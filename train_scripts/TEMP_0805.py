import numpy as np
import cupy as cp
import os
import time
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass, field
from pathlib import Path

from dataset import load_all_trajectories_from_file
from model_pcesn_DHextend import PCESNpp

from scipy.signal import savgol_filter # 引入滤波器对输出结果平滑化处理

@dataclass
class Config:
    """
    Centralized configuration for the experiment.
    Using a dataclass provides type hints and a clean structure.
    """
    # --- Paths and Naming ---
    DATA_FILE_PATH: str = 'venv/ForwardDynamics/KukaDirectDynamics.mat'
    RESULTS_DIR: Path = Path("results")
    EXPERIMENT_NAME: str = f'PCESN_PhysicsInformed_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # --- Experiment Setup ---
    NUM_FOLDS: int = 2  # Use K-fold cross-validation
    VISUALIZE_FOLDS: bool = True  # Generate plots for each fold
    RESERVOIR_HEATMAP: bool = False  # Set to True for detailed diagnostics

    # --- Model Hyperparameters ---
    INPUT_SIZE: int = 21
    OUTPUT_SIZE: int = 14
    RESERVOIR_SIZE: int = 400
    SPECTRAL_RADIUS: float = 0.9
    SPARSITY: float = 0.76
    LEAK_RATE: float = 0.6
    REGULARIZATION_FACTOR: float = 1e-4
    GHL_ETA: float = 1e-5
    GHL_DECAY_STEPS: int = 5000
    CORRECTION_INTERVAL: int | None = None

    # --- Smoothing Filter Settings ---
    APPLY_SMOOTHING: bool = True  # Master switch to enable/disable the filter
    SMOOTHING_WINDOW_LEN: int = 59  # Must be an odd integer
    SMOOTHING_POLY_ORDER: int = 2  # Must be less than window_length

    # --- Physics Parameters ---
    PHYSICS_PARAMS: dict = field(default_factory=lambda: {
        'dh_alpha': [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2],
        'dh_a': [0, 0, 0, 0, 0, 0, 0],
        'dh_d': [0.310, 0, 0.400, 0, 0.390, 0, 0.078],
        'link_masses': [2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5]
    })

def calculate_nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculates the Normalized Mean Squared Error (nMSE).
    nMSE = MSE(y_true, y_pred) / Var(y_true)
    """
    mse = np.mean((y_true - y_pred) ** 2)
    var_true = np.var(y_true)
    return mse / var_true if var_true > 1e-9 else mse

class DataHandler:
    """Handles all data loading, preparation, and splitting."""

    def __init__(self, config: Config):
        self.config = config
        self.all_trajectories = load_all_trajectories_from_file(config.DATA_FILE_PATH)
        self.physics_vector_cpu = self._get_physics_vector()
        self.physics_vector_gpu = cp.asarray(self.physics_vector_cpu)
        print(f"Loaded {len(self.all_trajectories)} trajectories.")

    def _get_physics_vector(self) -> np.ndarray:
        """Creates the static physics parameter vector."""
        vector = []
        for key in sorted(self.config.PHYSICS_PARAMS.keys()):
            vector.extend(self.config.PHYSICS_PARAMS[key])
        return np.array(vector, dtype=np.float32).reshape(-1, 1)

    def get_cross_validation_folds(self):
        """Creates a generator for K-fold cross-validation."""
        kf = KFold(n_splits=self.config.NUM_FOLDS, shuffle=True, random_state=42)
        trajectory_indices = np.arange(len(self.all_trajectories))

        for fold_idx, (train_indices, test_indices) in enumerate(kf.split(trajectory_indices)):
            train_trajs = [self.all_trajectories[i] for i in train_indices]
            test_trajs = [self.all_trajectories[i] for i in test_indices]  # Can have multiple test trajs

            # Fit scaler ONLY on training data for this fold
            all_train_inputs = np.concatenate([t['inputs'] for t in train_trajs])
            scaler = StandardScaler().fit(all_train_inputs)

            yield fold_idx, train_trajs, test_trajs, scaler

    def single_test(self, test_index: int = -1, random_state: int = None):
        """
        Provides a single train/test split for quick testing.

        Args:
            test_index (int): The index of the trajectory to use for testing.
                              Defaults to the last one.
            random_state (int): If provided, selects a random trajectory for testing.
                                Overrides test_index.

        """
        if random_state is not None:
            np.random.seed(random_state)
            test_index = np.random.randint(0, len(self.all_trajectories))
            print(f"Randomly selected trajectory {test_index} for testing.")

        all_indices = list(range(len(self.all_trajectories)))

        # Ensure test_index is valid
        if not -len(all_indices) <= test_index < len(all_indices):
            print(f"Warning: test_index {test_index} is out of bounds. Defaulting to the last trajectory.")
            test_index = -1

        test_traj = self.all_trajectories[test_index]

        # Remove the test index to get training indices
        train_indices = all_indices
        # Correctly remove the element at test_index, handling negative indices
        del train_indices[test_index % len(all_indices)]

        train_trajs = [self.all_trajectories[i] for i in train_indices]

        # Fit scaler ONLY on training data
        all_train_inputs = np.concatenate([t['inputs'] for t in train_trajs])
        scaler = StandardScaler().fit(all_train_inputs)

        return train_trajs, test_traj, scaler


class ExperimentRunner:
    """Manages the training and evaluation of the model for a single fold."""

    def __init__(self, config: Config, physics_vector_gpu: cp.ndarray):
        self.config = config
        self.physics_vector_gpu = physics_vector_gpu
        self.model = self._initialize_model()

    def _initialize_model(self) -> PCESNpp:
        """Initializes the PCESNpp model with parameters from the config."""
        return PCESNpp(
            input_size=self.config.INPUT_SIZE,
            reservoir_size=self.config.RESERVOIR_SIZE,
            output_size=self.config.OUTPUT_SIZE,
            physics_vector_size=len(self.physics_vector_gpu),
            spectral_radius=self.config.SPECTRAL_RADIUS,
            sparsity=self.config.SPARSITY,
            leak_rate=self.config.LEAK_RATE,
            regularization_factor=self.config.REGULARIZATION_FACTOR,
            ghl_eta=self.config.GHL_ETA,
            ghl_decay_steps=self.config.GHL_DECAY_STEPS
        )

    def train(self, train_trajs: list, scaler: StandardScaler):
        """Trains the model on a list of trajectories."""
        print("Training model...")
        start_time = time.time()
        state_history = []

        for traj in train_trajs:
            inputs = scaler.transform(traj['inputs'])
            targets = traj['targets']
            for t in range(len(inputs)):
                self.model.train_step(
                    inputs[t].reshape(-1, 1),
                    targets[t].reshape(-1, 1),
                    self.physics_vector_gpu
                )
                if self.config.RESERVOIR_HEATMAP:
                    state_history.append(cp.asnumpy(self.model.r_state))

        print(f"Training complete. Duration: {time.time() - start_time:.2f}s")
        if self.config.RESERVOIR_HEATMAP:
            Visualizer.plot_reservoir_heatmap(state_history, "Training Reservoir Dynamics")

    def evaluate(self, test_traj: dict, scaler: StandardScaler) -> dict:
        """Evaluates the model on a single test trajectory."""
        print("Evaluating trajectory...")
        self.model.reset_state()  # Crucial for a fair test on a new trajectory

        inputs = scaler.transform(test_traj['inputs'])
        targets = test_traj['targets']

        # --- 1. Single-Step-Ahead (SBS) Prediction ---
        # Uses ground truth input at each step. Tests the model's one-step mapping.
        sbs_predictions = np.array([
            self.model.predict(u_t.reshape(-1, 1), self.physics_vector_gpu).flatten()
            for u_t in inputs
        ])

        # --- 2. Full Trajectory (Autonomous) Prediction ---
        # Uses its own prediction to generate the next input. A much harder test.
        self.model.reset_state()  # Reset state again for a fair comparison
        full_predictions = []
        state_history = []
        current_input = inputs[0].reshape(-1, 1)

        for t in range(len(inputs)):
            predicted_output = self.model.predict(current_input, self.physics_vector_gpu)
            full_predictions.append(predicted_output.flatten())

            if self.config.RESERVOIR_HEATMAP:
                state_history.append(cp.asnumpy(self.model.r_state))

            # Prepare the next input if not the last step
            if t < len(inputs) - 1:
                # The model predicts the first 14 elements of the next state (positions)
                # We need to combine this with the ground truth of the other elements (velocities, etc.)
                # And Check whether the progress reaches check point
                if self.config.CORRECTION_INTERVAL and (t+1) % self.config.CORRECTION_INTERVAL==0:
                    print(f"  (Model Correction Point: Step- {t + 1})")
                    # Using real data from the next time step to Correct
                    current_input = inputs[t + 1].reshape(-1, 1)
                else:
                    next_input_unscaled = np.concatenate([
                    predicted_output.flatten(),
                    test_traj['inputs'][t + 1, self.config.OUTPUT_SIZE:]
                    ])
                    current_input = scaler.transform(next_input_unscaled.reshape(1, -1)).T

        if self.config.RESERVOIR_HEATMAP:
            Visualizer.plot_reservoir_heatmap(state_history, "Autonomous Prediction Reservoir Dynamics")

        results = {
            "sbs_predictions": np.array(sbs_predictions),
            "full_predictions": np.array(full_predictions),
            "targets": targets,
            "sbs_nmse": calculate_nmse(targets, sbs_predictions),
            "full_nmse": calculate_nmse(targets, full_predictions)
        }

        # --- 3. Apply Smoothing Filter (if enabled) ---
        if self.config.APPLY_SMOOTHING:
            print(
                f"Applying Savitzky-Golay filter (window: {self.config.SMOOTHING_WINDOW_LEN}, order: {self.config.SMOOTHING_POLY_ORDER})...")
            # Ensure window length is odd and greater than polyorder
            if self.config.SMOOTHING_WINDOW_LEN <= self.config.SMOOTHING_POLY_ORDER or self.config.SMOOTHING_WINDOW_LEN % 2 == 0:
                print("Warning: Invalid Savgol filter parameters. Skipping smoothing.")
                return results
            # Apply filter to each prediction type, column by column (axis=0)
            sbs_smoothed = savgol_filter(sbs_predictions, self.config.SMOOTHING_WINDOW_LEN,
                                         self.config.SMOOTHING_POLY_ORDER, axis=0)
            full_smoothed = savgol_filter(full_predictions, self.config.SMOOTHING_WINDOW_LEN,
                                          self.config.SMOOTHING_POLY_ORDER, axis=0)

            # Add smoothed results and their metrics to the dictionary
            results["sbs_predictions_smoothed"] = sbs_smoothed
            results["full_predictions_smoothed"] = full_smoothed
            results["sbs_nmse_smoothed"] = calculate_nmse(targets, sbs_smoothed)
            results["full_nmse_smoothed"] = calculate_nmse(targets, full_smoothed)

        return results


class Visualizer:
    """Handles all plotting and visualization tasks."""

    @staticmethod
    def plot_predictions(results: dict, fold_idx: int, save_dir: Path):
        """Generates and saves prediction plots for a fold."""
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"fold_{fold_idx + 1}_predictions.png"

        ground_truth = results["targets"]
        sbs_predictions = results["sbs_predictions"]
        full_predictions = results["full_predictions"]

        # Check if smoothed data exists in the results
        has_smoothed_data = "full_predictions_smoothed" in results

        num_plots = 7
        fig, axes = plt.subplots(num_plots, 1, figsize=(18, 36), sharex=True)
        fig.suptitle(f'Prediction vs. Ground Truth (Fold {fold_idx + 1})', fontsize=20)

        for i in range(num_plots):
            ax = axes[i]
            ax.plot(ground_truth[:, i], 'b-', label='Ground Truth', linewidth=2)
            ax.plot(sbs_predictions[:, i], 'g--', label='Single-Step Pred.', linewidth=1.5)
            ax.plot(full_predictions[:, i], 'r:', label=f'Autonomous Pred (Raw, nMSE:{results["full_nmse"]:.4f}).', linewidth=1.5)
            if has_smoothed_data:
                full_smoothed = results["full_predictions_smoothed"]
                ax.plot(full_smoothed[:, i], 'darkred', linestyle='-',
                        label=f'Autonomous (Smoothed, nMSE: {results["full_nmse_smoothed"]:.4f})', linewidth=2)

            ax.set_title(f'Joint {i + 1} Position')
            ax.set_ylabel('Position (rad)')
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel('Time Steps')
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(save_path)
        plt.close(fig)  # Close figure to free memory
        print(f"Saved plot to {save_path}")

    @staticmethod
    def plot_reservoir_heatmap(history: list, title: str):
        """Generates a heatmap of reservoir neuron activations over time."""
        if not history: return
        history_matrix = np.hstack(history)
        plt.figure(figsize=(18, 7))
        plt.imshow(history_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
        plt.colorbar(label="Neuron Activation")
        plt.title(title)
        plt.xlabel("Time Steps")
        plt.ylabel("Reservoir Neurons")
        plt.show()


def main():
    """Main execution function with interactive mode selection."""
    config = Config()
    data_handler = DataHandler(config)

    # --- Interactive selection of operating modes---
    run_mode = ""
    while run_mode not in ["1", "2"]:
        print("\nChoose Experiment Mode:")
        print("  1: 单次测试运行 (Single Test Run)")
        print("  2: 完整交叉验证 (Full Cross-Validation)")
        run_mode = input("Input Choice (1 或 2): ")

        # --- Step 2: Interactive Correction Interval Selection ---
        correction_interval_str = input(
            "\n请输入自主预测的校正步长 (Enter correction interval for autonomous prediction) "
            "\n[输入数字, 或留空/输入0代表不校正 (Enter a number, or leave blank/enter 0 for no correction)]: "
        )
        try:
            correction_interval = int(correction_interval_str)
            if correction_interval <= 0:
                config.CORRECTION_INTERVAL = None  # No correction
                print("--> 已设置为: 不进行周期性校正 (No periodic correction will be applied).")
            else:
                config.CORRECTION_INTERVAL = correction_interval
                print(
                    f"--> 已设置为: 每 {config.CORRECTION_INTERVAL} 步校正一次 (Correction will be applied every {config.CORRECTION_INTERVAL} steps).")
        except ValueError:
            config.CORRECTION_INTERVAL = None  # Default to no correction if input is invalid
            print("--> 无效输入。已设置为: 不进行周期性校正 (Invalid input. No periodic correction will be applied).")

    if run_mode == "2":
        # --- 交叉验证模式 ---
        print("\n===== Execute Cross Validation =====")
        all_results = []
        for fold_idx, train_trajs, test_trajs, scaler in data_handler.get_cross_validation_folds():
            print(f"\n--- Fold {fold_idx + 1}/{config.NUM_FOLDS}  ---")
            runner = ExperimentRunner(config, data_handler.physics_vector_gpu)
            runner.train(train_trajs, scaler)

            if test_trajs:
                test_results = runner.evaluate(test_trajs[0], scaler)
                all_results.append(test_results)
                if config.VISUALIZE_FOLDS:
                    fold_results_dir = config.RESULTS_DIR / config.EXPERIMENT_NAME
                    Visualizer.plot_predictions(test_results, fold_idx, fold_results_dir)

        print("\n===== Cross-Validation Complete =====")
        avg_sbs_nmse = np.mean([r['sbs_nmse'] for r in all_results])
        avg_full_nmse = np.mean([r['full_nmse'] for r in all_results])
        print(f"Average Single-Step nMSE across {len(all_results)} folds: {avg_sbs_nmse:.6f}")
        print(f"Average Autonomous nMSE across {len(all_results)} folds: {avg_full_nmse:.6f}")

    elif run_mode == "1":
        # --- 单次测试模式 ---
        print("\n===== Execute Single Running Test =====")
        train_trajs, test_traj, scaler = data_handler.single_test(test_index=5)

        print(f"Using {len(train_trajs)} Trajectories for Training")

        runner = ExperimentRunner(config, data_handler.physics_vector_gpu)
        runner.train(train_trajs, scaler)

        test_results = runner.evaluate(test_traj, scaler)

        if config.VISUALIZE_FOLDS:
            fold_results_dir = config.RESULTS_DIR / config.EXPERIMENT_NAME
            Visualizer.plot_predictions(test_results, 0, fold_results_dir)

        # mse = np.mean((test_results['full_predictions'] - test_results['targets']) ** 2)
        # print(f"\nFull_Prediction MSE: {mse:.6f}")

        print(f"\nSingle-Step nMSE: {test_results['sbs_nmse']:.6f}")
        print(f"Autonomous nMSE: {test_results['full_nmse']:.6f}")

if __name__ == '__main__':
    main()

