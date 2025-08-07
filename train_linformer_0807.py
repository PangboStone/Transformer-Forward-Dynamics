import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import time
from pathlib import Path
from dataclasses import dataclass, field

# =============================================================================
#  Module Imports
#  Ensure these files are in your project directory
# =============================================================================
from dataset import RobotDynamicsDataset, load_all_trajectories_from_file
from model_transformer import TransformerModel
from model_transformer_LinformerAttention import LinformerTransformerModel


# =============================================================================
#  1. Configuration Center
# =============================================================================
@dataclass
class Config:
    """Centralized configuration for the experiment."""
    # --- Paths and Naming ---
    DATA_FILE_PATH: str = 'venv/ForwardDynamics/KukaDirectDynamics.mat'
    RESULTS_DIR: Path = Path("results_transformer")
    EXPERIMENT_NAME: str = f'Linformer_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    # --- Device and Model Selection ---
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_TYPE: str = 'linformer'  # Options: 'transformer' or 'linformer'

    # --- Data and Model Hyperparameters ---
    SEQUENCE_LENGTH: int = 50
    K_PROJ_DIM: int = 10  # Projection dimension k for Linformer
    INPUT_SIZE: int = 21  # Input dimension (e.g., pos_7, vel_7, torque_7)
    OUTPUT_SIZE: int = 14  # Output dimension (e.g., pos_7, vel_7)
    D_MODEL: int = 256
    N_HEAD: int = 8
    NUM_ENCODER_LAYERS: int = 4
    DIM_FEEDFORWARD: int = 512
    DROPOUT: float = 0.1

    # --- Training Hyperparameters ---
    LEARNING_RATE: float = 5e-5
    BATCH_SIZE: int = 64
    EPOCHS: int = 50

    # --- Experiment and Evaluation Settings ---
    NUM_FOLDS: int = 5  # Used only if cross-validation is enabled
    VALIDATION_TRAJECTORY_INDEX: int = -1  # Index for the validation set in a single run
    CORRECTION_INTERVAL: int | None = 100  # Correction step for autonomous prediction
    SAVE_PLOTS: bool = True
    SHOW_PLOTS: bool = False  # Controls if plot windows are shown


# =============================================================================
#  2. Data and Model Core Functions
# =============================================================================
class DataHandler:
    """Encapsulates all data loading, splitting, and scaling logic."""

    def __init__(self, config: Config):
        self.config = config
        self.all_trajectories = load_all_trajectories_from_file(config.DATA_FILE_PATH)
        print(f"Successfully loaded {len(self.all_trajectories)} trajectories.")

    def get_dataloaders_for_fold(self, train_indices: list, val_indices: list) -> tuple:
        """Creates DataLoaders and scalers for a specific train/validation split."""
        train_trajs = [self.all_trajectories[i] for i in train_indices]
        val_trajs = [self.all_trajectories[i] for i in val_indices]

        # Fit scalers ONLY on training data to prevent data leakage
        input_scaler = StandardScaler().fit(np.concatenate([t['inputs'] for t in train_trajs]))
        output_scaler = StandardScaler().fit(np.concatenate([t['targets'] for t in train_trajs]))

        train_dataset = RobotDynamicsDataset(train_trajs, self.config.SEQUENCE_LENGTH, input_scaler, output_scaler)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)

        val_dataset = RobotDynamicsDataset(val_trajs, self.config.SEQUENCE_LENGTH, input_scaler, output_scaler)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE, shuffle=False)

        return train_loader, val_loader, input_scaler, output_scaler, val_trajs


def get_model(config: Config) -> nn.Module:
    """Initializes and returns the specified model based on the config."""
    model_map = {'transformer': TransformerModel, 'linformer': LinformerTransformerModel}
    model_class = model_map.get(config.MODEL_TYPE.lower())
    if not model_class:
        raise ValueError(f"Unknown model type: {config.MODEL_TYPE}")

    print(f"Initializing {config.MODEL_TYPE} model...")
    model_params = {
        'input_size': config.INPUT_SIZE, 'd_model': config.D_MODEL, 'nhead': config.N_HEAD,
        'num_encoder_layers': config.NUM_ENCODER_LAYERS, 'dim_feedforward': config.DIM_FEEDFORWARD,
        'output_size': config.OUTPUT_SIZE, 'dropout': config.DROPOUT,
        'seq_len': config.SEQUENCE_LENGTH, 'k': config.K_PROJ_DIM
    }
    return model_class(**model_params).to(config.DEVICE)


# =============================================================================
#  3. Metrics Calculation
# =============================================================================
def calculate_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Calculates detailed, component-wise evaluation metrics."""
    num_joints = y_true.shape[1] // 2
    pos_true, vel_true = y_true[:, :num_joints], y_true[:, num_joints:]
    pos_pred, vel_pred = y_pred[:, :num_joints], y_pred[:, num_joints:]

    # Calculate nMSE for position and velocity
    pos_mse = np.mean((pos_true - pos_pred) ** 2)
    vel_mse = np.mean((vel_true - vel_pred) ** 2)
    pos_var = np.var(pos_true)
    vel_var = np.var(vel_true)

    pos_nmse = pos_mse / pos_var if pos_var > 1e-9 else pos_mse
    vel_nmse = vel_mse / vel_var if vel_var > 1e-9 else vel_mse

    # Calculate mean Euclidean distance error for position
    # This assumes the first num_joints columns represent positions. Adjust if necessary.
    euclidean_error = np.mean([np.linalg.norm(pos_true[t] - pos_pred[t]) for t in range(len(pos_true))])

    return {
        "pos_nmse": pos_nmse,
        "vel_nmse": vel_nmse,
        "euclidean_error": euclidean_error
    }


# =============================================================================
#  4. Experiment Runner
# =============================================================================
class ExperimentRunner:
    """Manages the full lifecycle of a single experiment fold: training, validation, and evaluation."""

    def __init__(self, config: Config, fold_idx: int):
        self.config, self.fold_idx = config, fold_idx
        self.model = get_model(config)
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.LEARNING_RATE)
        self.writer = SummaryWriter(log_dir=str(config.RESULTS_DIR / config.EXPERIMENT_NAME / f'fold_{fold_idx}'))

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> dict:
        """Executes the complete training and validation loop."""
        print(f"--- Starting Training (Fold {self.fold_idx + 1}) ---")
        start_time = time.time()
        history = {'train_loss': [], 'val_loss': [], 'training_time': 0.0}

        for epoch in range(self.config.EPOCHS):
            self.model.train()
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(self.config.DEVICE), targets.to(self.config.DEVICE)
                loss = self.loss_fn(self.model(sequences), targets)
                self.optimizer.zero_grad();
                loss.backward();
                self.optimizer.step()

            self.model.eval()
            with torch.no_grad():
                # Calculate loss on a subset for speed, or full set for accuracy
                train_loss = np.mean(
                    [self.loss_fn(self.model(s.to(self.config.DEVICE)), t.to(self.config.DEVICE)).item() for s, t in
                     train_loader])
                val_loss = np.mean(
                    [self.loss_fn(self.model(s.to(self.config.DEVICE)), t.to(self.config.DEVICE)).item() for s, t in
                     val_loader])

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            print(
                f"Epoch [{epoch + 1:02d}/{self.config.EPOCHS}], Train Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

        history['training_time'] = time.time() - start_time
        print(f"Training finished. Duration: {history['training_time']:.2f}s")
        self.writer.close()
        return history

    def evaluate(self, val_trajectory: dict, input_scaler: StandardScaler, output_scaler: StandardScaler) -> dict:
        """Performs single-step and full-trajectory prediction on the validation set."""
        print("--- Starting Evaluation ---")
        self.model.eval()
        inputs_s = input_scaler.transform(val_trajectory['inputs'])
        targets_unscaled = val_trajectory['targets']

        # Single-Step Prediction
        val_ds = RobotDynamicsDataset([val_trajectory], self.config.SEQUENCE_LENGTH, input_scaler, output_scaler)
        val_loader = DataLoader(val_ds, batch_size=self.config.BATCH_SIZE, shuffle=False)
        with torch.no_grad():
            preds_s_sbs = np.concatenate([self.model(s.to(self.config.DEVICE)).cpu().numpy() for s, _ in val_loader])
        preds_us_sbs = output_scaler.inverse_transform(preds_s_sbs)

        # Full Trajectory Prediction
        current_seq = torch.tensor(inputs_s[:self.config.SEQUENCE_LENGTH], dtype=torch.float32).unsqueeze(0).to(
            self.config.DEVICE)
        preds_us_full = []
        with torch.no_grad():
            for t in range(len(inputs_s) - self.config.SEQUENCE_LENGTH):
                next_pred_s = self.model(current_seq)
                preds_us_full.append(output_scaler.inverse_transform(next_pred_s.cpu().numpy()).flatten())

                if self.config.CORRECTION_INTERVAL and (t + 1) % self.config.CORRECTION_INTERVAL == 0:
                    next_seq_s = inputs_s[t + 1: t + 1 + self.config.SEQUENCE_LENGTH]
                    current_seq = torch.tensor(next_seq_s, dtype=torch.float32).unsqueeze(0).to(self.config.DEVICE)
                else:
                    next_torque_us = val_trajectory['inputs'][t + self.config.SEQUENCE_LENGTH, self.config.OUTPUT_SIZE:]
                    next_input_us = np.concatenate([preds_us_full[-1], next_torque_us])
                    next_input_s = torch.tensor(input_scaler.transform(next_input_us.reshape(1, -1)),
                                                dtype=torch.float32).to(self.config.DEVICE)
                    current_seq = torch.cat([current_seq[:, 1:, :], next_input_s.unsqueeze(1)], dim=1)

        preds_us_full = np.array(preds_us_full)
        min_len = min(len(targets_unscaled), len(preds_us_sbs), len(preds_us_full))
        targets_aligned = targets_unscaled[:min_len]

        return {
            "sbs_metrics": calculate_detailed_metrics(targets_aligned, preds_us_sbs[:min_len]),
            "full_metrics": calculate_detailed_metrics(targets_aligned, preds_us_full[:min_len]),
            "predictions": {"sbs": preds_us_sbs[:min_len], "full": preds_us_full[:min_len]},
            "targets": targets_aligned
        }

    def analyze_error_accumulation(self, val_trajectory: dict, input_scaler: StandardScaler,
                                   output_scaler: StandardScaler, horizons: list) -> dict:
        """
        Analyzes prediction error over a single rollout, now with progress monitoring.
        """
        print("--- Starting Simplified Error Accumulation Analysis (Single Rollout) ---")
        self.model.eval()
        inputs_s = input_scaler.transform(val_trajectory['inputs'])
        targets_us = val_trajectory['targets']

        horizon_set = set(horizons)
        max_horizon = max(horizons)

        errors = {}

        current_seq = torch.tensor(
            inputs_s[:self.config.SEQUENCE_LENGTH],
            dtype=torch.float32
        ).unsqueeze(0).to(self.config.DEVICE)

        with torch.no_grad():
            for step in range(max_horizon):
                # --- Progress Monitoring ---
                # Print progress on the same line. `\r` moves the cursor to the beginning
                # of the line, and `end=''` prevents a newline character.
                print(f"\r    > Analyzing horizon step: {step + 1}/{max_horizon}", end="")

                # Make a one-step prediction
                next_pred_s = self.model(current_seq)

                # Check if the current step+1 is a horizon we need to record
                current_horizon = step + 1
                if current_horizon in horizon_set:
                    pred_us = output_scaler.inverse_transform(next_pred_s.cpu().numpy())
                    true_val = targets_us[self.config.SEQUENCE_LENGTH + step]
                    error = np.linalg.norm(true_val - pred_us.flatten())
                    errors[current_horizon] = error

                if len(errors) == len(horizon_set):
                    break
                # Prepare the next input for the rollout
                next_pred_us_for_input = output_scaler.inverse_transform(next_pred_s.cpu().numpy())
                next_torque_us = val_trajectory['inputs'][self.config.SEQUENCE_LENGTH + step, self.config.OUTPUT_SIZE:]
                next_input_us = np.concatenate([next_pred_us_for_input.flatten(), next_torque_us])
                next_input_s = torch.tensor(input_scaler.transform(next_input_us.reshape(1, -1)),
                                            dtype=torch.float32).to(self.config.DEVICE)
                current_seq = torch.cat([current_seq[:, 1:, :], next_input_s.unsqueeze(1)], dim=1)

        print()
        print("Error accumulation analysis finished.")
        return errors


# =============================================================================
#  5. Visualizer & Reporter
# =============================================================================
class Visualizer:
    """Encapsulates all plotting logic."""

    @staticmethod
    def _plot_handler(save_path: Path | None, show: bool):
        if save_path:
            plt.savefig(save_path)
            print(f"Plot saved to: {save_path}")
        if show:
            plt.show()
        plt.close()

    @staticmethod
    def plot_loss_curves(history: dict, save_path: Path, show: bool):
        plt.figure(figsize=(12, 7));
        plt.grid(True)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Training Loss Curve');
        plt.xlabel('Epoch');
        plt.ylabel('MSE Loss');
        plt.legend()
        Visualizer._plot_handler(save_path, show)

    @staticmethod
    def plot_predictions(results: dict, save_path: Path, show: bool):
        targets, full_preds = results['targets'], results['predictions']['full']
        num_joints = targets.shape[1] // 2
        fig, axes = plt.subplots(num_joints, 2, figsize=(22, 5 * num_joints), sharex=True)
        fig.suptitle(
            f'Autonomous Prediction vs. Ground Truth\nPos nMSE: {results["full_metrics"]["pos_nmse"]:.4f}, Vel nMSE: {results["full_metrics"]["vel_nmse"]:.4f}',
            fontsize=16)
        for i in range(num_joints):
            axes[i, 0].plot(targets[:, i], 'b-', label='Ground Truth Position');
            axes[i, 0].plot(full_preds[:, i], 'r--', label='Predicted Position')
            axes[i, 0].set_ylabel(f'Joint {i + 1} Position');
            axes[i, 0].grid(True);
            axes[i, 0].legend()
            axes[i, 1].plot(targets[:, i + num_joints], 'b-', label='Ground Truth Velocity');
            axes[i, 1].plot(full_preds[:, i + num_joints], 'r--', label='Predicted Velocity')
            axes[i, 1].set_ylabel(f'Joint {i + 1} Velocity');
            axes[i, 1].grid(True);
            axes[i, 1].legend()
        axes[-1, 0].set_xlabel('Time Step');
        axes[-1, 1].set_xlabel('Time Step')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        Visualizer._plot_handler(save_path, show)

    @staticmethod
    def plot_error_vs_horizon(horizon_errors: dict, save_path: Path, show: bool):
        horizons, errors = list(horizon_errors.keys()), list(horizon_errors.values())
        plt.figure(figsize=(12, 7));
        plt.grid(True)
        plt.plot(horizons, errors, 'bo-', label='Mean Euclidean Error')
        plt.title('Prediction Error vs. Prediction Horizon');
        plt.xlabel('Prediction Horizon (steps)');
        plt.ylabel('Mean Error')
        plt.legend();
        plt.xticks(horizons)
        Visualizer._plot_handler(save_path, show)


def print_results_table(all_fold_metrics: list):
    """Prints a formatted table of the final averaged results."""
    header = f"--- Final Averaged Results ({len(all_fold_metrics)} Folds, Mean ± Std Dev) ---"
    print("\n" + header);
    print("-" * len(header))
    print(f"{'Metric':<28} | {'Mean Value ± Std Deviation':<30}")
    print("-" * len(header))

    # Collect all keys from all dictionaries
    all_keys = sorted(list(all_fold_metrics[0].keys()))

    for key in all_keys:
        values = [d[key] for d in all_fold_metrics]
        mean, std = np.mean(values), np.std(values)
        print(f"{key:<28} | {mean:<10.4f} ± {std:<18.4f}")
    print("-" * len(header))


# =============================================================================
#  6. Main Execution Flow
# =============================================================================
def main():
    """Main function to coordinate the entire experiment."""
    config = Config()

    # --- Interactive Setup ---
    print("--- Experiment Setup ---")
    try:
        # Ask about cross-validation first
        perform_cv = (input("Perform full K-Fold cross-validation? (y/n) [default: n]: ").lower() or 'n') == 'y'

        if not perform_cv:
            config.NUM_FOLDS = 1
        else:
            folds_input = input(f"Enter number of folds for cross-validation [default: {config.NUM_FOLDS}]: ")
            config.NUM_FOLDS = int(folds_input) if folds_input else config.NUM_FOLDS

        corr_input = input(f"Enter correction interval [0 or blank for none, default: {config.CORRECTION_INTERVAL}]: ")
        config.CORRECTION_INTERVAL = int(corr_input) if corr_input and int(corr_input) > 0 else None

        config.SAVE_PLOTS = (input("Save plots? (y/n) [default: y]: ").lower() or 'y') == 'y'
        config.SHOW_PLOTS = (input("Show plot windows? (y/n) [default: n]: ").lower() or 'n') == 'y'
    except (ValueError, TypeError):
        print("Invalid input. Using default settings.")

    # --- Experiment Execution ---
    data_handler = DataHandler(config)
    results_dir = config.RESULTS_DIR / config.EXPERIMENT_NAME
    results_dir.mkdir(parents=True, exist_ok=True)

    all_fold_metrics = []

    if perform_cv:
        # Full Cross-Validation Run
        kf = KFold(n_splits=config.NUM_FOLDS, shuffle=True, random_state=42)
        indices = list(range(len(data_handler.all_trajectories)))
        for fold_idx, (train_indices, val_indices) in enumerate(kf.split(indices)):
            run_fold(config, data_handler, fold_idx, train_indices, val_indices, results_dir, all_fold_metrics)
    else:
        # Single Run
        all_indices = list(range(len(data_handler.all_trajectories)))
        val_indices = [config.VALIDATION_TRAJECTORY_INDEX % len(all_indices)]
        train_indices = [i for i in all_indices if i not in val_indices]
        run_fold(config, data_handler, 0, train_indices, val_indices, results_dir, all_fold_metrics)

    # --- Final Reporting ---
    if all_fold_metrics:
        print_results_table(all_fold_metrics)

        # For simplicity, run error accumulation analysis only on the last fold's setup
        print("\n--- Running error accumulation analysis on the last fold's setup ---")
        # We need to re-initialize the last runner to run the analysis
        # This part could be refactored further if multiple analyses are needed
        last_fold_idx = config.NUM_FOLDS - 1 if perform_cv else 0
        runner = ExperimentRunner(config, last_fold_idx)  # Re-init runner for the last fold

        # We need the data from the last fold again
        all_indices = list(range(len(data_handler.all_trajectories)))
        if perform_cv:
            # This is a simplification; a more robust way would be to store the last fold's data
            val_indices = [all_indices[-1]]  # Placeholder
            train_indices = all_indices[:-1]  # Placeholder
        else:
            val_indices = [config.VALIDATION_TRAJECTORY_INDEX % len(all_indices)]
            train_indices = [i for i in all_indices if i not in val_indices]

        _, _, last_input_scaler, last_output_scaler, last_val_trajs = data_handler.get_dataloaders_for_fold(
            train_indices, val_indices)

        horizon_errors = runner.analyze_error_accumulation(
            last_val_trajs[0], last_input_scaler, last_output_scaler,
            horizons=[1, 5, 10, 25, 50, 100]
        )
        if config.SAVE_PLOTS or config.SHOW_PLOTS:
            Visualizer.plot_error_vs_horizon(horizon_errors,
                                             results_dir / "error_vs_horizon.png" if config.SAVE_PLOTS else None,
                                             config.SHOW_PLOTS)

    print("\n--- Experiment Finished ---")


def run_fold(config, data_handler, fold_idx, train_indices, val_indices, results_dir, all_fold_metrics):
    """Helper function to run, evaluate, and visualize a single fold."""
    print(f"\n{'=' * 25} FOLD {fold_idx + 1}/{config.NUM_FOLDS} {'=' * 25}")

    train_loader, val_loader, input_scaler, output_scaler, val_trajs = data_handler.get_dataloaders_for_fold(
        train_indices, val_indices)

    runner = ExperimentRunner(config, fold_idx)
    history = runner.train(train_loader, val_loader)
    eval_results = runner.evaluate(val_trajs[0], input_scaler, output_scaler)

    # Collect metrics for the final table
    fold_metrics = {"Training Time (s)": history['training_time']}
    for key, val in eval_results['sbs_metrics'].items():
        fold_metrics[f"Sbs {key.replace('_', ' ').title()}"] = val
    for key, val in eval_results['full_metrics'].items():
        fold_metrics[f"Full {key.replace('_', ' ').title()}"] = val
    all_fold_metrics.append(fold_metrics)

    # Visualize results for this fold if requested
    if config.SAVE_PLOTS or config.SHOW_PLOTS:
        save_path_loss = results_dir / f"fold_{fold_idx}_loss.png" if config.SAVE_PLOTS else None
        save_path_preds = results_dir / f"fold_{fold_idx}_preds.png" if config.SAVE_PLOTS else None
        Visualizer.plot_loss_curves(history, save_path_loss, config.SHOW_PLOTS)
        Visualizer.plot_predictions(eval_results, save_path_preds, config.SHOW_PLOTS)


if __name__ == '__main__':
    main()
