import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

# --- 从自定义模块导入 ---
from dataset import load_all_trajectories_from_file
from model_pcesn import PCESNpp

# --- 1. Setting hyperparameters ---
# 数据与交叉验证
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
NUM_FOLDS = 10  # 10轮交叉验证

# PC-ESN++ 模型参数
INPUT_SIZE = 21
OUTPUT_SIZE = 14
RESERVOIR_SIZE = 400   # 关键超参数 储备池大小
SPECTRAL_RADIUS = 0.7
SPARSITY = 0.1
LEAK_RATE = 0.3
REGULARIZATION_FACTOR = 1e-4
GHL_LEARNING_RATE = 1e-3

# 结果
# ==================== 最终结果 ====================
# 平均单步预测 nMSE (2轮交叉验证): 0.792272 +/- 0.047574
# 平均全轨迹预测 nMSE (2轮交叉验证): 22.648288 +/- 1.544850


# --- Error Calculating Function ---
def calculate_nmse_overall(predictions, targets):
    """
    计算整体归一化均方误差 (nMSE)。
    如果输入是 (N_steps, N_dims)，则返回所有输出维度的平均 nMSe。
    如果输入是 (N_dims,)，则返回该单步的平均 nMSe。
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Ensure predictions and targets have at least 2 dimensions for consistent mean/var operations
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)

    # Calculate MSE per output dimension
    mse_per_output = np.mean((predictions - targets) ** 2, axis=0)
    # Calculate Variance per output dimension
    var_targets_per_output = np.var(targets, axis=0)

    # Avoid division by zero, then take mean over output dimensions
    nmse_per_output = mse_per_output / (var_targets_per_output + 1e-9)
    return np.mean(nmse_per_output)

def calculate_component_nmse(predictions, targets):
    """
    计算位置和速度的归一化均方误差 (nMSE)。
    Args:
        predictions (np.ndarray): 预测值，形状为 (N_steps, 14) 或 (14,).
        targets (np.ndarray): 真实值，形状为 (N_steps, 14) 或 (14,).
    Returns:
        tuple: (nMSE_position, nMSE_velocity)
    """
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    # Ensure predictions and targets have at least 2 dimensions for consistent slicing
    if predictions.ndim == 1:
        predictions = predictions.reshape(1, -1)
        targets = targets.reshape(1, -1)

    # Split into position (first 7 dims) and velocity (next 7 dims)
    pred_pos = predictions[:, :7]
    pred_vel = predictions[:, 7:]
    target_pos = targets[:, :7]
    target_vel = targets[:, 7:]

    nmse_pos = calculate_nmse_overall(pred_pos, target_pos)
    nmse_vel = calculate_nmse_overall(pred_vel, target_vel)

    return nmse_pos, nmse_vel

# def calculate_nmse(predictions, targets):
#     """计算归一化均方误差 (nMSE)"""
#     mse = np.mean((predictions - targets) ** 2)
#     var_targets = np.var(targets, axis=0)
#     # 避免除以零
#     nmse_per_output = mse / (var_targets + 1e-9)
#     return np.mean(nmse_per_output)  # 返回所有输出维度的平均nMSE



if __name__ == '__main__':
    # --- TensorBoard train log set up ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = f'pcesn_{NUM_FOLDS}_crossvalidation_{current_time}'
    parent_log_dir_tb = os.path.join('runs', experiment_name)
    os.makedirs(parent_log_dir_tb, exist_ok=True)  # Ensure parent directory exists

    # Initial console message for the entire run
    print(f"--- PC-ESN++ Training and Evaluation ---")
    print(f"Start Time: {current_time}")

    # load data
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

    #  nMSE results storing list
    all_folds_nmse_sbs_pos = []
    all_folds_nmse_sbs_vel = []
    all_folds_nmse_full_pos = []
    all_folds_nmse_full_vel = []

    # ———— Cross Valiadation ————
    for i in range(NUM_FOLDS):
        #     Generate a SummaryWriter for each fold
        writer = SummaryWriter(os.path.join(parent_log_dir_tb,f'Fold_{i+1}'))
        writer.add_text('Cross_Validation_Info', f'Start of Cross Validation Round {i + 1}/{NUM_FOLDS}', 0)
        print(f"\n{'=' * 15} Cross Validation: Round {i + 1}/{NUM_FOLDS}  {'=' * 15}")

        #     divide train and validate set
        test_trajectory = all_trajectories[i]
        train_trajectories = all_trajectories[:i] + all_trajectories[i + 1:]

        # date scalaring
        combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
        scaler = StandardScaler().fit(combined_train_inputs)

        # Scaling training data and test data
        scaled_train_inputs = [scaler.transform(t['inputs']) for t in train_trajectories]
        scaled_test_inputs = scaler.transform(test_trajectory['inputs'])

        # Target data keep the same
        train_targets = [t['targets'] for t in train_trajectories]
        test_targets = test_trajectory['targets']

        # Model initialization
        model = PCESNpp(input_size=INPUT_SIZE,
                        reservoir_size=RESERVOIR_SIZE,
                        output_size=OUTPUT_SIZE,
                        spectral_radius=SPECTRAL_RADIUS,
                        sparsity=SPARSITY,
                        leak_rate=LEAK_RATE,
                        regularization_factor=REGULARIZATION_FACTOR,
                        ghl_learning_rate=GHL_LEARNING_RATE)

        # Training Phase
        writer.add_text('Training_Info', '--- Start Training ---', 0)
        print("\n--- Start Training ---")
        global_step_counter = 0  # Global counter for X-axis in Tnsorboard
        all_training_losses = []  # colloct loss of all training samples

        for traj_idx, inputs in enumerate(scaled_train_inputs):
            targets = train_targets[traj_idx]
            trajectory_losses = []
            writer.add_text('Training_Info',
                            f"Training on Trajectory {traj_idx + 1}/{len(scaled_train_inputs)} ...",
                            global_step_counter)
            print(f"Training on Trajectory. {traj_idx + 1}/{len(scaled_train_inputs)} ...")

            # train model by each sample
            for t in range(len(inputs)):
                u_t = inputs[t].reshape(-1, 1)
                target_t = targets[t].reshape(-1, 1)
                loss = model.train_step(u_t, target_t)  # This 'loss' is MSE
                trajectory_losses.append(loss)
                writer.add_scalar('Training/MSE_per_step', loss, global_step_counter)
                global_step_counter += 1

            avg_traj_loss = np.mean(trajectory_losses)
            writer.add_scalar('Training/Average_MSE_per_trajectory', avg_traj_loss, traj_idx)
            writer.add_text('Training_Info',
                            f"Average Training MSE for Trajectory {traj_idx + 1}: {avg_traj_loss:.6f}",
                            global_step_counter)
            print(f"Average Training MSE: {avg_traj_loss:.6f}")
            all_training_losses.extend(trajectory_losses)

        overall_avg_train_loss = np.mean(all_training_losses)
        writer.add_text('Training_Info', "--- Training Complete ---", global_step_counter)
        writer.add_scalar('Training/Overall_Average_MSE', overall_avg_train_loss, global_step_counter)
        print("--- Training Complete ---")

        # Evaluation Phase
        writer.add_text('Evaluation_Info', '--- Start Evaluation ---', 0)
        print("\n--- Start Evaluation ---")

        # a) Single Step Prediction (Step-by-step prediction)
        # Reset the internal state of model for fair assessment
        model.r_state.fill(0)
        predictions_step_by_step = []

        for t in range(len(scaled_test_inputs)):
            u_t = scaled_test_inputs[t].reshape(-1, 1)
            pred_t = model.predict(u_t)  # Shape (14, 1)
            predictions_step_by_step.append(pred_t.flatten())  # Store flattened for overall calculation later

            # Calculate and record per-step position and velocity nMSE for convergence curve
            target_t = test_targets[t]  # Shape (14,)
            nmse_pos_t, nmse_vel_t = calculate_component_nmse(pred_t.flatten(), target_t)
            writer.add_scalar('Evaluation_Convergence/SBS_Position_nMSE', nmse_pos_t, t)
            writer.add_scalar('Evaluation_Convergence/SBS_Velocity_nMSE', nmse_vel_t, t)

        # Calculate overall nMSE for SBS for the current fold
        final_nmse_sbs_pos, final_nmse_sbs_vel = calculate_component_nmse(np.array(predictions_step_by_step),
                                                                          test_targets)
        writer.add_scalar('Evaluation/Final_SBS_Position_nMSE', final_nmse_sbs_pos, i)
        writer.add_scalar('Evaluation/Final_SBS_Velocity_nMSE', final_nmse_sbs_vel, i)
        print(f"nMSE (Single Step Prediction) - Position: {final_nmse_sbs_pos:.6f}, Velocity: {final_nmse_sbs_vel:.6f}")

        all_folds_nmse_sbs_pos.append(final_nmse_sbs_pos)
        all_folds_nmse_sbs_vel.append(final_nmse_sbs_vel)

        # b) Full Trajectory Prediction
        model.r_state.fill(0)  # Reset model state for full trajectory prediction
        predictions_full = []

        # Use the first actual input of the test set as the starting point
        current_input = scaled_test_inputs[0].reshape(-1, 1)

        for i in range(len(scaled_test_inputs)):
            # Used current input
            predicted_output = model.predict(current_input)  # Shape (14, 1)
            predictions_full.append(predicted_output.flatten())

            # Calculate and record per-step position and velocity nMSE for convergence curve
            target_t = test_targets[t]  # Shape (14,)
            nmse_pos_t, nmse_vel_t = calculate_component_nmse(predicted_output.flatten(), target_t)
            writer.add_scalar('Evaluation_Convergence/Full_Position_nMSE', nmse_pos_t, t)
            writer.add_scalar('Evaluation_Convergence/Full_Velocity_nMSE', nmse_vel_t, t)

            # Generate input signal for nest time step
            if t < len(scaled_test_inputs)-1:
                predicted_pos_vel = predicted_output.flatten()[:14] # use predicted position and velocity

                # Inverse transform to original date form
                # Create a dummy full input to inverse transform (since scaler was fitted on 21 dims)
                dummy_input_for_inverse = np.zeros(INPUT_SIZE)
                dummy_input_for_inverse[:14] = predicted_pos_vel
                unscaled_pred_full = scaler.inverse_transform(dummy_input_for_inverse.reshape(1, -1)).flatten()

                # Get torque input for next time step
                true_torque_next = test_trajectory['input'][t+1, 14:21]

                # Construct next input (unscaled)
                next_input_unscaled = np.concatenate([unscaled_pred_full[:14], true_torque_next])

                # Scale the next input for the next loop iteration
                current_input = scaler.transform(next_input_unscaled.reshape(1, -1)).reshape(-1, 1)

        # Calculate overall nMSE for Full Trajectory for the current fold
        final_nmse_full_pos, final_nmse_full_vel = calculate_component_nmse(np.array(predictions_full),test_targets)
        writer.add_scalar('Evaluation/Final_Full_Position_nMSE', final_nmse_full_pos, i)
        writer.add_scalar('Evaluation/Final_Full_Velocity_nMSE', final_nmse_full_vel, i)
        print(f"nMSE (Full Trajectory Prediction) - Position: {final_nmse_full_pos:.6f}, Velocity: {final_nmse_full_vel:.6f}")

        all_folds_nmse_full_pos.append(final_nmse_full_pos)
        all_folds_nmse_full_vel.append(final_nmse_full_vel)

        writer.close()  # Close writer for the current fold

    # --- Summarize total results of CrossValiadation ---
    print(f"\n\n{'=' * 20} Final Results Across {NUM_FOLDS} Folds {'=' * 20}")

    # Use a dedicated writer for the overall summary to make it accessible at the root of the TensorBoard run
    final_summary_writer = SummaryWriter(parent_log_dir_tb)

    summary_table_string = (
        "| **Metric** | **Mean (Position)** | **Std (Position)** | **Mean (Velocity)** | **Std (Velocity)** |\n"
        "|---|---|---|---|---|\n"
    )

    if all_folds_nmse_sbs_pos and all_folds_nmse_sbs_vel:
        avg_sbs_pos = np.mean(all_folds_nmse_sbs_pos)
        std_sbs_pos = np.std(all_folds_nmse_sbs_pos)
        avg_sbs_vel = np.mean(all_folds_nmse_sbs_vel)
        std_sbs_vel = np.std(all_folds_nmse_sbs_vel)

        summary_table_string += f"| One-step nMSE | {avg_sbs_pos:.6f} | {std_sbs_pos:.6f} | {avg_sbs_vel:.6f} | {std_sbs_vel:.6f} |\n"
        final_summary_writer.add_scalar('Summary/Mean_SBS_Position_nMSE', avg_sbs_pos, 0)
        final_summary_writer.add_scalar('Summary/Std_SBS_Position_nMSE', std_sbs_pos, 0)
        final_summary_writer.add_scalar('Summary/Mean_SBS_Velocity_nMSE', avg_sbs_vel, 0)
        final_summary_writer.add_scalar('Summary/Std_SBS_Velocity_nMSE', std_sbs_vel, 0)

    if all_folds_nmse_full_pos and all_folds_nmse_full_vel:
        avg_full_pos = np.mean(all_folds_nmse_full_pos)
        std_full_pos = np.std(all_folds_nmse_full_pos)
        avg_full_vel = np.mean(all_folds_nmse_full_vel)
        std_full_vel = np.std(all_folds_nmse_full_vel)

        summary_table_string += f"| Full-trajectory nMSE | {avg_full_pos:.6f} | {std_full_pos:.6f} | {avg_full_vel:.6f} | {std_full_vel:.6f} |\n"
        final_summary_writer.add_scalar('Summary/Mean_Full_Position_nMSE', avg_full_pos, 0)
        final_summary_writer.add_scalar('Summary/Std_Full_Position_nMSE', std_full_pos, 0)
        final_summary_writer.add_scalar('Summary/Mean_Full_Velocity_nMSE', avg_full_vel, 0)
        final_summary_writer.add_scalar('Summary/Std_Full_Velocity_nMSE', std_full_vel, 0)

    final_summary_writer.add_text('Final_Results_Summary_Table', summary_table_string, 0)
    print(summary_table_string)  # Also print to console for immediate visibility
    final_summary_writer.close()

    print(f"\n评估完成！TensorBoard 日志已保存至: {parent_log_dir_tb}")

#  ！！！Tensorboard log : tensorboard --logdir "D:\TransformerForwardDynamic\runs"
