import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# --- 从自定义模块导入 ---
# 注意：我们只需要数据加载函数，不需要PyTorch的Dataset类
from dataset import load_all_trajectories_from_file
from model_pcesn import PCESNpp

# --- 1. 设置超参数 ---
# 数据与交叉验证
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
NUM_FOLDS = 10  # 10轮交叉验证

# PC-ESN++ 模型参数
INPUT_SIZE = 21
OUTPUT_SIZE = 14
RESERVOIR_SIZE = 400   # 关键超参数 储备池大小
SPECTRAL_RADIUS = 0.9
SPARSITY = 0.1
LEAK_RATE = 0.5
REGULARIZATION_FACTOR = 1e-4
GHL_LEARNING_RATE = 1e-3

# 结果
# ==================== 最终结果 ====================
# 平均单步预测 nMSE (2轮交叉验证): 0.792272 +/- 0.047574
# 平均全轨迹预测 nMSE (2轮交叉验证): 22.648288 +/- 1.544850


# --- 辅助函数 ---
def calculate_nmse(predictions, targets):
    """计算归一化均方误差 (nMSE)"""
    mse = np.mean((predictions - targets) ** 2)
    var_targets = np.var(targets, axis=0)
    # 避免除以零
    nmse_per_output = mse / (var_targets + 1e-9)
    return np.mean(nmse_per_output)  # 返回所有输出维度的平均nMSE


# --- 2. 主训练与评估流程 ---
if __name__ == '__main__':
    # --- 初始化日志文件 ---
    log_dir = "venv/training_logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_file_path = os.path.join(log_dir, f'pcesn_reproduction_{current_time}.log')

    with open(log_file_path, 'w') as log_file:
        def log_print(message):
            print(message)
            log_file.write(message + '\n')
            log_file.flush()


        log_print("--- PC-ESN++ 复现模型 训练与评估 ---")
        log_print(f"开始时间: {current_time}")

        # 加载数据
        all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

        all_folds_nmse_sbs = []
        all_folds_nmse_full = []

        # --- 交叉验证循环 ---
        for i in range(NUM_FOLDS):
            log_print(f"\n{'=' * 15} 交叉验证: 第 {i + 1}/{NUM_FOLDS} 轮 {'=' * 15}")

            # 1. 划分训练集和测试集
            test_trajectory = all_trajectories[i]
            train_trajectories = all_trajectories[:i] + all_trajectories[i + 1:]

            # 2. 数据标准化
            combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
            scaler = StandardScaler().fit(combined_train_inputs)

            # 标准化训练数据和测试数据
            scaled_train_inputs = [scaler.transform(t['inputs']) for t in train_trajectories]
            scaled_test_inputs = scaler.transform(test_trajectory['inputs'])

            # 目标数据不需要标准化
            train_targets = [t['targets'] for t in train_trajectories]
            test_targets = test_trajectory['targets']

            # 3. 初始化模型
            model = PCESNpp(input_size=INPUT_SIZE,
                            reservoir_size=RESERVOIR_SIZE,
                            output_size=OUTPUT_SIZE,
                            spectral_radius=SPECTRAL_RADIUS,
                            sparsity=SPARSITY,
                            leak_rate=LEAK_RATE,
                            regularization_factor=REGULARIZATION_FACTOR,
                            ghl_learning_rate = GHL_LEARNING_RATE)

            # 4. 训练阶段 (在线学习)
            log_print("\n--- 开始训练 ---")
            all_training_losses = []  # 用于收集所有训练样本的loss
            for traj_idx, inputs in enumerate(scaled_train_inputs):
                targets = train_targets[traj_idx]
                trajectory_losses = []
                log_print(f"正在训练第 {traj_idx + 1}/{len(scaled_train_inputs)} 个轨迹...")
                # 逐个样本进行训练
                for t in range(len(inputs)):
                    u_t = inputs[t].reshape(-1, 1)
                    target_t = targets[t].reshape(-1, 1)
                    loss = model.train_step(u_t, target_t)
                    trajectory_losses.append(loss)

                avg_traj_loss = np.mean(trajectory_losses)
                log_print(f"平均训练MSE: {avg_traj_loss:.6f}")
                all_training_losses.extend(trajectory_losses)

            overall_avg_train_loss = np.mean(all_training_losses)
            log_print("--- 训练完成 ---")

            # 5. 评估阶段
            log_print("\n--- 开始评估 ---")
            # a) 单步预测 (Step-by-step prediction)
            # 重置模型内部状态以进行公平评估
            model.r_state.fill(0)
            predictions_step_by_step = []
            for t in range(len(scaled_test_inputs)):
                u_t = scaled_test_inputs[t].reshape(-1, 1)
                pred = model.predict(u_t)
                predictions_step_by_step.append(pred.flatten())

            nmse_sbs = calculate_nmse(np.array(predictions_step_by_step), test_targets)
            log_print(f"单步预测 nMSE: {nmse_sbs:.6f}")
            all_folds_nmse_sbs.append(nmse_sbs)

            # b) 全轨迹预测
            model.r_state.fill(0)
            predictions_full = []

            # 使用测试集的第一个真实输入作为起点
            current_input = scaled_test_inputs[0].reshape(-1, 1)

            for t in range(len(scaled_test_inputs)):
                # 使用当前输入进行预测
                predicted_output = model.predict(current_input)  # 形状 (14, 1)
                predictions_full.append(predicted_output.flatten())

                # 构造下一个时间步的输入
                # 使用预测出的位置和速度，但使用真实的力矩
                if t < len(scaled_test_inputs) - 1:
                    predicted_pos_vel = predicted_output.flatten()  # 形状 (14,)

                    # 反向标准化，得到原始尺度的位置和速度
                    # 注意：scaler是对21维输入进行训练的，我们需要构造一个假的21维向量来反向变换
                    # 这里是一个简化，更精确的做法是分别对pos/vel/torque部分进行scaler
                    # 但为了流程一致性，我们先用一个通用的scaler
                    dummy_input_for_inverse = np.zeros(INPUT_SIZE)
                    dummy_input_for_inverse[:14] = predicted_pos_vel
                    unscaled_pred = scaler.inverse_transform(dummy_input_for_inverse.reshape(1, -1)).flatten()

                    # 获取真实的力矩
                    true_torque_next = test_trajectory['inputs'][t + 1, 14:21]  # 未标准化的力矩

                    # 构造下一个输入（未标准化的）
                    next_input_unscaled = np.concatenate([unscaled_pred[:14], true_torque_next])

                    # 对下一个输入进行标准化，作为下一次循环的输入
                    current_input = scaler.transform(next_input_unscaled.reshape(1, -1)).reshape(-1, 1)

            nmse_full = calculate_nmse(np.array(predictions_full), test_targets)
            log_print(f"全轨迹预测 nMSE: {nmse_full:.6f}")
            all_folds_nmse_full.append(nmse_full)

        # --- 总结所有交叉验证的结果 ---
        log_print(f"\n\n{'=' * 20} 最终结果 {'=' * 20}")
        if all_folds_nmse_sbs:
            avg_nmse_sbs = np.mean(all_folds_nmse_sbs)
            std_nmse_sbs = np.std(all_folds_nmse_sbs)
            log_print(f"平均单步预测 nMSE ({NUM_FOLDS}轮交叉验证): {avg_nmse_sbs:.6f} +/- {std_nmse_sbs:.6f}")
        if all_folds_nmse_full:
            avg_nmse_full = np.mean(all_folds_nmse_full)
            std_nmse_full = np.std(all_folds_nmse_full)
            log_print(f"平均全轨迹预测 nMSE ({NUM_FOLDS}轮交叉验证): {avg_nmse_full:.6f} +/- {std_nmse_full:.6f}")

    print(f"\n评估完成！日志已保存至: {log_file_path}")
