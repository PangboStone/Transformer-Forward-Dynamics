import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import time

# --- 从自定义模块导入 ---
from dataset import load_all_trajectories_from_file
from model_pcesn import PCESNpp

# --- 1. 设置超参数 ---
# 使用我们之前调优找到的最佳或接近最佳的参数
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
RESERVOIR_SIZE = 400
SPECTRAL_RADIUS = 0.7
SPARSITY = 0.7
LEAK_RATE = 0.3
REGULARIZATION_FACTOR = 1e-2
GHL_LEARNING_RATE = 1e-5  # 适当调整GHL学习率

INPUT_SIZE = 21
OUTPUT_SIZE = 14


# --- 辅助函数 ---
def calculate_nmse(predictions, targets):
    """计算单个部分的归一化均方误差 (nMSE)"""
    if targets.shape[0] < 2: return float('inf')
    mse = np.mean((predictions - targets) ** 2)
    var_targets = np.var(targets)
    return mse / (var_targets + 1e-9)


# --- 2. 主复现流程 ---
if __name__ == '__main__':
    # --- 初始化日志 ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = f'pcesn_convergence_repro_{current_time}'
    log_dir_tb = os.path.join('runs', experiment_name)
    writer = SummaryWriter(log_dir_tb)
    print(f"--- 开始复现论文收敛图实验 ---")
    print(f"TensorBoard 日志将保存在: {log_dir_tb}")

    # --- 加载并划分数据 ---
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)
    # 固定最后一个轨迹作为测试集，前9个作为训练池
    test_trajectory = all_trajectories[9]
    training_pool = all_trajectories[:9]

    # --- 增量训练循环 ---
    # 循环将使用 1, 2, 3, ..., 9 个轨迹进行训练
    for num_train_trajectories in range(1, len(training_pool) + 1):
        start_time = time.time()
        print(f"\n--- 正在使用 {num_train_trajectories} 个轨迹进行训练 ---")

        # 1. 准备当前轮次的训练数据
        current_train_trajectories = training_pool[:num_train_trajectories]

        # 2. 数据标准化 (只在当前训练集上拟合)
        combined_train_inputs = np.concatenate([traj['inputs'] for traj in current_train_trajectories], axis=0)
        scaler = StandardScaler().fit(combined_train_inputs)

        # 标准化训练和测试数据
        scaled_train_inputs = [scaler.transform(t['inputs']) for t in current_train_trajectories]
        scaled_test_inputs = scaler.transform(test_trajectory['inputs'])
        train_targets = [t['targets'] for t in current_train_trajectories]
        test_targets = test_trajectory['targets']

        # 3. 初始化并训练模型
        model = PCESNpp(input_size=INPUT_SIZE, reservoir_size=RESERVOIR_SIZE, output_size=OUTPUT_SIZE,
                        spectral_radius=SPECTRAL_RADIUS, sparsity=SPARSITY, leak_rate=LEAK_RATE,
                        regularization_factor=REGULARIZATION_FACTOR, ghl_learning_rate=GHL_LEARNING_RATE)

        for traj_idx, inputs in enumerate(scaled_train_inputs):
            targets = train_targets[traj_idx]
            for t in range(len(inputs)):
                model.train_step(inputs[t].reshape(-1, 1), targets[t].reshape(-1, 1))

        print("模型训练完成。")

        # 4. 评估阶段
        print("正在评估模型...")
        # a) 单步预测 (复现 Figure 4)
        model.r_state.fill(0)
        predictions_sbs = np.array([model.predict(u_t.reshape(-1, 1)).flatten() for u_t in scaled_test_inputs])
        # *** 只提取基座关节的位置数据 (第0列) 进行比较 ***
        pred_base_joint_pos_sbs = predictions_sbs[:, 0]
        target_base_joint_pos_sbs = test_targets[:, 0]
        nmse_sbs_single_joint = calculate_nmse(pred_base_joint_pos_sbs, target_base_joint_pos_sbs)

        # b) 全轨迹预测 (复现 Figure 5)
        model.r_state.fill(0)
        predictions_full = []
        current_input = scaled_test_inputs[0].reshape(-1, 1)
        for t in range(len(scaled_test_inputs)):
            predicted_output = model.predict(current_input)
            predictions_full.append(predicted_output.flatten())
            if t < len(scaled_test_inputs) - 1:
                # 简化版：直接在标准化空间中构造下一个输入
                true_next_torque_scaled = scaled_test_inputs[t + 1, 14:]
                next_input_feature = np.concatenate([predicted_output.flatten(), true_next_torque_scaled])
                current_input = next_input_feature.reshape(-1, 1)
        pred_base_joint_pos_full = np.array(predictions_full)[:, 0]
        target_base_joint_pos_full = test_targets[:, 0]
        nmse_full_single_joint = calculate_nmse(pred_base_joint_pos_full, target_base_joint_pos_full)

        # 5. 将结果写入TensorBoard
        # X轴是训练所用的轨迹数量
        writer.add_scalar('Convergence/StepByStep_nMSE', nmse_sbs_single_joint, num_train_trajectories)
        writer.add_scalar('Convergence/FullTrajectory_nMSE', nmse_full_single_joint, num_train_trajectories)

        end_time = time.time()
        print(f"完成。单步nMSE: {nmse_sbs_single_joint:.6f}, 全轨迹nMSE: {nmse_full_single_joint:.6f} (耗时: {end_time - start_time:.2f}s)")

    writer.close()
    print(f"\n实验完成！请启动 TensorBoard 查看收敛曲线图。")
    print(f"TensorBoard 日志已保存至: {log_dir_tb}")

