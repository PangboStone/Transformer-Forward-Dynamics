import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# --- 从自定义模块导入 ---
from dataset import RobotDynamicsDataset, load_all_trajectories_from_file
from model import TransformerModel

# --- 1. 设置超参数 ---
# 环境与数据
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
SEQUENCE_LENGTH = 50
NUM_FOLDS = 10  # 执行完整的10轮交叉验证

# 模型维度
INPUT_SIZE = 21
OUTPUT_SIZE = 14

# Transformer模型专属超参数
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 3
DIM_FEEDFORWARD = 512

# 训练相关
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20  # 增加训练周期以获得更好的性能

# 误差计算函数
def calculate_nmse(predictions, targets):
    """计算归一化均方误差 (nMSE)"""
    if targets.shape[0] < 2: return float('inf')
    mse = np.mean((predictions - targets) ** 2)
    var_targets = np.var(targets)  # 计算所有输出的整体方差
    return mse / (var_targets + 1e-9)


def evaluate_model(model, test_loader, scaler, device, sequence_length):
    """
    在测试集上评估模型性能，包括单步和全轨迹预测。
    """
    model.eval()  # 设置模型为评估模式

    # --- 场景 A: 单步预测 (Step-by-step) ---
    predictions_sbs, actuals_sbs = [], []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            output = model(sequences)
            predictions_sbs.append(output.cpu().numpy())
            actuals_sbs.append(targets.cpu().numpy())

    predictions_sbs = np.concatenate(predictions_sbs, axis=0)
    actuals_sbs = np.concatenate(actuals_sbs, axis=0)
    nmse_sbs = calculate_nmse(predictions_sbs, actuals_sbs)

    # --- 场景 B: 全轨迹预测 (Full Trajectory) ---
    # 我们需要原始的测试轨迹数据来进行自回归预测
    test_inputs_scaled = test_loader.dataset.scaled_inputs[0]
    test_targets_raw = test_loader.dataset.targets[0]

    predictions_full = []
    # 使用轨迹的第一个序列作为初始输入
    current_sequence = torch.from_numpy(test_inputs_scaled[:sequence_length]).unsqueeze(0).to(device)

    with torch.no_grad():
        for t in range(len(test_targets_raw)):
            # 预测下一个时间步
            next_prediction = model(current_sequence)  # 形状 (1, output_size)
            predictions_full.append(next_prediction.cpu().numpy().flatten())

            # 构造下一个输入序列
            # 移除最旧的时间步
            next_sequence_scaled = current_sequence.cpu().numpy().squeeze(0)[1:]

            # 获取真实的下一个力矩 (已标准化)
            # 注意: t+sequence_length 必须在 test_inputs_scaled 的范围内
            if t + sequence_length < len(test_inputs_scaled):
                true_next_torque_scaled = test_inputs_scaled[t + sequence_length, 14:]
            else:  # 如果超出范围，则使用最后一个已知的力矩
                true_next_torque_scaled = test_inputs_scaled[-1, 14:]

            # 组合成新的输入特征 (位置+速度来自预测, 力矩来自真实数据)
            new_feature_unscaled = np.concatenate([
                next_prediction.cpu().numpy().flatten(),
                np.zeros(INPUT_SIZE - OUTPUT_SIZE)  # 占位符
            ])
            # 需要反向变换来获取未标准化的力矩，再组合，再变换，过程复杂
            # 简化方法：直接在标准化空间中组合
            new_feature_scaled = np.concatenate([
                next_prediction.cpu().numpy().flatten(),
                true_next_torque_scaled
            ])

            # 将新特征添加到序列末尾
            next_sequence_scaled = np.vstack([next_sequence_scaled, new_feature_scaled])
            current_sequence = torch.from_numpy(next_sequence_scaled).unsqueeze(0).to(device)

    nmse_full = calculate_nmse(np.array(predictions_full), test_targets_raw)

    return nmse_sbs, nmse_full


# --- 3. 主训练与评估流程 ---
if __name__ == '__main__':
    # --- 初始化日志 ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    experiment_name = f'transformer_full_cv_{current_time}'
    parent_log_dir_tb = os.path.join('venv/runs', experiment_name)

    log_dir_text = "venv/training_logs"
    os.makedirs(log_dir_text, exist_ok=True)
    log_file_path = os.path.join(log_dir_text, f'transformer_full_eval_{current_time}.log')

    model_save_dir = "venv/saved_models"
    os.makedirs(model_save_dir, exist_ok=True)

    log_file = open(log_file_path, 'w')

    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()


    # --- 记录配置 ---
    log_print("--- Transformer 模型 完整交叉验证 ---")
    # ... (在此处添加所有超参数的log_print)

    # --- 加载数据 ---
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

    # --- 用于存储每轮结果的列表 ---
    all_folds_nmse_sbs = []
    all_folds_nmse_full = []

    # --- 交叉验证循环 ---
    for i in range(NUM_FOLDS):
        fold_start_time = datetime.now()
        log_print(f"\n{'=' * 15} 交叉验证: 第 {i + 1}/{NUM_FOLDS} 轮 {'=' * 15}")

        # --- TensorBoard 日志 ---
        # writer = SummaryWriter(f'runs/transformer_fold_{i + 1}_{current_time}')
        writer = SummaryWriter(os.path.join(parent_log_dir_tb, f'Fold_{i+1}'))

        # 划分数据
        test_trajectories = [all_trajectories[i]]
        train_trajectories = all_trajectories[:i] + all_trajectories[i + 1:]

        # 数据准备
        combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
        scaler = StandardScaler().fit(combined_train_inputs)
        train_dataset = RobotDynamicsDataset(train_trajectories, SEQUENCE_LENGTH, scaler)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 初始化模型
        model = TransformerModel(INPUT_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_SIZE).to(
            DEVICE)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 训练循环
        log_print(f"开始训练...")
        for epoch in range(EPOCHS):
            model.train()
            total_loss = 0
            for sequences, targets in train_loader:
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
                predictions = model(sequences)
                loss = loss_function(predictions, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            log_print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {avg_loss:.6f}")
            writer.add_scalar(f'Loss/Train', avg_loss, epoch + 1)

        # --- 保存模型 ---
        model_path = os.path.join(model_save_dir, f'transformer_fold_{i + 1}.pth')
        torch.save(model.state_dict(), model_path)
        log_print(f"模型已保存至: {model_path}")

        # --- 评估模型 ---
        log_print("开始评估...")
        # 创建用于评估的DataLoader (batch_size可以大一些, shuffle=False)
        test_dataset = RobotDynamicsDataset(test_trajectories, SEQUENCE_LENGTH, scaler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

        # 加载模型并评估
        eval_model = TransformerModel(INPUT_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_SIZE).to(
            DEVICE)
        eval_model.load_state_dict(torch.load(model_path))

        nmse_sbs, nmse_full = evaluate_model(eval_model, test_loader, scaler, DEVICE, SEQUENCE_LENGTH)

        all_folds_nmse_sbs.append(nmse_sbs)
        all_folds_nmse_full.append(nmse_full)

        log_print(f"第 {i + 1} 轮评估结果 -> 单步nMSE: {nmse_sbs:.6f}, 全轨迹nMSE: {nmse_full:.6f}")
        writer.close()

    # --- 总结最终结果 ---
    log_print(f"\n\n{'=' * 20} 最终结果 {'=' * 20}")
    avg_nmse_sbs = np.mean(all_folds_nmse_sbs)
    std_nmse_sbs = np.std(all_folds_nmse_sbs)
    avg_nmse_full = np.mean(all_folds_nmse_full)
    std_nmse_full = np.std(all_folds_nmse_full)
    log_print(f"平均单步预测 nMSE ({NUM_FOLDS}轮): {avg_nmse_sbs:.6f} +/- {std_nmse_sbs:.6f}")
    log_print(f"平均全轨迹预测 nMSE ({NUM_FOLDS}轮): {avg_nmse_full:.6f} +/- {std_nmse_full:.6f}")

    log_file.close()
    print(f"\n完整评估完成！日志已保存至: {log_file_path}")

#  ！！！查看日志用命令tensorboard --logdir "D:\TransformerForwardDynamic\venv\runs"
