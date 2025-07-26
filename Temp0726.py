import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import math
import time
from dataset import RobotDynamicsDataset, load_all_trajectories_from_file
from model_transformer import TransformerModel
from model_transformer_LinformerAttention import LinformerTransformerModel
DATA_FILE_PATH = 'venv/ForwardDynamics/KukaDirectDynamics.mat'

# 数据与模型超参数
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQUENCE_LENGTH = 50
K_PROJ_DIM = 10      # Linformer的投影维度k (k < SEQ_LEN)
INPUT_SIZE = 21
OUTPUT_SIZE = 14
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# 训练超参数
LEARNING_RATE = 0.00005
BATCH_SIZE = 64
EPOCHS = 50  # 增加训练周期以充分训练
VALIDATION_TRAJECTORY_INDEX = 9  # 使用最后一个轨迹作为固定的验证集


# --- 预测函数 ---
def predict_single_step(model, data_loader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for sequences, _ in data_loader:
            sequences = sequences.to(device)
            prediction = model(sequences)
            predictions.append(prediction.cpu().numpy())
    return np.concatenate(predictions, axis=0)
def predict_full_trajectory(model, initial_sequence, true_torques_unscaled, input_scaler, output_scaler, device):
    model.eval()
    predictions_unscaled = []
    current_sequence_scaled = torch.from_numpy(initial_sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        for t in range(len(true_torques_unscaled)):
            next_prediction_scaled = model(current_sequence_scaled)
            next_prediction_unscaled = output_scaler.inverse_transform(next_prediction_scaled.cpu().numpy())
            predictions_unscaled.append(next_prediction_unscaled.flatten())
            next_input_unscaled = np.concatenate([next_prediction_unscaled.flatten(), true_torques_unscaled[t]])
            next_input_scaled = input_scaler.transform(next_input_unscaled.reshape(1, -1))
            prev_sequence_scaled = current_sequence_scaled.cpu().numpy().squeeze(0)[1:]
            new_sequence_scaled = np.vstack([prev_sequence_scaled, next_input_scaled])
            current_sequence_scaled = torch.from_numpy(new_sequence_scaled).unsqueeze(0).to(device)
    return np.array(predictions_unscaled)

# 可视化
def plot_comparison(ground_truth, sbs_predictions, full_predictions):
    num_plots = 7
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(f'Prediction Comparison (Validation Trajectory {VALIDATION_TRAJECTORY_INDEX + 1})', fontsize=20)
    for i in range(num_plots):
        axes[i].plot(ground_truth[:, i], 'b-', label='Ground Truth', linewidth=2.5, alpha=0.8)
        axes[i].plot(sbs_predictions[:, i], 'lime', linestyle='--', label='Single-Step Prediction', linewidth=2)
        axes[i].plot(full_predictions[:, i], 'r:', label='Full Trajectory Prediction', linewidth=2)
        axes[i].set_title(f'Joint {i + 1} Position', fontsize=14)
        axes[i].set_ylabel('Position (rad)')
        axes[i].grid(True)
        axes[i].legend()
    axes[-1].set_xlabel('Time Steps')
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.show()


# --- 3. 主流程 ---
if __name__ == '__main__':
    def log_print(message):
        print(message)

    log_print(f"--- Transformer 大规模训练与评估实验 ---")
    # --- 数据加载与准备 ---
    log_print("\n[1/5] 正在加载和准备数据...")
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)
    val_trajectory = all_trajectories[VALIDATION_TRAJECTORY_INDEX]
    train_trajectories = all_trajectories[:VALIDATION_TRAJECTORY_INDEX] + all_trajectories[VALIDATION_TRAJECTORY_INDEX + 1:]

    input_scaler = StandardScaler().fit(np.concatenate([t['inputs'] for t in train_trajectories], axis=0))
    output_scaler = StandardScaler().fit(np.concatenate([t['targets'] for t in train_trajectories], axis=0))

    train_dataset = RobotDynamicsDataset(train_trajectories, SEQUENCE_LENGTH, input_scaler, output_scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataset = RobotDynamicsDataset([val_trajectory], SEQUENCE_LENGTH, input_scaler, output_scaler)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    log_print("数据准备完成。")

    # --- 模型训练 ---
    log_print("\n[2/5] 正在初始化和训练模型...")
    # model = TransformerModel(INPUT_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_SIZE, DROPOUT).to(DEVICE)
    model = LinformerTransformerModel(
        input_size=INPUT_SIZE,
        d_model=D_MODEL,
        nhead=N_HEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        output_size=OUTPUT_SIZE,
        seq_len=SEQUENCE_LENGTH,
        k=K_PROJ_DIM,
        dropout=DROPOUT
    )
    model.to(DEVICE)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    start_time = time.time()
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        for sequences, targets in train_loader:
            sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
            predictions = model(sequences)
            loss = loss_function(predictions, targets)
            optimizer.zero_grad();
            loss.backward();
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for sequences, targets in val_loader:
                sequences, targets = sequences.to(DEVICE), targets.to(DEVICE)
                predictions = model(sequences)
                loss = loss_function(predictions, targets)
                total_val_loss += loss.item()
        avg_val_loss = total_val_loss / len(val_loader)
        log_print(f"Epoch [{epoch + 1:02d}/{EPOCHS}], 训练损失: {avg_train_loss:.6f}, 验证损失: {avg_val_loss:.6f}")

    training_time = time.time() - start_time
    log_print(f"训练完成。总耗时: {training_time / 60:.2f} 分钟。")

    # --- 模型评估与预测 ---
    log_print("\n[3/5] 加载模型并进行预测...")

    # 1. 单步预测
    predictions_scaled_sbs = predict_single_step(model, val_loader, DEVICE)
    predictions_unscaled_sbs = output_scaler.inverse_transform(predictions_scaled_sbs)
    log_print("单步预测完成。")

    # 2. 全轨迹预测
    initial_sequence_scaled = input_scaler.transform(val_trajectory['inputs'])[:SEQUENCE_LENGTH]
    true_torques_unscaled = val_trajectory['inputs'][SEQUENCE_LENGTH:, 14:]
    predictions_unscaled_full = predict_full_trajectory(model, initial_sequence_scaled, true_torques_unscaled,
                                                        input_scaler, output_scaler, DEVICE)
    log_print("全轨迹预测完成。")

    # --- 数据收集 ---
    log_print("\n[4/5] 正在收集评估数据...")
    aligned_ground_truth = val_trajectory['targets'][SEQUENCE_LENGTH - 1:-1]
    min_len = min(len(aligned_ground_truth), len(predictions_unscaled_sbs), len(predictions_unscaled_full))
    # (此处可以添加nMSE等数值指标的计算和记录)

    # --- 结果可视化 ---
    log_print("\n[5/5] 正在生成可视化图表...")
    plot_comparison(
        aligned_ground_truth[:min_len],
        predictions_unscaled_sbs[:min_len],
        predictions_unscaled_full[:min_len],
    )

