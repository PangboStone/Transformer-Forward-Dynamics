import torch
import torch.nn as nn
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import math
import time

from model_transformer import TransformerModel, PositionalEncoding
# --- 从我们自己的模块中导入 ---
# 为了让此脚本独立，我们直接从dataset.py导入加载函数
from dataset import load_all_trajectories_from_file


# --- 2. 辅助函数 ---
def predict_single_step(model, scaled_inputs, sequence_length, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(scaled_inputs) - sequence_length):
            sequence = scaled_inputs[i:i + sequence_length]
            sequence_tensor = torch.from_numpy(sequence).float().unsqueeze(0).to(device)
            prediction = model(sequence_tensor)
            predictions.append(prediction.cpu().numpy().flatten())
    return np.array(predictions)


def predict_full_trajectory(model, initial_sequence, true_torques_unscaled, input_scaler, output_scaler, device):
    model.eval()
    predictions_unscaled = []
    current_sequence_scaled = torch.from_numpy(initial_sequence).float().unsqueeze(0).to(device)
    with torch.no_grad():
        for t in range(len(true_torques_unscaled)):
            next_prediction_scaled = model(current_sequence_scaled)
            next_prediction_unscaled = output_scaler.inverse_transform(next_prediction_scaled.cpu().numpy())
            predictions_unscaled.append(next_prediction_unscaled.flatten())
            next_input_unscaled = np.concatenate([next_prediction_unscaled.flatten(), true_torques_unscaled[t]])
            next_input_scaled = input_scaler.transform(next_input_unscaled.reshape(1, -1))
            prev_sequence_scaled = current_sequence_scaled.cpu().numpy().squeeze(0)[1:]
            new_sequence_scaled = np.vstack([prev_sequence_scaled, next_input_scaled])
            current_sequence_scaled = torch.from_numpy(new_sequence_scaled).float().unsqueeze(0).to(device)
    return np.array(predictions_unscaled)


def plot_comparison(ground_truth, sbs_predictions, full_predictions, title_prefix):
    num_plots = 7
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(title_prefix, fontsize=20)
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


# --- 3. 交互式主流程 ---
if __name__ == '__main__':
    # --- 模型超参数 (必须与您加载的模型训练时使用的参数完全一致) ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # --- 交互式选择模型 ---
    model_dir = "saved_models"
    available_models = [f for f in os.listdir(model_dir) if f.endswith('.pth')]
    if not available_models:
        print(f"错误: 在 '{model_dir}' 文件夹中找不到任何模型文件 (.pth)。")
        exit()

    print("--- 请选择要加载的模型 ---")
    for i, model_name in enumerate(available_models):
        print(f"[{i + 1}] {model_name}")

    try:
        choice = int(input(f"请输入选择 (1-{len(available_models)}): ")) - 1
        if not 0 <= choice < len(available_models): raise ValueError
        model_path = os.path.join(model_dir, available_models[choice])
    except (ValueError, IndexError):
        print("无效输入，退出。")
        exit()

    # --- 交互式选择数据集 ---
    data_dir = "venv/ForwardDynamics"
    available_datasets = [f for f in os.listdir(data_dir) if f.endswith('.mat')]
    if not available_datasets:
        print(f"错误: 在 '{data_dir}' 文件夹中找不到任何数据集文件 (.mat)。")
        exit()

    print("\n--- 请选择要使用的数据集 ---")
    for i, data_name in enumerate(available_datasets):
        print(f"[{i + 1}] {data_name}")

    try:
        choice = int(input(f"请输入选择 (1-{len(available_datasets)}): ")) - 1
        if not 0 <= choice < len(available_datasets): raise ValueError
        data_file_path = os.path.join(data_dir, available_datasets[choice])
    except (ValueError, IndexError):
        print("无效输入，退出。")
        exit()

    # --- 加载数据并选择轨迹 ---
    all_trajectories = load_all_trajectories_from_file(data_file_path)
    print(f"\n--- 数据集 '{os.path.basename(data_file_path)}' 包含 {len(all_trajectories)} 条轨迹 ---")
    try:
        traj_choice = int(input(f"请输入要测试的轨迹编号 (1-{len(all_trajectories)}): ")) - 1
        if not 0 <= traj_choice < len(all_trajectories): raise ValueError
    except (ValueError, IndexError):
        print("无效输入，退出。")
        exit()

    # --- 数据准备 ---
    print("\n[1/4] 正在准备数据 scaler...")
    test_trajectory = all_trajectories[traj_choice]
    train_trajectories = all_trajectories[:traj_choice] + all_trajectories[traj_choice + 1:]
    input_scaler = StandardScaler().fit(np.concatenate([t['inputs'] for t in train_trajectories], axis=0))
    output_scaler = StandardScaler().fit(np.concatenate([t['targets'] for t in train_trajectories], axis=0))
    scaled_inputs = input_scaler.transform(test_trajectory['inputs'])
    ground_truth_outputs = test_trajectory['targets']
    print("数据准备完成。")

    # --- 加载模型 ---
    print(f"\n[2/4] 正在从 {model_path} 加载模型...")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 '{model_path}'。")
        exit()
    print("模型加载成功。")
    print("读取模型超参数")
    hyperparams = checkpoint['hyperparameters']
    for key, value in hyperparams.items():
        print(f"  - {key}: {value}")
    model = TransformerModel(
        input_size=hyperparams['INPUT_SIZE'],
        d_model=hyperparams['D_MODEL'],
        nhead=hyperparams['N_HEAD'],
        num_encoder_layers=hyperparams['NUM_ENCODER_LAYERS'],
        dim_feedforward=hyperparams['DIM_FEEDFORWARD'],
        output_size=hyperparams['OUTPUT_SIZE'],
        dropout=hyperparams['DROPOUT']
    ).to(DEVICE)
    print("加载模型权重")
    model.load_state_dict(checkpoint['model_state_dict'])
    SEQUENCE_LENGTH = hyperparams['SEQUENCE_LENGTH']


    # --- 执行预测 ---
    print("\n[3/4] 正在执行两种预测...")
    # 1. 单步预测
    predictions_scaled_sbs = predict_single_step(model, scaled_inputs, SEQUENCE_LENGTH, DEVICE)
    predictions_unscaled_sbs = output_scaler.inverse_transform(predictions_scaled_sbs)

    # 2. 全轨迹预测
    initial_sequence_scaled = scaled_inputs[:SEQUENCE_LENGTH]
    true_torques_unscaled = test_trajectory['inputs'][SEQUENCE_LENGTH:, 14:]
    predictions_unscaled_full = predict_full_trajectory(model, initial_sequence_scaled, true_torques_unscaled,
                                                        input_scaler, output_scaler, DEVICE)
    print("预测完成。")

    # --- 可视化 ---
    print("\n[4/4] 正在生成可视化图表...")
    aligned_ground_truth = ground_truth_outputs[SEQUENCE_LENGTH:]
    min_len = min(len(aligned_ground_truth), len(predictions_unscaled_sbs), len(predictions_unscaled_full))

    plot_comparison(
        aligned_ground_truth[:min_len],
        predictions_unscaled_sbs[:min_len],
        predictions_unscaled_full[:min_len],
        title_prefix=f"Prediction on '{os.path.basename(data_file_path)}' Trajectory {traj_choice + 1}"
    )
