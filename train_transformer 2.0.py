import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import time

from dataset import RobotDynamicsDataset, load_all_trajectories_from_file
from model_transformer import TransformerModel

# --- 1. 超参数设置 ---
# 环境与数据
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
SEQUENCE_LENGTH = 50
NUM_FOLDS = 10  # 执行完整的10轮交叉验证

# 模型维度
INPUT_SIZE = 21
OUTPUT_SIZE = 14

# Transformer模型参数
D_MODEL = 256
N_HEAD = 8
NUM_ENCODER_LAYERS = 4
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# 训练相关
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 20  # 建议使用20-30个周期以充分训练


# --- 2. 辅助函数 ---
def calculate_nmse(predictions, targets):
    """分别计算位置和速度的归一化均方误差 (nMSE)"""
    if targets.shape[0] < 2: return float('inf'), float('inf')

    pred_pos, pred_vel = predictions[:, :7], predictions[:, 7:]
    target_pos, target_vel = targets[:, :7], targets[:, 7:]

    mse_pos = np.mean((pred_pos - target_pos) ** 2)
    var_pos = np.var(target_pos)
    nmse_pos = mse_pos / (var_pos + 1e-9)

    mse_vel = np.mean((pred_vel - target_vel) ** 2)
    var_vel = np.var(target_vel)
    nmse_vel = mse_vel / (var_vel + 1e-9)

    return nmse_pos, nmse_vel


def evaluate_model(model_path, test_loader, device):
    """
    加载已保存的模型，并在测试集上进行评估。
    """
    # 初始化一个新的模型实例
    eval_model = TransformerModel(INPUT_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_SIZE,
                                  DROPOUT).to(device)
    # 加载已保存的权重
    eval_model.load_state_dict(torch.load(model_path))
    eval_model.eval()  # 设置为评估模式

    predictions_sbs, actuals_sbs = [], []
    with torch.no_grad():
        for sequences, targets in test_loader:
            sequences, targets = sequences.to(device), targets.to(device)
            output = eval_model(sequences)
            predictions_sbs.append(output.cpu().numpy())
            actuals_sbs.append(targets.cpu().numpy())

    predictions_sbs = np.concatenate(predictions_sbs, axis=0)
    actuals_sbs = np.concatenate(actuals_sbs, axis=0)

    nmse_pos_sbs, nmse_vel_sbs = calculate_nmse(predictions_sbs, actuals_sbs)

    # 全轨迹预测的实现较为复杂，暂时作为占位符
    nmse_pos_full, nmse_vel_full = float('nan'), float('nan')

    return nmse_pos_sbs, nmse_vel_sbs, nmse_pos_full, nmse_vel_full


# --- 3. 主流程 ---
if __name__ == '__main__':
    # --- 初始化日志和模型保存目录 ---
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    experiment_name = f'transformer_eval_{current_time}'

    log_dir_text = "training_logs"
    model_save_dir = "saved_models"
    parent_log_dir_tb = os.path.join('runs', experiment_name)

    os.makedirs(log_dir_text, exist_ok=True)
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(parent_log_dir_tb, exist_ok=True)

    log_file_path = os.path.join(log_dir_text, f'{experiment_name}.log')
    log_file = open(log_file_path, 'w')


    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()


    log_print(f"--- Transformer Evaluation ---")
    log_print(f"实验名称: {experiment_name}")
    # ... 在此记录所有超参数 ...

    # --- 加载数据 ---
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

    results_sbs_pos, results_sbs_vel = [], []

    # --- 交叉验证循环 ---
    for i in range(NUM_FOLDS):
        fold_start_time = time.time()
        log_print(f"\n{'=' * 15} CorssValidation: Fold {i + 1}/{NUM_FOLDS}  {'=' * 15}")

        writer = SummaryWriter(os.path.join(parent_log_dir_tb, f'Fold_{i + 1}'))

        # 数据划分与准备
        test_trajectories = [all_trajectories[i]]
        train_trajectories = all_trajectories[:i] + all_trajectories[i + 1:]

        input_scaler = StandardScaler().fit(np.concatenate([t['inputs'] for t in train_trajectories], axis=0))
        output_scaler = StandardScaler().fit(np.concatenate([t['targets'] for t in train_trajectories], axis=0))

        train_dataset = RobotDynamicsDataset(train_trajectories, SEQUENCE_LENGTH, input_scaler, output_scaler)
        train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        # 初始化模型
        model = TransformerModel(INPUT_SIZE, D_MODEL, N_HEAD, NUM_ENCODER_LAYERS, DIM_FEEDFORWARD, OUTPUT_SIZE,
                                 DROPOUT).to(DEVICE)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

        # 训练循环
        log_print("Training...")
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
            if (epoch + 1) % 5 == 0:  # 每5个周期打印一次日志
                log_print(f"  Epoch [{epoch + 1}/{EPOCHS}], Average loss: {avg_loss:.6f}")
            writer.add_scalar('Training_Loss/per_epoch', avg_loss, epoch + 1)

        # --- 保存模型 ---
        model_path = os.path.join(model_save_dir, f'{experiment_name}_fold_{i + 1}.pth')
        torch.save(model.state_dict(), model_path)
        log_print(f"Training finished, model saved to : {model_path}")

        # --- 评估模型 ---
        log_print("Evaluating...")
        test_dataset = RobotDynamicsDataset(test_trajectories, SEQUENCE_LENGTH, input_scaler, output_scaler)
        test_loader = DataLoader(dataset=test_dataset, batch_size=512, shuffle=False)

        nmse_pos, nmse_vel, _, _ = evaluate_model(model_path, test_loader, DEVICE)

        results_sbs_pos.append(nmse_pos)
        results_sbs_vel.append(nmse_vel)

        log_print(f" {i + 1} fold -> single step Pos nMSE: {nmse_pos:.6f}, single step Vel nMSE: {nmse_vel:.6f}")
        fold_end_time = time.time()
        log_print(f"time consumption: {(fold_end_time - fold_start_time) / 60:.2f} minutes")
        writer.close()

    # --- 总结最终结果 ---
    log_print(f"\n\n{'=' * 20} Final Results {'=' * 20}")
    avg_pos = np.mean(results_sbs_pos)
    std_pos = np.std(results_sbs_pos)
    avg_vel = np.mean(results_sbs_vel)
    std_vel = np.std(results_sbs_vel)

    log_print(f"Average single step prediction Position nMSE (fold{NUM_FOLDS}): {avg_pos:.6f} +/- {std_pos:.6f}")
    log_print(f"Average single step prediction Velocity nMSE (fold{NUM_FOLDS}轮): {avg_vel:.6f} +/- {std_vel:.6f}")

    log_file.close()
    print(f"\n完整评估完成！请查看日志文件和TensorBoard。")
