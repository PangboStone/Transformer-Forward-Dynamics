import os
import torch
from datetime import datetime
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from dataset import RobotDynamicsDataset, load_all_trajectories_from_file
from model import LSTMBaseline


# --- 1. 设置超参数 ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"正在使用设备: {DEVICE}")

# 数据相关
DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'  # 修改为你的数据文件路径
SEQUENCE_LENGTH = 50

# 训练相关
LEARNING_RATE = 0.001
BATCH_SIZE = 64
EPOCHS = 10 # 先用少量epoch测试流程

# 模型相关
INPUT_SIZE = 21
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_SIZE = 14

# --- 2. 主训练流程 ---
if __name__ == '__main__':
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = f'runs/lstm_baseline_{current_time}'
    print(f"TensorBoard 日志将保存在: {log_dir}")
    writer = SummaryWriter()

    log_dir_text = "venv/training_logs"
    os.makedirs(log_dir_text, exist_ok=True)
    log_file_path = os.path.join(log_dir_text, f'lstm_baseline_{current_time}.log')
    log_file = open(log_file_path, 'w')

    def log_print(message):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

    print(f"正在使用设备: {DEVICE}")

    # 加载所有轨迹数据
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

    # --- 我们先只跑一轮交叉验证来测试流程 ---
    log_print("\n--- 开始交叉验证: 第1轮/共10轮 ---")
    train_trajectories = all_trajectories[1:]  # 用后9个轨迹训练
    test_trajectories = all_trajectories[:1]  # 用第1个轨迹测试

    # 准备并拟合Scaler
    combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
    scaler = StandardScaler().fit(combined_train_inputs)

    # 创建Dataset和DataLoader
    train_dataset = RobotDynamicsDataset(train_trajectories, SEQUENCE_LENGTH, scaler)
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 初始化模型、损失函数和优化器
    model = LSTMBaseline(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE).to(DEVICE)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. 训练循环 ---
    print(f"\n开始在 {len(train_dataset)} 个训练样本上进行训练...")
    for epoch in range(EPOCHS):
        model.train()  # 设置模型为训练模式
        total_loss = 0
        for batch_idx, (sequences, targets) in enumerate(train_loader):
            # 将数据移动到指定设备 (CPU或GPU)
            sequences = sequences.to(DEVICE)
            targets = targets.to(DEVICE)

            # 前向传播
            predictions = model(sequences)
            loss = loss_function(predictions, targets)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        log_print(f"Epoch [{epoch + 1}/{EPOCHS}], 平均损失 (Loss): {avg_loss:.6f}")
        # 写入TensorBoard
        writer.add_scalar('Loss/Train', avg_loss, epoch + 1)
    writer.flush()
    writer.close()
log_print("\n基线模型训练流程测试完成！")

#  ！！！查看日志用命令tensorboard --logdir "D:\TransformerForwardDynamic\venv\runs"
