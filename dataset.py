import scipy.io
import numpy as np
import os
import torch
from torch.utils.data import Dataset


def load_and_parse_trajectory(file_path):
    """
    从单个 .mat 文件加载所有轨迹并解析。

    参数:
    file_path (str): .mat 文件的完整路径。

    返回:
    一个包含10个轨迹数据的列表，每个元素是一个字典。
    """
    if not os.path.exists(file_path):
        print(f"错误: 文件不存在 -> {file_path}")
        return []

    try:
        mat_data = scipy.io.loadmat(file_path)
        # print(f"文件 '{os.path.basename(file_path)}' 中包含的变量: {mat_data.keys()}")

        all_parsed_trajectories = []

        for i in range(1, 11):
            variable_name = f'baxtertraj{i}'  # 构建变量名: 'baxtertraj1', 'baxtertraj2'

            if variable_name in mat_data:
                # print(f"\n正在处理变量: '{variable_name}'...")
                data_array = mat_data[variable_name]

                # --- 根据 readme 文件内容，对数据进行切片 ---
                inputs_p = data_array[:, 0:7]
                inputs_v = data_array[:, 7:14]
                inputs_tau = data_array[:, 14:21]
                targets_p = data_array[:, 21:28]
                targets_v = data_array[:, 28:35]

                all_inputs = np.concatenate([inputs_p, inputs_v, inputs_tau], axis=1)
                all_targets = np.concatenate([targets_p, targets_v], axis=1)

                # print(f"'{variable_name}' 解析后的输入数据形状: {all_inputs.shape}")
                # print(f"'{variable_name}' 解析后的目标数据形状: {all_targets.shape}")

                trajectory_dict = {
                    'name': variable_name,
                    'inputs': all_inputs,
                    'targets': all_targets
                }
                all_parsed_trajectories.append(trajectory_dict)
            else:
                print(f"警告: 在文件中找不到变量 '{variable_name}'。")

        return all_parsed_trajectories

    except Exception as e:
        print(f"加载文件时发生错误: {e}")
        return []

def load_all_trajectories_from_file(file_path):
    """
    从单个 .mat 文件加载所有轨迹。
    （这是我们之前完善的函数，现在作为数据加载的辅助函数）
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"错误: 文件不存在 -> {file_path}")

    mat_data = scipy.io.loadmat(file_path)
    all_parsed_trajectories = []

    # 假设变量名前缀是'baxter'或'kuka'
    prefix = os.path.basename(file_path).split('traj')[0].lower().replace('directdynamics.mat', '')

    for i in range(1, 11):
        variable_name = f'{prefix}traj{i}'
        if variable_name in mat_data:
            data_array = mat_data[variable_name]

            inputs_p = data_array[:, 0:7]
            inputs_v = data_array[:, 7:14]
            inputs_tau = data_array[:, 14:21]
            targets_p = data_array[:, 21:28]
            targets_v = data_array[:, 28:35]

            all_inputs = np.concatenate([inputs_p, inputs_v, inputs_tau], axis=1)
            all_targets = np.concatenate([targets_p, targets_v], axis=1)

            trajectory_dict = {
                'inputs': all_inputs.astype(np.float32),
                'targets': all_targets.astype(np.float32)
            }
            all_parsed_trajectories.append(trajectory_dict)

    return all_parsed_trajectories

class RobotDynamicsDataset(Dataset):
    """
    自定义PyTorch数据集，用于处理机器人动力学数据。
    这个类负责将轨迹数据转换为(序列, 目标)的样本对。
    """

    def __init__(self, trajectories, sequence_length, scaler):
        """
        初始化数据集。

        参数:
        trajectories (list): 一个包含多个轨迹字典的列表。
                             每个字典包含 'inputs' 和 'targets'。
        sequence_length (int): 输入序列的窗口长度。
        scaler (sklearn.preprocessing.StandardScaler): 用于标准化输入数据的scaler对象。
        """
        super().__init__()
        self.sequence_length = sequence_length
        self.trajectories = trajectories
        self.scaler = scaler

        # --- 数据处理核心逻辑 ---
        # 1. 对所有轨迹的输入数据进行标准化
        self.scaled_inputs = [self.scaler.transform(traj['inputs']) for traj in self.trajectories]
        # 注意：目标数据通常不需要标准化，因为我们希望模型直接预测原始值。
        self.targets = [traj['targets'] for traj in self.trajectories]

        # 2. 创建一个索引映射，方便__getitem__快速查找
        # 这个列表将存储每个样本的 (轨迹索引, 在该轨迹中的起始位置)
        self.indices = []
        for traj_idx, traj_inputs in enumerate(self.scaled_inputs):
            num_samples_in_traj = len(traj_inputs) - self.sequence_length
            if num_samples_in_traj > 0:
                for i in range(num_samples_in_traj):
                    self.indices.append((traj_idx, i))

    def __len__(self):
        """返回数据集中样本的总数。"""
        return len(self.indices)

    def __getitem__(self, idx):
        """
        根据索引idx，获取一个 (输入序列, 目标) 样本对。
        这是DataLoader在后台调用的函数。
        """
        # 1. 从索引映射中找到该样本所在的轨迹和起始位置
        traj_idx, start_idx = self.indices[idx]

        # 2. 根据起始位置和序列长度，切片出输入序列
        end_idx = start_idx + self.sequence_length
        input_sequence = self.scaled_inputs[traj_idx][start_idx:end_idx]

        # 3. 目标值对应的是输入序列结束时的那个时间步的目标
        target = self.targets[traj_idx][end_idx - 1]

        # 4. 将Numpy数组转换为PyTorch张量
        return torch.from_numpy(input_sequence), torch.from_numpy(target)


if __name__ == '__main__':
    from sklearn.preprocessing import StandardScaler

    # 1. 加载数据
    file_path = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'  # 修改为你的实际路径
    all_trajectories = load_all_trajectories_from_file(file_path)

    # 2. 模拟交叉验证的数据划分（用9个轨迹训练，1个测试）
    train_trajectories = all_trajectories[:9]
    test_trajectories = all_trajectories[9:]

    # 3. 准备并拟合Scaler (只在训练集上！)
    # 将所有训练轨迹的输入数据拼接起来
    combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)

    scaler = StandardScaler()
    print("正在使用训练数据拟合Scaler...")
    scaler.fit(combined_train_inputs)
    print("Scaler拟合完成。")

    # 4. 创建训练数据集和测试数据集的实例
    SEQUENCE_LENGTH = 50

    print("\n正在创建训练数据集...")
    train_dataset = RobotDynamicsDataset(train_trajectories, SEQUENCE_LENGTH, scaler)
    print("正在创建测试数据集...")
    test_dataset = RobotDynamicsDataset(test_trajectories, SEQUENCE_LENGTH, scaler)

    print(f"\n训练集中的样本总数: {len(train_dataset)}")
    print(f"测试集中的样本总数: {len(test_dataset)}")

    # 5. 取出第一个训练样本并检查其形状
    if len(train_dataset) > 0:
        first_sequence, first_target = train_dataset[0]
        print("\n取出一个训练样本进行检查:")
        print(f"输入序列的形状: {first_sequence.shape}")  # 应该为 (SEQUENCE_LENGTH, 21)
        print(f"目标值的形状: {first_target.shape}")  # 应该为 (14,)