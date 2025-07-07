import scipy.io
import numpy as np
import os


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
        print(f"文件 '{os.path.basename(file_path)}' 中包含的变量: {mat_data.keys()}")

        all_parsed_trajectories = []

        # 根据我们从keys中看到的信息，循环1到10来构建变量名
        for i in range(1, 11):
            variable_name = f'baxtertraj{i}'  # 构建变量名，如 'baxtertraj1', 'baxtertraj2'

            if variable_name in mat_data:
                print(f"\n正在处理变量: '{variable_name}'...")
                data_array = mat_data[variable_name]

                # --- 根据 readme 文件内容，对数据进行切片 ---
                inputs_p = data_array[:, 0:7]
                inputs_v = data_array[:, 7:14]
                inputs_tau = data_array[:, 14:21]
                targets_p = data_array[:, 21:28]
                targets_v = data_array[:, 28:35]

                all_inputs = np.concatenate([inputs_p, inputs_v, inputs_tau], axis=1)
                all_targets = np.concatenate([targets_p, targets_v], axis=1)

                print(f"'{variable_name}' 解析后的输入数据形状: {all_inputs.shape}")
                print(f"'{variable_name}' 解析后的目标数据形状: {all_targets.shape}")

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

# --- 主程序 ---
if __name__ == '__main__':

    data_folder_path = 'D:/TransformerForwardDynamic/venv/dataset'

    # 加载第一个轨迹为例
    file_name = 'BaxterDirectDynamics.mat'  # 假设文件名是这样的
    full_file_path = os.path.join(data_folder_path, file_name)

    baxter_trajectories =load_and_parse_trajectory(full_file_path)

    if baxter_trajectories:
        print(f"\n数据加载和解析成功！总共加载了 {len(baxter_trajectories)} 个轨迹。")
        # 例如，你可以访问第一个轨迹的数据：
        first_trajectory_inputs = baxter_trajectories[0]['inputs']
        print(f"第一个轨迹 (baxtertraj1) 的输入数据形状: {first_trajectory_inputs.shape}")
