import numpy as np
import cupy as cp
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time
import json

# --- 从自定义模块导入 ---
from dataset import load_all_trajectories_from_file
from model_pcesn_DHextend import PCESNpp  # 导入我们更新后的模型


# --- 1. 实验配置与超参数 ---
class Config:
    DATA_FILE_PATH = 'venv/ForwardDynamics/KukaDirectDynamics.mat'  # <-- 注意：使用KUKA数据
    NUM_FOLDS = 1
    GENERATE_HEATMAP = True
    # 模型超参数
    INPUT_SIZE = 21
    OUTPUT_SIZE = 14
    RESERVOIR_SIZE = 400
    SPECTRAL_RADIUS = 0.9
    SPARSITY = 0.76
    LEAK_RATE = 0.6
    REGULARIZATION_FACTOR = 1e-4
    GHL_ETA = 1e-5
    GHL_DECAY_STEPS = 5000
    # 物理参数 (使用您提供的KUKA LWR4+参数)
    PHYSICS_PARAMS = {
        'dh_alpha': [0, -np.pi / 2, np.pi / 2, np.pi / 2, -np.pi / 2, -np.pi / 2, np.pi / 2],
        'dh_a': [0, 0, 0, 0, 0, 0, 0],
        'dh_d': [0.310, 0, 0.400, 0, 0.390, 0, 0.078],
        'link_masses': [2.0, 2.0, 2.0, 2.0, 1.5, 1.0, 0.5]
    }


# --- 2. 核心功能函数 ---
def get_physics_vector(params):
    """根据物理参数配置，创建并返回静态物理向量。"""
    vector = []
    for key in sorted(params.keys()):  # 排序以保证顺序一致
        vector.extend(params[key])
    return np.array(vector, dtype=np.float32).reshape(-1, 1)


def run_single_fold(fold_idx, train_trajs, test_traj, config, physics_vector_gpu):
    """执行单轮交叉验证的训练和评估。"""
    print(f"\n--- 开始处理第 {fold_idx + 1}/{config.NUM_FOLDS} 折 ---")

    # 1. 数据标准化
    input_scaler = StandardScaler().fit(np.concatenate([t['inputs'] for t in train_trajs]))

    scaled_train_inputs = [input_scaler.transform(t['inputs']) for t in train_trajs]
    scaled_test_inputs = input_scaler.transform(test_traj['inputs'])
    train_targets = [t['targets'] for t in train_trajs]
    test_targets = test_traj['targets']

    # 2. 初始化模型
    model = PCESNpp(
        input_size=config.INPUT_SIZE, reservoir_size=config.RESERVOIR_SIZE,
        output_size=config.OUTPUT_SIZE, physics_vector_size=len(physics_vector_gpu),
        spectral_radius=config.SPECTRAL_RADIUS, sparsity=config.SPARSITY,
        leak_rate=config.LEAK_RATE, regularization_factor=config.REGULARIZATION_FACTOR,
        ghl_eta=config.GHL_ETA, ghl_decay_steps=config.GHL_DECAY_STEPS
    )

    # 3. 训练
    print("正在训练...")
    start_time = time.time()
    train_state_history = []
    for inputs, targets in zip(scaled_train_inputs, train_targets):
        for t in range(len(inputs)):
            model.train_step(inputs[t].reshape(-1, 1), targets[t].reshape(-1, 1), physics_vector_gpu)
            train_state_history.append(cp.asnumpy(model.r_state))
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f}s")
    visualize_reservoir_heatmap(train_state_history,"Reservoir Dynamics Record during Training")

    model.r_state.fill(0)

    # 4. 评估
    print("正在评估...")
    # a) 单步预测
    # model.r_state.fill(0)
    '''目前尚不清楚是否需要重置储蓄库状态'''
    predictions_sbs = np.array(
        [model.predict(u_t.reshape(-1, 1), physics_vector_gpu).flatten() for u_t in scaled_test_inputs])

    # b) 全轨迹预测
    model.r_state.fill(0)
    predictions_full = []
    full_pred_state_history = []
    current_input = scaled_test_inputs[0].reshape(-1, 1)
    for t in range(len(scaled_test_inputs)):
        predicted_output = model.predict(current_input, physics_vector_gpu)
        predictions_full.append(predicted_output.flatten())
        full_pred_state_history.append(cp.asnumpy(model.r_state))
        if t < len(scaled_test_inputs) - 1:
            next_input_unscaled = np.concatenate([predicted_output.flatten(), test_traj['inputs'][t + 1, 14:]])
            current_input = input_scaler.transform(next_input_unscaled.reshape(1, -1)).reshape(-1, 1)
    predictions_full = np.array(predictions_full)
    visualize_reservoir_heatmap(full_pred_state_history, "Reservoir Dynamics Record during Full Trajectory Prediction")
    return predictions_sbs, predictions_full, test_targets, training_time, model, scaled_test_inputs


def summarize_and_log_results(all_results, config, experiment_name):
    """汇总所有交叉验证的结果并记录。"""
    log_dir = "training_logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{experiment_name}.json")

    # 此处可以添加计算平均nMSE和标准差的详细逻辑
    # ...

    # 示例：仅保存配置
    final_summary = {'hyperparameters': {k: v for k, v in vars(Config).items() if not k.startswith('__')}}

    with open(log_path, 'w') as f:
        json.dump(final_summary, f, indent=4)
    print(f"\n最终结果摘要已保存至: {log_path}")


def visualize_predictions(ground_truth, sbs_predictions, full_predictions, fold_idx, experiment_name):
    """为单轮的结果生成可视化图表。"""
    plot_dir = "result_plots"
    os.makedirs(plot_dir, exist_ok=True)
    save_path = os.path.join(plot_dir, f"{experiment_name}_fold_{fold_idx + 1}.png")

    num_plots = 7
    fig, axes = plt.subplots(num_plots, 1, figsize=(16, 22), sharex=True)
    fig.suptitle(f'PCESN++ Physics-Informed Prediction (Fold {fold_idx + 1})', fontsize=20)
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
    plt.savefig(save_path)
    plt.close()
    print(f"第 {fold_idx + 1} 折的可视化图表已保存至: {save_path}")

def visualize_reservoir_heatmap(reservoir_history, title):
    if not reservoir_history: return
    history_matrix = np.hstack(reservoir_history)
    plt.figure(figsize=(20, 8))
    plt.imshow(history_matrix, aspect='auto', interpolation='nearest', cmap='viridis')
    plt.colorbar(label="Neuron Activation")
    plt.title(title, fontsize=16)
    plt.xlabel("Time Steps");
    plt.ylabel("Reservoir Neurons")
    plt.show()


# --- 3. 主执行流程 ---
if __name__ == '__main__':
    config = Config()
    experiment_name = f'pcesn_physics_informed_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'

    # 1. 准备数据和物理向量
    all_trajectories = load_all_trajectories_from_file(config.DATA_FILE_PATH)
    physics_vector_cpu = get_physics_vector(config.PHYSICS_PARAMS)
    physics_vector_gpu = cp.asarray(physics_vector_cpu)

    all_fold_results = []

    # 2. 执行交叉验证
    for i in range(config.NUM_FOLDS):
        test_traj = all_trajectories[i]
        train_trajs = all_trajectories[:i] + all_trajectories[i + 1:]
        # train_trajs = all_trajectories[:] 直接作假实验，用测试机训练模型
        sbs_preds, full_preds, targets, train_time, trained_model, scaled_test_data = run_single_fold(i, train_trajs, test_traj, config, physics_vector_gpu)

        all_fold_results.append({
            'fold': i, 'sbs_predictions': sbs_preds,
            'full_predictions': full_preds, 'targets': targets
        })

        # 仅为最后一轮生成可视化图表
        if i == config.NUM_FOLDS - 1:
            min_len = min(len(targets), len(sbs_preds), len(full_preds))
            visualize_predictions(targets[:min_len], sbs_preds[:min_len], full_preds[:min_len], i, experiment_name)


    # 3. 汇总并记录最终结果
    # summarize_and_log_results(all_fold_results, config, experiment_name)
