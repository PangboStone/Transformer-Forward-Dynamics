import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import itertools
import time

# --- 从自定义模块导入 ---
from dataset import load_all_trajectories_from_file
from model_pcesn import PCESNpp

# --- 1. 定义超参数搜索空间 ---
# 在这里定义我们想要测试的每一组超参数的值
# HYPERPARAM_SEARCH_SPACE = {
#     'RESERVOIR_SIZE': [300],  # 暂时固定储备池大小以加快搜索速度
#     'LEAK_RATE': [0.1, 0.3, 0.5, 0.8],
#     'SPECTRAL_RADIUS': [0.9, 0.99, 1.1],
#     'REGULARIZATION_FACTOR': [1e-2, 1e-4, 1e-6],
#     'GHL_LEARNING_RATE': [1e-3]  # 暂时固定GHL学习率
# }

# Grid Search on Parameter space
HYPERPARAM_SEARCH_SPACE = {
    'RESERVOIR_SIZE': [400],    # 测试不同容量的模型
    'LEAK_RATE': [0.3],         # 围绕上次找到的最佳值0.5进行精细搜索
    'SPECTRAL_RADIUS': [0.7],   # 围绕上次找到的最佳值0.9进行精细搜索
    'REGULARIZATION_FACTOR': [0.0001],  # 固定上次找到的最佳值
    'GHL_LEARNING_RATE': [0.001]        # 固定上次找到的最佳值
}

# --- 2. 训练与评估的核心函数 ---
# 我们将核心逻辑封装成一个函数，便于循环调用
def run_single_experiment(params, all_trajectories):
    """
    使用一组给定的超参数，运行一轮交叉验证并返回结果。
    """
    try:
        # 使用第0折作为快速测试
        fold_index = 0
        test_trajectory = all_trajectories[fold_index]
        train_trajectories = all_trajectories[1:]  # 使用剩余的轨迹

        # 数据标准化
        combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
        scaler = StandardScaler().fit(combined_train_inputs)
        scaled_train_inputs = [scaler.transform(t['inputs']) for t in train_trajectories]
        scaled_test_inputs = scaler.transform(test_trajectory['inputs'])
        train_targets = [t['targets'] for t in train_trajectories]
        test_targets = test_trajectory['targets']

        # 初始化模型
        model = PCESNpp(
            input_size=21,
            output_size=14,
            reservoir_size=params['RESERVOIR_SIZE'],
            spectral_radius=params['SPECTRAL_RADIUS'],
            leak_rate=params['LEAK_RATE'],
            regularization_factor=params['REGULARIZATION_FACTOR'],
            ghl_learning_rate=params['GHL_LEARNING_RATE']
        )

        # 训练
        for inputs, targets in zip(scaled_train_inputs, train_targets):
            for t in range(len(inputs)):
                model.train_step(inputs[t].reshape(-1, 1), targets[t].reshape(-1, 1))

        # 评估 (只评估单步预测以加快速度)
        model.r_state.fill(0)
        predictions_sbs = []
        for t in range(len(scaled_test_inputs)):
            pred = model.predict(scaled_test_inputs[t].reshape(-1, 1))
            predictions_sbs.append(pred.flatten())

        nmse_sbs = np.mean((np.array(predictions_sbs) - test_targets) ** 2) / np.var(test_targets)
        return nmse_sbs, params

    except Exception as e:
        print(f"参数组合 {params} 运行时出错: {e}")
        return float('inf'), params  # 返回一个极大值表示失败


# --- 3. 主调优流程 ---
if __name__ == '__main__':
    # --- 初始化日志 ---
    # log_dir = "venv/tuning_logs"
    # os.makedirs(log_dir, exist_ok=True)
    # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    # log_file_path = os.path.join(log_dir, f'pcesn_tuning_{current_time}.log')

    # with open(log_file_path, 'w') as log_file:
        def log_print(message):
            print(message)
            # log_file.write(message + '\n')
            # log_file.flush()


        log_print("--- PC-ESN++ 超参数网格搜索 ---")
        log_print(f"搜索空间: {HYPERPARAM_SEARCH_SPACE}")

        # 加载数据
        DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
        all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

        # 从搜索空间中生成所有可能的参数组合
        keys, values = zip(*HYPERPARAM_SEARCH_SPACE.items())
        param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

        log_print(f"\n将要测试 {len(param_combinations)} 组超参数组合...")

        results = []
        best_nmse = float('inf')
        best_params = None

        start_time = time.time()
        for i, params in enumerate(param_combinations):
            iter_start_time = time.time()
            log_print(f"\n--- 正在测试第 {i + 1}/{len(param_combinations)} 组 ---")
            log_print(f"参数: {params}")

            # 运行实验
            nmse, _ = run_single_experiment(params, all_trajectories)
            results.append((nmse, params))

            iter_end_time = time.time()
            log_print(f"结果 -> 单步预测 nMSE: {nmse:.6f} (耗时: {iter_end_time - iter_start_time:.2f} 秒)")

            # 记录最佳结果
            if nmse < best_nmse:
                best_nmse = nmse
                best_params = params
                log_print("!!! 发现了新的最佳参数 !!!")

        end_time = time.time()
        log_print("\n\n" + "=" * 20 + " 搜索完成 " + "=" * 20)
        log_print(f"总耗时: {(end_time - start_time) / 60:.2f} 分钟")
        log_print(f"最佳单步预测 nMSE: {best_nmse:.6f}")
        log_print(f"对应的最佳超参数组合: {best_params}")

    # print(f"\n调优完成！详细日志已保存至: {log_file_path}")


# ==================== 搜索完成 ====================
# 总耗时: 13.67 分钟
# 最佳单步预测 nMSE: 0.188193
# 对应的最佳超参数组合: {'RESERVOIR_SIZE': 400,
#              'LEAK_RATE': 0.3,
#              'SPECTRAL_RADIUS': 0.7,
#              'REGULARIZATION_FACTOR': 0.0001,
#              'GHL_LEARNING_RATE': 0.001}