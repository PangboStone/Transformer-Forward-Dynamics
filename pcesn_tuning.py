import numpy as np
import os
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import time
import csv
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

from dataset import load_all_trajectories_from_file
from model_pcesn import PCESNpp

# --- 1. 定义超参数的搜索空间 (使用skopt的维度对象) ---
# 这比网格搜索的字典更灵活
search_space = [
    Integer(200, 800, name='reservoir_size'),
    Real(0.1, 1.0, prior='uniform', name='spectral_radius'),
    Real(0.1, 1.0, prior='uniform', name='leak_rate'),
    Categorical([1e-2, 1e-3, 1e-4, 1e-5, 1e-6], name='regularization_factor'),
    Real(1e-4, 1e-2, prior='log-uniform', name='ghl_learning_rate'),
    Integer(1000, 10000, name='ghl_decay_steps')
]
# 将维度名称提取出来，方便后续使用
search_space_names = [dim.name for dim in search_space]


# --- 2. 定义目标函数 ---
# 这是贝叶斯优化器需要最小化的函数
# 它接收一组超参数，返回一个性能分数（nMSE）
@use_named_args(search_space)
def objective_function(**params):
    """
    接收一组超参数，运行一次精简实验，并返回单步预测nMSE。
    """
    print(f"\n--- 正在测试参数组合: {params} ---")

    try:
        # 使用第0折作为快速测试
        test_trajectory = all_trajectories[0]
        train_trajectories = all_trajectories[1:]

        # 数据标准化
        combined_train_inputs = np.concatenate([traj['inputs'] for traj in train_trajectories], axis=0)
        scaler = StandardScaler().fit(combined_train_inputs)
        scaled_train_inputs = [scaler.transform(t['inputs']) for t in train_trajectories]
        scaled_test_inputs = scaler.transform(test_trajectory['inputs'])
        train_targets = [t['targets'] for t in train_trajectories]
        test_targets = test_trajectory['targets']

        # 初始化模型
        model = PCESNpp(
            input_size=21, output_size=14,
            reservoir_size=params['reservoir_size'],
            spectral_radius=params['spectral_radius'],
            leak_rate=params['leak_rate'],
            regularization_factor=params['regularization_factor'],
            ghl_learning_rate=params['ghl_learning_rate'],
            ghl_decay_steps=params['ghl_decay_steps']
        )

        # 训练
        for inputs, targets in zip(scaled_train_inputs, train_targets):
            for t in range(len(inputs)):
                model.train_step(inputs[t].reshape(-1, 1), targets[t].reshape(-1, 1))

        # 评估 (只评估单步预测以加快速度)
        model.r_state.fill(0)
        predictions_sbs = np.array([model.predict(u_t.reshape(-1, 1)).flatten() for u_t in scaled_test_inputs])

        # 计算nMSE
        mse = np.mean((predictions_sbs - test_targets) ** 2)
        var_targets = np.var(test_targets)
        nmse = mse / (var_targets + 1e-9)

        print(f"结果 -> 单步预测 nMSE: {nmse:.6f}")
        return nmse

    except Exception as e:
        print(f"参数组合运行时出错: {e}")
        return 50.0  # 返回一个很大的惩罚值


# --- 3. 主调优流程 ---
if __name__ == '__main__':
    # 加载一次数据，供所有实验使用
    DATA_FILE_PATH = 'venv/ForwardDynamics/BaxterDirectDynamics.mat'
    all_trajectories = load_all_trajectories_from_file(DATA_FILE_PATH)

    print("--- Starting Bayesian hyperparameter optimisation ---")
    start_time = time.time()

    # --- CSV日志记录设置 ---
    log_dir = "tuning_logs"
    os.makedirs(log_dir, exist_ok=True)
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    csv_log_path = os.path.join(log_dir, f'pcesn_bayesian_tuning_results_{current_time}.csv')

    print(f"详细调优结果将保存至: {csv_log_path}")

    # 打开CSV文件并写入表头
    with open(csv_log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = search_space_names + ['nMSE']
        writer.writerow(header)


        # 定义一个回调函数，在每次迭代后被调用
        def on_step(res):
            # res.x_iters[-1] 是最新测试的参数列表
            # res.func_vals[-1] 是最新得到的分数 (nMSE)
            latest_params = res.x_iters[-1]
            latest_nmse = res.func_vals[-1]

            # 将参数和结果写入CSV文件
            row = latest_params + [latest_nmse]
            writer.writerow(row)
            print("Results of current round have been recorded.")


        # 运行贝叶斯优化，并传入回调函数
        result = gp_minimize(
            func=objective_function,
            dimensions=search_space,
            n_calls=50,  # 建议尝试 30-50 次
            random_state=42,
            callback=[on_step],  # 传入回调函数
            verbose=True
        )

    end_time = time.time()
    print("\n\n" + "=" * 20 + " Bayesian optimisation complete " + "=" * 20)
    print(f"Total time consumption: {(end_time - start_time) / 60:.2f} 分钟")
    print(f"Best single-step prediction found nMSE: {result.fun:.6f}")

    best_parameters = dict(zip(search_space_names, result.x))
    print("Corresponding optimal hyperparameter set:")
    for param, value in best_parameters.items():
        print(f"  {param}: {value}")

