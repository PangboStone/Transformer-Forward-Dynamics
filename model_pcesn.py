import numpy as np
import cupy as cp
from scipy.sparse import random as sparse_random
from scipy.linalg import eigh


class PCESNpp:
    """
    对论文 "A Reservoir Computing Approach for Learning Forward Dynamics"
    中的 PC-ESN++ 模型进行复现。

    这个类将包含模型的三个核心组件：
    1. 广义赫布学习 (GHL) - (已实现)
    2. 动态存储库 (Dynamic Reservoir) - (已实现)
    3. 迭代贝叶斯线性回归 (RLS实现) - (已实现)

    采用cupy进行GPU加速
    """

    def __init__(self, input_size, reservoir_size, output_size,
                 spectral_radius=0.99, sparsity=0.1, leak_rate=0.1,
                 regularization_factor=1e-3, ghl_learning_rate=1e-3):
        """
        初始化 PC-ESN++ 模型。

        参数:
        //... (其他参数不变)
        ghl_learning_rate (float): GHL层的初始学习率。
        """
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.regularization_factor = regularization_factor
        self.ghl_initial_learning_rate = ghl_learning_rate

        print("--- 初始化 PC-ESN++ 模型 ---")

        # --- 组件 A: 广义赫布学习 (GHL) ---
        print("正在初始化GHL层...")
        # W_ghl 对应论文中的 W^in，它是一个下三角矩阵
        W_ghl_cpu = np.tril(np.random.randn(self.input_size, self.input_size))
        self.W_ghl = cp.asarray(W_ghl_cpu)
        # 训练时间步计数器，用于学习率衰减
        self.t_counter = 0
        print("GHL层初始化完成。")

        # --- 组件 B: 动态存储库 (Dynamic Reservoir) ---
        print("正在创建动态存储库...")
        W_res_sparse = sparse_random(self.reservoir_size, self.reservoir_size,
                                     density=self.sparsity, data_rvs=np.random.randn)
        W_res_cpu = W_res_sparse.toarray()
        current_spectral_radius = np.max(np.abs(eigh(W_res_cpu, eigvals_only=True)))
        if current_spectral_radius > 0:
            W_res_cpu *= self.spectral_radius / current_spectral_radius

        # W_in 对应论文中的 W^self
        W_in_cpu = np.random.randn(self.reservoir_size, self.input_size)

        self.W_res = cp.asarray(W_res_cpu)
        self.W_in = cp.asarray(W_in_cpu)

        self.r_state = cp.zeros((self.reservoir_size, 1))
        print("动态存储库创建完成。")

        # --- 组件 C: 迭代贝叶斯线性回归 (以RLS实现) ---
        print("正在初始化RLS学习器...")
        self.combined_state_size = self.reservoir_size + 1
        self.W_out = cp.zeros((self.output_size, self.combined_state_size))
        self.P = (1.0 / self.regularization_factor) * cp.identity(self.combined_state_size)
        print("RLS学习器初始化完成。")

        print("--- 模型初始化完成 ---")

    def _update_ghl(self, u_t_gpu):
        """内部方法：使用GPU执行GHL层的前向传播和权重更新。"""
        # 1. 计算GHL层输出 h_t (论文公式 1)
        h_t = cp.dot(self.W_ghl, u_t_gpu)

        # 2. 更新GHL权重 (论文公式 2)
        # η_t (eta_t) 是一个随时间衰减的学习率
        self.t_counter += 1
        eta_t = self.ghl_initial_learning_rate / (1 + self.t_counter / 1000)

        # 计算权重更新量 ΔW
        h_h_T = cp.dot(h_t, h_t.T)
        delta_W = eta_t * (cp.dot(u_t_gpu, h_t.T) - cp.tril(h_h_T) @ self.W_ghl)

        # 更新权重
        self.W_ghl += delta_W

        return h_t


    def _update_reservoir_state(self, u_t_gpu):
        """内部方法：在GPU上更新并返回当前的存储库状态"""
        # --- 1. GHL层处理输入 ---
        h_t = self._update_ghl(u_t_gpu)
        # --- 2. 更新存储库状态 r ---
        activation = cp.tanh(cp.dot(self.W_res, self.r_state) + cp.dot(self.W_in, h_t))
        self.r_state = (1 - self.leak_rate) * self.r_state + self.leak_rate * activation
        return self.r_state

    def train_step(self, u_t_cpu, target_t_cpu):
        """
        使用单个时间步的数据对模型进行训练（在线学习）。
        """
        # 将数据从CPU转移到GPU
        u_t_gpu = cp.asarray(u_t_cpu)
        target_t_gpu = cp.asarray(target_t_cpu)

        # 1. 更新内部的存储库状态 (这会自动调用GHL更新)
        r_state = self._update_reservoir_state(u_t_gpu)

        # 2. 构造用于线性回归的组合状态向量 c_t
        c_t = cp.vstack((r_state, np.ones((1, 1))))

        # 3. 计算预测误差
        prediction = cp.dot(self.W_out, c_t)
        error = target_t_gpu - prediction

        # 计算该步的MSE用于返回
        mse = cp.mean(error**2)

        # 4. RLS 算法更新步骤
        P_c = cp.dot(self.P, c_t)
        k_t = P_c / (1 + cp.dot(c_t.T, P_c))
        self.W_out += cp.dot(error, k_t.T)
        self.P = cp.dot((cp.identity(self.combined_state_size) - cp.dot(k_t, c_t.T)), self.P)

        return mse.item()

    def predict(self, u_t_cpu):
        """
        使用当前模型对单个时间步的输入进行预测。
        """
        # 转移数据
        u_t_gpu = cp.asarray(u_t_cpu)

        # 1. 更新内部的存储库状态 (这会自动调用GHL更新)
        # 注意：在纯预测模式下，GHL层依然会更新，因为它是一个无监督的自适应过程。
        r_state = self._update_reservoir_state(u_t_gpu)

        # 2. 构造组合状态向量 c_t
        c_t = cp.vstack((r_state, cp.ones((1, 1))))

        # 3. 进行预测
        prediction_gpu = cp.dot(self.W_out, c_t)
        return cp.asnumpy(prediction_gpu)


# 测试代码
# if __name__ == '__main__':
#     # 定义模型参数
#     INPUT_SIZE = 21
#     RESERVOIR_SIZE = 500
#     OUTPUT_SIZE = 14
#
#     # 创建模型实例
#     pcesn_model = PCESNpp(input_size=INPUT_SIZE,
#                           reservoir_size=RESERVOIR_SIZE,
#                           output_size=OUTPUT_SIZE)
#
#     # 创建一个模拟的单步输入和目标
#     sample_input = np.random.rand(INPUT_SIZE, 1)
#     sample_target = np.random.rand(OUTPUT_SIZE, 1)
#
#     # 1. 预测一次 (在训练前)
#     print("\n--- 训练前 ---")
#     prediction_before = pcesn_model.predict(sample_input)
#     error_before = np.mean((sample_target - prediction_before) ** 2)
#     print(f"预测输出的形状: {prediction_before.shape}")
#     print(f"预测误差 (MSE): {error_before:.6f}")
#
#     # 2. 训练一步
#     print("\n--- 执行一步训练 ---")
#     pcesn_model.train_step(sample_input, sample_target)
#     print("模型权重已更新。")
#
#     # 3. 再次预测 (在训练后)
#     print("\n--- 训练后 ---")
#     # 注意：为了得到可比较的结果，我们重新生成了存储库状态
#     # 在实际应用中，状态是连续的
#     pcesn_model.r_state = cp.zeros((RESERVOIR_SIZE, 1))  # 重置状态
#     prediction_after = pcesn_model.predict(sample_input)
#     error_after = cp.mean((sample_target - prediction_after) ** 2)
#     print(f"预测输出的形状: {prediction_after.shape}")
#     print(f"预测误差 (MSE): {error_after:.6f}")
#
#     # 检查误差是否减小
#     if error_after < error_before:
#         print("\n成功: 经过一步训练后，预测误差减小了！")
#     else:
#         print("\n注意: 经过一步训练后，预测误差没有减小。这在单步中是可能发生的。")
