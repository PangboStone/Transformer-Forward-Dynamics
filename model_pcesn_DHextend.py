import numpy as np
import cupy as cp
from scipy.sparse import random as sparse_random
from scipy.linalg import eigh


class PCESNpp:
    def __init__(self, input_size, reservoir_size, output_size, physics_vector_size,
                 spectral_radius=0.99, sparsity=0.1, leak_rate=0.1,
                 regularization_factor=1e-3, ghl_eta=1e-3,ghl_decay_steps=5000.0,
                 tanh_r = 0.8, sico_r =0.1):
        self.input_size = input_size
        self.reservoir_size = reservoir_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius
        self.sparsity = sparsity
        self.leak_rate = leak_rate
        self.regularization_factor = regularization_factor
        self.ghl_eta = ghl_eta #        ghl_learning_rate (float): GHL层的初始学习率。
        self.ghl_decay_steps = ghl_decay_steps
        self.physics_vector_size = physics_vector_size
        self.tanh_r = tanh_r
        self.sico_r = sico_r

        assert tanh_r + sico_r * 2 == 1.0
        num_tanh = int(reservoir_size * tanh_r)
        num_sin = int(reservoir_size * sico_r)
        num_cos = int(reservoir_size * sico_r)

        self.tanh_index = slice(0,num_tanh)
        self.sin_index = slice(num_tanh, num_tanh + num_sin)
        self.cos_index = slice(num_tanh+num_sin, reservoir_size)

        print(f"储蓄池结构: {num_tanh/reservoir_size} %(tanh), {num_sin/reservoir_size} %(sin), {num_cos/reservoir_size} %(cos)")

        # --- GHL layer ---
        print("正在初始化GHL层...")
        # W_ghl 对应论文中的 W^in，它是一个下三角矩阵
        W_ghl_cpu = np.tril(np.random.randn(self.input_size, self.input_size))
        self.W_ghl = cp.asarray(W_ghl_cpu)
        # 训练时间步计数器，用于学习率衰减
        self.t_counter = 0
        print("GHL层初始化完成。")

        # --- Dynamic Reservoir ---
        print("正在创建动态存储库...")
        W_res_sparse = sparse_random(self.reservoir_size, self.reservoir_size, density=self.sparsity, data_rvs=np.random.randn)
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

        # --- . IterativeBayesianLinearRegression (RLS) ---
        print("正在初始化RLS学习器...")
        self.combined_state_size = self.reservoir_size + self.physics_vector_size + 1
        self.W_out = cp.zeros((self.output_size, self.combined_state_size))
        self.P = (1.0 / self.regularization_factor) * cp.identity(self.combined_state_size)
        print("RLS学习器初始化完成。")

        print("--- 模型初始化完成 ---")

    def _update_ghl(self, u_t_gpu):
        """Internal method: GPU performs the forward propagation and weight update of the GHL layer."""
        h_t = cp.dot(self.W_ghl, u_t_gpu)
        self.t_counter += 1
        eta_t = self.ghl_eta / (1 + self.t_counter / self.ghl_decay_steps)
        h_h_T = cp.dot(h_t, h_t.T)
        delta_W = eta_t * (cp.dot(u_t_gpu, h_t.T) - cp.tril(h_h_T) @ self.W_ghl)
        self.W_ghl += delta_W
        return h_t

    """Physics Injection Core Part"""
    def _update_reservoir_state(self, u_t_gpu):
        """
        Internal method: update and return the current repository state on the GPU.
        The core modification is here, applying mixed activation functions.
        """
        # --- Step 1: Process input through the GHL layer ---
        # The GHL layer performs an initial, adaptive transformation of the input.
        h_t = self._update_ghl(u_t_gpu)

        # --- Step 2: Calculate the total input before activation ---
        # This computes the linear combination of the recurrent state and the processed input.
        # This intermediate variable `pre_activation` is necessary because we need to
        # apply different activation functions to different parts of this vector.
        pre_activation = cp.dot(self.W_res, self.r_state) + cp.dot(self.W_in, h_t)

        # --- Step 3: Apply different activation functions based on pre-defined neuron types ---
        # Initialize an empty container for the activated neuron states.
        activation = cp.zeros_like(pre_activation)

        # Apply tanh to the 'tanh' neurons. These are responsible for stable dynamics and memory.
        # self.tanh_indices is a pre-computed slice object for efficiency.
        activation[self.tanh_index] = cp.tanh(pre_activation[self.tanh_index])

        # Apply sin to the 'sin' neurons. These provide intrinsic trigonometric computation.
        activation[self.sin_index] = cp.sin(pre_activation[self.sin_index])

        # Apply cos to the 'cos' neurons, complementing the sin neurons.
        activation[self.cos_index] = cp.cos(pre_activation[self.cos_index])

        # --- Step 4: Update the reservoir state using the leak rate ---
        # This is the standard leaky integrator update rule for ESNs.
        self.r_state = (1 - self.leak_rate) * self.r_state + self.leak_rate * activation

        return self.r_state

    def _get_extended_state(self, r_state, physics_vector_gpu):
        """sue fusion physical parameter vectors， extend: 21+7 = 28"""
        return cp.vstack((r_state, physics_vector_gpu, cp.ones((1, 1))))

    def train_step(self, u_t_cpu, target_t_cpu,physics_vector_gpu):
        """Train by single time step data input (online learning)"""
        u_t_gpu = cp.asarray(u_t_cpu)
        target_t_gpu = cp.asarray(target_t_cpu)
        r_state = self._update_reservoir_state(u_t_gpu)
        extend_state = self._get_extended_state(r_state,physics_vector_gpu)
        prediction = cp.dot(self.W_out, extend_state)
        error = target_t_gpu - prediction

        mse = cp.mean(error**2)

        # RLS layer update // Iterative BayesianLinear Regression
        P_c = cp.dot(self.P, extend_state)
        k_t = P_c / (1 + cp.dot(extend_state.T, P_c))
        self.W_out += cp.dot(error, k_t.T)
        self.P = cp.dot((cp.identity(self.combined_state_size) - cp.dot(k_t, extend_state.T)), self.P)

        return mse.item()


    def predict(self, u_t_cpu, physics_vector_gpu):
        u_t_gpu = cp.asarray(u_t_cpu)
        r_state = self._update_reservoir_state(u_t_gpu)
        extended_state = self._get_extended_state(r_state, physics_vector_gpu)
        prediction_gpu = cp.dot(self.W_out, extended_state)
        return cp.asnumpy(prediction_gpu)

    def reset_state(self):
        """ Reset the internal state of the model for clean evaluation between rounds or datasets """
        print("Reservoir internal state RESET...")
        self.r_state = cp.zeros((self.reservoir_size, 1))