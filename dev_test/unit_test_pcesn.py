import unittest
import numpy as np
import cupy as cp

# 从模型文件中导入我们要测试的类
from model_pcesn import PCESNpp


class TestPCESNpp(unittest.TestCase):
    """
    为 PC-ESN++ 模型的复现版本编写单元测试。
    独立验证模型三个核心组件的正确性。
    """

    def setUp(self):
        """
        这个方法在每个测试开始前都会被调用，在这里创建一个通用的模型实例供所有测试使用。
        """
        self.input_size = 21
        self.reservoir_size = 100  # 使用较小的尺寸以加快测试速度
        self.output_size = 14

        self.model = PCESNpp(
            input_size=self.input_size,
            reservoir_size=self.reservoir_size,
            output_size=self.output_size
        )

        # 创建一个CPU上的NumPy测试输入
        self.sample_input_np = np.random.rand(self.input_size, 1)

    def test_01_initialization(self):
        """
        Test 1: Verify that all weight matrices of the model have the correct shape after initialisation.
        """
        print("\n--- Executing Test 1: Weight Matrices Initialization ---")
        self.assertEqual(self.model.W_ghl.shape, (self.input_size, self.input_size))
        self.assertEqual(self.model.W_res.shape, (self.reservoir_size, self.reservoir_size))
        self.assertEqual(self.model.W_in.shape, (self.reservoir_size, self.input_size))
        self.assertEqual(self.model.W_out.shape, (self.output_size, self.reservoir_size + 1))
        self.assertEqual(self.model.P.shape, (self.reservoir_size + 1, self.reservoir_size + 1))
        print("Test 1 Passed: all weight matrices is correct shaped")

    def test_02_reservoir_state_update(self):
        """
        Test 2: Verify the dynamic reservoir‘s state after receiving input
        """
        print("\n--- Executing Test 2: Dynamic Reservoir State Update ---")
        # 获取初始状态 (GPU上的CuPy数组)
        initial_r_state = self.model.r_state.copy()

        # 执行一步预测，这会更新内部状态
        self.model.predict(self.sample_input_np)

        # 获取更新后的状态
        updated_r_state = self.model.r_state

        # 验证：更新后的状态不应该等于初始的全零状态
        self.assertFalse(cp.all(updated_r_state == initial_r_state), "Dynamic Reservoir Update Fails！")

        # 验证：状态值应该在tanh的范围内 (-1, 1)
        self.assertTrue(cp.all(updated_r_state > -1) and cp.all(updated_r_state < 1))
        print("Test 2 Passed: Dynamic Reservoir Update succeeds")

    def test_03_ghl_weights_update(self):
        """
        测试3：验证GHL层的权重是否在模型步骤中被更新。
        """
        print("\n--- Executing Test 3: GHL weight update ---")
        # 存储初始的GHL权重
        initial_W_ghl = self.model.W_ghl.copy()

        # 执行一步训练，这会触发GHL的更新
        self.model.train_step(self.sample_input_np, np.random.rand(self.output_size, 1))

        # 获取更新后的GHL权重
        updated_W_ghl = self.model.W_ghl

        # 验证：更新后的权重不应该等于初始权重
        self.assertFalse(cp.all(updated_W_ghl == initial_W_ghl), "GHL weights did not change after one step of training!")
        print("Test 3 Passed: GHL weights have been successfully updated.")

    def test_04_rls_learning_on_single_sample(self):
        """
        测试4：验证RLS学习算法的有效性。
        在同一个样本上反复训练，预测误差应该持续减小。
        """
        print("\n--- Executing Test 4: RLS learning validation  ---")
        # 创建一个固定的训练样本
        fixed_input = np.random.rand(self.input_size, 1)
        fixed_target = np.random.rand(self.output_size, 1)

        # 第一次预测，得到初始误差
        prediction = self.model.predict(fixed_input)
        initial_error = np.mean((fixed_target - prediction) ** 2)

        # 在同一个样本上反复训练5次
        last_error = initial_error
        for i in range(3):
            # 每次训练前都重置状态，以确保我们测试的是权重学习，而不是状态演化
            self.model.r_state.fill(0)

            # 训练并记录误差
            self.model.train_step(fixed_input, fixed_target)

            # 再次预测
            self.model.r_state.fill(0)
            prediction = self.model.predict(fixed_input)
            current_error = np.mean((fixed_target - prediction) ** 2)

            print(f"RLS training Iteration {i + 1}: MSE = {current_error:.8f}")

            # 验证：当前误差应该小于上一次的误差
            self.assertLess(current_error, last_error, f"Iteration {i + 1}，RLS fails, error dose not decent !")
            last_error = current_error

        print("Test 4 Passed: The RLS learning algorithm is effective in reducing the prediction error for a single sample.")


if __name__ == '__main__':
    # 运行所有测试
    unittest.main()
