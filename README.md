# Industrial Manipulator Forward Dynamics Modelling with Transformer-Based and Physics-Informed ESN Models

This repository contains the complete implementation for a research project focused on learning the forward dynamics of industrial manipulators. It provides a replication of the PC-ESN++ baseline model and introduces several advanced alternatives, including a novel Transformer-based framework (TFD-Net), an efficient Linformer variant, and a Physics-Informed PC-ESN.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Goal

This project primarily aims to develop and evaluate data-driven models for predicting the forward dynamics of robotic manipulators, including:
1.  Reproducing and critically analyzing the performance of a baseline **PC-ESN++** model from prior work.
2.  Designing and implementing a more powerful **Transformer-based architecture (TFD-Net)** to overcome the limitations of recurrent models.
3.  Developing an efficient **Linformer** variant to reduce the computational complexity of the standard Transformer.
4.  Exploring a **Physics-Informed PC-ESN** that injects domain knowledge (DH parameters) into the model to improve performance and interpretability.

## 📁 File Structure & Description

The repository contains a flat structure of scripts for data handling, model definitions, training, and evaluation. Here is a breakdown of the key files:

#### Core Modules
* **`dataset.py`**: Contains the core `RobotDynamicsDataset` class for PyTorch. It handles loading `.mat` files, parsing trajectories, and creating sequence-based samples for training.
* **`model_*.py`**: These files contain the architecture definitions for all models used in this project.
    * `model_pcesn.py`: Implements the baseline PC-ESN++ using **Cupy**.
    * `model_pcesn_DHextend.py`: The physics-informed version of the PC-ESN.
    * `model_transformer.py`: Implements the main **TFD-Net** using standard PyTorch `nn.TransformerEncoder`.
    * `model_transformer_LinformerAttention.py`: Implements the **Linformer** variant with a custom low-rank attention mechanism.
    * `model_LSTM.py`: A simple LSTM baseline used for initial pipeline validation.

#### Experiment & Evaluation Scripts
* **`train_linformer_0807.py`**: The primary, most up-to-date, and interactive script for training and evaluating the Transformer and Linformer models. **This is the recommended script for running experiments.**
* **`prediction_visualization.py`**: An interactive script to load a saved model checkpoint and generate detailed prediction plots for a chosen trajectory.
* **`pcesn_tuning.py`**: A script that uses **Bayesian Optimization** (`scikit-optimize`) to find the best hyperparameters for the PC-ESN++ model.
* **`train_pcesn.py` & `train_pcesn_DHextend.py`**: Scripts dedicated to running cross-validation experiments on the ESN-based models.

#### A Note on File Naming
Some script names include numeric suffixes (e.g., `_0726`, `_0807`). These suffixes represent the date on which that version of the script was developed (e.g., July 26th, August 7th). They mark different stages of the project's development. As a rule, **later versions are more complete and refined.** For instance, `train_linformer_0807.py` is the final and most comprehensive training script.

## 🦾 Models Implemented

This project compares four primary architectures:

1.  **PC-ESN++ (Baseline)**: A reservoir computing model with an online learning mechanism. It's computationally efficient to train but is inherently sequential.
2.  **Physics-Informed PC-ESN (PI-PCESN)**: An enhanced ESN that incorporates the robot's Denavit-Hartenberg (DH) parameters as features, creating a "grey-box" model.
3.  **Transformer (TFD-Net)**: A powerful encoder-only architecture that uses a self-attention mechanism to capture long-range dependencies in the trajectory data.
4.  **Linformer**: An efficient Transformer variant that approximates the self-attention mechanism to achieve linear time complexity, making it faster and less memory-intensive.

## ⚙️ Installation

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/PangboStone/Transformer-Base-Manipulator-Forward-Dynamics.git](https://github.com/PangboStone/Transformer-Base-Manipulator-Forward-Dynamics.git)
    cd Transformer-Base-Manipulator-Forward-Dynamics
    ```
2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install Dependencies:**
    The project requires PyTorch with CUDA support for best performance. Please install it first by following the [official PyTorch instructions](https://pytorch.org/get-started/locally/). Then, install the remaining packages.
    ```bash
    pip install -r requirements.txt
    ```
4.  **Download Datasets:**
    Download the benchmark datasets (`.mat` files) from the **[Real Robot Manipulation Datasets For Learning Dynamics](https://gabriellapizzuto.github.io/Real-Robot-Manipulation-Datasets-For-Learning-Dynamics/)** page. Create a `venv/ForwardDynamics/` directory at the project root and place the files inside it.

## 🛠 How to Use

This project's experiments are designed to be run directly from the scripts in the root directory.

### Training a Transformer or Linformer Model

The most comprehensive training script is `train_linformer_0807.py`, which runs interactively.

1.  **Run the script:**
    ```bash
    python train_linformer_0807.py
    ```
2.  **Follow the prompts:** The script will ask you to:
    * Choose between a single run or full cross-validation.
    * Select the model type (`transformer` or `linformer`).
    * Set a correction interval for autonomous prediction (or disable it).
    * Choose whether to save and/or display plots.

The script will handle data loading, training, evaluation, and saving all results (logs, plots, and TensorBoard files) to the `results/` directory and model checkpoints to `saved_models/`.

### Visualizing a Trained Model's Predictions

The `prediction_visualization.py` script allows you to load any trained model checkpoint and test it on any trajectory.

1.  **Run the script:**
    ```bash
    python prediction_visualization.py
    ```
2.  **Follow the prompts:**
    * Select a saved model checkpoint (`.pth` file) from the `saved_models/` directory.
    * Select a dataset (`.mat` file).
    * Select a specific trajectory number to evaluate.

The script will then generate a detailed plot comparing the model's single-step and full-trajectory predictions against the ground truth.

## 📊 Key Results

The experiments consistently showed that the Transformer-based models significantly outperformed the ESN baselines, especially in the more challenging full-trajectory prediction task.

### Prediction Performance (Full Trajectory on KUKA)

| Model | Position nMSE (Mean ± Std) | Velocity nMSE (Mean ± Std) |
| :--- | :--- | :--- |
| PI-PCESN | 1.1929 ± 0.3487 | 2.2965 ± 0.9638 |
| **TFD-Net** | **0.058 ± 0.0012** | **0.2931 ± 0.014** |
| **Linformer** | 0.103 ± 0.0007 | 0.6292 ± 0.036 |

### Computational Efficiency

The Transformer models are nearly **10x faster** at inference time due to their parallelizable architecture. The Linformer provides the best balance of speed and memory efficiency.

| Model | Avg. Inference Time (ms) | Peak Memory (MB) |
| :--- | :--- | :--- |
| PI-PCESN | 1.6659 | 2037.85 |
| **TFD-Net** | 0.1115 | 2112.16 |
| **Linformer** | **0.0985** | **1957.93** |

### 📈Prediction Result Visual Example

The plot below shows the Linformer's superior performance on the KUKA dataset. Its predictions (green and red dotted lines) closely follow the ground truth (blue line), whereas the ESN models typically show significant drift.

![Prediction and Ground-truth Trajectories Comparison](./assets/Prediction%20and%20Ground-truth%20Trajectories%20Comparison.png)

## 📰Key References

[1] Alkhodary, A., & Gur, B. (2024). Learning Soft Robotic Arm Control: A Data-Driven Approach with Forward Dynamics Transformer and Reinforcement Learning. *EAI/Springer Innovations in Communication and Computing*.
[23] Polydoros, A. S., & Nalpantidis, L. (2016). A Reservoir Computing Approach for Learning Forward Dynamics of Industrial Manipulators. *2016 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 612-618.
[22] Pizzuto, G. (2020). *Real robot manipulation datasets for learning dynamics*. Retrieved from https://gabriellapizzuto.github.io/Real-Robot-Manipulation-Datasets-For-Learning-Dynamics/

