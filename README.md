# Transformer-Forward-Dynamics

🔧 **Project Overview**  
This repository reproduces and improves learning-based forward dynamics modeling for industrial manipulators. It explores Transformer-based architectures and physics-informed neural networks to enhance prediction accuracy and real-world applicability.

---

## 📁 Directory Structure

```bash
Transformer-Forward-Dynamics/
├── dataset.py                     # Data loading and preprocessing
├── dataparse_demo.py             # Data parsing demonstration
├── model_transformer.py          # Standard Transformer model
├── model_transformer_LinformerAttention.py  # Linformer attention variant
├── model_pcesn.py                # Physics-Informed Echo State Network (PCESN)
├── model_pcesn_DHextend.py       # PCESN with extended DH parameters
├── train_transformer.py          # Transformer training script
├── train_pcesn.py                # PCESN training script
├── train_pcesn_DHextend.py       # Extended PCESN training
├── train_linformer_0807.py       # Linformer training script
├── prediction_visualization.py   # Visualization of prediction results
├── unit_test_pcesn.py            # Unit tests for PCESN
├── results/                      # Output figures and prediction results
└── Development Log.md            # Development notes and progress
