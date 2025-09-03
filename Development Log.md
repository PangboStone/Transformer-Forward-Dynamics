# Development Log

This document appendix provides a detailed week-by-week log of the development process and experimental work carried out during the project. The log complements the research diary and concludes technical implementation effort.
In total, approximately 390 hours were devoted over 10 weeks, resulting in over 6000 lines of Python code including data pipelines, model implementations, training routines, and evaluation scripts.

## Week 1 (June 22 – June 28): Project Foundation and Data Pipeline
* Established the development environment (`PyTorch`, `scikit-learn`, `scikit-optimize`, `Matplotlib`...).
* Implemented the first version of the data processing pipeline.
* Added `dataset.py` to parse raw `.mat` files into structured trajectory objects.
* Verified pipeline correctness by visualizing sample trajectories.
* Implemented a simple LSTM baseline to validate end-to-end functionality.

## Week 2 (June 29 – July 5): Reservoir Computing Baseline
* Implemented the PC-ESN (Predictive Coding Echo State Network) baseline.
* Added unit tests for core modules (reservoir dynamics, gradient-based Hebbian learning (GHL), recursive least squares (RLS)).
* Created `test_pcesn_model.py` to check reproducibility and correctness.
* Experimented with key hyperparameters (spectral radius, sparsity) using grid search.

## Week 3 (July 6 – July 12): Hyperparameter Optimization
* Replaced grid search with Bayesian optimization using `scikit-optimize`, ran optimization for about 10 hours.
* Developed `train_pcesn_bayesian_tuning.py`, logging tuning results to CSV.
* Reproduced plots from the original PC-ESN++ paper via `reproduce_plots.py`.
* Corrected evaluation metrics: focused on single-joint nMSE in line with baseline methodology.

## Week 4 (July 13 – July 19): Transformer Baseline
* Designed a Transformer architecture (TFD-Net) for forward dynamics prediction.
* Implemented sinusoidal positional encoding layer.
* Introduced separated input and output scalers for improved normalization.
* Built visualization utilities to compare Transformer predictions against ground truth.
* Evaluated sensitivity to `NUM_ENCODER_LAYERS`.
* Added functionality to load saved model checkpoints (`.pth`) and run inference.

## Week 5 (July 20 – July 26): Linformer Improvement Approach
* Implemented Linformer attention mechanism as a low-rank approximation to Transformer attention.
* Code contributions included `model_transformer_LinformerAttention.py` and training script `temp0726.py`.
* Refined comments and documentation in `model_transformer.py` for clarity.
* Benchmarked Linformer against the standard Transformer in terms of runtime and memory footprint.

## Week 6 (July 27 – August 2): Physics-Informed PC-ESN++
* Extended PC-ESN++ with physics-informed feature injection.
* Adopted hybrid activation functions to improve reservoir expressivity.
* Refactored the experimental pipeline into a modular, class-based framework:
    * `Config`: experiment configuration management.
    * `DataHandler`: standardized dataset loading and preprocessing.
    * `ExperimentRunner`: unified training/evaluation loop.
    * `Visualizer`: automated generation of evaluation plots.
* Integrated an output smoothing layer (Savitzky–Golay filter) to reduce prediction noise.

## Week 7 (August 3 – August 9): Comparative Experiments
* Ran systematic comparison between PC-ESN++, PI-PCESN, Transformer, and Linformer.
* Conducted leave-one-trajectory-out cross-validation across KUKA and Baxter datasets.
* Logged detailed results (nMSE per joint, per trajectory).
* Visualized multi-step prediction degradation across horizons (1, 10, 50, 100, 1000 steps).

## Week 8 (August 10 – August 16): Ablation and Analysis
* Performed ablation study on hybrid physics-informed features.
* Analyzed effect of reservoir size and spectral radius scaling.
* Benchmarked Transformer depth and attention head count.
* Generated error growth curves and comparative tables for appendix inclusion.

## Week 9 (August 17 – August 23): Thesis Writing and Documentation
* Consolidated experimental results into tables and figures.
* Wrote initial drafts for methodology and experiment chapters.
* Drew and organised detailed figures directly from plotting scripts.
* Documented codebase (docstrings, inline comments, README updates).

## Week 10 (August 24 – August 29): Finalization
* Completed dissertation writing: results, discussion, and conclusion chapters.
* Finalized appendices (derivations, dataset details, development log).
* Performed proofreading and integrated supervisor feedback.
* Packaged supplementary code and data for submission.
