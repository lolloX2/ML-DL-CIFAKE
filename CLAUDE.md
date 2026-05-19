# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CIFAKE binary image classification study: real vs. AI-generated face images (120k images, 32×32). Compares classical ML (SVM, Random Forest) against deep learning (ResNet-50 fine-tuning).

## Running the Pipeline

Install dependencies once:
```bash
pip install -r requirements.txt
```

Run scripts in order:
```bash
python 01_eda.py           # Exploratory data analysis → output/eda/
python 02_ml_pipeline.py   # Train SVM + Random Forest → output/metrics/ml_results.json
python 03_dl_pipeline.py   # Fine-tune ResNet-50 → output/models/resnet50_best.pth
python 04_evaluation.py    # Compare all models → output/evaluation/
```

Each script is self-contained and can be re-run independently. There is no test suite or linter configured.

## Architecture

All hyperparameters and file paths live in `config.py` — edit there before re-running scripts.

`utils.py` provides two data-loading paths used across scripts:
- `load_images_flat(path, size)` — flattens images to 1-D numpy arrays for scikit-learn models
- `get_dataloaders(train_dir, test_dir, batch_size)` — returns PyTorch DataLoaders (224×224, ImageNet normalization) for the ResNet pipeline

**Classical ML (`02_ml_pipeline.py`):** Loads 64×64 images, subsamples to 15k/class for SVM tractability, fits SVM (RBF, C=1.0) and Random Forest (200 trees), saves models via joblib.

**Deep Learning (`03_dl_pipeline.py`):** Two-phase fine-tuning of pretrained ResNet-50.
- Phase 1 (5 epochs, lr=1e-3): only the classifier head is trainable.
- Phase 2 (10 epochs, lr=1e-4): unfreezes `layer3`, `layer4`, and classifier. Best checkpoint saved by validation accuracy.

**Evaluation (`04_evaluation.py`):** Loads `ml_results.json` and `dl_results.json`, generates per-model confusion matrices, ROC curves, and a side-by-side accuracy/AUC comparison chart.

## Dataset Layout

```
CIFAKE/
  train/  REAL/ (50k)   FAKE/ (50k)
  test/   REAL/ (10k)   FAKE/ (10k)
```

The dataset is not version-controlled. If missing, download CIFAKE from Kaggle and place it at `CIFAKE/` relative to this directory.
