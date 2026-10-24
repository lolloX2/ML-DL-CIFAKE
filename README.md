# CIFAKE — Real vs. AI-Generated Face Classification

Binary image classification study on the [CIFAKE](https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images) dataset (120k 32×32 images): distinguishing real photographs from AI-generated (synthetic) images. Compares classical ML baselines (SVM, Random Forest) against a fine-tuned ResNet-18.

## Results

| Model | Accuracy | AUC |
|---|---|---|
| SVM (RBF) | 0.642 | 0.688 |
| Random Forest | 0.802 | 0.879 |
| ResNet-18 (fine-tuned) | 0.958 | 0.996 |

Full metrics (per-class precision/recall/F1) in `output/metrics/`.

## Setup

```bash
pip install -r requirements.txt
```

Download CIFAKE from Kaggle and place it at `CIFAKE/` relative to the project root:

```
CIFAKE/
  train/  REAL/ (50k)   FAKE/ (50k)
  test/   REAL/ (10k)   FAKE/ (10k)
```

## Usage

Run scripts in order:

```bash
python 01_eda.py             # Exploratory data analysis → output/eda/
python 02_ml_pipeline.py     # Train SVM + Random Forest → output/metrics/ml_results.json
python 03_dl_pipeline.py     # Fine-tune ResNet-18 → output/models/resnet18_best.pth
python 04_evaluation.py      # Compare all models → output/evaluation/
python 05_val_test_eval.py   # Re-evaluate the trained ResNet-18 on val vs. test → output/metrics/val_test_results.json
```

Each script is self-contained and can be re-run independently. There is no test suite or linter configured.

## Architecture

All hyperparameters and file paths live in `config.py` — edit there before re-running scripts.

`utils.py` provides two data-loading paths used across scripts:
- `load_images_flat(path, size)` — flattens images to 1-D numpy arrays for scikit-learn models
- `get_dataloaders(...)` — returns PyTorch DataLoaders (128×128, ImageNet normalization, train/val/test split) for the deep learning pipeline

**Classical ML (`02_ml_pipeline.py`):** Loads 32×32 images (flattened to 3,072-feature vectors), subsamples to 5k/class for tractability, fits SVM (RBF, C=1.0) and Random Forest (200 trees), saves models via joblib.

**Deep Learning (`03_dl_pipeline.py`):** Two-phase fine-tuning of a pretrained ResNet-18, with a `Linear → ReLU → Dropout → Linear` classifier head.
- Phase 1 (3 epochs, lr=1e-3): only the classifier head is trainable.
- Phase 2 (5 epochs, lr=1e-4): unfreezes `layer3`, `layer4`, and the classifier head. Best checkpoint saved by validation loss.

Training uses an 80/20 train/validation split (`VAL_SPLIT_RATIO` in `config.py`).

**Evaluation (`04_evaluation.py`):** Loads `ml_results.json` and `dl_results.json`, generates per-model confusion matrices, ROC curves, and a side-by-side accuracy/AUC comparison chart.

**Val/test check (`05_val_test_eval.py`):** Loads the trained ResNet-18 checkpoint and reports accuracy/AUC separately on validation and test sets, to check for a generalization gap.

## Project Structure

```
01_eda.py             # exploratory data analysis
02_ml_pipeline.py     # SVM + Random Forest baselines
03_dl_pipeline.py     # ResNet-18 fine-tuning
04_evaluation.py      # cross-model comparison
05_val_test_eval.py   # val vs. test evaluation of the trained ResNet-18
config.py             # paths + hyperparameters
utils.py              # data loading + plotting helpers
output/               # eda plots, trained models, metrics, evaluation charts
```
