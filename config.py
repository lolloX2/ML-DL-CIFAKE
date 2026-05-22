"""
config.py – Configurazione centralizzata del progetto CIFAKE.
Tutti i path, gli iperparametri e le costanti sono definiti qui.
"""

import os

# ---------------------------------------------------------------------------
# Percorsi dataset
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_REAL_DIR = os.path.join(BASE_DIR, "CIFAKE", "train", "REAL")
TRAIN_FAKE_DIR = os.path.join(BASE_DIR, "CIFAKE", "train", "FAKE")
TEST_REAL_DIR  = os.path.join(BASE_DIR, "CIFAKE", "test",  "REAL")
TEST_FAKE_DIR  = os.path.join(BASE_DIR, "CIFAKE", "test",  "FAKE")

# ---------------------------------------------------------------------------
# Cartelle di output
# ---------------------------------------------------------------------------
OUTPUT_DIR      = os.path.join(BASE_DIR, "output")
EDA_DIR         = os.path.join(OUTPUT_DIR, "eda")
MODELS_DIR      = os.path.join(OUTPUT_DIR, "models")
METRICS_DIR     = os.path.join(OUTPUT_DIR, "metrics")
EVALUATION_DIR  = os.path.join(OUTPUT_DIR, "evaluation")

# Crea automaticamente le cartelle se non esistono
for _dir in [EDA_DIR, MODELS_DIR, METRICS_DIR, EVALUATION_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Iperparametri generali
# ---------------------------------------------------------------------------
RANDOM_SEED = 42

# Frazione del training set da riservare come validation (split in codice)
VAL_SPLIT_RATIO = 0.2   # 10% → ~9k val su ~100k train → train ~90k / val ~10k

# Dimensione immagini per ML classico (flatten)
IMAGE_SIZE_ML = 32          # pxs × px → vettore 32*32*3 = 3 072 feature

# Dimensione immagini per Deep Learning (ResNet-50 richiede almeno 224)
IMAGE_SIZE_DL = 128

# ---------------------------------------------------------------------------
# Iperparametri ML
# ---------------------------------------------------------------------------
ML_SUBSAMPLE_PER_CLASS = 5_000    # campioni per classe per SGD
RF_N_ESTIMATORS        = 200      # alberi Random Forest
SVM_C                  = 1.0
SGD_N_EPOCHS           = 15

# ---------------------------------------------------------------------------
# Iperparametri Deep Learning
# ---------------------------------------------------------------------------
BATCH_SIZE     = 64
NUM_WORKERS    = 4

# Fase 1: addestramento solo del classificatore
PHASE1_EPOCHS  = 3
PHASE1_LR      = 1e-3

# Fase 2: fine-tuning di layer3 + layer4 + classificatore
PHASE2_EPOCHS  = 5
PHASE2_LR      = 1e-4

DROPOUT_RATE   = 0.3
WEIGHT_DECAY   = 1e-4

# Numero di classi
NUM_CLASSES = 2
CLASS_NAMES = ["REAL", "FAKE"]   # indice 0 = REAL, 1 = FAKE

# ---------------------------------------------------------------------------
# Path file salvati
# ---------------------------------------------------------------------------
ML_RESULTS_JSON    = os.path.join(METRICS_DIR,     "ml_results.json")
SVM_MODEL_PATH     = os.path.join(MODELS_DIR,      "svm_model.joblib")
RF_MODEL_PATH      = os.path.join(MODELS_DIR,      "rf_model.joblib")
DL_BEST_MODEL_PATH = os.path.join(MODELS_DIR,      "resnet18_best.pth")
DL_RESULTS_JSON    = os.path.join(METRICS_DIR,     "dl_results.json")
