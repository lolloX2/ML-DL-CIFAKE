"""
02_ml_pipeline.py – Pipeline di Machine Learning classico per CIFAKE.

Modelli addestrati:
  - SVM lineare con SGDClassifier(loss='hinge') + calibrazione isotonica
  - Random Forest (200 alberi)

Flusso:
  1. Carica immagini train/test (64×64, flatten, normalizzate /255)
  2. Subsample training per SVM (15.000 campioni/classe)
  3. Addestra e valuta entrambi i modelli
  4. Salva modelli in output/models/
  5. Salva metriche in output/metrics/ml_results.json
"""

import os
import sys
import json
import time

import numpy as np
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score
)
from tqdm import tqdm

# Importa configurazione e utility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    TEST_REAL_DIR,  TEST_FAKE_DIR,
    IMAGE_SIZE_ML, RANDOM_SEED,
    ML_SUBSAMPLE_PER_CLASS,
    RF_N_ESTIMATORS, SVM_C, SGD_N_EPOCHS,
    VAL_SPLIT_RATIO,
    SVM_MODEL_PATH, RF_MODEL_PATH,
    ML_RESULTS_JSON, EDA_DIR,
    CLASS_NAMES
)
from utils import load_images_flat, plot_confusion_matrix, plot_roc_curve, _SGDCalibrated


# ---------------------------------------------------------------------------
# Funzione principale
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  02_ML_PIPELINE – SVM e Random Forest su CIFAKE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Caricamento dati
    # ------------------------------------------------------------------
    print("\n[1/4] Caricamento dataset train (subsample per SVM)...")
    t0 = time.time()
    X_all_train, y_all_train = load_images_flat(
        TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
        image_size=IMAGE_SIZE_ML,
        max_per_class=ML_SUBSAMPLE_PER_CLASS,
        seed=RANDOM_SEED
    )
    print(f"  Campioni caricati: {X_all_train.shape}  |  "
          f"Labels: {np.bincount(y_all_train)}  |  "
          f"Tempo: {time.time()-t0:.1f}s")

    # Split stratificato train / validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_all_train, y_all_train,
        test_size=VAL_SPLIT_RATIO,
        stratify=y_all_train,
        random_state=RANDOM_SEED
    )
    print(f"  Train shape: {X_train.shape}  |  Val shape: {X_val.shape}")

    print("\n  Caricamento dataset test (completo)...")
    t0 = time.time()
    X_test, y_test = load_images_flat(
        TEST_REAL_DIR, TEST_FAKE_DIR,
        image_size=IMAGE_SIZE_ML,
        max_per_class=None,   # usa tutti i campioni di test
        seed=RANDOM_SEED
    )
    print(f"  Test shape: {X_test.shape}  |  "
          f"Labels: {np.bincount(y_test)}  |  "
          f"Tempo: {time.time()-t0:.1f}s")

    # Dizionario dove accumuliamo i risultati per il JSON
    results = {}

    # ------------------------------------------------------------------
    # 2. SVM lineare (LinearSVC)
    # ------------------------------------------------------------------
    print("\n[2/4] Addestramento SVM (SGDClassifier hinge + calibrazione isotonica)...")
    print(f"  Campioni train usati: {len(X_train):,}")

    t0 = time.time()
    svm = _fit_sgd_with_progress(
        X_train, y_train,
        C=SVM_C, n_epochs=SGD_N_EPOCHS, seed=RANDOM_SEED,
    )
    train_time_svm = time.time() - t0
    print(f"  Addestramento completato in {train_time_svm:.1f}s")

    # Valutazione SVM su validation
    print("  Valutazione SVM sul validation set...")
    acc_svm_val = accuracy_score(y_val, svm.predict(X_val))
    auc_svm_val = roc_auc_score(y_val, svm.predict_proba(X_val)[:, 1])
    print(f"  Val Accuracy: {acc_svm_val:.4f}  |  Val AUC: {auc_svm_val:.4f}")

    # Valutazione SVM sul test set (una sola volta, a fine pipeline)
    print("  Valutazione SVM sul test set...")
    y_pred_svm   = svm.predict(X_test)
    y_proba_svm  = svm.predict_proba(X_test)[:, 1]   # probabilità classe FAKE
    acc_svm      = accuracy_score(y_test, y_pred_svm)
    auc_svm      = roc_auc_score(y_test, y_proba_svm)
    report_svm   = classification_report(y_test, y_pred_svm,
                                         target_names=CLASS_NAMES,
                                         output_dict=True)

    print(f"  Test Accuracy SVM: {acc_svm:.4f}  |  AUC: {auc_svm:.4f}")
    print(classification_report(y_test, y_pred_svm, target_names=CLASS_NAMES))

    # Salva modello SVM
    joblib.dump(svm, SVM_MODEL_PATH)
    print(f"  Modello SVM salvato: {SVM_MODEL_PATH}")

    # Grafici SVM
    plot_confusion_matrix(
        y_test, y_pred_svm, CLASS_NAMES,
        title="Matrice di Confusione – SVM (SGD hinge)",
        save_path=os.path.join(EDA_DIR, "ml_svm_confusion_matrix.png")
    )
    auc_svm = plot_roc_curve(
        y_test, y_proba_svm,
        model_name="SVM (SGD hinge)",
        save_path=os.path.join(EDA_DIR, "ml_svm_roc_curve.png")
    )

    results["SVM"] = {
        "accuracy":         round(acc_svm, 4),
        "auc":              round(auc_svm, 4),
        "val_accuracy":     round(acc_svm_val, 4),
        "val_auc":          round(auc_svm_val, 4),
        "train_time_s":     round(train_time_svm, 1),
        "train_samples":    int(len(X_train)),
        "classification_report": report_svm
    }

    # ------------------------------------------------------------------
    # 3. Random Forest
    # ------------------------------------------------------------------
    print(f"\n[3/4] Addestramento Random Forest ({RF_N_ESTIMATORS} alberi)...")
    t0 = time.time()
    rf = RandomForestClassifier(
        n_estimators=_RF_PROGRESS_BATCH,
        warm_start=True,
        n_jobs=-1,
        random_state=RANDOM_SEED,
    )
    steps = list(range(_RF_PROGRESS_BATCH, RF_N_ESTIMATORS + 1, _RF_PROGRESS_BATCH))
    if steps[-1] < RF_N_ESTIMATORS:
        steps.append(RF_N_ESTIMATORS)
    with tqdm(steps, desc="  RF alberi", unit="step", ncols=72,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
              ) as pbar:
        for n in pbar:
            rf.n_estimators = n
            rf.fit(X_train, y_train)
            pbar.set_postfix(alberi=n)
    train_time_rf = time.time() - t0
    print(f"  Addestramento completato in {train_time_rf:.1f}s")

    # Valutazione Random Forest su validation
    print("  Valutazione Random Forest sul validation set...")
    acc_rf_val = accuracy_score(y_val, rf.predict(X_val))
    auc_rf_val = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
    print(f"  Val Accuracy: {acc_rf_val:.4f}  |  Val AUC: {auc_rf_val:.4f}")

    # Valutazione Random Forest sul test set (una sola volta, a fine pipeline)
    print("  Valutazione Random Forest sul test set...")
    y_pred_rf   = rf.predict(X_test)
    y_proba_rf  = rf.predict_proba(X_test)[:, 1]
    acc_rf      = accuracy_score(y_test, y_pred_rf)
    auc_rf      = roc_auc_score(y_test, y_proba_rf)
    report_rf   = classification_report(y_test, y_pred_rf,
                                        target_names=CLASS_NAMES,
                                        output_dict=True)

    print(f"  Test Accuracy RF: {acc_rf:.4f}  |  AUC: {auc_rf:.4f}")
    print(classification_report(y_test, y_pred_rf, target_names=CLASS_NAMES))

    # Salva modello Random Forest
    joblib.dump(rf, RF_MODEL_PATH)
    print(f"  Modello RF salvato: {RF_MODEL_PATH}")

    # Grafici Random Forest
    plot_confusion_matrix(
        y_test, y_pred_rf, CLASS_NAMES,
        title="Matrice di Confusione – Random Forest",
        save_path=os.path.join(EDA_DIR, "ml_rf_confusion_matrix.png")
    )
    auc_rf = plot_roc_curve(
        y_test, y_proba_rf,
        model_name="Random Forest",
        save_path=os.path.join(EDA_DIR, "ml_rf_roc_curve.png")
    )

    # Importanza delle feature per RF (top pixel)
    _plot_feature_importance(rf, IMAGE_SIZE_ML)

    results["RandomForest"] = {
        "accuracy":         round(acc_rf, 4),
        "auc":              round(auc_rf, 4),
        "val_accuracy":     round(acc_rf_val, 4),
        "val_auc":          round(auc_rf_val, 4),
        "train_time_s":     round(train_time_rf, 1),
        "train_samples":    int(len(X_train)),
        "classification_report": report_rf
    }

    # ------------------------------------------------------------------
    # 4. Salva metriche in JSON
    # ------------------------------------------------------------------
    print(f"\n[4/4] Salvataggio metriche in {ML_RESULTS_JSON}...")
    with open(ML_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"  Salvato: {ML_RESULTS_JSON}")

    # ------------------------------------------------------------------
    # Riepilogo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RIEPILOGO RISULTATI ML")
    print("=" * 60)
    print(f"  {'Modello':<20} {'Accuracy':>10} {'AUC':>10} {'Tempo (s)':>12}")
    print(f"  {'-'*55}")
    for name, res in results.items():
        print(f"  {name:<20} {res['accuracy']:>10.4f} {res['auc']:>10.4f} "
              f"{res['train_time_s']:>12.1f}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Funzione ausiliaria: SVM calibrato con progress bar per fold
# ---------------------------------------------------------------------------

_RF_PROGRESS_BATCH = 20   # alberi per step nella barra RF


def _fit_sgd_with_progress(X, y, C, n_epochs, seed):
    """
    Addestra SGDClassifier(loss='hinge') con partial_fit su n_epochs epoche,
    mostrando una barra tqdm per epoca. Calibra le probabilità su un hold-out
    del 20% tramite regressione isotonica sulla decision_function.
    """
    X_sgd, X_cal, y_sgd, y_cal = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # alpha = 1 / (C * n) è l'equivalente di C nella formulazione primale
    alpha = 1.0 / (C * len(X_sgd))
    sgd = SGDClassifier(
        loss="hinge",
        alpha=alpha,
        max_iter=1,
        tol=None,
        random_state=seed,
    )
    classes = np.array([0, 1])
    rng = np.random.RandomState(seed)

    with tqdm(range(n_epochs), desc="  SGD epoche", unit="ep", ncols=72,
              bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} ep [{elapsed}<{remaining}]"
              ) as pbar:
        for _ in pbar:
            idx = rng.permutation(len(X_sgd))
            sgd.partial_fit(X_sgd[idx], y_sgd[idx], classes=classes)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(sgd.decision_function(X_cal), y_cal)
    return _SGDCalibrated(sgd, iso)


# ---------------------------------------------------------------------------
# Funzione ausiliaria per importanza feature Random Forest
# ---------------------------------------------------------------------------

def _plot_feature_importance(rf: RandomForestClassifier,
                              image_size: int) -> None:
    """
    Visualizza l'importanza delle feature del Random Forest come
    heatmap sui canali R, G, B dell'immagine 64×64.
    """
    importances = rf.feature_importances_   # shape (image_size*image_size*3,)
    n_pixels = image_size * image_size

    # Riorganizza in (H, W, 3)
    imp_map = importances.reshape(image_size, image_size, 3)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    channel_names = ["Rosso (R)", "Verde (G)", "Blu (B)"]
    cmaps = ["Reds", "Greens", "Blues"]

    for i, (ax, name, cmap) in enumerate(zip(axes[:3], channel_names, cmaps)):
        im = ax.imshow(imp_map[:, :, i], cmap=cmap, aspect="equal")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"Importanza {name}", fontsize=11)
        ax.axis("off")

    # Mappa combinata (somma canali)
    combined = imp_map.sum(axis=2)
    im = axes[3].imshow(combined, cmap="hot", aspect="equal")
    plt.colorbar(im, ax=axes[3], shrink=0.8)
    axes[3].set_title("Importanza combinata (R+G+B)", fontsize=11)
    axes[3].axis("off")

    fig.suptitle("Importanza delle feature – Random Forest",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "ml_rf_feature_importance.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Feature importance salvata: {save_path}")


if __name__ == "__main__":
    main()
