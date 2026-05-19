"""
05_val_test_eval.py – Valutazione del modello ResNet-18 già addestrato
                      su validation set e test set separatamente.

Carica il checkpoint salvato da 03_dl_pipeline.py e produce:
  - Accuracy e AUC su val e test
  - Classification report su test
  - Confusion matrix e ROC curve su test  (output/eda/)
  - JSON con entrambe le metriche         (output/metrics/val_test_results.json)

Non esegue alcun training. Richiede che 03_dl_pipeline.py sia già stato eseguito.
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    TEST_REAL_DIR,  TEST_FAKE_DIR,
    IMAGE_SIZE_DL, RANDOM_SEED,
    BATCH_SIZE, NUM_WORKERS,
    DROPOUT_RATE, NUM_CLASSES, CLASS_NAMES,
    VAL_SPLIT_RATIO,
    DL_BEST_MODEL_PATH, EDA_DIR, METRICS_DIR
)
from utils import get_dataloaders, plot_confusion_matrix, plot_roc_curve

VAL_TEST_RESULTS_JSON = os.path.join(METRICS_DIR, "val_test_results.json")


def build_resnet18(dropout: float, num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )
    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_proba = [], [], []

    for images, labels in tqdm(loader, desc="  Evaluating", leave=False, ncols=80):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        proba = torch.softmax(outputs, dim=1)[:, 1]
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())

    avg_loss = running_loss / len(loader.dataset)
    y_true  = np.array(all_labels)
    y_pred  = np.array(all_preds)
    y_proba = np.array(all_proba)
    acc = accuracy_score(y_true, y_pred)
    return avg_loss, acc, y_true, y_pred, y_proba


def main():
    print("=" * 60)
    print("  05_VAL_TEST_EVAL – Valutazione ResNet-18 su Val e Test")
    print("=" * 60)

    if not os.path.exists(DL_BEST_MODEL_PATH):
        print(f"\n  ERRORE: checkpoint non trovato: {DL_BEST_MODEL_PATH}")
        print("  Esegui prima 03_dl_pipeline.py.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")

    # ------------------------------------------------------------------
    # DataLoader (stesso split usato in training)
    # ------------------------------------------------------------------
    print("\n  Creazione DataLoader...")
    train_loader, val_loader, test_loader = get_dataloaders(
        TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
        TEST_REAL_DIR,  TEST_FAKE_DIR,
        image_size=IMAGE_SIZE_DL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED,
        val_split=VAL_SPLIT_RATIO
    )

    # ------------------------------------------------------------------
    # Caricamento checkpoint
    # ------------------------------------------------------------------
    print(f"\n  Caricamento checkpoint: {DL_BEST_MODEL_PATH}")
    model = build_resnet18(DROPOUT_RATE, NUM_CLASSES)
    checkpoint = torch.load(DL_BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    print(f"  Salvato a: epoch={checkpoint['epoch']}, "
          f"phase={checkpoint['phase']}, "
          f"val_loss={checkpoint['val_loss']:.4f}")

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------
    # Valutazione validation
    # ------------------------------------------------------------------
    print("\n  Valutazione su Validation set...")
    val_loss, val_acc, y_true_v, y_pred_v, y_proba_v = evaluate(
        model, val_loader, criterion, device
    )
    val_auc = roc_auc_score(y_true_v, y_proba_v)
    print(f"  Val   – Loss: {val_loss:.4f}  Accuracy: {val_acc:.4f}  AUC: {val_auc:.4f}")

    # ------------------------------------------------------------------
    # Valutazione test
    # ------------------------------------------------------------------
    print("\n  Valutazione su Test set...")
    test_loss, test_acc, y_true_t, y_pred_t, y_proba_t = evaluate(
        model, test_loader, criterion, device
    )
    test_auc = roc_auc_score(y_true_t, y_proba_t)
    print(f"  Test  – Loss: {test_loss:.4f}  Accuracy: {test_acc:.4f}  AUC: {test_auc:.4f}")

    print("\n  Classification report (Test):")
    print(classification_report(y_true_t, y_pred_t, target_names=CLASS_NAMES))

    # ------------------------------------------------------------------
    # Grafici (test)
    # ------------------------------------------------------------------
    plot_confusion_matrix(
        y_true_t, y_pred_t, CLASS_NAMES,
        title="Matrice di Confusione – ResNet-18 (Test)",
        save_path=os.path.join(EDA_DIR, "eval_resnet18_confusion_matrix.png")
    )
    plot_roc_curve(
        y_true_t, y_proba_t,
        model_name="ResNet-18 (Test)",
        save_path=os.path.join(EDA_DIR, "eval_resnet18_roc_curve.png")
    )

    # ------------------------------------------------------------------
    # Riepilogo e JSON
    # ------------------------------------------------------------------
    gap_acc = test_acc - val_acc
    gap_auc = test_auc - val_auc

    print("\n" + "=" * 60)
    print("  RIEPILOGO")
    print("=" * 60)
    print(f"  {'':10}  {'Accuracy':>10}  {'AUC':>8}")
    print(f"  {'Val':10}  {val_acc:>10.4f}  {val_auc:>8.4f}")
    print(f"  {'Test':10}  {test_acc:>10.4f}  {test_auc:>8.4f}")
    print(f"  {'Gap':10}  {gap_acc:>+10.4f}  {gap_auc:>+8.4f}")
    print("=" * 60)

    results = {
        "model": "ResNet-18",
        "checkpoint": DL_BEST_MODEL_PATH,
        "val": {
            "loss":     round(float(val_loss), 4),
            "accuracy": round(float(val_acc),  4),
            "auc":      round(float(val_auc),   4),
        },
        "test": {
            "loss":     round(float(test_loss), 4),
            "accuracy": round(float(test_acc),  4),
            "auc":      round(float(test_auc),   4),
            "classification_report": classification_report(
                y_true_t, y_pred_t, target_names=CLASS_NAMES, output_dict=True
            )
        },
        "gap": {
            "accuracy": round(float(gap_acc), 4),
            "auc":      round(float(gap_auc),  4),
        }
    }
    with open(VAL_TEST_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n  Risultati salvati: {VAL_TEST_RESULTS_JSON}")


if __name__ == "__main__":
    main()
