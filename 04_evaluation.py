"""
04_evaluation.py – Confronto finale tra tutti i modelli su CIFAKE.

Carica i risultati salvati da 02_ml_pipeline.py e 03_dl_pipeline.py
e genera:
  - Classification report testuale per ogni modello
  - Matrice di confusione per ogni modello (se disponibili le predizioni)
  - Curva ROC per ogni modello
  - Grafico comparativo barre: Accuracy + AUC di tutti i modelli affiancati

Tutti i grafici vengono salvati in output/evaluation/
"""

import os
import sys
import json

import numpy as np
import joblib
import torch
import torch.nn as nn
from torchvision import models

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, roc_curve, auc,
    confusion_matrix
)

# Importa configurazione e utility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    TEST_REAL_DIR,  TEST_FAKE_DIR,
    IMAGE_SIZE_ML, IMAGE_SIZE_DL,
    RANDOM_SEED, BATCH_SIZE, NUM_WORKERS,
    ML_RESULTS_JSON, DL_RESULTS_JSON,
    SVM_MODEL_PATH, RF_MODEL_PATH, DL_BEST_MODEL_PATH,
    EVALUATION_DIR, CLASS_NAMES,
    DROPOUT_RATE, NUM_CLASSES
)
from utils import (
    load_images_flat, get_dataloaders,
    plot_confusion_matrix, plot_roc_curve, _SGDCalibrated  # noqa: F401 — needed for joblib/pickle deserialization
)


# ---------------------------------------------------------------------------
# Caricamento dati di test
# ---------------------------------------------------------------------------

def load_test_data_ml() -> tuple[np.ndarray, np.ndarray]:
    """Carica i dati di test appiattiti per i modelli ML."""
    print("  Caricamento test set per ML (flatten 64×64)...")
    X_test, y_test = load_images_flat(
        TEST_REAL_DIR, TEST_FAKE_DIR,
        image_size=IMAGE_SIZE_ML,
        max_per_class=None,
        seed=RANDOM_SEED
    )
    return X_test, y_test


def load_test_dataloader() -> torch.utils.data.DataLoader:
    """Crea il DataLoader di test per ResNet-18."""
    print("  Creazione DataLoader test per DL (224×224)...")
    _, _, test_loader = get_dataloaders(
        TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
        TEST_REAL_DIR,  TEST_FAKE_DIR,
        image_size=IMAGE_SIZE_DL,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        seed=RANDOM_SEED
    )
    return test_loader


# ---------------------------------------------------------------------------
# Funzione per ricaricare ResNet-18 dal checkpoint
# ---------------------------------------------------------------------------

def load_resnet18(checkpoint_path: str,
                  device: torch.device) -> nn.Module:
    """
    Ricostruisce l'architettura ResNet-18 modificata e carica i pesi
    dal checkpoint salvato da 03_dl_pipeline.py.
    """
    model = models.resnet18(weights=None)   # non ricarichiamo ImageNet
    in_features = model.fc.in_features      # 512 per ResNet-18
    model.fc = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=DROPOUT_RATE),
        nn.Linear(in_features, NUM_CLASSES)
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    print(f"  ResNet-18 caricato da: {checkpoint_path}")
    print(f"  (epoch={checkpoint['epoch']}, phase={checkpoint['phase']}, "
          f"val_loss={checkpoint['val_loss']:.4f})")
    return model


@torch.no_grad()
def predict_dl(model: nn.Module,
               loader: torch.utils.data.DataLoader,
               device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Esegue le predizioni sul test set con il modello DL.

    Ritorna
    -------
    y_true  : etichette vere
    y_pred  : predizioni argmax
    y_proba : probabilità classe FAKE (per la curva ROC)
    """
    all_labels, all_preds, all_proba = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        outputs = model(images)
        proba   = torch.softmax(outputs, dim=1)[:, 1]
        preds   = outputs.argmax(dim=1)

        all_labels.extend(labels.numpy())
        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())

    return (np.array(all_labels),
            np.array(all_preds),
            np.array(all_proba))


# ---------------------------------------------------------------------------
# Grafici di confronto
# ---------------------------------------------------------------------------

def plot_roc_all_models(models_data: list[dict], save_path: str) -> None:
    """
    Disegna le curve ROC di tutti i modelli su un unico grafico.

    Parametri
    ----------
    models_data : lista di dizionari con chiavi
                  'name', 'y_true', 'y_proba', 'color'
    save_path   : percorso di salvataggio
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    for m in models_data:
        fpr, tpr, _ = roc_curve(m["y_true"], m["y_proba"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=m["color"],
                label=f"{m['name']}  (AUC = {roc_auc:.4f})")
        ax.fill_between(fpr, tpr, alpha=0.05, color=m["color"])

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Classificatore casuale")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title("Confronto Curve ROC – Tutti i Modelli",
                 fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confronto ROC salvato: {save_path}")


def plot_comparison_bars(model_names: list[str],
                         accuracies: list[float],
                         aucs: list[float],
                         save_path: str) -> None:
    """
    Grafico a barre affiancate: Accuracy e AUC per ogni modello.

    Parametri
    ----------
    model_names : lista dei nomi dei modelli
    accuracies  : lista dei valori di accuracy
    aucs        : lista dei valori AUC
    save_path   : percorso di salvataggio
    """
    x = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars_acc = ax.bar(x - width / 2, accuracies, width,
                      label="Accuracy", color="#2196F3", alpha=0.85,
                      edgecolor="white", linewidth=0.5)
    bars_auc = ax.bar(x + width / 2, aucs, width,
                      label="AUC", color="#F44336", alpha=0.85,
                      edgecolor="white", linewidth=0.5)

    # Etichette sui valori
    for bar in bars_acc:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")
    for bar in bars_auc:
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005,
                f"{bar.get_height():.4f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Confronto Modelli – Accuracy e AUC sul Test Set",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=12)
    ax.set_ylabel("Punteggio", fontsize=12)
    ax.set_ylim(0, 1.08)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)

    # Linea di riferimento a 0.5 (classificatore casuale)
    ax.axhline(y=0.5, color="gray", linestyle="--", lw=1, alpha=0.5,
               label="Baseline casuale")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Confronto barre salvato: {save_path}")


def plot_metrics_heatmap(model_names: list[str],
                         metrics_dict: dict,
                         save_path: str) -> None:
    """
    Heatmap delle metriche principali (precision, recall, f1) per classe
    e per modello.
    """
    # Raccoglie le metriche dal classification_report
    rows = []
    index = []
    for name in model_names:
        report = metrics_dict[name]
        for cls in CLASS_NAMES:
            rows.append([
                report[cls]["precision"],
                report[cls]["recall"],
                report[cls]["f1-score"],
            ])
            index.append(f"{name}\n({cls})")

    data = np.array(rows)
    col_labels = ["Precision", "Recall", "F1-score"]

    fig, ax = plt.subplots(figsize=(9, max(4, len(index) * 0.7)))
    im = ax.imshow(data, cmap="YlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticks(range(len(index)))
    ax.set_yticklabels(index, fontsize=9)

    # Aggiungi i valori nelle celle
    for i in range(len(index)):
        for j in range(len(col_labels)):
            ax.text(j, i, f"{data[i, j]:.3f}",
                    ha="center", va="center",
                    color="black" if data[i, j] > 0.4 else "white",
                    fontsize=10)

    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Metriche per Classe e Modello",
                 fontsize=13, fontweight="bold", pad=15)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap metriche salvata: {save_path}")


# ---------------------------------------------------------------------------
# Script principale
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  04_EVALUATION – Confronto finale modelli CIFAKE")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")

    # ------------------------------------------------------------------
    # Caricamento dati di test
    # ------------------------------------------------------------------
    print("\n[1/5] Caricamento dati di test...")
    X_test_ml, y_test_ml = load_test_data_ml()
    test_loader = load_test_dataloader()

    # ------------------------------------------------------------------
    # Caricamento modelli ML e predizioni
    # ------------------------------------------------------------------
    print("\n[2/5] Caricamento modelli ML e predizioni...")
    models_predictions = {}  # {nome: (y_true, y_pred, y_proba)}

    # SVM
    if os.path.exists(SVM_MODEL_PATH):
        print(f"  Caricamento SVM da {SVM_MODEL_PATH}...")
        svm = joblib.load(SVM_MODEL_PATH)
        y_pred_svm  = svm.predict(X_test_ml)
        y_proba_svm = svm.predict_proba(X_test_ml)[:, 1]
        models_predictions["SVM"] = (y_test_ml, y_pred_svm, y_proba_svm)
        print(f"  SVM – Accuracy: {accuracy_score(y_test_ml, y_pred_svm):.4f}")
    else:
        print(f"  [SKIP] SVM non trovato: {SVM_MODEL_PATH}")

    # Random Forest
    if os.path.exists(RF_MODEL_PATH):
        print(f"  Caricamento Random Forest da {RF_MODEL_PATH}...")
        rf = joblib.load(RF_MODEL_PATH)
        y_pred_rf  = rf.predict(X_test_ml)
        y_proba_rf = rf.predict_proba(X_test_ml)[:, 1]
        models_predictions["RandomForest"] = (y_test_ml, y_pred_rf, y_proba_rf)
        print(f"  RF  – Accuracy: {accuracy_score(y_test_ml, y_pred_rf):.4f}")
    else:
        print(f"  [SKIP] Random Forest non trovato: {RF_MODEL_PATH}")

    # ResNet-18
    if os.path.exists(DL_BEST_MODEL_PATH):
        print(f"\n  Caricamento ResNet-18 da {DL_BEST_MODEL_PATH}...")
        resnet = load_resnet18(DL_BEST_MODEL_PATH, device)
        y_true_dl, y_pred_dl, y_proba_dl = predict_dl(resnet, test_loader, device)
        models_predictions["ResNet18"] = (y_true_dl, y_pred_dl, y_proba_dl)
        print(f"  ResNet-18 – Accuracy: {accuracy_score(y_true_dl, y_pred_dl):.4f}")
    else:
        print(f"  [SKIP] ResNet-18 non trovato: {DL_BEST_MODEL_PATH}")

    if not models_predictions:
        print("\n  ERRORE: Nessun modello trovato. Esegui prima 02 e 03.")
        return

    # ------------------------------------------------------------------
    # 3. Report dettagliati e grafici per ogni modello
    # ------------------------------------------------------------------
    print("\n[3/5] Report e grafici per ogni modello...")

    summary_metrics = {}     # per il confronto finale
    all_reports     = {}     # per la heatmap metriche

    for model_name, (y_true, y_pred, y_proba) in models_predictions.items():
        print(f"\n  {'='*40}")
        print(f"  Modello: {model_name}")
        print(f"  {'='*40}")

        acc = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)
        report  = classification_report(y_true, y_pred,
                                        target_names=CLASS_NAMES,
                                        output_dict=True)

        print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
        print(f"  AUC: {roc_auc:.4f}")

        summary_metrics[model_name] = {"accuracy": acc, "auc": roc_auc}
        all_reports[model_name]     = report

        # Matrice di confusione
        plot_confusion_matrix(
            y_true, y_pred, CLASS_NAMES,
            title=f"Matrice di Confusione – {model_name}",
            save_path=os.path.join(EVALUATION_DIR,
                                   f"{model_name.lower()}_confusion_matrix.png")
        )

        # Curva ROC
        plot_roc_curve(
            y_true, y_proba,
            model_name=model_name,
            save_path=os.path.join(EVALUATION_DIR,
                                   f"{model_name.lower()}_roc_curve.png")
        )

    # ------------------------------------------------------------------
    # 4. Confronto ROC sovrapposto
    # ------------------------------------------------------------------
    print("\n[4/5] Grafico confronto ROC e barre comparativi...")

    color_map = {
        "SVM":          "#E91E63",
        "RandomForest": "#FF9800",
        "ResNet18":     "#2196F3",
    }

    roc_data = [
        {
            "name":    name,
            "y_true":  y_true,
            "y_proba": y_proba,
            "color":   color_map.get(name, "gray")
        }
        for name, (y_true, _, y_proba) in models_predictions.items()
    ]
    plot_roc_all_models(
        roc_data,
        save_path=os.path.join(EVALUATION_DIR, "all_models_roc_comparison.png")
    )

    # ------------------------------------------------------------------
    # 5. Grafico a barre comparativo (Accuracy + AUC)
    # ------------------------------------------------------------------
    model_names = list(summary_metrics.keys())
    accuracies  = [summary_metrics[n]["accuracy"] for n in model_names]
    aucs        = [summary_metrics[n]["auc"]      for n in model_names]

    plot_comparison_bars(
        model_names, accuracies, aucs,
        save_path=os.path.join(EVALUATION_DIR, "all_models_bar_comparison.png")
    )

    # Heatmap metriche per classe
    plot_metrics_heatmap(
        model_names, all_reports,
        save_path=os.path.join(EVALUATION_DIR, "all_models_metrics_heatmap.png")
    )

    # ------------------------------------------------------------------
    # Salva il report riepilogativo in JSON
    # ------------------------------------------------------------------
    final_report = {
        name: {
            "accuracy": round(float(m["accuracy"]), 4),
            "auc":      round(float(m["auc"]),      4),
        }
        for name, m in summary_metrics.items()
    }
    report_path = os.path.join(EVALUATION_DIR, "final_comparison.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2, ensure_ascii=False)
    print(f"  Report finale salvato: {report_path}")

    # ------------------------------------------------------------------
    # Riepilogo tabellare finale
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RIEPILOGO CONFRONTO FINALE")
    print("=" * 60)
    print(f"  {'Modello':<20} {'Accuracy':>10} {'AUC':>10}")
    print(f"  {'-'*42}")
    for name in model_names:
        acc = summary_metrics[name]["accuracy"]
        a   = summary_metrics[name]["auc"]
        print(f"  {name:<20} {acc:>10.4f} {a:>10.4f}")
    print("=" * 60)

    best_model = max(summary_metrics, key=lambda n: summary_metrics[n]["accuracy"])
    print(f"\n  Modello migliore per Accuracy: {best_model}  "
          f"({summary_metrics[best_model]['accuracy']:.4f})")
    best_auc_model = max(summary_metrics, key=lambda n: summary_metrics[n]["auc"])
    print(f"  Modello migliore per AUC:      {best_auc_model}  "
          f"({summary_metrics[best_auc_model]['auc']:.4f})")

    print(f"\n  Output salvati in: {EVALUATION_DIR}")
    for fname in sorted(os.listdir(EVALUATION_DIR)):
        print(f"    - {fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
