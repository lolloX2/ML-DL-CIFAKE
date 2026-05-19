"""
03_dl_pipeline.py – Fine-tuning ResNet-18 su CIFAKE.

Architettura:
  ResNet-18 pretrained su ImageNet
  → layer fc sostituito con: Linear(512→512) → ReLU → Dropout(0.3) → Linear(512→2)

Strategia di training in 2 fasi:
  Fase 1 (3 epoch) : solo il classificatore head viene addestrato (feature extractor frozen)
  Fase 2 (5 epoch) : si sblocca solo layer4 + head; layer1-3 rimangono frozen

Salva:
  - Il modello migliore per validation loss in output/models/resnet18_best.pth
  - Le curve di loss e accuracy in output/eda/
  - Le metriche finali in output/metrics/dl_results.json
"""

import os
import sys
import json
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

# Importa configurazione e utility
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    TEST_REAL_DIR,  TEST_FAKE_DIR,
    IMAGE_SIZE_DL, RANDOM_SEED,
    BATCH_SIZE, NUM_WORKERS,
    PHASE1_EPOCHS, PHASE1_LR,
    PHASE2_EPOCHS, PHASE2_LR,
    DROPOUT_RATE, WEIGHT_DECAY,
    NUM_CLASSES, CLASS_NAMES,
    VAL_SPLIT_RATIO,
    DL_BEST_MODEL_PATH, DL_RESULTS_JSON, EDA_DIR
)
from utils import get_dataloaders, plot_confusion_matrix, plot_roc_curve


# ---------------------------------------------------------------------------
# Riproducibilità
# ---------------------------------------------------------------------------
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Costruzione del modello
# ---------------------------------------------------------------------------

def build_model(num_classes: int = 2, dropout: float = 0.3) -> nn.Module:
    """
    Carica ResNet-18 pretrained su ImageNet e sostituisce il layer fully
    connected finale con un classificatore personalizzato a due livelli.

    Architettura del nuovo classificatore:
      Linear(512 → 512) → ReLU → Dropout(dropout) → Linear(512 → num_classes)
    """
    # Carica i pesi pretrained (usa il nuovo API di torchvision)
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)

    # Freeze di tutti i parametri (fase 1: solo il classifier è trainable)
    for param in model.parameters():
        param.requires_grad = False

    # Sostituisci il layer fc con il nuovo classificatore
    in_features = model.fc.in_features   # 512 per ResNet-18
    model.fc = nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(inplace=True),
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes)
    )

    return model


def unfreeze_phase2(model: nn.Module) -> None:
    """
    Sblocca solo layer4 e il classificatore per il fine-tuning della fase 2.
    layer1, layer2 e layer3 rimangono frozen per preservare le feature
    di basso e medio livello.
    """
    # Sblocca solo layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
    # Il classificatore (fc) era già trainable dalla fase 1
    for param in model.fc.parameters():
        param.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parametri trainable fase 2: {n_trainable:,}")


# ---------------------------------------------------------------------------
# Loop di training
# ---------------------------------------------------------------------------

def train_one_epoch(model: nn.Module,
                    loader: torch.utils.data.DataLoader,
                    criterion: nn.Module,
                    optimizer: optim.Optimizer,
                    device: torch.device,
                    epoch: int,
                    total_epochs: int) -> tuple[float, float]:
    """
    Esegue un'epoca di training.

    Ritorna
    -------
    avg_loss : loss media sull'epoca
    accuracy : accuracy sull'epoca
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f"  Epoch {epoch}/{total_epochs} [Train]",
                leave=False, ncols=90)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model: nn.Module,
             loader: torch.utils.data.DataLoader,
             criterion: nn.Module,
             device: torch.device) -> tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
    """
    Valuta il modello sul set di test o validazione.

    Ritorna
    -------
    avg_loss : loss media
    accuracy : accuracy
    y_true   : etichette vere (numpy)
    y_pred   : predizioni (numpy)
    y_proba  : probabilità della classe positiva (numpy)
    """
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds  = []
    all_proba  = []

    pbar = tqdm(loader, desc="  Evaluation", leave=False, ncols=90)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)

        proba = torch.softmax(outputs, dim=1)[:, 1]   # prob classe FAKE
        preds = outputs.argmax(dim=1)

        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_proba.extend(proba.cpu().numpy())

    n = sum(len(b) for b in loader)   # conta i campioni totali
    avg_loss = running_loss / len(loader.dataset)
    y_true   = np.array(all_labels)
    y_pred   = np.array(all_preds)
    y_proba  = np.array(all_proba)
    accuracy = accuracy_score(y_true, y_pred)

    return avg_loss, accuracy, y_true, y_pred, y_proba


# ---------------------------------------------------------------------------
# Visualizzazione curve di training
# ---------------------------------------------------------------------------

def plot_training_curves(history: dict, phase: int, save_path: str) -> None:
    """
    Salva le curve di loss e accuracy per una fase di training.

    Parametri
    ----------
    history   : dizionario con chiavi 'train_loss', 'val_loss',
                'train_acc', 'val_acc'
    phase     : numero della fase (1 o 2)
    save_path : percorso di salvataggio
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss",
                 markersize=5)
    axes[0].plot(epochs, history["val_loss"],   "r-o", label="Val Loss",
                 markersize=5)
    axes[0].set_title(f"Loss – Fase {phase}", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=11)
    axes[0].grid(alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-o", label="Train Acc",
                 markersize=5)
    axes[1].plot(epochs, history["val_acc"],   "r-o", label="Val Acc",
                 markersize=5)
    axes[1].set_title(f"Accuracy – Fase {phase}", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_ylim(0, 1.05)
    axes[1].legend(fontsize=11)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curve di training fase {phase} salvate: {save_path}")


# ---------------------------------------------------------------------------
# Script principale
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  03_DL_PIPELINE – Fine-tuning ResNet-18 su CIFAKE")
    print("=" * 60)

    # Dispositivo: GPU se disponibile, altrimenti CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n  Dispositivo: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM disponibile: "
              f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ------------------------------------------------------------------
    # 1. DataLoader
    # ------------------------------------------------------------------
    print("\n[1/4] Creazione DataLoader...")
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
    # 2. Costruzione modello
    # ------------------------------------------------------------------
    print("\n[2/4] Costruzione modello ResNet-18...")
    model = build_model(num_classes=NUM_CLASSES, dropout=DROPOUT_RATE)
    model = model.to(device)

    n_trainable_p1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total        = sum(p.numel() for p in model.parameters())
    print(f"  Parametri totali:     {n_total:,}")
    print(f"  Parametri trainable (fase 1): {n_trainable_p1:,}")

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float("inf")

    # Storia completa per i grafici
    full_history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  []
    }

    # ------------------------------------------------------------------
    # 3. Fase 1: Addestramento solo del classificatore head
    # ------------------------------------------------------------------
    print(f"\n[3/4] Fase 1 – Addestramento classifier head "
          f"({PHASE1_EPOCHS} epoch, lr={PHASE1_LR})...")

    optimizer_p1 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_p1 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p1, T_max=PHASE1_EPOCHS
    )

    history_p1 = {"train_loss": [], "val_loss": [],
                  "train_acc":  [], "val_acc":  []}

    for epoch in range(1, PHASE1_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p1,
            device, epoch, PHASE1_EPOCHS
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler_p1.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{PHASE1_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"Tempo: {elapsed:.0f}s")

        history_p1["train_loss"].append(train_loss)
        history_p1["val_loss"].append(val_loss)
        history_p1["train_acc"].append(train_acc)
        history_p1["val_acc"].append(val_acc)

        # Salva il modello migliore
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "phase":       1,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
            }, DL_BEST_MODEL_PATH)
            print(f"    Modello migliore salvato (val_loss={val_loss:.4f})")

    # Grafico fase 1
    plot_training_curves(
        history_p1, phase=1,
        save_path=os.path.join(EDA_DIR, "dl_training_curves_phase1.png")
    )
    full_history["train_loss"].extend(history_p1["train_loss"])
    full_history["val_loss"].extend(history_p1["val_loss"])
    full_history["train_acc"].extend(history_p1["train_acc"])
    full_history["val_acc"].extend(history_p1["val_acc"])

    # ------------------------------------------------------------------
    # 4. Fase 2: Fine-tuning layer4 e classificatore
    # ------------------------------------------------------------------
    print(f"\n[4/4] Fase 2 – Fine-tuning layer4+head "
          f"({PHASE2_EPOCHS} epoch, lr={PHASE2_LR})...")

    unfreeze_phase2(model)

    # Nuovo ottimizzatore con LR più basso per il fine-tuning
    optimizer_p2 = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE2_LR, weight_decay=WEIGHT_DECAY
    )
    scheduler_p2 = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_p2, T_max=PHASE2_EPOCHS
    )

    history_p2 = {"train_loss": [], "val_loss": [],
                  "train_acc":  [], "val_acc":  []}

    for epoch in range(1, PHASE2_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer_p2,
            device, epoch, PHASE2_EPOCHS
        )
        val_loss, val_acc, _, _, _ = evaluate(
            model, val_loader, criterion, device
        )
        scheduler_p2.step()

        elapsed = time.time() - t0
        print(f"  Epoch {epoch:2d}/{PHASE2_EPOCHS}  |  "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}  |  "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  |  "
              f"Tempo: {elapsed:.0f}s")

        history_p2["train_loss"].append(train_loss)
        history_p2["val_loss"].append(val_loss)
        history_p2["train_acc"].append(train_acc)
        history_p2["val_acc"].append(val_acc)

        # Salva il modello migliore (aggiorna se migliorato nella fase 2)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":       epoch,
                "phase":       2,
                "model_state": model.state_dict(),
                "val_loss":    val_loss,
                "val_acc":     val_acc,
            }, DL_BEST_MODEL_PATH)
            print(f"    Modello migliore aggiornato (val_loss={val_loss:.4f})")

    # Grafico fase 2
    plot_training_curves(
        history_p2, phase=2,
        save_path=os.path.join(EDA_DIR, "dl_training_curves_phase2.png")
    )
    full_history["train_loss"].extend(history_p2["train_loss"])
    full_history["val_loss"].extend(history_p2["val_loss"])
    full_history["train_acc"].extend(history_p2["train_acc"])
    full_history["val_acc"].extend(history_p2["val_acc"])

    # Grafico combinato di tutte le epoch
    plot_training_curves(
        full_history, phase="1+2",
        save_path=os.path.join(EDA_DIR, "dl_training_curves_full.png")
    )

    # ------------------------------------------------------------------
    # 5. Valutazione finale con il modello migliore
    # ------------------------------------------------------------------
    print("\n  Caricamento del modello migliore per la valutazione finale...")
    checkpoint = torch.load(DL_BEST_MODEL_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    print(f"  Checkpoint da: epoch={checkpoint['epoch']}, "
          f"phase={checkpoint['phase']}, "
          f"val_loss={checkpoint['val_loss']:.4f}")

    _, final_acc, y_true, y_pred, y_proba = evaluate(
        model, test_loader, criterion, device
    )
    final_auc = roc_auc_score(y_true, y_proba)

    print(f"\n  Accuracy finale: {final_acc:.4f}  |  AUC: {final_auc:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    # Grafici di valutazione
    plot_confusion_matrix(
        y_true, y_pred, CLASS_NAMES,
        title="Matrice di Confusione – ResNet-18",
        save_path=os.path.join(EDA_DIR, "dl_resnet18_confusion_matrix.png")
    )
    final_auc = plot_roc_curve(
        y_true, y_proba,
        model_name="ResNet-18",
        save_path=os.path.join(EDA_DIR, "dl_resnet18_roc_curve.png")
    )

    # Salva metriche DL in JSON
    dl_results = {
        "ResNet18": {
            "accuracy": round(float(final_acc), 4),
            "auc":      round(float(final_auc), 4),
            "best_val_loss": round(float(best_val_loss), 4),
            "best_epoch":    int(checkpoint["epoch"]),
            "best_phase":    int(checkpoint["phase"]),
            "phase1_epochs": PHASE1_EPOCHS,
            "phase2_epochs": PHASE2_EPOCHS,
            "classification_report": classification_report(
                y_true, y_pred, target_names=CLASS_NAMES, output_dict=True
            )
        }
    }
    with open(DL_RESULTS_JSON, "w", encoding="utf-8") as f:
        json.dump(dl_results, f, indent=2, ensure_ascii=False)
    print(f"\n  Metriche DL salvate: {DL_RESULTS_JSON}")

    # ------------------------------------------------------------------
    # Riepilogo
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  RIEPILOGO RISULTATI DL")
    print("=" * 60)
    print(f"  Modello   : ResNet-18 (fine-tuned)")
    print(f"  Accuracy  : {final_acc:.4f}")
    print(f"  AUC       : {final_auc:.4f}")
    print(f"  Modello migliore salvato: {DL_BEST_MODEL_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
