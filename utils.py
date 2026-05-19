"""
utils.py – Funzioni condivise tra i vari script del progetto CIFAKE.

Contiene:
  - load_images_flat()      : carica immagini come array NumPy appiattiti (per ML)
  - get_dataloaders()       : restituisce DataLoader PyTorch (per DL)
  - plot_confusion_matrix() : visualizza e salva la matrice di confusione
  - plot_roc_curve()        : visualizza e salva la curva ROC
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")   # backend non-interattivo per salvataggio file
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, roc_curve, auc

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import train_test_split as _sklearn_tts


# ---------------------------------------------------------------------------
# Funzioni per ML classico
# ---------------------------------------------------------------------------

def load_images_flat(real_dir: str, fake_dir: str,
                     image_size: int = 64,
                     max_per_class: int = None,
                     seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Carica immagini da due cartelle (REAL e FAKE), le ridimensiona a
    (image_size × image_size), normalizza i valori in [0, 1] e
    appiattisce ogni immagine in un vettore 1-D.

    Parametri
    ----------
    real_dir       : path cartella immagini reali
    fake_dir       : path cartella immagini false
    image_size     : lato del resize (immagini quadrate)
    max_per_class  : numero massimo di campioni per classe (subsample)
    seed           : seme per la riproducibilità del subsample

    Ritorna
    -------
    X : array (N, image_size*image_size*3) di float32
    y : array (N,) con etichette 0=REAL, 1=FAKE
    """
    rng = np.random.default_rng(seed)

    def _load_folder(folder: str, label: int, max_samples: int = None):
        # Raccogli tutti i path di immagini supportati
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        paths = []
        for ext in extensions:
            paths.extend(glob.glob(os.path.join(folder, ext)))

        # Subsample se richiesto
        if max_samples is not None and len(paths) > max_samples:
            indices = rng.choice(len(paths), size=max_samples, replace=False)
            paths = [paths[i] for i in indices]

        vectors, labels = [], []
        for path in tqdm(paths, desc=f"Carico {'REAL' if label==0 else 'FAKE'} da {os.path.basename(folder)}"):
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize((image_size, image_size), Image.LANCZOS)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                vectors.append(arr.flatten())
                labels.append(label)
            except Exception as e:
                print(f"  [SKIP] {path}: {e}")

        return vectors, labels

    real_X, real_y = _load_folder(real_dir, label=0, max_samples=max_per_class)
    fake_X, fake_y = _load_folder(fake_dir, label=1, max_samples=max_per_class)

    X = np.array(real_X + fake_X, dtype=np.float32)
    y = np.array(real_y + fake_y, dtype=np.int64)

    # Mescola i campioni
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ---------------------------------------------------------------------------
# Dataset PyTorch e DataLoader per DL
# ---------------------------------------------------------------------------

class CIFAKEDataset(Dataset):
    """Dataset personalizzato per CIFAKE che legge immagini da cartelle."""

    def __init__(self, real_dir: str, fake_dir: str, transform=None):
        self.samples = []    # lista di tuple (path, label)
        self.transform = transform

        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

        for ext in extensions:
            for path in glob.glob(os.path.join(real_dir, ext)):
                self.samples.append((path, 0))   # 0 = REAL
            for path in glob.glob(os.path.join(fake_dir, ext)):
                self.samples.append((path, 1))   # 1 = FAKE

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    """
    Wrapper su un Subset di CIFAKEDataset che applica un transform specifico.

    Il dataset padre viene creato senza transform (transform=None) in modo
    che le immagini arrivino come PIL.Image grezze; questo wrapper le
    trasforma al momento del __getitem__, permettendo di assegnare transform
    diversi (es. augmentation vs. solo resize) a train e val sullo stesso
    insieme di file.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # img è PIL.Image (nessun transform sul padre)
        if self.transform:
            img = self.transform(img)
        return img, label


def get_dataloaders(train_real_dir: str, train_fake_dir: str,
                    test_real_dir: str,  test_fake_dir: str,
                    image_size: int = 224,
                    batch_size: int = 64,
                    num_workers: int = 4,
                    seed: int = 42,
                    val_split: float = 0.0) -> tuple:
    """
    Crea i DataLoader per train, (val opzionale) e test.

    Augmentation sul train: flip orizzontale, ColorJitter, RandomResizedCrop.
    Val / Test: solo resize + CenterCrop + normalizzazione ImageNet.

    Parametri
    ----------
    train_real_dir / train_fake_dir : cartelle training
    test_real_dir  / test_fake_dir  : cartelle test
    image_size     : dimensione finale delle immagini (default 224 per ResNet)
    batch_size     : dimensione del batch
    num_workers    : processi paralleli per il DataLoader
    seed           : seme per la riproducibilità
    val_split      : frazione del train da usare come validation (0 = disabilitato)

    Ritorna
    -------
    Se val_split > 0: (train_loader, val_loader, test_loader)
    Se val_split = 0: (train_loader, None, test_loader)
    """
    # Media e deviazione standard di ImageNet per la normalizzazione
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                               saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    test_dataset = CIFAKEDataset(test_real_dir, test_fake_dir, transform=eval_transform)

    # Generatore per riproducibilità
    generator = torch.Generator()
    generator.manual_seed(seed)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    if val_split > 0.0:
        # Crea il dataset senza transform: le immagini restano PIL.Image grezze
        # così TransformSubset può applicare transform diversi a train e val
        raw_dataset = CIFAKEDataset(train_real_dir, train_fake_dir, transform=None)
        labels = [raw_dataset.samples[i][1] for i in range(len(raw_dataset))]

        train_idx, val_idx = _sklearn_tts(
            range(len(raw_dataset)),
            test_size=val_split,
            stratify=labels,
            random_state=seed
        )

        train_subset = TransformSubset(Subset(raw_dataset, train_idx), train_transform)
        val_subset   = TransformSubset(Subset(raw_dataset, val_idx),   eval_transform)

        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            generator=generator, drop_last=True
        )
        val_loader = DataLoader(
            val_subset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )

        print(f"  Train: {len(train_subset):,}  |  "
              f"Val: {len(val_subset):,}  |  "
              f"Test: {len(test_dataset):,}")
        print(f"  Batch size: {batch_size}  |  "
              f"Train batches: {len(train_loader)}  |  "
              f"Val batches: {len(val_loader)}  |  "
              f"Test batches: {len(test_loader)}")

        return train_loader, val_loader, test_loader

    else:
        train_dataset = CIFAKEDataset(train_real_dir, train_fake_dir, transform=train_transform)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True,
            generator=generator, drop_last=True
        )

        print(f"  Train samples: {len(train_dataset):,}  |  "
              f"Test samples: {len(test_dataset):,}")
        print(f"  Batch size: {batch_size}  |  "
              f"Train batches: {len(train_loader)}  |  "
              f"Test batches: {len(test_loader)}")

        return train_loader, None, test_loader


# ---------------------------------------------------------------------------
# Visualizzazioni condivise
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          class_names: list[str],
                          title: str, save_path: str) -> None:
    """
    Genera e salva una heatmap della matrice di confusione normalizzata.

    Parametri
    ----------
    y_true      : etichette vere
    y_pred      : etichette predette
    class_names : lista dei nomi delle classi
    title       : titolo del grafico
    save_path   : percorso dove salvare l'immagine
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)   # normalizzata per riga

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ["d", ".2%"],
        ["Conteggi assoluti", "Normalizzata per riga"]
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8}
        )
        ax.set_xlabel("Predetto", fontsize=12)
        ax.set_ylabel("Reale",    fontsize=12)
        ax.set_title(subtitle,    fontsize=12)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Matrice di confusione salvata: {save_path}")


def plot_roc_curve(y_true: np.ndarray, y_score: np.ndarray,
                   model_name: str, save_path: str) -> float:
    """
    Genera e salva la curva ROC con l'area AUC.

    Parametri
    ----------
    y_true      : etichette binarie vere (0/1)
    y_score     : probabilità della classe positiva (1)
    model_name  : nome del modello (usato nel titolo)
    save_path   : percorso dove salvare l'immagine

    Ritorna
    -------
    roc_auc : valore AUC (float)
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color="steelblue", lw=2,
            label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--",
            label="Classificatore casuale")
    ax.fill_between(fpr, tpr, alpha=0.1, color="steelblue")

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate",  fontsize=12)
    ax.set_title(f"Curva ROC – {model_name}", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Curva ROC salvata: {save_path} (AUC={roc_auc:.4f})")
    return roc_auc
