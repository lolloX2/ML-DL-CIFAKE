"""
01_eda.py – Analisi Esplorativa dei Dati (EDA) per il dataset CIFAKE.

Analisi prodotte:
  1. Distribuzione delle classi (countplot)
  2. Istogrammi RGB medi per classe sovrapposti
  3. Griglia 4×4 esempi per classe (REAL e FAKE)
  4. Blur score (varianza del Laplaciano) boxplot per classe
  5. Analisi FFT: spettro di potenza medio per classe

Tutti i grafici vengono salvati in output/eda/
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import cv2   # per il Laplaciano (blur score)

# Importa la configurazione centralizzata
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
    EDA_DIR, IMAGE_SIZE_ML, RANDOM_SEED
)

# Riproducibilità
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Funzioni di supporto
# ---------------------------------------------------------------------------

def collect_image_paths(folder: str, max_samples: int = None) -> list[str]:
    """Raccoglie tutti i path di immagini in una cartella."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(folder, ext)))
    if max_samples and len(paths) > max_samples:
        paths = random.sample(paths, max_samples)
    return paths


def load_image_array(path: str, size: int = IMAGE_SIZE_ML) -> np.ndarray:
    """Carica un'immagine RGB ridimensionata come array NumPy uint8."""
    img = Image.open(path).convert("RGB")
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.uint8)


def compute_mean_rgb_histograms(paths: list[str], size: int = IMAGE_SIZE_ML,
                                n_bins: int = 64) -> tuple:
    """
    Calcola gli istogrammi medi dei canali R, G, B su un insieme di immagini.
    Ritorna (bin_edges, hist_r, hist_g, hist_b) mediati su tutti i campioni.
    """
    bins = np.linspace(0, 255, n_bins + 1)
    acc_r = np.zeros(n_bins)
    acc_g = np.zeros(n_bins)
    acc_b = np.zeros(n_bins)

    for path in tqdm(paths, desc="  Istogrammi RGB", leave=False):
        arr = load_image_array(path, size)
        acc_r += np.histogram(arr[:, :, 0], bins=bins)[0]
        acc_g += np.histogram(arr[:, :, 1], bins=bins)[0]
        acc_b += np.histogram(arr[:, :, 2], bins=bins)[0]

    n = len(paths)
    return bins, acc_r / n, acc_g / n, acc_b / n


def compute_blur_scores(paths: list[str], size: int = IMAGE_SIZE_ML) -> list[float]:
    """
    Calcola il blur score di ogni immagine come varianza del Laplaciano.
    Valori bassi → immagine sfocata; valori alti → immagine nitida.
    """
    scores = []
    for path in tqdm(paths, desc="  Blur scores", leave=False):
        arr = load_image_array(path, size)
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)
    return scores


def compute_mean_fft_spectrum(paths: list[str],
                              size: int = IMAGE_SIZE_ML) -> np.ndarray:
    """
    Calcola lo spettro di potenza FFT medio (in dB) su scala log
    per un insieme di immagini in scala di grigi.
    Ritorna una matrice (size × size) con lo spettro mediato.
    """
    acc = np.zeros((size, size), dtype=np.float64)

    for path in tqdm(paths, desc="  FFT spectrum", leave=False):
        arr = load_image_array(path, size)
        gray = np.mean(arr, axis=2)                      # media sui canali → scala di grigi
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        power = np.abs(fshift) ** 2
        acc += power

    mean_spectrum = acc / len(paths)
    # Conversione in dB con log10 (aggiungiamo 1 per evitare log(0))
    return 10 * np.log10(mean_spectrum + 1)


# ---------------------------------------------------------------------------
# Script principale
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  01_EDA – Analisi Esplorativa CIFAKE")
    print("=" * 60)

    # Raccogli i path (limitati a 5000 per classe per la velocità dell'EDA)
    MAX_EDA = 5_000
    print(f"\n[1/5] Raccolta path immagini (max {MAX_EDA} per classe)...")
    real_paths = collect_image_paths(TRAIN_REAL_DIR, max_samples=MAX_EDA)
    fake_paths = collect_image_paths(TRAIN_FAKE_DIR, max_samples=MAX_EDA)
    print(f"  REAL: {len(real_paths):,}  |  FAKE: {len(fake_paths):,}")

    # ------------------------------------------------------------------
    # 1. Distribuzione delle classi
    # ------------------------------------------------------------------
    print("\n[1/5] Distribuzione delle classi...")
    # Conta tutte le immagini (non solo il subsample)
    all_real = collect_image_paths(TRAIN_REAL_DIR)
    all_fake = collect_image_paths(TRAIN_FAKE_DIR)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(
        x=["REAL", "FAKE"],
        y=[len(all_real), len(all_fake)],
        palette=["#2196F3", "#F44336"],
        ax=ax
    )
    for bar, count in zip(ax.patches, [len(all_real), len(all_fake)]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 200,
            f"{count:,}", ha="center", va="bottom", fontsize=12, fontweight="bold"
        )
    ax.set_title("Distribuzione delle classi nel training set",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Numero di immagini", fontsize=12)
    ax.set_xlabel("Classe", fontsize=12)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "01_class_distribution.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}")

    # ------------------------------------------------------------------
    # 2. Istogrammi RGB medi sovrapposti per classe
    # ------------------------------------------------------------------
    print("\n[2/5] Istogrammi RGB medi per classe...")
    bins_r, hr_r, hg_r, hb_r = compute_mean_rgb_histograms(real_paths)
    bins_f, hr_f, hg_f, hb_f = compute_mean_rgb_histograms(fake_paths)

    bin_centers = (bins_r[:-1] + bins_r[1:]) / 2

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    channel_data = [
        ("Rosso (R)",  [hr_r, hr_f], ["#EF9A9A", "#B71C1C"]),
        ("Verde (G)",  [hg_r, hg_f], ["#A5D6A7", "#1B5E20"]),
        ("Blu (B)",    [hb_r, hb_f], ["#90CAF9", "#0D47A1"]),
    ]
    for ax, (channel_name, hists, colors) in zip(axes, channel_data):
        for hist, color, label in zip(hists, colors, ["REAL", "FAKE"]):
            ax.plot(bin_centers, hist, color=color, lw=1.5, label=label)
            ax.fill_between(bin_centers, hist, alpha=0.2, color=color)
        ax.set_title(f"Canale {channel_name}", fontsize=12)
        ax.set_xlabel("Valore pixel", fontsize=10)
        ax.set_ylabel("Conteggio medio", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Istogrammi RGB medi per classe (REAL vs FAKE)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "02_rgb_histograms.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}")

    # ------------------------------------------------------------------
    # 3. Griglia 4×4 esempi per classe
    # ------------------------------------------------------------------
    print("\n[3/5] Griglia esempi per classe...")
    n_grid = 4  # griglia 4×4

    fig, axes = plt.subplots(2, n_grid * n_grid,
                             figsize=(n_grid * n_grid * 2, 5))

    for col_idx, (paths, label) in enumerate(
        [(real_paths, "REAL"), (fake_paths, "FAKE")]
    ):
        sampled = random.sample(paths, n_grid * n_grid)
        for i, path in enumerate(sampled):
            ax = axes[col_idx, i]
            arr = load_image_array(path)
            ax.imshow(arr)
            ax.axis("off")
        axes[col_idx, 0].set_ylabel(label, fontsize=14, fontweight="bold",
                                    rotation=0, labelpad=50, va="center")

    fig.suptitle("Esempi di immagini: REAL (riga 1) e FAKE (riga 2)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "03_image_grid.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}")

    # ------------------------------------------------------------------
    # 4. Blur score boxplot per classe
    # ------------------------------------------------------------------
    print("\n[4/5] Blur score (varianza Laplaciano)...")
    real_blur = compute_blur_scores(real_paths)
    fake_blur = compute_blur_scores(fake_paths)

    fig, ax = plt.subplots(figsize=(8, 6))
    data_plot = [real_blur, fake_blur]
    bp = ax.boxplot(
        data_plot,
        labels=["REAL", "FAKE"],
        patch_artist=True,
        notch=True,
        medianprops={"color": "black", "linewidth": 2}
    )
    # Colora i box
    colors = ["#2196F3", "#F44336"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Aggiungi violinplot sovrapposto per mostrare la distribuzione
    parts = ax.violinplot(data_plot, positions=[1, 2], showmedians=False,
                          showextrema=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.3)

    ax.set_title("Blur Score (varianza del Laplaciano) per classe",
                 fontsize=14, fontweight="bold")
    ax.set_ylabel("Varianza Laplaciano (nitidezza)", fontsize=12)
    ax.set_xlabel("Classe", fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Statistiche nel grafico
    for i, (scores, label) in enumerate([(real_blur, "REAL"), (fake_blur, "FAKE")], 1):
        ax.text(i, ax.get_ylim()[1] * 0.95,
                f"media={np.mean(scores):.1f}\nmediana={np.median(scores):.1f}",
                ha="center", fontsize=9, color="black",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))

    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "04_blur_score_boxplot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}")

    # ------------------------------------------------------------------
    # 5. Analisi FFT: spettro di potenza medio per classe
    # ------------------------------------------------------------------
    print("\n[5/5] Analisi FFT (spettro di potenza medio)...")
    # Usa un sottoinsieme più piccolo per la FFT (computazionalmente pesante)
    MAX_FFT = 1_000
    real_fft_paths = random.sample(real_paths, min(MAX_FFT, len(real_paths)))
    fake_fft_paths = random.sample(fake_paths, min(MAX_FFT, len(fake_paths)))

    real_spectrum = compute_mean_fft_spectrum(real_fft_paths)
    fake_spectrum = compute_mean_fft_spectrum(fake_fft_paths)
    diff_spectrum = real_spectrum - fake_spectrum

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    titles = ["Spettro medio REAL", "Spettro medio FAKE",
              "Differenza (REAL − FAKE)"]
    spectra = [real_spectrum, fake_spectrum, diff_spectrum]
    cmaps   = ["viridis", "viridis", "RdBu_r"]

    for ax, spectrum, title, cmap in zip(axes, spectra, titles, cmaps):
        im = ax.imshow(spectrum, cmap=cmap, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.8, label="Potenza (dB)")
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Frequenza X", fontsize=10)
        ax.set_ylabel("Frequenza Y", fontsize=10)

    fig.suptitle("Analisi FFT – Spettro di potenza medio per classe",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(EDA_DIR, "05_fft_spectrum.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Salvato: {save_path}")

    # ------------------------------------------------------------------
    # Riepilogo finale
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  EDA completata! Grafici salvati in:", EDA_DIR)
    print("  File prodotti:")
    for fname in sorted(os.listdir(EDA_DIR)):
        print(f"    - {fname}")
    print("=" * 60)


if __name__ == "__main__":
    main()
