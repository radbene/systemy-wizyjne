"""Uruchomienie analizy i preprocessingu datasetu apples_tomatoes."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from dataset_analysis import analyze_dataset, print_report, save_report
from image_preprocessing import preprocess_image, preprocess_dataset


def create_comparison_grid(
    dataset_dir: Path,
    output_path: Path,
    samples_per_class: int = 3,
) -> None:
    """Generuje siatkę porównawczą: oryginał vs poszczególne etapy preprocessingu."""
    fig, axes = plt.subplots(6, 5, figsize=(15, 16))
    stages = ["original", "denoised", "enhanced", "final"]
    stage_titles = ["Oryginał", "Denoising", "Korekta kolorów", "Wynik końcowy"]

    row = 0
    for cls in ("apples", "tomatoes"):
        paths = sorted((dataset_dir / "train" / cls).glob("*.jpeg"))
        # wybierz zróżnicowane próbki: mały, średni, duży obraz
        paths_sorted = sorted(paths, key=lambda p: cv2.imread(str(p)).shape[0] * cv2.imread(str(p)).shape[1])
        indices = [
            0,
            len(paths_sorted) // 2,
            len(paths_sorted) - 1,
        ][:samples_per_class]

        for idx in indices:
            path = paths_sorted[idx]
            image = cv2.imread(str(path))
            result = preprocess_image(image)

            for col, stage in enumerate(stages):
                img = result[stage]
                axes[row, col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                axes[row, col].axis("off")
                if row == 0:
                    axes[row, col].set_title(stage_titles[col], fontsize=10)

            axes[row, 4].axis("off")
            axes[row, 4].text(
                0.1, 0.5,
                f"{cls}\n{path.name}\n{image.shape[1]}×{image.shape[0]}",
                va="center", fontsize=9,
            )
            row += 1

    plt.suptitle("Preprocessing datasetu apples_tomatoes", fontsize=14, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close()


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "apples_tomatoes"
    output_dir = root / "apples_tomatoes_preprocessed"
    report_dir = root / "preprocess" / "output"

    print(">>> Analiza datasetu...")
    stats = analyze_dataset(dataset_dir)
    print_report(stats)
    save_report(stats, report_dir / "dataset_stats.json")

    print("\n>>> Generowanie wizualizacji etapów preprocessingu...")
    create_comparison_grid(dataset_dir, report_dir / "preprocessing_comparison.png")

    print("\n>>> Preprocessing wszystkich obrazów...")
    count = preprocess_dataset(dataset_dir, output_dir)
    print(f"Przetworzono {count} obrazów → {output_dir}")

    print("\n>>> Gotowe.")


if __name__ == "__main__":
    main()
