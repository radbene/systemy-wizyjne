"""Analiza datasetu apples_tomatoes."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class DatasetStats:
    total_images: int
    train_apples: int
    train_tomatoes: int
    test_apples: int
    test_tomatoes: int
    by_source: dict[str, int]
    width_range: tuple[int, int]
    height_range: tuple[int, int]
    unique_sizes: int
    small_images_lt_120px: int
    brightness_mean: float
    brightness_std: float
    brightness_min: float
    brightness_max: float
    class_brightness: dict[str, float]


def analyze_dataset(dataset_dir: Path) -> DatasetStats:
    sizes: list[tuple[int, int]] = []
    brightness: list[float] = []
    class_brightness: dict[str, list[float]] = defaultdict(list)
    by_source: Counter[str] = Counter()
    counts = {"train_apples": 0, "train_tomatoes": 0, "test_apples": 0, "test_tomatoes": 0}

    for split in ("train", "test"):
        for cls in ("apples", "tomatoes"):
            folder = dataset_dir / split / cls
            for path in sorted(folder.glob("*.jpeg")):
                img = cv2.imread(str(path))
                if img is None:
                    continue

                h, w = img.shape[:2]
                sizes.append((w, h))
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                mean_b = float(gray.mean())
                brightness.append(mean_b)
                class_brightness[cls].append(mean_b)

                prefix = path.stem.split("_")[1]
                by_source[f"{split}/{cls}/{prefix}"] += 1
                counts[f"{split}_{cls}"] += 1

    ws, hs = zip(*sizes)
    small = sum(1 for w, h in sizes if w < 120 or h < 120)

    return DatasetStats(
        total_images=len(sizes),
        train_apples=counts["train_apples"],
        train_tomatoes=counts["train_tomatoes"],
        test_apples=counts["test_apples"],
        test_tomatoes=counts["test_tomatoes"],
        by_source=dict(by_source),
        width_range=(min(ws), max(ws)),
        height_range=(min(hs), max(hs)),
        unique_sizes=len(set(sizes)),
        small_images_lt_120px=small,
        brightness_mean=float(np.mean(brightness)),
        brightness_std=float(np.std(brightness)),
        brightness_min=float(min(brightness)),
        brightness_max=float(max(brightness)),
        class_brightness={k: float(np.mean(v)) for k, v in class_brightness.items()},
    )


def print_report(stats: DatasetStats) -> None:
    print("=" * 60)
    print("ANALIZA DATASETU apples_tomatoes")
    print("=" * 60)
    print(f"Liczba obrazów łącznie:     {stats.total_images}")
    print(f"  train/apples:             {stats.train_apples}")
    print(f"  train/tomatoes:           {stats.train_tomatoes}")
    print(f"  test/apples:              {stats.test_apples}")
    print(f"  test/tomatoes:            {stats.test_tomatoes}")
    print()
    print("Rozkład wg źródła (p1/p2/p3):")
    for key, count in sorted(stats.by_source.items()):
        print(f"  {key}: {count}")
    print()
    print(f"Rozmiary (szerokość):       {stats.width_range[0]}–{stats.width_range[1]} px")
    print(f"Rozmiary (wysokość):        {stats.height_range[0]}–{stats.height_range[1]} px")
    print(f"Unikalne rozmiary:          {stats.unique_sizes}")
    print(f"Małe obrazy (<120 px):      {stats.small_images_lt_120px}")
    print()
    print("Jasność (średnia szarości):")
    print(f"  całość:  μ={stats.brightness_mean:.1f}, σ={stats.brightness_std:.1f}, "
          f"min={stats.brightness_min:.1f}, max={stats.brightness_max:.1f}")
    for cls, val in stats.class_brightness.items():
        print(f"  {cls}:   μ={val:.1f}")
    print()
    print("Obserwacje:")
    print("  - 3 źródła obrazów (p1, p2, p3) o różnej jakości i tle")
    print("  - p1: zdjęcia studyjne (białe tło) oraz naturalne (liście, etykiety)")
    print("  - p2: głównie produkty na białym tle")
    print("  - p3: opakowania, folia, etykiety sklepowe")
    print("  - duża zmienność rozmiaru i ekspozycji → wymagany preprocessing")
    print("=" * 60)


def save_report(stats: DatasetStats, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(asdict(stats), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    dataset_dir = root / "apples_tomatoes"
    stats = analyze_dataset(dataset_dir)
    print_report(stats)
    save_report(stats, root / "preprocess" / "output" / "dataset_stats.json")
