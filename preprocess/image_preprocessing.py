"""Preprocessing obrazów: denoising, korekta kolorów, normalizacja rozmiaru."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = (224, 224)


def remove_artifacts(image: np.ndarray) -> np.ndarray:
    """Usuwa szum i artefakty kompresji JPEG."""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def enhance_brightness_and_colors(image: np.ndarray) -> np.ndarray:
    """Korekta jasności (CLAHE) i nasycenia kolorów."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)

    enhanced = cv2.merge([l, a, b])
    enhanced = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def preprocess_image(image: np.ndarray) -> dict[str, np.ndarray]:
    """Pipeline: denoising → korekta kolorów → resize."""
    denoised = remove_artifacts(image)
    enhanced = enhance_brightness_and_colors(denoised)
    final = resize_image(enhanced)

    return {
        "original": image,
        "denoised": denoised,
        "enhanced": enhanced,
        "final": final,
    }


def preprocess_dataset(
    input_dir: Path,
    output_dir: Path,
    splits: tuple[str, ...] = ("train", "test"),
    classes: tuple[str, ...] = ("apples", "tomatoes"),
) -> int:
    """Przetwarza cały dataset, zachowując strukturę katalogów."""
    count = 0
    for split in splits:
        for cls in classes:
            src_folder = input_dir / split / cls
            dst_folder = output_dir / split / cls
            dst_folder.mkdir(parents=True, exist_ok=True)

            for src_path in sorted(src_folder.glob("*.jpeg")):
                image = cv2.imread(str(src_path))
                if image is None:
                    continue

                result = preprocess_image(image)
                dst_path = dst_folder / src_path.name
                cv2.imwrite(str(dst_path), result["final"])
                count += 1

    return count
