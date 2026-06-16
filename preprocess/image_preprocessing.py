"""Preprocessing obrazów: denoising, segmentacja (Lab 1 + Lab 3), korekta kolorów."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = (224, 224)
BACKGROUND_COLOR = (240, 240, 240)
WHITE_BG_THRESHOLD = 0.35
MIN_FOREGROUND_RATIO = 0.05
KMEANS_K = 4


def remove_artifacts(image: np.ndarray) -> np.ndarray:
    """Usuwa szum i artefakty kompresji JPEG (filtr mediany + denoising)."""
    median = cv2.medianBlur(image, 3)
    denoised = cv2.fastNlMeansDenoisingColored(median, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
    return denoised


def has_white_background(image: np.ndarray) -> bool:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray > 220)) > WHITE_BG_THRESHOLD


def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.02) -> np.ndarray:
    """Operacje morfologiczne (Lab 1) + wybór największego obiektu (connected components)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    min_area = h * w * min_area_ratio
    best_label = 0
    best_area = 0
    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]
        if area >= min_area and area > best_area:
            best_area = area
            best_label = label

    if best_label == 0:
        return mask

    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def segment_otsu_morphology(image: np.ndarray) -> np.ndarray:
    """
    Segmentacja metodami z Lab 1:
    - binaryzacja Otsu na kanale jasności
    - operacje morfologiczne (closing, opening)
    - indeksacja obiektów (connected components) — wybór największego
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return clean_mask(mask)


def segment_kmeans(image: np.ndarray) -> np.ndarray:
    """
    Segmentacja metodą KMeans w przestrzeni LAB (Lab 3).
    Tło identyfikowane na podstawie pikseli z obrzeża obrazu,
    pierwszy plan — klaster o najwyższym nasyceniu koloru.
    """
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    sample_step = max(1, int(np.sqrt(h * w) / 100))
    pixels = lab[::sample_step, ::sample_step].reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, _, centers = cv2.kmeans(
        pixels, KMEANS_K, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )

    border = max(2, int(min(h, w) * 0.05))
    border_pixels = np.vstack([
        lab[:border, :].reshape(-1, 3),
        lab[-border:, :].reshape(-1, 3),
        lab[:, :border].reshape(-1, 3),
        lab[:, -border:].reshape(-1, 3),
    ])
    border_mean = border_pixels.mean(axis=0)

    bg_cluster = int(np.argmin(np.linalg.norm(centers - border_mean, axis=1)))

    all_pixels = lab.reshape(-1, 3).astype(np.float32)
    full_labels = np.argmin(
        np.linalg.norm(all_pixels[:, None, :] - centers[None, :, :], axis=2),
        axis=1,
    ).reshape(h, w)

    best_cluster = bg_cluster
    best_saturation = -1.0
    for idx in range(KMEANS_K):
        if idx == bg_cluster:
            continue
        cluster_mask = full_labels == idx
        if not np.any(cluster_mask):
            continue
        mean_sat = float(hsv[:, :, 1][cluster_mask].mean())
        if mean_sat > best_saturation:
            best_saturation = mean_sat
            best_cluster = idx

    mask = np.where(full_labels == best_cluster, 255, 0).astype(np.uint8)
    return clean_mask(mask)


def segment_object(image: np.ndarray, fruit_class: str) -> tuple[np.ndarray, str]:
    """Wybiera metodę segmentacji w zależności od typu tła."""
    if has_white_background(image):
        return segment_otsu_morphology(image), "Otsu + morfologia"
    return segment_kmeans(image), "KMeans (LAB)"


def apply_segmentation(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Nakłada maskę — obiekt na neutralnym tle."""
    background = np.full_like(image, BACKGROUND_COLOR)
    fg = mask > 0
    result = background.copy()
    result[fg] = image[fg]
    return result


def crop_to_object(image: np.ndarray, mask: np.ndarray, padding: float = 0.06) -> tuple[np.ndarray, np.ndarray]:
    """Przycina obraz do bounding box największego obiektu."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image, mask

    x, y, w, h = cv2.boundingRect(coords)
    pad_x = int(w * padding)
    pad_y = int(h * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + w + pad_x)
    y2 = min(image.shape[0], y + h + pad_y)

    return image[y1:y2, x1:x2], mask[y1:y2, x1:x2]


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


def preprocess_image(image: np.ndarray, fruit_class: str = "apples") -> dict[str, np.ndarray]:
    """Pipeline: denoising → segmentacja → korekta kolorów → resize."""
    denoised = remove_artifacts(image)
    mask, method = segment_object(denoised, fruit_class)

    if cv2.countNonZero(mask) < image.shape[0] * image.shape[1] * MIN_FOREGROUND_RATIO:
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        method = "fallback (brak maski)"

    segmented = apply_segmentation(denoised, mask)
    cropped, _ = crop_to_object(segmented, mask)
    enhanced = enhance_brightness_and_colors(cropped)
    final = resize_image(enhanced)

    return {
        "original": image,
        "denoised": denoised,
        "mask": mask,
        "segmented": cropped,
        "enhanced": enhanced,
        "final": final,
        "segmentation_method": method,
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

                result = preprocess_image(image, cls)
                dst_path = dst_folder / src_path.name
                cv2.imwrite(str(dst_path), result["final"])
                count += 1

    return count
