"""Feature extraction utilities for apples vs tomatoes classification."""

from __future__ import annotations

from hashlib import sha1
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops
from tqdm import tqdm


CLASS_TO_LABEL = {"apples": 0, "tomatoes": 1}
LABEL_TO_CLASS = {value: key for key, value in CLASS_TO_LABEL.items()}

GLCM_DISTANCES = (1, 2)
GLCM_ANGLES = (0.0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0)
GLCM_LEVELS = 32


def _extract_source_prefix(path: Path) -> str:
    parts = path.stem.split("_")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def load_dataset(split_dir: Path, split_name: str) -> pd.DataFrame:
    """Return dataframe with image paths, labels, class names and source prefix."""
    rows: list[dict[str, object]] = []
    for class_name, label in CLASS_TO_LABEL.items():
        class_dir = split_dir / class_name
        for image_path in sorted(class_dir.glob("*.jpeg")):
            rows.append(
                {
                    "path": str(image_path.resolve()),
                    "label": int(label),
                    "class_name": class_name,
                    "split": split_name,
                    "source": _extract_source_prefix(image_path),
                    "file_name": image_path.name,
                }
            )
    return pd.DataFrame(rows)


def load_train_test(preprocessed_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train and test metadata from preprocessed dataset."""
    train_df = load_dataset(preprocessed_root / "train", "train")
    test_df = load_dataset(preprocessed_root / "test", "test")
    return train_df, test_df


def _build_shape_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Create binary mask of fruit-like regions for shape descriptors."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

    non_white = ~((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180))

    red_1 = cv2.inRange(hsv, (0, 30, 30), (18, 255, 255))
    red_2 = cv2.inRange(hsv, (160, 30, 30), (180, 255, 255))
    green = cv2.inRange(hsv, (20, 20, 20), (95, 255, 255))
    yellow = cv2.inRange(hsv, (8, 20, 20), (45, 255, 255))

    color_mask = (red_1 | red_2 | green | yellow) > 0
    mask = np.where(non_white | color_mask, 255, 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    image_area = image_bgr.shape[0] * image_bgr.shape[1]
    min_area = max(16, int(image_area * 0.01))
    kept = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            kept[labels == label] = 255

    if cv2.countNonZero(kept) == 0:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        kept[labels == largest] = 255
    return kept


def extract_glcm_features(image_bgr: np.ndarray) -> dict[str, float]:
    """Extract 12 GLCM texture features (6 properties x 2 distances)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray_q = np.clip(gray // (256 // GLCM_LEVELS), 0, GLCM_LEVELS - 1).astype(np.uint8)
    glcm = graycomatrix(
        gray_q,
        distances=list(GLCM_DISTANCES),
        angles=list(GLCM_ANGLES),
        levels=GLCM_LEVELS,
        symmetric=True,
        normed=True,
    )

    feature_values: dict[str, float] = {}
    props = ("contrast", "dissimilarity", "homogeneity", "energy", "correlation", "ASM")
    for prop in props:
        values = graycoprops(glcm, prop)
        for idx, distance in enumerate(GLCM_DISTANCES):
            feature_values[f"glcm_{prop}_d{distance}"] = float(np.mean(values[idx, :]))
    return feature_values


def extract_shape_features(image_bgr: np.ndarray) -> dict[str, float]:
    """Extract shape descriptors based on contour from binary mask."""
    mask = _build_shape_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = mask.shape
    image_area = float(h * w)
    if not contours:
        return {
            "shape_area_ratio": 0.0,
            "shape_perimeter_norm": 0.0,
            "shape_compactness": 0.0,
            "shape_aspect_ratio": 1.0,
            "shape_extent": 0.0,
            "shape_solidity": 0.0,
            **{f"shape_hu_{idx}": 0.0 for idx in range(1, 8)},
        }

    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    x, y, bw, bh = cv2.boundingRect(contour)
    bbox_area = float(max(1, bw * bh))

    hull = cv2.convexHull(contour)
    hull_area = float(max(1.0, cv2.contourArea(hull)))

    compactness = float((4.0 * np.pi * area) / (perimeter * perimeter + 1e-9))
    aspect_ratio = float(bw / max(1, bh))
    extent = float(area / bbox_area)
    solidity = float(area / hull_area)

    moments = cv2.moments(contour)
    hu = cv2.HuMoments(moments).flatten()
    hu_log = -np.sign(hu) * np.log10(np.abs(hu) + 1e-12)

    features = {
        "shape_area_ratio": float(area / image_area),
        "shape_perimeter_norm": float(perimeter / (2.0 * (h + w))),
        "shape_compactness": compactness,
        "shape_aspect_ratio": aspect_ratio,
        "shape_extent": extent,
        "shape_solidity": solidity,
    }
    for idx, value in enumerate(hu_log, start=1):
        features[f"shape_hu_{idx}"] = float(value)
    return features


def extract_hsv_features(image_bgr: np.ndarray) -> dict[str, float]:
    """Extract HSV color statistics (mean and std per channel)."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    channels = ("h", "s", "v")
    features: dict[str, float] = {}
    for idx, name in enumerate(channels):
        values = hsv[:, :, idx]
        features[f"hsv_{name}_mean"] = float(np.mean(values))
        features[f"hsv_{name}_std"] = float(np.std(values))
    return features


def extract_feature_dict(image_bgr: np.ndarray) -> dict[str, float]:
    """Extract full feature dictionary for a single image."""
    features: dict[str, float] = {}
    features.update(extract_glcm_features(image_bgr))
    features.update(extract_shape_features(image_bgr))
    features.update(extract_hsv_features(image_bgr))
    return features


def _paths_cache_key(paths: Iterable[str]) -> str:
    digest = sha1()
    for raw_path in paths:
        path = Path(raw_path)
        stat = path.stat()
        digest.update(str(path).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
    return digest.hexdigest()


def extract_features_matrix(
    paths: list[str],
    cache_path: Path | None = None,
    use_cache: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract feature matrix from image paths.

    Optionally stores and loads a compressed cache (.npz) with shape:
    - X: ndarray [N, F]
    - feature_names: ndarray [F]
    - paths: ndarray [N]
    - cache_key: scalar string
    """
    if not paths:
        return np.empty((0, 0), dtype=np.float32), []

    cache_key = _paths_cache_key(paths)
    if use_cache and cache_path is not None and cache_path.exists():
        cached = np.load(cache_path, allow_pickle=True)
        cached_paths = cached["paths"].tolist()
        cached_key = str(cached["cache_key"].item())
        if cached_paths == paths and cached_key == cache_key:
            return cached["X"], cached["feature_names"].tolist()

    rows: list[dict[str, float]] = []
    for path in tqdm(paths, desc="Extracting features", leave=False):
        image = cv2.imread(path)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        rows.append(extract_feature_dict(image))

    feature_df = pd.DataFrame(rows)
    feature_names = feature_df.columns.tolist()
    X = feature_df.to_numpy(dtype=np.float32)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            X=X,
            feature_names=np.array(feature_names, dtype=object),
            paths=np.array(paths, dtype=object),
            cache_key=np.array(cache_key, dtype=object),
        )

    return X, feature_names

