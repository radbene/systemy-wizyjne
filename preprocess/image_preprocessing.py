"""Preprocessing obrazów: denoising, segmentacja, korekta kolorów."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


TARGET_SIZE = (224, 224)
WHITE_BG_THRESHOLD = 0.30
CROP_PADDING = 0.14
MIN_FOREGROUND_RATIO = 0.04
MAX_FOREGROUND_RATIO = 0.90
MIN_COMPONENT_RATIO = 0.008


def remove_artifacts(image: np.ndarray) -> np.ndarray:
    """Usuwa szum i artefakty kompresji JPEG."""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=6, hColor=6, templateWindowSize=7, searchWindowSize=21)
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharpened = cv2.filter2D(denoised, -1, kernel)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def has_white_background(image: np.ndarray) -> bool:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white_like = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 180)
    return float(np.mean(white_like)) > WHITE_BG_THRESHOLD


def border_is_colorful(image: np.ndarray) -> bool:
    """Obiekt wypełnia kadr — brak wyraźnego tła na brzegu."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = image.shape[:2]
    border = max(2, int(min(h, w) * 0.05))
    edge = np.concatenate([
        hsv[:border, :, 1].ravel(),
        hsv[-border:, :, 1].ravel(),
        hsv[:, :border, 1].ravel(),
        hsv[:, -border:, 1].ravel(),
    ])
    return float(np.mean(edge)) > 50


def morph_clean(mask: np.ndarray, close_iter: int = 2, open_iter: int = 1, dilate_iter: int = 1) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    if dilate_iter:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)
    return mask


def union_significant_components(mask: np.ndarray, min_area_ratio: float = MIN_COMPONENT_RATIO) -> np.ndarray:
    """Zachowuje wszystkie istotne obiekty (np. dwa pomidory), nie tylko największy."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    h, w = mask.shape
    min_area = h * w * min_area_ratio
    result = np.zeros((h, w), dtype=np.uint8)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= min_area:
            result[labels == label] = 255

    if cv2.countNonZero(result) == 0:
        return largest_component(mask)

    return morph_clean(result, close_iter=3, open_iter=1, dilate_iter=2)


def largest_component(mask: np.ndarray) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels <= 1:
        return mask

    best_label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == best_label, 255, 0).astype(np.uint8)


def fruit_color_mask(hsv: np.ndarray) -> np.ndarray:
    """Maska kolorów owoców: czerwień, zieleń, żółć, pomarańcz (Lab 1 — progowanie HSV)."""
    red1 = cv2.inRange(hsv, (0, 35, 35), (18, 255, 255))
    red2 = cv2.inRange(hsv, (160, 35, 35), (180, 255, 255))
    green = cv2.inRange(hsv, (22, 20, 25), (95, 255, 255))
    yellow = cv2.inRange(hsv, (8, 25, 35), (40, 255, 255))
    return red1 | red2 | green | yellow


def segment_white_background(image: np.ndarray) -> np.ndarray:
    """
    Segmentacja na białym tle: HSV (niskie S + wysokie V) + Otsu, łączona metoda OR.
    Lepsza dla jasnożółtych/jasnoróżowych owoców niż samo Otsu.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    not_white = ~((hsv[:, :, 1] < 38) & (hsv[:, :, 2] > 175))
    hsv_mask = not_white.astype(np.uint8) * 255

    gray = cv2.GaussianBlur(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, otsu_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    mask = cv2.bitwise_or(hsv_mask, otsu_mask)
    mask = morph_clean(mask, close_iter=2, open_iter=1, dilate_iter=1)
    return union_significant_components(mask)


def segment_watershed(image: np.ndarray) -> np.ndarray | None:
    """
    Segmentacja watershed (Lab 1/2) z markerami:
    - tło: jasne piksele brzegu
    - pierwszy plan: maska kolorów owoców
    """
    h, w = image.shape[:2]
    if border_is_colorful(image):
        return None

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    color_mask = fruit_color_mask(hsv)
    color_mask = morph_clean(color_mask, close_iter=3, open_iter=1, dilate_iter=0)

    if cv2.countNonZero(color_mask) < h * w * MIN_FOREGROUND_RATIO:
        return None

    sure_fg = cv2.erode(color_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)

    sure_bg = np.zeros((h, w), dtype=np.uint8)
    border = max(3, int(min(h, w) * 0.06))
    sure_bg[:border, :] = 255
    sure_bg[-border:, :] = 255
    sure_bg[:, :border] = 255
    sure_bg[:, -border:] = 255

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    sure_bg = cv2.bitwise_or(sure_bg, cv2.bitwise_and(thresh, cv2.bitwise_not(color_mask)))

    unknown = cv2.subtract(cv2.dilate(color_mask, None, iterations=2), sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0
    markers[sure_bg == 255] = 1

    img_ws = image.copy()
    cv2.watershed(img_ws, markers)

    fg = np.where((markers > 1), 255, 0).astype(np.uint8)
    fg = union_significant_components(fg, min_area_ratio=0.015)
    return fg if cv2.countNonZero(fg) > 0 else None


def crop_around_object(image: np.ndarray, mask: np.ndarray, padding: float = CROP_PADDING) -> np.ndarray:
    coords = cv2.findNonZero(mask)
    if coords is None:
        return image

    x, y, bw, bh = cv2.boundingRect(coords)
    pad_x = int(bw * padding)
    pad_y = int(bh * padding)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(image.shape[1], x + bw + pad_x)
    y2 = min(image.shape[0], y + bh + pad_y)
    return image[y1:y2, x1:x2]


def apply_segmentation(image: np.ndarray) -> tuple[np.ndarray, str]:
    h, w = image.shape[:2]

    if border_is_colorful(image):
        return image, "pełny kadr (obiekt na brzegu)"

    if has_white_background(image):
        mask = segment_white_background(image)
        method = "HSV + Otsu"
    else:
        mask = segment_watershed(image)
        method = "Watershed + HSV"
        if mask is None:
            mask = union_significant_components(morph_clean(fruit_color_mask(cv2.cvtColor(image, cv2.COLOR_BGR2HSV)), 3, 1, 2))
            method = "HSV kolory owoców"

    if mask is None or cv2.countNonZero(mask) == 0:
        return image, f"{method} → pełny kadr"

    fg_ratio = cv2.countNonZero(mask) / (h * w)
    if fg_ratio > MAX_FOREGROUND_RATIO or fg_ratio < MIN_FOREGROUND_RATIO:
        return image, f"{method} → pełny kadr"

    cropped = crop_around_object(image, mask)
    return cropped if cropped.size else image, method


def enhance_brightness_and_colors(image: np.ndarray) -> np.ndarray:
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
    denoised = remove_artifacts(image)
    segmented, method = apply_segmentation(denoised)
    enhanced = enhance_brightness_and_colors(segmented)
    final = resize_image(enhanced)

    return {
        "original": image,
        "denoised": denoised,
        "segmented": segmented,
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
                cv2.imwrite(str(dst_folder / src_path.name), result["final"])
                count += 1

    return count
