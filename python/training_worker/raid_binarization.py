"""Binarize raid name crops exactly as dfragon does before its OCR model reads them.

Ported from dfragon ``packages/client-core/src/lib/nickname-pixels.ts``
(``binarizeRaidNicknamePixels``) through the dnf-ocr-synth 0.2.0 calibration port, and checked
against the same fixed vectors. Training must see what the product feeds the model, so a product
change to that function has to be ported here too.
"""

import math
from dataclasses import dataclass

import numpy as np

MIN_CLASS_PROBABILITY = 2.0**-23
BACKGROUND = 255
FRAME_EDGE_MARGIN_RATIO = 1 / 5
FRAME_LINE_WIDTH_RATIO = 3 / 4
# dfragon main splits a raid name into clusters at a gap of round(height * 4 / 17) empty columns;
# dfragon #1152 widens it to 6/17 so a digit 1's trailing gap no longer cuts a name in two.
DETACHED_GAPS = {"4/17": 4 / 17, "6/17": 6 / 17}
DEFAULT_DETACHED_GAP = "4/17"


def _otsu_threshold(histogram: list[int], count: int, total: int) -> int:
    # Keep findOtsuThreshold's float operation order so ties resolve like the product.
    scale = 1 / count
    mean = total * scale
    dark_probability = 0.0
    dark_mean = 0.0
    max_variance = 0.0
    threshold = 0
    for gray, bin_count in enumerate(histogram):
        probability = bin_count * scale
        dark_mean *= dark_probability
        dark_probability += probability
        light_probability = 1 - dark_probability
        if (
            min(dark_probability, light_probability) < MIN_CLASS_PROBABILITY
            or max(dark_probability, light_probability) > 1 - MIN_CLASS_PROBABILITY
        ):
            continue
        dark_mean = (dark_mean + gray * probability) / dark_probability
        light_mean = (mean - dark_probability * dark_mean) / light_probability
        difference = dark_mean - light_mean
        variance = dark_probability * light_probability * difference * difference
        if variance > max_variance:
            max_variance = variance
            threshold = gray

    return threshold


def _binarize_darkest_channel(pixels: np.ndarray) -> np.ndarray:
    darkest = pixels[..., :3].min(axis=-1)
    if darkest.size == 0:
        return np.zeros(darkest.shape, dtype=bool)
    histogram = np.bincount(darkest.reshape(-1), minlength=256).tolist()
    threshold = _otsu_threshold(histogram, darkest.size, int(darkest.sum(dtype=np.int64)))

    return darkest > threshold


def _remove_frame_lines(ink: np.ndarray) -> np.ndarray:
    height, width = ink.shape
    margin = math.floor(height * FRAME_EDGE_MARGIN_RATIO)
    min_line_width = math.ceil(width * FRAME_LINE_WIDTH_RATIO)
    result = ink.copy()
    for y in (*range(margin), *range(height - margin, height)):
        start = 0
        for x in range(width + 1):
            if x < width and ink[y, x]:
                continue
            if x - start >= min_line_width:
                result[y, start:x] = False
            start = x + 1

    return result


@dataclass
class _ColumnCluster:
    left: int
    right: int
    count: int


def _remove_detached_pixels(ink: np.ndarray, gap_ratio: float) -> np.ndarray:
    height, width = ink.shape
    # JS Math.round rounds halves up, unlike Python's round().
    min_gap = max(1, math.floor(height * gap_ratio + 0.5))
    counts = ink.sum(axis=0)
    clusters: list[_ColumnCluster] = []
    for x in range(width):
        count = int(counts[x])
        if count == 0:
            continue

        if clusters and x - clusters[-1].right - 1 < min_gap:
            clusters[-1].right = x
            clusters[-1].count += count
        else:
            clusters.append(_ColumnCluster(x, x, count))
    result = ink.copy()
    if clusters:
        main = max(clusters, key=lambda cluster: cluster.count)
        for cluster in clusters:
            if cluster is not main:
                result[:, cluster.left : cluster.right + 1] = False

    return result


def binarize_raid_nickname(
    pixels: np.ndarray, detached_gap: str = DEFAULT_DETACHED_GAP
) -> np.ndarray:
    """Return black ink on white as (H, W, 3) uint8 from (H, W, 3+) uint8 pixels.

    Channel order does not matter: every step reads the darkest channel or the binary map.
    ``detached_gap`` names a key of ``DETACHED_GAPS``.
    """
    source = np.asarray(pixels)
    if source.dtype != np.uint8 or source.ndim != 3 or source.shape[2] < 3:
        raise ValueError("raid crop must be an (H, W, 3+) uint8 array.")
    ink = _remove_detached_pixels(
        _remove_frame_lines(_binarize_darkest_channel(source)), DETACHED_GAPS[detached_gap]
    )
    gray = np.where(ink, 0, BACKGROUND).astype(np.uint8)

    return np.repeat(gray[..., None], 3, axis=2)


class RaidNicknameBinarize:
    """PaddleOCR transform operator placed right after DecodeImage."""

    def __init__(self, detached_gap: str = DEFAULT_DETACHED_GAP) -> None:
        if detached_gap not in DETACHED_GAPS:
            raise ValueError(f"detached gap must be one of {tuple(DETACHED_GAPS)}.")
        self.detached_gap = detached_gap

    def __call__(self, data: dict) -> dict:
        data["image"] = binarize_raid_nickname(data["image"], self.detached_gap)

        return data
