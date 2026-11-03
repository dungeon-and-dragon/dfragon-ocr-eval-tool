"""Place synthetic names in raid-window name crops the way dfragon cuts them."""

import math
import random
from dataclasses import dataclass

# Native raid-window layout from dnf-ocr-synth 0.2.0 calibration.layout and geometry,
# which copied dfragon's dnf-raid-participants.ts.
RAID_NAME_LEFT = 182
RAID_NAME_TOP = 21
RAID_NAME_HEIGHT = 17
# calibrated-v1 was measured from s=1 up and rejects smaller scales.
MIN_SCALE = 1.0
MAX_SCALE = 16.0
# Median text position over the 33 captures of the calibrated-v1 holdout anchors, in native px
# from the unrounded cell corner: the baseline 12.03 below the top and the ink 1.0 left of
# name_text_left. The s=1 capture is exactly 12 and -1.
NAME_BASELINE = 12
NAME_INK_LEFT_OFFSET = -1
ROUNDINGS = ("up", "down")
PHASE_ATTEMPTS = 8


@dataclass(frozen=True)
class CellPlacement:
    scale: float
    cell: tuple[float, float]
    crop: tuple[int, int, int, int]
    rounding: str


def scale_range(height: int) -> tuple[float, float] | None:
    """Return the scales whose name crop can be ``height`` px tall, or None.

    name_crop_geometry rounds both edges of the 17 native px cell, so a crop is
    floor(17s) or ceil(17s) tall depending on where the window sits. Both ends are open, except
    that s=1 always gives exactly 17px.
    """
    low = max(MIN_SCALE, (height - 1) / RAID_NAME_HEIGHT)
    high = min(MAX_SCALE, (height + 1) / RAID_NAME_HEIGHT)

    return (low, high) if low < high else None


def _top_phase(scale: float, height: int, rng: random.Random) -> float:
    span = RAID_NAME_HEIGHT * scale
    fraction = span - math.floor(span)
    if height == math.floor(span) + 1 and fraction > 0:
        return (0.5 - fraction + rng.random() * fraction) % 1

    if height == math.floor(span):
        return (0.5 + rng.random() * (1 - fraction)) % 1
    raise ValueError(f"{height}px name crop cannot come from scale {scale}.")


def place_cell(
    crop_geometry, scale: float, height: int | None, rng: random.Random
) -> CellPlacement:
    """Pick a sub-pixel window position for ``scale`` and cut its row 1 name crop.

    ``height`` fixes the crop height; None leaves it to a uniform phase.
    """
    for _ in range(PHASE_ATTEMPTS):
        left = rng.random()
        top = rng.random() if height is None else _top_phase(scale, height, rng)
        box = crop_geometry(scale, 1, (left - RAID_NAME_LEFT * scale, top - RAID_NAME_TOP * scale))
        # An exact half rounds to the even pixel, so redraw the rare phase that lands there.
        if height is None or box.height == height:
            rounding = rng.choice(ROUNDINGS)
            crop = (box.x, box.y, box.width, box.height)

            return CellPlacement(scale, (left, top), crop, rounding)
    raise ValueError(f"{height}px name crop was not reached at scale {scale}.")


def text_offset(
    placement: CellPlacement, box: tuple[int, int, int], padding: int, text_left
) -> tuple[int, int]:
    """Return where a rendered name image goes in the crop, to the nearest pixel.

    ``box`` is the native ink box (left, top, width) with ``top`` relative to the baseline, and
    ``padding`` the renderer's native padding around that ink.
    """
    scale = placement.scale
    _, top, width = box
    ink_left = placement.cell[0] + scale * (
        NAME_INK_LEFT_OFFSET + text_left(width, placement.rounding)
    )
    ink_top = placement.cell[1] + scale * (NAME_BASELINE + top)
    x = round(ink_left - scale * padding - placement.crop[0])
    y = round(ink_top - scale * padding - placement.crop[1])

    return x, y
