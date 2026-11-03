import json
from pathlib import Path

import numpy as np
import pytest

from training_worker.raid_binarization import (
    DETACHED_GAPS,
    RaidNicknameBinarize,
    binarize_raid_nickname,
)

# Synthetic arrays with dfragon's own binarizeRaidNicknamePixels outputs at each detached gap
# ratio (main and dfragon #1152); no screenshot pixels.
VECTORS = json.loads(
    (Path(__file__).parent / "fixtures/raid-binarization-vectors.json").read_text(encoding="utf-8")
)["vectors"]
CASES = [(vector, gap) for vector in VECTORS for gap in vector["outputs"]]


def test_vectors_cover_every_gap_and_tell_them_apart():
    assert {gap for _, gap in CASES} == set(DETACHED_GAPS)
    assert any(
        len({str(output) for output in vector["outputs"].values()}) > 1 for vector in VECTORS
    )


@pytest.mark.parametrize(
    ("vector", "gap"), CASES, ids=[f"{vector['name']}-{gap}" for vector, gap in CASES]
)
def test_training_binarization_matches_the_product_vectors(vector, gap):
    rgba = np.array(vector["rgba"], dtype=np.uint8).reshape(vector["height"], vector["width"], 4)
    expected = np.array(vector["outputs"][gap], dtype=np.uint8).reshape(rgba.shape)[..., :3]
    np.testing.assert_array_equal(binarize_raid_nickname(rgba, gap), expected)
    # PaddleOCR decodes BGR without alpha; the product reads RGBA.
    data = RaidNicknameBinarize(gap)(
        {"image": np.ascontiguousarray(rgba[..., 2::-1]), "label": "x"}
    )
    np.testing.assert_array_equal(data["image"], expected)
    assert data["image"].dtype == np.uint8


def test_default_gap_is_the_product_main_ratio():
    vector = next(vector for vector in VECTORS if vector["name"] == "raid-gap-between-ratios-30x17")
    rgba = np.array(vector["rgba"], dtype=np.uint8).reshape(vector["height"], vector["width"], 4)
    expected = np.array(vector["outputs"]["4/17"], dtype=np.uint8).reshape(rgba.shape)[..., :3]
    np.testing.assert_array_equal(binarize_raid_nickname(rgba), expected)
    with pytest.raises(ValueError):
        RaidNicknameBinarize("5/17")
