import copy
import json
import random
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from test_training_snapshot import save_dataset
from test_training_snapshot import snapshot as snapshot

from training_worker.dictionary import expand_state_arrays, extend_characters, write_run_dictionary
from training_worker.snapshot import digest, load_snapshot
from training_worker.supplement import (
    Synthesizer,
    background_files,
    composite,
    nearest_backgrounds,
    prepare_supplement,
    scale_heights,
)
from training_worker.supplement_geometry import place_cell, scale_range
from training_worker.supplement_plan import GROUPS, character_group, distribution, plan_supplement
from training_worker.supplement_snapshot import load_supplement


def valid(text):
    try:
        good = 0 < len(text.encode("cp949")) <= 12 and not any(c.isspace() for c in text)
    except UnicodeEncodeError:
        good = False
    return SimpleNamespace(is_valid=good, reason="invalid test nickname")


def options(**overrides):
    value = {
        "pythonExecutable": "/fixture/python",
        "additionalCharacters": "ABCD",
        "targets": dict.fromkeys(GROUPS, None),
        "characters": dict.fromkeys(GROUPS, ""),
        "tolerance": 0,
        "maxImages": 1000,
        "seed": 42,
        "profile": "nanum-neo",
        "fonts": dict.fromkeys(("gulim", "batang", "nanum", "uttum"), ""),
        "scaleModel": "affine-bilinear",
        "scaleSampling": "fixed",
        "placement": "center",
        "scale": 1,
        "width": 40,
        "height": 20,
        "padding": 1,
        "color": [75, 209, 255],
        "backgroundMode": "solid",
        "backgroundColor": [20, 40, 60],
        "backgroundDirectory": "",
    }
    value.update(overrides)
    return value


def test_character_counts_and_balanced_unique_names_use_only_selected_shortages():
    assert [character_group(char) for char in "ⓐ⒜Ⅰⅰ"] == ["special", "special", "latin", "latin"]
    assert distribution(["가", "가"])["nicknames"] == 1
    assert distribution(["검사★龍", "あアA2Ω"])["groups"] == dict(
        zip(GROUPS, (2, 1, 1, 1, 1, 1, 1, 1))
    )
    rows = [{"text": "가나다라마바", "split": "train"} for _ in range(100)] + [
        {"text": "★☆", "split": "test"}
    ]
    config = options()
    config["targets"]["special"] = 20
    config["characters"]["special"] = "★☆♥♡◆◇○●◎"
    before = copy.deepcopy(rows)
    plan = plan_supplement(rows, config, valid)
    assert plan == plan_supplement(rows, config, valid)
    assert rows == before
    assert plan["real"]["images"] == 100
    assert abs(plan["differences"]["special"]) < 1e-8
    assert len(set(plan["nicknames"])) == len(plan["nicknames"])
    assert "★☆" not in plan["nicknames"]
    assert all(len(set(name)) == len(name) and valid(name).is_valid for name in plan["nicknames"])
    frequencies = [item["count"] for item in plan["synthetic"]["frequencies"]]
    assert max(frequencies) - min(frequencies) <= 1
    assert plan["synthetic"]["groups"]["hangul"] == 0


def test_real_growth_can_reduce_or_increase_supplement_and_tolerance_is_user_controlled():
    config = options()
    config["targets"]["latin"] = 50
    config["characters"]["latin"] = "ABCDEFGH"
    baseline = [{"text": "가나", "split": "train"} for _ in range(10)]
    need = plan_supplement(baseline, config, valid)["synthetic"]["characters"]
    less = plan_supplement(baseline + [{"text": "ABCDEF", "split": "train"}], config, valid)[
        "synthetic"
    ]["characters"]
    more = plan_supplement(baseline + [{"text": "다라마", "split": "train"}], config, valid)[
        "synthetic"
    ]["characters"]
    assert less < need < more
    config["tolerance"] = 10
    tolerant = plan_supplement(baseline, config, valid)
    assert tolerant["synthetic"]["characters"] < need
    assert abs(tolerant["differences"]["latin"]) <= 10


def test_unreachable_goals_and_finite_name_space_warn_without_blocking():
    config = options(maxImages=3)
    config["targets"]["special"] = 100
    config["characters"]["special"] = "★"
    plan = plan_supplement([{"text": "가나다", "split": "train"}], config, valid)
    assert plan["nicknames"] == ["★"]
    assert plan["warnings"]
    assert plan["real"]["images"] == 1
    with pytest.raises(InterruptedError):
        plan_supplement([{"text": "가", "split": "train"}], config, valid, lambda: True)


def test_invalid_settings_are_distinct_from_unmet_targets():
    config = options()
    config["targets"]["latin"] = 110
    with pytest.raises(ValueError):
        plan_supplement([{"text": "가", "split": "train"}], config, valid)
    config["targets"]["latin"] = 20
    config["characters"]["latin"] = "★"
    with pytest.raises(ValueError, match="문자군"):
        plan_supplement([{"text": "가", "split": "train"}], config, valid)


def test_rgba_is_composited_not_discarded_and_text_cannot_be_cropped(tmp_path):
    config = options()
    image, metadata = composite(
        Image.new("RGBA", (4, 4), (200, 100, 0, 128)), config, [], random.Random(0)
    )
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (20, 40, 60)
    assert image.getpixel((19, 9)) == (110, 70, 30)
    assert metadata["textOffset"] == [18, 8]
    with pytest.raises(ValueError, match="크롭"):
        composite(Image.new("RGBA", (50, 4)), config, [], random.Random(0))
    background = tmp_path / "background.png"
    Image.new("RGBA", (50, 25), (20, 30, 40, 200)).save(background)
    with pytest.raises(ValueError, match="불투명"):
        composite(Image.new("RGBA", (4, 4)), config, [background], random.Random(0))


def raid_crop_geometry(scale, row, heading):
    # Test double of dnf-ocr-synth name_crop_geometry: both cell edges rounded in frame pixels.
    x, y = heading
    top = 21 + 21 * (row - 1)
    left, upper = round(x + 182 * scale), round(y + top * scale)
    return SimpleNamespace(
        x=left,
        y=upper,
        width=round(x + 268 * scale) - left,
        height=round(y + (top + 17) * scale) - upper,
    )


class FakeRenderer:
    def __init__(self):
        self.calls = []

    def render(self, text, *, scale, **options):
        # This tests alpha composition and input selection, not game font appearance.
        self.calls.append({"text": text, "scale": scale, **options})
        size = (len(text) * 4, 5)
        return SimpleNamespace(
            image=Image.new("RGBA", (size[0] + 4, size[1] + 4), (255, 255, 0, 128)),
            metadata={
                "text": text,
                "scale": scale,
                "scale_model": {"name": options["scale_model"]},
                "padding_px": 2,
                "native_ink_size": list(size),
            },
        )


def fake_synthesizer(renderer=None):
    return Synthesizer(
        renderer or FakeRenderer(),
        valid,
        "test-double",
        raid_crop_geometry,
        lambda width, rounding: (86 - width + (rounding == "up")) // 2,
        lambda text, _profile, _model: (0, -4, len(text) * 4),
    )


@pytest.mark.parametrize("version", ["0.1.2", "0.2.1"])
def test_unsupported_synthesizer_fails_before_constructing_renderer(snapshot, monkeypatch, version):
    root, _ = snapshot
    output = root / "previews/version-check"
    output.mkdir(parents=True)
    renderer = Mock()
    module = SimpleNamespace(FontPaths=Mock(), Renderer=renderer, validate_nickname=valid)
    monkeypatch.setitem(sys.modules, "dnf_ocr_synth", module)
    monkeypatch.setattr("importlib.metadata.version", lambda _: version)

    with pytest.raises(ValueError, match="0.2.0"):
        prepare_supplement(
            {
                "directory": str(root),
                "outputDirectory": str(output),
                "mode": "preview",
                "options": options(),
            }
        )

    renderer.assert_not_called()
    assert not (output / "images").exists()


def test_generate_uses_frozen_real_input_and_one_selected_supplement_only(snapshot):
    root, dataset = snapshot
    before = {name: digest(root / name) for name in ("dataset.json", "model/characters.txt")}
    config = options()
    config["targets"]["latin"] = 75
    config["characters"]["latin"] = "ABCD"
    run = root / "runs/first"
    output = run / "synthetic"
    output.mkdir(parents=True)
    request = {
        "directory": str(root),
        "outputDirectory": str(output),
        "mode": "generate",
        "options": config,
    }
    data = prepare_supplement(request, synthesizer=fake_synthesizer())
    chars = set("가나다ABCD ")
    rows, _ = load_supplement(root, run, digest(output / "supplement.json"), dataset, chars)
    assert len(rows) == data["plan"]["synthetic"]["images"] > 0
    assert all(row["split"] == "train" for row in rows)
    assert dataset["counts"] == {"train": 1, "val": 1, "test": 1, "skipped": 0}
    assert all(digest(root / name) == sha for name, sha in before.items())
    with Image.open(root / rows[0]["image"]) as image:
        assert image.mode == "RGB"
    data["samples"][0]["text"] = "나"
    (output / "supplement.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="중복"):
        load_supplement(root, run, digest(output / "supplement.json"), dataset, chars)


def test_expansion_is_explicit_and_preserves_original_dictionary_and_input(snapshot):
    root, _ = snapshot
    original = (root / "model/characters.txt").read_bytes()
    added = write_run_dictionary(
        root / "model/characters.txt", root / "expanded.txt", "★龍あア ★가"
    )
    assert added == ["★", "龍", "あ", "ア"]
    assert (root / "expanded.txt").read_text(encoding="utf-8").splitlines() == [
        "가",
        "나",
        "다",
        *added,
    ]
    assert (root / "model/characters.txt").read_bytes() == original
    assert extend_characters(["가", "나"], "가★") == ["가", "나", "★"]
    load_snapshot(root, "★龍あア")


def test_weight_extension_preserves_all_existing_classes_space_special_tokens_and_non_heads():
    original = ["가", "나"]
    extended = [*original, "★", "龍", "あ", "ア"]
    axes = {
        "head.ctc_head.fc.weight": (1, 1, 0),
        "head.ctc_head.fc.bias": (0, 1, 0),
        "head.gtc_head.embedding.embedding.weight": (0, 4, 1),
        "head.gtc_head.tgt_word_prj.weight": (1, 4, 1),
    }
    state = {"backbone.weight": np.arange(4, dtype=np.float32)}
    initialized = {"backbone.weight": np.full(4, -1, dtype=np.float32)}
    for name, (axis, prefix, trailing) in axes.items():
        source_shape = (
            [3, prefix + len(original) + 1 + trailing]
            if axis == 1
            else [prefix + len(original) + 1 + trailing, 3]
        )
        if name.endswith("bias"):
            source_shape = source_shape[:1]
        target_shape = source_shape.copy()
        target_shape[axis] += len(extended) - len(original)
        state[name] = np.arange(np.prod(source_shape), dtype=np.float32).reshape(source_shape)
        initialized[name] = np.full(target_shape, -1, dtype=np.float32)
    result = expand_state_arrays(state, initialized, original, extended)
    np.testing.assert_array_equal(result["backbone.weight"], state["backbone.weight"])
    for name, (axis, prefix, trailing) in axes.items():
        indices = list(range(prefix + len(original))) + list(
            range(prefix + len(extended), prefix + len(extended) + 1 + trailing)
        )
        np.testing.assert_array_equal(np.take(result[name], indices, axis=axis), state[name])
        np.testing.assert_array_equal(
            np.take(
                result[name], list(range(prefix + len(original), prefix + len(extended))), axis=axis
            ),
            -1,
        )
        assert np.all(initialized[name] == -1)
    with pytest.raises(ValueError, match="순서"):
        expand_state_arrays(state, initialized, original, ["나", "가", "★"])
    broken = {**initialized, "backbone.weight": np.zeros(8, dtype=np.float32)}
    with pytest.raises(ValueError, match="출력층 밖"):
        expand_state_arrays(state, broken, original, extended)


@pytest.mark.parametrize("height", [17, 18, 19, 24, 31])
def test_train_height_maps_to_scales_whose_name_crop_has_that_height(height):
    low, high = scale_range(height)
    rng = random.Random(height)
    # Both ends are open except s=1 itself: there the crop is always exactly 17px.
    for step in range(200):
        scale = low + (high - low) * (step + 0.5) / 200
        placement = place_cell(raid_crop_geometry, scale, height, rng)
        assert placement.crop[3] == height


def test_scale_range_ends_where_the_height_stops_being_reachable():
    assert scale_range(19) == (18 / 17, 20 / 17)
    assert scale_range(17) == (1, 18 / 17)
    rng = random.Random(0)
    for scale in (18 / 17 - 0.001, 20 / 17 + 0.001):
        with pytest.raises(ValueError):
            place_cell(raid_crop_geometry, scale, 19, rng)


def test_heights_that_need_a_scale_below_one_have_no_range():
    assert scale_range(16) is None
    assert scale_range(4) is None
    assert place_cell(raid_crop_geometry, 1, 17, random.Random(0)).crop[2:] == (86, 17)
    rows = [
        {"split": "train", "height": 19},
        {"split": "train", "height": 16},
        {"split": "train", "height": 16},
        {"split": "val", "height": 30},
    ]
    assert scale_heights(rows) == ([19], {16: 2})


def scale_folders(root, folders):
    for name in folders:
        (root / name).mkdir()
        Image.new("RGB", (90, 20), (10, 20, 30)).save(root / name / "band.png")


def test_background_comes_from_the_nearest_scale_folder_and_root_files_are_ignored(tmp_path):
    scale_folders(tmp_path, ["s1.00", "s1.50", "glow"])
    Image.new("RGB", (90, 20)).save(tmp_path / "root.png")
    config = options(placement="raid-name", backgroundMode="images")
    config["backgroundDirectory"] = str(tmp_path)
    backgrounds = background_files(config)
    assert set(backgrounds) == {1.0, 1.5}
    assert nearest_backgrounds(backgrounds, 1.2) == [tmp_path / "s1.00/band.png"]
    assert nearest_backgrounds(backgrounds, 1.3) == [tmp_path / "s1.50/band.png"]
    assert nearest_backgrounds(backgrounds, 1.8) == [tmp_path / "s1.50/band.png"]


def test_background_folder_without_scale_folders_uses_its_own_files(tmp_path):
    scale_folders(tmp_path, ["glow"])
    Image.new("RGB", (90, 20)).save(tmp_path / "root.png")
    config = options(placement="raid-name", backgroundMode="images")
    config["backgroundDirectory"] = str(tmp_path)
    assert nearest_backgrounds(background_files(config), 1.7) == [tmp_path / "root.png"]


def with_train_heights(root, dataset, heights):
    rows = []
    for row, height in zip(dataset["samples"], heights, strict=True):
        path = root / row["image"]
        Image.new("RGB", (90, height), (row["slot"] * 40, 50, 90)).save(path)
        rows.append({**row, "width": 90, "height": height, "sha256": digest(path)})
    save_dataset(root, {**dataset, "samples": rows})


def test_train_heights_records_scale_model_and_draws_each_crop_at_a_real_train_height(snapshot):
    root, dataset = snapshot
    with_train_heights(root, dataset, [19, 24, 30])
    config = options(
        scaleModel="calibrated-v1", scaleSampling="train-heights", placement="raid-name"
    )
    config["targets"]["latin"] = 75
    config["characters"]["latin"] = "ABCD"
    output = root / "previews/heights"
    output.mkdir(parents=True)
    renderer = FakeRenderer()
    data = prepare_supplement(
        {
            "directory": str(root),
            "outputDirectory": str(output),
            "mode": "preview",
            "options": config,
        },
        synthesizer=fake_synthesizer(renderer),
    )
    saved = json.loads((output / "supplement.json").read_text(encoding="utf-8"))
    assert saved == data
    assert saved["options"]["scaleSampling"] == "train-heights"
    assert saved["renderer"]["scaleModel"] == "calibrated-v1"
    # Only the train row counts; val and test heights must not shape the synthetic scales.
    assert saved["scaleSampling"] == {
        "mode": "train-heights",
        "trainHeights": {"19": 1},
        "skippedHeights": {},
    }
    assert {call["scale_model"] for call in renderer.calls} == {"calibrated-v1"}
    assert saved["samples"]
    for sample in saved["samples"]:
        scale = sample["placement"]["scale"]
        assert 18 / 17 <= scale < 20 / 17
        assert sample["rendering"]["scale"] == scale
        assert sample["height"] == 19
        with Image.open(root / sample["image"]) as image:
            assert image.size == (sample["width"], 19)


def test_train_heights_reject_a_train_without_any_height_at_scale_one_or_more(snapshot):
    root, dataset = snapshot
    config = options(
        scaleModel="calibrated-v1", scaleSampling="train-heights", placement="raid-name"
    )
    config["targets"]["latin"] = 75
    config["characters"]["latin"] = "ABCD"
    output = root / "previews/too-small"
    output.mkdir(parents=True)
    request = {"directory": str(root), "outputDirectory": str(output), "mode": "preview"}
    with pytest.raises(ValueError, match="배율 1 이상"):
        prepare_supplement({**request, "options": config}, synthesizer=fake_synthesizer())


def test_calibrated_model_rejects_a_fixed_scale_below_one(snapshot):
    root, _ = snapshot
    output = root / "previews/small-scale"
    output.mkdir(parents=True)
    config = options(scaleModel="calibrated-v1", scale=0.8)
    with pytest.raises(ValueError, match="calibrated-v1"):
        prepare_supplement(
            {
                "directory": str(root),
                "outputDirectory": str(output),
                "mode": "preview",
                "options": config,
            },
            synthesizer=fake_synthesizer(),
        )
