"""Composite local-font RGBA renders onto explicit opaque crop backgrounds."""

import importlib.metadata
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from eval_worker.dataset import read_object

from .dictionary import extend_characters
from .snapshot import digest, load_snapshot
from .supplement_geometry import CellPlacement, place_cell, scale_range, text_offset
from .supplement_plan import GROUPS, plan_supplement, validate_plan_options

SYNTHESIZER_VERSION = "0.2.0"
SCALE_MODELS = ("calibrated-v1", "affine-bilinear")
SCALE_SAMPLINGS = ("fixed", "train-heights")
PLACEMENTS = ("raid-name", "center")
SCALE_FOLDER = re.compile(r"s(\d+(?:\.\d+)?)")


@dataclass(frozen=True)
class Synthesizer:
    renderer: object
    validate_nickname: Callable
    version: str
    crop_geometry: Callable
    text_left: Callable[[int, str], int]
    text_box: Callable[[str, str, str], tuple[int, int, int]]


def load_synthesizer(fonts: dict) -> Synthesizer:
    """Check the supported local runtime before importing or constructing its renderer."""
    try:
        version = importlib.metadata.version("dnf-ocr-synth")
    except importlib.metadata.PackageNotFoundError as error:
        raise ValueError(
            f"합성용 Python에 dnf-ocr-synth {SYNTHESIZER_VERSION}를 설치해 주세요."
        ) from error
    if version != SYNTHESIZER_VERSION:
        raise ValueError(
            f"지원하는 dnf-ocr-synth 버전은 {SYNTHESIZER_VERSION}입니다. 현재 버전: {version}"
        )

    try:
        from dnf_ocr_synth import SCALE_MODELS as SYNTH_SCALE_MODELS
        from dnf_ocr_synth import (
            FontPaths,
            Renderer,
            name_crop_geometry,
            name_text_left,
            validate_nickname,
        )
        from dnf_ocr_synth.calibration.layout import raid_layout, text_box
    except ImportError as error:
        raise ValueError("합성용 Python의 dnf-ocr-synth 의존성을 확인해 주세요.") from error

    renderer = Renderer(
        FontPaths(**{key: Path(path) if path else None for key, path in fonts.items()})
    )

    def box(text: str, profile: str, scale_model: str) -> tuple[int, int, int]:
        # render() lays raid-name models out like this but does not return the baseline offset.
        if SYNTH_SCALE_MODELS[scale_model].glyph_layout == "raid-name":
            return text_box(raid_layout(renderer, text, profile, "ink"))

        return text_box(renderer.layout(text, profile=profile))

    return Synthesizer(
        renderer, validate_nickname, version, name_crop_geometry, name_text_left, box
    )


def validate_render_options(options: dict) -> None:
    validate_plan_options(options)
    if options.get("profile") not in {"dotum", "nanum-neo"}:
        raise ValueError("지원하지 않는 합성 글꼴 모드입니다.")
    if (
        options.get("scaleModel") not in SCALE_MODELS
        or options.get("scaleSampling") not in SCALE_SAMPLINGS
        or options.get("placement") not in PLACEMENTS
    ):
        raise ValueError("확대 방식, 배율 분포, 글자 배치를 확인해 주세요.")
    scale = options.get("scale")
    if type(scale) not in (int, float) or not math.isfinite(scale) or not 0 < scale <= 16:
        raise ValueError("배율은 0보다 크고 16 이하여야 합니다.")
    if (
        options["scaleSampling"] == "fixed"
        and options["scaleModel"] == "calibrated-v1"
        and scale < 1
    ):
        raise ValueError("calibrated-v1은 배율 1 이상만 지원합니다.")
    for key in ("width", "height"):
        if type(options.get(key)) is not int or not 4 <= options[key] <= 2048:
            raise ValueError("크롭 너비·높이는 4~2048px이어야 합니다.")
    padding = options.get("padding")
    if type(padding) is not int or not 0 <= padding < min(options["width"], options["height"]) / 2:
        raise ValueError("여백은 크롭의 절반보다 작은 0 이상의 정수여야 합니다.")
    for key in ("color", "backgroundColor"):
        value = options.get(key)
        if (
            not isinstance(value, list)
            or len(value) != 3
            or any(type(x) is not int or not 0 <= x <= 255 for x in value)
        ):
            raise ValueError("색상은 RGB 정수 3개여야 합니다.")
    if options.get("backgroundMode") not in {"solid", "images"}:
        raise ValueError("배경 종류를 선택해 주세요.")
    if not isinstance(options.get("fonts"), dict) or set(options["fonts"]) != {
        "gulim",
        "batang",
        "nanum",
        "uttum",
    }:
        raise ValueError("로컬 폰트 경로를 확인해 주세요.")
    for path in options["fonts"].values():
        if not isinstance(path, str) or (path and not Path(path).is_absolute()):
            raise ValueError("폰트에는 로컬 절대 경로를 사용해 주세요.")
    if options["backgroundMode"] == "images" and (
        not isinstance(options.get("backgroundDirectory"), str)
        or not Path(options["backgroundDirectory"]).is_absolute()
    ):
        raise ValueError("글자가 없는 배경 이미지 폴더를 선택해 주세요.")


def _image_files(folder: Path, options: dict) -> list[Path]:
    paths = sorted(
        path
        for path in folder.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg"} and path.is_file()
    )
    if not paths or len(paths) > 1000:
        raise ValueError("배경 폴더에는 PNG/JPEG 1~1000개가 필요합니다.")
    for path in paths:
        with Image.open(path) as source:
            if source.width * source.height > 16_777_216 or (
                options["placement"] == "center"
                and (source.width < options["width"] or source.height < options["height"])
            ):
                raise ValueError("배경 이미지는 크롭보다 커야 하며 16777216픽셀 이하여야 합니다.")
            if source.convert("RGBA").getextrema()[3] != (255, 255):
                raise ValueError("배경 이미지가 불투명해야 합니다.")
    return paths


def background_files(options: dict) -> dict[float | None, list[Path]]:
    """Group backgrounds by the scale of their s{scale} subfolder, or None for the root."""
    if options["backgroundMode"] == "solid":
        return {}
    root = Path(options["backgroundDirectory"]).resolve(strict=True)
    folders = {
        float(match[1]): path
        for path in root.iterdir()
        if path.is_dir() and (match := SCALE_FOLDER.fullmatch(path.name))
    }
    if not folders:
        return {None: _image_files(root, options)}

    return {scale: _image_files(path, options) for scale, path in sorted(folders.items())}


def nearest_backgrounds(backgrounds: dict[float | None, list[Path]], scale: float) -> list[Path]:
    if not backgrounds:
        return []

    if None in backgrounds:
        return backgrounds[None]

    return backgrounds[min(backgrounds, key=lambda folder: (abs(folder - scale), folder))]


def _background(
    size: tuple[int, int], options: dict, paths: list[Path], rng: random.Random
) -> tuple[Image.Image, dict]:
    if paths:
        path = rng.choice(paths)
        with Image.open(path) as source:
            rgba = source.convert("RGBA")
            if rgba.getextrema()[3] != (255, 255):
                raise ValueError("배경 이미지가 불투명해야 합니다.")
        # Only name-cell placement gets here with a smaller background: a band cut at the nearest
        # scale is stretched by the few pixels a slightly larger scale or phase adds.
        stretched = (max(rgba.width, size[0]), max(rgba.height, size[1]))
        metadata = {"mode": "images", "folder": path.parent.name, "file": path.name}
        if stretched != rgba.size:
            metadata["stretchedFrom"] = list(rgba.size)
            rgba = rgba.resize(stretched, Image.Resampling.BILINEAR)
        x = rng.randint(0, rgba.width - size[0])
        y = rng.randint(0, rgba.height - size[1])
        background = rgba.crop((x, y, x + size[0], y + size[1]))
        metadata |= {"sha256": digest(path), "crop": [x, y, *size]}

        return background, metadata

    background = Image.new("RGBA", size, (*options["backgroundColor"], 255))

    return background, {"mode": "solid", "rgb": options["backgroundColor"]}


def composite(
    foreground: Image.Image, options: dict, paths: list[Path], rng: random.Random
) -> tuple[Image.Image, dict]:
    size = (options["width"], options["height"])
    if (
        foreground.width + 2 * options["padding"] > size[0]
        or foreground.height + 2 * options["padding"] > size[1]
    ):
        raise ValueError("합성 글자가 크롭을 벗어납니다. 배율·크롭 크기·여백을 확인해 주세요.")
    background, metadata = _background(size, options, paths, rng)
    offset = ((size[0] - foreground.width) // 2, (size[1] - foreground.height) // 2)
    background.alpha_composite(foreground.convert("RGBA"), offset)
    return background.convert("RGB"), {**metadata, "textOffset": list(offset)}


def composite_name_cell(
    foreground: Image.Image,
    ink_size: tuple[int, int],
    padding: int,
    placement: CellPlacement,
    offset: tuple[int, int],
    options: dict,
    paths: list[Path],
    rng: random.Random,
) -> tuple[Image.Image, dict]:
    """Paste the render where the game draws it in a name crop, clipping its outer edge.

    The game's own crop also cuts the outline's soft edge, but never the fill.
    """
    size = placement.crop[2:]
    left = offset[0] + placement.scale * padding
    top = offset[1] + placement.scale * padding
    if (
        left < 0
        or top < 0
        or left + placement.scale * ink_size[0] > size[0]
        or top + placement.scale * ink_size[1] > size[1]
    ):
        raise ValueError("합성 글자가 이름 칸을 벗어납니다. 닉네임 길이와 배율을 확인해 주세요.")
    background, metadata = _background(size, options, paths, rng)
    layer = Image.new("RGBA", size)
    layer.paste(foreground.convert("RGBA"), offset)
    background.alpha_composite(layer)

    return background.convert("RGB"), {**metadata, "textOffset": list(offset)}


def scale_heights(rows: list[dict]) -> tuple[list[int], dict[int, int]]:
    """Return the real train crop heights a scale can be drawn for, and the rest by height."""
    usable = []
    skipped = Counter()
    for row in rows:
        if row["split"] != "train":
            continue

        if scale_range(row["height"]) is None:
            skipped[row["height"]] += 1
        else:
            usable.append(row["height"])

    return usable, dict(sorted(skipped.items()))


def _height_counts(counts: dict[int, int]) -> dict[str, int]:
    return {str(height): count for height, count in sorted(counts.items())}


def sample_scale(options: dict, heights: list[int], rng: random.Random) -> tuple[float, int | None]:
    if options["scaleSampling"] == "fixed":
        return options["scale"], None
    height = rng.choice(heights)
    low, high = scale_range(height)

    return rng.uniform(low, high), height


def prepare_supplement(
    request: dict, cancelled=lambda: False, emit=lambda _: None, synthesizer=None
) -> dict:
    root = Path(request["directory"]).resolve(strict=True)
    output = Path(request["outputDirectory"]).resolve(strict=True)
    if (
        not output.is_relative_to(root)
        or output == root
        or request.get("mode") not in {"preview", "generate"}
    ):
        raise ValueError("합성 출력 경로가 올바르지 않습니다.")
    options = request["options"]
    validate_render_options(options)
    dataset, _, _ = load_snapshot(root, options.get("additionalCharacters", ""))
    dictionary = set(
        extend_characters(
            (root / "model/characters.txt").read_text(encoding="utf-8").splitlines(),
            options.get("additionalCharacters", ""),
        )
    ) | {" "}
    cleaned = validate_plan_options(options)
    unsupported = sorted(
        {
            char
            for group in GROUPS
            for char in cleaned["characters"][group]
            if char not in dictionary
        }
    )
    if unsupported:
        raise ValueError("현재 모델 사전에 없는 합성 문자: " + " ".join(unsupported))
    heights, skipped_heights = scale_heights(dataset["samples"])
    if options["scaleSampling"] == "train-heights" and not heights:
        raise ValueError("실제 train 높이에서 배율 1 이상을 얻을 수 없습니다.")

    if synthesizer is None:
        synthesizer = load_synthesizer(options["fonts"])
    renderer = synthesizer.renderer
    plan = plan_supplement(dataset["samples"], options, synthesizer.validate_nickname, cancelled)
    if options["scaleSampling"] == "train-heights" and skipped_heights:
        plan["warnings"].append(
            "배율 1 미만인 실제 train 높이는 배율 분포에서 뺐습니다: "
            + ", ".join(f"{height}px {count}장" for height, count in skipped_heights.items())
        )
    backgrounds = background_files(options)
    render_options = {
        "profile": options["profile"],
        "color": tuple(options["color"]),
        "scale_model": options["scaleModel"],
    }
    probe_scale = options["scale"] if options["scaleSampling"] == "fixed" else 1
    # Probe the actual font chain, including fallbacks, before generating any training rows.
    for group in GROUPS:
        for char in cleaned["characters"][group]:
            if cancelled():
                raise InterruptedError("합성 준비를 취소했습니다.")
            renderer.render(char, scale=probe_scale, **render_options)
    all_names = plan.pop("nicknames")
    count = min(6, len(all_names)) if request["mode"] == "preview" else len(all_names)
    images = output / "images"
    images.mkdir(exist_ok=False)
    rng = random.Random(options["seed"])
    samples = []
    for index, text in enumerate(all_names[:count]):
        if cancelled():
            raise InterruptedError("합성 준비를 취소했습니다.")
        scale, height = sample_scale(options, heights, rng)
        sample = renderer.render(text, scale=scale, **render_options)
        if options["placement"] == "center":
            placement = {"mode": "center", "scale": scale}
            image, background = composite(
                sample.image, options, nearest_backgrounds(backgrounds, scale), rng
            )
        else:
            cell = place_cell(synthesizer.crop_geometry, scale, height, rng)
            box = synthesizer.text_box(text, options["profile"], options["scaleModel"])
            padding = sample.metadata["padding_px"]
            offset = text_offset(cell, box, padding, synthesizer.text_left)
            placement = {
                "mode": "raid-name",
                "scale": scale,
                "cell": list(cell.cell),
                "rounding": cell.rounding,
            }
            image, background = composite_name_cell(
                sample.image,
                tuple(sample.metadata["native_ink_size"]),
                padding,
                cell,
                offset,
                options,
                nearest_backgrounds(backgrounds, scale),
                rng,
            )
        name = f"{index:06d}.png"
        target = images / name
        image.save(target)
        samples.append(
            {
                "id": f"synthetic-{index:06d}",
                "text": text,
                "split": "train",
                "image": str(target.relative_to(root)).replace("\\", "/"),
                "sha256": digest(target),
                "width": image.width,
                "height": image.height,
                "rendering": sample.metadata,
                "placement": placement,
                "background": background,
            }
        )
        if index % 100 == 0:
            emit({"type": "message", "message": f"합성 이미지 {index + 1}/{count}개 생성 중"})
    if options["scaleSampling"] == "fixed":
        scale_sampling = {"mode": "fixed", "scale": options["scale"]}
    else:
        scale_sampling = {
            "mode": "train-heights",
            "trainHeights": _height_counts(Counter(heights)),
            "skippedHeights": _height_counts(skipped_heights),
        }
    result = {
        "schemaVersion": 2,
        "mode": request["mode"],
        "datasetSha256": digest(root / "dataset.json"),
        "dictionarySha256": digest(root / "model/characters.txt"),
        "renderer": {
            "package": "dnf-ocr-synth",
            "version": synthesizer.version,
            "scaleModel": options["scaleModel"],
        },
        "options": options,
        "scaleSampling": scale_sampling,
        "plan": plan,
        "samples": samples,
    }
    (output / "supplement.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def read_supplement_request(path: Path) -> dict:
    return read_object(path)
