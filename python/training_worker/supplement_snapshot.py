"""Validate the selected run's synthetic train input before GPU work."""

import hashlib
import unicodedata
from pathlib import Path

from PIL import Image

from eval_worker.dataset import read_object

from .snapshot import digest
from .supplement_plan import distribution


def pixel_key(path: Path, width: int, height: int) -> str:
    with Image.open(path) as source:
        if source.format != "PNG" or source.size != (width, height) or width * height > 16_777_216:
            raise ValueError("합성 이미지 크기가 기록과 다릅니다.")
        image = source.convert("RGBA")
        if image.getextrema()[3] != (255, 255):
            raise ValueError("합성 학습 입력은 배경과 합성한 불투명 PNG여야 합니다.")
        return hashlib.sha256(f"{width}x{height}:".encode() + image.tobytes()).hexdigest()


def load_supplement(
    root: Path, run: Path, expected_hash: str, dataset: dict, characters: set[str]
) -> tuple[list[dict], dict]:
    directory = run / "synthetic"
    manifest = directory / "supplement.json"
    if digest(manifest) != expected_hash:
        raise ValueError("합성 입력 목록이 변경되었습니다.")
    data = read_object(manifest)
    if (
        data.get("schemaVersion") != 2
        or data.get("mode") != "generate"
        or data.get("datasetSha256") != digest(root / "dataset.json")
        or data.get("dictionarySha256") != digest(root / "model/characters.txt")
    ):
        raise ValueError("합성 입력과 실제 데이터·모델이 다릅니다.")
    forbidden = {unicodedata.normalize("NFC", row["text"]) for row in dataset["samples"]}
    evaluation_pixels = {
        pixel_key(root / row["image"], row["width"], row["height"])
        for row in dataset["samples"]
        if row["split"] in {"val", "test"}
    }
    rows = data.get("samples")
    if not isinstance(rows, list) or len(rows) > 100_000:
        raise ValueError("합성 입력 목록 크기가 올바르지 않습니다.")
    for index, row in enumerate(rows):
        text = row.get("text")
        expected = directory / "images" / f"{index:06d}.png"
        if (
            row.get("id") != f"synthetic-{index:06d}"
            or row.get("split") != "train"
            or row.get("image") != expected.relative_to(root).as_posix()
            or expected.resolve(strict=True) != expected
        ):
            raise ValueError("합성 데이터는 현재 실행의 train에만 추가할 수 있습니다.")
        if (
            not isinstance(text, str)
            or not text
            or len(text) > 25
            or any(char not in characters for char in text)
            or unicodedata.normalize("NFC", text) in forbidden
        ):
            raise ValueError("합성 닉네임이 중복되거나 사전 밖 문자가 있습니다.")
        forbidden.add(unicodedata.normalize("NFC", text))
        if (
            digest(expected) != row.get("sha256")
            or pixel_key(expected, row["width"], row["height"]) in evaluation_pixels
        ):
            raise ValueError("합성 이미지가 변경되었거나 val/test 픽셀과 겹칩니다.")
    actual = distribution([row["text"] for row in rows])
    if data.get("plan", {}).get("synthetic") != actual:
        raise ValueError("합성 입력의 실제 분포가 저장된 결과와 다릅니다.")
    return rows, data
