import copy
import os
import subprocess
import sys
from pathlib import Path

from .raid_binarization import DEFAULT_DETACHED_GAP, RaidNicknameBinarize

REVISION = "b03f46425e8ff4442b268ce449e3eef758146cd4"
_dll_handles = []


def prepare_runtime(
    upstream: Path,
    dictionary: Path,
    weights: Path,
    source_dictionary: Path | None = None,
    detached_gap: str = DEFAULT_DETACHED_GAP,
):
    revision = subprocess.run(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    changed = subprocess.run(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if revision != REVISION or changed.strip():
        raise ValueError("PaddleOCR source must be the unmodified pinned revision: " + REVISION)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
    if sys.platform == "win32":
        directories = sorted((Path(sys.prefix) / "Lib/site-packages/nvidia").glob("*/bin"))
        for directory in directories:
            _dll_handles.append(os.add_dll_directory(str(directory)))
        os.environ["PATH"] = (
            os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
        )
    sys.path.insert(0, str(upstream))
    import numpy as np
    import paddle
    import yaml
    from ppocr.data.imaug import create_operators, transform
    from ppocr.losses import build_loss
    from ppocr.modeling.architectures import build_model
    from ppocr.postprocess import build_post_process

    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() < 1:
        raise RuntimeError("A working NVIDIA CUDA GPU is required.")
    paddle.set_device("gpu:0")
    config_path = upstream / "configs/rec/PP-OCRv5/multi_language/korean_PP-OCRv5_mobile_rec.yml"
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["Global"]["character_dict_path"] = str(dictionary)
    config["Global"]["use_space_char"] = True
    decoder = build_post_process(config["PostProcess"], config["Global"])
    count = len(decoder.character)
    config["Architecture"]["Head"]["out_channels_list"] = {
        "CTCLabelDecode": count,
        "SARLabelDecode": count + 2,
        "NRTRLabelDecode": count + 3,
    }
    paddle.seed(42)
    np.random.seed(42)
    model = build_model(copy.deepcopy(config["Architecture"]))
    state = paddle.load(str(weights))
    target = model.state_dict()
    original = (source_dictionary or dictionary).read_text(encoding="utf-8").splitlines()
    extended = dictionary.read_text(encoding="utf-8").splitlines()
    if original != extended:
        from .dictionary import expand_state_arrays

        expanded = expand_state_arrays(
            {name: value.numpy() for name, value in state.items()},
            {name: value.numpy() for name, value in target.items()},
            original,
            extended,
        )
        state = {
            name: paddle.to_tensor(value, place=target[name].place)
            for name, value in expanded.items()
        }
    elif state.keys() != target.keys() or any(
        list(state[name].shape) != list(value.shape) for name, value in target.items()
    ):
        raise ValueError("Model parameter names or shapes do not match the Korean PP-OCRv5 preset.")
    missing, unexpected = model.set_state_dict(state)
    if missing or unexpected:
        raise ValueError("Model weights could not be loaded completely.")
    transforms = copy.deepcopy(config["Eval"]["dataset"]["transforms"])
    operators = create_operators(transforms, config["Global"])
    if type(operators[0]).__name__ != "DecodeImage":
        raise ValueError("The Korean PP-OCRv5 Eval transforms must start with DecodeImage.")
    operators.insert(1, RaidNicknameBinarize(detached_gap))
    loss = build_loss(copy.deepcopy(config["Loss"]))
    return paddle, np, model, decoder, operators, transform, loss
