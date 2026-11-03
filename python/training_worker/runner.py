import json
import math
from datetime import datetime, timezone
from pathlib import Path

from eval_worker.dataset import read_object

from .dictionary import write_run_dictionary
from .raid_binarization import DEFAULT_DETACHED_GAP, DETACHED_GAPS
from .recognition.distance import edit_distance
from .recognition.metrics import recognition_metrics
from .runtime import REVISION, prepare_runtime
from .snapshot import digest, load_snapshot
from .supplement_snapshot import load_supplement


class Cancelled(Exception):
    pass


def validate_options(value: dict) -> None:
    for key, maximum in (("epochs", 1000), ("batchSize", 128)):
        if type(value.get(key)) is not int or not 1 <= value[key] <= maximum:
            raise ValueError(f"Invalid {key}.")
    learning_rate = value.get("learningRate")
    if (
        type(learning_rate) not in (int, float)
        or not math.isfinite(learning_rate)
        or not 1e-8 <= learning_rate <= 0.1
    ):
        raise ValueError("Invalid learningRate.")
    if value.get("nicknameDetachedGap", DEFAULT_DETACHED_GAP) not in DETACHED_GAPS:
        raise ValueError("Invalid nicknameDetachedGap.")
    for key in ("directory", "runDirectory", "upstreamDirectory"):
        if not isinstance(value.get(key), str) or not Path(value[key]).is_absolute():
            raise ValueError(f"{key} must be an absolute path.")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def summarize(samples: list[dict]) -> dict:
    values = recognition_metrics([(row["truth"], row["prediction"]) for row in samples])
    return {
        "cer": values["cer"],
        "exactMatch": values["exact_match"],
        "sampleCount": len(samples),
        "characterCount": sum(len(row["truth"]) for row in samples),
        "exactMatchCount": sum(row["truth"] == row["prediction"] for row in samples),
        "editDistance": sum(row["editDistance"] for row in samples),
    }


def run_training(request: dict, cancelled, emit) -> None:
    validate_options(request)
    root = Path(request["directory"]).resolve(strict=True)
    run = Path(request["runDirectory"]).resolve(strict=True)
    if run.parent != root / "runs":
        raise ValueError("Training output must be a new run inside this experiment.")
    if (run / "weights.pdparams").exists():
        raise ValueError("Training output already exists.")
    started = datetime.now(timezone.utc).isoformat()
    detached_gap = request.get("nicknameDetachedGap", DEFAULT_DETACHED_GAP)

    def check_cancel():
        if cancelled():
            raise Cancelled()

    check_cancel()
    additions = request.get("additionalCharacters", "")
    dataset, source_model, warnings = load_snapshot(root, additions)
    added_characters = write_run_dictionary(
        root / "model/characters.txt", run / "characters.txt", additions
    )
    dictionary_sha = digest(run / "characters.txt")
    supplemental = []
    supplement = None
    if request.get("supplementSha256") is not None:
        supplemental, supplement = load_supplement(
            root,
            run,
            request["supplementSha256"],
            dataset,
            set((run / "characters.txt").read_text(encoding="utf-8").splitlines()) | {" "},
        )
    actual_rows = [{**row, "source": "real"} for row in dataset["samples"]] + [
        {**row, "source": "synthetic"} for row in supplemental
    ]
    write_json(
        run / "input.json",
        {
            "schemaVersion": 1,
            "datasetSha256": digest(root / "dataset.json"),
            "dictionarySha256": dictionary_sha,
            "supplementSha256": request.get("supplementSha256"),
            "samples": actual_rows,
        },
    )
    input_sha = digest(run / "input.json")
    dataset_sha = digest(root / "dataset.json")
    for warning in warnings:
        emit({"type": "message", "message": warning})
    emit({"type": "message", "message": "GPU 모델과 학습 입력을 확인하고 있습니다."})
    paddle, np, model, decoder, operators, transform, loss_function = prepare_runtime(
        Path(request["upstreamDirectory"]),
        run / "characters.txt",
        root / "model/weights.pdparams",
        source_dictionary=root / "model/characters.txt",
        detached_gap=detached_gap,
    )
    paddle.seed(42)
    rng = np.random.default_rng(42)
    rows = {
        split: [row for row in actual_rows if row["split"] == split]
        for split in ("train", "val", "test")
    }

    def transformed(row: dict):
        check_cancel()
        path = root / row["image"]
        if digest(path) != row["sha256"]:
            raise ValueError("Image changed during training/evaluation.")
        result = transform({"image": path.read_bytes(), "label": row["text"]}, operators)
        if result is None or len(result) != 5 or int(result[3]) != len(row["text"]):
            raise ValueError("PaddleOCR rejected or truncated a selected label.")
        return result

    for row in actual_rows:
        transformed(row)

    def evaluate(split: str, publish: bool = False) -> tuple[dict, list[dict]]:
        check_cancel()
        model.eval()
        results = []
        with paddle.no_grad():
            for row in rows[split]:
                batch = transformed(row)
                output = model(paddle.to_tensor(np.stack([batch[0]]))).numpy()
                prediction, confidence = decoder(output)[0]
                sample = {
                    "id": row["id"],
                    "imagePath": str(root / row["image"]),
                    "truth": row["text"],
                    "prediction": prediction,
                    "editDistance": edit_distance(row["text"], prediction),
                    "confidence": float(confidence),
                    "imageSha256": row["sha256"],
                }
                results.append(sample)
                if publish:
                    emit({"type": "sample", "sample": sample})
        return summarize(results), results

    optimizer = paddle.optimizer.Adam(
        learning_rate=request["learningRate"],
        parameters=model.parameters(),
        weight_decay=3e-5,
        grad_clip=paddle.nn.ClipGradByGlobalNorm(5.0),
    )
    history = []
    best = None
    optimizer_steps = 0
    for epoch in range(1, request["epochs"] + 1):
        check_cancel()
        model.train()
        order = rng.permutation(len(rows["train"])).tolist()
        losses = []
        for offset in range(0, len(order), request["batchSize"]):
            batch_rows = [
                transformed(rows["train"][index])
                for index in order[offset : offset + request["batchSize"]]
            ]
            batch = [
                paddle.to_tensor(np.stack([sample[index] for sample in batch_rows]))
                for index in range(5)
            ]
            loss = loss_function(model(batch[0], data=batch[1:]), batch)["loss"]
            value = float(loss.item())
            if not math.isfinite(value):
                raise ValueError("Non-finite training loss.")
            loss.backward()
            optimizer.step()
            optimizer.clear_grad()
            optimizer_steps += 1
            losses.append(value)
            emit(
                {
                    "type": "progress",
                    "epoch": epoch,
                    "epochs": request["epochs"],
                    "message": f"학습 {epoch}/{request['epochs']} epoch · {min(offset + len(batch_rows), len(order))}/{len(order)} · loss {value:.4f}",
                }
            )
        validation, _ = evaluate("val")
        entry = {"epoch": epoch, "loss": sum(losses) / len(losses), "validation": validation}
        history.append(entry)
        if (
            best is None
            or validation["cer"] < best["validation"]["cer"]
            or (
                validation["cer"] == best["validation"]["cer"]
                and validation["exactMatch"] > best["validation"]["exactMatch"]
            )
        ):
            best = entry
            paddle.save(model.state_dict(), str(run / "weights.pdparams"))
        write_json(run / "progress.json", {"history": history, "bestEpoch": best["epoch"]})
        emit(
            {
                "type": "progress",
                "epoch": epoch,
                "epochs": request["epochs"],
                "message": f"검증 {epoch}/{request['epochs']} · CER {validation['cer']:.4f} · 최적 epoch {best['epoch']}",
            }
        )
    check_cancel()
    if (
        digest(root / "dataset.json") != dataset_sha
        or digest(run / "characters.txt") != dictionary_sha
        or digest(run / "input.json") != input_sha
    ):
        raise ValueError("Dataset metadata changed during training.")
    missing, unexpected = model.set_state_dict(paddle.load(str(run / "weights.pdparams")))
    if missing or unexpected:
        raise ValueError("Selected checkpoint could not be reloaded.")
    emit({"type": "evaluating", "message": "val로 선택한 모델을 test 데이터로 평가합니다."})
    metrics, samples = evaluate("test", publish=True)
    check_cancel()
    finished = datetime.now(timezone.utc).isoformat()
    scorer_root = Path(__file__).parent / "recognition"
    report = {
        "schemaVersion": 1,
        "status": "completed",
        "totalSamples": len(samples),
        "processedSamples": len(samples),
        "summary": metrics,
        "samples": samples,
        "error": None,
        "startedAt": started,
        "finishedAt": finished,
        "reproducibility": {
            "normalization": "none",
            "settings": {"datasetDirectory": str(root)},
            "datasetSha256": dataset_sha,
            "inputSha256": input_sha,
            "dictionarySha256": dictionary_sha,
            "addedCharacters": added_characters,
            "syntheticImages": len(supplemental),
            "supplementSha256": request.get("supplementSha256"),
            "modelId": source_model["id"],
            "revision": REVISION,
            "checkpointSha256": digest(run / "weights.pdparams"),
            "sourceSha256": {
                f"recognition/{name}": digest(scorer_root / name)
                for name in ("metrics.py", "distance.py")
            },
            "preprocessing": f"dfragon raid nickname binarization (detached gap {detached_gap} of the height) after DecodeImage, then official Korean PP-OCRv5 Eval transforms; resize and padding [3,48,320]",
            "nicknameDetachedGap": detached_gap,
            "seed": 42,
            "paddle": paddle.__version__,
            "cuda": paddle.version.cuda(),
            "gpu": paddle.device.cuda.get_device_name(0),
        },
    }
    write_json(run / "report.json", report)
    write_json(
        run / "evaluation.json",
        {
            "schemaVersion": 1,
            "sourceModelId": source_model["id"],
            "datasetSha256": dataset_sha,
            "inputSha256": input_sha,
            "dictionarySha256": dictionary_sha,
            "addedCharacters": added_characters,
            "syntheticImages": len(supplemental),
            "supplementSha256": request.get("supplementSha256"),
            "summary": metrics,
            "validation": best["validation"],
            "bestEpoch": best["epoch"],
            "training": {key: request[key] for key in ("epochs", "batchSize", "learningRate")},
            "supplement": None
            if supplement is None
            else {
                "targets": supplement["options"]["targets"],
                "tolerance": supplement["options"]["tolerance"],
                "syntheticImages": len(supplemental),
                "finalGroups": supplement["plan"]["final"]["groups"],
            },
            "optimizerSteps": optimizer_steps,
            "history": history,
            "revision": REVISION,
            "checkpointSha256": digest(run / "weights.pdparams"),
            "finishedAt": finished,
        },
    )
    emit({"type": "finished", "metrics": metrics, "reportPath": str(run / "report.json")})


def read_request(path: Path) -> dict:
    value = read_object(path)
    validate_options(value)
    return value
