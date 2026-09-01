import json
from pathlib import Path

from PIL import Image

from aigc_recognizer.baseline_examples import RECORD_ID, SCENARIO, build_example


def _predictions(path: Path, *, univfd: bool) -> None:
    rows = [
        {
            "id": RECORD_ID,
            "image_path": f"images/fake/{RECORD_ID}.jpg",
            "label": 1,
            "pred": score,
            "scenario": scenario,
        }
        for scenario, score in (
            ("clean", 0.9),
            (SCENARIO, 0.1 if univfd else 0.9),
        )
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_build_example_requires_univfd_flip_and_writes_light_asset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path = dataset_root / "images" / "fake" / f"{RECORD_ID}.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (480, 320), "#BFD7EA").save(image_path)
    predictions = {}
    results = {}
    for model in ("UnivFD", "RobustFake"):
        prediction_path = tmp_path / f"{model}.jsonl"
        result_path = tmp_path / f"{model}.json"
        _predictions(prediction_path, univfd=model == "UnivFD")
        result_path.write_text(json.dumps({"threshold": 0.5}), encoding="utf-8")
        predictions[model] = prediction_path
        results[model] = result_path
    output = tmp_path / "output"
    case = build_example(
        dataset_root=dataset_root,
        predictions=predictions,
        results=results,
        output_directory=output,
    )
    assert case["models"]["UnivFD"]["clean_correct"]
    assert not case["models"]["UnivFD"]["transformed_correct"]
    assert case["models"]["RobustFake"]["transformed_correct"]
    with Image.open(output / "univfd-transform-flip.png") as slide:
        assert slide.size == (1600, 900)
        assert slide.getpixel((0, 0)) == (246, 248, 251)
