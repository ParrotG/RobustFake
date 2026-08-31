from PIL import Image, ImageFilter, ImageEnhance

from aigc_recognizer.degradation_restore import (
    _adaptive_gaussian_unsharp,
    _gradient_metrics,
    _to_rgb_array,
    analyze_degradation,
    restore_image,
)


def _sample() -> Image.Image:
    image = Image.new("RGB", (192, 160), (40, 70, 120))
    pixels = image.load()
    for y in range(image.height):
        for x in range(image.width):
            pixels[x, y] = ((x * 5 + y) % 256, (y * 7) % 256, ((x + y) * 3) % 256)
    return image


def test_report_contains_fixed_residual_artifact_scores() -> None:
    report = analyze_degradation(_sample())
    assert set(report.artifacts) == {"gaussian_blur", "gaussian_noise", "resize", "color_jitter"}
    assert report.metrics["hf_energy"] >= 0
    assert all(0 <= evidence.confidence <= 1 for evidence in report.artifacts.values())


def test_blurred_input_gets_restoration_without_shape_change() -> None:
    blurred = _sample().filter(ImageFilter.GaussianBlur(4.0))
    report = analyze_degradation(blurred)
    restored, operations = restore_image(blurred, report)
    assert restored.size == blurred.size
    assert restored.mode == "RGB"
    assert any(operation.startswith("gaussian_unsharp(") for operation in operations)


def test_adaptive_gaussian_parameters_raise_blurred_edge_response() -> None:
    blurred = _sample().filter(ImageFilter.GaussianBlur(4.0))
    restored, parameters = _adaptive_gaussian_unsharp(blurred, confidence=0.9)
    before_gradient = _gradient_metrics(_to_rgb_array(blurred))[0]
    after_gradient = _gradient_metrics(_to_rgb_array(restored))[0]

    assert 0.5 <= parameters.radius <= 3.0
    assert parameters.percent >= 70
    assert parameters.threshold in {2, 4, 7}
    assert parameters.residual_gain > 1.0
    assert after_gradient > before_gradient


def test_color_shift_restoration_is_non_destructive() -> None:
    shifted = ImageEnhance.Color(_sample()).enhance(1.8)
    restored, _ = restore_image(shifted)
    assert restored.size == shifted.size
    assert restored.getbbox() is not None
