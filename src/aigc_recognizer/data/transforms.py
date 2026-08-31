"""Paired geometric views and realistic image redistribution degradations."""

from __future__ import annotations

import io
import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torchvision.transforms import functional as tvf

from aigc_recognizer.config import AppConfig

CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class Geometry:
    """Geometry shared by clean and transformed versions of one view."""

    crop_box: tuple[int, int, int, int] | None
    interpolation: int


def canonical_rgb(image: Image.Image, padding_color: int) -> Image.Image:
    """Apply orientation and convert all supported modes to opaque RGB."""
    image = ImageOps.exif_transpose(image)
    if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (padding_color,) * 3 + (255,))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    else:
        image = image.convert("RGB")
    return image


class RobustPairTransform:
    """Create clean/degraded global and local views using shared geometry."""

    def __init__(self, config: AppConfig) -> None:
        self.views = config.views
        self.standardization = config.standardization
        self.augment = config.augmentations
        self.seed = config.project.seed

    def _interpolation(self, rng: random.Random) -> int:
        choices = [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]
        return int(rng.choice(choices) if self.views.random_interpolation else Image.Resampling.BICUBIC)

    def _geometries(self, image: Image.Image, rng: random.Random) -> tuple[Geometry, Geometry]:
        interpolation = self._interpolation(rng)
        width, height = image.size

        def square_crop(scale_min: float, scale_max: float) -> Geometry:
            fraction = rng.uniform(scale_min, scale_max)
            side = max(1, round(min(width, height) * fraction))
            left = rng.randint(0, max(0, width - side))
            top = rng.randint(0, max(0, height - side))
            return Geometry((left, top, left + side, top + side), interpolation)

        # Both views crop every source to a square before resize. In particular,
        # the global view must not expose source aspect ratio through constant
        # letterbox padding, which is strongly correlated with the label in the
        # current multi-source pool. The wider global scale retains substantially
        # more context than the local view while applying the same label-agnostic
        # geometry to real and generated images.
        return (
            square_crop(
                self.views.global_crop_scale_min,
                self.views.global_crop_scale_max,
            ),
            square_crop(self.views.local_scale_min, self.views.local_scale_max),
        )

    def _render(self, image: Image.Image, geometry: Geometry) -> Image.Image:
        if geometry.crop_box is not None:
            image = image.crop(geometry.crop_box)
        size = self.views.input_size
        width, height = image.size
        scale = min(size / width, size / height)
        resized = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=geometry.interpolation,
        )
        canvas = Image.new("RGB", (size, size), (self.views.padding_color,) * 3)
        canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
        return canvas

    @staticmethod
    def _encode(image: Image.Image, image_format: str, **kwargs: object) -> Image.Image:
        buffer = io.BytesIO()
        image.save(buffer, format=image_format, **kwargs)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB").copy()

    def _standardize_resize(self, image: Image.Image, rng: random.Random) -> Image.Image:
        scale = rng.uniform(
            self.standardization.resize_scale_min,
            self.standardization.resize_scale_max,
        )
        width, height = image.size
        interpolation = self._interpolation(rng)
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=interpolation,
        )
        return reduced.resize((width, height), resample=interpolation)

    def _standardize_codec(self, image: Image.Image, rng: random.Random) -> Image.Image:
        quality = rng.randint(
            self.standardization.quality_min,
            self.standardization.quality_max,
        )
        codec = rng.choices(
            ["JPEG", "WEBP"],
            weights=[
                self.standardization.jpeg_weight,
                self.standardization.webp_weight,
            ],
            k=1,
        )[0]
        options: dict[str, object] = {"quality": quality}
        if codec == "JPEG":
            options["subsampling"] = 2
        return self._encode(image, codec, **options)

    def standardize(self, image: Image.Image, rng: random.Random) -> Image.Image:
        """Apply conservative image-only random standardization before view creation."""
        if not self.standardization.enabled:
            return image
        if rng.random() >= self.standardization.application_probability:
            return image
        mode = rng.choices(
            ["resize", "codec", "resize_codec"],
            weights=[
                self.standardization.resize_weight,
                self.standardization.codec_weight,
                self.standardization.resize_codec_weight,
            ],
            k=1,
        )[0]
        if mode in {"resize", "resize_codec"}:
            image = self._standardize_resize(image, rng)
        if mode in {"codec", "resize_codec"}:
            image = self._standardize_codec(image, rng)
        return image

    def _jpeg(self, image: Image.Image, rng: random.Random) -> Image.Image:
        quality = rng.randint(self.augment.jpeg_quality_min, self.augment.jpeg_quality_max)
        return self._encode(image, "JPEG", quality=quality, subsampling=2)

    def _double_jpeg(self, image: Image.Image, rng: random.Random) -> Image.Image:
        return self._jpeg(self._jpeg(image, rng), rng)

    def _webp(self, image: Image.Image, rng: random.Random) -> Image.Image:
        quality = rng.randint(40, 90)
        return self._encode(image, "WEBP", quality=quality)

    def _blur(self, image: Image.Image, rng: random.Random) -> Image.Image:
        sigma = rng.uniform(self.augment.blur_sigma_min, self.augment.blur_sigma_max)
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))

    def _resize(self, image: Image.Image, rng: random.Random) -> Image.Image:
        scale = rng.uniform(self.augment.resize_scale_min, self.augment.resize_scale_max)
        width, height = image.size
        interpolation = self._interpolation(rng)
        reduced = image.resize(
            (max(1, round(width * scale)), max(1, round(height * scale))),
            resample=interpolation,
        )
        return reduced.resize((width, height), resample=interpolation)

    def _noise(self, image: Image.Image, rng: random.Random) -> Image.Image:
        sigma = rng.uniform(
            self.augment.gaussian_noise_sigma_min,
            self.augment.gaussian_noise_sigma_max,
        )
        array = np.asarray(image, dtype=np.float32) / 255.0
        generator = np.random.default_rng(rng.getrandbits(64))
        noisy = np.clip(array + generator.normal(0.0, sigma, array.shape), 0.0, 1.0)
        return Image.fromarray(np.round(noisy * 255.0).astype(np.uint8), mode="RGB")

    def _color(self, image: Image.Image, rng: random.Random) -> Image.Image:
        strength = self.augment.color_jitter_strength
        operations: list[tuple[Callable[[Image.Image], ImageEnhance._Enhance], float]] = [
            (ImageEnhance.Brightness, rng.uniform(1 - strength, 1 + strength)),
            (ImageEnhance.Contrast, rng.uniform(1 - strength, 1 + strength)),
            (ImageEnhance.Color, rng.uniform(1 - strength, 1 + strength)),
        ]
        rng.shuffle(operations)
        for enhancer, factor in operations:
            image = enhancer(image).enhance(factor)
        return image

    def _center_crop(self, image: Image.Image, rng: random.Random) -> Image.Image:
        fraction = rng.uniform(
            self.augment.center_crop_min_fraction,
            self.augment.center_crop_max_fraction,
        )
        width, height = image.size
        crop_width, crop_height = max(1, round(width * fraction)), max(1, round(height * fraction))
        left, top = (width - crop_width) // 2, (height - crop_height) // 2
        cropped = image.crop((left, top, left + crop_width, top + crop_height))
        return cropped.resize((width, height), resample=self._interpolation(rng))

    def _operation_pool(self) -> list[tuple[float, Callable[[Image.Image, random.Random], Image.Image]]]:
        pool: list[tuple[float, Callable[[Image.Image, random.Random], Image.Image]]] = [
            (1.0, self._jpeg),
            (1.0, self._blur),
            (1.0, self._resize),
            (1.0, self._noise),
            (1.0, self._color),
            (1.0, self._center_crop),
        ]
        if self.augment.enable_double_jpeg:
            pool.append((self.augment.double_jpeg_weight, self._double_jpeg))
        if self.augment.enable_webp:
            pool.append((self.augment.webp_weight, self._webp))
        return pool

    def _degrade(self, image: Image.Image, rng: random.Random) -> Image.Image:
        draw = rng.random()
        if draw < self.augment.transformed_clean_probability:
            operation_count = 0
        elif draw < (
            self.augment.transformed_clean_probability
            + self.augment.single_operation_probability
        ):
            operation_count = 1
        else:
            operation_count = 2
        pool = self._operation_pool()
        available = list(range(len(pool)))
        for _ in range(operation_count):
            weights = [pool[index][0] for index in available]
            chosen = rng.choices(available, weights=weights, k=1)[0]
            image = pool[chosen][1](image, rng)
            available.remove(chosen)
        return image

    @staticmethod
    def _tensor(image: Image.Image) -> torch.Tensor:
        tensor = tvf.pil_to_tensor(image).float().div_(255.0)
        return tvf.normalize(tensor, CLIP_MEAN, CLIP_STD)

    def __call__(self, image: Image.Image, *, seed: int | None = None) -> dict[str, torch.Tensor]:
        rng = random.Random(seed) if seed is not None else random.Random(random.getrandbits(64))
        image = canonical_rgb(image, self.views.padding_color)
        image = self.standardize(image, rng)
        global_geometry, local_geometry = self._geometries(image, rng)
        transformed = self._degrade(image.copy(), rng)
        clean_views = torch.stack(
            [self._tensor(self._render(image, geometry)) for geometry in (global_geometry, local_geometry)]
        )
        transformed_views = torch.stack(
            [
                self._tensor(self._render(transformed, geometry))
                for geometry in (global_geometry, local_geometry)
            ]
        )
        return {"clean_views": clean_views, "transformed_views": transformed_views}
