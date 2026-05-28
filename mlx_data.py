from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Iterator

import mlx.core as mx
import numpy as np
from datasets import load_dataset
from PIL import Image, ImageEnhance, ImageOps


os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"


def _normalize(image: Image.Image, mean: list[float], std: list[float]) -> np.ndarray:
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    return (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)


def _resize_short(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    if width == height:
        return image.resize((size, size), Image.BICUBIC)
    if width < height:
        new_width = size
        new_height = round(size * height / width)
    else:
        new_height = size
        new_width = round(size * width / height)
    return image.resize((new_width, new_height), Image.BICUBIC)


def _center_crop(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    left = max((width - size) // 2, 0)
    top = max((height - size) // 2, 0)
    return image.crop((left, top, left + size, top + size))


def _random_crop(image: Image.Image, size: int, padding: int = 0) -> Image.Image:
    if padding:
        image = ImageOps.expand(image, border=padding, fill=0)
    width, height = image.size
    if width == size and height == size:
        return image
    left = random.randint(0, max(width - size, 0))
    top = random.randint(0, max(height - size, 0))
    return image.crop((left, top, left + size, top + size))


def _rand_augment(image: Image.Image) -> Image.Image:
    op = random.choice(["identity", "autocontrast", "brightness", "contrast", "rotate"])
    if op == "autocontrast":
        return ImageOps.autocontrast(image)
    if op == "brightness":
        return ImageEnhance.Brightness(image).enhance(random.uniform(0.75, 1.25))
    if op == "contrast":
        return ImageEnhance.Contrast(image).enhance(random.uniform(0.75, 1.25))
    if op == "rotate":
        return image.rotate(random.uniform(-15, 15), resample=Image.BICUBIC)
    return image


@dataclass
class ImageDataset:
    data_name: str
    image_size: int
    num_labels: int
    batch_size: int
    eval_batch_size: int
    num_workers: int
    add_rand_aug: bool
    train_key: str = "train"
    val_key: str | None = "validation"
    test_key: str | None = "test"
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    image_col: str = "image"
    label_col: str = "label"

    def setup(self):
        data = load_dataset(self.data_name)
        if self.data_name == "uoft-cs/cifar100":
            data = data.remove_columns(["coarse_label"])
            data = data.rename_columns({"img": "image", "fine_label": "label"})
        self.datasets = data

    def _transform_train(self, image: Image.Image) -> np.ndarray:
        image = image.convert("RGB")
        if self.image_size == 32:
            image = image.resize((self.image_size, self.image_size), Image.BICUBIC)
            image = _random_crop(image, self.image_size, padding=4)
        else:
            image = _resize_short(image, self.image_size)
            image = _random_crop(image, self.image_size)
        if random.random() < 0.5:
            image = ImageOps.mirror(image)
        if self.add_rand_aug:
            image = _rand_augment(image)
        return _normalize(image, list(self.mean), list(self.std))

    def _transform_eval(self, image: Image.Image) -> np.ndarray:
        image = _resize_short(image.convert("RGB"), self.image_size)
        image = _center_crop(image, self.image_size)
        return _normalize(image, list(self.mean), list(self.std))

    def _batch_iter(self, split: str, batch_size: int, train: bool) -> Iterator[dict]:
        dataset = self.datasets[split]
        indices = list(range(len(dataset)))
        if train:
            random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            if train and len(batch_indices) < batch_size:
                continue
            images = []
            labels = []
            for idx in batch_indices:
                example = dataset[idx]
                transform = self._transform_train if train else self._transform_eval
                images.append(transform(example[self.image_col]))
                labels.append(example[self.label_col])
            yield {
                "pixel_values": mx.array(np.stack(images), dtype=mx.float32),
                "labels": mx.array(np.asarray(labels), dtype=mx.int32),
            }

    def train_batches(self):
        return self._batch_iter(self.train_key, self.batch_size, train=True)

    def val_batches(self):
        if self.val_key is None:
            return None
        return self._batch_iter(self.val_key, self.eval_batch_size, train=False)

    def test_batches(self):
        if self.test_key is None:
            return None
        return self._batch_iter(self.test_key, self.eval_batch_size, train=False)

    def num_train_batches(self) -> int:
        return len(self.datasets[self.train_key]) // self.batch_size


def load_data(args):
    kwargs = {
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "num_workers": args.num_workers or os.cpu_count() or 1,
        "add_rand_aug": args.add_rand_aug,
    }
    if args.data_name == "cifar100":
        data = ImageDataset(
            data_name="uoft-cs/cifar100",
            image_size=32,
            num_labels=100,
            mean=(0.5071, 0.4867, 0.4408),
            std=(0.2675, 0.2565, 0.2761),
            val_key=None,
            test_key="test",
            **kwargs,
        )
    elif args.data_name == "tiny-imagenet":
        data = ImageDataset(
            data_name="zh-plus/tiny-imagenet",
            image_size=64,
            num_labels=200,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
            val_key="valid",
            test_key="valid",
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid data name: {args.data_name}")
    data.setup()
    return data
