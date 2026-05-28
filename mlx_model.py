from __future__ import annotations

import random
from statistics import NormalDist

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_dblock_modules import get_block_sigmas, get_discrete_sigmas
from mlx_vit import load_vit


_NORMAL = NormalDist()


def load_model(args):
    if args.model_type == "vit":
        return ViTModel(args)
    if args.model_type == "dblock":
        return ViTDBlockModel(args)
    raise ValueError(f"Invalid model type: {args.model_type}")


def accuracy(logits: mx.array, labels: mx.array) -> mx.array:
    return mx.mean(mx.argmax(logits, axis=-1) == labels)


def macro_f1(logits: mx.array, labels: mx.array, num_labels: int) -> mx.array:
    preds = mx.argmax(logits, axis=-1)
    f1s = []
    for cls in range(num_labels):
        pred_is_cls = preds == cls
        label_is_cls = labels == cls
        tp = mx.sum(pred_is_cls & label_is_cls)
        fp = mx.sum(pred_is_cls & ~label_is_cls)
        fn = mx.sum(~pred_is_cls & label_is_cls)
        denom = 2 * tp + fp + fn
        f1s.append(mx.where(denom > 0, (2 * tp) / denom, mx.array(0.0)))
    return mx.mean(mx.stack(f1s))


class ViTModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.image_size = args.image_size
        self.num_labels = args.num_labels
        self.model = load_vit(image_size=self.image_size, num_labels=self.num_labels)

    def __call__(self, pixel_values: mx.array) -> mx.array:
        return self.model(pixel_values)

    def loss(self, batch):
        logits = self(batch["pixel_values"])
        labels = batch["labels"]
        loss = nn.losses.cross_entropy(logits, labels, reduction="mean")
        return loss, {"loss": loss}

    def eval_step(self, batch):
        logits = self(batch["pixel_values"])
        labels = batch["labels"]
        return {
            "loss": nn.losses.cross_entropy(logits, labels, reduction="mean"),
            "acc": accuracy(logits, labels),
            "f1": macro_f1(logits, labels, self.num_labels),
        }


class ViTDBlockModel(ViTModel):
    def __init__(self, args):
        nn.Module.__init__(self)
        self.args = args
        self.image_size = args.image_size
        self.num_labels = args.num_labels
        self.gamma = args.gamma
        self.sigma_data = 0.5
        self.cfg_scale = args.cfg_scale
        self.class_dropout_prob = (
            args.class_dropout_prob if self.cfg_scale > 0.0 else 0.0
        )
        self.num_inference_steps = args.num_inference_steps or args.num_blocks
        self.block_sigmas = get_block_sigmas(num_layers=args.num_blocks)
        self.sigmas = get_discrete_sigmas(
            num_steps=self.num_inference_steps, dblock=True
        )
        self.model = load_vit(
            image_size=self.image_size, num_labels=self.num_labels, is_dblock=True
        )

    def normalize_embeddings(self, x: mx.array) -> mx.array:
        denom = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return x / mx.maximum(denom, mx.array(1e-12, dtype=x.dtype))

    def get_embeds(self, input_ids: mx.array) -> mx.array:
        embeds = self.model.get_input_embeddings()(input_ids)
        return self.normalize_embeddings(embeds)

    def get_sigmas_np(
        self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2
    ) -> np.ndarray:
        block_idx = random.choices(range(self.args.num_blocks), k=1)[0]
        sigma_min_block = self.block_sigmas[block_idx]
        sigma_max_block = self.block_sigmas[block_idx + 1]
        if self.gamma > 0.0:
            log_sigma_min = np.log(sigma_min_block)
            log_sigma_max = np.log(sigma_max_block)
            log_range = log_sigma_max - log_sigma_min
            sigma_min_block = np.exp(log_sigma_min - self.gamma * log_range)
            sigma_max_block = np.exp(log_sigma_max + self.gamma * log_range)
            sigma_min_block = max(sigma_min_block, self.block_sigmas[0])
            sigma_max_block = min(sigma_max_block, self.block_sigmas[-1])

        cdf_min_block = _NORMAL.cdf((np.log(sigma_min_block) - p_mean) / p_std)
        cdf_max_block = _NORMAL.cdf((np.log(sigma_max_block) - p_mean) / p_std)
        rand = np.random.uniform(cdf_min_block, cdf_max_block, n_samples)
        sigma = np.exp([p_mean + p_std * _NORMAL.inv_cdf(float(v)) for v in rand])
        return sigma.astype(np.float32)

    def get_sigmas(self, n_samples: int, p_mean: float = -1.2, p_std: float = 1.2):
        return mx.array(self.get_sigmas_np(n_samples, p_mean, p_std), dtype=mx.float32)

    def get_weights(self, sigmas: mx.array) -> mx.array:
        return (sigmas**2 + self.sigma_data**2) / (sigmas * self.sigma_data) ** 2

    def estimate_target_layer_np(self, sigma_np: np.ndarray) -> int:
        block_idx = np.searchsorted(self.block_sigmas, sigma_np, side="right") - 1
        block_idx = (self.args.num_blocks - 1) - block_idx
        block_idx = np.clip(block_idx, 0, self.args.num_blocks - 1).astype(np.int64)
        values, counts = np.unique(block_idx, return_counts=True)
        return int(values[counts.argmax()])

    def estimate_target_layer(self, sigma: mx.array) -> int:
        return self.estimate_target_layer_np(np.array(sigma))

    def _layer_assignment(self):
        split_size = self.model.config.num_hidden_layers // self.args.num_blocks
        return [
            list(range(i * split_size, (i + 1) * split_size))
            for i in range(self.args.num_blocks)
        ]

    def denoise(self, x, zt, sigma, block_idx=None):
        if block_idx is None:
            block_idx = self.estimate_target_layer(sigma)

        if self.training and self.class_dropout_prob > 0.0:
            drop_x = mx.random.uniform(shape=(x.shape[0],)) < self.class_dropout_prob
            x = mx.where(drop_x[:, None, None, None], mx.zeros_like(x), x)
        elif not self.training and self.cfg_scale > 0.0:
            x = mx.concatenate([mx.zeros_like(x), x], axis=0)
            zt = mx.concatenate([zt, zt], axis=0)
            sigma = mx.concatenate([sigma, sigma], axis=0)

        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / mx.sqrt(sigma**2 + self.sigma_data**2)
        c_in = 1 / mx.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = 0.25 * mx.log(sigma)

        outputs = self.model.forward_block(
            layer_indices=self._layer_assignment()[block_idx],
            pixel_values=x,
            noisy_embeds=zt * c_in[:, None],
            timesteps=c_noise,
        )
        hidden_states = outputs.last_hidden_state
        model_out = hidden_states * c_out[:, None] + zt * c_skip[:, None]
        logits = self.model.forward_output_embeddings(
            mx.expand_dims(model_out, 1), outputs.conditioning
        )
        if not self.training and self.cfg_scale > 0.0:
            logits_uncond, logits_cond = mx.split(logits, 2, axis=0)
            logits = logits_uncond + self.cfg_scale * (logits_cond - logits_uncond)
        return logits

    def loss(self, batch):
        pixel_values = batch["pixel_values"]
        labels = batch["labels"]
        z = self.get_embeds(labels)
        sigmas_np = self.get_sigmas_np(z.shape[0])
        block_idx = self.estimate_target_layer_np(sigmas_np)
        sigmas = mx.array(sigmas_np, dtype=z.dtype)
        zt = z + sigmas[:, None] * mx.random.normal(z.shape, dtype=z.dtype)
        logits = self.denoise(pixel_values, zt, sigmas, block_idx)
        losses = nn.losses.cross_entropy(logits, labels, reduction="none")
        ce_loss = mx.mean(losses)
        weights = self.get_weights(sigmas)
        loss = mx.mean(losses * weights)
        return loss, {
            "loss": loss,
            f"loss_{block_idx}": loss,
            "ce_loss": ce_loss,
            f"ce_loss_{block_idx}": ce_loss,
        }

    def diffusion_step(self, x):
        bsz = x.shape[0]
        hidden_size = self.model.config.hidden_size
        z = mx.random.normal((bsz, hidden_size), dtype=mx.float32)
        z = z * mx.sqrt(1.0 + self.sigmas[0] ** 2.0)
        s_in = mx.ones((x.shape[0],), dtype=mx.float32)
        for i in range(self.sigmas.shape[0] - 1):
            sigma = self.sigmas[i] * s_in
            next_sigma = self.sigmas[i + 1] * s_in
            logits = self.denoise(x, z, sigma)
            probs = mx.softmax(logits, axis=1)
            denoised = probs @ self.model.get_input_embeddings().weight
            d = (z - denoised) / sigma[:, None]
            dt = next_sigma - sigma
            z = z + dt[:, None] * d
        sigmas = mx.full((x.shape[0],), self.sigmas[-1].item(), dtype=mx.float32)
        return self.denoise(x, z, sigmas)

    def eval_step(self, batch):
        logits = self.diffusion_step(batch["pixel_values"])
        labels = batch["labels"]
        return {
            "loss": nn.losses.cross_entropy(logits, labels, reduction="mean"),
            "acc": accuracy(logits, labels),
            "f1": macro_f1(logits, labels, self.num_labels),
        }
