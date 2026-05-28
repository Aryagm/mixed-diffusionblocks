from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from statistics import NormalDist
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np

from llm_dblocks.adapters import ModelAdapter


_NORMAL = NormalDist()


def scalar(x) -> float:
    return float(x.item())


@dataclass
class DBlockTrainingConfig:
    num_blocks: int = 3
    lr: float = 1e-4
    weight_decay: float = 0.01
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    p_mean: float = -1.2
    p_std: float = 1.2
    gamma: float = 0.05
    aux_lm_weight: float = 0.1
    clean_lm_weight: float = 0.0
    gradient_clip_norm: float = 1.0
    objective: str = "paper_ar"
    sigma_data: float = 0.5


def block_ranges(num_layers: int, num_blocks: int) -> list[tuple[int, int]]:
    if num_blocks < 1:
        raise ValueError("num_blocks must be >= 1")
    if num_blocks > num_layers:
        raise ValueError("num_blocks cannot exceed num_layers")
    ranges = []
    for i in range(num_blocks):
        start = round(i * num_layers / num_blocks)
        end = round((i + 1) * num_layers / num_blocks)
        ranges.append((start, end))
    return ranges


def block_sigmas(config: DBlockTrainingConfig) -> list[float]:
    cdf_min = _NORMAL.cdf((np.log(config.sigma_min) - config.p_mean) / config.p_std)
    cdf_max = _NORMAL.cdf((np.log(config.sigma_max) - config.p_mean) / config.p_std)
    sigmas = []
    for i in range(config.num_blocks + 1):
        p = cdf_min + (cdf_max - cdf_min) * (i / config.num_blocks)
        sigmas.append(float(np.exp(config.p_mean + config.p_std * _NORMAL.inv_cdf(p))))
    return sigmas


def ar_concat_mask(seq_len: int) -> mx.array:
    """Allowed-attention mask for [clean tokens, noisy tokens].

    Clean rows use standard causal attention over clean columns. Noisy token i
    can attend to clean tokens < i and its own noisy embedding, which implements
    D(z_i_sigma, z_<i, sigma) without leaking clean future tokens.
    """
    total = 2 * seq_len
    rows = np.arange(total)[:, None]
    cols = np.arange(total)[None, :]
    allowed = np.zeros((total, total), dtype=bool)

    clean_rows = rows < seq_len
    clean_cols = cols < seq_len
    allowed |= clean_rows & clean_cols & (cols <= rows)

    noisy_i = rows - seq_len
    noisy_rows = rows >= seq_len
    allowed |= noisy_rows & clean_cols & (cols < noisy_i)
    allowed |= noisy_rows & (cols == rows)
    return mx.array(allowed)


def discrete_sigmas(
    num_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
) -> mx.array:
    ramp = np.linspace(0, 1, num_steps, dtype=np.float32)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return mx.array(sigmas.astype(np.float32))


class DBlockTrainer:
    def __init__(self, adapter: ModelAdapter, config: DBlockTrainingConfig):
        self.adapter = adapter
        self.config = config
        self.ranges = block_ranges(adapter.num_layers, config.num_blocks)
        self.sigmas = block_sigmas(config)
        self.adapter.freeze_for_dblocks()
        self.optimizers = [
            optim.AdamW(learning_rate=config.lr, weight_decay=config.weight_decay)
            for _ in range(config.num_blocks)
        ]
        mx.eval(self.adapter.model.parameters())

    def set_trainable_block(self, block_idx: int):
        start, end = self.ranges[block_idx]
        self.adapter.model.freeze()
        for layer in self.adapter.layers[start:end]:
            layer.unfreeze()

    def _sample_sigmas(self, batch_size: int, block_idx: int) -> mx.array:
        sigma_min = self.sigmas[block_idx]
        sigma_max = self.sigmas[block_idx + 1]
        if self.config.gamma > 0.0:
            log_min = np.log(sigma_min)
            log_max = np.log(sigma_max)
            width = log_max - log_min
            sigma_min = max(np.exp(log_min - self.config.gamma * width), self.sigmas[0])
            sigma_max = min(np.exp(log_max + self.config.gamma * width), self.sigmas[-1])

        cdf_min = _NORMAL.cdf((np.log(sigma_min) - self.config.p_mean) / self.config.p_std)
        cdf_max = _NORMAL.cdf((np.log(sigma_max) - self.config.p_mean) / self.config.p_std)
        rand = np.random.uniform(cdf_min, cdf_max, batch_size)
        values = np.exp(
            [
                self.config.p_mean + self.config.p_std * _NORMAL.inv_cdf(float(v))
                for v in rand
            ]
        ).astype(np.float32)
        return mx.array(values, dtype=mx.float32)[:, None, None]

    def normalize_embeddings(self, x: mx.array) -> mx.array:
        denom = mx.sqrt(mx.sum(x * x, axis=-1, keepdims=True))
        return x / mx.maximum(denom, mx.array(1e-12, dtype=x.dtype))

    def edm_coefficients(self, sigmas: mx.array):
        sigma_data = self.config.sigma_data
        c_skip = sigma_data**2 / (sigmas**2 + sigma_data**2)
        c_out = sigmas * sigma_data / mx.sqrt(sigmas**2 + sigma_data**2)
        c_in = 1 / mx.sqrt(sigmas**2 + sigma_data**2)
        weights = (sigmas**2 + sigma_data**2) / (sigmas * sigma_data) ** 2
        return c_skip, c_out, c_in, weights

    def clean_next_token_loss(self, model, batch: dict[str, mx.array]) -> mx.array:
        logits = model(batch["input_ids"])
        return nn.losses.cross_entropy(
            logits.reshape(-1, self.adapter.vocab_size),
            batch["labels"].reshape(-1),
            reduction="mean",
        )

    def loss(self, model, batch: dict[str, mx.array], block_idx: int | None = None):
        if self.config.objective == "hidden":
            return self.hidden_state_loss(model, batch, block_idx=block_idx)
        if self.config.objective == "paper_ar":
            return self.paper_ar_loss(model, batch, block_idx=block_idx)
        raise ValueError(f"Unknown objective: {self.config.objective}")

    def hidden_state_loss(
        self, model, batch: dict[str, mx.array], block_idx: int | None = None
    ):
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        if block_idx is None:
            block_idx = random.randrange(self.config.num_blocks)
        start, end = self.ranges[block_idx]

        hidden = self.adapter.embed(input_ids)
        prefix = self.adapter.run_layers(hidden, 0, start)
        prefix = mx.stop_gradient(prefix)

        clean_target = self.adapter.run_layers(prefix, start, end)
        clean_target = mx.stop_gradient(clean_target)

        sigmas = self._sample_sigmas(prefix.shape[0], block_idx)
        noisy_prefix = prefix + sigmas * mx.random.normal(prefix.shape, dtype=prefix.dtype)
        denoised = self.adapter.run_layers(noisy_prefix, start, end)

        weights = (sigmas**2 + 0.5**2) / (sigmas * 0.5) ** 2
        denoise_loss = mx.mean(weights * (denoised - clean_target) ** 2)

        logits = self.adapter.logits_from_hidden(denoised)
        aux_lm_loss = nn.losses.cross_entropy(
            logits.reshape(-1, self.adapter.vocab_size),
            labels.reshape(-1),
            reduction="mean",
        )
        if self.config.clean_lm_weight > 0:
            clean_lm_loss = self.clean_next_token_loss(model, batch)
        else:
            clean_lm_loss = mx.array(0.0, dtype=denoise_loss.dtype)
        loss = (
            denoise_loss
            + self.config.aux_lm_weight * aux_lm_loss
            + self.config.clean_lm_weight * clean_lm_loss
        )
        return loss, {
            "loss": loss,
            "denoise_loss": denoise_loss,
            "aux_lm_loss": aux_lm_loss,
            "clean_lm_loss": clean_lm_loss,
            "block": mx.array(block_idx),
        }

    def paper_ar_loss(
        self, model, batch: dict[str, mx.array], block_idx: int | None = None
    ):
        input_ids = batch["input_ids"]
        if block_idx is None:
            block_idx = random.randrange(self.config.num_blocks)
        start, end = self.ranges[block_idx]

        clean = self.normalize_embeddings(self.adapter.embed(input_ids))
        sigmas = self._sample_sigmas(clean.shape[0], block_idx).astype(clean.dtype)
        noisy = clean + sigmas * mx.random.normal(clean.shape, dtype=clean.dtype)
        c_skip, c_out, c_in, weights = self.edm_coefficients(sigmas)
        hidden = mx.concatenate([clean, noisy * c_in], axis=1)
        mask = ar_concat_mask(input_ids.shape[1])

        hidden = self.adapter.run_layers(hidden, 0, start, mask=mask)
        hidden = mx.stop_gradient(hidden)
        hidden = self.adapter.run_layers(hidden, start, end, mask=mask)

        noisy_hidden = hidden[:, input_ids.shape[1] :, :]
        model_out = noisy_hidden * c_out + noisy * c_skip
        noisy_logits = self.adapter.logits_from_hidden(model_out)
        losses = nn.losses.cross_entropy(
            noisy_logits[:, 1:, :].reshape(-1, self.adapter.vocab_size),
            input_ids[:, 1:].reshape(-1),
            reduction="none",
        )
        losses = losses.reshape(input_ids.shape[0], -1)
        token_loss = mx.mean(losses * weights.reshape(-1, 1))
        ce_loss = mx.mean(losses)
        if self.config.clean_lm_weight > 0:
            clean_lm_loss = self.clean_next_token_loss(model, batch)
        else:
            clean_lm_loss = mx.array(0.0, dtype=token_loss.dtype)
        loss = token_loss + self.config.clean_lm_weight * clean_lm_loss
        return loss, {
            "loss": loss,
            "denoise_loss": token_loss,
            "aux_lm_loss": ce_loss,
            "clean_lm_loss": clean_lm_loss,
            "block": mx.array(block_idx),
        }

    def block_for_sigma(self, sigma: mx.array) -> int:
        sigma_value = float(sigma.item())
        idx = np.searchsorted(self.sigmas, sigma_value, side="right") - 1
        idx = int(np.clip(idx, 0, self.config.num_blocks - 1))
        return idx

    def paper_ar_logits_from_noisy(
        self,
        input_ids: mx.array,
        noisy: mx.array,
        sigma: mx.array,
        block_idx: int,
    ) -> mx.array:
        clean = self.normalize_embeddings(self.adapter.embed(input_ids))
        c_skip, c_out, c_in, _ = self.edm_coefficients(sigma)
        hidden = mx.concatenate([clean, noisy * c_in], axis=1)
        mask = ar_concat_mask(input_ids.shape[1])
        start, end = self.ranges[block_idx]
        hidden = self.adapter.run_layers(hidden, 0, start, mask=mask)
        hidden = self.adapter.run_layers(hidden, start, end, mask=mask)
        noisy_hidden = hidden[:, input_ids.shape[1] :, :]
        model_out = noisy_hidden * c_out + noisy * c_skip
        return self.adapter.logits_from_hidden(model_out)

    def diffusion_reconstruct_logits(
        self,
        input_ids: mx.array,
        *,
        num_steps: int = 8,
    ) -> mx.array:
        """Teacher-forced DiffusionBlocks reconstruction logits.

        This is not free-running generation. It mirrors language-model CE by
        conditioning on clean previous tokens, but reconstructs each noisy token
        through the DiffusionBlocks denoising process before scoring it.
        """
        clean = self.normalize_embeddings(self.adapter.embed(input_ids))
        sigmas = discrete_sigmas(
            num_steps,
            sigma_min=self.config.sigma_min,
            sigma_max=self.config.sigma_max,
        )
        z = mx.random.normal(clean.shape, dtype=clean.dtype) * sigmas[0].astype(clean.dtype)
        embedding_weight = self.normalize_embeddings(self.adapter.embedding_weight())

        for i in range(num_steps - 1):
            sigma = sigmas[i]
            next_sigma = sigmas[i + 1]
            block_idx = self.block_for_sigma(sigma)
            sigma_batch = mx.full((input_ids.shape[0], 1, 1), sigma.item(), dtype=z.dtype)
            logits = self.paper_ar_logits_from_noisy(input_ids, z, sigma_batch, block_idx)
            probs = mx.softmax(logits, axis=-1)
            denoised = probs @ embedding_weight.astype(probs.dtype)
            d = (z - denoised) / sigma.astype(z.dtype)
            z = z + (next_sigma - sigma).astype(z.dtype) * d

        block_idx = self.block_for_sigma(sigmas[-1])
        sigma_batch = mx.full((input_ids.shape[0], 1, 1), sigmas[-1].item(), dtype=z.dtype)
        return self.paper_ar_logits_from_noisy(input_ids, z, sigma_batch, block_idx)

    def diffusion_reconstruction_ce(
        self,
        batch: dict[str, mx.array],
        *,
        num_steps: int = 8,
    ) -> mx.array:
        logits = self.diffusion_reconstruct_logits(batch["input_ids"], num_steps=num_steps)
        return nn.losses.cross_entropy(
            logits[:, 1:, :].reshape(-1, self.adapter.vocab_size),
            batch["input_ids"][:, 1:].reshape(-1),
            reduction="mean",
        )

    def train(self, batches, *, iters: int, log_every: int = 10):
        def loss_fn(model, batch, block_idx):
            loss, _ = self.loss(model, batch, block_idx=block_idx)
            return loss

        loss_and_grad = nn.value_and_grad(self.adapter.model, loss_fn)
        totals = {
            "loss": 0.0,
            "denoise_loss": 0.0,
            "aux_lm_loss": 0.0,
            "clean_lm_loss": 0.0,
        }
        start_time = time.perf_counter()
        last_time = start_time

        for step in range(1, iters + 1):
            batch = next(batches)
            block_idx = random.randrange(self.config.num_blocks)
            self.set_trainable_block(block_idx)
            optimizer = self.optimizers[block_idx]

            loss, grads = loss_and_grad(self.adapter.model, batch, block_idx)
            if self.config.gradient_clip_norm > 0:
                grads, _ = optim.clip_grad_norm(grads, self.config.gradient_clip_norm)
            optimizer.update(self.adapter.model, grads)
            metrics = self.loss(self.adapter.model, batch, block_idx=block_idx)[1]
            mx.eval(self.adapter.model.parameters(), optimizer.state, *metrics.values())

            for key in totals:
                totals[key] += scalar(metrics[key])

            if step % log_every == 0 or step == iters:
                now = time.perf_counter()
                denom = log_every if step % log_every == 0 else step % log_every
                denom = max(denom, 1)
                summary = {key: value / denom for key, value in totals.items()}
                steps_s = denom / max(now - last_time, 1e-9)
                print(
                    f"step={step} loss={summary['loss']:.4f} "
                    f"denoise={summary['denoise_loss']:.4f} "
                    f"aux_lm={summary['aux_lm_loss']:.4f} "
                    f"clean_lm={summary['clean_lm_loss']:.4f} "
                    f"steps_s={steps_s:.2f}"
                )
                totals = {key: 0.0 for key in totals}
                last_time = now

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        weights_path = os.path.join(output_dir, "dblock_trainable.safetensors")
        params = dict(tree_flatten(self.adapter.model.parameters()))
        trainable = {
            key: value
            for key, value in params.items()
            if ".layers." in f".{key}." or key.startswith("layers.")
        }
        mx.save_safetensors(
            weights_path,
            trainable,
            metadata={"format": "diffusionblocks-mlx-trainable"},
        )
        with open(os.path.join(output_dir, "dblock_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "num_layers": self.adapter.num_layers,
                    "block_ranges": self.ranges,
                    "training": self.config.__dict__,
                },
                f,
                indent=2,
            )
        return weights_path
