from __future__ import annotations

import argparse
import datetime
import json
import os
import random

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from mlx_data import load_data
from mlx_model import load_model


def to_float(x) -> float:
    return float(np.array(x))


def make_lr_schedule(args, total_steps: int):
    base_lr = args.lr
    warmup = args.num_warmup_steps
    if args.scheduler_type == "constant":
        return base_lr
    if args.scheduler_type == "constant_with_warmup":
        if warmup <= 0:
            return base_lr

        def schedule(step):
            step = mx.minimum(step, warmup)
            return base_lr * step / warmup

        return schedule
    if args.scheduler_type == "cosine_with_min_lr":
        min_lr = 0.0
        if args.scheduler_specific_kwargs is not None:
            min_lr = args.scheduler_specific_kwargs.get("min_lr", min_lr)

        def schedule(step):
            warm = mx.minimum(step / max(warmup, 1), 1.0)
            progress = mx.clip((step - warmup) / max(total_steps - warmup, 1), 0, 1)
            cosine = 0.5 * (1.0 + mx.cos(mx.array(np.pi) * progress))
            return warm * (min_lr + (base_lr - min_lr) * cosine)

        return schedule
    raise ValueError(f"Unsupported scheduler_type for MLX: {args.scheduler_type}")


def make_optimizer(args, total_steps: int):
    lr = make_lr_schedule(args, total_steps)
    if args.optimizer == "adamw":
        return optim.AdamW(learning_rate=lr, weight_decay=args.weight_decay)
    if args.optimizer == "adam":
        return optim.Adam(learning_rate=lr)
    raise ValueError(f"Unsupported optimizer for MLX: {args.optimizer}")


def evaluate(model, batches, prefix: str, max_batches: int | None = None):
    if batches is None:
        return {}
    model.eval()
    totals = {"loss": 0.0, "acc": 0.0, "f1": 0.0}
    count = 0
    for batch in batches:
        metrics = model.eval_step(batch)
        mx.eval(*metrics.values())
        for key in totals:
            totals[key] += to_float(metrics[key])
        count += 1
        if max_batches is not None and count >= max_batches:
            break
    if count == 0:
        return {}
    return {f"{prefix}/{key}": value / count for key, value in totals.items()}


def save_checkpoint(model, logdir: str, name: str):
    os.makedirs(logdir, exist_ok=True)
    path = os.path.join(logdir, name)
    model.save_weights(path)
    return path


def train(args, data, model, logdir: str):
    epochs = args.num_epochs
    if args.model_type == "dblock":
        epochs *= args.num_blocks
    total_steps = max(epochs * data.num_train_batches(), 1)
    optimizer = make_optimizer(args, total_steps)

    def loss_fn(model, batch):
        loss, _ = model.loss(batch)
        return loss

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    best_score = -float("inf")
    global_step = 0

    mx.eval(model.parameters())
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_count = 0
        for batch in data.train_batches():
            loss, grads = loss_and_grad_fn(model, batch)
            if args.gradient_clip_val is not None and args.gradient_clip_val > 0:
                grads, _ = optim.clip_grad_norm(grads, args.gradient_clip_val)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            train_loss += to_float(loss)
            train_count += 1
            global_step += 1
            if args.debug and global_step >= 2:
                break

        train_loss /= max(train_count, 1)
        print(f"epoch={epoch + 1} step={global_step} train/loss={train_loss:.4f}")

        should_eval = (epoch + 1) % args.save_every_n_epochs == 0 or epoch == epochs - 1
        if should_eval:
            val_metrics = evaluate(
                model, data.val_batches(), "val", max_batches=2 if args.debug else None
            )
            if val_metrics:
                print(" ".join(f"{k}={v:.4f}" for k, v in val_metrics.items()))
                score = val_metrics["val/acc"]
                if score > best_score:
                    best_score = score
                    save_checkpoint(model, logdir, "best.npz")
            elif data.val_key is None:
                save_checkpoint(model, logdir, f"epoch-{epoch + 1}.npz")
            save_checkpoint(model, logdir, "last.npz")

        if args.debug:
            break


def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    data = load_data(args)
    args.image_size = data.image_size
    args.num_labels = data.num_labels
    model = load_model(args)
    if args.ckpt_path is not None:
        model.load_weights(args.ckpt_path)
        nowname = os.path.basename(os.path.dirname(args.ckpt_path))
    else:
        now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
        nowname = now + f"-mlx-{args.model_type}" + args.postfix
    logdir = os.path.join("logs", nowname)
    print("Experiment Name:", nowname)

    if args.stage == "train":
        train(args, data, model, logdir)
        test_metrics = evaluate(
            model, data.test_batches(), "test", max_batches=2 if args.debug else None
        )
        if test_metrics:
            print(" ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))
    else:
        if args.ckpt_path is None:
            raise ValueError("--ckpt_path is required for test")
        test_metrics = evaluate(
            model, data.test_batches(), "test", max_batches=2 if args.debug else None
        )
        print(" ".join(f"{k}={v:.4f}" for k, v in test_metrics.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", type=str, default="train", choices=["train", "test"])
    parser.add_argument("data_name", type=str, default="cifar100")
    parser.add_argument(
        "--model_type", type=str, default="vit", choices=["vit", "dblock"]
    )
    parser.add_argument("--num_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--add_rand_aug", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)
    parser.add_argument("--scheduler_type", type=str, default="constant_with_warmup")
    parser.add_argument(
        "--scheduler_specific_kwargs",
        type=json.loads,
        default=None,
        help="scheduler-specific kwargs, e.g. '{\"min_lr\": 5e-5}'",
    )
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam"])
    parser.add_argument("--num_warmup_steps", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--ckpt_path", type=str, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--postfix", type=str, default="")
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--num_blocks", type=int, default=3)
    parser.add_argument("--gamma", type=float, default=0.05)
    parser.add_argument("--num_inference_steps", type=int, default=None)
    parser.add_argument("--cfg_scale", type=float, default=0.0)
    parser.add_argument("--class_dropout_prob", type=float, default=0.0)
    args = parser.parse_args()
    main(args)
