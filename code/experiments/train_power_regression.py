from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V3_Small_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v3_small,
    resnet18,
)


CAPACITY_KW = 15.0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class CachedSkyDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, cache_dir: Path, use_solar: bool, training: bool, ramp_loss_weight: float):
        self.frame = frame.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.use_solar = use_solar
        self.training = training
        self.ramp_loss_weight = float(ramp_loss_weight)
        self._arrays: dict[str, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.frame)

    def _array(self, shard: str) -> np.ndarray:
        if shard not in self._arrays:
            self._arrays[shard] = np.load(
                self.cache_dir / f"{shard}_pixels_128.npy", mmap_mode="r"
            )
        return self._arrays[shard]

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        image = np.asarray(self._array(str(row["shard"]))[int(row["shard_row"])], dtype=np.float32).copy()
        image = torch.from_numpy(image).permute(2, 0, 1).div_(255.0)
        # No geometric augmentation: orientation and solar position are physically meaningful.
        target = torch.tensor(float(row["power_operational_qc_kw"]) / CAPACITY_KW, dtype=torch.float32)
        solar = torch.tensor(float(row["solar_elevation_deg"]) / 90.0, dtype=torch.float32)
        is_ramp = str(row.get("ramp_label", "")) in {"ramp_up", "ramp_down"}
        weight = torch.tensor(self.ramp_loss_weight if is_ramp else 1.0, dtype=torch.float32)
        return image, solar, target, weight, int(row["_row_id"])


class TinyCNN(nn.Module):
    def __init__(self, use_solar: bool):
        super().__init__()
        self.use_solar = use_solar
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.BatchNorm2d(32), nn.SiLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.BatchNorm2d(64), nn.SiLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.BatchNorm2d(128), nn.SiLU(),
            nn.Conv2d(128, 192, 3, stride=2, padding=1), nn.BatchNorm2d(192), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(192 + int(use_solar), 128), nn.SiLU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )

    def forward(self, image: torch.Tensor, solar: torch.Tensor) -> torch.Tensor:
        image = (image - 0.5) / 0.25
        features = self.features(image).flatten(1)
        if self.use_solar:
            features = torch.cat([features, solar[:, None]], dim=1)
        return 1.25 * torch.sigmoid(self.head(features).squeeze(1))


class TorchvisionRegressor(nn.Module):
    def __init__(self, architecture: str, use_solar: bool, pretrained: bool):
        super().__init__()
        self.use_solar = use_solar
        if architecture == "resnet18":
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = resnet18(weights=weights)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif architecture == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = mobilenet_v3_small(weights=weights)
            feature_dim = self.backbone.classifier[0].in_features
            self.backbone.classifier = nn.Identity()
        elif architecture == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = efficientnet_b0(weights=weights)
            feature_dim = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.head = nn.Sequential(
            nn.Linear(feature_dim + int(use_solar), 128), nn.SiLU(), nn.Dropout(0.15), nn.Linear(128, 1)
        )
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", mean)
        self.register_buffer("image_std", std)

    def forward(self, image: torch.Tensor, solar: torch.Tensor) -> torch.Tensor:
        features = self.backbone((image - self.image_mean) / self.image_std)
        if self.use_solar:
            features = torch.cat([features, solar[:, None]], dim=1)
        return 1.25 * torch.sigmoid(self.head(features).squeeze(1))


def metrics(observed_kw: np.ndarray, prediction_kw: np.ndarray) -> dict[str, float | int]:
    residual = prediction_kw - observed_kw
    denominator = np.sum((observed_kw - observed_kw.mean()) ** 2)
    return {
        "n": int(len(observed_kw)),
        "mae_kw": float(np.mean(np.abs(residual))),
        "rmse_kw": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator else float("nan"),
        "mbe_kw": float(np.mean(residual)),
        "nmae_capacity_pct": float(np.mean(np.abs(residual)) / CAPACITY_KW * 100),
    }


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    row_ids, observed, predicted = [], [], []
    for images, solar, targets, _, ids in loader:
        images = images.to(device, non_blocking=True)
        solar = solar.to(device, non_blocking=True)
        output = model(images, solar)
        row_ids.append(ids.numpy())
        observed.append(targets.numpy() * CAPACITY_KW)
        predicted.append(output.float().cpu().numpy() * CAPACITY_KW)
    return np.concatenate(row_ids), np.concatenate(observed), np.concatenate(predicted)


def evaluate_groups(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for group_type, column in [
        ("overall", None), ("ramp_label", "ramp_label"),
        ("camera_color_state", "camera_color_state"), ("month", "month"),
    ]:
        groups = [("all", predictions)] if column is None else predictions.groupby(column, dropna=False)
        for value, group in groups:
            rows.append({
                "group_type": group_type,
                "group_value": str(value),
                **metrics(group["observed_kw"].to_numpy(float), group["prediction_kw"].to_numpy(float)),
            })
    return pd.DataFrame(rows)


def day_block_bootstrap(predictions: pd.DataFrame, replicates: int, seed: int) -> pd.DataFrame:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["image_timestamp"]).dt.date
    dates = np.array(sorted(frame["date"].unique()), dtype=object)
    daily = []
    for day in dates:
        group = frame.loc[frame["date"].eq(day)]
        y = group["observed_kw"].to_numpy(float)
        r = group["prediction_kw"].to_numpy(float) - y
        daily.append([len(y), np.abs(r).sum(), (r**2).sum(), r.sum(), y.sum(), (y**2).sum()])
    daily = np.asarray(daily, dtype=float)
    rng = np.random.default_rng(seed)
    rows = []
    for replicate in range(replicates):
        count = np.bincount(rng.integers(0, len(dates), len(dates)), minlength=len(dates))
        n, sae, sse, sr, sy, sy2 = count @ daily
        denominator = sy2 - sy**2 / n
        rows.append({
            "replicate": replicate, "n": int(n), "mae_kw": sae / n,
            "rmse_kw": np.sqrt(sse / n), "r2": 1 - sse / denominator, "mbe_kw": sr / n,
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--ramp-labels", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--model",
        choices=["tinycnn", "resnet18", "mobilenet_v3_small", "efficientnet_b0"],
        required=True,
    )
    parser.add_argument("--use-solar", action="store_true")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--split-column", default="split_chronological")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--ramp-loss-weight", type=float, default=1.0)
    args = parser.parse_args()

    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features_path, cache_dir, labels_path = Path(args.features), Path(args.cache_dir), Path(args.ramp_labels)
    frame = pd.read_csv(features_path, low_memory=False)
    labels = pd.read_csv(labels_path, usecols=["sample_id", "ramp_label"])
    frame = frame.merge(labels, on="sample_id", how="left", validate="one_to_one")
    frame["_row_id"] = np.arange(len(frame))
    frame["month"] = pd.to_datetime(frame["image_timestamp"]).dt.to_period("M").astype(str)
    frame["camera_color_state"] = np.where(frame["effective_grayscale"].astype(bool), "effective_grayscale", "color")
    if frame["sample_id"].duplicated().any() or frame["power_operational_qc_kw"].isna().any():
        raise ValueError("Duplicate sample IDs or missing regression targets")
    if set(frame[args.split_column].dropna().unique()) != {"train", "validation", "test"}:
        raise ValueError(f"Invalid split column: {args.split_column}")

    subsets = {name: frame.loc[frame[args.split_column].eq(name)].copy() for name in ["train", "validation", "test"]}
    datasets = {
        name: CachedSkyDataset(part, cache_dir, args.use_solar, name == "train", args.ramp_loss_weight)
        for name, part in subsets.items()
    }
    loaders = {
        name: DataLoader(
            dataset, batch_size=args.batch_size, shuffle=name == "train", num_workers=args.workers,
            pin_memory=True, persistent_workers=args.workers > 0, drop_last=False,
            generator=torch.Generator().manual_seed(args.seed),
        ) for name, dataset in datasets.items()
    }

    if args.model == "tinycnn":
        model = TinyCNN(args.use_solar)
    else:
        model = TorchvisionRegressor(args.model, args.use_solar, args.pretrained)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("GPU is required for the deep baseline run")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    criterion = nn.SmoothL1Loss(beta=0.05, reduction="none")
    scaler = torch.amp.GradScaler("cuda")
    history, best_score, best_epoch, stale = [], float("inf"), 0, 0
    checkpoint_path = output / "best_model.pt"
    started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, count = 0.0, 0
        for images, solar, targets, sample_weights, _ in loaders["train"]:
            images = images.to(device, non_blocking=True)
            solar = solar.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_weights = sample_weights.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda"):
                prediction = model(images, solar)
                per_sample_loss = criterion(prediction, targets)
                loss = (per_sample_loss * sample_weights).sum() / sample_weights.sum()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += float(loss.detach()) * len(targets)
            count += len(targets)
        val_ids, val_y, val_p = predict(model, loaders["validation"], device)
        val_metrics = metrics(val_y, val_p)
        val_is_ramp = frame.iloc[val_ids]["ramp_label"].isin(["ramp_up", "ramp_down"]).to_numpy()
        val_ramp_mae = float(np.mean(np.abs(val_p[val_is_ramp] - val_y[val_is_ramp])))
        selection_weights = np.where(val_is_ramp, args.ramp_loss_weight, 1.0)
        val_selection_mae = float(np.average(np.abs(val_p - val_y), weights=selection_weights))
        history.append({
            "epoch": epoch, "train_loss": running_loss / count,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "val_ramp_mae_kw": val_ramp_mae,
            "val_selection_weighted_mae_kw": val_selection_mae,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })
        print(json.dumps(history[-1]), flush=True)
        if val_selection_mae < best_score - 1e-4:
            best_score, best_epoch, stale = val_selection_mae, epoch, 0
            torch.save({
                "model_state_dict": model.state_dict(), "args": vars(args),
                "epoch": epoch, "validation_metrics": val_metrics,
                "validation_ramp_mae_kw": val_ramp_mae,
                "validation_selection_weighted_mae_kw": val_selection_mae,
            }, checkpoint_path)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False, encoding="utf-8-sig")

    metric_rows, prediction_tables = [], []
    for split, loader in loaders.items():
        ids, y, p = predict(model, loader, device)
        prediction = frame.iloc[ids][[
            "sample_id", "image_timestamp", args.split_column, "ramp_label",
            "effective_grayscale", "camera_color_state", "month", "solar_elevation_deg",
        ]].copy()
        prediction["observed_kw"] = y
        prediction["prediction_kw"] = p
        prediction["residual_kw"] = p - y
        prediction["split"] = split
        prediction_tables.append(prediction)
        metric_rows.append({"split": split, **metrics(y, p)})
    predictions = pd.concat(prediction_tables, ignore_index=True)
    predictions.to_csv(output / "predictions.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False, encoding="utf-8-sig")

    test_predictions = predictions.loc[predictions["split"].eq("test")].copy()
    evaluate_groups(test_predictions).to_csv(output / "test_metrics_by_group.csv", index=False, encoding="utf-8-sig")
    bootstrap = day_block_bootstrap(test_predictions, 1000, args.seed)
    bootstrap.to_csv(output / "test_day_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    ci = {}
    for column in ["mae_kw", "rmse_kw", "r2", "mbe_kw"]:
        ci[column] = {
            "lower95": float(bootstrap[column].quantile(0.025)),
            "median": float(bootstrap[column].median()),
            "upper95": float(bootstrap[column].quantile(0.975)),
        }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(test_predictions["observed_kw"], test_predictions["prediction_kw"], s=4, alpha=0.15)
    ax.plot([0, 15], [0, 15], "r--", linewidth=1)
    ax.set(xlabel="Observed power (kW)", ylabel="Predicted power (kW)", title="Chronological test set")
    fig.tight_layout(); fig.savefig(output / "test_scatter.png", dpi=180); plt.close(fig)

    history_frame = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history_frame["epoch"], history_frame["val_mae_kw"], "o-")
    ax.axvline(best_epoch, color="r", linestyle="--", linewidth=1)
    ax.set(xlabel="Epoch", ylabel="Validation MAE (kW)", title="Training selection curve")
    ax.grid(alpha=0.25); fig.tight_layout(); fig.savefig(output / "validation_curve.png", dpi=180); plt.close(fig)

    summary = {
        "schema_version": "1.0-deep-regression",
        "model": args.model,
        "pretrained": bool(args.pretrained),
        "inputs": "image+solar_elevation" if args.use_solar else "image_only",
        "split_column": args.split_column,
        "seed": args.seed,
        "ramp_loss_weight": args.ramp_loss_weight,
        "selection_metric": "validation ramp-weighted MAE using the configured training-only ramp weight",
        "best_selection_weighted_mae_kw": best_score,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "runtime_minutes": (time.time() - started) / 60.0,
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "metrics": metric_rows,
        "test_day_block_bootstrap_ci": ci,
        "input_sha256": {"features": sha256(features_path), "ramp_labels": sha256(labels_path)},
        "scientific_notes": [
            "No geometric augmentation was used because camera orientation and solar position are physical signals.",
            "The primary image-only model receives no timestamp, solar geometry, irradiance, weather, or power history.",
            "Solar elevation fusion is diagnostic and must not be described as image-only.",
            "Model selection uses validation MAE only; the chronological test set is evaluated once after selection.",
            "Ramp weighting uses labels from the training partition only and introduces no inference-time auxiliary input.",
        ],
        "software": {"torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(0)},
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "SHA256SUMS.txt").write_text(
        "\n".join(f"{sha256(path)}  {path.name}" for path in [checkpoint_path, output / "predictions.csv", output / "metrics.csv", output / "summary.json"]) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
