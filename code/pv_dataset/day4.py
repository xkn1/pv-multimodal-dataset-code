from __future__ import annotations

import io
import json
import math
import random
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageOps


def _geometry_masked_array(data: bytes, camera: dict, size: int) -> np.ndarray:
    with Image.open(io.BytesIO(data)) as opened:
        original_width, original_height = opened.size
        opened.draft("RGB", (max(512, size * 2), max(512, size * 2)))
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    sx, sy = width / original_width, height / original_height
    cx = float(camera["optical_center_x_px"]) * sx
    cy = float(camera["optical_center_y_px"]) * sy
    radius = float(camera["sky_radius_px"]) * min(sx, sy)
    box = (round(cx - radius), round(cy - radius), round(cx + radius), round(cy + radius))
    square = image.crop(box).resize((size, size), Image.Resampling.LANCZOS)
    array = np.asarray(square, dtype=np.uint8).copy()
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    mask = (xx - center) ** 2 + (yy - center) ** 2 <= (size / 2.0 - 1.0) ** 2
    array[~mask] = 0
    return array


def _normalized_fisheye_radius(theta: np.ndarray, model: str) -> np.ndarray:
    """Return fisheye radius normalized to the radius at 90 degrees."""
    if model == "equidistant":
        return theta / (math.pi / 2.0)
    if model == "equisolid":
        return np.sin(theta / 2.0) / math.sin(math.pi / 4.0)
    if model == "stereographic":
        return np.tan(theta / 2.0)
    if model == "orthographic":
        return np.sin(theta)
    raise ValueError(f"Unsupported fisheye model: {model}")


def rectilinear_square(
    source: np.ndarray,
    *,
    center_x: float,
    center_y: float,
    radius_90deg: float,
    output_size: int = 224,
    corner_zenith_degrees: float = 72.0,
    fisheye_model: str = "stereographic",
    camera_yaw_degrees: float = 0.0,
) -> np.ndarray:
    """Map a calibrated all-sky fisheye image to a dense square pinhole view.

    ``corner_zenith_degrees`` defines the retained field of view explicitly:
    every corner reaches that zenith angle, so no undefined exterior or black
    padding is presented to the model. ``camera_yaw_degrees`` rotates the world
    azimuth basis into source-image coordinates and can be used for north-up
    standardisation once the camera orientation has been calibrated.
    """
    if source.ndim != 3 or source.shape[2] != 3:
        raise ValueError("source must have shape (height, width, 3)")
    if not 0.0 < corner_zenith_degrees < 90.0:
        raise ValueError("corner_zenith_degrees must be between 0 and 90")
    if radius_90deg <= 0 or output_size < 2:
        raise ValueError("radius_90deg and output_size must be positive")

    axis = np.linspace(-1.0, 1.0, output_size, dtype=np.float64)
    plane_y, plane_x = np.meshgrid(axis, axis, indexing="ij")
    scale = math.tan(math.radians(corner_zenith_degrees)) / math.sqrt(2.0)
    plane_x *= scale
    plane_y *= scale
    theta = np.arctan(np.hypot(plane_x, plane_y))
    azimuth = np.arctan2(plane_y, plane_x) + math.radians(camera_yaw_degrees)
    source_radius = radius_90deg * _normalized_fisheye_radius(theta, fisheye_model)
    sample_x = center_x + source_radius * np.cos(azimuth)
    sample_y = center_y + source_radius * np.sin(azimuth)

    height, width = source.shape[:2]
    if (sample_x.min() < 0 or sample_y.min() < 0
            or sample_x.max() > width - 1 or sample_y.max() > height - 1):
        raise ValueError("projection samples outside the source image")
    x0 = np.floor(sample_x).astype(np.int32)
    y0 = np.floor(sample_y).astype(np.int32)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = sample_x - x0
    wy = sample_y - y0
    image = source.astype(np.float32)
    top = image[y0, x0] * (1.0 - wx[..., None]) + image[y0, x1] * wx[..., None]
    bottom = image[y1, x0] * (1.0 - wx[..., None]) + image[y1, x1] * wx[..., None]
    output = top * (1.0 - wy[..., None]) + bottom * wy[..., None]
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _rectilinear_array(data: bytes, camera: dict, day4: dict, size: int) -> np.ndarray:
    """Decode and geometrically rectify one camera frame using calibrated values."""
    with Image.open(io.BytesIO(data)) as opened:
        original_width, original_height = opened.size
        target = int(day4.get("decode_draft_target", max(512, size * 2)))
        opened.draft("RGB", (target, target))
        image = ImageOps.exif_transpose(opened).convert("RGB")
    source = np.asarray(image, dtype=np.uint8)
    width, height = image.size
    sx, sy = width / original_width, height / original_height
    return rectilinear_square(
        source,
        center_x=float(day4["rectilinear_center_x_px"]) * sx,
        center_y=float(day4["rectilinear_center_y_px"]) * sy,
        radius_90deg=float(day4["rectilinear_radius_90deg_px"]) * min(sx, sy),
        output_size=size,
        corner_zenith_degrees=float(day4["rectilinear_corner_zenith_degrees"]),
        fisheye_model=str(day4["rectilinear_fisheye_model"]),
        camera_yaw_degrees=float(day4.get("rectilinear_camera_yaw_degrees", 0.0)),
    )


def _erode_binary_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    """Conservatively shrink valid sky so obstacle boundary pixels are excluded."""
    result = mask.astype(bool)
    if radius <= 0:
        return result
    padded = np.pad(result, radius, mode="constant", constant_values=False)
    windows = []
    height, width = result.shape
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            windows.append(padded[dy:dy + height, dx:dx + width])
    return np.logical_and.reduce(windows)


def rectilinear_valid_sky_mask(source_mask: np.ndarray, day4: dict, size: int) -> np.ndarray:
    """Project a fixed circular-camera sky mask into the rectified square grid."""
    if source_mask.ndim != 2:
        raise ValueError("source_mask must be a two-dimensional array")
    eroded = _erode_binary_mask(
        source_mask.astype(bool), int(day4.get("rectilinear_mask_erosion_px", 0))
    )
    mask_rgb = np.repeat((eroded.astype(np.uint8) * 255)[..., None], 3, axis=2)
    projected = rectilinear_square(
        mask_rgb,
        center_x=float(day4["rectilinear_mask_center_x_px"]),
        center_y=float(day4["rectilinear_mask_center_y_px"]),
        radius_90deg=float(day4["rectilinear_mask_radius_90deg_px"]),
        output_size=size,
        corner_zenith_degrees=float(day4["rectilinear_corner_zenith_degrees"]),
        fisheye_model=str(day4["rectilinear_fisheye_model"]),
        camera_yaw_degrees=float(day4.get("rectilinear_camera_yaw_degrees", 0.0)),
    )[..., 0]
    # Requiring nearly all bilinear support avoids mixing obstacle boundary pixels.
    result = projected >= 250
    polygons = day4.get("rectilinear_exclusion_polygons_normalized", [])
    if polygons:
        canvas = Image.fromarray(result.astype(np.uint8) * 255)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            points = [
                (round(float(x) * (size - 1)), round(float(y) * (size - 1)))
                for x, y in polygon
            ]
            if len(points) < 3:
                raise ValueError("each exclusion polygon must contain at least three points")
            draw.polygon(points, fill=0)
        result = np.asarray(canvas) >= 128
    return result


def _neutral_fill_invalid(array: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Remove obstacle pixels without pretending to reconstruct missing clouds."""
    if not valid_mask.any():
        raise ValueError("valid sky mask contains no pixels")
    output = array.copy()
    output[~valid_mask] = np.median(array[valid_mask], axis=0).astype(np.uint8)
    return output


def image_statistics(array: np.ndarray, valid_mask: np.ndarray | None = None) -> dict[str, float]:
    size = array.shape[0]
    yy, xx = np.mgrid[:size, :size]
    center = (size - 1) / 2.0
    rr = np.sqrt((xx - center) ** 2 + (yy - center) ** 2) / (size / 2.0)
    if valid_mask is None:
        mask = rr <= 0.98
    else:
        if valid_mask.shape != array.shape[:2]:
            raise ValueError("valid_mask must match the image height and width")
        mask = valid_mask.astype(bool).copy()
    if not mask.any():
        raise ValueError("valid_mask contains no usable pixels")
    rgb = array.astype(np.float32) / 255.0
    pixels = rgb[mask]
    features: dict[str, float] = {}
    for channel, name in enumerate("rgb"):
        values = pixels[:, channel]
        features[f"{name}_mean"] = float(values.mean())
        features[f"{name}_std"] = float(values.std())
        for quantile in (0.10, 0.50, 0.90):
            features[f"{name}_q{int(quantile * 100):02d}"] = float(np.quantile(values, quantile))
    maximum = pixels.max(axis=1)
    minimum = pixels.min(axis=1)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 1e-6)
    channel_range = maximum - minimum
    features["brightness_mean"] = float(maximum.mean())
    features["brightness_std"] = float(maximum.std())
    features["saturation_mean"] = float(saturation.mean())
    features["saturation_q90"] = float(np.quantile(saturation, 0.90))
    features["channel_range_mean"] = float(channel_range.mean())
    features["dark_fraction"] = float((maximum < 0.10).mean())
    features["bright_fraction"] = float((maximum > 0.90).mean())
    features["red_blue_log_ratio"] = float(np.log((pixels[:, 0].mean() + 1e-3) / (pixels[:, 2].mean() + 1e-3)))
    for ring_index, (lower, upper) in enumerate(((0.0, 0.33), (0.33, 0.66), (0.66, 0.98))):
        selected = rgb[mask & (rr >= lower) & (rr < upper)]
        for channel, name in enumerate("rgb"):
            features[f"ring{ring_index}_{name}_mean"] = float(selected[:, channel].mean()) if len(selected) else float("nan")
    angle = (np.arctan2(yy - center, xx - center) + 2 * np.pi) % (2 * np.pi)
    for sector in range(8):
        selected = rgb[mask & (angle >= sector * np.pi / 4) & (angle < (sector + 1) * np.pi / 4)]
        for channel, name in enumerate("rgb"):
            features[f"sector{sector}_{name}_mean"] = float(selected[:, channel].mean()) if len(selected) else float("nan")
    features["effective_grayscale"] = bool(
        features["channel_range_mean"] < 0.012 and features["saturation_q90"] < 0.035
    )
    return features


def polar_unwrap_valid_sky(
    array: np.ndarray,
    valid_mask: np.ndarray,
    output_height: int = 128,
    output_width: int = 256,
    boundary_margin_px: float = 3.0,
) -> np.ndarray:
    """Remap only valid sky pixels to a dense radius-by-azimuth tensor.

    The horizontal dimension is azimuth and is periodic; the vertical dimension
    runs from the optical centre to the last continuously valid pixel on each
    ray. This is an empirical polar remapping, not calibrated fisheye
    undistortion. Invalid exterior pixels are never sampled.
    """
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("array must have shape (height, width, 3)")
    if valid_mask.shape != array.shape[:2]:
        raise ValueError("valid_mask must match the image height and width")
    if output_height < 2 or output_width < 4:
        raise ValueError("polar output is too small")
    mask = valid_mask.astype(bool)
    height, width = mask.shape
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    if not mask[int(round(cy)), int(round(cx))]:
        raise ValueError("valid_mask must contain the optical centre")

    angles = np.linspace(-np.pi, np.pi, output_width, endpoint=False, dtype=np.float64)
    maximum_radius = float(np.hypot(max(cx, width - 1 - cx), max(cy, height - 1 - cy)))
    probe_radius = np.linspace(0.0, maximum_radius, max(height, width) * 4)
    probe_x = np.rint(cx + np.cos(angles)[:, None] * probe_radius[None, :]).astype(int)
    probe_y = np.rint(cy + np.sin(angles)[:, None] * probe_radius[None, :]).astype(int)
    in_bounds = (probe_x >= 0) & (probe_x < width) & (probe_y >= 0) & (probe_y < height)
    ray_valid = np.zeros_like(in_bounds)
    ray_valid[in_bounds] = mask[probe_y[in_bounds], probe_x[in_bounds]]
    # Only the continuous centre-connected prefix is usable; pixels behind an
    # obstruction or outside the mask must not re-enter the representation.
    invalid_seen = np.maximum.accumulate(~ray_valid, axis=1)
    continuous = ~invalid_seen
    valid_counts = continuous.sum(axis=1)
    if np.any(valid_counts < 2):
        raise ValueError("valid_mask has a degenerate radial direction")
    boundary_radius = np.maximum(
        probe_radius[np.maximum(valid_counts - 1, 0)] - float(boundary_margin_px), 1.0
    )

    radial_fraction = np.linspace(0.0, 1.0, output_height, dtype=np.float64)
    sample_radius = radial_fraction[:, None] * boundary_radius[None, :]
    sample_x = cx + sample_radius * np.cos(angles)[None, :]
    sample_y = cy + sample_radius * np.sin(angles)[None, :]

    x0 = np.floor(sample_x).astype(int); y0 = np.floor(sample_y).astype(int)
    x1 = np.minimum(x0 + 1, width - 1); y1 = np.minimum(y0 + 1, height - 1)
    x0 = np.maximum(x0, 0); y0 = np.maximum(y0, 0)
    wx = sample_x - x0; wy = sample_y - y0
    source = array.astype(np.float32)
    weights = np.stack(
        [(1.0 - wx) * (1.0 - wy), wx * (1.0 - wy), (1.0 - wx) * wy, wx * wy], axis=-1
    )
    neighbour_valid = np.stack(
        [mask[y0, x0], mask[y0, x1], mask[y1, x0], mask[y1, x1]], axis=-1
    )
    weights *= neighbour_valid
    weight_sum = weights.sum(axis=-1, keepdims=True)
    if np.any(weight_sum <= 0):
        raise ValueError("polar sampling reached a location without valid neighbours")
    weights /= weight_sum
    neighbours = np.stack(
        [source[y0, x0], source[y0, x1], source[y1, x0], source[y1, x1]], axis=-2
    )
    output = np.sum(neighbours * weights[..., None], axis=-2)
    return np.clip(np.rint(output), 0, 255).astype(np.uint8)


def _extract_shard(args: tuple[str, list[dict], str, dict, dict, int]) -> dict:
    container, records, cache_directory, camera, day4, size = args
    cache = Path(cache_directory)
    shard_id = Path(container).stem
    pixel_path = cache / f"{shard_id}_pixels_{size}.npy"
    feature_path = cache / f"{shard_id}_features.csv"
    if pixel_path.exists() and feature_path.exists():
        existing = pd.read_csv(feature_path)
        if len(existing) == len(records):
            return {"shard": shard_id, "rows": len(records), "resumed": True}
    pixels = np.lib.format.open_memmap(pixel_path, mode="w+", dtype=np.uint8, shape=(len(records), size, size, 3))
    rows: list[dict] = []
    valid_mask = None
    if day4.get("preprocessing_mode") == "rectilinear_square":
        valid_mask = np.ones((size, size), dtype=bool)
        if day4.get("rectilinear_static_mask_path"):
            with Image.open(day4["rectilinear_static_mask_path"]) as opened_mask:
                source_mask = np.asarray(opened_mask.convert("L")) >= 128
            valid_mask = rectilinear_valid_sky_mask(source_mask, day4, size)
    with zipfile.ZipFile(container) as archive:
        for row_number, record in enumerate(records):
            result = {"sample_id": record["sample_id"], "shard": shard_id, "shard_row": row_number, "decode_ok": False, "decode_error": ""}
            try:
                data = archive.read(record["image_member"])
                if day4.get("preprocessing_mode") == "rectilinear_square":
                    array = _rectilinear_array(data, camera, day4, size)
                else:
                    array = _geometry_masked_array(data, camera, size)
                result.update(image_statistics(array, valid_mask))
                pixels[row_number] = _neutral_fill_invalid(array, valid_mask) if valid_mask is not None else array
                result["decode_ok"] = True
            except Exception as exc:
                pixels[row_number] = 0
                result["decode_error"] = f"{type(exc).__name__}: {exc}"
            rows.append(result)
    pixels.flush()
    pd.DataFrame(rows).to_csv(feature_path, index=False, encoding="utf-8-sig")
    return {"shard": shard_id, "rows": len(records), "resumed": False}


def extract_feature_cache(index_csv: str | Path, output_dir: str | Path, config: dict) -> pd.DataFrame:
    index = pd.read_csv(index_csv, low_memory=False)
    day4 = config.get("day4", {})
    size = int(day4.get("input_size", 128))
    workers = int(day4.get("extraction_workers", 4))
    cache = Path(output_dir) / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    if day4.get("preprocessing_mode") == "rectilinear_square":
        valid_mask = np.ones((size, size), dtype=bool)
        if day4.get("rectilinear_static_mask_path"):
            with Image.open(day4["rectilinear_static_mask_path"]) as opened_mask:
                source_mask = np.asarray(opened_mask.convert("L")) >= 128
            valid_mask = rectilinear_valid_sky_mask(source_mask, day4, size)
        Image.fromarray(valid_mask.astype(np.uint8) * 255).save(Path(output_dir) / "valid_sky_mask.png")
    tasks = []
    for container, group in index.groupby("image_container", sort=True):
        tasks.append((container, group[["sample_id", "image_member"]].to_dict("records"),
                      str(cache), config["camera"], day4, size))
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(_extract_shard, task) for task in tasks]
        for future in as_completed(futures):
            print(json.dumps(future.result(), ensure_ascii=False), flush=True)
    feature_frames = [pd.read_csv(path) for path in sorted(cache.glob("*_features.csv"))]
    features = pd.concat(feature_frames, ignore_index=True)
    if features["sample_id"].duplicated().any() or len(features) != len(index):
        raise RuntimeError("Feature cache is incomplete or contains duplicate sample IDs")
    manifest = index.merge(features, on="sample_id", how="left", validate="one_to_one")
    manifest.to_csv(Path(output_dir) / "image_features.csv", index=False, encoding="utf-8-sig")
    return manifest


def _metrics(y_true: np.ndarray, prediction: np.ndarray, capacity_kw: float) -> dict:
    residual = prediction - y_true
    mse = float(np.mean(residual**2))
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "n": int(len(y_true)),
        "mae_kw": float(np.mean(np.abs(residual))),
        "rmse_kw": math.sqrt(mse),
        "r2": float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else None,
        "mbe_kw": float(residual.mean()),
        "nmae_capacity_pct": float(np.mean(np.abs(residual)) / capacity_kw * 100.0),
    }


def _ridge_fit_predict(train_x, train_y, validation_x, validation_y, test_x, alphas):
    mean = train_x.mean(axis=0)
    scale = train_x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    tx = np.column_stack([np.ones(len(train_x)), (train_x - mean) / scale])
    vx = np.column_stack([np.ones(len(validation_x)), (validation_x - mean) / scale])
    sx = np.column_stack([np.ones(len(test_x)), (test_x - mean) / scale])
    best = None
    identity = np.eye(tx.shape[1]); identity[0, 0] = 0.0
    for alpha in alphas:
        weights = np.linalg.solve(tx.T @ tx + float(alpha) * identity, tx.T @ train_y)
        prediction = vx @ weights
        mae = float(np.mean(np.abs(prediction - validation_y)))
        if best is None or mae < best[0]:
            best = (mae, float(alpha), weights)
    model = {"weights": best[2], "mean": mean, "scale": scale, "alpha": best[1]}
    return vx @ best[2], sx @ best[2], model


def run_statistical_baselines(features: pd.DataFrame, output_dir: str | Path, config: dict) -> dict:
    output = Path(output_dir)
    valid = features["decode_ok"].astype(bool)
    data = features.loc[valid].copy()
    target = "power_operational_qc_kw"
    feature_columns = [
        column for column in data.columns
        if column.endswith(("_mean", "_std")) or "_q" in column or column.endswith("_fraction") or column == "red_blue_log_ratio"
    ]
    # Explicitly exclude label-derived or metadata columns accidentally matching suffixes.
    feature_columns = [column for column in feature_columns if column not in {"power_raw_kw", "power_operational_qc_kw"}]
    splits = {name: data.loc[data["split_chronological"].eq(name)] for name in ("train", "validation", "test")}
    y = {name: frame[target].to_numpy(float) for name, frame in splits.items()}
    capacity = float(config["site"]["installed_dc_kwp"])
    predictions: dict[str, dict[str, np.ndarray]] = {name: {} for name in splits}

    train_mean = float(y["train"].mean())
    for name in splits:
        predictions[name]["train_mean"] = np.full(len(splits[name]), train_mean)

    solar_x = {}
    for name, frame in splits.items():
        elevation = np.deg2rad(frame["solar_elevation_deg"].to_numpy(float))
        solar_x[name] = np.column_stack([np.sin(elevation), np.sin(elevation) ** 2, np.sin(elevation) ** 3])
    solar_val, solar_test, solar_model = _ridge_fit_predict(solar_x["train"], y["train"], solar_x["validation"], y["validation"], solar_x["test"], [0.0, 0.1, 1, 10, 100, 1000])
    predictions["validation"]["solar_only_ridge"] = solar_val
    predictions["test"]["solar_only_ridge"] = solar_test

    image_x = {name: frame[feature_columns].to_numpy(float) for name, frame in splits.items()}
    image_val, image_test, image_model = _ridge_fit_predict(image_x["train"], y["train"], image_x["validation"], y["validation"], image_x["test"], [0.0, 0.1, 1, 10, 100, 1000])
    predictions["validation"]["image_statistics_ridge"] = image_val
    predictions["test"]["image_statistics_ridge"] = image_test
    combined_x = {name: np.column_stack([image_x[name], solar_x[name]]) for name in splits}
    combined_val, combined_test, combined_model = _ridge_fit_predict(combined_x["train"], y["train"], combined_x["validation"], y["validation"], combined_x["test"], [0.0, 0.1, 1, 10, 100, 1000])
    predictions["validation"]["image_plus_solar_ridge"] = combined_val
    predictions["test"]["image_plus_solar_ridge"] = combined_test

    rows = []
    for split in ("validation", "test"):
        for model, prediction in predictions[split].items():
            clipped = np.clip(prediction, 0.0, capacity * 1.25)
            rows.append({"split": split, "model": model, **_metrics(y[split], clipped, capacity)})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "baseline_metrics.csv", index=False, encoding="utf-8-sig")
    prediction_table = splits["test"][["sample_id", "image_timestamp", target]].copy()
    for model, prediction in predictions["test"].items():
        prediction_table[f"prediction_{model}_kw"] = np.clip(prediction, 0.0, capacity * 1.25)
    prediction_table.to_csv(output / "test_predictions.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        output / "ridge_models.npz",
        train_mean_kw=np.asarray([train_mean]),
        image_feature_names=np.asarray(feature_columns),
        solar_feature_names=np.asarray(["sin_solar_elevation", "sin_solar_elevation_squared", "sin_solar_elevation_cubed"]),
        solar_weights=solar_model["weights"], solar_mean=solar_model["mean"], solar_scale=solar_model["scale"], solar_alpha=np.asarray([solar_model["alpha"]]),
        image_weights=image_model["weights"], image_mean=image_model["mean"], image_scale=image_model["scale"], image_alpha=np.asarray([image_model["alpha"]]),
        combined_weights=combined_model["weights"], combined_mean=combined_model["mean"], combined_scale=combined_model["scale"], combined_alpha=np.asarray([combined_model["alpha"]]),
    )
    summary = {
        "schema_version": "4.0-day4-baseline",
        "feature_count": len(feature_columns),
        "decoded_count": int(valid.sum()),
        "decode_failure_count": int((~valid).sum()),
        "effective_grayscale_count": int(features.loc[valid, "effective_grayscale"].astype(bool).sum()),
        "effective_grayscale_by_split": {key: int(value) for key, value in data.groupby("split_chronological")["effective_grayscale"].sum().items()},
        "ridge_alpha": {"solar_only": solar_model["alpha"], "image_statistics": image_model["alpha"], "image_plus_solar": combined_model["alpha"]},
        "metrics": rows,
        "notes": [
            "The image-statistics model uses only the current masked fisheye image.",
            "Solar-only is a contextual baseline, not an image model.",
            "Image-plus-solar is diagnostic and must not be described as image-only.",
            "All predictions are clipped to [0, 1.25 times DC nameplate] after fitting.",
        ],
    }
    (output / "day4_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_day4(index_csv: str | Path, output_dir: str | Path, config: dict) -> list[Path]:
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    features = extract_feature_cache(index_csv, output, config)
    run_statistical_baselines(features, output, config)
    return [output / "image_features.csv", output / "baseline_metrics.csv", output / "test_predictions.csv", output / "day4_summary.json"]
