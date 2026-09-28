from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def resolved_geometry(width: int, height: int, camera_config: dict) -> dict:
    center_x = camera_config.get("optical_center_x_px")
    center_y = camera_config.get("optical_center_y_px")
    radius = camera_config.get("sky_radius_px")
    provisional = center_x is None or center_y is None or radius is None
    if center_x is None:
        center_x = width / 2.0
    if center_y is None:
        center_y = height / 2.0
    if radius is None:
        radius = min(width, height) / 2.0
    if radius <= 0:
        raise ValueError("sky_radius_px must be positive")
    if center_x - radius < 0 or center_x + radius > width or center_y - radius < 0 or center_y + radius > height:
        raise ValueError("Configured sky circle falls outside the image")
    return {
        "center_x_px": float(center_x),
        "center_y_px": float(center_y),
        "radius_px": float(radius),
        "geometry_status": "provisional" if provisional else camera_config.get("geometry_status", "calibrated"),
    }


def mask_and_resize_full_hemisphere(
    input_path: str | Path,
    output_path: str | Path,
    camera_config: dict,
    output_size: int = 512,
    jpeg_quality: int = 95,
) -> dict:
    """Crop the complete fisheye circle, mask non-sky corners and resize.

    This operation intentionally does not claim geometric undistortion. Until
    calibrated projection parameters are available it is safer than discarding
    the horizon through a 90-degree rectilinear crop.
    """
    source = Path(input_path)
    destination = Path(output_path)
    with Image.open(source) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    geometry = resolved_geometry(width, height, camera_config)
    cx, cy, radius = geometry["center_x_px"], geometry["center_y_px"], geometry["radius_px"]
    box = (round(cx - radius), round(cy - radius), round(cx + radius), round(cy + radius))
    square = image.crop(box)
    diameter = square.size[0]
    yy, xx = np.ogrid[: square.size[1], : square.size[0]]
    local_cx = (square.size[0] - 1) / 2.0
    local_cy = (square.size[1] - 1) / 2.0
    circle = ((xx - local_cx) ** 2 + (yy - local_cy) ** 2) <= (diameter / 2.0) ** 2
    array = np.asarray(square).copy()
    array[~circle] = 0
    result = Image.fromarray(array, mode="RGB")
    rotation = camera_config.get("north_rotation_degrees_clockwise")
    if rotation is not None and float(rotation) != 0:
        result = result.rotate(-float(rotation), resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
    result = result.resize((int(output_size), int(output_size)), Image.Resampling.LANCZOS)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        result.save(destination, quality=int(jpeg_quality), subsampling=0, optimize=True)
    else:
        result.save(destination)
    return {
        "input": str(source),
        "output": str(destination),
        "input_width": width,
        "input_height": height,
        "output_size": int(output_size),
        **geometry,
        "projection_model": camera_config.get("projection_model", "unknown"),
        "operation": "full_fisheye_circle_mask_and_resize",
        "publication_geometry_calibrated": geometry["geometry_status"] == "calibrated",
    }


def create_provisional_angular_map(output_path: str | Path, output_size: int, camera_config: dict) -> Path:
    """Create a provisional pixel-to-angle map under an explicit equidistant assumption."""
    if camera_config.get("projection_model") not in {"equidistant", "unknown"}:
        raise ValueError("Angular-map generation currently supports only an equidistant model")
    size = int(output_size)
    yy, xx = np.mgrid[0:size, 0:size]
    center = (size - 1) / 2.0
    dx, dy = xx - center, yy - center
    radius = size / 2.0
    normalized_radius = np.sqrt(dx * dx + dy * dy) / radius
    valid = normalized_radius <= 1.0
    zenith = np.where(valid, normalized_radius * 90.0, np.nan)
    azimuth = (np.rad2deg(np.arctan2(dx, -dy)) + 360.0) % 360.0
    rotation = camera_config.get("north_rotation_degrees_clockwise")
    if rotation is not None:
        azimuth = (azimuth + float(rotation)) % 360.0
    azimuth = np.where(valid, azimuth, np.nan)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        zenith_degrees=zenith.astype(np.float32),
        azimuth_degrees=azimuth.astype(np.float32),
        valid_sky_mask=valid,
        metadata_json=json.dumps({
            "projection_model": "equidistant_assumption",
            "geometry_status": camera_config.get("geometry_status", "provisional"),
            "warning": "Do not describe this map as calibrated until optical center, radius and orientation are validated",
        }, ensure_ascii=False),
    )
    return path
