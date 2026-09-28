from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image


CURRENT_VALUES = {0, 1, 2, 3}
LEGACY_VALUES = {0, 1, 2, 255}


def verify(package: Path) -> dict:
    manifest_path = package / "manifest_500.csv"
    if not manifest_path.is_file():
        manifest_path = package / "annotation_manifest_500.csv"
    image_dir = package / "images"
    if not image_dir.is_dir():
        image_dir = package / "images_512"
    mask_dir = package / "semantic_masks"
    current_release = mask_dir.is_dir()
    if not current_release:
        mask_dir = package / "masks"
        if not mask_dir.is_dir():
            mask_dir = package / "masks_empty_255_ignore"
    valid_values = CURRENT_VALUES if current_release else LEGACY_VALUES
    expected_size = (1024, 1024) if current_release else (512, 512)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not image_dir.is_dir() or not mask_dir.is_dir():
        raise FileNotFoundError("Missing annotation images or mask directory")

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))

    errors: list[str] = []
    seen: set[str] = set()
    value_counts = {value: 0 for value in sorted(valid_values)}
    for row in rows:
        sample_id = (row.get("sample_id") or row.get("annotation_id") or "").strip()
        if not sample_id or sample_id in seen:
            errors.append(f"missing or duplicate sample_id: {sample_id!r}")
            continue
        seen.add(sample_id)
        file_id = row.get("annotation_id", "").strip() or sample_id
        image_path = image_dir / f"{file_id}.jpg"
        mask_path = mask_dir / f"{file_id}.png"
        if not image_path.is_file() or not mask_path.is_file():
            errors.append(f"missing image or mask for {file_id} ({sample_id})")
            continue
        with Image.open(image_path) as image:
            if image.mode != "RGB" or image.size != expected_size:
                errors.append(f"invalid image mode/size for {sample_id}: {image.mode} {image.size}")
        with Image.open(mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L"), dtype=np.uint8)
        if mask.shape != expected_size[::-1]:
            errors.append(f"invalid mask size for {sample_id}: {mask.shape}")
            continue
        values, counts = np.unique(mask, return_counts=True)
        invalid = sorted(set(map(int, values)) - valid_values)
        if invalid:
            errors.append(f"invalid mask values for {sample_id}: {invalid}")
        for value, count in zip(values, counts):
            if int(value) in value_counts:
                value_counts[int(value)] += int(count)

    return {
        "package": str(package.resolve()),
        "manifest_rows": len(rows),
        "unique_sample_ids": len(seen),
        "mask_value_counts": value_counts,
        "error_count": len(errors),
        "errors": errors[:100],
        "passed": not errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the cloud-reference annotation package")
    parser.add_argument("package", type=Path, help="Directory containing the manifest, images/ and masks/")
    parser.add_argument("--output-json", type=Path)
    args = parser.parse_args()
    result = verify(args.package)
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
