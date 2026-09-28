from __future__ import annotations

import argparse
import json
from pathlib import Path

from .clouds import red_blue_ratio_cloud_baseline
from .config import load_config
from .day3 import run_day3_pairing
from .day4 import run_day4
from .images import create_provisional_angular_map, mask_and_resize_full_hemisphere
from .manifest import build_manifest, save_manifest
from .power import run_power_qc
from .ramps import run_ramp_detection


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="All-sky PV dataset processing pipeline")
    root.add_argument("--config", help="Path to site_config.json")
    commands = root.add_subparsers(dest="command", required=True)

    power = commands.add_parser("power-qc", help="Build a non-interpolated power QC table")
    power.add_argument("--raw-dir")
    power.add_argument("--irradiance", help="Optional irradiance workbook for operational-state QC")
    power.add_argument("--output-dir", required=True)

    ramps = commands.add_parser("ramps", help="Create multi-scale ramp candidates and merged events")
    ramps.add_argument("--power-qc", required=True)
    ramps.add_argument("--output-dir", required=True)
    ramps.add_argument("--training-end", help="Optional cutoff for train-only statistical thresholds")

    manifest = commands.add_parser("manifest", help="Pair image timestamps with power and optional covariates")
    manifest.add_argument("--images", required=True, help="Image directory, ZIP directory or one ZIP")
    manifest.add_argument("--power-qc", required=True)
    manifest.add_argument("--irradiance")
    manifest.add_argument("--weather")
    manifest.add_argument("--output", required=True)

    day3 = commands.add_parser("pair-day3", help="Pair audited images with Day-2 power QC")
    day3.add_argument("--image-inventory", required=True)
    day3.add_argument("--power-qc", required=True)
    day3.add_argument("--output-dir", required=True)

    day4 = commands.add_parser("day4-baseline", help="Extract masked-image features and run baseline models")
    day4.add_argument("--index", required=True)
    day4.add_argument("--output-dir", required=True)

    image = commands.add_parser("preprocess-image", help="Preserve and resize the full fisheye hemisphere")
    image.add_argument("--input", required=True)
    image.add_argument("--output", required=True)
    image.add_argument("--size", type=int, default=512)

    angular = commands.add_parser("angular-map", help="Create an explicitly provisional equidistant angular map")
    angular.add_argument("--output", required=True)
    angular.add_argument("--size", type=int, default=512)

    cloud = commands.add_parser("cloud-baseline", help="Generate a heuristic red/blue-ratio cloud mask")
    cloud.add_argument("--input", required=True)
    cloud.add_argument("--output-mask")
    cloud.add_argument("--threshold", type=float)
    cloud.add_argument("--valid-sky-mask", help="Static mask excluding buildings, trees and camera housing")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "power-qc":
        raw = args.raw_dir or config["paths"]["raw_power_directory"]
        outputs = run_power_qc(raw, args.output_dir, config, args.irradiance)
        print("\n".join(str(path) for path in outputs))
    elif args.command == "ramps":
        outputs = run_ramp_detection(args.power_qc, args.output_dir, config, args.training_end)
        print("\n".join(str(path) for path in outputs))
    elif args.command == "manifest":
        manifest = build_manifest(
            args.images,
            args.power_qc,
            config,
            args.irradiance,
            args.weather,
        )
        print(save_manifest(manifest, args.output))
    elif args.command == "pair-day3":
        outputs = run_day3_pairing(
            args.image_inventory,
            args.power_qc,
            args.output_dir,
            config,
        )
        print("\n".join(str(path) for path in outputs))
    elif args.command == "day4-baseline":
        outputs = run_day4(args.index, args.output_dir, config)
        print("\n".join(str(path) for path in outputs))
    elif args.command == "preprocess-image":
        metadata = mask_and_resize_full_hemisphere(
            args.input, args.output, config["camera"], args.size
        )
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    elif args.command == "angular-map":
        print(create_provisional_angular_map(args.output, args.size, config["camera"]))
    elif args.command == "cloud-baseline":
        result = red_blue_ratio_cloud_baseline(
            args.input, args.output_mask, args.threshold, args.valid_sky_mask
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0
