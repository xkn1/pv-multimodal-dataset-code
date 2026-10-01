"""Repair the label spacing in the two archived Fig. 8 event panels.

This script deliberately does not replot or resample scientific measurements.
It copies the chart and sky-image pixels from the two original PNGs, replaces
only the crowded strip between them, and typesets the same labels in that strip.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PLOT_BOTTOM = 708
SKY_TOP = 840
NEW_GAP = 150
PANEL_GAP = 44

CAPTIONS = {
    "down": [
        ("15:20", "P=3.5 kW, C=0.84"),
        ("15:30", "P=4.4 kW, C=0.46"),
        ("15:40", "P=2.6 kW, C=0.33"),
        ("15:55", "P=0.9 kW, C=0.91"),
        ("16:05", "P=1.4 kW, C=0.71"),
    ],
    "up": [
        ("09:30", "P=2.5 kW, C=0.90"),
        ("09:40", "P=2.6 kW, C=0.88"),
        ("09:50", "P=3.9 kW, C=0.90"),
        ("10:00", "P=8.1 kW, C=0.49"),
        ("10:15", "P=9.0 kW, C=0.16"),
    ],
}

TICK_LABELS = {
    "down": [f"05 {hour:02d}:{minute:02d}" for hour, minute in
             [(14, 45), (15, 0), (15, 15), (15, 30), (15, 45), (16, 0), (16, 15), (16, 30)]],
    "up": [f"14 {hour:02d}:{minute:02d}" for hour, minute in
           [(9, 0), (9, 15), (9, 30), (9, 45), (10, 0), (10, 15), (10, 30), (10, 45)]],
}


def dark_runs(row, minimum: int) -> list[tuple[int, int]]:
    """Return large dark horizontal spans at a known chart or image row."""
    spans = []
    start = None
    for x, color in enumerate(row):
        dark = max(color[:3]) < 100
        if dark and start is None:
            start = x
        elif not dark and start is not None:
            if x - start >= minimum:
                spans.append((start, x))
            start = None
    if start is not None and len(row) - start >= minimum:
        spans.append((start, len(row)))
    return spans


def font(size: int) -> ImageFont.FreeTypeFont:
    path = Path("C:/Windows/Fonts/arial.ttf")
    if not path.is_file():
        raise FileNotFoundError(f"Required font not found: {path}")
    return ImageFont.truetype(str(path), size)


def repair(source: Path, direction: str) -> Image.Image:
    with Image.open(source) as raw:
        original = raw.convert("RGB")
    if original.height != 1160 or original.width not in (2151, 2176):
        raise ValueError(f"Unexpected source dimensions for {source}: {original.size}")

    width, height = original.size
    result = Image.new("RGB", (width, PLOT_BOTTOM + NEW_GAP + height - SKY_TOP), "white")
    result.paste(original.crop((0, 0, width, PLOT_BOTTOM)), (0, 0))
    result.paste(original.crop((0, SKY_TOP, width, height)), (0, PLOT_BOTTOM + NEW_GAP))

    draw = ImageDraw.Draw(result)
    # Re-type the tick labels in a clean band. The source's dotted connectors
    # started below the plot and crossed the old captions in that band.
    tick_spans = dark_runs([original.getpixel((x, 710)) for x in range(width)], 2)
    image_spans = dark_runs([original.getpixel((x, 840)) for x in range(width)], 200)
    if len(tick_spans) != 8 or len(image_spans) != 5:
        raise ValueError(f"Could not locate original ticks/images: {source}")
    for (left, right), label in zip(tick_spans, TICK_LABELS[direction]):
        center = (left + right) / 2
        draw.line((center, PLOT_BOTTOM, center, PLOT_BOTTOM + 7), fill="#222222", width=2)
        draw.text((center, PLOT_BOTTOM + 15), label,
                  fill="#222222", font=font(23), anchor="mt")
    draw.text((width / 2, PLOT_BOTTOM + 51), "Local time (Asia/Shanghai)",
              fill="#222222", font=font(23), anchor="mt")
    centers = [(left + right) / 2 for left, right in image_spans]
    for center, (time, details) in zip(centers, CAPTIONS[direction]):
        draw.text((center, PLOT_BOTTOM + 87), time, fill="#222222",
                  font=font(23), anchor="mt")
        draw.text((center, PLOT_BOTTOM + 115), details, fill="#222222",
                  font=font(20), anchor="mt")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--down", type=Path, required=True)
    parser.add_argument("--up", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    down = repair(args.down, "down")
    up = repair(args.up, "up")
    down.save(args.output_dir / "Fig8_ramp_down_labels_fixed.png", dpi=(300, 300))
    up.save(args.output_dir / "Fig8_ramp_up_labels_fixed.png", dpi=(300, 300))

    width = max(down.width, up.width)
    combined = Image.new("RGB", (width, down.height + PANEL_GAP + up.height), "white")
    combined.paste(down, ((width - down.width) // 2, 0))
    combined.paste(up, ((width - up.width) // 2, down.height + PANEL_GAP))
    combined.save(args.output_dir / "Fig8_ramp_events_labels_fixed.png", dpi=(300, 300))


if __name__ == "__main__":
    main()
