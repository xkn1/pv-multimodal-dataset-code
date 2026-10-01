# Manuscript figure sources

These scripts were omitted from code release v1.0.0 and added in the next
revision. They are the source scripts used for the figure families listed below.
They do **not** package site-private intermediate predictions or the original
JPEG containers. A script being present does not by itself mean its historical
output can be regenerated from the public dataset alone.

| Manuscript graphic | Generator |
| --- | --- |
| Resource workflow; image-processing examples; dataset overview | `build_inserted_paper_figures.py` |
| Temporal alignment; power–GHI consistency; sky geometry; NWP integrity; semantic-label validation; ramp robustness | `build_validation_figures.py` |
| Four image-supported ramp examples (the two selected panels in the manuscript and two supplementary examples) | `build_cloud_power_ramp_examples.py` |
| Typical semantic predictions (`fig05b_typical_semantic_predictions.jpg`) | **Original generator not located**; see audit below |

The ramp script was revised to separate the chart ticks from thumbnail captions.
It places the timezone below the thumbnails and removes lines that crossed
caption text. Its event-selection and data values were not changed. The
wording of its module description was also corrected from causal to directional
evidence; the paper does not infer causality from these plots.

## Usage

Install `numpy`, `pandas`, `matplotlib`, and `Pillow`. All scripts accept
explicit source/output paths instead of embedding a private machine path.

```bash
python build_validation_figures.py --validation-root /path/to/validation --output-dir /path/to/figures
python build_inserted_paper_figures.py --source-root /path/to/source --output-dir /path/to/figures
python build_cloud_power_ramp_examples.py --features /path/to/features.csv --cloud-features /path/to/cloud_features.csv --ramp-labels /path/to/ramp_labels.csv --output-dir /path/to/figures
```

The last script expects `image_container` values in its feature table to
point to ZIP archives containing original JPEG members, with `image_member`
as the ZIP member name. Paths from the historical server must be remapped to
local paths. The published monthly ZIPs contain lossless JPEG XL files; restore
JPEG bytes as described in the dataset documentation before using this script.

## Provenance and input audit

The scripts were compared with the figure references in the manuscript's
`中文论文初稿_v4_插图版.tex` on 2026-10-01. The `fig00` family, `fig01`–`fig06`
family, and four ramp examples have identified scripts. The original generator
for `fig05b_typical_semantic_predictions.jpg` could not be found in the local
Python, notebook, or shell sources. Its image file exists, but its exact
rendering recipe and input prediction masks are not supplied here. Do not claim
that this figure is reproducible from the code release until that gap is closed.

The historical ramp-event figure could not be redrawn from the local public
package during this patch: the required joined cloud-feature time series and
the original JPEG ZIP path referenced by the old CSV are absent. A synthetic
layout render verified the spacing change, but does not verify scientific
values or replace the manuscript figure. Re-render the real figure from the
archived input tables and image ZIPs before updating the manuscript graphic.
