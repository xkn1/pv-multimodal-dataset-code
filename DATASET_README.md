# A 26-month photovoltaic multimodal dataset with sky images, irradiance, reanalysis and forecasts

Prepublication package for data covering 2024-01-01 through 2026-02-28.

## Verified contents

- 122,213 all-sky images in 26 monthly lossless JPEG XL archives
- 96,481 unique image-power pairs
- chronological splits: 65,837 train, 17,032 validation, 13,612 test
- 500 reviewed annotation images and 500 semantic masks
- 6,405 primary 15-minute 1.5 kW ramp events
- 18,960 ERA5 hours and 1,429 available IFS runs

JPEG XL archives reproduce every original JPEG byte-for-byte. Use the supplied restoration tool with `djxl --reconstruct_jpeg`. Twenty-five exceptional JPEG files are stored as original-byte passthrough entries.

## Licenses

Original software: MIT. Original data: CC BY 4.0. Third-party source-derived material remains subject to its upstream terms. Read `LICENSE_SCOPE.md` and `THIRD_PARTY_NOTICES.md`.

Power values are source-reported measurements at five-minute timestamps; the pipeline does not average or interpolate them. See POWER_DATA_SEMANTICS.md.

## Important quality notes

The semantic-mask encoding is 0=invalid/outside ROI, 1=cloud, 2=sun, 3=sky. Six masks are retained with explicit QC warnings; see `KNOWN_ANNOTATION_LIMITATIONS.md`.

Author metadata and the public contact are complete. Reserved DOI: https://doi.org/10.5281/zenodo.22844897. The DOI will become publicly resolvable when the Zenodo record is published.
