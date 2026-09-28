# Photovoltaic multimodal dataset: analysis code

This repository contains the processing, quality-control, pairing, and baseline-model code released with the 26-month photovoltaic multimodal dataset. The dataset files are hosted in the [Hugging Face dataset repository](https://huggingface.co/datasets/keningxie-pv/pv-multimodal-dataset). See [DATASET_README.md](DATASET_README.md) and [README_zh.md](README_zh.md) for the dataset description.

## Contents

- `code/pv_dataset/`: image, power, weather, pairing, ramp, and baseline processing modules.
- `code/experiments/`: example power-regression and future-ramp training scripts.
- `code/config/site_config.example.json`: public configuration template. Replace placeholders with paths to data you downloaded; do not commit a private site configuration.
- `code/tests/`: release smoke tests.

## Setup and a first check

Use Python 3.10 or later. From the `code/` directory:

```bash
python -m pip install -r requirements.txt
python -m pytest -q tests/test_release_smoke.py
python run_pipeline.py --config config/site_config.example.json --help
```

Install `requirements-models.txt` to run the PyTorch training scripts. The full pipeline requires dataset files and a local configuration with valid paths. The example configuration does not include private site coordinates and is not a record of final camera-calibration parameters. The released code therefore should not be taken as evidence that every image used in the reported experiments can be reconstructed from the public configuration alone.

## Data, rights, and citation

Download data from [Hugging Face](https://huggingface.co/datasets/keningxie-pv/pv-multimodal-dataset); large image archives are not stored here. Original code is MIT-licensed under [LICENSE_CODE_MIT.txt](LICENSE_CODE_MIT.txt). Dataset content has a separate CC BY 4.0 licence, and third-party material retains upstream terms; see [LICENSE_SCOPE.md](LICENSE_SCOPE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

Please cite the dataset and associated paper using [CITATION.cff](CITATION.cff). The reserved Zenodo DOI in the dataset documentation is not a published citation until that record is released.
