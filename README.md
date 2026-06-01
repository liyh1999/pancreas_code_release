# Pancreatic segmentation and regional fat quantification code

This repository contains the analysis scripts accompanying the manuscript:

**Automated Pancreatic Segmentation and Regional Fat Quantification Suggest Tail Fat Association with Type 2 Diabetes**

The scripts implement the reproducible processing workflow described in the manuscript:

1. patient-level stratified dataset splitting;
2. nnU-Net training and inference command wrappers;
3. semi-automated head-body-tail partitioning of whole-pancreas masks;
4. regional volume and fat quantification;
5. statistical analysis and random forest classification;
6. figure generation from de-identified tabular outputs.

No clinical imaging data are included in this repository. Clinical imaging data, annotations, and related analysis files are not publicly released because of patient privacy protection, institutional data governance requirements, and ethics approval conditions.

## Repository structure

```text
pancreas_code_release/
  README.md
  requirements.txt
  config.example.yaml
  CITATION.cff
  scripts/
    split_dataset.py
    run_nnunet.py
    partition_pancreas.py
    quantify_fat.py
    analyze_statistics.py
    generate_figures.py
  pancreas_region_analysis/
    __init__.py
    imaging.py
    partition.py
    quantification.py
```

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements.txt
```

For nnU-Net training and inference, install `nnunetv2` separately following the official nnU-Net v2 documentation, and configure the required environment variables:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

On Windows PowerShell:

```powershell
$env:nnUNet_raw="C:\path\to\nnUNet_raw"
$env:nnUNet_preprocessed="C:\path\to\nnUNet_preprocessed"
$env:nnUNet_results="C:\path\to\nnUNet_results"
```

## Typical workflow

### 1. Create patient-level folds

Input CSV must contain at least:

- `patient_id`
- `group`, for example `Healthy`, `Prediabetic`, `Diabetic`

```bash
python scripts/split_dataset.py metadata.csv outputs/folds.csv --n-splits 5 --seed 42
```

### 2. Train or predict with nnU-Net

This script is a thin wrapper around official nnU-Net v2 commands.

```bash
python scripts/run_nnunet.py plan --dataset-id 501 --verify-dataset-integrity
python scripts/run_nnunet.py train --dataset-id 501 --configuration 3d_fullres --fold 0
python scripts/run_nnunet.py predict --input imagesTs --output predictions --dataset-id 501 --configuration 3d_fullres --fold 0
```

### 3. Partition whole-pancreas masks into head, body, and tail

```bash
python scripts/partition_pancreas.py \
  --mask predictions/case001.nii.gz \
  --output outputs/regions/case001_regions.nii.gz \
  --head-side max
```

Region labels in the output mask:

- `1`: tail
- `2`: body
- `3`: head

The script performs the algorithmic initialization. Expert review and correction can then be performed in 3D Slicer or a comparable medical image viewer.

### 4. Quantify regional volume and fat fraction

Using a PDFF map:

```bash
python scripts/quantify_fat.py \
  --region-mask outputs/regions/case001_regions.nii.gz \
  --pdff pdff/case001_pdff.nii.gz \
  --patient-id case001 \
  --group Healthy \
  --output-csv outputs/quantification.csv
```

Using in-phase and opposed-phase images:

```bash
python scripts/quantify_fat.py \
  --region-mask outputs/regions/case001_regions.nii.gz \
  --in-phase dixon/case001_in.nii.gz \
  --opposed-phase dixon/case001_opp.nii.gz \
  --patient-id case001 \
  --group Healthy \
  --output-csv outputs/quantification.csv
```

### 5. Statistical analysis and model evaluation

```bash
python scripts/analyze_statistics.py \
  --input-csv outputs/quantification.csv \
  --output-dir outputs/statistics \
  --seed 42
```

### 6. Generate figures

```bash
python scripts/generate_figures.py \
  --input-csv outputs/quantification.csv \
  --stats-dir outputs/statistics \
  --output-dir outputs/figures
```

## Notes

- This repository provides the reproducible analysis workflow and does not include patient data.
- Because the regional partitioning is semi-automated, algorithmic outputs should be reviewed by experienced radiologists before final quantitative analysis.
- The `--head-side` option should be confirmed according to image orientation and visual inspection.

