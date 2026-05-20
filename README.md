# Spectral-Aware CodeFormer

This repository contains the code-only implementation for a spectral-aware extension of CodeFormer. The base model remains a VQ/codebook restoration pipeline, and the project change is an optional FFT/Fourier magnitude loss for fine-detail preservation.

This repo intentionally excludes demo pictures, sample inputs, and report images so the GitHub project focuses on the implementation and reproducible commands.

## Main Code Changes

- `basicsr/losses/losses.py`
  - Adds `FrequencyL1Loss`, a log-magnitude FFT L1 loss.
- `basicsr/models/codeformer_model.py`
  - Adds optional `train.frequency_opt` support.
- `basicsr/models/codeformer_joint_model.py`
  - Adds optional `train.frequency_opt` support for joint training.
- `options/CodeFormer_stage3_nofft.yml`
  - Baseline fine-tuning config without FFT loss.
- `options/CodeFormer_stage3_spectral.yml`
  - FFT-loss fine-tuning config.
- `scripts/compare_fft_ablation.py`
  - Compares no-FFT and FFT prediction folders with PSNR, global SSIM, and Freq-L1.

## Setup

```bash
conda create -n spectral-codeformer python=3.8 -y
conda activate spectral-codeformer
pip install -r requirements.txt
python basicsr/setup.py develop
```

Download pretrained weights:

```bash
python scripts/download_pretrained_models.py CodeFormer
python scripts/download_pretrained_models.py facelib
```

Place aligned training images here, or edit the config paths:

```text
datasets/ffhq/ffhq_512
```

## Train Without FFT Loss

```bash
python basicsr/train.py -opt options/CodeFormer_stage3_nofft.yml
```

## Train With FFT Loss

```bash
python basicsr/train.py -opt options/CodeFormer_stage3_spectral.yml
```

The FFT config enables:

```yaml
frequency_opt:
  type: FrequencyL1Loss
  loss_weight: 0.1
  reduction: mean
  log_amplitude: true
```

## Generate Outputs For Comparison

Run inference with the no-FFT checkpoint and save outputs to a folder such as:

```text
results/nofft/restored_faces
```

Run inference with the FFT-loss checkpoint and save outputs to:

```text
results/fft/restored_faces
```

Keep the same validation images and filenames for both runs.

## Evaluate No-FFT vs FFT

```bash
python scripts/compare_fft_ablation.py \
  --nofft_dir results/nofft/restored_faces \
  --fft_dir results/fft/restored_faces \
  --gt_dir datasets/validation/gt
```

Expected output format:

```text
| model | images | PSNR up | SSIM up | Freq-L1 down |
|---|---:|---:|---:|---:|
| nofft | ... | ... | ... | ... |
| fft | ... | ... | ... | ... |

Delta fft - nofft:
PSNR:    ...
SSIM:    ...
Freq-L1: ...
```

## Notes

The repository is based on CodeFormer and keeps its license terms. The project-specific contribution is the FFT/Fourier loss integration and the ablation workflow for comparing restoration with and without spectral supervision.
