# Spectral-Aware CodeFormer

This repository extends the official CodeFormer codebase with a small, trainable spectral-awareness experiment for high-fidelity image reconstruction. The base restoration pipeline remains CodeFormer: a VQGAN codebook provides discrete visual tokens and a transformer predicts code indices from degraded inputs. The project contribution is an optional Fourier-magnitude loss that can be enabled during image-level fine tuning.

## What changed

- Added `FrequencyL1Loss` in `basicsr/losses/losses.py`.
- Wired `train.frequency_opt` into `CodeFormerModel` and `CodeFormerJointModel`.
- Added `options/CodeFormer_stage3_spectral.yml` for spectral fine tuning.
- Added `scripts/evaluate_reconstruction.py` for quick PSNR, global SSIM, and Freq-L1 checks on paired folders.

## Setup

```bash
conda create -n spectral-codeformer python=3.8 -y
conda activate spectral-codeformer
pip install -r requirements.txt
python basicsr/setup.py develop
```

Download the pretrained CodeFormer/VQGAN weights using the original helper:

```bash
python scripts/download_pretrained_models.py CodeFormer
python scripts/download_pretrained_models.py facelib
```

For training, place FFHQ/CelebA-HQ style images under `datasets/ffhq/ffhq_512`, or edit `options/CodeFormer_stage3_spectral.yml` to point to your dataset.

## Training

Baseline CodeFormer stage-3 fine tuning:

```bash
python basicsr/train.py -opt options/CodeFormer_stage3.yml
```

Spectral-aware fine tuning:

```bash
python basicsr/train.py -opt options/CodeFormer_stage3_spectral.yml
```

The spectral run uses:

```yaml
frequency_opt:
  type: FrequencyL1Loss
  loss_weight: 0.1
  reduction: mean
  log_amplitude: true
```

Set `loss_weight: 0.0` or remove `frequency_opt` for the no-spectral ablation.

## Inference

```bash
python inference_codeformer.py -w 0.7 --input_path inputs/cropped_faces --bg_upsampler realesrgan
```

## Evaluation

After generating reconstructed images with matching ground-truth filenames:

```bash
python scripts/evaluate_reconstruction.py \
  --pred_dir results/spectral/restored_faces \
  --gt_dir datasets/validation/gt
```

Final report numbers should be produced from the same validation split for the baseline and spectral runs. The included evaluator is a lightweight sanity checker; final publication-quality LPIPS/FID/MUSIQ should use standard benchmark implementations.

## Notes

This repo inherits CodeFormer's upstream license and implementation structure. The spectral loss is intentionally small and isolated so future work can test stronger frequency objectives, multi-scale spectra, or task-specific degradation models without rewriting the restoration backbone.
