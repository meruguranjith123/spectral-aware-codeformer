import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def read_rgb(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32) / 255.0


def psnr(pred, target):
    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * np.log10(1.0 / np.sqrt(mse))


def ssim_simple(pred, target):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var()
    sigma_y = target.var()
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    return ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    )


def freq_l1(pred, target):
    pred_fft = np.fft.rfft2(pred, axes=(0, 1), norm='ortho')
    target_fft = np.fft.rfft2(target, axes=(0, 1), norm='ortho')
    return np.mean(np.abs(np.log1p(np.abs(pred_fft)) - np.log1p(np.abs(target_fft))))


def main():
    parser = argparse.ArgumentParser(description='Evaluate paired reconstruction folders.')
    parser.add_argument('--pred_dir', required=True, help='Folder containing reconstructed images.')
    parser.add_argument('--gt_dir', required=True, help='Folder containing ground-truth images with matching names.')
    parser.add_argument('--ext', default='png,jpg,jpeg', help='Comma-separated image extensions.')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    exts = {f'.{ext.strip().lower()}' for ext in args.ext.split(',')}
    pred_paths = sorted(p for p in pred_dir.iterdir() if p.suffix.lower() in exts)
    if not pred_paths:
        raise SystemExit(f'No prediction images found in {pred_dir}')

    rows = []
    for pred_path in pred_paths:
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred = read_rgb(pred_path)
        target = read_rgb(gt_path)
        if pred.shape != target.shape:
            target = np.asarray(Image.open(gt_path).convert('RGB').resize(pred.shape[1::-1]), dtype=np.float32) / 255.0
        rows.append((psnr(pred, target), ssim_simple(pred, target), freq_l1(pred, target)))

    if not rows:
        raise SystemExit('No matching prediction/ground-truth filenames were found.')

    metrics = np.asarray(rows)
    print(f'Images: {len(rows)}')
    print(f'PSNR:    {metrics[:, 0].mean():.3f}')
    print(f'SSIM*:   {metrics[:, 1].mean():.4f}')
    print(f'Freq-L1: {metrics[:, 2].mean():.6f}')
    print('*SSIM is a lightweight global implementation; use a full benchmark script for final publication numbers.')


if __name__ == '__main__':
    main()
