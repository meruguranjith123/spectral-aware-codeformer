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


def ssim_global(pred, target):
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
    pred_mag = np.log1p(np.abs(pred_fft))
    target_mag = np.log1p(np.abs(target_fft))
    return np.mean(np.abs(pred_mag - target_mag))


def collect(pred_dir, gt_dir, exts):
    rows = []
    for pred_path in sorted(pred_dir.iterdir()):
        if pred_path.suffix.lower() not in exts:
            continue
        gt_path = gt_dir / pred_path.name
        if not gt_path.exists():
            continue
        pred = read_rgb(pred_path)
        target = read_rgb(gt_path)
        if pred.shape != target.shape:
            target_img = Image.open(gt_path).convert('RGB').resize(pred.shape[1::-1])
            target = np.asarray(target_img, dtype=np.float32) / 255.0
        rows.append((psnr(pred, target), ssim_global(pred, target), freq_l1(pred, target)))
    if not rows:
        raise ValueError(f'No matching images found for {pred_dir} against {gt_dir}')
    return np.asarray(rows)


def summarize(name, metrics):
    return {
        'name': name,
        'count': int(metrics.shape[0]),
        'psnr': float(metrics[:, 0].mean()),
        'ssim': float(metrics[:, 1].mean()),
        'freq_l1': float(metrics[:, 2].mean()),
    }


def main():
    parser = argparse.ArgumentParser(description='Compare no-FFT and FFT-loss reconstruction folders.')
    parser.add_argument('--nofft_dir', required=True, help='Predictions from baseline/no frequency-loss model.')
    parser.add_argument('--fft_dir', required=True, help='Predictions from FFT/frequency-loss model.')
    parser.add_argument('--gt_dir', required=True, help='Ground-truth images with matching filenames.')
    parser.add_argument('--ext', default='png,jpg,jpeg', help='Comma-separated image extensions.')
    args = parser.parse_args()

    exts = {f'.{ext.strip().lower()}' for ext in args.ext.split(',')}
    nofft = summarize('nofft', collect(Path(args.nofft_dir), Path(args.gt_dir), exts))
    fft = summarize('fft', collect(Path(args.fft_dir), Path(args.gt_dir), exts))

    print('| model | images | PSNR up | SSIM up | Freq-L1 down |')
    print('|---|---:|---:|---:|---:|')
    for row in (nofft, fft):
        print(f"| {row['name']} | {row['count']} | {row['psnr']:.3f} | {row['ssim']:.4f} | {row['freq_l1']:.6f} |")

    print('\\nDelta fft - nofft:')
    print(f"PSNR:    {fft['psnr'] - nofft['psnr']:+.3f}")
    print(f"SSIM:    {fft['ssim'] - nofft['ssim']:+.4f}")
    print(f"Freq-L1: {fft['freq_l1'] - nofft['freq_l1']:+.6f}")


if __name__ == '__main__':
    main()
