import argparse
import os
from pathlib import Path

import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse

from src.metrics.topology_metrics import compute_ph_diagram, betti_distance

from src.utils.image_utils import load_and_prep
from src.metrics.metrics import ssim_metric, mse_metric, wasserstein_metric, bottleneck_metric, file_size_kb, compression_ratio

from src.compressors.ph_compressor import process_image
from src.compressors.jpeg_compressor import compress_image_jpeg
from pathlib import Path
import ot



SIZE = 128
EXTENSIONS = {".png", ".jpg", ".jpeg"}



def compare_images(original_path: Path, ph_path: Path, jpeg_path: Path) -> dict:
    """
    Evaluate the compressed images against the original using the six metrics 
    from the paper.
    """
    # Load and Preprocess
    

    orig = load_and_prep(original_path)
    ph = load_and_prep(ph_path)
    jpeg = load_and_prep(jpeg_path)

    pd_orig = compute_ph_diagram(orig)
    pd_ph = compute_ph_diagram(ph)
    pd_jpeg = compute_ph_diagram(jpeg)

    results = {}

    # 1 & 2: Visual Metrics
    results["ssim_ph"] = ssim_metric(orig, ph)
    results["ssim_jpeg"] = ssim_metric(orig, jpeg)

    results["mse_ph"] = mse_metric(orig, ph)
    results["mse_jpeg"] = mse_metric(orig, jpeg)

    results["wasser_ph"] = wasserstein_metric(pd_orig, pd_ph)
    results["wasser_jpeg"] = wasserstein_metric(pd_orig, pd_jpeg)

    results["bottle_ph"] = bottleneck_metric(pd_orig, pd_ph)
    results["bottle_jpeg"] = bottleneck_metric(pd_orig, pd_jpeg)

    results["size_ph"] = file_size_kb(ph_path)
    results["size_jpeg"] = file_size_kb(jpeg_path)

    results["betti_ph"] = betti_distance(pd_orig, pd_ph)
    results["betti_jpeg"] = betti_distance(pd_orig, pd_jpeg)

    results["compression_ratio_ph"] = compression_ratio(original_path, ph_path)
    results["compression_ratio_jpeg"] = compression_ratio(original_path, jpeg_path)

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PH vs JPEG using paper metrics")
    parser.add_argument("--input", type=str, default="inputs/images", help="Original images folder")
    parser.add_argument("--output", type=str, default="outputs/compressed", help="Compressed images folder")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    originals = [p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS]

    print(f"{'Image':<20} | {'Metric':<12} | {'PH':<10} | {'JPEG':<10}")
    print("-" * 60)

    for orig_p in originals:
        ph_stats = process_image(orig_p, output_dir)
        ph_p = Path(ph_stats["output_path"])



        jpeg_stats = compress_image_jpeg(
            input_path=orig_p, output_dir=output_dir, target_kb=ph_stats["ph_size_kb"], tolerance=0.5)
        jpeg_p = Path(jpeg_stats["output_path"])

        res = compare_images(orig_p, ph_p, jpeg_p)
        print(f"{orig_p.name:<20}")
        print(f"{'':<20} | SSIM         | {res['ssim_ph']:<10.4f} | {res['ssim_jpeg']:<10.4f}")
        print(f"{'':<20} | MSE          | {res['mse_ph']:<10.6f} | {res['mse_jpeg']:<10.6f}")
        print(f"{'':<20} | Wasserstein  | {res['wasser_ph']:<10.4f} | {res['wasser_jpeg']:<10.4f}")
        print(f"{'':<20} | Bottleneck   | {res['bottle_ph']:<10.4f} | {res['bottle_jpeg']:<10.4f}")
        print(f"{'':<20} | Betti Dist   | {res['betti_ph']:<10.4f} | {res['betti_jpeg']:<10.4f}")
        print(f"{'':<20} | Size (KB)    | {res['size_ph']:<10.2f} | {res['size_jpeg']:<10.2f}")
        print("-" * 60)

if __name__ == "__main__":
    main()