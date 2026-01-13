import argparse
import os
from pathlib import Path

import numpy as np
from gudhi import CubicalComplex, bottleneck_distance, wasserstein_distance
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.transform import resize


SIZE = 128
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compute_ph_diagram(img: np.ndarray) -> np.ndarray:
    """
    Compute persistence intervals for 0- and 1-dim using GUDHI.
    Matches Algorithm 1 in Section 4.
    """
    img = np.array(img, dtype=np.float32)
    cc = CubicalComplex(top_dimensional_cells=img)
    cc.persistence()
    return np.vstack((
        cc.persistence_intervals_in_dimension(0),
        cc.persistence_intervals_in_dimension(1)
    ))


def betti_distance(img1: np.ndarray, img2: np.ndarray, p: int = 1, num_alphas: int = 100) -> float:
    """
    Compute Betti Number Distance using the Lp norm of the difference 
    between Betti curves as defined in Section 2.4.
    """
    alpha_values = np.linspace(0, 1, num_alphas)
    pd1 = compute_ph_diagram(img1)
    pd2 = compute_ph_diagram(img2)

    # Calculate integral of absolute difference over the filtration
    total_diff = 0.0
    for alpha in alpha_values:
        # Count Betti numbers (holes) active at threshold alpha
        b1 = np.sum((pd1[:, 0] <= alpha) & (pd1[:, 1] > alpha))
        b2 = np.sum((pd2[:, 0] <= alpha) & (pd2[:, 1] > alpha))
        total_diff += np.abs(b1 - b2) ** p

    # Return the normalized Lp norm [cite: 250, 255]
    return (total_diff / num_alphas) ** (1.0 / p)


def compare_images(original_path: Path, ph_path: Path, jpeg_path: Path) -> dict:
    """
    Evaluate the compressed images against the original using the six metrics 
    from the paper.
    """
    # Load and Preprocess
    def load_and_prep(path):
        img = imread(str(path))
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        return resize(img, (SIZE, SIZE), anti_aliasing=True)

    orig = load_and_prep(original_path)
    ph = load_and_prep(ph_path)
    jpeg = load_and_prep(jpeg_path)

    pd_orig = compute_ph_diagram(orig)
    pd_ph = compute_ph_diagram(ph)
    pd_jpeg = compute_ph_diagram(jpeg)

    results = {}

    # 1 & 2: Visual Metrics
    results["ssim_ph"] = ssim(orig, ph, data_range=1.0)
    results["ssim_jpeg"] = ssim(orig, jpeg, data_range=1.0)
    results["mse_ph"] = mse(orig, ph)
    results["mse_jpeg"] = mse(orig, jpeg)

    # 3: Wasserstein Distance (p=1)
    results["wasser_ph"] = wasserstein_distance(pd_orig, pd_ph, order=1.0)
    results["wasser_jpeg"] = wasserstein_distance(pd_orig, pd_jpeg, order=1.0)

    # 4: Bottleneck Distance
    results["bottle_ph"] = bottleneck_distance(pd_orig, pd_ph)
    results["bottle_jpeg"] = bottleneck_distance(pd_orig, pd_jpeg)

    # 5: Betti Number Distance (Integral/Lp Norm)
    results["betti_ph"] = betti_distance(orig, ph, p=1)
    results["betti_jpeg"] = betti_distance(orig, jpeg, p=1)

    # 6: File Size (KB)
    results["size_ph"] = os.path.getsize(ph_path) / 1024.0
    results["size_jpeg"] = os.path.getsize(jpeg_path) / 1024.0

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate PH vs JPEG using paper metrics")
    parser.add_argument("--input", type=str, default="input", help="Original images folder")
    parser.add_argument("--output", type=str, default="output", help="Compressed images folder")
    args = parser.parse_args()

    input_dir, output_dir = Path(args.input), Path(args.output)
    originals = [p for p in input_dir.iterdir() if p.suffix.lower() in EXTENSIONS]

    print(f"{'Image':<20} | {'Metric':<12} | {'PH':<10} | {'JPEG':<10}")
    print("-" * 60)

    for orig_p in originals:
        ph_p, jpeg_p = output_dir / f"{orig_p.stem}_ph.png", output_dir / f"{orig_p.stem}_jpeg.jpg"
        if not ph_p.exists() or not jpeg_p.exists():
            continue

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