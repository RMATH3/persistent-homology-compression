
import argparse
from pathlib import Path

import numpy as np
from gudhi import CubicalComplex
from gudhi.bottleneck import bottleneck_distance
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from skimage.transform import resize


SIZE = 128
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compute_ph_diagram(img: np.ndarray) -> np.ndarray:
    """Compute persistence intervals for 0- and 1-dim using Gudhi."""
    img = np.array(img, dtype=np.float32)
    cc = CubicalComplex(top_dimensional_cells=img)
    cc.persistence()
    return np.vstack((
        cc.persistence_intervals_in_dimension(0),
        cc.persistence_intervals_in_dimension(1)
    ))


def compute_betti_numbers(img: np.ndarray, alpha_values: np.ndarray) -> dict:
    """Compute Betti numbers (β0, β1) for each filtration threshold alpha."""
    ph_diagram = compute_ph_diagram(img)
    betti = {0: [], 1: []}
    
    for alpha in alpha_values:
        for dim in [0, 1]:
            dim_intervals = ph_diagram[ph_diagram[:, 0] != np.inf]  # filter valid intervals
            count = np.sum((dim_intervals[:, 0] <= alpha) & (dim_intervals[:, 1] > alpha))
            betti[dim].append(count)
    
    return betti


def betti_distance(img1: np.ndarray, img2: np.ndarray, p: int = 1, num_alphas: int = 50) -> float:
    """Compute Betti Number Distance using L_p norm over alpha range."""
    alpha_values = np.linspace(0, 1, num_alphas)
    
    betti1 = compute_betti_numbers(img1, alpha_values)
    betti2 = compute_betti_numbers(img2, alpha_values)
    
    # Compute L_p norm for each dimension and combine
    distance = 0.0
    for dim in [0, 1]:
        diff = np.abs(np.array(betti1[dim]) - np.array(betti2[dim]))
        distance += np.sum(diff ** p)
    
    return distance ** (1.0 / p)


def compare_images(original_path: Path, ph_path: Path, jpeg_path: Path) -> dict:
    """Run all four metrics comparing original vs PH vs JPEG."""
    # Load and prepare images
    original = imread(str(original_path))
    if original.ndim == 3 and original.shape[-1] == 4:
        original = original[..., :3]
    original_gray = rgb2gray(original)
    original_resized = resize(original_gray, (SIZE, SIZE), anti_aliasing=True)
    
    ph_img = imread(str(ph_path))
    if ph_img.ndim == 3:
        ph_img = rgb2gray(ph_img)
    ph_resized = resize(ph_img, (SIZE, SIZE), anti_aliasing=True)
    
    jpeg_img = imread(str(jpeg_path))
    if jpeg_img.ndim == 3:
        jpeg_img = rgb2gray(jpeg_img)
    jpeg_resized = resize(jpeg_img, (SIZE, SIZE), anti_aliasing=True)
    
    # Compute metrics
    results = {}
    
    # SSIM (higher is better, max 1.0)
    results["ssim_ph"] = ssim(original_resized, ph_resized, data_range=1.0)
    results["ssim_jpeg"] = ssim(original_resized, jpeg_resized, data_range=1.0)
    
    # MSE (lower is better)
    results["mse_ph"] = mse(original_resized, ph_resized)
    results["mse_jpeg"] = mse(original_resized, jpeg_resized)
    
    # Bottleneck Distance (lower is better)
    original_ph = compute_ph_diagram(original_resized)
    ph_diagram = compute_ph_diagram(ph_resized)
    jpeg_diagram = compute_ph_diagram(jpeg_resized)
    
    results["bottleneck_ph"] = bottleneck_distance(original_ph, ph_diagram)
    results["bottleneck_jpeg"] = bottleneck_distance(original_ph, jpeg_diagram)
    
    # Betti Number Distance (lower is better)
    results["betti_ph"] = betti_distance(original_resized, ph_resized, p=1)
    results["betti_jpeg"] = betti_distance(original_resized, jpeg_resized, p=1)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Compare PH vs JPEG compression using multiple metrics")
    parser.add_argument("--input", type=str, default="input", help="Input folder with original images")
    parser.add_argument("--output", type=str, default="output", help="Output folder with compressed images")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_dir.exists():
        print(f"Input folder '{input_dir}' does not exist.")
        return
    
    if not output_dir.exists():
        print(f"Output folder '{output_dir}' does not exist.")
        return
    
    # Find all original images
    originals = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
    
    if not originals:
        print(f"No images found in '{input_dir}'.")
        return
    
    print("\nComparing PH vs JPEG Compression\n" + "="*60)
    
    for original_path in originals:
        stem = original_path.stem
        ph_path = output_dir / f"{stem}_ph.png"
        jpeg_path = output_dir / f"{stem}_jpeg.jpg"
        
        if not ph_path.exists():
            print(f"Skipping {original_path.name}: PH output not found")
            continue
        
        if not jpeg_path.exists():
            print(f"Skipping {original_path.name}: JPEG output not found")
            continue
        
        print(f"\n{original_path.name}:")
        results = compare_images(original_path, ph_path, jpeg_path)
        
        print(f"  SSIM:       PH={results['ssim_ph']:.4f}  JPEG={results['ssim_jpeg']:.4f}")
        print(f"  MSE:        PH={results['mse_ph']:.6f}  JPEG={results['mse_jpeg']:.6f}")
        print(f"  Bottleneck: PH={results['bottleneck_ph']:.6f}  JPEG={results['bottleneck_jpeg']:.6f}")
        print(f"  Betti Dist: PH={results['betti_ph']:.6f}  JPEG={results['betti_jpeg']:.6f}")


if __name__ == "__main__":
    main()
