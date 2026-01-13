import argparse
import os
import time
from pathlib import Path

import numpy as np
from gudhi import CubicalComplex
from gudhi.wasserstein import wasserstein_distance
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage import img_as_ubyte
from scipy.ndimage import gaussian_filter


def compute_ph_diagram(img: np.ndarray) -> np.ndarray:
    """Compute persistence intervals for 0- and 1-dim using Gudhi."""
    img = np.array(img, dtype=np.float32)
    cc = CubicalComplex(top_dimensional_cells=img)
    cc.persistence()
    return np.vstack(
        (
            cc.persistence_intervals_in_dimension(0),
            cc.persistence_intervals_in_dimension(1),
        )
    )


def _conj_index(shape: tuple[int, int], idx: tuple[int, int]) -> tuple[int, int]:
    """Return the conjugate-symmetric index for an FFT coordinate."""
    h, w = shape
    i, j = idx
    return (-i) % h, (-j) % w


def _unique_freqs(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Iterate only the unique half-plane freqs for a real image FFT."""
    h, w = shape
    freqs = []
    for i in range(h):
        for j in range(w):
            ci, cj = _conj_index(shape, (i, j))
            if (i, j) == (0, 0) or (i, j) <= (ci, cj):
                freqs.append((i, j))
    return freqs


def rank_and_mask_freqs(
        original_image: np.ndarray,
        alpha: float = 0.3,
):
    """
    Ranks frequencies by Importance Score (W1 * 1/f) and retains top alpha%.

    Returns the masked FFT image and count of kept frequencies.
    """
    original_ph = compute_ph_diagram(original_image)
    fft_image = np.fft.fft2(original_image)
    height, width = fft_image.shape
    freqs = _unique_freqs((height, width))

    scores = {}
    print(f"Ranking {len(freqs)} unique frequencies...")

    for idx, (u, v) in enumerate(freqs):
        # Reconstruct using only this frequency pair [cite: 276]
        single_freq_mask = np.zeros_like(fft_image, dtype=complex)
        ci, cj = _conj_index((height, width), (u, v))
        single_freq_mask[u, v] = fft_image[u, v]
        single_freq_mask[ci, cj] = fft_image[ci, cj]

        inv = np.fft.ifft2(single_freq_mask).real
        freq_ph = compute_ph_diagram(inv)

        # Importance Score formula[cite: 281]:
        # score = W1(D_full, D_single) * (1 / sqrt(fx^2 + fy^2))
        try:
            dist = wasserstein_distance(original_ph, freq_ph, order=1.0)
        except Exception:
            dist = 0.0

        freq_norm = np.sqrt(u ** 2 + v ** 2) + 1e-8
        # Higher score = more important [cite: 286]
        scores[(u, v)] = dist * (1.0 / freq_norm)

        if idx % 500 == 0:
            print(f"  Ranked {idx}/{len(freqs)} frequencies...")

    # Sort frequencies by score in descending order [cite: 287]
    ranked_freqs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # Retain the top alpha percentage
    keep_count_limit = max(1, int(len(ranked_freqs) * alpha))
    kept_mask = np.zeros_like(fft_image, dtype=complex)

    actual_kept_count = 0
    for i in range(keep_count_limit):
        (u, v), _ = ranked_freqs[i]
        ci, cj = _conj_index((height, width), (u, v))
        kept_mask[u, v] = fft_image[u, v]
        kept_mask[ci, cj] = fft_image[ci, cj]
        actual_kept_count += 1

    return fft_image * (kept_mask != 0), actual_kept_count


def process_image(
        input_path: Path,
        output_dir: Path,
        size: int = 128,
        alpha: float = 0.3,
) -> dict:
    """Process a single image using the Importance Score ranking[cite: 311]."""
    print(f"\nProcessing {input_path.name} with alpha={alpha:.2f} ...")
    image = imread(str(input_path))
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = rgb2gray(image)  # [cite: 261]

    gray_resized = resize(image, (size, size), anti_aliasing=True)  # [cite: 267]

    start = time.time()
    kept_fft, kept_count = rank_and_mask_freqs(
        gray_resized,
        alpha=alpha
    )
    compression_time = time.time() - start

    # Inverse FFT to reconstruct [cite: 191]
    reconstructed = np.fft.ifft2(kept_fft).real

    # Apply Gaussian smoothing (sigma=1) to stabilize topological features
    smoothed = gaussian_filter(reconstructed, sigma=1.0)

    clipped = np.clip(smoothed, 0, 1)
    uint8_img = img_as_ubyte(clipped)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = input_path.stem + "_ph.png"
    out_path = output_dir / out_name
    imsave(str(out_path), uint8_img)
    ph_size = os.path.getsize(out_path)

    stats = {
        "file": input_path.name,
        "kept_count": int(kept_count),
        "compression_time_sec": float(f"{compression_time:.4f}"),
        "ph_size_kb": float(f"{ph_size / 1024:.4f}"),
        "output_path": str(out_path),
    }

    print(f"Saved {out_name} | size: {stats['ph_size_kb']:.2f} KB | Kept: {kept_count}")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="PH-based image compression using Equation 3 Importance Ranking [cite: 1]"
    )
    parser.add_argument(
        "--input", type=str, default="input", help="Input folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output folder for compressed images",
    )
    parser.add_argument("--size", type=int, default=128, help="Resize dimension (NxN)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.3,
        help="Percentage of top frequencies to retain (0.0 to 1.0) [cite: 319]",
    )
    parser.add_argument(
        "--extensions",
        type=str,
        default=".png,.jpg,.jpeg",
        help="Comma-separated image extensions to process",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    exts = {e.strip().lower() for e in args.extensions.split(",")}

    if not input_dir.exists():
        print(f"Input folder '{input_dir}' does not exist.")
        return

    images = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]

    summary = []
    for img_path in images:
        stats = process_image(
            img_path,
            output_dir,
            size=args.size,
            alpha=args.alpha,
        )
        summary.append(stats)

    print("\nSummary:")
    for s in summary:
        print(
            f"- {s['file']}: kept={s['kept_count']} | time={s['compression_time_sec']}s | size={s['ph_size_kb']:.2f}KB"
        )


if __name__ == "__main__":
    main()
