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


def keep_freqs(
    original_image: np.ndarray,
    initial_epsilon: float = 11.040,
    max_freqs: int | None = None,
    decay_rate: float = 1.0,
    sample_ratio: float = 1.0,
):
    """Select FFT frequencies whose single-frequency reconstruction preserves PH diagram within epsilon.

    Returns the masked FFT image and count of kept frequencies.
    """
    original_ph = compute_ph_diagram(original_image)
    fft_image = np.fft.fft2(original_image)
    kept_mask = np.zeros_like(fft_image, dtype=bool)

    height, width = fft_image.shape
    freqs = _unique_freqs((height, width))

    # sort by magnitude (descending) and optionally subsample
    freqs.sort(key=lambda pos: np.abs(fft_image[pos]), reverse=True)
    if 0 < sample_ratio < 1.0:
        freqs = freqs[: max(1, int(len(freqs) * sample_ratio))]

    kept_count = 0
    evaluated = 0
    epsilon = initial_epsilon
    max_freqs = max_freqs or len(freqs)

    for u, v in freqs:
        if evaluated >= max_freqs:
            break

        single_freq = np.zeros_like(fft_image, dtype=complex)
        ci, cj = _conj_index((height, width), (u, v))
        single_freq[u, v] = fft_image[u, v]
        single_freq[ci, cj] = fft_image[ci, cj]

        inv = np.fft.ifft2(single_freq).real
        freq_ph = compute_ph_diagram(inv)

        try:
            dist = wasserstein_distance(original_ph, freq_ph, order=1.0)
        except Exception:
            dist = float("inf")

        if dist < epsilon:
            kept_mask[u, v] = True
            kept_mask[ci, cj] = True
            kept_count += 1

        evaluated += 1
        if evaluated % 100 == 0:
            print(f"Evaluated {evaluated}, kept {kept_count}, epsilon = {epsilon:.3f}")
            epsilon *= decay_rate  # gradually loosen

    return fft_image * kept_mask, kept_count


def process_image(
    input_path: Path,
    output_dir: Path,
    size: int = 128,
    epsilon: float = 11.040,
    max_freqs: int | None = None,
    decay_rate: float = 1.0,
    sample_ratio: float = 1.0,
    save_plot: bool = False,
) -> dict:
    """Process a single image and write PH-compressed output PNG.

    Returns a dict with summary stats.
    """
    print(f"\nProcessing {input_path.name} ...")
    image = imread(str(input_path))
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    gray = rgb2gray(image)
    gray_resized = resize(gray, (size, size), anti_aliasing=True)

    start = time.time()
    kept_fft, kept_count = keep_freqs(
        gray_resized,
        initial_epsilon=epsilon,
        max_freqs=max_freqs,
        decay_rate=decay_rate,
        sample_ratio=sample_ratio,
    )
    compression_time = time.time() - start
    print(f"Compression time: {compression_time:.2f} seconds")

    reconstructed = np.fft.ifft2(kept_fft).real
    clipped = np.clip(reconstructed, 0, 1)
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

    print(f"Saved {out_name} | size: {stats['ph_size_kb']:.2f} KB")
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Persistent Homology-based image compression (batch over a folder)"
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
        "--epsilon",
        type=float,
        default=11.040,
        help="Initial epsilon for PH distance threshold",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=1.0,
        help="Decay rate applied to epsilon every 100 evals",
    )
    parser.add_argument(
        "--max-freqs",
        type=int,
        default=None,
        help="Limit number of frequencies evaluated (performance control)",
    )
    parser.add_argument(
        "--sample-ratio",
        type=float,
        default=1.0,
        help="Fraction of top-magnitude freqs to evaluate (0<r<=1)",
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
        print(f"Input folder '{input_dir}' does not exist. Create it and add images.")
        return

    images = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in exts
    ]
    if not images:
        print(f"No images in '{input_dir}'. Supported: {', '.join(sorted(exts))}")
        return

    summary = []
    for img_path in images:
        stats = process_image(
            img_path,
            output_dir,
            size=args.size,
            epsilon=args.epsilon,
            max_freqs=args.max_freqs,
            decay_rate=args.decay_rate,
            sample_ratio=args.sample_ratio,
        )
        summary.append(stats)

    print("\nSummary:")
    for s in summary:
        print(
            f"- {s['file']}: kept={s['kept_count']} | time={s['compression_time_sec']}s | size={s['ph_size_kb']:.2f}KB -> {s['output_path']}"
        )


if __name__ == "__main__":
    main()
