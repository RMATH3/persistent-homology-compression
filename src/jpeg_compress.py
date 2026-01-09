import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize


SIZE = 128
MAX_QUALITY = 100
MIN_QUALITY = 1
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compress_image_jpeg(
    input_path: Path,
    output_dir: Path,
    target_kb: float,
    tolerance: float = 0.5,
) -> dict:
    """Compress a single image to a JPEG near the target size (KB)."""
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    image = imread(str(input_path))
    if image.ndim == 3 and image.shape[-1] == 4:
        image = image[..., :3]

    gray = rgb2gray(image)
    gray_resized = resize(gray, (SIZE, SIZE), anti_aliasing=True)
    img_uint8 = (gray_resized * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode="L")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = input_path.stem + "_jpeg.jpg"
    out_path = output_dir / out_name

    chosen_quality = None
    chosen_size_kb = None
    elapsed = None

    for q in range(MAX_QUALITY, MIN_QUALITY - 1, -1):
        start = time.time()
        img_pil.save(out_path, format="JPEG", quality=q)
        elapsed = time.time() - start
        size_kb = os.path.getsize(out_path) / 1024.0
        if abs(size_kb - target_kb) <= tolerance:
            chosen_quality = q
            chosen_size_kb = size_kb
            break

    stats = {
        "file": input_path.name,
        "quality": chosen_quality,
        "size_kb": chosen_size_kb,
        "compression_time_sec": float(f"{elapsed:.4f}" if elapsed else "0.0"),
        "output_path": str(out_path),
    }

    if chosen_quality is None:
        print(
            f"Failed to match target size for {input_path.name}; last size {size_kb:.2f} KB"
        )
    else:
        print(
            f"{input_path.name}: quality {chosen_quality} -> {chosen_size_kb:.2f} KB in {elapsed:.2f}s"
        )

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="JPEG compression to a target file size (batch over a folder)"
    )
    parser.add_argument(
        "--input", type=str, default="input", help="Input folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output folder for compressed JPEGs",
    )
    parser.add_argument(
        "--target-kb", type=float, default=5.0, help="Target output size in KB"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.5, help="Allowed +/- KB difference"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    exts = EXTENSIONS

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
        stats = compress_image_jpeg(
            img_path,
            output_dir,
            target_kb=args.target_kb,
            tolerance=args.tolerance,
        )
        summary.append(stats)

    print("\nSummary:")
    for s in summary:
        print(
            f"- {s['file']}: quality={s['quality']} | size={s['size_kb']:.2f}KB | time={s['compression_time_sec']}s -> {s['output_path']}"
        )


if __name__ == "__main__":
    main()
