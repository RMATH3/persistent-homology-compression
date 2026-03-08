import argparse
import os
import time
from pathlib import Path

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.io import imread
from skimage.transform import resize

# Parameters from Section 3.1 and 3.4 of the paper
SIZE = 128
MAX_QUALITY = 95  # Clamped maximum quality
MIN_QUALITY = 5  # Clamped minimum quality
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def compress_image_jpeg(
        input_path: Path,
        output_dir: Path,
        target_kb: float,
        tolerance: float = 0.5,
) -> dict:
    """
    Compresses image to JPEG by iteratively adjusting quality to match a
    target file size (KB) for fair comparison with PH compression.
    """
    if not input_path.exists():
        raise FileNotFoundError(f"{input_path} not found")

    # Image Preprocessing: Grayscale and Resize to 128x128
    image = imread(str(input_path))
    if image.ndim == 3:
        if image.shape[-1] == 4:
            image = image[..., :3]
        image = rgb2gray(image)

    gray_resized = resize(image, (SIZE, SIZE), anti_aliasing=True)  # [cite: 267]
    img_uint8 = (gray_resized * 255).astype(np.uint8)
    img_pil = Image.fromarray(img_uint8, mode="L")

    output_dir.mkdir(parents=True, exist_ok=True)
    out_name = input_path.stem + "_jpeg.jpg"
    out_path = output_dir / out_name

    chosen_quality = None
    chosen_size_kb = None
    start_time = time.time()

    # Iterative quality search to match target size
    # We iterate downwards from highest clamped quality
    for q in range(MAX_QUALITY, MIN_QUALITY - 1, -1):
        img_pil.save(out_path, format="JPEG", quality=q, subsampling=0)
        size_kb = os.path.getsize(out_path) / 1024.0

        # Check if we are within the target range
        if abs(size_kb - target_kb) <= tolerance:
            chosen_quality = q
            chosen_size_kb = size_kb
            break

        # If we drop below the target size without hitting the tolerance,
        # the previous quality was likely the closest match.
        if size_kb < (target_kb - tolerance):
            chosen_quality = q
            chosen_size_kb = size_kb
            break

    elapsed = time.time() - start_time

    stats = {
        "file": input_path.name,
        "quality": chosen_quality,
        "size_kb": chosen_size_kb if chosen_size_kb else size_kb,
        "compression_time_sec": float(f"{elapsed:.4f}"),
        "output_path": str(out_path),
    }

    if chosen_quality is None:
        print(f"  {input_path.name}: Closest match reached quality {q} ({size_kb:.2f} KB)")
    else:
        print(
            f"  {input_path.name}: Quality {chosen_quality} matched target ({chosen_size_kb:.2f} KB)")

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="JPEG compression matching target KB size (Paper Section 3.4)"
    )
    parser.add_argument(
        "--input", type=str, default="input", help="Folder containing original images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Folder for matched JPEG outputs",
    )
    parser.add_argument(
        "--target-kb",
        type=float,
        required=True,
        help="Target size in KB (should match PH output size)"
    )
    parser.add_argument(
        "--tolerance", type=float, default=0.5, help="Allowed +/- KB difference"
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"Input folder '{input_dir}' not found.")
        return

    images = [
        p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS
    ]

    if not images:
        print(f"No valid images found in {input_dir}")
        return

    print(f"Targeting size: {args.target_kb} KB...")
    summary = []
    for img_path in images:
        stats = compress_image_jpeg(
            img_path,
            output_dir,
            target_kb=args.target_kb,
            tolerance=args.tolerance,
        )
        summary.append(stats)

    print("\nBatch Compression Summary:")
    for s in summary:
        print(
            f"- {s['file']}: Quality={s['quality']} | Size={s['size_kb']:.2f}KB | Time={s['compression_time_sec']}s"
        )


if __name__ == "__main__":
    main()
