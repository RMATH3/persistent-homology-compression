# Persistent Homology Compression

NCSSM Morganton Research in Mathematics 2025  
By: Anil Chintapalli, Arjun Rao, Henry Chen, Peter Tenholder

A tidy, single-script workflow for compressing images using persistent homology (PH) to select FFT frequencies.

## Project Layout
- `environment.yml`: Conda environment spec (Python, NumPy, SciPy, scikit-image, Matplotlib, Gudhi, Pillow)
- `src/ph_compress.py`: Single CLI for batch PH compression
- `src/jpeg_compress.py`: JPEG compressor targeting a file size
- `src/comparison.py`: Placeholder for PH vs JPEG metrics (to be filled in later)
- `input/`: Put source images here (`.png`, `.jpg`, `.jpeg`)
- `output/`: Compressed images are written here (`*_ph.png`, `*_jpeg.jpg`)

## Setup (Conda)
```
conda env create -f environment.yml
conda activate ph
```

## Workflow
1. Place images in `input/` (e.g., `input/cat.png`).
2. Run the CLI to process all images and write outputs to `output/`.

### Windows PowerShell (PH)
```
python .\src\ph_compress.py --input input --output output --size 128 --epsilon 11.040 --decay-rate 1.0 --max-freqs 16384 --sample-ratio 1.0
```

### macOS/Linux (bash/zsh) (PH)
```
python ./src/ph_compress.py --input input --output output --size 128 --epsilon 11.040 --decay-rate 1.0 --max-freqs 16384 --sample-ratio 1.0
```

Alternative (module run, PH):
```
python -m src.ph_compress --input input --output output --size 128 --epsilon 11.040 --decay-rate 1.0 --max-freqs 16384 --sample-ratio 1.0
```

### JPEG compression (PowerShell)
```
python .\src\jpeg_compress.py --input input --output output --target-kb 5.0 --tolerance 0.5
```

### JPEG compression (bash/zsh)
```
python ./src/jpeg_compress.py --input input --output output --target-kb 5.0 --tolerance 0.5
```

Notes (PH):
- `--size`: images are resized to `NxN` (default `128`).
- `--epsilon`: PH threshold for accepting a single frequency.
- `--decay-rate`: multiplies `epsilon` every 100 evaluations (default `1.0` means no decay).
- `--max-freqs`: performance control; default evaluates all frequencies (`N*N`). For speed, reduce this (e.g., `4096`).
- `--sample-ratio`: evaluate only the top fraction of magnitudes before PH checks (e.g., `0.25`).
- `--extensions`: comma-separated list of input file types.

Notes (JPEG):
- `--target-kb`: desired output size in KB; `--tolerance` is +/- KB allowed.
- Quality search runs from 100 down to 1 until within tolerance.

## What It Does
- PH: Converts to grayscale/resizes, ranks FFT freqs, PH-filters, reconstructs, and writes `*_ph.png`.
- JPEG: Converts to grayscale/resizes, searches quality to hit `--target-kb` (within `--tolerance`), writes `*_jpeg.jpg`.
- Comparison: placeholder for future PH vs JPEG metrics.

## Tips
- PH evaluation is computationally heavy. Start with a small `--size` (e.g., `64`), lower `--max-freqs`, or reduce `--sample-ratio`.
- Gudhi and Pillow install best via `conda-forge` (already configured in `environment.yml`).

## Troubleshooting
- If `input/` is empty or missing, the script reports it and exits.
- If you need JPEG comparison or dataset benchmarks, use a separate branch; this repo now focuses on the single PH workflow.