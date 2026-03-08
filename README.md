# Persistent Homology Compression

NCSSM Morganton Research in Mathematics 2025  
By: Anil Chintapalli, Peter Tenholder, Henry Chen, Arjun Rao

A tidy, single-script workflow for compressing images using persistent homology (PH) to select FFT frequencies.

## Project Layout
- `requirements.txt`: pip dependencies for the project
- `src/compressors/ph_compressor.py`: Persistent homology–based image compressor CLI
- `src/compressors/jpeg_compressor.py`: JPEG compressor CLI targeting a file size
- `src/metrics/metrics.py`: Standard evaluation metrics (SSIM, MSE, Wasserstein distance, Bottleneck distance, file size, compression ratio)
- `src/metrics/topology_metrics.py`: Topological metrics (persistence diagrams, Betti number distance)
- `src/utils/image_utils.py`: Image loading and preprocessing functions
- `src/comparison/compare.py`: Experiment runner that loads images, applies metrics, compares PH vs JPEG outputs, and prints results
- `inputs/images/`: Original source images 
- `outputs/compressed/`: Compressed PH and JPEG images
- `outputs/figures/`: Optional plots or visualizations from experiments

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

### Comparison (PowerShell)
```
python .\src\comparison.py --input input --output output
```

### Comparison (bash/zsh)
```
python ./src/comparison.py --input input --output output
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

Notes (Comparison):
- Expects both `*_ph.png` and `*_jpeg.jpg` outputs for each input image.
- Reports SSIM (higher=better), MSE (lower=better), Bottleneck Distance (lower=better), Betti Number Distance (lower=better).

## What It Does
- PH: Converts to grayscale/resizes, ranks FFT freqs, PH-filters, reconstructs, and writes `*_ph.png`.
- JPEG: Converts to grayscale/resizes, searches quality to hit `--target-kb` (within `--tolerance`), writes `*_jpeg.jpg`.
- Comparison: runs SSIM, MSE, Bottleneck Distance, and Betti Number Distance to compare PH vs JPEG outputs against originals.

## Tips
- PH evaluation is computationally heavy. Start with a small `--size` (e.g., `64`), lower `--max-freqs`, or reduce `--sample-ratio`.
- Gudhi and Pillow install best via `conda-forge` (already configured in `environment.yml`).

## Troubleshooting
- If `input/` is empty or missing, the script reports it and exits.
- If you need JPEG comparison or dataset benchmarks, use a separate branch; this repo now focuses on the single PH workflow.