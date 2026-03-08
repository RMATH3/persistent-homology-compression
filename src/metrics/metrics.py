import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error as mse
from gudhi import wasserstein_distance, bottleneck_distance


# 1. Structural Similarity (SSIM)
def ssim_metric(original, reconstructed):
    return ssim(original, reconstructed, data_range=1.0)


# 2. Mean Squared Error (MSE)
def mse_metric(original, reconstructed):
    return mse(original, reconstructed)


# 3. Wasserstein Distance 
def wasserstein_metric(pd1, pd2, order=1.0):
    return wasserstein_distance(pd1, pd2, order=order)


# 4. Bottleneck Distance
def bottleneck_metric(pd1, pd2):
    return bottleneck_distance(pd1, pd2)


# 5. File size in KB
def file_size_kb(path):
    return os.path.getsize(path) / 1024.0


# 6. Compression Ratio
def compression_ratio(original_path, compressed_path):
    return os.path.getsize(original_path) / os.path.getsize(compressed_path)