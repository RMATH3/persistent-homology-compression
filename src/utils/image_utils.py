from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize
from src.config import SIZE

def load_and_prep(path):
        img = imread(str(path))
        if img.ndim == 3:
            img = rgb2gray(img[..., :3])
        return resize(img, (SIZE, SIZE), anti_aliasing=True)