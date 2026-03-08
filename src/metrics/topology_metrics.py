import numpy as np
from gudhi import CubicalComplex

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
