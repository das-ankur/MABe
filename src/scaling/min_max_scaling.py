from cachetools import LRUCache
import numpy as np


# ---------- Global LRU Caches (size = 3) ----------
_SCALING_CACHE = LRUCache(maxsize=3)   # Stores global min/max per tracking file



def compute_global_min_max(df, file_path):
    """
    Compute and cache global min/max for x and y coordinates.
    """
    if file_path in _SCALING_CACHE:
        return _SCALING_CACHE[file_path]

    # Only consider valid (non -1) entries
    valid_df = df[(df["x"] != -1) & (df["y"] != -1)]
    if valid_df.empty:
        stats = {"x_min": 0, "x_max": 1, "y_min": 0, "y_max": 1}
    else:
        stats = {
            "x_min": valid_df["x"].min(),
            "x_max": valid_df["x"].max(),
            "y_min": valid_df["y"].min(),
            "y_max": valid_df["y"].max(),
        }

    _SCALING_CACHE[file_path] = stats
    return stats


def min_max_scaling(values, min_val, max_val):
    """
    Perform Min-Max normalization using provided global min/max values,
    ignoring entries with value -1 (treated as missing).
    """
    arr = np.array(values, dtype=np.float32)  # Explicitly use float32
    mask = arr != -1
    if not np.any(mask):
        return arr

    # Add small epsilon to prevent division by zero
    eps = 1e-7
    denom = max_val - min_val + eps
    
    # Clip input values to prevent extreme outliers
    arr[mask] = np.clip(arr[mask], min_val - eps, max_val + eps)
    
    # Perform scaling
    arr[mask] = (arr[mask] - min_val) / denom
    
    # Clip output to [0, 1] range for safety
    arr[mask] = np.clip(arr[mask], 0, 1)
    
    # Replace any potential NaN or Inf values with -1 (missing)
    arr[~np.isfinite(arr)] = -1
    
    return arr