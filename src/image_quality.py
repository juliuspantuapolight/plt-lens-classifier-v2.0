"""Lightweight, configurable blur assessment."""
from __future__ import annotations

import cv2
import numpy as np


def laplacian_variance(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def assess_blur(image: np.ndarray, threshold: float) -> dict:
    score = laplacian_variance(image)
    return {"laplacian_variance": score, "threshold": threshold, "is_severely_blurred": score < threshold}
