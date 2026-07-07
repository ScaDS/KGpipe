from typing import Tuple

import numpy as np


def wilson_interval(p: float, n: int) -> Tuple[float, float]:
    z = 1.96
    return p ± np.sqrt(p * (1 - p) / n) * z

def wald_interval(p: float, n: int) -> Tuple[float, float]:
    return p ± np.sqrt(p * (1 - p) / n) * z