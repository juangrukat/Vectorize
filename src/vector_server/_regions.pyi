import numpy as np
import numpy.typing as npt

def ward(
    image: npt.NDArray[np.float64],
    threshold: float,
    min_pixels: int,
    max_regions: int,
    refine_passes: int = 2,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.float64]]: ...
