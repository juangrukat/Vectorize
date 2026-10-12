from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

def render(
    vertices: npt.NDArray[np.float64],
    loops: Sequence[Sequence[int]],
    loop_colors: Sequence[int],
    palette: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> npt.NDArray[np.float64]: ...
def loss_gradient(
    vertices: npt.NDArray[np.float64],
    loops: Sequence[Sequence[int]],
    loop_colors: Sequence[int],
    palette: npt.NDArray[np.float64],
    width: int,
    height: int,
    target: npt.NDArray[np.float64],
) -> tuple[float, npt.NDArray[np.float64]]: ...
