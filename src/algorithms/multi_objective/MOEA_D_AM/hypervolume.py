from pymoo.indicators.hv import HV
import numpy as np

def hypervolume(front: np.ndarray, reference_point: np.ndarray) -> float:
    hv = HV(ref_point=reference_point)
    hv_value = hv.do(-front)

    return hv_value