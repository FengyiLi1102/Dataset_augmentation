from typing import Dict, List, Tuple

import numpy as np

LabelsType = Dict[str, List[np.ndarray]]
StitchLabelType = Dict[Tuple[int, int], LabelsType]

PointImageType = Tuple[int, int]                                 # All positive integers in an array coordinate
PointCoordinateType = Tuple[float, float]                        # Cartesian coordinate


