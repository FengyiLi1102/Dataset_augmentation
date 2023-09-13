from __future__ import annotations

from collections import defaultdict
from typing import Tuple, List, Dict

import numpy as np

from src.constant import AUGMENTATION, TRAINING, VALIDATION, TESTING, SIMPLE, BACKGROUND
from src.utils import ratio_to_number
from src.typeHint import PointImageType


class Task:
    """
    Each fake synthetic image corresponds to one task.
    """

    def __init__(self):
        self.background_id: int = 0
        self.split: int = 0

        self.n_structure: int = 1

        """
        # following 3 attributes serve for two applications:
        # 1) only one extended structure to make
        #   [[1, 2]
        #    [3, 4]]
        #
        # 2) for multiple extended structures in the background
             [[1, 2, 3, 4],
              [5, 6, 7, 8]]
        """
        self.n_chip: int = 1
        self.component_id: np.ndarray | None | List[int] = []  # (row, col) = id_chip
        self.flip: np.ndarray | None | List[int] = []  # (row, col) = flip
        # self.position: List[np.ndarray] = []

        self.rotation: List[int] = []  # [rotation for all chips in each structure]
        self.required_scale: float = .0

    @classmethod
    def initialise_list(cls,
                        mode: str,
                        num: int,
                        ratio: List[int] = None) -> Tuple[List[Task], List[int] | None]:
        init_list = []
        n_split = None

        if mode == SIMPLE:
            for _ in range(num):
                task = Task()
                # task.component_id = np.zeros(size, dtype=np.uint16)
                # task.flip = np.zeros(size, dtype=np.int8)
                init_list.append(task)
        elif mode == AUGMENTATION:
            if ratio is None or sum(ratio) != 10:
                raise Exception(f"Error: Split ratio for training, validation and testing is not given properly.")

            n_split = ratio_to_number(ratio, num)

            splits_list = [TRAINING for _ in range(n_split[0])] + \
                          [VALIDATION for _ in range(n_split[1])] + \
                          [TESTING for _ in range(n_split[-1])]

            for category in splits_list:
                task = Task()
                task.split = category
                init_list.append(task)

        return init_list, n_split

    def __str__(self):
        print(f"Background img: {self.background_id} \n",
              f"required_scale: {self.required_scale} \n")

        if type(self.component_id) is np.ndarray:
            for idx in range(self.component_id.shape[0]):
                print(f"Structure: {idx} \n")
                print(f"component img: {self.component_id[idx]} \n",
                      f"flip: {self.flip[idx]} \n")

            print(f"================================================")
        else:
            for idx in range(len(self.component_id)):
                print(f"component img: {self.component_id[idx]} \n",
                      f"flip: {self.flip[idx]} \n",
                      f"rotation: {self.rotation[idx]} \n")

            print(f"split: {self.split} \n")
            print(f"================================================")

        return ""
