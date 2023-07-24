from __future__ import annotations

import random
from typing import Tuple, List

from src.constant import AUGMENTATION, TRAINING, VALIDATION, TESTING, SIMPLE, BACKGROUND


class Task:

    def __init__(self):
        self.background_id: int = 0
        self.required_scale: List[float] = []
        self.component_id: List[int] = []
        self.position: List[Tuple[int, int]] = []
        self.flip: List[str] = []
        self.rotation: List[int] = []

        self.split: int = 0

    @classmethod
    def initialise_list(cls, mode: str, num: int, ratio: List[int] = None) -> List[Task]:
        init_list = []

        if mode == SIMPLE:
            for _ in range(num):
                init_list.append(Task())
        elif mode == AUGMENTATION:
            if ratio is None or sum(ratio) != 10:
                raise Exception(f"Error: Split ratio for training, validation and testing is not given properly.")

            splits_list = [TRAINING for _ in range(int(ratio[0] / 10 * num))] + \
                          [VALIDATION for _ in range(int(ratio[1] / 10 * num))] + \
                          [TESTING for _ in range(int(ratio[-1] / 10 * num))]

            for category in splits_list:
                task = Task()
                task.split = category
                init_list.append(task)

        return init_list

    def __str__(self):
        print(f"Background img: {self.background_id}")

        for idx in range(len(self.required_scale)):
            print(f"required_scale: {self.required_scale[idx]} \n",
                  f"component img: {self.component_id[idx]} \n",
                  f"flip: {self.flip[idx]} \n",
                  f"rotation: {self.rotation[idx]} \n")

        print(f"split: {self.split} \n")
        print(f"================================================")

        return ""
