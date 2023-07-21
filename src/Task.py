from __future__ import annotations

import random
from typing import Tuple, List

from src.constant import AUGMENTATION, TRAINING, VALIDATION, TESTING, SIMPLE, BACKGROUND


class Task:
    _required_scale: float = None
    _background_id: int = None
    _component_id: int = None
    _position: Tuple[float, float] = None
    _flip: str = None
    _rotation: int = None
    _split: int = None

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

    @property
    def required_scale(self):
        return self._required_scale

    @required_scale.setter
    def required_scale(self, value):
        self._required_scale = value

    @property
    def background_id(self):
        return self._background_id

    @background_id.setter
    def background_id(self, value):
        self._background_id = value

    @property
    def component_id(self):
        return self._component_id

    @component_id.setter
    def component_id(self, value):
        self._component_id = value

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def flip(self):
        return self._flip

    @flip.setter
    def flip(self, value):
        self._flip = value

    @property
    def rotation(self):
        return self._rotation

    @rotation.setter
    def rotation(self, value):
        self._rotation = value

    @property
    def split(self):
        return self._split

    @split.setter
    def split(self, value):
        self._split = value

    def __str__(self):
        return f"required_scale: {self.required_scale} \n" \
               f"background img: {self.background_id} \n" \
               f"component img: {self.component_id} \n" \
               f"position: {self.position} \n" \
               f"flip: {self.flip} \n" \
               f"rotation: {self.rotation} \n" \
               f"split: {self._split} \n" \
               f"================================================"


if __name__ == "__main__":
    test = Task.initialise_list(10)
    test[0].rotation = 10
    print(test[0].rotation)
