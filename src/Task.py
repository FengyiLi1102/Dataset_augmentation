from __future__ import annotations

from typing import Tuple, List

from src.constant import AUGMENTATION


class Task:
    _required_scale: float
    _background_id: int
    _component_id: int
    _position: Tuple[float, float]
    _flip: str
    _rotation: int

    none_placeholder = {
        "Required_scale": None,
        "Background_id": None,
        "Component_id": None,
        "Position": None,
        "Flip": None,
        "Rotation": None
    }

    def __init__(self, task_type: str, **kwargs):
        if task_type == "augmentation":
            if len(kwargs) != 0 and len(kwargs) != 6:
                raise Exception(f"Error: Provide {len(kwargs)} is not enough to form a augmented task")

            try:
                self._required_scale = kwargs["Required_scale"]
                self._background_id = kwargs["Background_id"]
                self._component_id = kwargs["Component_id"]
                self._position = kwargs["Position"]
                self._flip = kwargs["Flip"]
                self._rotation = kwargs["Rotation"]
            except Exception as e:
                raise Exception(f"{e} due to missing data provided")
        else:
            # TODO: for background and component tasks
            exit()

    @classmethod
    def initialise_list(cls, mode: str, num: int) -> List[Task]:
        init_list = []

        if mode == AUGMENTATION:
            for _ in range(num):
                init_list.append(Task(AUGMENTATION, **cls.none_placeholder))

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

    def __str__(self):
        return f"required_scale: {self.required_scale} \n" \
               f"background img: {self.background_id} \n" \
               f"component img: {self.component_id} \n" \
               f"position: {self.position} \n" \
               f"flip: {self.flip} \n" \
               f"rotation: {self.rotation} \n" \
               f"================================================"


if __name__ == "__main__":
    test = Task.initialise_list(10)
    test[0].rotation = 10
    print(test[0].rotation)
