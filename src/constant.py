from collections import defaultdict
from typing import NewType, Dict, List

import numpy as np

# DatabaseManager, Task
DNA_AUGMENTATION = "DNA_augmentation"
BACKGROUND = "background"
COMPONENT = "component"
AUGMENTATION = "augmentation"
SIMPLE = "simple"
AUGMENTED = "augmented"
MOSAICS = "mosaics"
CROPPED = "cropped"
RAW = "raw"

# TaskAssigner
TRAINING = 0
VALIDATION = 1
TESTING = 2

V = 0
H = 1
N = -1

flip_convertor = {
    V: "v",
    H: "h",
    N: "n"
}

split_convertor = {
    0: "train",
    1: "val",
    2: "test"
}

# run
GENERATE_FAKE_BACKGROUND = "generate_fake_backgrounds"
CROP_ORIGAMI = "crop_origami"
RUN = "run"
CREATE_CACHE = "create_cache"
GENERATE_EXTENDED_STRUCTURE = "generate_extended_structure"

# DataLoader
DNA_ORIGAMI = "DNA-origami"
ACTIVE_SITE = "active_site"

# New types
cv2_image = NewType("cv2_image", np.ndarray)


if __name__ == "__main__":
    l = {"a": [1, 2], "b": [2, 3]}
    print(type(l))
    print(type(l) == dict[str, list[int]])

