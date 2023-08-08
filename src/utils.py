import os
from collections import defaultdict
from typing import Union, Tuple, Dict, List, Callable

import numpy as np


def mkdir_if_not_exists(target_path: str) -> bool:
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return True
    else:
        return False


def process_labels(param: Union[np.ndarray, Tuple[int, int]],
                   labels: Dict[str, List[np.ndarray]],
                   operation_func: Callable) -> Dict[str, List[np.ndarray]]:
    processed_labels = defaultdict(list)

    for label_type in labels:
        for i in range(len(labels[label_type])):
            processed_labels[label_type].append(operation_func(param, labels[label_type][i]))

    return processed_labels


def ratio_to_number(ratio: List[float], num: int):
    return [int(ratio[0] / 10 * num), int(ratio[1] / 10 * num), int(ratio[-1] / 10 * num)]


def concatenate_txt(dir_1: str, dir_2: str, save_dir: str):
    mkdir_if_not_exists(save_dir)
    # Get the list of txt files in the new label folder
    new_label_txts = [f for f in os.listdir(dir_1) if f.endswith('.txt')]

    for txt in new_label_txts:
        # Check if the file also exists in the existing label files
        if txt in os.listdir(dir_2):
            with open(os.path.join(dir_1, txt), 'r') as f1, \
                    open(os.path.join(dir_2, txt), 'r') as f2, \
                    open(os.path.join(save_dir, txt), 'w') as fout:
                lines1 = f1.readlines()
                lines2 = f2.readlines()
                fout.writelines(lines1)
                fout.write("\n")
                fout.writelines(lines2)


if __name__ == "__main__":
    concatenate_txt(r"../dataset/cropped/labels", r"../dataset/cropped/active_sites", r"../dataset/cropped/final_label")
