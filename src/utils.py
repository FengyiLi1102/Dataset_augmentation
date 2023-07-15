import os


def mkdir_if_not_exists(target_path: str) -> bool:
    if not os.path.exists(target_path):
        os.makedirs(target_path)
        return True
    else:
        return False
