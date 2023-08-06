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

# DataLoader
DNA_ORIGAMI = "DNA-origami"
ACTIVE_SITE = "active_site"
