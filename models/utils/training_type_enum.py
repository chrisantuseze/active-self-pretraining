from enum import Enum
from dataclasses import dataclass


@dataclass(order=True)
class Params:
    batch_size: int
    image_size: int
    lr: float
    epochs: int
    weight_decay: float
    name: str

class TrainingType(Enum):
    BASE_PRETRAIN = "Base"
    SOURCE_PRETRAIN = "Source"
    TARGET_PRETRAIN = "Target"
    ACTIVE_LEARNING = "Active Learning"
    LINEAR_CLASSIFIER = "Linear Classifier"

