from enum import Enum
from dataclasses import dataclass


@dataclass(order=True)
class Params:
    batch_size: int
    image_size: int
    lr: float
    epochs: int
    optimizer: str
    weight_decay: float
    temperature: float

class TrainingType(Enum):
    SOURCE_PRETRAIN = "Source"
    TARGET_AL = "Target AL"
    TARGET_PRETRAIN = "Target"
    ACTIVE_LEARNING = "Active Learning"
    LINEAR_CLASSIFIER = "Linear Classifier"

