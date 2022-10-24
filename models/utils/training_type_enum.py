from enum import Enum
from dataclasses import dataclass, field


@dataclass(order=True)
class Params:
    batch_size: int
    image_size: int
    lr: int
    epochs: int

class TrainingType(Enum):
    BASE_PRETRAIN = 0
    TARGET_PRETRAIN = 1
    ACTIVE_LEARNING = 2
    FINETUNING = 3
    AL_FINETUNING = 4

