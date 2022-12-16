from enum import Enum
from dataclasses import dataclass, field


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
    BASE_PRETRAIN = "Base"
    TARGET_PRETRAIN = "Target"
    ACTIVE_LEARNING = "Active Learning"
    FINETUNING = "Finetuning"
    AL_FINETUNING = "AL Finetuning"

