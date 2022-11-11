from enum import Enum

class DatasetType(Enum):
    IMAGENET = 0
    CIFAR10 = 1
    QUICKDRAW = 2
    SKETCH = 3
    CLIPART = 4
    UCMERCED = 5
    IMAGENET_LITE = 6
    CHEST_XRAY = 7

def get_dataset_enum(value: int):
    if value == DatasetType.IMAGENET.value:
        return "imagenet"

    if value == DatasetType.CIFAR10.value:
        return "cifar10"

    if value == DatasetType.QUICKDRAW.value:
        return "quickdraw"

    if value == DatasetType.SKETCH.value:
        return "sketch"

    if value == DatasetType.CLIPART.value:
        return "clipart"

    if value == DatasetType.UCMERCED.value:
        return "ucmerced"