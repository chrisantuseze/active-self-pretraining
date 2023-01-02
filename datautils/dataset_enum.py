from enum import Enum

class DatasetType(Enum):
    IMAGENET = 0
    CIFAR10 = 1
    CHEST_XRAY = 2
    REAL = 3
    UCMERCED = 4
    FLOWERS = 5
    EUROSAT = 6
    FOOD101 = 7
    CLIPART = 8
    SKETCH = 9

def get_dataset_enum(value: int):
    if value == DatasetType.IMAGENET.value:
        return "imagenet"

    if value == DatasetType.CIFAR10.value:
        return "cifar10"

    if value == DatasetType.CHEST_XRAY.value:
        return "chest_xray"

    if value == DatasetType.REAL.value:
        return "real"

    if value == DatasetType.UCMERCED.value:
        return "ucmerced"

    if value == DatasetType.FLOWERS.value:
        return "flowers"
    
    if value == DatasetType.EUROSAT.value:
        return "eurosat"

    if value == DatasetType.FOOD101.value:
        return "food101"

    if value == DatasetType.CLIPART.value:
        return "clipart"

    if value == DatasetType.SKETCH.value:
        return "sketch"