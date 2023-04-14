from enum import Enum

class DatasetType(Enum):
    IMAGENET = 0
    CIFAR10 = 1
    CHEST_XRAY = 2
    REAL = 3
    UCMERCED = 4
    EUROSAT = 5
    FLOWERS = 6
    HAM10000 = 7
    CLIPART = 8
    SKETCH = 9
    QUICKDRAW = 10
    MODERN_OFFICE_31 = 11

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
    
    if value == DatasetType.EUROSAT.value:
        return "eurosat"

    if value == DatasetType.FLOWERS.value:
        return "flowers"

    if value == DatasetType.HAM10000.value:
        return "ham10000"

    if value == DatasetType.CLIPART.value:
        return "clipart"

    if value == DatasetType.SKETCH.value:
        return "sketch"

    if value == DatasetType.QUICKDRAW.value:
        return "quickdraw"

    if value == DatasetType.MODERN_OFFICE_31.value:
        return "modern_office_31"