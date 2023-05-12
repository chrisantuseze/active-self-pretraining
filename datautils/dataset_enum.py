from enum import Enum

class DatasetType(Enum):
    CHEST_XRAY = 0
    EUROSAT = 1
    FLOWERS = 2
    HAM10000 = 3
    
    CLIPART = 4
    SKETCH = 5
    QUICKDRAW = 6
    PAINTING = 7

    MODERN_OFFICE_31 = 8

    AMAZON = 9
    WEBCAM = 10
    DSLR = 11

    ARTISTIC = 12
    CLIP_ART = 13
    PRODUCT = 14
    REAL_WORLD = 15

    MNIST = 16
    MNIST_M = 17
    SVHN = 18
    USPS = 19
    SYN_DIGITS = 20

def get_dataset_enum(value: int):
    if value == DatasetType.CHEST_XRAY.value:
        return "chest_xray"
    
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

    if value == DatasetType.AMAZON.value:
        return "amazon"

    if value == DatasetType.WEBCAM.value:
        return "webcam"

    if value == DatasetType.DSLR.value:
        return "dslr"

    if value == DatasetType.PAINTING.value:
        return "painting"

    if value == DatasetType.ARTISTIC.value:
        return "artistic"

    if value == DatasetType.CLIP_ART.value:
        return "clip_art"

    if value == DatasetType.PRODUCT.value:
        return "product"

    if value == DatasetType.REAL_WORLD.value:
        return "real_world"

    if value == DatasetType.MNIST.value:
        return "mnist"

    if value == DatasetType.MNIST_M.value:
        return "mnist_m"

    if value == DatasetType.SVHN.value:
        return "svhn"

    if value == DatasetType.USPS.value:
        return "usps"

    if value == DatasetType.SYN_DIGITS.value:
        return "syn_digits"