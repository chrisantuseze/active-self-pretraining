from enum import Enum

class DatasetType(Enum):
    CLIPART = 0
    SKETCH = 1
    QUICKDRAW = 2
    PAINTING = 3

    AMAZON = 4
    WEBCAM = 5
    DSLR = 6

    ARTISTIC = 7
    CLIP_ART = 8
    PRODUCT = 9
    REAL_WORLD = 10

def get_dataset_enum(value: int):
    if value == DatasetType.CLIPART.value:
        return "clipart"

    if value == DatasetType.SKETCH.value:
        return "sketch"

    if value == DatasetType.QUICKDRAW.value:
        return "quickdraw"

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