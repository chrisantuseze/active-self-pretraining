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

def get_dataset_info(value: int):
    dataset = {
        DatasetType.CLIPART.value: [345, "clipart", "/clipart"],
        DatasetType.SKETCH.value: [345, "sketch", "/sketch"],
        DatasetType.QUICKDRAW.value: [345, "quickdraw", "/quickdraw"],
        DatasetType.PAINTING.value: [345, "painting", "/painting"],
        DatasetType.AMAZON.value: [31, "amazon", "/office-31/amazon/images"],
        DatasetType.WEBCAM.value: [31, "webcam", "/office-31/webcam/images"],
        DatasetType.DSLR.value: [31, "dslr", "/office-31/dslr/images"],
        DatasetType.ARTISTIC.value:  [65, "artistic", "/officehome/artistic"],
        DatasetType.CLIP_ART.value:  [65, "clip_art", "/officehome/clip_art"],
        DatasetType.PRODUCT.value:  [65, "product", "/officehome/product"],
        DatasetType.REAL_WORLD.value:  [65, "real_world", "/officehome/real_world"]
    }
    return dataset[value]