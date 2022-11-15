from enum import Enum

class AL_Method(Enum):
    LEAST_CONFIDENCE = 0
    ENTROPY = 1
    BOTH = 2

def get_al_method_enum(value: int):
    if value == AL_Method.LEAST_CONFIDENCE.value:
        return "least_confidence"

    if value == AL_Method.ENTROPY.value:
        return "entropy"

    if value == AL_Method.BOTH.value:
        return "both"