from enum import Enum

class SSL_Method(Enum):
    SIMCLR = 0
    DCL = 1
    SWAV = 2

def get_ssl_method(value: int):
    if value == SSL_Method.SIMCLR.value:
        prefix = "simclr"

    elif value == SSL_Method.DCL.value:
        prefix = "dcl"

    elif value == SSL_Method.SWAV.value:
        prefix = "swav"

    return prefix