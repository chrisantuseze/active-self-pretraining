import logging
import os
import sys


def init():
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler(stream=sys.stdout)
    logger.addHandler(handler)

    if not os.path.isdir("save/misc"):
        os.makedirs("save/misc")

    logging.basicConfig(filename="save/misc/casl.log", format="%(asctime)s %(levelname)s %(message)s", datefmt="%m-%d-%Y %I:%M:%S %p", level=logging.INFO)
    logging.info("CASL started...")

    def handle_exception(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        logging.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))

    sys.excepthook = handle_exception
    
def debug(message: str):
    print(message)
    logging.debug(message)

def info(message: str):
    print(message)
    logging.info(message)

def warn(message: str):
    print(message)
    logging.warn(message)

def error(message: str):
    print(message)
    logging.error(message)
