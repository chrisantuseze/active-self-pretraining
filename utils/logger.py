import logging

def init():
    logging.basicConfig(filename="casl.log", encoding="utf-8", format="%(asctime)s %(levelname)s %(message)s", datefmt="%m-%d-%Y %I:%M:%S %p", level=logging.INFO)
    logging.info("CASL started...")
    
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