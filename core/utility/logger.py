import logging
import sys

def getLogger(name, log_path, log_file_name):
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # std out
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # file
    file_handler = logging.FileHandler("{0}/{1}.log".format(log_path, log_file_name))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger