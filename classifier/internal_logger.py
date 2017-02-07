import logging


logger = logging.getLogger('utilization_classifier')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('train.log')
fh.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(message)s')
fh.setFormatter(formatter)
# add the handlers to the logger
logger.addHandler(fh)
