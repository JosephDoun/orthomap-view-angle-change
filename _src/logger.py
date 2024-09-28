import logging

logger = logging.getLogger(__file__);
logger.setLevel(logging.DEBUG);

logging.basicConfig(
    format='%(asctime)s:%(levelname)s'
    ':%(filename)s:%(processName)s'
    ':%(funcName)s:%(lineno)d: %(message)s',
    level=logging.WARN,
    datefmt='%H:%M:%S %b%d'
)
