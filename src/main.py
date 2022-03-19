import logging

from dense_visual_odometry.core import dummy_function
from dense_visual_odometry.log import set_root_logger


logger = logging.getLogger(__name__)


def main():

    set_root_logger()

    a = 1
    b = 2
    logger.info(f"Calling dummy function with {a} and {b}..")
    c = dummy_function(a, b)
    logger.info(f"Graceful execution with {c} as result :)")


if __name__ == "__main__":
    main()
