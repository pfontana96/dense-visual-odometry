from pathlib import Path
import sys
import os


def add_src_dir_to_path():
    """
        Adds the project 'src' dir to the environment's path in order to allow importing project's modules when testing

    :return: None
    """
    src_dir = Path(__file__).resolve().parent.parent.joinpath("src")

    # Assume running locally
    os.environ["PATH"] += ':' + str(src_dir)
    sys.path.insert(0, str(src_dir))


def add_tests_dir_to_path():
    """
        Adds the project 'tests' dir to the environment's path in order to allow importing project's modules when
        testing

    :return: None
    """
    sys.path.insert(0, Path(__file__).resolve().parent)


add_src_dir_to_path()
add_tests_dir_to_path()
