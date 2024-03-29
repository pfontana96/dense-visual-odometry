from setuptools import setup, find_packages
from pathlib import Path


def read(filename: str, encoding: str = "utf8"):
    curr_dir = Path(__file__).resolve().parent
    file = curr_dir / filename
    
    if not file.exists():
        raise FileNotFoundError("Could not find file at '{}'".format(str(file)))

    with file.open(encoding=encoding) as fp:
        content = fp.read()

    return content

setup(
    name="Dense Visual Odometry",
    description="Dense Visual Odometry Library",
    license="GNU",
    version=read("VERSION"),
    author="Pedro Fontana",
    author_email="pedro.fontana.1996@gmail.com",
    long_description=read("README.md"),
    url="https://github.com/pfontana96/dense-visual-odometry",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Libraries"
    ],
    python_requires=">=3.7",
    install_requires=read("requirements.txt").splitlines(),
    zip_safe=False
)
