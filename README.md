# dense-visual-odometry
[![Coverage Status](https://coveralls.io/repos/github/pfontana96/dense-visual-odometry/badge.png?branch=main)](https://coveralls.io/github/pfontana96/dense-visual-odometry?branch=main)

Python implemented Dense Visual Odometry

## Generate documentation
This repo uses [Numpy docstrings](https://numpydoc.readthedocs.io/en/latest/format.html). To generate the documentation locally as HTML pages one can use [pydoctor](https://pydoctor.readthedocs.io/en/latest/):

```bash
python3 -m venv venv
source venv/bin/activate
pip install pydoctor==22.5.1
pydoctor --project-name=dense-visual-odometry --project-version=0.0.1 --project-url=https://github.com/pfontana96/dense-visual-odometry --make-html --html-output=docs --project-base-dir="src/dense_visual_odometry/" --docformat=numpy src/dense_visual_odometry/
```
which will generate the code documentation under a directory named `docs` (ommitted on .gitignore)
