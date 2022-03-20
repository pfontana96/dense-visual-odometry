FROM python:3.9-slim-buster

WORKDIR dense-visual-odometry

COPY src src
COPY tests tests
COPY requirements.txt requirements.txt

RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["/bin/bash"]