FROM python:3.9-slim-buster

RUN apt-get update; apt-get upgrade -y; apt-get install git python3-opencv -y

WORKDIR dense-visual-odometry

COPY . .

RUN python -m pip install --no-cache-dir --upgrade pip
RUN pip install -r requirements.txt --no-cache-dir

ENTRYPOINT ["/bin/bash"]
