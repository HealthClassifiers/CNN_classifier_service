FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip --timeout=100000 install -r requirements.txt