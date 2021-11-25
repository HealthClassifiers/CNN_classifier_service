FROM python:3.6
ADD . /app
WORKDIR /app
RUN pip --timeout=1000 install -r requirements.txt