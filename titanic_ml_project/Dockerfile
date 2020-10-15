FROM python:3.8

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

WORKDIR /usr/src/app

COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

COPY . /usr/src/app

CMD uvicorn main:app