FROM python:3.8

ENV PYTHONUNBUFFERED=1

WORKDIR /usr/src/app

COPY requirements.txt .

RUN pip3 install -r requirements.txt

EXPOSE 8000

COPY . .

CMD uvicorn main:app --reload --host 0.0.0.0

