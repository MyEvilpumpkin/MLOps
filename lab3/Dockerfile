FROM python:3.11.7-slim

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

COPY . /app

WORKDIR /app

RUN python3 create_model.py > model_test_report.txt

CMD cat model_test_report.txt && python3 app.py
