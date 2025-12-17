FROM python:3.13

WORKDIR /code

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./main.py /code/
COPY ./data /code/data

CMD sh -c "uvicorn main:app --host 0.0.0.0 --port $PORT"
