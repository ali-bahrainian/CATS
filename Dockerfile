FROM python:3.8

WORKDIR /api

RUN python3 -m pip install --upgrade pip

COPY requirements.txt .

RUN pip3 install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python3", "./api.py"]