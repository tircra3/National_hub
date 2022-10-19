FROM python:3.7.13

COPY . /workspace
WORKDIR /workspace


RUN apt-get update
RUN apt-get install tk -y
RUN apt-get install -qqy x11-apps
RUN pip install --upgrade pip setuptools==45.2.0
RUN pip install -r requirements.txt

CMD ["python","model_ui.py"]