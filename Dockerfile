FROM apache/airflow:2.7.1-python3.10

COPY requirements.txt /
USER airflow
RUN pip install --no-cache-dir --user -r /requirements.txt
RUN pip install awscli