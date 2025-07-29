FROM astronomerinc/ap-airflow:2.2.3-ubuntu20.04
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
