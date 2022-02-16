FROM python:3.8.2-slim

RUN pip install mlflow==1.21.0 && \
    pip install psycopg2-binary && \
    pip install boto3

RUN mkdir /mnt/mlruns/
RUN chmod 777 -R /mnt/mlruns/

EXPOSE 5000
ENTRYPOINT ["mlflow", "server"]