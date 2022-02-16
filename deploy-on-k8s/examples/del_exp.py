import mlflow
import os

# 設置 mlflow address
mlflow.set_tracking_uri('http://localhost:30986')

# 設置 minio endpoint
os.environ['MLFLOW_S3_ENDPOINT_URL'] = 'http://localhost:32197'
os.environ['AWS_ACCESS_KEY_ID'] = 'minio'
os.environ['AWS_SECRET_ACCESS_KEY'] = 'minio123'

mlflow.delete_experiment(3)