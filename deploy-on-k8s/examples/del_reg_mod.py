from mlflow.tracking import MlflowClient

client = MlflowClient(tracking_uri="http://localhost:30986")

client.delete_registered_model(name="test-pyfile-model")