import mlflow

# client = MlflowClient(tracking_uri="http://localhost:30986")
client = mlflow.tracking.MlflowClient(tracking_uri="http://localhost:30986")

# 獲取該實驗的相關訊息
experiment = client.get_experiment_by_name("hr-resume-selector")
print("Experiment_id: {}".format(experiment.experiment_id))
print("Artifact Location: {}".format(experiment.artifact_location))
print("Tags: {}".format(experiment.tags))
print("Lifecycle_stage: {}".format(experiment.lifecycle_stage))
print("------------------------------")

# 列出這個實驗的所有run_id
run_infos = client.list_run_infos(experiment.experiment_id)
print(run_infos)
print("------------------------------")

# Fetch latest version; this will be version 2
client.create_registered_model("almee-test")
# print("--")
# print_models_info(client.get_latest_versions(name, stages=["None"]))

# client.get
# client.get_model_version_download_uri

# 獲取run_id
# with mlflow.start_run() as run:
#     mlflow.log_param("p", 0)

#     run_id = run.info.run_id
#     print("run_id: {}; lifecycle_stage: {}".format(run_id,
#         mlflow.get_run(run_id).info.lifecycle_stage))


# client.get_experiment("hr-resume-selector")

# Load the model 
# s3://mlflow/1/caf236f4e61b4d688b178603316d9500/artifacts/test-pyfile-model/model.pkl
# model_uri = "runs:/{}/model".format(run.info.run_id)
# loaded_model = mlflow.fastai.load_model(model_uri)
# results = loaded_model.predict(predict_data)

# mlflow.sklearn.log_model(sk_model = model,
#                              artifact_path = 'test-pyfile-model',
#                              registered_model_name = 'test-pyfile-model')