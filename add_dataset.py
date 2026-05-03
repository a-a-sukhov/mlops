from clearml import Dataset

dataset = Dataset.create(
    dataset_project="students-datasets",
    dataset_name="sms-spam-catboost",
    dataset_version="1.0.0",
)

dataset.add_files("/Users/andreisuhov/Desktop/сode/mlops/data/sms_spam.csv")
dataset.upload()
dataset.finalize()

print("Dataset ID:", dataset.id)