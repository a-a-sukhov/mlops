import argparse
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from catboost import CatBoostClassifier, Pool
from clearml import Dataset, Task
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split


def get_git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True
        ).strip()
    except Exception:
        return "not_git_repo"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_id", required=True)

    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=42)

    args = parser.parse_args()

    task = Task.init(
        project_name="students-demo",
        task_name="catboost-text-classification"
    )

    params = {
        "dataset_id": args.dataset_id,
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "git_commit": get_git_commit(),
    }

    task.connect(params)

    logger = task.get_logger()

    print("Using dataset_id:", args.dataset_id)

    dataset = Dataset.get(dataset_id=args.dataset_id)
    dataset_path = Path(dataset.get_local_copy())

    print("Dataset local path:", dataset_path)

    csv_files = list(dataset_path.rglob("*.csv"))
    if not csv_files:
        raise FileNotFoundError("No CSV file found in ClearML Dataset")

    csv_path = csv_files[0]
    print("Using CSV:", csv_path)

    df = pd.read_csv(csv_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("CSV must contain columns: text, label")

    df = df[["text", "label"]].dropna()

    df["label"] = df["label"].map({
        "ham": 0,
        "spam": 1,
        0: 0,
        1: 1
    })

    df = df.dropna()
    df["label"] = df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df[["text"]],
        df["label"],
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=df["label"]
    )

    train_pool = Pool(
        data=X_train,
        label=y_train,
        text_features=["text"]
    )

    test_pool = Pool(
        data=X_test,
        label=y_test,
        text_features=["text"]
    )

    model = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        eval_metric="F1",
        verbose=20,
        random_seed=args.random_state
    )

    model.fit(train_pool, eval_set=test_pool)

    preds = model.predict(test_pool)
    preds = [int(x) for x in preds]

    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds)

    print("accuracy:", accuracy)
    print("f1:", f1)

    logger.report_scalar(
        title="metrics",
        series="accuracy",
        value=accuracy,
        iteration=args.iterations
    )

    logger.report_scalar(
        title="metrics",
        series="f1",
        value=f1,
        iteration=args.iterations
    )

    cm = confusion_matrix(y_test, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["ham", "spam"])
    ax.set_yticklabels(["ham", "spam"])

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")

    cm_path = Path("confusion_matrix.png")
    fig.tight_layout()
    fig.savefig(cm_path)
    plt.close(fig)

    logger.report_image(
        title="confusion_matrix",
        series="catboost",
        iteration=args.iterations,
        local_path=str(cm_path)
    )

    model_path = Path("catboost_sms_spam_model.cbm")
    model.save_model(model_path)

    task.upload_artifact(
        name="catboost_model",
        artifact_object=str(model_path)
    )

    task.upload_artifact(
        name="confusion_matrix_png",
        artifact_object=str(cm_path)
    )

    print("Model artifact saved:", model_path)
    print("Done")


if __name__ == "__main__":
    main()