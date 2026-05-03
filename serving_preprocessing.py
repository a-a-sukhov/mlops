from typing import Any, Optional, Callable
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool


class Preprocess:
    def __init__(self):
        self.model = None
        self.labels = {
            0: "ham",
            1: "spam",
        }

    def load(self, local_file_name: str):
        path = Path(local_file_name)

        if path.is_dir():
            cbm_files = list(path.rglob("*.cbm"))
            if not cbm_files:
                raise FileNotFoundError(f"No .cbm model file found in {path}")
            path = cbm_files[0]

        print(f"Loading CatBoost model from: {path}")

        self.model = CatBoostClassifier()
        self.model.load_model(blob=path.read_bytes())

        return self.model

    def preprocess(
        self,
        body: dict,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> Any:
        if "text" in body:
            texts = [body["text"]]
        elif "texts" in body:
            texts = body["texts"]
        else:
            raise ValueError('Request must contain "text" or "texts"')

        if not isinstance(texts, list):
            raise ValueError('"texts" must be a list')

        texts = [str(x) for x in texts]
        state["texts"] = texts

        return Pool(
            data=pd.DataFrame({"text": texts}),
            text_features=["text"],
        )

    def process(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn: Optional[Callable[[dict], None]] = None,
    ) -> Any:
        preds = self.model.predict(data)
        probs = self.model.predict_proba(data)

        return {
            "preds": preds,
            "probs": probs,
        }

    def postprocess(
        self,
        data: Any,
        state: dict,
        collect_custom_statistics_fn=None,
    ) -> dict:
        preds = [int(x) for x in data["preds"].reshape(-1).tolist()]
        probs = data["probs"].tolist()

        predictions = []

        for text, pred, prob in zip(state["texts"], preds, probs):
            predictions.append({
                "text": text,
                "class_id": pred,
                "class_name": self.labels.get(pred, str(pred)),
                "probability_ham": float(prob[0]),
                "probability_spam": float(prob[1]),
            })

        return {
            "predictions": predictions
        }