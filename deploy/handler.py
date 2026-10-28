import joblib
import pandas as pd


class EndpointHandler:
    def __init__(self, path=""):
        self.model = joblib.load(f"{path}/model.joblib")

    def __call__(self, data):
        # {"inputs": [[x_0, ..., x_39], ...]}
        X = pd.DataFrame(data["inputs"], columns=self.model.feature_names_in_)
        return {"predictions": self.model.predict(X).tolist()}
