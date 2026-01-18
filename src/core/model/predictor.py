import pickle
import os

class Predictor:
    def __init__(self, model_path='model.pkl'):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                return pickle.load(f)
        return None

    def predict(self, features):
        if self.model:
            score = self.model.predict(features)
            return int(round(score[0]))
        return None
