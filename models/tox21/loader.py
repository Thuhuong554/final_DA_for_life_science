import joblib

def load_tox_rf(model_path: str):
    return joblib.load(model_path)
