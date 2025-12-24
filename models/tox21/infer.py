import numpy as np
from typing import List
from pipeline.schemas import ToxOut
from .featurize import morgan_fp

def predict_tox_rf(
    smiles_list: List[str],
    rf_model,
    tau_tox: float = 0.5,
):
    X = np.stack([morgan_fp(s) for s in smiles_list], axis=0)
    p = rf_model.predict_proba(X)[:, 1].astype(float)  # class=1 assumed "toxic"
    outs = []
    for pi in p:
        outs.append(ToxOut(p_toxic=float(pi), non_toxic=bool(pi < tau_tox), xai=None))
    return outs
