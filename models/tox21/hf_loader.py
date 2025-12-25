from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Optional

def load_tox_hf(
    artifacts_dir: str = "artifacts/admet_chemberta_tox21",
    device: Optional[torch.device] = None,
):
    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifacts_dir)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    model.to(device)
    model.eval()
    return model, tokenizer
