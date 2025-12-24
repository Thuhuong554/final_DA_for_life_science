from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

def load_tox_hf(
    artifacts_dir: str = "artifacts/admet_chemberta_tox21",
    device: str = "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(artifacts_dir)
    model = AutoModelForSequenceClassification.from_pretrained(artifacts_dir)
    model.to(device)
    model.eval()
    return model, tokenizer
