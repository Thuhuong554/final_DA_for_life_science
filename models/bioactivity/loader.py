import os
import torch
from transformers import AutoTokenizer
from .model_def import ChemBERTaReference

def load_bioactivity(
    model_dir: str = "bioactivity",
    weights_name: str = "best_reference_chemberta_xai.pth",
    hf_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: str = "cpu",
):
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    model = ChemBERTaReference(model_name=hf_model_name)
    ckpt_path = os.path.join(model_dir, weights_name)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer
