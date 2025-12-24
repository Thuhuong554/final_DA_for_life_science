import torch
from typing import List
from pipeline.schemas import BioOut

@torch.no_grad()
def predict_bioactivity(
    smiles_list: List[str],
    model,
    tokenizer,
    device: str = "cpu",
    max_length: int = 128,
    tau_bio: float = 0.5,
):
    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    logits = model(input_ids=input_ids, attention_mask=attention_mask)
    probs = torch.sigmoid(logits).detach().cpu().tolist()

    outs = []
    for p in probs:
        outs.append(BioOut(p_active=float(p), active=bool(p > tau_bio), xai=None))
    return outs
