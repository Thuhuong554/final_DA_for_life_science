import torch
from typing import List
from pipeline.schemas import ToxOut
from typing import Optional

@torch.no_grad()
def predict_tox_hf(
    smiles_list: List[str],
    model,
    tokenizer,
    device: Optional[torch.device] = None,
    max_length: int = 128,
    tau_tox: float = 0.5,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = tokenizer(
        smiles_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # shape: [B, num_labels] OR [B, 1]

    # Case A: binary with single logit -> sigmoid
    if logits.ndim == 2 and logits.size(-1) == 1:
        probs_toxic = torch.sigmoid(logits.squeeze(-1))  # [B]
    # Case B: binary with 2 logits -> softmax, take class=1 as "toxic"
    elif logits.ndim == 2 and logits.size(-1) == 2:
        probs = torch.softmax(logits, dim=-1)  # [B, 2]
        probs_toxic = probs[:, 1]  # [B]
    else:
        raise ValueError(f"Unexpected logits shape: {tuple(logits.shape)}")

    probs_toxic = probs_toxic.detach().cpu().tolist()

    outs = []
    for p in probs_toxic:
        outs.append(
            ToxOut(
                p_toxic=float(p),
                non_toxic=bool(p < tau_tox),
                xai=None
            )
        )
    return outs
