from typing import List, Tuple
import torch
import torch.nn.functional as F

@torch.no_grad()
def _tokenize_single(smiles: str, tokenizer, device: str = "cpu", max_length: int = 128):
    enc = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}

def explain_bioactivity(
    smiles: str,
    model,
    tokenizer,
    device: str = "cpu",
    max_length: int = 128
) -> Tuple[List[str], List[float]]:
    """
    Token-level saliency for Bioactivity ChemBERTa model.

    Returns:
        tokens: list of token strings
        scores: list of importance scores (higher => more contribution)
    Notes:
        - This is a gradient-based saliency on input embeddings.
        - It is suitable for showing token-level attribution in the app/report.
    """
    model.eval()
    model.to(device)

    # Build inputs
    enc = tokenizer(
        smiles,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # We need gradients, so do not use @torch.no_grad here
    # Get embeddings and enable grad
    # Works for Roberta-like models (ChemBERTa)
    emb_layer = model.bert.embeddings.word_embeddings
    inputs_embeds = emb_layer(input_ids)
    inputs_embeds = inputs_embeds.detach().requires_grad_(True)  # make it a leaf
    inputs_embeds.retain_grad() # retain gradients

    # Forward with inputs_embeds
    outputs = model.bert(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask
    )
    cls_emb = outputs.last_hidden_state[:, 0, :]  # [CLS]
    logits = model.classifier(cls_emb).squeeze(-1)  # [B]
    prob = torch.sigmoid(logits)  # [B]

    # Backprop: d(prob)/d(embeds)
    model.zero_grad(set_to_none=True)
    prob.backward(torch.ones_like(prob))

    grads = inputs_embeds.grad  # [B, T, H]
    if grads is None:
        raise RuntimeError("XAI failed: inputs_embeds.grad is None. Gradient graph not retained.")
    # Saliency score per token: L2 norm of gradient
    scores = torch.norm(grads, dim=-1).squeeze(0)  # [T]

    # Convert ids -> tokens
    tokens = tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

    # Mask out padding tokens (attention_mask=0)
    am = attention_mask.squeeze(0).float()
    scores = scores * am

    # Convert to python list
    scores_list = scores.detach().cpu().tolist()

    return tokens, scores_list
