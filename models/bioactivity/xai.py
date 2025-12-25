from typing import List, Tuple, Dict, Any, Optional, Set
import torch
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Draw
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image


@torch.no_grad()
def _tokenize_single(
    smiles: str,
    tokenizer,
    device: Optional[torch.device] = None,
    max_length: int = 128
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    enc = tokenizer(
        smiles,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    return {k: v.to(device) for k, v in enc.items()}


def explain_bioactivity_chemberta(
    smiles: str,
    model,
    tokenizer,
    device: Optional[torch.device] = None,
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
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
    inputs_embeds.retain_grad()  # retain gradients

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


def explain_bioactivity_rf(
    smiles: str,
    rf_model,
    top_k: int = 5
) -> Tuple[List[str], List[float], Set[int], Chem.Mol]:
    """
    Explain bioactivity using Random Forest fingerprint bit importance.
    
    Returns:
        atom_indices: list of atom indices (as strings) that contribute most
        scores: list of importance scores
        highlight_atoms: set of atom indices to highlight in visualization
        mol: RDKit molecule object for visualization
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    
    # Fingerprint with bit info
    bitInfo = {}
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
        mol, radius=2, nBits=1024, bitInfo=bitInfo
    )
    
    # RF feature importance
    importances = rf_model.feature_importances_
    
    # Top contributing bits
    top_bits = np.argsort(importances)[::-1][:top_k]
    
    # Collect atom indices involved
    highlight_atoms = set()
    bit_scores = {}
    
    for b in top_bits:
        if b in bitInfo:
            score = importances[b]
            for atom_idx, radius in bitInfo[b]:
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                for bond_id in env:
                    bond = mol.GetBondWithIdx(bond_id)
                    highlight_atoms.add(bond.GetBeginAtomIdx())
                    highlight_atoms.add(bond.GetEndAtomIdx())
                # Also track scores for aggregation
                if atom_idx not in bit_scores:
                    bit_scores[atom_idx] = []
                bit_scores[atom_idx].append(score)
    
    # Aggregate scores per atom (only for atoms that have scores)
    atom_indices = []
    scores = []
    for atom_idx in sorted(highlight_atoms):
        if atom_idx in bit_scores:
            atom_indices.append(str(atom_idx))
            scores.append(float(np.mean(bit_scores[atom_idx])))
    
    return atom_indices, scores, highlight_atoms, mol


def explain_bioactivity_cnn_lstm(
    smiles: str,
    model,
    tokenizer_meta: Dict[str, Any],
    device: Optional[torch.device] = None
) -> Tuple[List[str], List[float]]:
    """
    Character-level saliency for CNN-LSTM model.
    
    Returns:
        characters: list of character strings
        scores: list of importance scores (normalized)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    char_map = tokenizer_meta["char_map"]
    max_len = tokenizer_meta["max_len"]
    
    # Fast tokenize function
    def fast_tokenize(smiles_str):
        seq = [char_map.get(c, 0) for c in smiles_str]
        return seq + [0] * (max_len - len(seq)) if len(seq) < max_len else seq[:max_len]
    
    was_training = model.training
    model.train()  # Need gradients
    model.zero_grad()
    
    seq = torch.tensor(
        [fast_tokenize(smiles)],
        dtype=torch.long,
        device=device
    )
    
    # Forward
    out = model(seq)
    
    # Backward (scalar output → gradient)
    out.backward()
    
    # Get gradient at embedding layer
    emb_grad = model.embedding.weight.grad.detach()
    
    seq_ids = seq[0].cpu().numpy()
    
    saliency = emb_grad[seq_ids].norm(dim=1).cpu().numpy()
    saliency = saliency / (saliency.max() + 1e-8)
    
    # Set model back to original state
    if not was_training:
        model.eval()
    
    # Convert to characters (only for actual SMILES length)
    characters = list(smiles)
    scores = saliency[:len(smiles)].tolist()
    
    return characters, scores


def visualize_rf_molecule(mol: Chem.Mol, highlight_atoms: Set[int], size: Tuple[int, int] = (400, 400)) -> Image.Image:
    """
    Visualize molecule with highlighted atoms for RF XAI.
    
    Returns:
        PIL Image of the molecule
    """
    img = Draw.MolToImage(mol, highlightAtoms=list(highlight_atoms), size=size)
    return img


def visualize_cnn_lstm_saliency(smiles: str, saliency_scores: List[float], figsize: Tuple[int, int] = (14, 3)) -> Image.Image:
    """
    Visualize CNN-LSTM character-level saliency as bar chart.
    
    Returns:
        PIL Image of the bar chart
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(range(len(smiles)), saliency_scores[:len(smiles)])
    ax.set_xticks(range(len(smiles)))
    ax.set_xticklabels(list(smiles), rotation=0, fontsize=8)
    ax.set_ylabel("Saliency")
    ax.set_title("CNN–LSTM Character-level Saliency")
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = BytesIO()
    plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def explain_bioactivity(
    smiles: str,
    model,
    tokenizer=None,
    device: Optional[torch.device] = None,
    max_length: int = 128,
    model_type: str = "chemberta",
    rf_model=None,
    tokenizer_meta: Optional[Dict[str, Any]] = None
) -> Tuple[List[str], List[float], Optional[Any]]:
    """
    Unified XAI function for bioactivity models.
    
    Args:
        smiles: Input SMILES string
        model: Model to explain (ChemBERTa, CNN-LSTM, or None for RF)
        tokenizer: Tokenizer for ChemBERTa (optional)
        device: PyTorch device
        max_length: Max sequence length for ChemBERTa
        model_type: One of "chemberta", "cnn_lstm", "rf"
        rf_model: Random Forest model (required if model_type="rf")
        tokenizer_meta: Tokenizer metadata for CNN-LSTM (required if model_type="cnn_lstm")
    
    Returns:
        tokens/characters/atom_indices: list of strings
        scores: list of importance scores
        visualization_data: Optional visualization data (mol+highlight_atoms for RF, None for others)
    """
    if model_type == "chemberta":
        if tokenizer is None:
            raise ValueError("tokenizer is required for ChemBERTa model")
        items, scores = explain_bioactivity_chemberta(smiles, model, tokenizer, device, max_length)
        return items, scores, None
    elif model_type == "cnn_lstm":
        if tokenizer_meta is None:
            raise ValueError("tokenizer_meta is required for CNN-LSTM model")
        items, scores = explain_bioactivity_cnn_lstm(smiles, model, tokenizer_meta, device)
        return items, scores, scores  # Return scores for visualization
    elif model_type == "rf":
        if rf_model is None:
            raise ValueError("rf_model is required for Random Forest model")
        items, scores, highlight_atoms, mol = explain_bioactivity_rf(smiles, rf_model)
        return items, scores, (mol, highlight_atoms)  # Return mol and highlight_atoms for visualization
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of: chemberta, cnn_lstm, rf")
