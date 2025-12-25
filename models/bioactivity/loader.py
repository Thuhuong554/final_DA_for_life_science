import os
import torch
import joblib
from transformers import AutoTokenizer
from .model_def import ChemBERTaReference, HybridCNNLSTM
from typing import Optional, Tuple, Dict, Any

def load_bioactivity(
    model_dir: str = "checkpoints/bioactivity",
    weights_name: str = "best_reference_chemberta_xai.pth",
    hf_model_name: str = "seyonec/ChemBERTa-zinc-base-v1",
    device: Optional[torch.device] = None,
):
    """
    Load ChemBERTa model for bioactivity prediction
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name)

    model = ChemBERTaReference(model_name=hf_model_name)
    ckpt_path = os.path.join(model_dir, weights_name)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, tokenizer


def load_rf_baseline(
    model_path: str = "experiments/bioactivity_baselines/best_baseline_rf.joblib"
):
    """
    Load Random Forest baseline model
    """
    return joblib.load(model_path)


def load_cnn_lstm(
    model_dir: str = "checkpoints/bioactivity",
    weights_name: str = "best_advanced_model.pth",
    tokenizer_meta_path: str = "experiments/bioactivity_baselines/cnn_tokenizer_meta.joblib",
    device: Optional[torch.device] = None,
) -> Tuple[HybridCNNLSTM, Dict[str, Any]]:
    """
    Load CNN-LSTM model and tokenizer metadata
    
    Returns:
        model: HybridCNNLSTM model
        tokenizer_meta: Dictionary containing char_map, vocab_sz, max_len
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load tokenizer metadata
    tokenizer_meta = joblib.load(tokenizer_meta_path)
    char_map = tokenizer_meta["char_map"]
    vocab_sz = tokenizer_meta["vocab_sz"]
    max_len = tokenizer_meta["max_len"]
    
    # Load model
    model = HybridCNNLSTM(vocab_sz).to(device)
    ckpt_path = os.path.join(model_dir, weights_name)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=True)
    model.eval()
    
    return model, tokenizer_meta
