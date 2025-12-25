from typing import List, Callable
from pipeline.schemas import ScreenOut, BioOut, ToxOut
from utils.smiles_validator import validate_smiles_list

def screen_end_to_end(
    smiles_list: List[str],
    bio_fn: Callable[[List[str]], List[BioOut]],
    tox_fn: Callable[[List[str]], List[ToxOut]],
) -> List[ScreenOut]:
    # Validate SMILES before prediction
    is_valid_list, error_messages = validate_smiles_list(smiles_list)
    
    # Filter valid SMILES for prediction
    valid_smiles = [smi for smi, is_valid in zip(smiles_list, is_valid_list) if is_valid]
    valid_indices = [i for i, is_valid in enumerate(is_valid_list) if is_valid]
    
    # Only predict for valid SMILES
    if valid_smiles:
        bio_out = bio_fn(valid_smiles)
        tox_out = tox_fn(valid_smiles)
    else:
        bio_out = []
        tox_out = []
    
    # Create results for all SMILES (valid and invalid)
    results = []
    valid_idx = 0
    
    for i, smi in enumerate(smiles_list):
        is_valid = is_valid_list[i]
        error_msg = error_messages[i]
        
        if not is_valid:
            # Invalid SMILES: create dummy outputs
            results.append(ScreenOut(
                smiles=smi,
                bio=BioOut(p_active=0.0, active=False),
                tox=ToxOut(p_toxic=1.0, non_toxic=False),
                keep=False,
                reason="Invalid SMILES",
                is_valid=False,
                validation_error=error_msg
            ))
        else:
            # Valid SMILES: use actual predictions
            b = bio_out[valid_idx]
            t = tox_out[valid_idx]
            valid_idx += 1
            
            keep = bool(b.active and t.non_toxic)
            if keep:
                reason = "Active & Non-Toxic"
            elif (not b.active) and (not t.non_toxic):
                reason = "Inactive & Toxic"
            elif not b.active:
                reason = "Inactive"
            else:
                reason = "Toxic"

            results.append(ScreenOut(
                smiles=smi,
                bio=b,
                tox=t,
                keep=keep,
                reason=reason,
                is_valid=True,
                validation_error=""
            ))
    
    return results
