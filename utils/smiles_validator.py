"""
SMILES validation utilities
"""
from typing import Tuple, List
from rdkit import Chem


def validate_smiles(smiles: str) -> Tuple[bool, str]:
    """
    Validate a SMILES string.
    
    Args:
        smiles: SMILES string to validate
    
    Returns:
        is_valid: True if SMILES is valid, False otherwise
        error_message: Error message if invalid, empty string if valid
    """
    if not smiles or not isinstance(smiles, str):
        return False, "SMILES is empty or not a string"
    
    # Strip whitespace
    smiles = smiles.strip()
    
    if not smiles:
        return False, "SMILES is empty after stripping whitespace"
    
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False, "Invalid SMILES: Cannot parse molecule"
        
        # Additional validation: check if molecule has at least one atom
        if mol.GetNumAtoms() == 0:
            return False, "Invalid SMILES: Molecule has no atoms"
        
        return True, ""
    except Exception as e:
        return False, f"Invalid SMILES: {str(e)}"


def validate_smiles_list(smiles_list: List[str]) -> Tuple[List[bool], List[str]]:
    """
    Validate a list of SMILES strings.
    
    Args:
        smiles_list: List of SMILES strings to validate
    
    Returns:
        is_valid_list: List of boolean values indicating validity
        error_messages: List of error messages (empty string if valid)
    """
    is_valid_list = []
    error_messages = []
    
    for smiles in smiles_list:
        is_valid, error_msg = validate_smiles(smiles)
        is_valid_list.append(is_valid)
        error_messages.append(error_msg)
    
    return is_valid_list, error_messages

