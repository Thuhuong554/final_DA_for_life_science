from dataclasses import dataclass
from typing import Any, Optional

@dataclass
class BioOut:
    p_active: float
    active: bool
    xai: Optional[Any] = None

@dataclass
class ToxOut:
    p_toxic: float
    non_toxic: bool
    xai: Optional[Any] = None

@dataclass
class ScreenOut:
    smiles: str
    bio: BioOut
    tox: ToxOut
    keep: bool
    reason: str
    is_valid: bool = True
    validation_error: str = ""
