from typing import List, Callable
from pipeline.schemas import ScreenOut, BioOut, ToxOut

def screen_end_to_end(
    smiles_list: List[str],
    bio_fn: Callable[[List[str]], List[BioOut]],
    tox_fn: Callable[[List[str]], List[ToxOut]],
) -> List[ScreenOut]:
    bio_out = bio_fn(smiles_list)
    tox_out = tox_fn(smiles_list)

    results = []
    for smi, b, t in zip(smiles_list, bio_out, tox_out):
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
            reason=reason
        ))
    return results
