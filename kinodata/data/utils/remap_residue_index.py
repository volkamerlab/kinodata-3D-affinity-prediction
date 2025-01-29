import numpy as np

BLANK_CHR = "_"


def compute_residue_reindex(
    pocket_sequence: str,
) -> np.ndarray:
    reindex = np.arange(len(pocket_sequence) + 1)
    j = 1
    for chr in pocket_sequence:
        if chr == BLANK_CHR:
            reindex[j:] += 1
        else:
            j += 1
    return reindex


def remap_residue_index(
    pocket_sequence: str,
    residue_index: list[int],
) -> list[int]:
    if (num_blanks := pocket_sequence.count(BLANK_CHR)) == 0:
        return residue_index
    if (85 - max(residue_index)) != num_blanks:
        print(
            f"Warning: 85 - max(residue_index) = {85 - max(residue_index)}",
            f"is not equal to num_blanks ({BLANK_CHR}) = {num_blanks}",
            f"for pocket sequence {pocket_sequence}",
        )
        return residue_index
    remap = compute_residue_reindex(pocket_sequence)
    return remap[residue_index]
