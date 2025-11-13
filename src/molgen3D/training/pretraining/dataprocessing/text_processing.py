# pretraining/data/textproc/tagged_units.py
from __future__ import annotations
from typing import Optional
import torch
from typing import List, Iterator, Tuple, Dict

def build_unit(canonical: str, embedded: str, newline: bool = False) -> str:
    """Compose one training unit with your legacy tags."""
    s = f"[SMILES]{canonical}[/SMILES][CONFORMER]{embedded}[/CONFORMER]"
    return s + ("\n" if newline else "")

def is_valid_unit(
    canonical: str,
    embedded: str,
    *,
    min_emb_len: int = 16,
    max_chars: Optional[int] = None,
) -> bool:
    """Very fast pre-filter to skip obviously bad lines before tokenization."""
    if not canonical or not embedded:
        return False
    if len(embedded) < min_emb_len:
        return False
    if max_chars is not None and (len(canonical) + len(embedded)) > max_chars:
        return False
    return True

class ChunkPacker:
    def __init__(self, seq_len: int, bos_id: int, eos_id: int):
        self.seq_len = int(seq_len)
        self.need = self.seq_len + 1
        self.bos_id = int(bos_id)
        self.eos_id = int(eos_id)
        self.buf: List[int] = []  # start each chunk empty
        self.pending_unit: List[int] = []  # partial unit being continued across chunks

    def try_add_unit(self, toks: List[int]) -> bool:
        """Add as much of the unit as possible to fill the chunk.
        Units already start with EOS, so no separators needed between them.
        Returns True if entire unit was added, False if cut off."""
        # First, continue any pending unit from previous chunk
        if self.pending_unit:
            remaining_space = self.need - len(self.buf)
            if remaining_space > 0:
                add_count = min(remaining_space, len(self.pending_unit))
                self.buf.extend(self.pending_unit[:add_count])
                self.pending_unit = self.pending_unit[add_count:]

                # If pending unit completed, no need to add separator since next unit starts with EOS

        # Now add the new unit (each unit already starts with EOS)
        remaining_space = self.need - len(self.buf)
        if remaining_space > 0:
            # Add as many tokens as possible from the unit
            add_count = min(remaining_space, len(toks))
            if add_count > 0:
                self.buf.extend(toks[:add_count])
                remaining_toks = toks[add_count:]

                if remaining_toks:
                    # Unit was cut off, save remainder for next chunk
                    self.pending_unit = remaining_toks
                    return False  # Partial unit

                # Unit completed fully
                return True  # Complete unit

        # Couldn't add anything
        return False

    def start_next_block_with(self, toks: List[int]):
        """Start a new block with remaining partial unit."""
        self.buf = []
        self.pending_unit = toks[:]  # Store the unit to continue

        # Try to add as much as possible of the pending unit
        if self.pending_unit:
            remaining_space = self.need - len(self.buf)
            if remaining_space > 1:  # Need space for at least one token + potential EOS
                add_count = min(remaining_space - 1, len(self.pending_unit))
                if add_count > 0:
                    self.buf.extend(self.pending_unit[:add_count])
                    self.pending_unit = self.pending_unit[add_count:]

                    # If pending unit completed, add EOS
                    if not self.pending_unit and len(self.buf) < self.need:
                        self.buf.append(self.eos_id)

    def yield_blocks(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        while len(self.buf) >= self.need:
            x = torch.tensor(self.buf[: self.need], dtype=torch.long)
            del self.buf[: self.need]
            # No need to ensure EOS at start since units already start with EOS
            yield x[:-1], x[1:]

    def state_dict(self) -> Dict[str, List[int]]:
        return {"buf": self.buf, "pending_unit": self.pending_unit}

    def load_state_dict(self, s: Dict):
        self.buf = list(s.get("buf", [])) or []
        self.pending_unit = list(s.get("pending_unit", []))