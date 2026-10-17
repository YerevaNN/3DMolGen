import csv
import gc
import os
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from utils import encode_cartesian_v2


CoordItem = Union[torch.Tensor, np.ndarray]
PygSlices = dict[str, torch.Tensor]
PygCoordStore = tuple[Data, PygSlices]
CoordStore = Union[Sequence[CoordItem], torch.Tensor, PygCoordStore]

chembl_dir: Path = Path(os.environ.get("CHEMBL_SOURCE_DIR", "/nfs/h100/raid/chem/chembl3d/processed"))
base_output_dir: Path = Path(os.environ.get("CHEMBL_OUTPUT_DIR", "/nfs/dgx/raid/chem/TokenizerData"))


def load_coords(coords_path: Path) -> CoordStore:
    if not coords_path.exists():
        raise FileNotFoundError(f"Missing coordinates file at {coords_path}")
    load_kwargs = {
        "map_location": torch.device("cpu"),
        "weights_only": False,
    }
    try:
        data = torch.load(coords_path, mmap=True, **load_kwargs)
    except TypeError:
        data = torch.load(coords_path, **load_kwargs)
    return data


class CoordAccessor:
    store: CoordStore
    data: Data | None
    slices: PygSlices | None
    pos_slices: torch.Tensor | None
    tensor_store: torch.Tensor | None
    seq_store: Sequence[CoordItem] | None

    def __init__(self, store: CoordStore):
        self.store = store
        self.data = None
        self.slices = None
        self.pos_slices = None
        self.tensor_store = None
        self.seq_store = None
        if isinstance(store, tuple):
            data_obj, slices = store
            pos_slices = slices.get("pos") if isinstance(slices, dict) else None
            if not isinstance(data_obj, Data):
                raise TypeError(f"PyG coordinates payload must start with Data, received {type(data_obj)}")
            if not isinstance(pos_slices, torch.Tensor):
                raise TypeError("PyG pos slices must be tensor")
            if not hasattr(data_obj, "pos"):
                raise AttributeError("PyG Data missing pos attribute")
            self.data = data_obj
            self.slices = slices
            self.pos_slices = pos_slices
        elif isinstance(store, torch.Tensor):
            if store.ndim < 2:
                raise ValueError(f"Tensor coordinates must be at least 2D, got {store.shape}")
            self.tensor_store = store
        elif isinstance(store, Sequence):
            self.seq_store = store
        else:
            raise TypeError(f"Unsupported coordinates container {type(store)}")

    def length(self) -> int:
        if self.pos_slices is not None:
            return int(self.pos_slices.shape[0] - 1)
        if self.tensor_store is not None:
            return int(self.tensor_store.shape[0])
        if self.seq_store is not None:
            return len(self.seq_store)
        raise RuntimeError("Uninitialized coordinates store")

    def coords_at(self, idx: int) -> np.ndarray:
        if self.pos_slices is not None and self.data is not None:
            if idx + 1 >= self.pos_slices.shape[0]:
                raise IndexError(f"Index {idx} out of bounds for PyG coordinates")
            start = int(self.pos_slices[idx].item())
            end = int(self.pos_slices[idx + 1].item())
            positions = self.data.pos[start:end]
            arr = positions.detach().cpu().numpy()
        elif self.tensor_store is not None:
            item = self.tensor_store[idx]
            arr = item.detach().cpu().numpy()
        elif self.seq_store is not None:
            item = self.seq_store[idx]
            arr = item.detach().cpu().numpy() if isinstance(item, torch.Tensor) else np.asarray(item)
        else:
            raise RuntimeError("Uninitialized coordinates store")
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Coordinates at index {idx} have invalid shape {arr.shape}")
        return arr

    def mol_at(self, idx: int):
        if self.data is None or self.slices is None:
            return None
        mol_slices = self.slices.get("mol")
        if not isinstance(mol_slices, torch.Tensor) or not hasattr(self.data, "mol"):
            return None
        if idx + 1 >= mol_slices.shape[0]:
            raise IndexError(f"Index {idx} out of bounds for PyG mol slices")
        start = int(mol_slices[idx].item())
        end = int(mol_slices[idx + 1].item())
        mols = self.data.mol[start:end]
        if len(mols) == 0:
            return None
        if len(mols) != 1:
            raise ValueError(f"Expected one mol at index {idx}, found {len(mols)}")
        return mols[0]

    def has_mols(self) -> bool:
        return self.data is not None and self.slices is not None and isinstance(self.slices.get("mol"), torch.Tensor) and hasattr(self.data, "mol")


def write_split(
    split: str,
    indices: Sequence[int],
    accessor: CoordAccessor,
    name_prefix: str,
) -> None:
    split_dir = base_output_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    output_csv = split_dir / "chembl3d.csv"
    with output_csv.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["name", "enriched_text"])
        processed = 0
        skipped = 0
        for idx in tqdm(indices, desc=f"chembl3d {split}"):
            try:
                mol = accessor.mol_at(idx)
                if mol is None:
                    raise RuntimeError("Source RDKit mol missing; refusing to rebuild chembl3d from smiles+coords")
                enriched_text = encode_cartesian_v2(mol, precision=4)
                writer.writerow([f"{name_prefix}_{idx}", enriched_text])
                processed += 1
            except Exception as exc:
                print(f"Skipped index {idx}: {type(exc).__name__}: {exc}")
                skipped += 1
                continue
            if accessor.seq_store is not None:
                accessor.seq_store[idx] = None
            if processed % 50000 == 0 and processed > 0:
                gc.collect()
        csvfile.flush()
        print(f"{split}: Processed: {processed:,}, Skipped: {skipped:,}")
        print(f"Saved to {output_csv}")


def main() -> None:
    split_specs = [
        ("train", "train_h.pt", "chembl3d"),
        ("val", "val_h.pt", "chembl3d"),
        ("test", "test_small_h.pt", "chembl3d_test"),
    ]
    requested_splits_raw = os.environ.get("CHEMBL_SPLITS", "")
    requested_splits = {part.strip() for part in requested_splits_raw.split(",") if part.strip()}
    for split, coords_name, name_prefix in split_specs:
        if requested_splits and split not in requested_splits:
            continue
        print(f"\nProcessing {split} split...")
        accessor = CoordAccessor(load_coords(chembl_dir / coords_name))
        n_mols = accessor.length()
        if not accessor.has_mols():
            raise RuntimeError(f"{split} payload at {chembl_dir / coords_name} does not include source RDKit mols")
        print(f"{split.capitalize()} molecules: {n_mols}")
        write_split(split, range(n_mols), accessor, name_prefix=name_prefix)
        del accessor
        gc.collect()
    print("\nDone!")


if __name__ == "__main__":
    main()
