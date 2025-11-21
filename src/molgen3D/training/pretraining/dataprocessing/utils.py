import io
import glob
from pathlib import Path
from typing import List, Sequence, Union
import numpy as np
from loguru import logger


def expand_paths(paths: Union[str, Sequence[str]]) -> List[str]:
    ps = [paths] if isinstance(paths, (str, Path)) else list(paths)
    out: List[str] = []
    for p in ps:
        p = str(p)
        path_obj = Path(p)

        # If it's already a glob pattern, expand it
        if any(ch in p for ch in "*?["):
            out.extend(glob.glob(p))
        # If it's a directory, find all .jsonl files within it
        elif path_obj.is_dir():
            out.extend(glob.glob(str(path_obj / "**" / "*.jsonl"), recursive=True))
        # If it's a file, use it directly
        else:
            out.append(p)

    out = sorted(set(out))
    if not out:
        raise FileNotFoundError(f"No JSONL files matched: {paths}")
    return out


def idx_path(p: Union[str, Path]) -> Path:
    return Path(str(p) + ".idx")


def build_line_index(p: Union[str, Path]) -> np.ndarray:
    p = Path(p)
    idxp = idx_path(p)
    if idxp.exists() and idxp.stat().st_mtime >= p.stat().st_mtime:
        return np.fromfile(idxp, dtype=np.uint64)
    logger.info(f"[index] building .idx for {p}")
    offs, pos = [], 0
    with open(p, "rb") as f:
        while True:
            offs.append(pos)
            line = f.readline()
            if not line:
                break
            pos += len(line)
    arr = np.array(offs[:-1], dtype=np.uint64)  # drop EOF sentinel
    arr.tofile(idxp)
    return arr


def read_line_at(fp: io.BufferedReader, byte_off: int) -> bytes:
    fp.seek(byte_off)
    return fp.readline()