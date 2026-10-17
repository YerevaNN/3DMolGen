#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import sqlite3
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

from tqdm.auto import tqdm


SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}
SCAN_SPLITS = ("test", "val", "train")
INSERT_BATCH_SIZE = 200000


def log(msg: str) -> None:
    print(msg, flush=True)


def digest_sample(enriched_text: str) -> bytes:
    return hashlib.blake2b(enriched_text.encode("utf-8"), digest_size=16).digest()


def ensure_split_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise ValueError(f"Split directory not found: {split_dir}")


def list_split_files(root: Path, split: str) -> List[Path]:
    files = sorted((root / split).glob("*.csv"))
    if not files:
        raise ValueError(f"No CSV files found in {root / split}")
    return files


def ordered_source_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for split in SCAN_SPLITS:
        files.extend(list_split_files(root, split))
    return files


def drop_list_path(drop_lists_root: Path, split: str, dataset: str) -> Path:
    return drop_lists_root / split / f"{dataset}.csv"


def write_csv(rows: Iterable[Dict[str, int | str]], fieldnames: List[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def init_db(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    conn.execute(
        """
        CREATE TABLE files (
            file_id INTEGER PRIMARY KEY,
            split TEXT NOT NULL,
            dataset TEXT NOT NULL,
            source_path TEXT NOT NULL,
            drop_list_path TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE sample_hashes (
            file_id INTEGER NOT NULL,
            digest BLOB NOT NULL,
            PRIMARY KEY (file_id, digest)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE shared_digests (
            digest BLOB PRIMARY KEY
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE shared_samples (
            sample_text TEXT NOT NULL,
            file_id INTEGER NOT NULL,
            PRIMARY KEY (sample_text, file_id)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE sample_owners (
            sample_text TEXT PRIMARY KEY,
            keep_file_id INTEGER NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def open_existing_db(db_path: Path) -> sqlite3.Connection:
    if not db_path.exists():
        raise ValueError(f"Stats DB not found: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    conn.execute("PRAGMA cache_size=-200000")
    return conn


def scan_unique_samples(
    source_path: Path,
    on_unique_sample: Callable[[str], None],
) -> Dict[str, int]:
    seen_samples = set()
    total_rows = 0
    unique_rows = 0
    duplicate_rows = 0
    skipped_rows = 0

    with source_path.open("r", encoding="utf-8", buffering=1048576) as in_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {source_path}")
        if "enriched_text" not in reader.fieldnames:
            raise ValueError(f"'enriched_text' column not found in {source_path}")

        for row in reader:
            enriched_text = row.get("enriched_text", "")
            if not enriched_text:
                skipped_rows += 1
                continue
            total_rows += 1
            if enriched_text in seen_samples:
                duplicate_rows += 1
                continue

            seen_samples.add(enriched_text)
            unique_rows += 1
            on_unique_sample(enriched_text)

    return {
        "source_rows": total_rows,
        "unique_rows": unique_rows,
        "within_file_duplicate_rows": duplicate_rows,
        "skipped_rows": skipped_rows,
    }


def register_files(conn: sqlite3.Connection, source_root: Path, drop_lists_root: Path) -> List[Path]:
    files = ordered_source_files(source_root)
    for file_id, source_path in enumerate(files, start=1):
        split = source_path.parent.name
        dataset = source_path.stem
        conn.execute(
            """
            INSERT INTO files (file_id, split, dataset, source_path, drop_list_path)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                file_id,
                split,
                dataset,
                str(source_path),
                str(drop_list_path(drop_lists_root, split, dataset)),
            ),
        )
    conn.commit()
    return files


def build_digest_index(
    conn: sqlite3.Connection,
    files: List[Path],
) -> Dict[Tuple[str, str], Dict[str, int | str]]:
    summary_rows: Dict[Tuple[str, str], Dict[str, int | str]] = {}

    for file_id, source_path in enumerate(tqdm(files, desc="Indexing files", mininterval=5), start=1):
        buffer: List[Tuple[int, bytes]] = []

        def on_unique_sample(enriched_text: str) -> None:
            digest = digest_sample(enriched_text)
            buffer.append((file_id, sqlite3.Binary(digest)))
            if len(buffer) >= INSERT_BATCH_SIZE:
                conn.executemany("INSERT INTO sample_hashes (file_id, digest) VALUES (?, ?)", buffer)
                conn.commit()
                buffer.clear()

        stats = scan_unique_samples(source_path, on_unique_sample)
        if buffer:
            conn.executemany("INSERT INTO sample_hashes (file_id, digest) VALUES (?, ?)", buffer)
            conn.commit()

        split = source_path.parent.name
        dataset = source_path.stem
        summary_rows[(split, dataset)] = {
            "split": split,
            "dataset": dataset,
            "source_rows": stats["source_rows"],
            "unique_rows": stats["unique_rows"],
            "within_file_duplicate_rows": stats["within_file_duplicate_rows"],
            "cross_file_duplicate_rows_to_drop": 0,
            "kept_rows_if_applied": stats["unique_rows"],
            "skipped_rows": stats["skipped_rows"],
        }
        log(
            f"{split}/{dataset}: source_rows={stats['source_rows']:,}, "
            f"unique_rows={stats['unique_rows']:,}, "
            f"within_file_duplicate_rows={stats['within_file_duplicate_rows']:,}"
        )

    return summary_rows


def populate_shared_digests(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM shared_digests")
    conn.execute(
        """
        INSERT INTO shared_digests (digest)
        SELECT digest
        FROM sample_hashes
        GROUP BY digest
        HAVING COUNT(*) > 1
        """
    )
    conn.commit()
    shared_digest_count = conn.execute("SELECT COUNT(*) FROM shared_digests").fetchone()[0]
    return int(shared_digest_count)


def load_shared_digests(conn: sqlite3.Connection) -> set[bytes]:
    return {row[0] for row in conn.execute("SELECT digest FROM shared_digests")}


def collect_exact_shared_samples(
    conn: sqlite3.Connection,
    files: List[Path],
    shared_digests: set[bytes],
) -> None:
    for file_id, source_path in enumerate(tqdm(files, desc="Verifying shared samples", mininterval=5), start=1):
        buffer: List[Tuple[str, int]] = []

        def on_unique_sample(enriched_text: str) -> None:
            digest = digest_sample(enriched_text)
            if digest not in shared_digests:
                return
            buffer.append((enriched_text, file_id))
            if len(buffer) >= 50000:
                conn.executemany(
                    "INSERT OR IGNORE INTO shared_samples (sample_text, file_id) VALUES (?, ?)",
                    buffer,
                )
                conn.commit()
                buffer.clear()

        scan_unique_samples(source_path, on_unique_sample)
        if buffer:
            conn.executemany(
                "INSERT OR IGNORE INTO shared_samples (sample_text, file_id) VALUES (?, ?)",
                buffer,
            )
            conn.commit()

    conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_samples_text ON shared_samples (sample_text)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_shared_samples_file_id ON shared_samples (file_id)")
    conn.commit()


def build_sample_owners(conn: sqlite3.Connection) -> int:
    conn.execute("DELETE FROM sample_owners")
    conn.execute(
        """
        INSERT INTO sample_owners (sample_text, keep_file_id)
        SELECT repeated.sample_text,
               (
                   SELECT ss.file_id
                   FROM shared_samples ss
                   JOIN files f ON f.file_id = ss.file_id
                   WHERE ss.sample_text = repeated.sample_text
                   ORDER BY
                       CASE f.split
                           WHEN 'test' THEN 0
                           WHEN 'val' THEN 1
                           ELSE 2
                       END,
                       f.dataset,
                       ss.file_id
                   LIMIT 1
               ) AS keep_file_id
        FROM (
            SELECT sample_text
            FROM shared_samples
            GROUP BY sample_text
            HAVING COUNT(*) > 1
        ) repeated
        """
    )
    conn.commit()
    owner_count = conn.execute("SELECT COUNT(*) FROM sample_owners").fetchone()[0]
    return int(owner_count)


def write_drop_lists(conn: sqlite3.Connection, drop_lists_root: Path) -> Dict[Tuple[str, str], int]:
    drop_counts: Dict[Tuple[str, str], int] = {}

    files = list(
        conn.execute(
            """
            SELECT file_id, split, dataset, drop_list_path
            FROM files
            ORDER BY
                CASE split
                    WHEN 'test' THEN 0
                    WHEN 'val' THEN 1
                    ELSE 2
                END,
                dataset,
                file_id
            """
        )
    )

    for file_id, split, dataset, drop_path_text in files:
        drop_path = Path(drop_path_text)
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with drop_path.open("w", newline="", encoding="utf-8", buffering=1048576) as f:
            writer = csv.DictWriter(f, fieldnames=["enriched_text"])
            writer.writeheader()
            for (sample_text,) in conn.execute(
                """
                SELECT ss.sample_text
                FROM shared_samples ss
                JOIN sample_owners o ON o.sample_text = ss.sample_text
                WHERE ss.file_id = ? AND o.keep_file_id != ?
                ORDER BY ss.sample_text
                """,
                (file_id, file_id),
            ):
                writer.writerow({"enriched_text": sample_text})
                count += 1

        drop_counts[(split, dataset)] = count
        log(f"{split}/{dataset}: cross_file_duplicate_rows_to_drop={count:,}")

    return drop_counts


def fetch_exact_intersections(conn: sqlite3.Connection) -> List[Dict[str, int | str]]:
    query = """
        SELECT
            f1.split AS split_a,
            f1.dataset AS dataset_a,
            f2.split AS split_b,
            f2.dataset AS dataset_b,
            COUNT(*) AS intersection_rows
        FROM shared_samples s1
        JOIN shared_samples s2
            ON s1.sample_text = s2.sample_text
           AND s1.file_id < s2.file_id
        JOIN sample_owners o
            ON o.sample_text = s1.sample_text
        JOIN files f1 ON f1.file_id = s1.file_id
        JOIN files f2 ON f2.file_id = s2.file_id
        GROUP BY s1.file_id, s2.file_id
        HAVING COUNT(*) > 0
        ORDER BY intersection_rows DESC, split_a, dataset_a, split_b, dataset_b
    """
    rows: List[Dict[str, int | str]] = []
    for split_a, dataset_a, split_b, dataset_b, intersection_rows in conn.execute(query):
        rows.append(
            {
                "split_a": split_a,
                "dataset_a": dataset_a,
                "split_b": split_b,
                "dataset_b": dataset_b,
                "intersection_rows": intersection_rows,
            }
        )
    return rows


def run_stats_phase(
    source_root: Path,
    sqlite_db: Path,
    drop_lists_root: Path,
    stats_summary_csv: Path,
    intersections_csv: Path,
) -> None:
    ensure_split_dirs(source_root)
    drop_lists_root.mkdir(parents=True, exist_ok=True)

    conn = init_db(sqlite_db)
    files = register_files(conn, source_root, drop_lists_root)
    summary_rows = build_digest_index(conn, files)

    shared_digest_count = populate_shared_digests(conn)
    log(f"Shared candidate digests across files: {shared_digest_count:,}")

    shared_digests = load_shared_digests(conn)
    collect_exact_shared_samples(conn, files, shared_digests)
    repeated_sample_count = build_sample_owners(conn)
    log(f"Exact repeated samples across files: {repeated_sample_count:,}")

    drop_counts = write_drop_lists(conn, drop_lists_root)
    intersections = fetch_exact_intersections(conn)

    for key, count in drop_counts.items():
        summary_rows[key]["cross_file_duplicate_rows_to_drop"] = count
        summary_rows[key]["kept_rows_if_applied"] = int(summary_rows[key]["unique_rows"]) - count

    ordered_summary_rows = [
        summary_rows[(source_path.parent.name, source_path.stem)]
        for source_path in files
    ]
    write_csv(
        rows=ordered_summary_rows,
        fieldnames=[
            "split",
            "dataset",
            "source_rows",
            "unique_rows",
            "within_file_duplicate_rows",
            "cross_file_duplicate_rows_to_drop",
            "kept_rows_if_applied",
            "skipped_rows",
        ],
        path=stats_summary_csv,
    )
    write_csv(
        rows=intersections,
        fieldnames=["split_a", "dataset_a", "split_b", "dataset_b", "intersection_rows"],
        path=intersections_csv,
    )

    log(f"Saved exact duplicate stats to {stats_summary_csv}")
    log(f"Saved exact pairwise intersections to {intersections_csv}")
    log(f"Saved stats DB to {sqlite_db}")
    log(f"Saved per-file drop lists to {drop_lists_root}")


def load_drop_samples(drop_path: Path) -> set[str]:
    if not drop_path.exists():
        return set()

    drop_samples: set[str] = set()
    with drop_path.open("r", encoding="utf-8", buffering=1048576) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "enriched_text" not in reader.fieldnames:
            raise ValueError(f"Invalid drop-list CSV: {drop_path}")
        for row in reader:
            enriched_text = row.get("enriched_text", "")
            if enriched_text:
                drop_samples.add(enriched_text)
    return drop_samples


def apply_file(
    source_path: Path,
    output_path: Path,
    drop_samples: set[str],
) -> Dict[str, int | str]:
    seen_samples = set()
    source_rows = 0
    kept_rows = 0
    within_file_duplicate_rows = 0
    cross_file_duplicate_rows_removed = 0
    skipped_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with source_path.open("r", encoding="utf-8", buffering=1048576) as in_f, output_path.open(
        "w", newline="", encoding="utf-8", buffering=1048576
    ) as out_f:
        reader = csv.DictReader(in_f)
        if reader.fieldnames is None:
            raise ValueError(f"Missing header in {source_path}")
        if "enriched_text" not in reader.fieldnames:
            raise ValueError(f"'enriched_text' column not found in {source_path}")
        writer = csv.DictWriter(out_f, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            enriched_text = row.get("enriched_text", "")
            if not enriched_text:
                skipped_rows += 1
                continue
            source_rows += 1
            if enriched_text in seen_samples:
                within_file_duplicate_rows += 1
                continue

            seen_samples.add(enriched_text)
            if enriched_text in drop_samples:
                cross_file_duplicate_rows_removed += 1
                continue

            writer.writerow(row)
            kept_rows += 1

    return {
        "split": source_path.parent.name,
        "dataset": source_path.stem,
        "source_rows": source_rows,
        "kept_rows": kept_rows,
        "within_file_duplicate_rows_removed": within_file_duplicate_rows,
        "cross_file_duplicate_rows_removed": cross_file_duplicate_rows_removed,
        "skipped_rows": skipped_rows,
    }


def run_apply_phase(
    source_root: Path,
    output_root: Path,
    drop_lists_root: Path,
    apply_summary_csv: Path,
) -> None:
    ensure_split_dirs(source_root)
    output_root.mkdir(parents=True, exist_ok=True)

    apply_rows: List[Dict[str, int | str]] = []
    files = ordered_source_files(source_root)
    for source_path in tqdm(files, desc="Writing deduplicated files", mininterval=5):
        split = source_path.parent.name
        dataset = source_path.stem
        drops = load_drop_samples(drop_list_path(drop_lists_root, split, dataset))
        output_path = output_root / split / source_path.name
        stats = apply_file(source_path, output_path, drops)
        apply_rows.append(stats)
        log(
            f"{split}/{dataset}: kept_rows={stats['kept_rows']:,}, "
            f"within_file_duplicate_rows_removed={stats['within_file_duplicate_rows_removed']:,}, "
            f"cross_file_duplicate_rows_removed={stats['cross_file_duplicate_rows_removed']:,}"
        )

    write_csv(
        rows=apply_rows,
        fieldnames=[
            "split",
            "dataset",
            "source_rows",
            "kept_rows",
            "within_file_duplicate_rows_removed",
            "cross_file_duplicate_rows_removed",
            "skipped_rows",
        ],
        path=apply_summary_csv,
    )
    log(f"Saved apply summary to {apply_summary_csv}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Collect exact enriched_text duplicate statistics across a flat train/val/test root, "
            "save per-file drop lists for cross-file duplicates, and optionally write a globally "
            "deduplicated copy that keeps one deterministic copy of each repeated sample."
        )
    )
    parser.add_argument(
        "--phase",
        choices=("stats", "apply", "all"),
        default="all",
        help="Run stats collection only, apply only, or both in sequence.",
    )
    parser.add_argument("--source-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data_new"))
    parser.add_argument("--output-root", type=Path, default=Path("/nfs/h100/raid/chem/3D_big_data_new_dedup"))
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_exact_duplicate_stats.csv"),
        help="Stats-phase per-file summary CSV.",
    )
    parser.add_argument(
        "--intersections-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_exact_sample_intersections.csv"),
    )
    parser.add_argument(
        "--apply-summary-csv",
        type=Path,
        default=Path("/auto/home/filya/fsq/3d_big_data_new_dedup_apply_summary.csv"),
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new_dedup_state/sample_duplicate_stats.sqlite"),
    )
    parser.add_argument(
        "--drop-lists-root",
        type=Path,
        default=Path("/nfs/h100/raid/chem/3D_big_data_new_dedup_state/drop_lists"),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.phase in {"stats", "all"}:
        run_stats_phase(
            source_root=args.source_root,
            sqlite_db=args.sqlite_db,
            drop_lists_root=args.drop_lists_root,
            stats_summary_csv=args.summary_csv,
            intersections_csv=args.intersections_csv,
        )

    if args.phase in {"apply", "all"}:
        if args.phase == "apply":
            open_existing_db(args.sqlite_db).close()
        run_apply_phase(
            source_root=args.source_root,
            output_root=args.output_root,
            drop_lists_root=args.drop_lists_root,
            apply_summary_csv=args.apply_summary_csv,
        )

    log("Done.")


if __name__ == "__main__":
    main()
