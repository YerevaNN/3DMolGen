from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path
import json
from datetime import datetime
from collections import deque
import numpy as np
from loguru import logger


@dataclass
class RunStatistics:
    output_dir: str
    processed_prompts: int = 0
    successful_generations: int = 0
    distinct_prompts: int = 0
    failed_ground_truth: int = 0
    failed_conformer_generation: int = 0
    failed_matching_smiles: int = 0
    failed_rmsd: int = 0
    rmsd_values: list = field(default_factory=list)
    total_rmsd: float = 0.0
    rmsd_counts: int = 0
    start_time: datetime = field(default_factory=datetime.now)
    
    def add_rmsd(self, rmsd: float) -> None:
        """Track RMSD values for averaging"""
        self.total_rmsd += rmsd
        self.rmsd_counts += 1
        self.rmsd_values.append(rmsd)
    
    def add_success(self, success: bool) -> None:
        """Track success for rolling statistics"""
        if success:
            self.successful_generations += 1
    
    @property
    def average_rmsd(self) -> float:
        """Calculate average RMSD across successful generations"""
        return np.nanmean(self.rmsd_values) if self.rmsd_values else 0.0
    
    @property
    def success_rate(self) -> float:
        """Calculate successful generation rate"""
        return self.successful_generations / self.processed_prompts if self.processed_prompts > 0 else 0.0
    
    @property
    def failure_rates(self) -> Dict[str, float]:
        """Calculate failure rates for different types of failures"""
        if self.processed_prompts == 0:
            return {
                "ground_truth": 0.0,
                "conformer_generation": 0.0,
                "matching_smiles": 0.0,
                "rmsd": 0.0
            }
        
        return {
            "ground_truth": self.failed_ground_truth / self.processed_prompts,
            "conformer_generation": self.failed_conformer_generation / self.processed_prompts,
            "matching_smiles": self.failed_matching_smiles / self.processed_prompts,
            "rmsd": self.failed_rmsd / self.processed_prompts
        }
    
    @property
    def runtime(self) -> float:
        """Calculate total runtime in minutes"""
        return (datetime.now() - self.start_time).total_seconds() / 60.0

    def log_global_stats(self):
        """Log global statistics for the entire run."""
        failure_rates = self.failure_rates
        stats = {
            "processed_prompts": self.processed_prompts,
            "distinct_prompts": self.distinct_prompts,
            "successful_generations": self.successful_generations,
            "failed_ground_truth": self.failed_ground_truth,
            "failed_conformer_generation": self.failed_conformer_generation,
            "failed_matching_smiles": self.failed_matching_smiles,
            "failed_rmsd": self.failed_rmsd,
            "success_rate": self.success_rate,
            "average_rmsd": self.average_rmsd,
            "failure_rates": failure_rates,
            "runtime_minutes": self.runtime,
        }
        return stats

    def update_stats(self) -> Dict:
        import os
        import time
        import glob
        import fcntl
        pid = os.getpid()
        stats_dir = Path(self.output_dir)
        stats_dir.mkdir(parents=True, exist_ok=True)
        own_stats = self.log_global_stats()
        own_stats_file = stats_dir / f"statistics_{pid}.json"
        with open(own_stats_file, 'w') as f:
            json.dump(own_stats, f, indent=4)
        lock_file = stats_dir / "statistics.lock"
        aggregate = {}
        with open(lock_file, 'w') as lock:
            acquired = False
            while not acquired:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    time.sleep(0.1)
            stats_files = glob.glob(str(stats_dir / "statistics_*.json"))
            aggregate = {
                "processed_prompts": 0,
                "distinct_prompts": 0,
                "successful_generations": 0,
                "failed_ground_truth": 0,
                "failed_conformer_generation": 0,
                "failed_matching_smiles": 0,
                "failed_rmsd": 0,
                "rmsd_values": [],
                "runtime_minutes": 0.0
            }
            for file in stats_files:
                with open(file, 'r') as f:
                    try:
                        stats = json.load(f)
                    except json.JSONDecodeError:
                        continue
                    aggregate["processed_prompts"] += stats.get("processed_prompts", 0)
                    aggregate["distinct_prompts"] += stats.get("distinct_prompts", 0)
                    aggregate["successful_generations"] += stats.get("successful_generations", 0)
                    aggregate["failed_ground_truth"] += stats.get("failed_ground_truth", 0)
                    aggregate["failed_conformer_generation"] += stats.get("failed_conformer_generation", 0)
                    aggregate["failed_matching_smiles"] += stats.get("failed_matching_smiles", 0)
                    aggregate["failed_rmsd"] += stats.get("failed_rmsd", 0)
                    aggregate["runtime_minutes"] += stats.get("runtime_minutes", 0.0)
                    aggregate["rmsd_values"].extend(stats.get("rmsd_values", []))
            aggregate["success_rate"] = (
                aggregate["successful_generations"] / aggregate["processed_prompts"]
                if aggregate["processed_prompts"] > 0 else 0.0
            )
            aggregate["average_rmsd"] = (
                float(np.nanmean(aggregate["rmsd_values"]))
                if aggregate["rmsd_values"] else 0.0
            )
            aggregate["failure_rates"] = {
                "ground_truth": aggregate["failed_ground_truth"] / aggregate["processed_prompts"] if aggregate["processed_prompts"] > 0 else 0.0,
                "conformer_generation": aggregate["failed_conformer_generation"] / aggregate["processed_prompts"] if aggregate["processed_prompts"] > 0 else 0.0,
                "matching_smiles": aggregate["failed_matching_smiles"] / aggregate["processed_prompts"] if aggregate["processed_prompts"] > 0 else 0.0,
                "rmsd": aggregate["failed_rmsd"] / aggregate["processed_prompts"] if aggregate["processed_prompts"] > 0 else 0.0
            }
            stats_file = stats_dir / "statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(aggregate, f, indent=4)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return aggregate