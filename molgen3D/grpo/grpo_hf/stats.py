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
    """Tracks various statistics during model training and generation."""
    # Global statistics
    processed_prompts: int = 0
    successful_generations: int = 0
    failed_ground_truth: int = 0
    failed_conformer_generation: int = 0
    failed_matching_smiles: int = 0
    failed_rmsd: int = 0
    rmsd_values: list = field(default_factory=list)
    
    # RMSD statistics
    total_rmsd: float = 0.0
    rmsd_counts: int = 0
    
    # Timing statistics
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
        return sum(self.rmsd_values) / len(self.rmsd_values) if self.rmsd_values else 0.0
    
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
            "successful_generations": self.successful_generations,
            "failed_ground_truth": self.failed_ground_truth,
            "failed_conformer_generation": self.failed_conformer_generation,
            "failed_matching_smiles": self.failed_matching_smiles,
            "failed_rmsd": self.failed_rmsd,
            "success_rate": self.success_rate,
            "average_rmsd": self.average_rmsd,
            "failure_rates": failure_rates,
            "runtime_minutes": self.runtime
        }
        return stats

    def save(self, run_dir: Path) -> None:
        """Save statistics to a JSON file"""
        stats = self.log_global_stats()
        stats_file = run_dir / "statistics.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=4)