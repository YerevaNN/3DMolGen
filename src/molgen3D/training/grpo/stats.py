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
    precision_rewards: list = field(default_factory=list)
    coverage_rewards: list = field(default_factory=list)
    match_rewards: list = field(default_factory=list)
    validity_rewards: list = field(default_factory=list)
    diversity_rewards: list = field(default_factory=list)
    final_rewards: list = field(default_factory=list)
    coverage_claims: int = 0
    coverage_opportunities: int = 0
    posebusters_successes: int = 0
    posebusters_failures: int = 0
    
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
        return float(np.nanmean(self.rmsd_values)) if self.rmsd_values else 0.0
    
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
        posebusters_checks = self.posebusters_successes + self.posebusters_failures

        def safe_mean(values):
            return float(np.nanmean(values)) if values else 0.0

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
            "reward_precision_mean": safe_mean(self.precision_rewards),
            "reward_coverage_mean": safe_mean(self.coverage_rewards),
            "reward_match_mean": safe_mean(self.match_rewards),
            "reward_validity_rate": safe_mean(self.validity_rewards),
            "reward_diversity_mean": safe_mean(self.diversity_rewards),
            "reward_final_mean": safe_mean(self.final_rewards),
            "reward_precision_count": len(self.precision_rewards),
            "reward_coverage_count": len(self.coverage_rewards),
            "reward_match_count": len(self.match_rewards),
            "reward_validity_count": len(self.validity_rewards),
            "reward_diversity_count": len(self.diversity_rewards),
            "reward_final_count": len(self.final_rewards),
            "reward_coverage_claims": self.coverage_claims,
            "reward_coverage_opportunities": self.coverage_opportunities,
            "reward_coverage_rate": (
                self.coverage_claims / self.coverage_opportunities if self.coverage_opportunities > 0 else 0.0
            ),
            "posebusters_successes": self.posebusters_successes,
            "posebusters_failures": self.posebusters_failures,
            "posebusters_checks": posebusters_checks,
            "posebusters_pass_rate": (
                self.posebusters_successes / posebusters_checks if posebusters_checks > 0 else 0.0
            ),
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
                "runtime_minutes": 0.0,
                "reward_precision_sum": 0.0,
                "reward_precision_count": 0,
                "reward_coverage_sum": 0.0,
                "reward_coverage_count": 0,
                "reward_match_sum": 0.0,
                "reward_match_count": 0,
                "reward_validity_sum": 0.0,
                "reward_validity_count": 0,
                "reward_diversity_sum": 0.0,
                "reward_diversity_count": 0,
                "reward_final_sum": 0.0,
                "reward_final_count": 0,
                "reward_coverage_claims": 0,
                "reward_coverage_opportunities": 0,
                "posebusters_successes": 0,
                "posebusters_failures": 0,
                "posebusters_checks": 0,
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
                    precision_count = stats.get("reward_precision_count", 0)
                    aggregate["reward_precision_sum"] += stats.get("reward_precision_mean", 0.0) * precision_count
                    aggregate["reward_precision_count"] += precision_count
                    coverage_count = stats.get("reward_coverage_count", 0)
                    aggregate["reward_coverage_sum"] += stats.get("reward_coverage_mean", 0.0) * coverage_count
                    aggregate["reward_coverage_count"] += coverage_count
                    match_count = stats.get("reward_match_count", 0)
                    aggregate["reward_match_sum"] += stats.get("reward_match_mean", 0.0) * match_count
                    aggregate["reward_match_count"] += match_count
                    validity_count = stats.get("reward_validity_count", 0)
                    aggregate["reward_validity_sum"] += stats.get("reward_validity_rate", 0.0) * validity_count
                    aggregate["reward_validity_count"] += validity_count
                    diversity_count = stats.get("reward_diversity_count", 0)
                    aggregate["reward_diversity_sum"] += stats.get("reward_diversity_mean", 0.0) * diversity_count
                    aggregate["reward_diversity_count"] += diversity_count
                    final_count = stats.get("reward_final_count", 0)
                    aggregate["reward_final_sum"] += stats.get("reward_final_mean", 0.0) * final_count
                    aggregate["reward_final_count"] += final_count
                    aggregate["reward_coverage_claims"] += stats.get("reward_coverage_claims", 0)
                    aggregate["reward_coverage_opportunities"] += stats.get("reward_coverage_opportunities", 0)
                    aggregate["posebusters_successes"] += stats.get("posebusters_successes", 0)
                    aggregate["posebusters_failures"] += stats.get("posebusters_failures", 0)
                    aggregate["posebusters_checks"] += stats.get("posebusters_checks", 0)
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
            def finalize_mean(sum_key: str, count_key: str, out_key: str):
                count = aggregate.get(count_key, 0)
                aggregate[out_key] = (aggregate.get(sum_key, 0.0) / count) if count else 0.0

            finalize_mean("reward_precision_sum", "reward_precision_count", "reward_precision_mean")
            finalize_mean("reward_coverage_sum", "reward_coverage_count", "reward_coverage_mean")
            finalize_mean("reward_match_sum", "reward_match_count", "reward_match_mean")
            finalize_mean("reward_validity_sum", "reward_validity_count", "reward_validity_rate")
            finalize_mean("reward_diversity_sum", "reward_diversity_count", "reward_diversity_mean")
            finalize_mean("reward_final_sum", "reward_final_count", "reward_final_mean")
            aggregate["reward_coverage_rate"] = (
                aggregate["reward_coverage_claims"] / aggregate["reward_coverage_opportunities"]
                if aggregate["reward_coverage_opportunities"] > 0 else 0.0
            )
            aggregate["posebusters_pass_rate"] = (
                aggregate["posebusters_successes"] / aggregate["posebusters_checks"]
                if aggregate["posebusters_checks"] > 0 else 0.0
            )
            stats_file = stats_dir / "statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(aggregate, f, indent=4)
            fcntl.flock(lock, fcntl.LOCK_UN)
        return aggregate
