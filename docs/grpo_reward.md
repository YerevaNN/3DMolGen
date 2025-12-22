# GRPO Reward V3 (GEOM-Drugs Aligned)

This document unifies the original reward specification with the implementation
guide for the Group Relative Policy Optimization (GRPO) V3 reward used in
`src/molgen3D/training/grpo/grpo_reward_v3.py`. It explains the mathematical
design, how it maps to code, default hyperparameters, and operational guidance.

## 1. Overview

- **Goal**: Provide a dense, group-aware reward that aligns with all four
  GEOM-Drugs metrics (AMR-P, AMR-R, COV-P, COV-R) under small rollout counts.
- **Key ideas**
  - Hard validity gate now requires (i) SMILES graph equality with the prompt
    and (ii) optional PoseBusters success before any reward terms are computed.
  - Dense quality term rewards every valid rollout for approaching any reference.
  - Smooth coverage term grants marginal value for covering new references even
    before reaching the benchmark threshold.
  - Hard matching term recreates GEOM-Drugs COV-R behavior via bipartite
    matching under a 0.75 Å threshold.
- **Why “inclusive”**: Duplicated good samples still earn quality reward, but
  group-aware terms gradually discount them in favor of diversity.

## 2. Setting and Notation

For each canonical SMILES (one GRPO group):

- Reference conformers: $G=\{g_j\}_{j=1}^M$ (up to 30 cached GEOM conformers).
- Rollouts: $Y=\{y_i\}_{i=1}^K$ generated conformers.
- Distances: $D_{ij}=d(y_i,g_j)$ = heavy-atom RMSD (Å) after Kabsch alignment.
- Hard threshold: $\delta = 0.75$ Å (GEOM-Drugs benchmark).
- Validity gate: $v_i\in\{0,1\}$ via RDKit sanity + optional PoseBusters.
- Nearest reference per rollout: $d_i = \min_j D_{ij}$.

## 3. Reward Terms

### Term 0 – Hard validity gate
$$
r_i =
\begin{cases}
r_{\text{floor}}, & v_i = 0\\
\tilde r_i, & v_i = 1
\end{cases}
\qquad r_{\text{floor}}=-1.0
$$
Rollouts are only considered valid if they **first** pass a graph-equality
check: the generated SMILES (after stripping) must encode the same molecular
graph as the prompt’s canonical SMILES (via `same_molecular_graph`). Any
mismatch (or parsing failure) immediately receives $r_{\text{floor}}$. If
`enable_posebusters` is true, the RDKit molecule must also pass PoseBusters;
failures are likewise assigned the floor reward.

### Term 1 – Dense quality (AMR-P / COV-P aligned)
$$
r_i^{\text{qual}} = \exp\!\left(-\frac{d_i}{\sigma}\right), \qquad \sigma = 0.25 \text{ Å}
$$
Reward decays smoothly with distance to the nearest reference; dense signal for
all valid conformers.

### Term 2 – Smooth marginal coverage (COV-R precursor)
$$
\begin{aligned}
k(d) &= \exp\!\left(-\left(\frac{d}{\rho}\right)^2\right), \quad \rho = 0.75 \text{ Å}\\
u_j &= 1 - \prod_{i=1}^K \big(1 - k(D_{ij})\big)\\
\Delta_{ij} &= k(D_{ij})\prod_{\ell\ne i} \big(1 - k(D_{\ell j})\big)\\
r_i^{\text{smcov}} &= \frac{1}{M}\sum_{j=1}^M \Delta_{ij}
\end{aligned}
$$
Interpretation: $\Delta_{ij}$ is the marginal contribution of rollout $i$ to
covering reference $j$. Early explorers get the highest reward; duplicates are
naturally discounted.

### Term 3 – Hard unique-coverage bonus (COV-R aligned)
$$
\begin{aligned}
E &= \{(i,j) : v_i = 1 \wedge D_{ij} < \delta\}\\
M^\star &= \arg\max_{M\subseteq E}\ |M| \quad \text{(break ties by } \sum D_{ij})\\
c_i &= \mathbf{1}\big[\exists j:(i,j)\in M^\star\big]
\end{aligned}
$$
Optional shaping (enabled in code):
$$
r_i^{\text{match}} =
\begin{cases}
1 - D_{ij}/\delta, & (i,j)\in M^\star\\
0, & \text{otherwise}
\end{cases}
$$
This enforces one-to-one coverage under $\delta$, mirroring GEOM-Drugs COV-R.

### Final combination (valid samples)
$$
\tilde r_i = \lambda_{\text{qual}} r_i^{\text{qual}}
            + \lambda_{\text{smcov}} r_i^{\text{smcov}}
            + \lambda_{\text{match}} r_i^{\text{match}}
$$
Default weights: $\lambda_{\text{qual}} = \lambda_{\text{smcov}} =
\lambda_{\text{match}} = 1.0$.

## 4. Implementation Flow

Located in `src/molgen3D/training/grpo/grpo_reward_v3.py`.

1. **Group samples** by canonical SMILES (GRPO prompts carry SMILES tags).
2. **Load reference conformers** for the group (cached pickles, max 30).
3. **Parse rollouts** into RDKit molecules, ensure the parsed SMILES matches
   the prompt’s graph, and optionally run PoseBusters. Invalid rollouts never
   reach the RMSD or reward stages.
4. **Compute RMSD matrix** $D$ (K × M) with Kabsch alignment (heavy atoms).
5. **Compute term rewards** (`compute_quality_reward`, `compute_smooth_coverage_reward`,
   `compute_matching_reward`).
6. **Combine terms** with weights + validity gate.
7. **Return per-sample rewards** in original order and log batch statistics.

### Validity gate (`compute_validity`)
- Ensures SMILES/graph equality and pose sanity before any RMSD work.
- PoseBusters is only run if `enable_posebusters` is true. Otherwise, the gate
  is determined solely by SMILES parsing and graph equality.
- Invalid rollouts are masked from all later computation and receive
  `r_floor`.

### RMSD matrix (`compute_distance_matrix`)
- Skips invalid rollouts (sets $D_{ij}=\infty$).
- Uses heavy-atom ordering consistent with benchmark alignment.

### Dense quality (`compute_quality_reward`)
```python
d_i = np.min(D, axis=1)
r_qual = np.where(validity == 1, np.exp(-d_i / sigma), 0.0)
```

### Smooth coverage (`compute_smooth_coverage_reward`)
```python
K_vals = np.exp(-(D / rho) ** 2)               # k(D_ij)
log_one_minus = np.log1p(-K_vals)              # log(1 - k)
log_prod_complement = np.sum(log_one_minus, axis=0) - log_one_minus
Delta = K_vals * np.exp(log_prod_complement)   # marginal Δ_ij
r_smcov = np.where(validity == 1, np.mean(Delta, axis=1), 0.0)
```
Log-space math keeps products stable for large K.

### Hard matching (`compute_matching_reward`)
```python
cost = np.where((D <= delta) & (validity[:, None] == 1), D, np.inf)
rows, cols = linear_sum_assignment(cost)
valid = cost[rows, cols] <= delta
r_match = np.zeros(K)
r_match[rows[valid]] = 1.0 - cost[rows[valid], cols[valid]] / delta  # shaped
```
The Hungarian algorithm first maximizes matches (finite entries) and then
prefers tighter RMSD pairs via the distance-based tie-break.

### Reward combination (`combine_rewards`)
```python
r_final = (
    lambda_qual * r_qual +
    lambda_smcov * r_smcov +
    lambda_match * r_match
)
r_final = np.where(validity == 1, r_final, r_floor)
```

## 5. Hyperparameters

| Parameter | Default | Description | Effect |
|-----------|---------|-------------|--------|
| `delta` | 0.75 Å | Matching threshold | ↑ favors recall, ↓ enforces strict matches |
| `sigma` | 0.25 Å | Quality decay scale | Smaller = sharper push toward refs |
| `rho` | 0.75 Å | Smooth coverage kernel | Smaller = rewards only very close hits |
| `lambda_qual` | 1.0 | Weight on quality term | Bias toward AMR-P |
| `lambda_smcov` | 1.0 | Weight on smooth coverage | Bias toward pre-threshold diversity |
| `lambda_match` | 1.0 | Weight on hard matching | Bias toward COV-R |
| `r_floor` | -1.0 | Invalid penalty | Lower discourages invalid generations |
| `max_ground_truths` | 30 | Reference cap per molecule | Controls compute/memory |

### Tuning guidance
- If AMR-P drops early (model becomes “diversity-pushy”), lower
  `lambda_smcov` (e.g. 0.5).
- If coverage metrics stall, raise `lambda_match` (1.5–2.0).
- Keep `rho ≈ delta` to maintain alignment with benchmark scale.
- `sigma ≈ delta / 3` balances dense gradients with accuracy pressure.

## 6. Integration with GRPO

### Grouping (`group_by_prompt`)
- Prompts include `[SMILES]..[/SMILES]` tags; canonical SMILES are used as
  group keys.
- Rewards are computed per group and then re-ordered to match completions.

### Configuration (`Config` / YAML)
```yaml
grpo:
  reward_strategy: "v3"
  delta: 0.75
  sigma: 0.25
  rho: 0.75
  lambda_qual: 1.0
  lambda_smcov: 1.0
  lambda_match: 1.0
  r_floor: -1.0
  enable_posebusters: false
  max_ground_truths: 30
```

### Statistics and logging (`RunStatistics`)
- `reward_v3/validity_rate`
- `reward_v3/mean_d_i`
- `reward_v3/mean_r_qual`, `mean_r_smcov`, `mean_r_match`
- `reward_v3/total_matched`, `fraction_under_delta`
- `reward_v3/avg_M`, `avg_K`

## 7. Computational Considerations

- Validity: $O(K)$ + optional PoseBusters cost.
- RMSD: $O(K \times M \times N)$ (N = atoms, Kabsch alignment dominates).
- Smooth coverage: $O(K^2 \times M)$ due to marginal products.
- Matching: $O(K^3)$ from Hungarian algorithm (small K keeps this manageable).
- Typical regime: $K=16$, $M=30$, $N≈40$.
- Use vectorized NumPy + cached reference loading to stay within wall-clock.

## 8. Edge Cases & Robustness

- **Missing references**: If no ground truth found, return `r_floor` for all K
  and log `failed_ground_truth`.
- **All invalid rollouts**: Entire group receives `r_floor`; downstream stats
  show 0 validity and 0 matches.
- **Numerical stability**: Smooth coverage uses log-space products and clamps
  kernel values to $[0,1]$ to prevent `nan`.
- **SMILES parsing errors**: Invalid RDKit molecules are treated as invalid
  rollouts; errors are logged for debugging.
- **PoseBusters optionality**: Keep disabled for throughput unless training
  drifts; enabling raises compute cost but enforces structural sanity.

## 9. Example Usage

```python
from molgen3D.training.grpo.grpo_reward_v3 import reward_function as reward_v3
from molgen3D.training.grpo.config import Config
from molgen3D.training.grpo.stats import RunStatistics
from transformers import AutoTokenizer

config = Config.from_yaml("configs/qwen3_v3_reward.yaml")
stats = RunStatistics(output_dir="./outputs")
tokenizer = AutoTokenizer.from_pretrained(config.model.tokenizer_path)

rewards = reward_v3(
    prompts=prompts,            # list[str], SMILES-tagged
    completions=completions,    # list[str], conformer blocks
    stats=stats,
    tokenizer=tokenizer,
    config=config,
)
```

The resulting list contains one reward per completion, already aligned with the
input order. These rewards feed directly into GRPO advantage estimation.

---

The V3 reward therefore couples dense precision pressure with group-aware
diversity incentives and a hard GEOM-Drugs-aligned coverage gate, providing a
single cohesive signal for training 3D molecular generators.