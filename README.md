# ESI: Exhaustive Symbolic Integration

Code and data for "Operator Basis Determines the Landscape of Symbolic Integrability" (Desmond, 2026).

ESI measures the integrability fraction rho(k) -- the proportion of functions in a bounded symbolic space whose derivatives also lie within that space -- and identifies integrals that expose gaps in current computer algebra systems.

## Quick start

### Use the integration lookup tool

```bash
# 1. Download the lookup database (~650MB)
python3 download_data.py

# 2. Look up an integral
python3 esi_integrate.py "1/x"                    # -> log(x)
python3 esi_integrate.py "x**x*(log(x)+1)"        # -> x^x
python3 esi_integrate.py "exp(x**2)"               # -> Not found (non-elementary)
python3 esi_integrate.py -v "exp(x)"               # Verbose: show all representations
python3 esi_integrate.py -i                         # Interactive mode
```

Or use the Jupyter notebook `esi_integrate_demo.ipynb` for interactive exploration.

### Reproduce the paper results

```bash
# 1. Download everything needed for lookup, auditing, and pipeline input regeneration (~1.9GB)
python3 download_data.py --all

# 2. Run the ESI pipeline (differentiate, fingerprint, cluster)
python3 run_parallel.py \
    --data-dir function_catalogues/ext_log_maths \
    --max-complexity 8 \
    --ncores 4 \
    --mode both

# 3. Analyse: rho decomposition, delta distribution
python3 analysis/analyse_rho_decomposition.py \
    --raw results/raw_results_ext_log_maths_k8.pkl \
    --max-k 8

# 4. Generate paper figures
python3 figures/make_figures.py
```

## Repository structure

```
ESI/
|-- README.md
|-- download_data.py         # Download databases from Zenodo
|-- esi_integrate.py         # Integration lookup tool (command-line)
|-- esi_integrate_demo.ipynb # Integration lookup tool (Jupyter notebook)
|-- esi_pipeline.py          # Core library: parse, canonicalise, differentiate, fingerprint, cluster
|-- run_parallel.py          # Main pipeline: parallelised over expressions, produces rho(k) + clusters
|
|-- analysis/
|   |-- analyse_rho_decomposition.py   # Decompose rho(k) by operator presence; delta distribution
|   |-- compute_trig_model.py          # Growth-rate crossover model (alpha, beta, alpha/beta)
|   |-- theoretical_model.py           # Theoretical model for rho(k) scaling
|   |-- theoretical_model_refined.py   # Refined model with basis-specific parameters
|   |-- cluster_stats.py               # Cluster-size distribution statistics
|   |-- collision_check.py             # Empirical validation of fingerprint collision rate
|   +-- abs_dedup_check.py             # Verify Abs-wrapping doesn't bias rho(k)
|
|-- cas_comparison/
|   |-- cas_full_sympy.py              # SymPy comparison on ext_log_maths (MPI, 4 strategies, 180s)
|   |-- cas_full_mathematica.py        # Mathematica comparison on ext_log_maths (MPI, assumptions, 180s)
|   |-- cas_sympy_allbases.py          # SymPy comparison across the CAS-tested bases (MPI)
|   |-- cas_mma_allbases.py            # Mathematica comparison across the CAS-tested bases (MPI, --start/--end for chunking)
|   |-- cas_stress_600s.py             # Deep stress test on ext_log_maths: 600s, 9 SymPy + 3 MMA strategies
|   |-- cas_stress_allbases.py         # Deep stress test across all bases (MPI, 600s, all strategies)
|   |-- cas_fricas_test.py             # FriCAS comparison on CAS-blind integrals
|   |-- cas_fricas_retest.py           # FriCAS retest with refined parameters
|   |-- cas_fricas_verify.py           # Final FriCAS verification of genuinely resistant integrals
|   |-- cas_singleton_test.py          # SymPy test on singleton equivalence classes
|   |-- extract_singleton_failures.py  # Extract SymPy-hard singletons for FriCAS testing
|   |-- rubi_comparison.py             # Compare ESI against RUBI test suite (72,401 integrals)
|   |-- rubi_comparison_fast.py        # Fast RUBI comparison via fingerprint matching
|   |-- check_rubi_failures.py         # Check RUBI's own failure cases against ESI
|   |-- merge_mma_ext_chunks.py        # Merge chunked Mathematica results into single JSON
|   +-- table_fg_integrals.py          # Build verification table of f^g CAS-blind integrals
|
|-- figures/
|   |-- make_figures.py                # Generate all paper figures (rho, decomposition, model, delta)
|   +-- fig_cluster_sizes.py           # Cluster-size distribution figure
|
|-- generation/
|   |-- gen_trig_all.py                # Generate trig_maths function set via ESR (MPI)
|   |-- gen_ext_log_mpi.py             # Generate ext_log_maths function set via ESR (MPI)
|   +-- gen_trig_mpi.py                # Generate trig_maths functions at specific complexities (MPI)
|
|-- paper/
|   |-- Draft.tex                      # Paper source
|   |-- references.bib                 # Bibliography
|   |-- table_cas_blind.tex            # 5 final all-engine CAS-resistant integrals (LaTeX table)
|   +-- table_fg_integrals.tex         # f^g integral table (LaTeX)
|
|-- function_catalogues/               # Downloaded unique-equation catalogues (see Data section)
|   |-- core_maths/compl_*/            # ESR function sets (symlinks or downloads)
|   |-- ext_maths/compl_*/
|   |-- ext_log_maths/compl_*/
|   +-- trig_maths/compl_*/
|
+-- results/                           # Pipeline outputs
    |-- rho_results_*.json             # rho(k) for each basis
    |-- rho_decomposition_*.json       # Operator decomposition + delta distribution
    |-- cas_full_*_results.json        # Comprehensive CAS comparison results (ext_log_maths)
    |-- theoretical_model_all_bases.json
    +-- *.pkl                          # Raw pipeline outputs (large)
```

## Reproducing the paper results

### Prerequisites

- Python 3.11+ with: `sympy`, `numpy`, `mpmath`, `matplotlib`
- For MPI-parallel runs: `mpi4py`
- For Mathematica comparison: Wolfram Mathematica (command-line `math` binary)
- For FriCAS comparison: FriCAS 1.3.12+ (via Singularity container or native install)
- For RUBI comparison: RUBI test suite files (Mathematica `.m` format)
- For function generation: [ESR](https://github.com/DeaglanBartlett/ESR)

### Data download

All data can be downloaded from Zenodo using `download_data.py`:

```bash
python3 download_data.py              # Lookup database only (~650MB) -- enough for esi_integrate.py
python3 download_data.py --all        # Database + raw outputs + function catalogues (~1.9GB)
python3 download_data.py --functions  # Function sets only (for reproducing the pipeline)
```

The prepared Zenodo data package contains:

- final JSON and PKL result artifacts used for the paper tables, figures, and CAS-resistance claims;
- lookup-equivalence databases for `core_maths`, `core_log_maths`, `ext_maths`, `ext_log_maths`, and `trig_maths`, plus `esi_hash_index.json`;
- raw derivative/fingerprint outputs needed for rho decomposition and audit scripts;
- `function_catalogues_unique_equations.tar.gz`, containing only the `unique_equations_k.txt` catalogues for the five bases.

Large ESR generation intermediates (`trees`, `orig_trees`, `matches`, `inv_subs`, etc.), cluster logs, and external CAS installations are intentionally omitted from the Zenodo package. They are reproducible or environment-specific and are not used directly by the paper analysis scripts. The staged local upload package is in `zenodo/upload/`; after Zenodo assigns a record, replace the placeholder `ZENODO_RECORD` in `download_data.py`.

### Step 1: Function space generation

Pre-generated function catalogues for all five operator bases are available via `download_data.py --functions`.
To regenerate from scratch using ESR:

```bash
# Core, ext bases (already available in ESR's function_library/)
# Ext_log basis:
mpirun -np 20 python3 generation/gen_ext_log_mpi.py
# Trig basis:
mpirun -np 20 python3 generation/gen_trig_all.py
```

This produces `unique_equations_k.txt` files in `ESR/esr/function_library/<basis>/compl_k/`.

### Step 2: ESI pipeline

For each basis, run the pipeline to differentiate all functions, compute
fingerprints, measure rho(k), and build equivalence classes:

```bash
python3 run_parallel.py \
    --data-dir /path/to/ESR/esr/function_library/ext_log_maths \
    --max-complexity 8 \
    --ncores 20 \
    --mode both \
    --output-prefix ext_log_maths_k8
```

This produces:
- `rho_results_ext_log_maths_k8.json` -- rho(k) at each complexity
- `results_ext_log_maths_k8.pkl` -- equivalence classes (clusters)
- `raw_results_ext_log_maths_k8.pkl` -- per-expression results

Repeat for each basis (core_maths, ext_maths, ext_log_maths, trig_maths).

### Step 3: Analysis

```bash
# Rho decomposition by operator presence + delta distribution
python3 analysis/analyse_rho_decomposition.py \
    --raw raw_results_ext_log_maths_k8.pkl --max-k 8

# Growth-rate model parameters
python3 analysis/compute_trig_model.py

# Fingerprint collision validation
python3 analysis/collision_check.py
```

### Step 4: CAS comparison

The CAS comparison tests multi-member equivalence classes against SymPy,
Mathematica, RUBI, and FriCAS. This is compute-intensive and best run on a cluster.

#### ext_log_maths (primary analysis, 49,538 clusters)

```bash
# SymPy (MPI, ~200 cores, ~1 hour)
mpirun -np 200 python3 cas_comparison/cas_full_sympy.py

# Mathematica (MPI, 5 cores due to license limits, ~6 hours)
mpirun -np 5 python3 cas_comparison/cas_full_mathematica.py

# Deep stress test on both-fail integrals (600s timeout, all strategies)
python3 cas_comparison/cas_stress_600s.py --index 0 --with-mma
```

#### All bases (core_maths, ext_maths, trig_maths)

```bash
# SymPy across all bases (MPI, no -s flag)
mpirun -np 84 python3 cas_comparison/cas_sympy_allbases.py \
    --pkl results_core_maths_k10.pkl --out cas_sympy_core_maths.json

# Mathematica across all bases (MPI, max 5 ranks for license)
mpirun -np 5 python3 cas_comparison/cas_mma_allbases.py \
    --pkl results_ext_maths_k9.pkl --out cas_mma_ext_maths.json

# For large bases, use --start/--end to chunk:
mpirun -np 5 python3 cas_comparison/cas_mma_allbases.py \
    --pkl results_ext_maths_k9.pkl --out cas_mma_ext_maths_chunk_00.json \
    --start 0 --end 9052

# Merge chunked results
python3 cas_comparison/merge_mma_ext_chunks.py

# Deep stress test across all bases (MPI, 600s, 9 SymPy + 3 MMA strategies)
mpirun -np 28 python3 cas_comparison/cas_stress_allbases.py \
    --input stress_input_core_maths.json --out stress_core_maths.json
```

#### FriCAS (on CAS-blind integrals)

```bash
# Test CAS-blind integrals against FriCAS
python3 cas_comparison/cas_fricas_test.py --start 0 --end 100

# Retest with refined parameters
python3 cas_comparison/cas_fricas_retest.py

# Final verification of genuinely resistant integrals
python3 cas_comparison/cas_fricas_verify.py
```

#### RUBI test suite cross-check

```bash
# Compare ESI database against RUBI's 72,401 test integrals
python3 cas_comparison/rubi_comparison.py

# Check RUBI's own failure cases
python3 cas_comparison/check_rubi_failures.py
```

### Step 5: Figures

```bash
python3 figures/make_figures.py           # Main paper figures
python3 figures/fig_cluster_sizes.py      # Cluster size distribution
```

## Key results

| Basis | k_max | deduplicated | rho(k=6) | rho(k_max) | CAS status |
|-------|-------|---------------|----------|------------|------------|
| core_maths | 10 | 77,053 | 0.058 | 0.035 | CAS cascade tested; no final all-engine failures |
| core_log_maths | 8 | 21,214 | **0.192** | 0.174 | SymPy-only fill-in: 5,669 tested, 144 hard, 59 stress-impossible; not in the full CAS cascade |
| ext_maths | 9 | 628,400 | 0.041 | 0.029 | 2 final all-engine failures |
| ext_log_maths | 9 | 1,454,666 | 0.116 | 0.083 | 3 final all-engine failures |
| trig_maths | 9 | 897,293 | 0.049 | 0.037 | CAS cascade tested; no final all-engine failures |

- Adding log to the basis boosts rho(k) by factors of order 2--3 over most shared complexity levels and produces a non-monotonic peak at k=6 in the ESR node-count metric.
- `core_log_maths` isolates the logarithm effect for rho(k). A lightweight SymPy-only fill-in was run after the main analysis, but the full six-engine CAS cascade still covers the four other bases.
- In `ext_log_maths`, 503 integrals fail both SymPy and Mathematica at 180 s. Extended SymPy/Mathematica stress tests solve 270, leaving 232 that still resist both engines; FriCAS solves 218 of these.
- The final six-engine cascade identifies **5 all-engine CAS-resistant integrals under the standard call**. Three are robust under every Mathematica variant tested; two are absolute-value cases solved only after a domain restriction or roundtrip transformation.
- Broad k=9 and `trig_maths` follow-up cascades add zero new all-engine CAS-resistant integrals (`k9_broad_final_resistant.json`, `trig_maths_final_resistant.json`).
- ESI solves 184 RUBI benchmark integrals, including 7 classified as "hard" by RUBI. A separate check of RUBI's reported failures gives 134 failures, 122 mappable into ESI's fingerprint framework, 3 raw matches, and 2 robust solves after removing one branch artifact.

Final CAS-result artifacts are in `results/final_cas_summary.json` and the supporting files named there. Table-level stress-test artifacts are also included as `results/stress_results_*.json`, including the late `core_log_maths` SymPy fill-in.

## The ESI pipeline

```
Load F_<=k  ->  Canonicalise params  ->  Differentiate (SymPy)
    ->  Numerical fingerprint (60 pts, 50-digit, 10 sig fig, MD5)
    ->  Hash  ->  Cluster by derivative hash
    ->  Compute rho(k), build equivalence classes
```

**Numerical fingerprinting:** Each expression is evaluated at 60 random points
(x in (0.2, 5), params in (0.5, 3.0)) at 50-digit precision, rounded to 10
significant figures, and MD5-hashed. False positive rate < 10^{-590} per pair;
empirically validated with 0 false positives in 518K pair comparisons.

## Citation

```bibtex
@article{Desmond2026,
  author = {Harry Desmond},
  title  = {Operator Basis Determines the Landscape of Symbolic Integrability},
  journal = {Journal of Symbolic Computation},
  year   = {2026}
}
```

## License

[TBD]
