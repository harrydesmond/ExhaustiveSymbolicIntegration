# Exhaustive Symbolic Integration

Code for the paper "Exhaustive Symbolic Integration: Integration by
Differentiation and the Landscape of Symbolic Integrability" (Desmond, 2026).

This GitHub repository is the maintained code and paper-source repository. The
larger data files used by the paper are deposited separately on Zenodo:

- DOI: `10.5281/zenodo.20027938`
- Record: <https://zenodo.org/records/20027938>

ESI measures the integrability fraction `rho(k)`: the proportion of functions in
a bounded symbolic space whose derivatives also lie within that space. It also
identifies integrals that expose gaps in current computer algebra systems.

## The ESI Pipeline

```text
Load F_<=k
  -> canonicalise parameters
  -> differentiate with SymPy
  -> numerical fingerprint at 60 points, 50-digit precision
  -> hash and cluster by derivative fingerprint
  -> compute rho(k) and build equivalence classes
```

Each expression is evaluated at 60 random points (`x` in `(0.2, 5)`,
parameters in `(0.5, 3.0)`), rounded to 10 significant figures, and MD5-hashed.
The false-positive rate is below `10^{-590}` per pair, with zero empirical false
positives found in 518K pair comparisons.

## Quick Start

### Use The Integration Lookup Tool

```bash
# 1. Download the lookup database from Zenodo.
python3 download_data.py

# 2. Look up an integral.
python3 esi_integrate.py "1/x"                    # -> log(x)
python3 esi_integrate.py "x**x*(log(x)+1)"        # -> x^x
python3 esi_integrate.py "exp(x**2)"              # -> Not found
python3 esi_integrate.py -v "exp(x)"              # Verbose mode
python3 esi_integrate.py -i                       # Interactive mode
```

The default download extracts only the precomputed lookup database needed by
`esi_integrate.py`. The ESR function catalogues are needed only when rerunning
the ESI pipeline from the enumerated functions.

The notebook `esi_integrate_demo.ipynb` provides the same lookup workflow in an
interactive form.

### Reproduce The Paper Results

```bash
# 1. Download lookup data, raw outputs, and function catalogues.
python3 download_data.py --all

# 2. Run the ESI pipeline.
python3 run_parallel.py \
    --data-dir function_catalogues/ext_log_maths \
    --max-complexity 8 \
    --ncores 4 \
    --mode both

# 3. Analyse rho decomposition and delta distribution.
python3 analysis/analyse_rho_decomposition.py \
    --raw results/raw_results_ext_log_maths_k8.pkl \
    --max-k 8

# 4. Generate paper figures.
python3 figures/make_figures.py
```

## Data Download

The paper data are deposited on Zenodo at
<https://zenodo.org/records/20027938>. The download script fetches the published
`Archive.tar.gz` file and extracts the requested subset.

```bash
python3 download_data.py              # Lookup database only
python3 download_data.py --raw        # Lookup database + raw outputs
python3 download_data.py --functions  # Function catalogues only
python3 download_data.py --all        # Lookup + raw outputs + catalogues
```

The Zenodo data package contains:

- final JSON and PKL result artifacts used for the paper tables, figures, and
  CAS-resistance claims;
- lookup-equivalence databases for `core_maths`, `core_log_maths`, `ext_maths`,
  `ext_log_maths`, and `trig_maths`, plus `esi_hash_index.json`;
- raw derivative and fingerprint outputs needed for rho decomposition and audit
  scripts;
- `function_catalogues_unique_equations.tar.gz`, containing only the
  `unique_equations_k.txt` catalogues for the five bases.

Large ESR generation intermediates (`trees`, `orig_trees`, `matches`,
`inv_subs`, etc.), cluster logs, and external CAS installations are intentionally
omitted from the Zenodo package. They are reproducible or environment-specific
and are not used directly by the paper analysis scripts.

GitHub contains the code: the ESI pipeline, lookup tool, CAS-comparison scripts,
figure-generation scripts, paper source, and documentation. Zenodo contains the
data files consumed by those scripts.

## Repository Structure

```text
ESI/
|-- README.md
|-- LICENSE
|-- download_data.py
|-- esi_integrate.py
|-- esi_integrate_demo.ipynb
|-- esi_pipeline.py
|-- run_parallel.py
|
|-- analysis/
|   |-- analyse_rho_decomposition.py
|   |-- compute_trig_model.py
|   |-- theoretical_model.py
|   |-- theoretical_model_refined.py
|   |-- cluster_stats.py
|   |-- collision_check.py
|   +-- abs_dedup_check.py
|
|-- cas_comparison/
|   |-- cas_full_sympy.py
|   |-- cas_full_mathematica.py
|   |-- cas_sympy_allbases.py
|   |-- cas_mma_allbases.py
|   |-- cas_stress_600s.py
|   |-- cas_stress_allbases.py
|   |-- cas_fricas_test.py
|   |-- cas_fricas_retest.py
|   |-- cas_fricas_verify.py
|   |-- cas_singleton_test.py
|   |-- extract_singleton_failures.py
|   |-- rubi_comparison.py
|   |-- rubi_comparison_fast.py
|   |-- check_rubi_failures.py
|   |-- merge_mma_ext_chunks.py
|   +-- table_fg_integrals.py
|
|-- figures/
|   |-- make_figures.py
|   +-- fig_cluster_sizes.py
|
|-- generation/
|   |-- gen_trig_all.py
|   |-- gen_ext_log_mpi.py
|   +-- gen_trig_mpi.py
|
|-- paper/
|   |-- Draft.tex
|   |-- references.bib
|   |-- table_cas_blind.tex
|   +-- table_fg_integrals.tex
|
|-- function_catalogues/
|   |-- core_maths/compl_*/
|   |-- ext_maths/compl_*/
|   |-- ext_log_maths/compl_*/
|   +-- trig_maths/compl_*/
|
+-- results/
    |-- rho_results_*.json
    |-- rho_decomposition_*.json
    |-- cas_full_*_results.json
    |-- theoretical_model_all_bases.json
    +-- *.pkl
```

`function_catalogues/` and `results/` are populated by `download_data.py`; large
files are not stored directly in Git.

## Reproducing The Paper Results

### Prerequisites

- Python 3.11+ with `sympy`, `numpy`, `mpmath`, and `matplotlib`
- `mpi4py` for MPI-parallel runs
- Wolfram Mathematica for Mathematica comparisons
- FriCAS 1.3.12+ for FriCAS comparisons
- RUBI test suite files for RUBI comparisons
- [ESR](https://github.com/DeaglanBartlett/ESR) for function generation

### Step 1: Function Space Generation

Pre-generated function catalogues for all five operator bases are available via
`download_data.py --functions` or `download_data.py --all`. To regenerate from
scratch using ESR:

```bash
# Ext_log basis
mpirun -np 20 python3 generation/gen_ext_log_mpi.py

# Trig basis
mpirun -np 20 python3 generation/gen_trig_all.py
```

This produces `unique_equations_k.txt` files in
`ESR/esr/function_library/<basis>/compl_k/`.

### Step 2: ESI Pipeline

For each basis, run the pipeline to differentiate all functions, compute
fingerprints, measure `rho(k)`, and build equivalence classes:

```bash
python3 run_parallel.py \
    --data-dir /path/to/ESR/esr/function_library/ext_log_maths \
    --max-complexity 8 \
    --ncores 20 \
    --mode both \
    --output-prefix ext_log_maths_k8
```

This produces:

- `rho_results_ext_log_maths_k8.json`: `rho(k)` at each complexity;
- `results_ext_log_maths_k8.pkl`: equivalence classes;
- `raw_results_ext_log_maths_k8.pkl`: per-expression results.

Repeat for each basis: `core_maths`, `core_log_maths`, `ext_maths`,
`ext_log_maths`, and `trig_maths`.

### Step 3: Analysis

```bash
# Rho decomposition by operator presence and delta distribution.
python3 analysis/analyse_rho_decomposition.py \
    --raw raw_results_ext_log_maths_k8.pkl \
    --max-k 8

# Growth-rate model parameters.
python3 analysis/compute_trig_model.py

# Fingerprint collision validation.
python3 analysis/collision_check.py
```

### Step 4: CAS Comparison

The CAS comparison tests multi-member equivalence classes against SymPy,
Mathematica, RUBI, and FriCAS. This is compute-intensive and best run on a
cluster.

For the primary `ext_log_maths` analysis:

```bash
# SymPy: MPI, around 200 cores, around 1 hour.
mpirun -np 200 python3 cas_comparison/cas_full_sympy.py

# Mathematica: MPI, around 5 ranks because of license limits.
mpirun -np 5 python3 cas_comparison/cas_full_mathematica.py

# Deep stress test on integrals failed by both engines.
python3 cas_comparison/cas_stress_600s.py --index 0 --with-mma
```

Across the other CAS-tested bases:

```bash
# SymPy across bases.
mpirun -np 84 python3 cas_comparison/cas_sympy_allbases.py \
    --pkl results_core_maths_k10.pkl \
    --out cas_sympy_core_maths.json

# Mathematica across bases.
mpirun -np 5 python3 cas_comparison/cas_mma_allbases.py \
    --pkl results_ext_maths_k9.pkl \
    --out cas_mma_ext_maths.json

# Chunk large bases.
mpirun -np 5 python3 cas_comparison/cas_mma_allbases.py \
    --pkl results_ext_maths_k9.pkl \
    --out cas_mma_ext_maths_chunk_00.json \
    --start 0 \
    --end 9052

# Merge chunked Mathematica results.
python3 cas_comparison/merge_mma_ext_chunks.py

# Deep stress test across bases.
mpirun -np 28 python3 cas_comparison/cas_stress_allbases.py \
    --input stress_input_core_maths.json \
    --out stress_core_maths.json
```

FriCAS checks:

```bash
python3 cas_comparison/cas_fricas_test.py --start 0 --end 100
python3 cas_comparison/cas_fricas_retest.py
python3 cas_comparison/cas_fricas_verify.py
```

RUBI test-suite cross-check:

```bash
python3 cas_comparison/rubi_comparison.py
python3 cas_comparison/check_rubi_failures.py
```

### Step 5: Figures

```bash
python3 figures/make_figures.py
python3 figures/fig_cluster_sizes.py
```

## Key Results

- `core_maths`: `k_max=10`, 77,053 deduplicated functions, `rho(k=6)=0.058`,
  `rho(k_max)=0.035`; CAS cascade tested with no final all-engine failures.
- `core_log_maths`: `k_max=8`, 21,214 deduplicated functions,
  `rho(k=6)=0.192`, `rho(k_max)=0.174`; SymPy-only fill-in gives 5,669 tested,
  144 hard, and 59 stress-impossible. This basis is not in the full CAS
  cascade.
- `ext_maths`: `k_max=9`, 628,400 deduplicated functions, `rho(k=6)=0.041`,
  `rho(k_max)=0.029`; two robust all-engine failures.
- `ext_log_maths`: `k_max=9`, 1,454,666 deduplicated functions,
  `rho(k=6)=0.116`, `rho(k_max)=0.083`; one robust all-engine failure plus two
  standard-call Abs-only failures.
- `trig_maths`: `k_max=9`, 897,293 deduplicated functions, `rho(k=6)=0.049`,
  `rho(k_max)=0.037`; CAS cascade tested with no final all-engine failures.

Adding `log` to the basis boosts `rho(k)` by factors of order 2-3 over most
shared complexity levels and produces a non-monotonic peak at `k=6` in the ESR
node-count metric.

The final six-engine cascade identifies five all-engine CAS-resistant integrals
under the standard call. Three resist all tested strategies; two are
absolute-value cases solved by Mathematica after a domain restriction or
roundtrip transformation, and are therefore not counted in the robust
all-strategy headline.

Final CAS-result artifacts are in `results/final_cas_summary.json` and the
supporting files named there. Table-level stress-test artifacts are included as
`results/stress_results_*.json`.

## Citation

```bibtex
@article{Desmond2026,
  author  = {Harry Desmond},
  title   = {Exhaustive Symbolic Integration: Integration by Differentiation
             and the Landscape of Symbolic Integrability},
  journal = {Journal of Symbolic Computation},
  year    = {2026}
}
```

Data package:

```bibtex
@dataset{Desmond2026ESIData,
  author    = {Desmond, Harry},
  title     = {Data for "Exhaustive Symbolic Integration: Integration by
               Differentiation and the Landscape of Symbolic Integrability"},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.20027938},
  url       = {https://zenodo.org/records/20027938}
}
```

## License

This repository is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE).

## Contact

For questions or comments, email Harry Desmond
(harry.desmond@port.ac.uk).
