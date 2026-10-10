# Exhaustive Symbolic Integration

Code for the paper "Exhaustive Symbolic Integration: Integration by
Differentiation and the Landscape of Symbolic Integrability" (Desmond, 2026).

This GitHub repository is the maintained code repository. The larger data
files used by the paper are deposited separately on Zenodo:

- Concept DOI (latest version): `10.5281/zenodo.20027937`
- Version 1: `10.5281/zenodo.20027938`; version 2 (`10.5281/zenodo.23279393`) adds
  `cas_audit.tar.gz`, the verified CAS audit

ESI measures the integrability fraction `rho(k)`: the proportion of functions in
a bounded symbolic space whose derivatives also lie within that space. It also
identifies integrals that expose gaps in current computer algebra systems; the
verified test of six systems that does this is in [`cas_audit/`](cas_audit/README.md).

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

The paper data are deposited on Zenodo (concept DOI
<https://doi.org/10.5281/zenodo.20027937>). The download script finds the latest
version of the record, fetches `Archive.tar.gz` (and, with `--cas-audit`,
`cas_audit.tar.gz`) and extracts the requested subset.

```bash
python3 download_data.py              # Lookup database only
python3 download_data.py --raw        # Lookup database + raw outputs
python3 download_data.py --functions  # Function catalogues only
python3 download_data.py --cas-audit  # Verified CAS audit (version 2)
python3 download_data.py --all        # Lookup + raw outputs + catalogues + CAS audit
```

The Zenodo data package contains:

- final JSON and PKL result artifacts used for the paper tables and figures, and the
  CAS-comparison outputs of the submitted version, whose CAS-resistance results are
  superseded by the verified audit;
- `cas_audit.tar.gz` (version 2): inputs, outputs and summary of the verified CAS audit
  behind Tables 3 and 4 of the revised paper;
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

GitHub contains the code: the ESI pipeline, lookup tool, CAS-audit harness, earlier
CAS-comparison scripts, figure-generation scripts, and documentation. Zenodo contains the data files
consumed by those scripts.

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
|-- cas_audit/                (verified CAS audit; see cas_audit/README.md)
|   |-- cas_audit.py
|   |-- cas_audit_build.py
|   |-- cas_audit_sample.py
|   |-- cas_audit_analyse.py
|   |-- cas_audit_reverify.py
|   |-- cas_audit_report.py
|   |-- cas_audit_summary.py
|   |-- cas_abs_share.py
|   |-- cas_const_check.py
|   +-- env_glamdring.sh
|
|-- cas_comparison/          (scripts of the submitted version; superseded)
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
- Wolfram Mathematica (13.3 used) and RUBI 4.17.3 for Mathematica and RUBI tests
- FriCAS 1.3.12, Maxima 5.47 and Giac 2.0.0 for the verified CAS audit
- RUBI test suite files for the RUBI benchmark comparison
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

The CAS results of the revised paper come from the verified audit in
[`cas_audit/`](cas_audit/README.md), which uses the screening sweeps below as its
first stage and then tests every surviving integrand with Giac, FriCAS, Maxima,
SymPy, Mathematica and RUBI, verifying each claimed antiderivative by
differentiation. The commands below are the screening sweeps by SymPy and
Mathematica. The FriCAS and other follow-up scripts listed after them belong to the
submitted version and are superseded by the audit.

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
  `rho(k_max)=0.035`; no integral resists all six CAS.
- `core_log_maths`: `k_max=8`, 21,214 deduplicated functions,
  `rho(k=6)=0.192`, `rho(k_max)=0.174`; no integral resists all six CAS.
- `ext_maths`: `k_max=9`, 628,400 deduplicated functions, `rho(k=6)=0.041`,
  `rho(k_max)=0.029`; one integral resists all six CAS.
- `ext_log_maths`: `k_max=9`, 1,454,666 deduplicated functions,
  `rho(k=6)=0.116`, `rho(k_max)=0.083`; eight integrals at `k=9` resist all six CAS.
- `trig_maths`: `k_max=9`, 897,293 deduplicated functions, `rho(k=6)=0.049`,
  `rho(k_max)=0.037`; no integral resists all six CAS.

Adding `log` to the basis boosts `rho(k)` by factors of order 2-3 over most
shared complexity levels and produces a non-monotonic peak at `k=6` in the ESR
node-count metric.

The verified audit finds nine integrals with elementary antiderivatives that resist
SymPy 1.14, Mathematica 13.3, RUBI 4.17.3, FriCAS 1.3.12, Maxima 5.47 and Giac 2.0.0
under every strategy tested, among 376,296 tested integrands; none of 10,648
integrands sampled from those eliminated in screening resists all six. The numbers
are in `results/cas_audit_summary.json`; `results/final_cas_summary.json` and the
other CAS files in `results/` belong to the submitted version and are superseded.

## Citation

```bibtex
@article{Desmond2026,
  author  = {Harry Desmond},
  title   = {Exhaustive Symbolic Integration: Integration by Differentiation
             and the Landscape of Symbolic Integrability},
  journal = {Mathematics in Computer Science (submitted)},
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
  doi       = {10.5281/zenodo.20027937},
  url       = {https://doi.org/10.5281/zenodo.20027937}
}
```

## License

This repository is licensed under the GNU General Public License v3.0. See
[LICENSE](LICENSE).

## Contact

For questions or comments, email Harry Desmond
(harry.desmond@port.ac.uk).
