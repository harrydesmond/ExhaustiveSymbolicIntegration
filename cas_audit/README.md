# Verified CAS audit

Scripts for the test of computer algebra systems (CAS) in Secs. 2.4 and 3.3 of the revised paper.
Every claimed antiderivative is verified: the system that produced it differentiates it, and the result
must match the integrand at 8 points in (0.2, 5) at fixed parameter values (relative tolerance 1e-6),
with no unevaluated integral or error in the output. These scripts supersede the CAS-resistance
results of `../cas_comparison/`, whose success checks were defective (see "Relation to earlier
scripts" below).

The inputs and outputs of the runs used for the paper are in `cas_audit.tar.gz` in version 2 of the
Zenodo deposit (concept DOI 10.5281/zenodo.20027937), and `../results/cas_audit_summary.json` holds
the numbers quoted in the paper.

## Systems

| System | Version used | Environment variable (default) |
|---|---|---|
| SymPy | 1.14.0 | Python package |
| Mathematica | 13.3.1 | `ESI_WOLFRAMSCRIPT` (`wolframscript`) |
| RUBI | 4.17.3.0 (paclet) | run through `ESI_WOLFRAMSCRIPT` |
| FriCAS | 1.3.12 | `ESI_FRICAS` (`fricas`) |
| Maxima | 5.47.0 | `ESI_MAXIMA` (`maxima`) |
| Giac | 2.0.0 | `ESI_GIAC` (`icas`) |

Each variable may hold a full command, e.g. a container invocation; `env_glamdring.sh` is the
configuration used for the paper. Note that SageMath's `integrate(..., algorithm='giac')` silently
returns the integral unevaluated when Giac is not installed, which is why the harness calls Giac's
`icas` directly.

## Scripts

- `cas_audit.py`: the harness. Runs each (form, system, strategy) in its own process with a hard
  timeout, verifies the result and appends one JSON record per task to a checkpoint file, so an
  interrupted run resumes where it stopped. `--strip` removes absolute values from the antiderivative
  before differentiating; `--strategies fast` runs one call per system, `full` the full set (7 SymPy,
  11 Mathematica, 2 RUBI, 3 FriCAS, 5 Maxima, 5 Giac).
- `cas_audit_build.py`: builds the candidate list (integrands failed by SymPy and Mathematica in the
  screening sweeps) and the corresponding harness forms.
- `cas_audit_analyse.py`: per-set solve counts and survivors of a stage; `--write-survivors` writes
  the input of the next stage.
- `cas_audit_sample.py`: stratified random sample of integrands eliminated in screening.
- `cas_audit_reverify.py`: re-verifies answers at 40 digits with mpmath, for integrands whose
  values overflow the in-system check.
- `cas_audit_report.py`: regenerates `cas_audit_summary.json` from the deposit.
- `cas_audit_summary.py`: status counts per system for result files.
- `cas_abs_share.py`, `cas_const_check.py`: fraction of catalogue functions containing absolute
  values, and the lowest complexity of selected constant-bearing functions (Sec. 4.3).

## Reproducing the paper's runs

`SCREEN` is a directory holding the screening outputs (`results/` of Zenodo version 1 together with
`cas_audit/inputs/screening/` of version 2) and `RES` the directory with the lookup pickles. FriCAS
and Mathematica/RUBI tasks are slow and licence-limited respectively; `--workers` and `--mma-workers`
set the parallelism.

```bash
source env_glamdring.sh        # or set ESI_GIAC, ESI_MAXIMA, ESI_FRICAS, ESI_WOLFRAMSCRIPT

# 1. Candidates: 3,231 integrands that fail SymPy and Mathematica screening
python3 cas_audit_build.py --screening-dir $SCREEN        # audit_items.json, audit_forms_jobB.json

# 2. Giac, FriCAS and Maxima on every candidate (180 s)
python3 cas_audit.py audit_forms_jobB.json jobB_results.jsonl --strip \
    --engines giac,fricas,maxima --timeout 180
# Mathematica screening step for core_log_maths, which the original sweeps omitted
python3 -c "import json; json.dump([f for f in json.load(open('audit_forms_jobB.json')) \
    if f['set']=='core_log_maths'], open('audit_forms_corelog.json','w'))"
python3 cas_audit.py audit_forms_corelog.json corelog_mma.jsonl --strip --engines mma --timeout 180

# 3. All six systems with all strategies (600 s) on the survivors
python3 cas_audit_analyse.py audit_forms_jobB.json jobB_results.jsonl \
    --write-survivors audit_forms_stage2.json
python3 cas_audit.py audit_forms_stage2.json stage2_results.jsonl --strip \
    --engines giac,fricas,maxima,sympy,mma,rubi --strategies full --timeout 600
python3 cas_audit_analyse.py audit_forms_stage2.json stage2_results.jsonl \
    --write-survivors audit_forms_stage3.json

# 4. Real integrands of survivors containing absolute values, on intervals of fixed sign
#    (audit_forms_stage3_real.json lists the pieces and in-interval verification points)
python3 cas_audit.py audit_forms_stage3_real.json stage3_results.jsonl \
    --engines giac,fricas,maxima,sympy,mma,rubi --strategies full --timeout 600

# Integrals reported as CAS-resistant in the submitted version, and the two RUBI-suite integrals
python3 cas_audit.py audit_forms_jobA.json jobA_results.jsonl \
    --engines giac,fricas,maxima,sympy,mma,rubi --strategies full --timeout 600
python3 cas_audit.py audit_forms_rubi.json rubi_results.jsonl \
    --engines giac,fricas,maxima,sympy,mma,rubi --strategies full --timeout 600

# 5. Sample check of the screening's eliminations
python3 cas_audit_sample.py --results-dir $RES --screening-dir $SCREEN   # audit_forms_sample.json
python3 cas_audit.py audit_forms_sample.json sample_giac.jsonl --strip --engines giac --timeout 180
python3 cas_audit_analyse.py audit_forms_sample.json sample_giac.jsonl --write-survivors sample_after_giac.json
python3 cas_audit.py sample_after_giac.json sample_maxima.jsonl --strip --engines maxima --timeout 180
python3 cas_audit_analyse.py sample_after_giac.json sample_maxima.jsonl --write-survivors sample_after_maxima.json
python3 cas_audit.py sample_after_maxima.json sample_fricas.jsonl --strip --engines fricas --timeout 180
python3 cas_audit_analyse.py sample_after_maxima.json sample_fricas.jsonl --write-survivors sample_after_fricas.json
python3 cas_audit.py sample_after_fricas.json sample_s2_fast.jsonl --strip \
    --engines sympy,giac,maxima --strategies full --timeout 600
python3 cas_audit.py sample_after_fricas.json sample_s2_fricas.jsonl --strip \
    --engines fricas --strategies full --timeout 600
python3 cas_audit_analyse.py sample_after_fricas.json sample_s2_fast.jsonl sample_s2_fricas.jsonl \
    --write-survivors sample_after_s2a.json
python3 cas_audit.py sample_after_s2a.json sample_s2_mma.jsonl --strip \
    --engines mma,rubi --strategies full --timeout 600
python3 cas_audit_analyse.py sample_after_s2a.json sample_s2_mma.jsonl \
    --write-survivors sample_final_survivors.json
python3 cas_audit_reverify.py sample_final_survivors.json sample_s2_mma.jsonl sample_s2_fast.jsonl

# Summary (arrange the files as inputs/, outputs/, intermediate/, logs/ as in the deposit)
python3 cas_audit_report.py <dir> cas_audit_summary.json
```

## Results

Of 376,296 tested integrands, 3,092 fail both screening systems. Giac, FriCAS and Maxima solve
3,022 of these, the full strategy set solves 55 of the remaining 70, and 6 of the last 15 are
absolute-value artefacts whose real integrands are solved. Nine integrals resist all six systems
under every strategy (Table 4 of the paper): sqrt(x^x - x)/sqrt(x), sqrt(x + x^(e^x))/sqrt(x),
e^sqrt(x)/sqrt(x + sqrt(x)), e^(sqrt(x)/2)/sqrt(x + sqrt(x)) and five towers
x^(s e^(e^(x^2))/m + t/2). None of 10,648 integrands sampled from the screening's eliminations
resists all six systems.

## Relation to earlier scripts

The scripts in `../cas_comparison/` and the summaries `../results/final_cas_summary.json`,
`mma_stress_5_survivors_results.json`, `k9_followup_results.json`, `k9_broad_final_resistant.json`
and `trig_maths_final_resistant.json` underlie the submitted version of the paper. Their FriCAS
success test counted some "implementation incomplete" errors as solutions, RUBI was skipped for two of
the three integrals reported as CAS-resistant, and Giac never ran (it was called through SageMath
without Giac installed). They are kept for reference only.
