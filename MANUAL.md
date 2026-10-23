# succd — user manual

This manual explains how to use the code on your own molecules and what every
function, script and data file does. `README.md` covers installation and the
reproduction of the article; `notebooks/2_tutorial_one_molecule.ipynb` walks
through the method step by step on N₂.

Contents

1. [The method in five calls](#1-the-method-in-five-calls)
2. [Running your own molecule](#2-running-your-own-molecule)
3. [Python API](#3-python-api)
4. [Command-line scripts](#4-command-line-scripts)
5. [Settings and environment variables](#5-settings-and-environment-variables)
6. [Data files](#6-data-files)
7. [Figures](#7-figures)
8. [Cost and limits](#8-cost-and-limits)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. The method in five calls

| Article step | What it does | Call |
|---|---|---|
| Step 1, Initialization | RHF, integrals, qubit Hamiltonian, PT2 score E_i of every double (eq 1) | `C = core.build(name)` |
| Step 2, Ansatz expansion | keep the doubles with \|E_i\| > δ (eq 2), ranked by \|E_i\| | `N = core.nsel(C)` |
| Step 3, Ansatz construction | s-CID coefficients c_m, c_m/c₀, s-CCD amplitudes t_m, signs from MP1 (eq 7) | `P = core.params(C, N, False)` |
| Step 3, eq 6 | rescaled parameters d_m = (2 − c₀²)/c₀ · c_m | `d = (2 - P['c0']**2) / P['c0'] * P['c']` |
| Step 4, outputs | circuit energy and the prepared state | `core.evaluator(C)(d, N)`, `core.state(C, d)` |

## 2. Running your own molecule

```python
from succd import geoms, core, units

# xyz block in Angstrom, and the number of frozen core spatial orbitals
geoms.SYSTEMS['H2O'] = ("""O   0.0000   0.0000   0.1173
H   0.0000   0.7572  -0.4692
H   0.0000  -0.7572  -0.4692""", 1)

C = core.build('H2O')                 # Step 1
N = core.nsel(C)                      # Step 2
P = core.params(C, N, False)          # Step 3
c0 = P['c0']
d = (2 - c0**2) / c0 * P['c']         # eq 6
E = core.evaluator(C)                 # E(params, N) in kcal/mol, relative to HF

for label, p in (('c_m', P['c']), ('d_m', d), ('t_m', P['t'])):
    e = E(p, N)
    print(label, '%.4f kcal/mol  %.3f mHa' % (e, units.kcalmol_to_mha(e)))
print('s-CID', P['E_sCID'], ' s-CCD', P['E_sCCD'])
psi = core.state(C, d)                # statevector of the d_m circuit
```

Output on this machine (STO-6G, frozen 1s, 12 qubits, about 30 s):

```
selected doubles N = 21   c0 = 0.98728   alpha_est = 1.0385
E_corr s-UCCD(c_m) = -31.0153 kcal/mol = -49.426 mHa
E_corr s-UCCD(d_m) = -31.0254 kcal/mol = -49.442 mHa
E_corr s-UCCD(t_m) = -31.0320 kcal/mol = -49.453 mHa
E_corr s-CID       = -30.6407 kcal/mol
E_corr s-CCD       = -31.0149 kcal/mol
```

Rules for the geometry block: one atom per line, `symbol  x  y  z`, Ångström.
Frozen core: 0 for H/He, 1 per first-row atom (Li–Ne), 5 per second-row atom
(Na–Ar). The basis is STO-6G (`core.build` passes `basis='sto-6g'`); a larger
basis needs `selected_cc_pipeline.build_union_context(xyz, basis=..., n_frozen=...)`
directly and grows the qubit count quickly (section 8).

Running the full analysis of the article on your molecule: add it to
`succd/geoms.py` (or to the `GEOMETRIES` cell of notebooks 4 and 5) and use
the scripts of section 4 with its key, e.g. `python scripts/s2_system.py H2O`.

## 3. Python API

All energies are **correlation energies in kcal/mol relative to RHF**, unless
a name says otherwise. `succd.units` converts (`kcalmol_to_mha`).

### `succd.core`

| Function | Returns |
|---|---|
| `build(name)` | context dict `C` for `geoms.SYSTEMS[name]`: integrals (`h1`, `eri`), orbital and electron counts (`M`, `na`, `nb`), PT2 table (`C['dfr']`, columns `E2_i`, `t_mp`), determinants, excitation generators; `C['_tap']` is True when a Z₂-tapered Hamiltonian is used (faster) |
| `nsel(C)` | N, the number of doubles with \|E_i\| > `core.DELTA`; the doubles in `C` are ranked and pruned in place |
| `params(C, N, False)` | dict: `c` (c_m with signs), `cp` (c_m/c₀), `t` (t_m), `c0`, `E_sCID`, `E_sCCD`, `dbl` (the N doubles as spin-orbital index tuples), `mp2` (signs) |
| `params(C, N, True)` | the singles-inclusive sets of Application 2 (`t`, raw c magnitudes, singles signs, `take` = indices of the selected singles) |
| `evaluator(C, take=())` | function `E(p, N)` → circuit energy for parameters `p` (doubles first, then the singles in `take`) |
| `state(C, p, take=())` | full Jordan–Wigner statevector (numpy array, length 2^(2M)) |
| `DELTA` | selection threshold in Hartree, default 3e-4; set `core.DELTA = ...` before `nsel` |
| `K` | 627.5094740631, Hartree → kcal/mol |

### `succd.drv`

| Function | Returns |
|---|---|
| `ci_energy(psi, C)` | ⟨H⟩ of a statevector, via PySCF FCI contraction |
| `sqd(psi, C, shots=[1e6, 1e8], seed=2024)` | SQD energies: dict per budget with `N` (determinants observed) and `E` |
| `_sd_sets(C, N)` | singles-inclusive parameter sets `c`, `cp`, `t`, the selected singles, and s-CISD info |
| `stage_*` | the blocks used by `scripts/s2_system.py` |

### `succd.geoms`, `succd.paths`, `succd.units`

`geoms.SYSTEMS` maps a name to `(xyz, n_frozen)`. `paths` holds the data
folders (`paths.RESULTS`, `paths.VALUES`, `paths.SI`, `paths.SHOT_SWEEP`,
`paths.FIGURES`) and helpers `paths.results(name)`, …. `units` holds the
conversion factors and `CHEMICAL_ACCURACY_MHA`.

### Lower level: `succd.selected_cc_pipeline`

The physics: `build_union_context` (everything `core.build` wraps),
`select_doubles_by_pt2`, `ci_corr` (selected CI by Slater–Condon rules),
`SelectedCCSD` (selected CCD/CCSD solver), `circuit_energy` and
`_propagate_ansatz` (sparse exponential propagation of the ansatz),
`cc_ansatz_params` (singles selection, eq 12). Read its module docstring.

## 4. Command-line scripts

Run from the repository root. Each stage reads `succd/geoms.py`, writes into
`data/results/` and skips what is already there (`FORCE=1` recomputes).

| Script | Arguments | Output | Article |
|---|---|---|---|
| `s1_selection.py` | systems (default: all six) | `selection.json` | Step 2 |
| `s2_system.py` | `SYSTEM [block ...]` | `res_<SYSTEM>.json` | Figs 5, 6, A1–A4 |
| `s3_sweep.py` | systems; `STEP=5` | `sweep.json` | Fig 4 |
| `s4_gates.py` | systems | `gates.json` | Fig 3, Table S5 |
| `s5_noise.py` | `[--singles] systems` | `noise.json` / `noise_singles.json` | Table S7, Fig S4 |
| `s6_stretch.py` | `N2 R1 R2 ...` or `H8 R1 R2 ...` (Å) | `stretch.json` | Table S1, Fig S1 |
| `s7_comp_check.py` | `SYSTEM` | `data/si/comp_<SYSTEM>.json` | Table S4, Fig S3 |
| `s8_fci_ref.py`, `s9_shot_sweep.py` | systems | `data/shot_sweep/*.json` | Table S8 |
| `export_values.py` | — | `data/values/*`, `data/si/si_tables.json` | all figures |
| `run_pipeline.sh` | — | everything, in order | — |
| `check_environment.py` | — | versions and import test | — |

Blocks of `s2_system.py`: `base` (FCI, c₀, s-CID, s-CCD), `D_c`, `D_cp`, `D_t`,
`Ddm` (doubles-only circuits and their SQD samples), `alpha` (ray optimum α*),
`vqe_c`, `vqe_t` (L-BFGS-B optimum p_m from c_m or t_m), `comp`
(collinearity of c and t), `quad` (explicit D+Q diagonalization), `fullcc`
(full CISD, CCSD, Davidson), `SD_c`, `SD_cp`, `SD_t`, `SDdm` (singles-inclusive
circuits). Without blocks, all are run.

`s6_stretch.py` knows two scans: `N2` (N–N distance) and any other tag, which
is treated as the uniform H₈ chain (H–H spacing).

## 5. Settings and environment variables

| Variable | Default | Effect |
|---|---|---|
| `SUCCD_DELTA` | `3e-4` | selection threshold δ in Hartree |
| `SUCCD_RESULTS` | `data/results` | where the stages write; point it to a scratch folder to keep the tracked data untouched |
| `SUCCD_VALUES`, `SUCCD_SI`, `SUCCD_SHOT_SWEEP`, `SUCCD_FIGURES` | `data/values`, `data/si`, `data/shot_sweep`, `figures/output` | the other folders |
| `SUCCD_ROOT` | the repository | where the data folders are |
| `FORCE` | unset | `FORCE=1` recomputes blocks already stored |
| `MAXIT` | 4000 | L-BFGS-B iterations for `vqe_c` / `vqe_t` |
| `GRID` | `1.00,1.05,...,1.25` | α grid for the ray optimum on untapered (20-qubit) systems |
| `STEP` | 5 | operator-count step of the Figure 4 sweep |

## 6. Data files

| Folder | Content |
|---|---|
| `data/results/res_<SYS>.json` | everything per system: `meta` (N, qubits, FCI, c₀, s-CID, s-CCD), `D_*` and `SD_*` (circuit energy `var`, SQD energies `sqd` at 1e6 and 1e8), `alpha`, `vqe_*`, `comp`, `quad`, `fullcc` |
| `data/results/selection.json` | sorted PT2 scores of every double, N at δ = 2e-4 and 3e-4 |
| `data/results/gates.json` | transpiled CX, total gates, depth for s-UCCD, UCCSD and s-UCCD + singles |
| `data/results/sweep.json` | ΔE′(c_m, d_m, t_m) at every operator count |
| `data/results/noise*.json`, `stretch.json` | noise condition, bond-stretching scans |
| `data/values/` | the tidy CSV/JSON each figure is drawn from, and `si_tables.json` |
| `data/si/` | component statistics of SI Section S2 |
| `data/shot_sweep/` | measurement-budget sweep and FCI references |

Energies are in kcal/mol relative to RHF, as in the code. `data/values` and
`data/si` are regenerated byte-for-byte from `data/results` by
`scripts/export_values.py`.

## 7. Figures

`python figures/make_figures.py` (data figures) and `python figures/schematics.py`
(TOC graphic, Figures 1 and 2) write PNG, PDF and TIFF into `figures/output/`,
at journal print width and 600 dpi (1200 dpi for line art), in mHa.
`python figures/make_figures.py --list` shows the names; pass names to draw a
subset. `python figures/package_tiffs.py` checks the TIFFs and zips them for
upload. Each figure function in `make_figures.py` is self-contained: colours
and sizes are at its top.

## 8. Cost and limits

* Qubits = 2 × (orbitals − frozen). The statevector is 2^qubits complex numbers:
  12–16 qubits take seconds, 20 qubits about one minute per circuit
  evaluation and ~16 MB per state; 24 qubits is the practical limit of this
  statevector code on a workstation.
* Z₂ tapering is used automatically when the qubit Hamiltonian fits in memory
  (`C['_tap']`); it makes energy evaluations much faster.
* The closed form of eq 6 is reliable while the reference dominates
  (c₀ ≳ 0.95; see SI Section S1.1). Below that, scan α with
  `minimize_scalar(lambda a: E(a * P['c'], N))`.
* The sign rule (eq 7) assumes a dominant RHF reference; check
  `P['mp2']` against the s-CCD signs for strongly correlated cases
  (`scripts/s7_comp_check.py` does this).
* Singles enter only through `params(C, N, True)` (Application 2).

## 9. Troubleshooting

| Symptom | Fix |
|---|---|
| `ImportError: cannot import name 'AbstractOdeFunction'` | import `succd` before `qiskit_nature` (it applies the qiskit-algorithms workaround) |
| `ModuleNotFoundError: succd` | run from the repository root, or `pip install -e .` |
| a notebook cannot find the data | open it from inside the repository; its first cell looks for `pyproject.toml` upwards |
| version differences | `python scripts/check_environment.py`; the tested versions are in `requirements.txt` |
| long progress messages (orbital lists, "JW Mapping" times) | printed by the Qiskit-Nature helper in `pyscf_problem_extended.py`; harmless |
| a stage does nothing | its results are already stored; use `FORCE=1` or another `SUCCD_RESULTS` folder |
| gate counts differ by a few units from `gates.json` | the transpiler is not bit-reproducible across machines; CX counts are stable |
