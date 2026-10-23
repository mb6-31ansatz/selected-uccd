# succd — a selected unitary coupled cluster ansatz with classically determined parameters

Code and data for

> M. Bovoloni, P. Macchi, A. Famulari, *A Selected Unitary Coupled Cluster Ansatz
> with Classically Determined Parameters*, J. Chem. Theory Comput. (submitted).

Repository: <https://github.com/mb6-31ansatz/selected-uccd> · archived release: DOI to be added after the Zenodo release (see `CITATION.cff`).

Every number, table and figure of the article and of its Supporting Information
can be regenerated from this repository: the raw results are tracked in `data/`,
the scripts that produced them are in `scripts/`, the figures are drawn by
`figures/`, and five notebooks walk through the method, verify every reported
quantity against the stored data, and run the pipeline end to end.

```
selected-uccd/
├── succd/                 the Python package (pipeline)
│   ├── geoms.py           the six MP2/def2-SVP geometries, frozen-core counts
│   ├── core.py            context, PT2 selection, the three parameter sets, circuit energy
│   ├── drv.py             SQD sampling, ray optimum, L-BFGS-B optimum, singles-inclusive sets
│   ├── selected_cc_pipeline.py   Slater–Condon CI, selected CCD/CCSD, sparse generators
│   ├── pyscf_problem_extended.py qubit Hamiltonian, Z2 tapering, excitation strings
│   ├── PySCFDriver_mod_vers.py   modified Qiskit Nature PySCF driver (IBM, Apache-2.0)
│   ├── paths.py           data folders (overridable with SUCCD_* variables)
│   └── units.py           Hartree / kcal mol⁻¹ / mHa
├── scripts/               the stages that produced data/, in order s1 … s9
├── figures/               make_figures.py (data figures), schematics.py (TOC, Figs 1–2), package_tiffs.py
├── notebooks/             1 figures · 2 tutorial · 3 verification · 4 pipeline · 5 SI
├── data/
│   ├── results/           raw stage outputs (res_<system>.json, gates, noise, selection, sweep, stretch)
│   ├── values/            the CSV/JSON every figure is drawn from, and the SI tables
│   ├── si/                component statistics of SI Section S2 (comp_<system>.json)
│   └── shot_sweep/        the measurement-budget sweep of SI Section S5
├── MANUAL.md            user manual: own molecules, API, scripts, data files, troubleshooting
├── requirements.txt       tested versions · pyproject.toml · LICENSE · NOTICE · CITATION.cff
└── CHANGELOG.md           what changed with respect to the bundle the article was assembled from
```

**New users:** read `MANUAL.md` (how to run your own molecule, every function and
script) and open `notebooks/2_tutorial_one_molecule.ipynb`.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # the tested versions (Python 3.13)
pip install -e .                         # optional: makes `import succd` work anywhere
python scripts/check_environment.py      # versions and import chain
```

Without `pip install -e .` everything still works: the scripts, the figure
makers and the first cell of every notebook put the repository root on
`sys.path` themselves.

`qiskit-algorithms 0.4.0` has a circular import that fires on some
installations the first time `qiskit-nature` pulls it in; `succd/__init__.py`
applies the workaround, so import `succd` (or anything in it) before
`qiskit_nature` in your own scripts.

## Reproducing the figures (minutes, no quantum-chemistry stack needed)

```bash
python figures/make_figures.py     # Figures 3–6, A1–A4, S1–S4  → figures/output/
python figures/schematics.py       # TOC graphic, Figure 1, Figure 2
python figures/package_tiffs.py    # checks dpi/size/mode and zips the TIFFs for upload
```

Only `numpy`, `pandas`, `matplotlib` and `pillow` are needed.  Each figure is
written as PNG and TIFF at 600 dpi (1200 dpi for the line-art Figure 1), RGB,
LZW-compressed, at the journal print width (7.0 in double column, 3.33 in
single column, 3.25 × 1.75 in for the TOC graphic), plus a vector PDF.  Stored
energies are in kcal mol⁻¹; the figures convert to mHa (1 kcal mol⁻¹ =
1.593601 mHa).  Figure 2 derives every bond length and angle from
`succd/geoms.py`.  Notebook 1 draws the same article figures cell by cell.

## Reproducing the numbers

| what | command | time |
|---|---|---|
| one system, every step, with explanations | `notebooks/2_tutorial_one_molecule.ipynb` | ~10 min (N₂) |
| every reported quantity re-derived from `data/`, with assertions | `notebooks/3_verification_and_geometries.ipynb` | seconds |
| the pipeline over a list of systems, into one table, then the figures | `notebooks/4_pipeline_all_systems.ipynb` | ~45 min for all six (stored); set `RUN = ['H6', 'N2']` for 2 min |
| the Supporting Information quantities | `notebooks/5_supporting_information_calcs.ipynb` | ~20 min for all six (stored); idem |
| everything, from the geometries to the TIFFs | `scripts/run_pipeline.sh` | hours |

The stages behind `data/` (each writes into `data/results/` and skips what it
already finds there; set `SUCCD_RESULTS=/scratch` to write elsewhere):

| stage | produces | article |
|---|---|---|
| `s1_selection.py` | PT2 scores, N(δ) | Step 2, Computational Details |
| `s2_system.py <sys> [block …]` | FCI, c₀, s-CID/s-CCD energies, circuit energies of c_m/c_m/c₀/d_m/t_m, SQD samples, ray optimum, L-BFGS-B optimum, D+Q space, full CISD/CCSD and Davidson | Figs 5, 6, A1–A4 |
| `s3_sweep.py` (`STEP=5`) | ΔE′ at every operator count | Fig 4 |
| `s4_gates.py` | transpiled gate counts, singles block | Fig 3, Table S5 |
| `s5_noise.py [--singles]` | SQD noise condition, eq S1 | Table S7, Fig S4 |
| `s6_stretch.py N2 1.10 …` | bond-stretching scans | Table S1, Fig S1 |
| `s7_comp_check.py` | angle and component ratios of c and t | Table S4, Fig S3 |
| `s8_fci_ref.py`, `s9_shot_sweep.py` | measurement-budget sweep | Table S8 |
| `export_values.py` | `data/values/`, `data/si/` from the above | all figures |

Both `data/values/` and `data/si/` are regenerated byte-for-byte by
`export_values.py` from `data/results/`.

### Geometries

The six structures are in `succd/geoms.py` under the keys `H6`, `H8`, `N2`,
`CO`, `CH2O`, `C2H2` (the article labels).  They are editable and they drive
the calculation: notebooks 2, 4 and 5 open with a `GEOMETRIES` cell that
overrides `geoms.SYSTEMS`, so moving an atom or stretching a bond there is all
that is needed to run on a new structure.

### Units and conventions

* Energies in `data/` are correlation energies in kcal mol⁻¹ relative to
  RHF, unless a key says otherwise (`succd/units.py` has the factors).
* The selection threshold δ is 3 × 10⁻⁴ Ha (0.3 mHa); `SUCCD_DELTA` overrides it.
* STO-6G, RHF reference, frozen core (`geoms.py` gives the count), Jordan–Wigner
  mapping, one Trotter step, Z₂ tapering where it applies (H₆, H₈, N₂, CO).
* SQD samples at 10⁶ and 10⁸ measurements with `SEED = 2024` (`succd/drv.py`);
  the shot sweep of SI Section S5 draws its own fixed set per system.

## Licence

Apache License 2.0 (`LICENSE`).  `succd/PySCFDriver_mod_vers.py` is a modified
copy of the Qiskit Nature PySCF driver, © IBM 2018, 2023, redistributed under
the same licence with its header preserved; see `NOTICE`.

## Citing

`CITATION.cff` carries the metadata; fill in the article DOI on acceptance and
the Zenodo DOI once a release is archived.
