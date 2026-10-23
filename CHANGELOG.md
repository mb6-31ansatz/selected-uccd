# Changelog

## 1.0.0 — reorganised for publication (October 2026)

Starting point: the flat `QC1_compact` bundle (every module, notebook and data
folder at the root).  Nothing in the physics was changed; the reorganisation is
structural, and `scripts/export_values.py` regenerates `data/values/` and
`data/si/` byte-for-byte from the stored `data/results/`.

### Layout
* Pipeline modules moved into the package `succd/` with relative imports
  (`from succd import core, drv`, `from succd import selected_cc_pipeline as scp`).
  New `succd/__init__.py` (applies the qiskit-algorithms import workaround once),
  `succd/paths.py` (data folders, env-overridable) and `succd/units.py`.
* Stage scripts moved to `scripts/` and numbered in dependency order
  (`comp_check.py` → `s7_comp_check.py`, `shot_sweep/fci_ref.py` → `s8_fci_ref.py`,
  `shot_sweep/shot_sweep.py` → `s9_shot_sweep.py`).  `scripts/run_pipeline.sh`
  runs them all.
* Data folders grouped under `data/`: `out/` → `data/results/`, `values/` →
  `data/values/`, `data/` → `data/si/`, `shot_sweep/*.json` → `data/shot_sweep/`.
  The duplicate `data/stretch.json` (identical to `out/stretch.json`) was dropped.
* Notebooks moved to `notebooks/`; their first cell finds the repository root
  (the folder holding `pyproject.toml`) and all paths go through `succd.paths`.
* Added `LICENSE` (Apache-2.0 text), `CITATION.cff`, `pyproject.toml`; filled
  the `[YEAR] [AUTHORS]` placeholders in every header and in `NOTICE`.
* `patch_tables.py` (rewrites tables inside the manuscript .docx) and
  `si_figures.py` (an earlier SI figure set in kcal/mol) are not part of the
  reproduction and were left out.

### Fixes
* Repository URL is now https://github.com/mb6-31ansatz/selected-uccd (the old
  name ended in a hyphen); updated in CITATION.cff, pyproject.toml and README.md.
* The acetylene system was keyed `C2H4` / `Ethylene` in `geoms.py`, in the
  result files (`res_C2H4.json`) and in JSON keys, while the article calls it
  C₂H₂.  Everything now uses `C2H2` (`geoms.Acetylene`); `res_C2H4.json` →
  `res_C2H2.json`.  The geometry itself is unchanged (two C, two H).
* The stage scripts defaulted to absolute paths of the machine the bundle was
  made on; they now resolve through
  `succd.paths`.
* `drv.save` wrote `res_<system>.json` into the current directory; it now
  writes into `data/results/`.
* `s7_comp_check.py` writes `comp_<system>.json` into `data/si/` instead of the
  current directory.

### Additions needed to redraw the submitted figures from the deposited data
* `s2_system.py fullcc` now also stores the CISD reference weight c₀ (with the
  spin-adapted CISD norm of PySCF) and the Davidson correction (1 − c₀²)E_CISD;
  `export_values.figA1` adds the `CISD+Davidson` row that Figure A1 plots.  The
  previous bundle plotted CCSD in that figure.
* `s5_noise.py --singles` evaluates the noise condition on the singles-inclusive
  d_m circuit, which is the state SI Table S7 and Figure S4 report (the bundle
  only had the doubles-only condition, quoted in the SI text for comparison).
  `export_values` writes it as `table9` and keeps the doubles-only rows as
  `table9_doubles_only`.  Output: `data/results/noise_singles.json`.
* `export_values.figA4` writes `figA4_circuit_vs_sqd.csv` for Figure A4, which
  had no data file in the bundle.
* `data/results/sweep.json` holds the Figure 4 sweep at step 5 for all six
  systems (the bundle had step 25, i.e. three points, for CH₂O and C₂H₂).
* `figures/make_figures.py` and `figures/schematics.py` replace the kcal/mol
  figure cells and `figA1_enhanced_cisd.py` as the reference figure set: mHa,
  journal print size, TIFF/PNG/PDF.

### Stored notebook outputs
All five notebooks were executed on this layout; 4 and 5 over all six systems
(`RUN = ALL`, plus the H₈ stretching scan in 5).  Their outputs reproduce the
stored results (gate totals differ by a few units from `gates.json` because the
transpiler is not bit-reproducible across machines).

### Numbering note
The notebooks were written against an earlier draft of the Supporting
Information.  Sections 2–6 of that draft are S1–S5 of the submitted SI, tables
2–11 are S1–S8 after removal of two, figures S5 and S6 were removed and the old
S5 became S4.  Notebook headings carry the current numbering where they were
updated; a cell headed "Table 9" refers to Table S7.
