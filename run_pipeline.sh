#!/usr/bin/env bash
# Regenerate every stored result from the geometries, in the order the stages
# depend on each other.  Run from the repository root with the environment of
# requirements.txt.  The whole set takes a few hours; the two 20-qubit systems
# (CH2O, C2H2) dominate.  Each stage skips what it already finds in
# data/results/, so the script can be interrupted and resumed.  Set
# SUCCD_RESULTS=/some/scratch to write somewhere else than the tracked data.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
ALL="H6 H8 N2 CO CH2O C2H2"

$PY scripts/check_environment.py
$PY scripts/s1_selection.py $ALL                     # selection scores, N(delta)
for m in $ALL; do $PY scripts/s2_system.py $m; done   # every per-system block
STEP=5 $PY scripts/s3_sweep.py $ALL                  # Figure 4
$PY scripts/s4_gates.py $ALL                         # Figure 3, Table S5
$PY scripts/s5_noise.py $ALL                         # doubles-only noise condition
$PY scripts/s5_noise.py --singles $ALL               # Table S7 / Figure S4
$PY scripts/s6_stretch.py N2 1.10 1.30 1.50 1.70 1.90 2.10 2.40      # Table S1 / Figure S1
$PY scripts/s6_stretch.py H8 0.90 1.00 1.10 1.20 1.30 1.40
for m in $ALL; do $PY scripts/s7_comp_check.py $m; done   # Tables S4 / Figure S3
$PY scripts/s8_fci_ref.py $ALL                       # FCI references for the shot sweep
$PY scripts/s9_shot_sweep.py $ALL                    # Table S8
$PY scripts/export_values.py                         # data/values/, data/si/
$PY figures/make_figures.py                          # article + SI data figures
$PY figures/schematics.py                            # TOC, Figure 1, Figure 2
