# succd — Selected Unitary Coupled Cluster with Classically Determined Parameters

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
![Tested on Python 3.13](https://img.shields.io/badge/tested%20on-Python%203.13-blue.svg)
<!-- After the Zenodo release, replace XXXXXXX below and remove this comment:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
-->

Code, data and notebooks for the article

> **A Selected Unitary Coupled Cluster Ansatz with Classically Determined Parameters**
> Michelangelo Bovoloni, Piero Macchi, Antonino Famulari
> *Journal of Chemical Theory and Computation* (submitted, 2026)

Every number, table and figure of the article and of its Supporting Information
can be regenerated from this repository.

---

## About

Unitary coupled cluster (UCC) circuits are usually built either from the full
singles-and-doubles pool, which is far too deep, or by adaptive loops that choose
operators and angles from repeated noisy measurements. **s-UCCD** takes a third
route: everything is decided classically, before the circuit is run.

1. **Selection.** A perturbative estimate ranks every double excitation; only those
   above a threshold δ = 0.3 mHa enter the circuit.
2. **Parameters.** One diagonalization of the selected CI problem gives the
   coefficients c<sub>m</sub>; a closed-form rescaling,
   d<sub>m</sub> = (2 − c<sub>0</sub>²)/c<sub>0</sub> · c<sub>m</sub>,
   turns them into circuit angles. The signs come from first-order perturbation theory.
3. **No optimization on the device.** The circuit is fixed, and its energy is
   known before execution.

On six small molecules (H₆, H₈, N₂, CO, CH₂O, C₂H₂; STO-6G, 12–20 qubits):

- **3–12× fewer two-qubit (CX) gates** than full UCCSD in the same active space;
- the circuit lies **within 0.3 mHa of its own variational optimum**, with
  coupled-cluster-quality parameters from a linear eigenvalue problem;
- the excess of the circuit energy over the selected CI energy measures the
  **quadruple-excitation correlation energy**, which gives a size-extensivity
  correction to CISD (Enhanced CISD);
- with the selected singles appended, the circuit is a compact
  **state-preparation front end for sample-based quantum diagonalization (SQD)**.

All quantum circuits are simulated exactly (statevector, no hardware noise).

## Quick start

```bash
git clone https://github.com/mb6-31ansatz/selected-uccd.git
cd selected-uccd
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                       # tested versions, Python 3.13
pip install -e .                                      # makes `import succd` work from any folder
python scripts/check_environment.py                   # checks versions and imports
```

Then open the step-by-step tutorial, `notebooks/2_tutorial_one_molecule.ipynb`
(N₂, about 5 minutes), or run the method on your own molecule:

```python
from succd import geoms, core

geoms.SYSTEMS['H2O'] = ("""O  0.0000  0.0000  0.1173
H  0.0000  0.7572 -0.4692
H  0.0000 -0.7572 -0.4692""", 1)          # xyz in Å, number of frozen core orbitals

C = core.build('H2O')                    # RHF, integrals, qubit Hamiltonian, PT2 ranking
N = core.nsel(C)                         # selected doubles
P = core.params(C, N, False)             # s-CID coefficients, s-CCD amplitudes, c0
d = (2 - P['c0']**2) / P['c0'] * P['c']  # closed-form circuit parameters
print(N, core.evaluator(C)(d, N))        # 21 doubles, -31.03 kcal/mol (correlation energy)
```

**[`MANUAL.md`](MANUAL.md)** explains every function, script, setting and data file.

## Repository contents

| Folder / file | Content |
|---|---|
| [`succd/`](succd) | the Python package: geometries, selection, parameters, circuit energies, SQD |
| [`notebooks/`](notebooks) | 1 figures · **2 tutorial** · 3 verification of every reported number · 4 full pipeline · 5 Supporting Information |
| [`scripts/`](scripts) | the stages `s1`–`s9` that produced `data/`, `export_values.py`, `run_pipeline.sh` |
| [`data/`](data) | stored results (`results/`), the values each figure is drawn from (`values/`), SI statistics (`si/`, `shot_sweep/`) |
| [`figures/`](figures) | scripts that draw every figure of the article and the SI (PNG, PDF, 600 dpi TIFF) |
| [`MANUAL.md`](MANUAL.md) · [`CHANGELOG.md`](CHANGELOG.md) | user manual · history of the code |
| `requirements.txt` · `pyproject.toml` · `CITATION.cff` · `LICENSE` · `NOTICE` | environment, packaging, citation, licence |

## Reproducing the article

| Goal | How | Time |
|---|---|---|
| redraw all figures from the stored data | `python figures/make_figures.py` and `python figures/schematics.py` | ~1 min |
| check every number quoted in the article | `notebooks/3_verification_and_geometries.ipynb` | seconds |
| recompute one molecule from its geometry | `notebooks/2_tutorial_one_molecule.ipynb` | ~5 min |
| recompute all six molecules / the SI quantities | `notebooks/4_pipeline_all_systems.ipynb`, `notebooks/5_supporting_information_calcs.ipynb` | ~45 / ~20 min |
| recompute everything from scratch | `bash scripts/run_pipeline.sh` | a few hours |

The figure scripts need only `numpy`, `pandas`, `matplotlib` and `pillow`; the
calculations need the full stack of `requirements.txt` (PySCF 2.14, Qiskit 2.5,
Qiskit Nature 0.8, Qiskit Aer 0.17, Qiskit Algorithms 0.4).

Conventions: STO-6G basis, RHF reference, frozen core, Jordan–Wigner mapping,
one Trotter step, Z₂ tapering where it applies. Stored energies are correlation
energies in kcal/mol relative to RHF; the figures convert to mHa.

## How to cite

If you use this code or data, please cite the article and the archived software:

```bibtex
@article{Bovoloni2026sUCCD,
  author  = {Bovoloni, Michelangelo and Macchi, Piero and Famulari, Antonino},
  title   = {A Selected Unitary Coupled Cluster Ansatz with Classically Determined Parameters},
  journal = {Journal of Chemical Theory and Computation},
  year    = {2026},
  note    = {submitted}
}

@software{Bovoloni2026succd,
  author    = {Bovoloni, Michelangelo and Macchi, Piero and Famulari, Antonino},
  title     = {succd: selected unitary coupled cluster doubles with classically determined parameters},
  version   = {1.0.0},
  year      = {2026},
  publisher = {Zenodo},
  url       = {https://github.com/mb6-31ansatz/selected-uccd}
}
```

GitHub's **"Cite this repository"** button (right column) gives the same
information from [`CITATION.cff`](CITATION.cff).

## Licence

[Apache License 2.0](LICENSE). `succd/PySCFDriver_mod_vers.py` is a modified copy
of the Qiskit Nature PySCF driver (© IBM 2018, 2023), redistributed under the same
licence with its header preserved; see [`NOTICE`](NOTICE).

## Authors and contact

Department of Chemistry, Materials and Chemical Engineering "G. Natta",
Politecnico di Milano, Italy

- Michelangelo Bovoloni: method, code and calculations
- Piero Macchi: supervision
- Antonino Famulari: supervision; corresponding author, antonino.famulari@polimi.it

Questions and bug reports: please open an
[issue](https://github.com/mb6-31ansatz/selected-uccd/issues).
