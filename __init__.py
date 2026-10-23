# Copyright 2026 Michelangelo Bovoloni, Piero Macchi, Antonino Famulari
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""succd - selected unitary coupled cluster doubles with classically determined
parameters.

Companion code of

    M. Bovoloni, P. Macchi, A. Famulari,
    "A Selected Unitary Coupled Cluster Ansatz with Classically Determined
    Parameters", J. Chem. Theory Comput. (submitted).

Modules
-------
geoms                     the six MP2/def2-SVP geometries and the frozen-core counts
core                      build the context, select the doubles, the three parameter
                          sets (c_m, c_m/c0, t_m), circuit energy and statevector
drv                       stage helpers: SQD sampling, ray optimum, L-BFGS-B optimum
selected_cc_pipeline      the physics: Slater-Condon CI, selected CCD/CCSD, PT2
                          selection, sparse generators, circuit propagation
pyscf_problem_extended    qubit Hamiltonian, Z2 tapering, JW/tapered excitation strings
PySCFDriver_mod_vers      modified Qiskit Nature PySCF driver (IBM, Apache-2.0)
paths                     the repository data folders
units                     Hartree / kcal mol^-1 / mHa conversion factors

Importing the light modules (``geoms``, ``paths``, ``units``) does not pull in
PySCF or Qiskit; ``core`` and ``drv`` do.
"""

__version__ = '1.0.0'

# qiskit-algorithms 0.4.0 contains a circular import that fires on some installs
# the first time qiskit-nature pulls it in.  Importing the solvers subpackage
# first makes solvers/__init__ run before abstract_ode_function, which resolves
# the cycle.  It is a no-op where the bug does not occur.
try:  # pragma: no cover
    import qiskit_algorithms.time_evolvers.variational.solvers  # noqa: F401
except Exception:  # noqa: BLE001
    pass

from . import paths, units  # noqa: E402,F401
from .geoms import SYSTEMS  # noqa: E402,F401

__all__ = ['SYSTEMS', 'paths', 'units', '__version__']
