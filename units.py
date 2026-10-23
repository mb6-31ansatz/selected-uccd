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

"""Energy units.

The pipeline and every stored file work in kcal mol^-1 (columns and keys are
suffixed ``_kcalmol`` or documented as such).  The article reports energies in
mHa; the figure scripts convert at plotting time with ``KCALMOL_TO_MHA``.
The selection threshold delta is a Hartree quantity (3e-4 Ha = 0.3 mHa).
"""

HARTREE_TO_KCALMOL = 627.5094740631          # CODATA 2018
KCALMOL_TO_HARTREE = 1.0 / HARTREE_TO_KCALMOL
KCALMOL_TO_MHA = 1000.0 / HARTREE_TO_KCALMOL  # 1 kcal/mol = 1.593601 mHa
MHA_TO_KCALMOL = 1.0 / KCALMOL_TO_MHA
CHEMICAL_ACCURACY_KCALMOL = 1.0
CHEMICAL_ACCURACY_MHA = CHEMICAL_ACCURACY_KCALMOL * KCALMOL_TO_MHA   # 1.594 mHa


def kcalmol_to_mha(x):
    """Convert kcal mol^-1 to mHa (works on floats and numpy arrays)."""
    return x * KCALMOL_TO_MHA


def mha_to_kcalmol(x):
    """Convert mHa to kcal mol^-1."""
    return x * MHA_TO_KCALMOL
