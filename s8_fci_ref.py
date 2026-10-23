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

"""Exact FCI correlation energy in the frozen-core active space, by direct
diagonalisation of the full particle-number- and spin-resolved sector.

Used to turn the correlation energies of shot_sweep.py into deviations from FCI,
which is what Figure A2 reports.

Output: data/shot_sweep/fci_ref.json
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from pyscf.fci import direct_spin1

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core

K = core.K
OUT = paths.shot_sweep('fci_ref.json')


def run(name):
    t0 = time.time()
    C = core.build(name)
    M, na, nb = C['M'], C['na'], C['nb']
    e, _ = direct_spin1.kernel(C['h1'], C['eri'], M, (na, nb),
                               max_space=30, max_cycle=400, conv_tol=1e-12)
    E = float((e + C['shift']) * K)
    A = json.load(open(OUT)) if os.path.exists(OUT) else {}
    A[name] = {'E_FCI_corr_kcalmol': E, 'M': int(M), 'na': int(na), 'nb': int(nb)}
    json.dump(A, open(OUT, 'w'), indent=1)
    print(f'{name:6} E_FCI_corr = {E:11.4f} kcal/mol  ({time.time() - t0:.0f}s)',
          flush=True)


if __name__ == '__main__':
    paths.ensure(paths.SHOT_SWEEP)
    for nm in sys.argv[1:]:
        run(nm)
