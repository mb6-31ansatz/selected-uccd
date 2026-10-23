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

"""
SQD shot-budget sweep — is the SQD error at 1e8 shots a sampling limit or an
ansatz limit?

For each system we rebuild the singles-inclusive d_m state (the state of
Figure A2), then diagonalise the Hamiltonian exactly in the subspace of
determinants observed at a sequence of shot budgets, and finally in the whole
support of the state (the S -> infinity ceiling).

Design: one set of uniform draws u_i is fixed per system, and a determinant is
observed at budget S when u_i < 1-(1-w_i)^S.  Since that threshold increases
with S, the subspaces are strictly nested and the comparison across budgets is
controlled rather than three independent experiments.

Reading:
  ceiling ~ value at 1e8   ->  saturated; the state cannot reach what FCI needs
                              (an ansatz limit, no shot budget helps)
  ceiling << value at 1e8  ->  shot-limited; more measurements help

Output: data/shot_sweep/shot_sweep.json
"""
import json
import os
import sys
import time
import warnings

warnings.filterwarnings('ignore')
import numpy as np
from pyscf.fci import direct_spin1, cistring
from scipy.sparse.linalg import LinearOperator, eigsh

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core, drv
from succd import selected_cc_pipeline as scp

K = core.K
SEED = 2024
SHOTS = [1e6, 1e8, 1e9, 1e10, 1e11]
OUT = paths.shot_sweep('shot_sweep.json')


def subspace_energy(addrs, h2e, M, na, nb, nstr, shift):
    """Exact ground-state energy in the given determinant subspace, kcal/mol."""
    addrs = np.unique(addrs)
    if len(addrs) == 1:
        v = np.zeros(nstr * nstr)
        v[addrs] = 1.0
        lam = float(direct_spin1.contract_2e(
            h2e, v.reshape(nstr, nstr), M, (na, nb)).ravel()[addrs][0])
    else:
        def mv(x):
            v = np.zeros(nstr * nstr)
            v[addrs] = x
            return direct_spin1.contract_2e(
                h2e, v.reshape(nstr, nstr), M, (na, nb)).ravel()[addrs]
        lam = eigsh(LinearOperator((len(addrs),) * 2, matvec=mv, dtype=float),
                    k=1, which='SA', tol=1e-10, maxiter=40000)[0][0]
    return float((lam + shift) * K)


def run(name):
    t0 = time.time()
    C = core.build(name)
    n = core.nsel(C)

    # the singles-inclusive d_m state, exactly as Figure A2 prepares it
    sets, take, info = drv._sd_sets(C, n)
    c0 = info['c0']
    a = (2 - c0 ** 2) / c0
    psi = core.state(C, a * sets['c'], take)

    M, na, nb = C['M'], C['na'], C['nb']
    nstr = cistring.num_strings(M, na)
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, (na, nb), .5)
    HF = (C['HFdet'][0], C['HFdet'][1])

    w = np.abs(np.asarray(psi).ravel()) ** 2
    w = w / w.sum()

    def addr_of(i):
        Ia = int(i) & ((1 << M) - 1)
        Ib = (int(i) >> M) & ((1 << M) - 1)
        if bin(Ia).count('1') != na or bin(Ib).count('1') != nb:
            return None
        return cistring.str2addr(M, na, Ia) * nstr + cistring.str2addr(M, nb, Ib)

    hf_addr = addr_of((HF[1] << M) | HF[0])

    # one fixed set of uniform draws -> nested subspaces across budgets
    rng = np.random.default_rng(SEED)
    u = rng.random(w.size)

    rec = {'system': name, 'n': int(n), 'n_singles': int(info['ns']),
           'c0': c0, 'factor': float(a), 'budgets': {}}

    for S in SHOTS:
        pseen = -np.expm1(int(S) * np.log1p(-np.clip(w, 0.0, 1.0 - 1e-18)))
        sel = np.where(u < pseen)[0]
        addrs = [q for q in (addr_of(i) for i in sel) if q is not None]
        if hf_addr not in addrs:
            addrs.append(hf_addr)
        E = subspace_energy(np.array(addrs), h2e, M, na, nb, nstr, C['shift'])
        rec['budgets'][f'{S:.0e}'] = {'n_dets': int(len(np.unique(addrs))),
                                      'E_corr_kcalmol': E}
        print(f'  {name:6} S={S:.0e}  dets={len(np.unique(addrs)):6d}  '
              f'E_corr={E:11.4f} kcal/mol', flush=True)

    # ceiling: the whole support of the prepared state
    idx = np.where(w > 1e-14)[0]
    addrs = [q for q in (addr_of(i) for i in idx) if q is not None]
    if hf_addr not in addrs:
        addrs.append(hf_addr)
    E_ceiling = subspace_energy(np.array(addrs), h2e, M, na, nb, nstr, C['shift'])
    rec['ceiling'] = {'n_dets': int(len(np.unique(addrs))),
                      'E_corr_kcalmol': E_ceiling}
    print(f'  {name:6} S=inf     dets={len(np.unique(addrs)):6d}  '
          f'E_corr={E_ceiling:11.4f} kcal/mol   ({time.time()-t0:.0f}s)', flush=True)

    A = json.load(open(OUT)) if os.path.exists(OUT) else {}
    A[name] = rec
    json.dump(A, open(OUT, 'w'), indent=1)
    return rec


if __name__ == '__main__':
    paths.ensure(paths.SHOT_SWEEP)
    for nm in sys.argv[1:]:
        run(nm)
