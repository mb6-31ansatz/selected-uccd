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

"""Stage 5 - the SQD noise condition of eq S1 (SI Table S7, Figure S4).

Two states can be assessed:

    python scripts/s5_noise.py H6 N2 ...            doubles-only d_m circuit (Step 2)
    python scripts/s5_noise.py --singles H6 N2 ...  singles-inclusive d_m circuit, the
                                                    state recommended by Application 2

The Supporting Information (Table S7, Figure S4) reports the singles-inclusive
condition; the doubles-only numbers are quoted in its text for comparison.  The
outputs are data/results/noise.json and data/results/noise_singles.json.

The determinants the ideal d_m state populates are ordered by decreasing weight and the
smallest leading set whose exact diagonalisation recovers 95 % of the converged
correlation energy of the whole populated space is located by bisection.  w_min is the
weight of the last member of that set.  D is the dimension of the particle-number- and
spin-resolved sector.  A configuration is visible above the depolarised background when
p w_i >= (1-p)/D, so the required survival probability is 1/(1 + D w_min) and the
admitted two-qubit gate count is the integer part of ln(required p)/ln f.

"""
import json, math, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core, drv
from math import comb
from pyscf.fci import direct_spin1, cistring
from scipy.sparse.linalg import LinearOperator, eigsh

K = core.K
OUT = paths.RESULTS
paths.ensure(OUT)
FID = 0.9977                      # median two-qubit fidelity of ref 54


def run(name, singles=False):
    t0 = time.time()
    F = os.path.join(OUT, 'noise_singles.json' if singles else 'noise.json')
    C = core.build(name)
    n = core.nsel(C)
    if singles:
        # the singles-inclusive d_m state, exactly as Application 2 prepares it
        sets, take, info = drv._sd_sets(C, n)
        c0 = info['c0']
        a = (2 - c0 ** 2) / c0
        psi = core.state(C, a * sets['c'], take)
        ns = int(info['ns'])
    else:
        P = core.params(C, n, False)
        a = (2 - P['c0'] ** 2) / P['c0']
        psi = core.state(C, a * P['c'])
        ns = 0
    w = np.abs(psi) ** 2
    w = w / w.sum()
    M, na, nb = C['M'], C['na'], C['nb']
    nstr = cistring.num_strings(M, na)
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, (na, nb), .5)
    idx = [i for i in np.where(w > 1e-14)[0]
           if bin(int(i) & ((1 << M) - 1)).count('1') == na
           and bin((int(i) >> M) & ((1 << M) - 1)).count('1') == nb]
    order = sorted(idx, key=lambda i: -w[i])

    def energy(sub):
        addrs = np.unique([cistring.str2addr(M, na, int(i) & ((1 << M) - 1)) * nstr +
                           cistring.str2addr(M, nb, (int(i) >> M) & ((1 << M) - 1))
                           for i in sub])
        def mv(x):
            v = np.zeros(nstr * nstr)
            v[addrs] = x
            return direct_spin1.contract_2e(h2e, v.reshape(nstr, nstr),
                                            M, (na, nb)).ravel()[addrs]
        lam = eigsh(LinearOperator((len(addrs),) * 2, matvec=mv, dtype=float), k=1,
                    which='SA', tol=1e-10, maxiter=40000)[0][0]
        return float((lam + C['shift']) * K)

    Efull = energy(order)
    target = 0.95 * Efull
    lo, hi = 1, len(order)
    while lo < hi:
        mid = (lo + hi) // 2
        if energy(order[:mid]) <= target:
            hi = mid
        else:
            lo = mid + 1
    k = lo
    wmin = float(w[order[k - 1]])
    D = comb(M, na) * comb(M, nb)
    Dw = D * wmin
    p_req = 1.0 / (1.0 + Dw)
    ncx = math.log(p_req) / math.log(FID)
    rec = dict(system=name, n=int(n), n_singles=ns, singles_inclusive=bool(singles),
               dets_95=int(k), populated=int(len(order)),
               D=int(D), w_min=wmin, D_wmin=float(Dw), p_required=float(p_req),
               N_CX_admitted=int(math.floor(ncx)), N_CX_exact=float(ncx),
               E_full_sampled=Efull, E_95=energy(order[:k]), f=FID)
    A = json.load(open(F)) if os.path.exists(F) else {}
    A[name] = rec
    json.dump(A, open(F, 'w'), indent=1)
    print('%-6s dets95=%d populated=%d D=%d w_min=%.3g D*w=%.4g p=%.4f N_CX=%d (%.0fs)'
          % (name, k, len(order), D, wmin, Dw, p_req, rec['N_CX_admitted'],
             time.time() - t0), flush=True)


if __name__ == '__main__':
    args = sys.argv[1:]
    singles = '--singles' in args
    for nm in [a for a in args if not a.startswith('--')]:
        run(nm, singles=singles)
