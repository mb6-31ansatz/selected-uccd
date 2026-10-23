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

"""Stage 6 - bond-stretching scans (SI Table 2, Figure S1).

At every geometry the whole procedure is repeated from scratch: perturbative selection at
the current delta, s-CID diagonalisation giving a new c0 and hence a new alpha_est, s-CCD
solution, and a Brent minimisation of the circuit energy along the ray to locate alpha*.
The prediction is therefore always scored against the optimum it is meant to reproduce at
that geometry.

Geometry conventions are those of the shipped stretch script: N2 is N 0 0 0 / N 0 0 R, and
H8 is the uniform linear chain H 0 0 i*R for i = 0..7, so R is the spacing.

Output: out/stretch.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core
from succd import selected_cc_pipeline as scp
from scipy.optimize import minimize_scalar

K = core.K
OUT = paths.RESULTS
paths.ensure(OUT)
F = os.path.join(OUT, 'stretch.json')


def one(tag, xyz, nfz, dist):
    C = scp.build_union_context(xyz, basis='sto-6g', n_frozen=nfz, verbose=False)
    C['_tap'] = (C.get('Hmat') is not None and C.get('gens_tap') is not None)
    n = core.nsel(C)
    Ev = core.evaluator(C)
    dbl = [tuple(map(int, x)) for x in C['dbl'][:n]]
    tmp = C['dfr']['t_mp'].values[:n].astype(float)
    sg = np.sign(tmp)
    E_cid, vec = scp.ci_corr([C['HFdet']] + C['double_dets'][:n], C['h1'], C['eri'])
    c0 = abs(vec[0])
    cm = np.abs(vec[1:1 + n])
    Ecc, _, t2 = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                  selected_singles=[], selected_doubles=dbl,
                                  so_labels=C['lbl']).kernel()
    tm = np.abs(np.array([t2[i, j, a, b] for (i, j, a, b) in dbl]))
    r = minimize_scalar(lambda a: Ev(a * cm * sg, n), bracket=(1.0, 1.2),
                        method='brent', options={'xtol': 1e-6})
    astar, Ea = float(r.x), float(r.fun)
    a_cl = (2 - c0 ** 2) / c0
    a_mp = float(np.abs(tmp).sum() / cm.sum()) / c0
    d = dict(dist=float(dist), n=int(n), c0=float(c0), astar=astar, E_astar=Ea,
             a_closed=float(a_cl), a_mp=float(a_mp),
             E_cm=float(Ev(cm * sg, n)), E_cp=float(Ev(cm / c0 * sg, n)),
             E_closed=float(Ev(a_cl * cm * sg, n)), E_mp=float(Ev(a_mp * cm * sg, n)),
             E_tm=float(Ev(tm * sg, n)), E_sCCD=float(Ecc * K), E_sCID=float(E_cid * K),
             norm_ratio=float(np.linalg.norm(tm) / np.linalg.norm(cm)))
    d['penalty'] = d['E_closed'] - Ea
    gap = Ea - d['E_cm']
    for k, e in (('f_cp', 'E_cp'), ('f_closed', 'E_closed'), ('f_mp', 'E_mp')):
        d[k] = (d[e] - d['E_cm']) / gap if abs(gap) > 1e-9 else float('nan')
    A = json.load(open(F)) if os.path.exists(F) else {}
    A.setdefault(tag, [])
    A[tag] = [x for x in A[tag] if abs(x['dist'] - dist) > 1e-9] + [d]
    A[tag].sort(key=lambda x: x['dist'])
    json.dump(A, open(F, 'w'), indent=1)
    print('%-4s R=%5.2f n=%3d c0=%.4f a*=%.4f a_cl=%.4f penalty=%8.3f'
          % (tag, dist, n, c0, astar, a_cl, d['penalty']), flush=True)


if __name__ == '__main__':
    tag = sys.argv[1]
    A = json.load(open(F)) if os.path.exists(F) else {}
    have = {round(x['dist'], 4) for x in A.get(tag, [])}
    for dv in [float(x) for x in sys.argv[2:]]:
        if round(dv, 4) in have:
            print('%s R=%.2f already present' % (tag, dv), flush=True)
            continue
        t0 = time.time()
        if tag == 'N2':
            one(tag, 'N 0 0 0; N 0 0 %.6f' % dv, 2, dv)
        else:
            one(tag, '; '.join('H 0 0 %.6f' % (i * dv) for i in range(8)), 0, dv)
        print('   [%s R=%.2f done in %.0fs]' % (tag, dv, time.time() - t0), flush=True)
