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

"""Stage 3 - the sweep behind Figure 4.

At every operator count N the whole construction is repeated from scratch: the s-CID
diagonalisation over HF + the first N selected doubles supplies a new c0 and hence a new
alpha_est, and the s-CCD equations are re-solved over the same N doubles.  Nothing is
carried over between sizes, which is what the Computational Details section asserts.

  dE'(x_m) = E_sCCD(t_m) - E_sUCCD(x_m),    x_m in {c_m, d_m, t_m}

Output: out/sweep.json, one record per (system, N).
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core
from succd import selected_cc_pipeline as scp

K = core.K
OUT = paths.RESULTS
paths.ensure(OUT)
F = os.path.join(OUT, 'sweep.json')


def grid(nmax, step=5):
    g = list(range(step, nmax + 1, step))
    if g and g[-1] != nmax:
        g.append(nmax)
    return g


def run(name, step=5):
    C = core.build(name)
    n = core.nsel(C)
    Ev = core.evaluator(C)
    A = json.load(open(F)) if os.path.exists(F) else {}
    A.setdefault(name, [])
    done = {r['N'] for r in A[name]}
    for N in grid(n, step):
        if N in done:
            continue
        t0 = time.time()
        dbl = [tuple(map(int, x)) for x in C['dbl'][:N]]
        sg = np.sign(C['dfr']['t_mp'].values[:N]).astype(float)
        E_cid, vec = scp.ci_corr([C['HFdet']] + C['double_dets'][:N], C['h1'], C['eri'])
        c0 = abs(vec[0])
        cm = np.abs(vec[1:1 + N])
        a = (2 - c0 ** 2) / c0
        E_ccd, _, t2 = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                        selected_singles=[], selected_doubles=dbl,
                                        so_labels=C['lbl']).kernel()
        tm = np.abs(np.array([t2[i, j, aa, bb] for (i, j, aa, bb) in dbl]))
        E_c = float(Ev(cm * sg, N))
        E_d = float(Ev(a * cm * sg, N))
        E_t = float(Ev(tm * sg, N))
        Ecc = float(E_ccd * K)
        A[name].append(dict(N=int(N), c0=float(c0), alpha_est=float(a),
                            E_sCID=float(E_cid * K), E_sCCD=Ecc,
                            E_cm=E_c, E_dm=E_d, E_tm=E_t,
                            dE_cm=Ecc - E_c, dE_dm=Ecc - E_d, dE_tm=Ecc - E_t))
        A[name].sort(key=lambda r: r['N'])
        json.dump(A, open(F, 'w'), indent=1)
        print('%-5s N=%4d c0=%.4f a=%.4f  dE_c=%+7.3f dE_d=%+7.3f dE_t=%+7.3f (%.0fs)'
              % (name, N, c0, a, Ecc - E_c, Ecc - E_d, Ecc - E_t, time.time() - t0),
              flush=True)


if __name__ == '__main__':
    step = int(os.environ.get('STEP', '5'))
    for nm in sys.argv[1:]:
        run(nm, step)
