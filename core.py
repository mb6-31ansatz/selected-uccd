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

"""Shared context/evaluator helpers for the s-UCCD parameterisation study."""
import warnings, numpy as np; warnings.filterwarnings('ignore')
from . import selected_cc_pipeline as scp
from .geoms import SYSTEMS
from pyscf.fci import direct_spin1, cistring
K = 627.5094740631
import os
DELTA = float(os.environ.get('SUCCD_DELTA', '3e-4'))   # PT2 selection threshold (r15B: 3e-4)

def build(name):
    xyz, nf = SYSTEMS[name]
    C = scp.build_union_context(xyz, basis='sto-6g', n_frozen=nf, verbose=False)
    C['_tap'] = (C.get('Hmat') is not None and C.get('gens_tap') is not None)
    return C

def nsel(C):
    return scp.select_doubles_by_pt2(C, delta=DELTA, order_by='energy', verbose=False)

def evaluator(C, take=()):
    """Return E(params) in kcal/mol; params = [doubles..., singles...] for `take`."""
    M, na_, nb_ = C['M'], C['na'], C['nb']; nelec = (na_, nb_)
    nstr = cistring.num_strings(M, na_); shift = C['shift']
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, nelec, .5)
    take = list(take)
    def to_ci(psi):
        ci = np.zeros(nstr*nstr)
        for k in np.where(np.abs(psi) > 1e-14)[0]:
            Ia = int(k) & ((1 << M)-1); Ib = (int(k) >> M) & ((1 << M)-1)
            if bin(Ia).count('1') == na_ and bin(Ib).count('1') == nb_:
                ci[cistring.str2addr(M, na_, Ia)*nstr + cistring.str2addr(M, nb_, Ib)] = np.real(psi[k])
        return ci
    if C['_tap']:
        def E(p, N):
            gens = [C['gens_tap'][k] for k in range(N)] + [C['gens_tap_singles'][t] for t in take]
            return scp.circuit_energy(p, C['psi0'], gens, C['Hmat'], shift)*K
    else:
        def E(p, N):
            psi = np.asarray(scp._propagate_ansatz(C, p, take, 'jw')).ravel()
            ci = to_ci(psi); ci /= np.linalg.norm(ci)
            return (float(np.dot(ci, direct_spin1.contract_2e(h2e, ci.reshape(nstr, nstr),
                    M, nelec).ravel())) + shift)*K
    return E

def state(C, p, take=()):
    """Full JW statevector of the ansatz (needed for SQD sampling)."""
    return np.asarray(scp._propagate_ansatz(C, p, list(take), 'jw')).ravel()

def params(C, N, with_singles):
    """All three parameter vectors on the same operator set.
    doubles: magnitudes from s-CID / s-CID/c0 / s-CCD, signs = MP2 (== s-CCD signs)
    singles: magnitudes from s-CISD / s-CISD/c0 / s-CCSD, signs = sign(-t1) from s-CCSD
    Returns dict with c, cp, t (concatenated vectors), plus diagnostics."""
    dbl = [tuple(map(int, x)) for x in C['dbl'][:N]]
    mp2 = np.sign(C['dfr']['t_mp'].values[:N]).astype(float)
    if with_singles:
        p_cc, take, _ = scp.cc_ansatz_params(N, C, include_singles='pt2')
        sd = scp.getting_save_all_dict(N, C, use_vqe=False, include_singles='pt2',
                                       threshold2=1e-6, compute_energies=False, verbose=False)
        p_ci = np.asarray(sd['vqe_params'])
        t_d, t_s = p_cc[:N], p_cc[N:]                       # t2 (already +MP2 signs), t1
        sel = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                               selected_singles=[C['sgl'][i] for i in take] if 'sgl' in C else [],
                               selected_doubles=dbl, so_labels=C['lbl'])
        # CI reference coefficient c0 from the s-CISD eigenvector
        c0 = float(sd.get('c_HF', 0.0)) or None
        c_d = np.abs(p_ci[:N]); c_s = np.abs(p_ci[N:])
        sgn_s = np.sign(-t_s); sgn_s[sgn_s == 0] = 1.0
        return dict(take=list(take),
                    t=np.concatenate([t_d, -t_s]),
                    c_raw_d=c_d, c_raw_s=c_s, sgn_s=sgn_s, mp2=mp2, t_s=t_s, dbl=dbl)
    else:
        base = [C['HFdet']] + C['double_dets'][:N]
        E_cid, vec = scp.ci_corr(base, C['h1'], C['eri'])
        c0 = abs(vec[0]); c_d = np.abs(vec[1:1+N])
        E_ccd, _, t2 = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                        selected_singles=[], selected_doubles=dbl,
                                        so_labels=C['lbl']).kernel()
        t_d = np.abs(np.array([t2[i, j, a, b] for (i, j, a, b) in dbl]))
        return dict(take=[], c=c_d*mp2, cp=(c_d/c0)*mp2, t=t_d*mp2, c0=c0,
                    E_sCID=E_cid*K, E_sCCD=E_ccd*K,
                    sum_c=float(c_d.sum()), sum_t=float(t_d.sum()), dbl=dbl, mp2=mp2)
