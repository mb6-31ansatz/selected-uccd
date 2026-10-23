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

"""Stage 2 - every per-system quantity at N = N(delta), delta = 3e-4.

One system per invocation.  Each block writes its result to out/res_<name>.json as soon
as it finishes, so an interrupted run resumes instead of restarting.  Nothing is
re-implemented: every number comes from the shipped core / selected_cc_pipeline / drv
routines.

Blocks
  base    FCI, c0, E_sCID, E_sCCD, sum|c|, sum|t|
  D_x     variational <H> of the doubles-only circuit and its SQD energies, x in c,cp,t
  alpha   ray optimum alpha* and E(alpha*)
  vqe_x   variational optimum p_m of the fixed ansatz from start x in c,t
  SD_x    the same as D_x with the PT2-selected singles appended
  comp    angle between c and t, component-wise ratio statistics, sign audit
  quad    N_Q, E(D+Q) by exact diagonalisation in HF + doubles + quadruples
  fullcc  full (unselected) CISD and CCSD, and FCI, for the appendix
  mp      the MP1 predictors rho_MP and rho_MP/c0
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core, drv
from succd import selected_cc_pipeline as scp
from succd.geoms import SYSTEMS
from pyscf import gto, scf, cc, ci
from pyscf.fci import direct_spin1, cistring
from scipy.sparse.linalg import LinearOperator, eigsh
from scipy.optimize import minimize, minimize_scalar

K = core.K
OUT = paths.RESULTS
paths.ensure(OUT)


def path(name):
    return os.path.join(OUT, 'res_%s.json' % name)


def load(name):
    p = path(name)
    return json.load(open(p)) if os.path.exists(p) else {}


def save(name, key, val):
    A = load(name)
    A[key] = val
    json.dump(A, open(path(name), 'w'), indent=1)


_CTX = {}


def get(name):
    if name not in _CTX:
        C = core.build(name)
        n = core.nsel(C)
        _CTX[name] = (C, n)
    return _CTX[name]


# ----------------------------------------------------------------- base
def b_base(name):
    C, n = get(name)
    M, nelec = C['M'], (C['na'], C['nb'])
    lamF, _ = direct_spin1.kernel(C['h1'], C['eri'], M, nelec, conv_tol=1e-13)
    P = core.params(C, n, False)
    save(name, 'meta', dict(system=name, n=int(n), nq=int(2 * M), tapered=bool(C['_tap']),
                            FCI=float((lamF + C['shift']) * K), c0=P['c0'],
                            E_sCID=P['E_sCID'], E_sCCD=P['E_sCCD'],
                            sum_c=P['sum_c'], sum_t=P['sum_t'],
                            ratio=P['sum_t'] / P['sum_c']))
    print('%s base n=%d nq=%d FCI=%.4f c0=%.5f' % (name, n, 2 * M, (lamF + C['shift']) * K,
                                                   P['c0']), flush=True)


# ----------------------------------------------------------------- doubles block
def b_D(name, which):
    C, n = get(name)
    P = core.params(C, n, False)
    Ev = core.evaluator(C) if C['_tap'] else None
    p = {'c': P['c'], 'cp': P['cp'], 't': P['t']}[which]
    psi = core.state(C, p)
    e = Ev(p, n) if C['_tap'] else drv.ci_energy(psi, C)
    save(name, 'D_' + which, dict(var=float(e), sqd=drv.sqd(psi, C)))
    print('%s D_%s var=%.4f' % (name, which, e), flush=True)


# ----------------------------------------------------------------- d_m block
def b_Ddm(name):
    """d_m = alpha_est * c_m with alpha_est = (2-c0^2)/c0, the parameterisation of Eq. 5."""
    C, n = get(name)
    P = core.params(C, n, False)
    a = (2 - P['c0'] ** 2) / P['c0']
    Ev = core.evaluator(C) if C['_tap'] else None
    p = a * P['c']
    psi = core.state(C, p)
    e = Ev(p, n) if C['_tap'] else drv.ci_energy(psi, C)
    save(name, 'D_dm', dict(alpha_est=float(a), var=float(e), sqd=drv.sqd(psi, C)))
    print('%s D_dm alpha_est=%.4f var=%.4f' % (name, a, e), flush=True)


# ----------------------------------------------------------------- ray optimum
def b_alpha(name):
    C, n = get(name)
    P = core.params(C, n, False)
    Ev = core.evaluator(C)
    if C['_tap']:
        r = minimize_scalar(lambda a: Ev(a * P['c'], n), bracket=(1.0, 1.15),
                            method='brent', options={'xtol': 1e-6})
        out = dict(astar=float(r.x), E=float(r.fun), nfev=int(r.nfev), how='Brent')
    else:
        # one circuit evaluation costs ~1 min on the 20-qubit systems, so the optimum
        # is located by a fixed grid plus a local parabola instead of by Brent.
        grid = [float(x) for x in os.environ.get(
            'GRID', '1.00,1.05,1.10,1.15,1.20,1.25').split(',')]
        Eg = [float(Ev(a * P['c'], n)) for a in grid]
        for a, e in zip(grid, Eg):
            print('   alpha=%.3f E=%.4f' % (a, e), flush=True)
        i = int(np.argmin(Eg))
        sl = slice(max(0, i - 1), min(len(grid), i + 2))
        q = np.polyfit(grid[sl], Eg[sl], 2)
        astar = float(-q[1] / (2 * q[0]))
        out = dict(astar=astar, E=float(np.polyval(q, astar)), grid=grid, Egrid=Eg,
                   nfev=len(grid), how='6-point grid + local parabola')
    save(name, 'alpha', out)
    print('%s alpha*=%.4f E=%.4f' % (name, out['astar'], out['E']), flush=True)


# ----------------------------------------------------------------- variational optimum
def b_vqe(name, start, maxit=None):
    C, n = get(name)
    P = core.params(C, n, False)
    Ev = core.evaluator(C)
    p0 = {'c': P['c'], 't': P['t'], 'zero': np.zeros_like(P['c'])}[start]
    if maxit is None:
        maxit = int(os.environ.get('MAXIT', 4000))
    t0 = time.time()
    r = minimize(lambda p: Ev(p, n), p0, method='L-BFGS-B',
                 options={'maxiter': int(maxit), 'ftol': 1e-13, 'gtol': 1e-9})
    save(name, 'vqe_' + start, dict(E=float(r.fun), nfev=int(r.nfev), nit=int(r.nit),
                                    maxiter=int(maxit), converged=bool(r.success),
                                    start=start))
    print('%s vqe(%s)=%.6f nit=%d nfev=%d (%.0fs)'
          % (name, start, r.fun, r.nit, r.nfev, time.time() - t0), flush=True)


# ----------------------------------------------------------------- singles-inclusive
def b_SD(name, which):
    C, n = get(name)
    sets, take, info = drv._sd_sets(C, n)
    save(name, 'SD_meta', info)
    Ev = core.evaluator(C, take) if C['_tap'] else None
    p = sets[which]
    psi = core.state(C, p, take)
    e = Ev(p, n) if C['_tap'] else drv.ci_energy(psi, C)
    save(name, 'SD_' + which, dict(var=float(e), sqd=drv.sqd(psi, C)))
    print('%s SD_%s var=%.4f' % (name, which, e), flush=True)


def b_SDdm(name):
    """singles-inclusive d_m: the whole s-CISD vector rescaled by alpha_est."""
    C, n = get(name)
    P = core.params(C, n, False)
    a = (2 - P['c0'] ** 2) / P['c0']
    sets, take, info = drv._sd_sets(C, n)
    Ev = core.evaluator(C, take) if C['_tap'] else None
    p = sets['c'] * a
    psi = core.state(C, p, take)
    e = Ev(p, n) if C['_tap'] else drv.ci_energy(psi, C)
    save(name, 'SD_dm', dict(var=float(e), sqd=drv.sqd(psi, C)))
    print('%s SD_dm var=%.4f' % (name, e), flush=True)


# ----------------------------------------------------------------- angles / ratios
def b_comp(name):
    C, n = get(name)
    P = core.params(C, n, False)
    c, t, c0 = P['c'], P['t'], P['c0']
    dbl = [tuple(map(int, x)) for x in C['dbl'][:n]]
    _, _, t2 = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                selected_singles=[], selected_doubles=dbl,
                                so_labels=C['lbl']).kernel()
    t_raw = np.array([t2[i, j, a, b] for (i, j, a, b) in dbl], dtype=float)
    s_mp1 = np.sign(C['dfr']['t_mp'].values[:n]).astype(float)

    ac, at = np.abs(c), np.abs(t)
    ratio = at / ac
    mean = float(ratio.mean())
    dev = np.abs(ratio - mean) / mean * 100.0
    order = np.argsort(ac)
    half = n // 2
    worst5 = np.argsort(dev)[::-1][:5]
    cos_mp1 = float(c @ t / (np.linalg.norm(c) * np.linalg.norm(t)))
    cos_raw = float(c @ t_raw / (np.linalg.norm(c) * np.linalg.norm(t_raw)))
    tmp = C['dfr']['t_mp'].values[:n].astype(float)
    rho_mp = float(np.abs(tmp).sum() / ac.sum())

    save(name, 'comp', dict(
        n=int(n), c0=float(c0),
        cos_mp1=cos_mp1, theta_mp1=float(np.degrees(np.arccos(min(1.0, cos_mp1)))),
        cos_raw=cos_raw, theta_raw=float(np.degrees(np.arccos(min(1.0, cos_raw)))),
        n_sign_disagree=int((np.sign(t_raw) != s_mp1).sum()),
        ratio_mean=mean,
        spread_pct=float((ratio.max() - ratio.min()) / mean * 100),
        norm_ratio=float(np.linalg.norm(t) / np.linalg.norm(c)),
        alpha_est=float((2 - c0 ** 2) / c0), inv_c0=float(1 / c0),
        dev_small_half=float(dev[order[:half]].mean()),
        dev_large_half=float(dev[order[half:]].mean()),
        corr_logmag_dev=float(np.corrcoef(np.log(ac), dev)[0, 1]),
        weight_of_5_worst=float((ac[worst5] ** 2).sum() / (ac ** 2).sum()),
        rho_mp=rho_mp, rho_mp_over_c0=float(rho_mp / c0),
        c_abs=[float(x) for x in ac], ratio=[float(x) for x in ratio]))
    print('%s comp cos=%.8f disagree=%d normratio=%.4f'
          % (name, cos_mp1, int((np.sign(t_raw) != s_mp1).sum()),
             np.linalg.norm(t) / np.linalg.norm(c)), flush=True)


# ----------------------------------------------------------------- quadruples
def b_quad(name):
    C, n = get(name)
    HFa, HFb = int(C['HFdet'][0]), int(C['HFdet'][1])
    dd = [(int(d[0]), int(d[1])) for d in C['double_dets'][:n]]
    ops = [((HFa & ~Da), (Da & ~HFa), (HFb & ~Db), (Db & ~HFb)) for Da, Db in dd]

    def apply(op, D):
        ra, aa, rb, ab = op
        Da, Db = D
        if (Da & ra) != ra or (Da & aa) or (Db & rb) != rb or (Db & ab):
            return None
        return ((Da & ~ra) | aa, (Db & ~rb) | ab)

    quads = set()
    for i in range(n):
        for j in range(i + 1, n):
            R = apply(ops[i], dd[j])
            if R and ((HFa ^ R[0]).bit_count() + (HFb ^ R[1]).bit_count()) // 2 >= 4:
                quads.add(R)
    M, na_, nb_ = C['M'], C['na'], C['nb']
    nelec = (na_, nb_)
    nstr = cistring.num_strings(M, na_)
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, nelec, .5)
    dets = [(HFa, HFb)] + dd + sorted(quads)
    a = np.unique([cistring.str2addr(M, na_, x) * nstr + cistring.str2addr(M, nb_, y)
                   for x, y in dets])

    def mv(x):
        v = np.zeros(nstr * nstr)
        v[a] = x
        return direct_spin1.contract_2e(h2e, v.reshape(nstr, nstr), M, nelec).ravel()[a]

    lam = eigsh(LinearOperator((len(a),) * 2, matvec=mv, dtype=float), k=1,
                which='SA', tol=1e-11, maxiter=60000)[0][0]
    save(name, 'quad', dict(n=int(n), NQ=int(len(quads)), dim=int(len(a)),
                            E_DQ=float((lam + C['shift']) * K),
                            FCI_dim=int(nstr * nstr)))
    print('%s quad NQ=%d E_DQ=%.4f' % (name, len(quads), (lam + C['shift']) * K), flush=True)


# ----------------------------------------------------------------- unselected CI / CC
def b_fullcc(name):
    """Full (unselected) CISD and CCSD, the CISD reference weight and the Davidson
    correction (1 - c0^2) E_CISD of Figure A1.  c0 is the reference coefficient of
    the CISD vector normalised with the spin-adapted metric of PySCF (cisd.dot), not
    with the Euclidean norm of the packed amplitude vector."""
    from pyscf.ci import cisd as _cisd
    xyz, nf = SYSTEMS[name]
    mol = gto.M(atom=xyz, basis='sto-6g', verbose=0)
    mf = scf.RHF(mol).run(conv_tol=1e-12)
    mycc = cc.CCSD(mf, frozen=nf if nf else None).run()
    myci = ci.CISD(mf, frozen=nf if nf else None)
    myci.conv_tol = 1e-10
    e_ci, vec = myci.kernel()
    c0 = float(abs(vec[0]) / np.sqrt(_cisd.dot(vec, vec, myci.nmo, myci.nocc)))
    e_dav = (1.0 - c0 ** 2) * e_ci
    save(name, 'fullcc', dict(E_CCSD=float(mycc.e_corr * K), E_CISD=float(e_ci * K),
                              c0_CISD=c0, E_Davidson=float(e_dav * K),
                              E_CISD_Davidson=float((e_ci + e_dav) * K)))
    print('%s fullcc CISD=%.4f CCSD=%.4f c0=%.5f Davidson=%.4f'
          % (name, e_ci * K, mycc.e_corr * K, c0, e_dav * K), flush=True)


BLOCKS = {
    'base': b_base, 'alpha': b_alpha, 'comp': b_comp, 'quad': b_quad,
    'fullcc': b_fullcc, 'Ddm': b_Ddm, 'SDdm': b_SDdm,
    'D_c': lambda n: b_D(n, 'c'), 'D_cp': lambda n: b_D(n, 'cp'),
    'D_t': lambda n: b_D(n, 't'),
    'SD_c': lambda n: b_SD(n, 'c'), 'SD_cp': lambda n: b_SD(n, 'cp'),
    'SD_t': lambda n: b_SD(n, 't'),
    'vqe_c': lambda n: b_vqe(n, 'c'), 'vqe_t': lambda n: b_vqe(n, 't'),
    'vqe_zero': lambda n: b_vqe(n, 'zero'),
}

ALL = ['base', 'D_c', 'D_cp', 'D_t', 'Ddm', 'alpha', 'comp', 'quad', 'fullcc',
       'SD_c', 'SD_cp', 'SD_t', 'SDdm', 'vqe_c', 'vqe_t']


def main():
    name = sys.argv[1]
    blocks = sys.argv[2:] or ALL
    A = load(name)
    keymap = {'D_c': 'D_c', 'D_cp': 'D_cp', 'D_t': 'D_t', 'Ddm': 'D_dm',
              'SD_c': 'SD_c', 'SD_cp': 'SD_cp', 'SD_t': 'SD_t', 'SDdm': 'SD_dm',
              'base': 'meta', 'alpha': 'alpha', 'comp': 'comp', 'quad': 'quad',
              'fullcc': 'fullcc', 'vqe_c': 'vqe_c', 'vqe_t': 'vqe_t',
              'vqe_zero': 'vqe_zero'}
    for b in blocks:
        if keymap.get(b) in A and os.environ.get('FORCE', '') != '1' and not (
                b == 'fullcc' and 'E_Davidson' not in A['fullcc']):
            print('%s %s already present, skipped' % (name, b), flush=True)
            continue
        t0 = time.time()
        BLOCKS[b](name)
        print('   [%s %s done in %.0fs]' % (name, b, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
