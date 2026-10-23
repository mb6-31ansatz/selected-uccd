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

"""Stage driver with on-disk context caching, so each stage fits in one call."""
import sys; sys.setrecursionlimit(300000)
import warnings, sys, json, os, time, pickle, numpy as np; warnings.filterwarnings('ignore')
from scipy.optimize import minimize, minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh
from . import selected_cc_pipeline as scp
from pyscf.fci import direct_spin1, cistring
from . import core
K = core.K; SHOTS = [1e6, 1e8]; SEED = 2024

_CACHE={}
def get(name):
    if name in _CACHE: return _CACHE[name]
    C = core.build(name); n = core.nsel(C); _CACHE[name]=(C,n); return C, n

def save(name, key, val):
    from . import paths
    paths.ensure(paths.RESULTS)
    f = paths.results(f'res_{name}.json'); A = json.load(open(f)) if os.path.exists(f) else {}
    A[key] = val; json.dump(A, open(f, 'w'), indent=1)

def ci_energy(psi, C):
    M, na_, nb_ = C['M'], C['na'], C['nb']; nelec = (na_, nb_)
    nstr = cistring.num_strings(M, na_)
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, nelec, .5)
    ci = np.zeros(nstr*nstr)
    for k in np.where(np.abs(psi) > 1e-14)[0]:
        Ia = int(k) & ((1 << M)-1); Ib = (int(k) >> M) & ((1 << M)-1)
        if bin(Ia).count('1') == na_ and bin(Ib).count('1') == nb_:
            ci[cistring.str2addr(M, na_, Ia)*nstr+cistring.str2addr(M, nb_, Ib)] = np.real(psi[k])
    ci /= np.linalg.norm(ci)
    return (float(np.dot(ci, direct_spin1.contract_2e(h2e, ci.reshape(nstr, nstr),
            M, nelec).ravel()))+C['shift'])*K

def sqd(psi, C, shots=SHOTS, seed=SEED):
    M, na_, nb_ = C['M'], C['na'], C['nb']; nelec = (na_, nb_)
    nstr = cistring.num_strings(M, na_)
    h2e = direct_spin1.absorb_h1e(C['h1'], C['eri'], M, nelec, .5)
    HF = (C['HFdet'][0], C['HFdet'][1])
    amp = np.abs(psi); p = amp**2; p = p/p.sum(); rng = np.random.default_rng(seed); out = {}
    for S in shots:
        pseen = -np.expm1(int(S)*np.log1p(-np.clip(p, 0, 1-1e-18)))
        sel = np.where(rng.random(p.size) < pseen)[0]; pairs = []
        for i in sel:
            Ia = int(i) & ((1 << M)-1); Ib = (int(i) >> M) & ((1 << M)-1)
            if bin(Ia).count('1') == na_ and bin(Ib).count('1') == nb_: pairs.append((Ia, Ib))
        if HF not in pairs: pairs.append(HF)
        a = np.unique([cistring.str2addr(M, na_, x)*nstr+cistring.str2addr(M, nb_, y) for x, y in pairs])
        def mv(x):
            v = np.zeros(nstr*nstr); v[a] = x
            return direct_spin1.contract_2e(h2e, v.reshape(nstr, nstr), M, nelec).ravel()[a]
        lam = eigsh(LinearOperator((len(a),)*2, matvec=mv, dtype=float), k=1, which='SA',
                    tol=1e-10, maxiter=40000)[0][0]
        out['%.0e' % S] = dict(N=int(len(a)), E=float((lam+C['shift'])*K))
    return out

def stage_base(name):
    C, n = get(name); M = C['M']; nelec = (C['na'], C['nb'])
    lamF, _ = direct_spin1.kernel(C['h1'], C['eri'], M, nelec, conv_tol=1e-13)
    P = core.params(C, n, False)
    save(name, 'meta', dict(system=name, n=int(n), nq=int(2*M), tapered=bool(C['_tap']),
         FCI=float((lamF+C['shift'])*K), c0=P['c0'], E_sCID=P['E_sCID'], E_sCCD=P['E_sCCD'],
         sum_c=P['sum_c'], sum_t=P['sum_t'], ratio=P['sum_t']/P['sum_c']))
    print(name, 'n=%d nq=%d tap=%s FCI=%.4f c0=%.5f ratio=%.4f' %
          (n, 2*M, C['_tap'], (lamF+C['shift'])*K, P['c0'], P['sum_t']/P['sum_c']), flush=True)

def stage_D(name, which):
    C, n = get(name); P = core.params(C, n, False)
    Ev = core.evaluator(C) if C['_tap'] else None
    p = {'c': P['c'], 'cp': P['cp'], 't': P['t']}[which]
    t0 = time.time(); psi = core.state(C, p)
    e = Ev(p, n) if C['_tap'] else ci_energy(psi, C)
    r = dict(var=float(e), sqd=sqd(psi, C))
    save(name, 'D_'+which, r)
    print(f'{name} D {which}: var={e:.4f} sqd1e6={r["sqd"]["1e+06"]["E"]:.4f}/{r["sqd"]["1e+06"]["N"]} '
          f'sqd1e8={r["sqd"]["1e+08"]["E"]:.4f}/{r["sqd"]["1e+08"]["N"]}  ({time.time()-t0:.0f}s)', flush=True)

def stage_alpha(name):
    C, n = get(name); P = core.params(C, n, False); t0 = time.time()
    if C['_tap']:
        Ev = core.evaluator(C)
        r = minimize_scalar(lambda a: Ev(a*P['c'], n), bracket=(1.0, 1.15),
                            method='brent', options={'xtol': 1e-5})
        out = dict(astar=float(r.x), E=float(r.fun), nfev=int(r.nfev), how='Brent')
    else:
        grid = [float(x) for x in os.environ.get('GRID','1.00,1.04,1.08,1.12,1.16,1.20').split(',')]
        E0 = os.environ.get('E0')
        Eg = []
        for a in grid:
            Eg.append(float(E0) if (a == 1.0 and E0) else ci_energy(core.state(C, a*P['c']), C))
            print('   alpha=%.3f E=%.4f' % (a, Eg[-1]), flush=True)
        i = int(np.argmin(Eg)); sl = slice(max(0, i-1), min(len(grid), i+2))
        q = np.polyfit(grid[sl], Eg[sl], 2); a = -q[1]/(2*q[0])
        out = dict(astar=float(a), E=float(np.polyval(q, a)), grid=grid, Egrid=Eg,
                   how='6-point grid + local parabola')
    save(name, 'alpha', out)
    print(f'{name} alpha*={out["astar"]:.4f} E={out["E"]:.4f} ({time.time()-t0:.0f}s)', flush=True)

def stage_vqe(name, start):
    C, n = get(name)
    if not C['_tap']: print(name, 'VQE skipped (JW)'); return
    P = core.params(C, n, False); Ev = core.evaluator(C); t0 = time.time()
    p0 = {'c': P['c'], 't': P['t']}[start]
    r = minimize(lambda p: Ev(p, n), p0, method='L-BFGS-B',
                 options={'maxiter': int(os.environ.get('MAXIT',4000)), 'ftol': 1e-13, 'gtol': 1e-9})
    save(name, 'vqe_'+start, dict(E=float(r.fun), nfev=int(r.nfev), nit=int(r.nit)))
    print(f'{name} VQE({start})={r.fun:.4f} nfev={r.nfev} ({time.time()-t0:.0f}s)', flush=True)

def _sd_sets(C, n):
    p_cc, take, _ = scp.cc_ansatz_params(n, C, include_singles='pt2'); take = list(take)
    ns = len(take)
    base = [C['HFdet']]+C['double_dets'][:n]+[C['single_dets'][t] for t in take]
    E_cisd, vec = scp.ci_corr(base, C['h1'], C['eri'])
    c0 = abs(vec[0]); cD = np.abs(vec[1:1+n]); cS = np.abs(vec[1+n:1+n+ns])
    mp2 = np.sign(C['dfr']['t_mp'].values[:n]).astype(float)
    tD, tS = p_cc[:n], p_cc[n:]
    sg = np.sign(-tS); sg[sg == 0] = 1.0
    sets = {'c': np.concatenate([cD*mp2, cS*sg]),
            'cp': np.concatenate([cD/c0*mp2, cS/c0*sg]),
            't': np.concatenate([tD, -tS])}
    info = dict(ns=int(ns), c0=float(c0), E_sCISD=float(E_cisd*K),
                t1_absmax=float(np.abs(tS).max()) if ns else 0.0,
                c2_over_c0_absmax=float((cD/c0).max()),
                t1t1_half_absmax=float((np.abs(tS).max()**2)/2) if ns else 0.0)
    return sets, take, info

def stage_SD(name, which):
    C, n = get(name); sets, take, info = _sd_sets(C, n)
    save(name, 'SD_meta', info)
    Ev = core.evaluator(C, take) if C['_tap'] else None
    t0 = time.time(); p = sets[which]; psi = core.state(C, p, take)
    e = Ev(p, n) if C['_tap'] else ci_energy(psi, C)
    r = dict(var=float(e), sqd=sqd(psi, C))
    save(name, 'SD_'+which, r)
    print(f'{name} SD {which}: var={e:.4f} sqd1e6={r["sqd"]["1e+06"]["E"]:.4f}/{r["sqd"]["1e+06"]["N"]} '
          f'sqd1e8={r["sqd"]["1e+08"]["E"]:.4f}/{r["sqd"]["1e+08"]["N"]}  ({time.time()-t0:.0f}s)', flush=True)

def stage_all(name):
    stage_base(name)
    for w in ('c','cp','t'): stage_D(name,w)
    stage_alpha(name)
    for st in ('c','t'): stage_vqe(name,st)
    for w in ('c','cp','t'): stage_SD(name,w)

def stage_cheap(name):
    stage_base(name)
    for w in ('c','cp','t'): stage_D(name,w)
    stage_alpha(name)

def stage_sdall(name):
    for w in ('c','cp','t'): stage_SD(name,w)



def stage_agrid(name, *alphas):
    C, n = get(name); P = core.params(C, n, False)
    out = {}
    for a in [float(x) for x in alphas]:
        t0 = time.time(); e = ci_energy(core.state(C, a*P['c']), C)
        out['%.4f' % a] = float(e)
        print(f'{name} alpha={a:.4f} E={e:.4f} ({time.time()-t0:.0f}s)', flush=True)
    from . import paths
    f = paths.results(f'res_{name}.json'); A = json.load(open(f)) if os.path.exists(f) else {}
    A.setdefault('agrid', {}).update(out); json.dump(A, open(f, 'w'), indent=1)

if __name__ == '__main__':
    fn = sys.argv[1]; globals()['stage_'+fn](*sys.argv[2:])
