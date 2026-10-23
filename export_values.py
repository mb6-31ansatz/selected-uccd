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

"""Export - assemble every figure CSV and every SI table from out/*.json.

Nothing is computed here.  Each value is read from the stage outputs and combined by the
definitions the article gives:

  Fig 3   transpiled gate counts, s-UCCD at N against UCCSD on the same active space
  Fig 4   dE'(x_m) = E_sCCD - E_sUCCD(x_m) against N
  Fig 5   dE_sUCCD(x_m) = E_sUCCD(x_m) - E_sUCCD(p_m) at the largest selection
  Fig 6   the surplus over s-CID against the explicit D+Q diagonalisation
  Fig A1  eq 11, E_Enh(d_m) = E_CISD + [E_sUCCD(d_m) - E_sCID], CISD and CISD +
          Davidson, deviation from FCI
  Fig A2  |E_SQD - E_FCI| at 1e8 measurements, doubles and singles-inclusive subspaces
  Fig A3  determinant counts of those subspaces against the FCI space
  Fig A4  circuit energy of s-UCCD(d_m) against the two sampled energies, from FCI

Output: data/values/*.csv, data/values/figure_values.json, data/values/si_tables.json
"""
import csv, json, math, os
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402

OUT = paths.RESULTS
VAL = paths.VALUES
paths.ensure(VAL)

# plotting order used by every figure of the article
ORDER = ['C2H2', 'N2', 'CH2O', 'CO', 'H8', 'H6']
KEY = {m: m for m in ORDER}     # system key -> result-file key (identical since v1.0)
DELTA = 3e-4
FID = 0.9977


def jload(fn):
    p = os.path.join(OUT, fn)
    return json.load(open(p)) if os.path.exists(p) else {}


RES = {m: jload('res_%s.json' % KEY[m]) for m in ORDER}
SEL = jload('selection.json')
GAT = jload('gates.json')
SWP = jload('sweep.json')
NOI = jload('noise.json')                 # doubles-only d_m circuit (Step 2)
NOIS = jload('noise_singles.json')        # singles-inclusive d_m circuit (Application 2)
STR = jload('stretch.json')


def have(m, *keys):
    r = RES[m]
    return all(k in r for k in keys)


def w(fn, header, rows):
    with open(os.path.join(VAL, fn), 'w', newline='') as f:
        c = csv.writer(f)
        c.writerow(header)
        c.writerows(rows)
    print('  %-28s %3d rows' % (fn, len(rows)))


# ------------------------------------------------------------------ Figure 3
def fig3():
    rows = []
    for m in ORDER:
        g = GAT.get(KEY[m])
        if not g:
            continue
        for ans in ('s-UCCD', 'UCCSD'):
            d = g[ans]
            rows.append([m, ans, d['nq'], d['cx'], d['total'], d['depth']])
    w('fig3_gates.csv', ['molecule', 'ansatz', 'n_qubits', 'cx_gates',
                         'total_gates', 'depth'], rows)


# ------------------------------------------------------------------ Figure 4
def fig4():
    rows = []
    for m in ORDER:
        for r in SWP.get(KEY[m], []):
            rows.append([m, r['N'], r['dE_cm'], r['dE_dm'], r['dE_tm'], r['c0']])
    w('fig4_sweep.csv', ['molecule', 'N', 'dE_cm_kcalmol', 'dE_dm_kcalmol',
                         'dE_tm_kcalmol', 'c0'], rows)


# ------------------------------------------------------------------ Figure 5
_KAPPA = {}


def capture_factor():
    """Fraction of the explicit quadruple correlation energy that the variational
    optimum of the fixed ansatz recovers, calibrated on the systems whose optimum was
    actually located:

        kappa = [E(p_m) - E_sCID] / [E(D+Q) - E_sCID]

    For the systems where one circuit evaluation is too expensive to optimise over, p_m
    is estimated by inverting this relation at the mean kappa.  This is the route the
    article already uses for its hatched bars."""
    if _KAPPA:
        return _KAPPA
    ks = {}
    for m in ORDER:
        r = RES[m]
        runs = [r[k] for k in ('vqe_c', 'vqe_t') if k in r]
        if not runs or 'quad' not in r or 'meta' not in r:
            continue
        if not all(x.get('converged') for x in runs):
            continue
        cid = r['meta']['E_sCID']
        den = r['quad']['E_DQ'] - cid
        if abs(den) < 1e-9:
            continue
        ks[m] = (min(x['E'] for x in runs) - cid) / den
    _KAPPA.update(dict(per_system=ks,
                       mean=sum(ks.values()) / len(ks) if ks else None,
                       spread=(max(ks.values()) - min(ks.values())) / 2 if ks else None))
    return _KAPPA


def pm(m):
    """Variational optimum of the fixed ansatz.

    Returned as (energy, estimated).  Measured when a converged optimisation exists;
    otherwise reconstructed from the calibrated capture factor and flagged, so the
    figure can hatch the bar."""
    r = RES[m]
    runs = [r[k] for k in ('vqe_c', 'vqe_t', 'vqe_zero') if k in r]
    if runs and all(x.get('converged', False) for x in runs):
        return min(x['E'] for x in runs), False
    K_ = capture_factor()
    if K_.get('mean') is None or 'quad' not in r or 'meta' not in r:
        return (min(x['E'] for x in runs), True) if runs else (None, None)
    cid = r['meta']['E_sCID']
    return cid + K_['mean'] * (r['quad']['E_DQ'] - cid), True


def fig5():
    rows = []
    for m in ORDER:
        Ep, est = pm(m)
        if Ep is None:
            continue
        for src, key in (('c_m', 'D_c'), ('d_m', 'D_dm'), ('t_m', 'D_t')):
            if key in RES[m]:
                rows.append([m, src, RES[m][key]['var'] - Ep, est])
    w('fig5_residual.csv', ['molecule', 'source', 'residual_kcalmol',
                            'pm_estimated'], rows)
    K_ = capture_factor()
    print('    capture factor %.4f +/- %.4f from %s'
          % (K_['mean'], K_['spread'], ', '.join(K_['per_system'])))
    json.dump(K_, open(os.path.join(VAL, 'capture_factor.json'), 'w'), indent=1)


# ------------------------------------------------------------------ Figure 6
def fig6():
    rows = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'meta', 'quad', 'D_t', 'D_dm'):
            continue
        cid = r['meta']['E_sCID']
        rows.append([m, '|E_sUCCD(t_m) - E_sCID|', abs(r['D_t']['var'] - cid)])
        rows.append([m, '|E_sUCCD(d_m) - E_sCID|', abs(r['D_dm']['var'] - cid)])
        rows.append([m, '|E_sCI(D+Q) - E_sCID|', abs(r['quad']['E_DQ'] - cid)])
    w('fig6_quadruples.csv', ['molecule', 'quantity', 'energy_kcalmol'], rows)


# ------------------------------------------------------------------ Figure A1
def figA1():
    rows = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'meta', 'fullcc', 'D_dm'):
            continue
        fci = r['meta']['FCI']
        cisd = r['fullcc']['E_CISD']
        ccsd = r['fullcc']['E_CCSD']
        enh = cisd + (r['D_dm']['var'] - r['meta']['E_sCID'])
        rows.append([m, 'CISD', cisd - fci])
        if 'E_CISD_Davidson' in r['fullcc']:
            rows.append([m, 'CISD+Davidson', r['fullcc']['E_CISD_Davidson'] - fci])
        rows.append([m, 'Enh(d_m)', enh - fci])
        rows.append([m, 'CCSD', ccsd - fci])
    w('figA1_enhanced_cisd.csv', ['molecule', 'method', 'dev_from_FCI_kcalmol'], rows)


# ------------------------------------------------------------------ Figures A2 / A3
def figA2():
    rows = []
    for m in ORDER:
        r = RES[m]
        if 'meta' not in r:
            continue
        fci = r['meta']['FCI']
        for tag, keyD, keySD in (('c', 'D_c', 'SD_c'), ('d', 'D_dm', 'SD_dm'),
                                 ('t', 'D_t', 'SD_t')):
            for sub, key in (('SQD-D', keyD), ('SQD-SD', keySD)):
                if key not in r:
                    continue
                s = r[key]['sqd']['1e+08']
                rows.append([m, tag, sub, abs(s['E'] - fci), s['N']])
    w('figA2_sqd.csv', ['molecule', 'parameters', 'subspace',
                        'abs_dev_from_FCI_kcalmol', 'n_determinants'], rows)


def figA3():
    rows = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'quad', 'D_dm', 'SD_dm'):
            continue
        rows.append([m, 'FCI space', r['quad']['FCI_dim']])
        rows.append([m, 'SQD-D', r['D_dm']['sqd']['1e+08']['N']])
        rows.append([m, 'SQD-SD', r['SD_dm']['sqd']['1e+08']['N']])
    w('figA3_determinants.csv', ['molecule', 'series', 'count'], rows)


def figA4():
    """Figure A4: deviation from FCI of the s-UCCD(d_m) circuit energy and of the SQD
    energies sampled from it at 1e8 measurements, doubles-only and singles-inclusive."""
    rows = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'meta', 'D_dm', 'SD_dm'):
            continue
        fci = r['meta']['FCI']
        rows.append([m, 'circuit s-UCCD(d_m)', r['D_dm']['var'] - fci])
        rows.append([m, 'SQD-D', r['D_dm']['sqd']['1e+08']['E'] - fci])
        rows.append([m, 'SQD-SD', r['SD_dm']['sqd']['1e+08']['E'] - fci])
    w('figA4_circuit_vs_sqd.csv', ['molecule', 'series', 'dev_from_FCI_kcalmol'], rows)


# ------------------------------------------------------------------ SI tables
def si_tables():
    T = {'delta': DELTA, 'fidelity': FID}

    # Table 1 - singles-inclusive variational energies
    T['table1'] = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'SD_c', 'SD_cp', 'SD_t', 'D_t'):
            continue
        T['table1'].append(dict(system=m, SD_cm=r['SD_c']['var'],
                                SD_cmp=r['SD_cp']['var'], SD_tm=r['SD_t']['var'],
                                D_tm=r['D_t']['var']))

    # Table 2 - bond-stretching scans
    T['table2'] = []
    for tag in ('N2', 'H8'):
        for d in STR.get(tag, []):
            T['table2'].append(dict(system=tag, R=d['dist'], N=d['n'], c0=d['c0'],
                                    astar=d['astar'], a_est=d['a_closed'],
                                    penalty=d['penalty']))

    # Table 3 - cost-free predictors
    T['table3'] = []
    for m in ORDER:
        r = RES[m]
        if not have(m, 'alpha', 'D_c', 'D_cp', 'D_dm', 'comp'):
            continue
        Ec, Ea = r['D_c']['var'], r['alpha']['E']
        gap = Ea - Ec
        T['table3'].append(dict(
            system=m, c0=r['comp']['c0'], astar=r['alpha']['astar'],
            inv_c0=r['comp']['inv_c0'],
            f_inv_c0=(r['D_cp']['var'] - Ec) / gap if abs(gap) > 1e-12 else None,
            a_est=r['comp']['alpha_est'],
            a_est_over_astar=r['comp']['alpha_est'] / r['alpha']['astar'],
            f_a_est=(r['D_dm']['var'] - Ec) / gap if abs(gap) > 1e-12 else None,
            penalty=r['D_dm']['var'] - Ea,
            rho_mp=r['comp']['rho_mp'], rho_mp_over_c0=r['comp']['rho_mp_over_c0']))

    # Tables 4, 5, 6 - angles, component ratios, norm ratios
    T['table4'] = []
    T['table5'] = []
    T['table6'] = []
    for m in ORDER:
        c = RES[m].get('comp')
        if not c:
            continue
        T['table4'].append(dict(system=m, N=c['n'], cos_eq6=c['cos_mp1'],
                                theta_eq6=c['theta_mp1'], cos_sccd=c['cos_raw'],
                                theta_sccd=c['theta_raw'],
                                signs_opposed=c['n_sign_disagree']))
        T['table5'].append(dict(system=m, mean_ratio=c['ratio_mean'],
                                spread_pct=c['spread_pct'],
                                dev_small=c['dev_small_half'],
                                dev_large=c['dev_large_half'],
                                corr=c['corr_logmag_dev'],
                                weight_5_worst=c['weight_of_5_worst']))
        nr = c['norm_ratio']
        T['table6'].append(dict(system=m, norm_ratio=nr, a_est=c['alpha_est'],
                                inv_c0=c['inv_c0'],
                                a_est_err=100 * (c['alpha_est'] / nr - 1),
                                inv_c0_err=100 * (c['inv_c0'] / nr - 1)))

    # Table 7 - gate cost of the singles block
    T['table7'] = []
    for m in ORDER:
        g = GAT.get(KEY[m])
        if not g:
            continue
        T['table7'].append(dict(system=m, singles=g['n_singles_all'],
                                cx_singles_block=g.get('cx_singles_block'),
                                cx_per_single=g.get('cx_per_single'),
                                N=g['N'], cx_succd=g['s-UCCD']['cx'],
                                cx_per_double=g['cx_per_double']))

    # Table 8 - the accuracy condition of Eq. 2
    if T['table1']:
        lo = min(abs(r['SD_tm']) for r in T['table1'])
        hi = max(abs(r['SD_tm']) for r in T['table1'])
    else:
        lo = hi = None
    percx = [r['cx_per_double'] for r in T['table7']] or [None]
    T['table8'] = []
    for eps in (0.01, 0.05, 0.10):
        ncx = int(math.floor(math.log(1 - eps) / math.log(FID)))
        row = dict(eps=eps, ncx=ncx,
                   energy_lo=lo * eps if lo else None,
                   energy_hi=hi * eps if hi else None,
                   doubles_this_work=int(ncx // max(percx)) if percx[0] else None,
                   doubles_13cx=int(ncx // 13))
        T['table8'].append(row)
    T['corr_range'] = [lo, hi]
    T['cx_per_double_range'] = [min(percx), max(percx)] if percx[0] else None

    # Table 9 - the SQD noise condition (SI Table S7, Figure S4).  The submitted SI
    # assesses the singles-inclusive circuit recommended by Application 2; the
    # doubles-only condition is kept under 'table9_doubles_only' because the SI text
    # quotes it for comparison.
    def _t9(src):
        rows = []
        for m in ORDER:
            if KEY[m] not in src:
                continue
            d = src[KEY[m]]
            rows.append(dict(system=m, dets_95=d['dets_95'], populated=d['populated'],
                             D=d['D'], w_min=d['w_min'], D_wmin=d['D_wmin'],
                             p_required=d['p_required'], N_CX=d['N_CX_admitted']))
        return rows
    T['table9_doubles_only'] = _t9(NOI)
    T['table9'] = _t9(NOIS) if NOIS else T['table9_doubles_only']
    T['table9_state'] = 'singles-inclusive d_m circuit' if NOIS else 'doubles-only d_m circuit'

    # selection: N at both thresholds
    T['selection'] = [dict(system=m, N_old=SEL[KEY[m]]['N_at_0.0002'],
                           N_new=SEL[KEY[m]]['N_at_0.0003'],
                           n_total=SEL[KEY[m]]['n_total'],
                           shell_split=SEL[KEY[m]].get('shell_split'),
                           gap_at_cut=SEL[KEY[m]].get('gap_at_cut'))
                      for m in ORDER if KEY[m] in SEL]

    json.dump(T, open(os.path.join(VAL, 'si_tables.json'), 'w'), indent=1)
    print('  si_tables.json              %d tables' % len([k for k in T if k.startswith('table')]))
    return T


def figure_values():
    """One machine-readable file carrying every plotted number and its provenance."""
    V = dict(delta=DELTA, order=ORDER,
             meta={m: RES[m].get('meta') for m in ORDER},
             alpha={m: RES[m].get('alpha') for m in ORDER},
             quad={m: RES[m].get('quad') for m in ORDER},
             fullcc={m: RES[m].get('fullcc') for m in ORDER},
             comp={m: {k: v for k, v in (RES[m].get('comp') or {}).items()
                       if k not in ('c_abs', 'ratio')} for m in ORDER},
             gates={m: GAT.get(KEY[m]) for m in ORDER},
             noise={m: NOI.get(KEY[m]) for m in ORDER},
             vqe={m: {k: RES[m][k] for k in ('vqe_c', 'vqe_t', 'vqe_zero')
                      if k in RES[m]} for m in ORDER},
             energies={m: {k: RES[m][k]['var'] for k in
                           ('D_c', 'D_cp', 'D_dm', 'D_t', 'SD_c', 'SD_cp',
                            'SD_dm', 'SD_t') if k in RES[m]} for m in ORDER})
    json.dump(V, open(os.path.join(VAL, 'figure_values.json'), 'w'), indent=1)
    print('  figure_values.json')
    return V




# ================================================================= SI notebook schema
def si_notebook_data(T):
    """Write data/si/si_tables.json and data/si/comp_*.json (the schema notebook 1 Part B reads)."""
    DATA = paths.SI
    paths.ensure(DATA)
    idx = lambda tab: {r['system']: r for r in T[tab]}
    t1, t3, t7, t9 = idx('table1'), idx('table3'), idx('table7'), idx('table9')
    sel = idx('selection')

    D = {
        '_provenance': {
            'regenerated': 'All values recomputed at delta = 3e-4 Hartree from the '
                           'shipped core / selected_cc_pipeline routines.',
            'units': 'energies in kcal/mol relative to Hartree-Fock unless stated',
        },
        'order': ORDER,
        'pretty': {'H6': 'H$_6$', 'H8': 'H$_8$', 'N2': 'N$_2$', 'CO': 'CO',
                   'CH2O': 'CH$_2$O', 'C2H2': 'C$_2$H$_2$'},
        'delta': DELTA,
        'table1_singles_inclusive': {
            m: dict(SD_c=t1[m]['SD_cm'], SD_cp=t1[m]['SD_cmp'],
                    SD_t=t1[m]['SD_tm'], D_t=t1[m]['D_tm'])
            for m in ORDER if m in t1},
        'table2_stretch': [dict(system=r['system'], R=r['R'], c0=r['c0'],
                                astar=r['astar'], aest=r['a_est'],
                                penalty=r['penalty'], N=r['N'])
                           for r in T['table2']],
        'table3_predictors': {
            m: dict(c0=t3[m]['c0'], astar=t3[m]['astar'], inv_c0=t3[m]['inv_c0'],
                    f_inv_c0=t3[m]['f_inv_c0'], aest=t3[m]['a_est'],
                    aest_over_astar=t3[m]['a_est_over_astar'],
                    f_aest=t3[m]['f_a_est'], penalty=t3[m]['penalty'],
                    rho_mp=t3[m]['rho_mp'], rho_mp_over_c0=t3[m]['rho_mp_over_c0'])
            for m in ORDER if m in t3},
        'table7_gate_cost': {
            m: dict(singles=t7[m]['singles'], cx_singles_block=t7[m]['cx_singles_block'],
                    cx_per_single=t7[m]['cx_per_single'], N_returned=t7[m]['N'],
                    cx_sUCCD=t7[m]['cx_succd'], cx_per_double=t7[m]['cx_per_double'])
            for m in ORDER if m in t7},
        'table8_accuracy_condition': T['table8'],
        'table9_sqd_noise': {'fidelity': FID, 'rows': T['table9']},
        'selection': {m: dict(N_old=sel[m]['N_old'], N_new=sel[m]['N_new'],
                              n_total=sel[m]['n_total'],
                              shell_split=sel[m]['shell_split'])
                      for m in ORDER if m in sel},
        'r15_cx_per_double_as_quoted': {'H6': 105.4, 'H8': 125.6, 'N2': 113.2,
                                        'CO': 104.6, 'CH2O': 137.4, 'C2H2': 136.8},
    }
    json.dump(D, open(os.path.join(DATA, 'si_tables.json'), 'w'), indent=1)
    for m in ORDER:
        c = RES[m].get('comp')
        if c:
            json.dump(c, open(os.path.join(DATA, 'comp_%s.json' % m), 'w'), indent=1)
    print('  data/si/si_tables.json + comp_*.json')


if __name__ == '__main__':
    print('writing values to %s' % VAL)
    fig3(); fig4(); fig5(); fig6(); figA1(); figA2(); figA3(); figA4()
    T = si_tables(); figure_values(); si_notebook_data(T)
    print('done')
