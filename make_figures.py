#!/usr/bin/env python
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

"""Every data figure of the article and of the Supporting Information.

    python figures/make_figures.py              all figures
    python figures/make_figures.py fig4 figS1   a selection
    python figures/make_figures.py --list       the names

Reads data/values/*.csv, data/values/si_tables.json and data/si/comp_*.json;
recomputes nothing.  Stored energies are in kcal/mol and are converted to mHa
here, which is the unit of the article.  Writes PNG + PDF + TIFF (600 dpi,
RGB, LZW) at journal print width into figures/output/ (or $SUCCD_FIGURES).

The schematic figures (TOC graphic, Figure 1, Figure 2) are in schematics.py.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from succd import paths  # noqa: E402
import style as st  # noqa: E402
from style import (ORDER, LABELS, PRETTY, C_SUCCD, C_UCCSD, C_CM, C_DM, C_TM,  # noqa: E402
                   C_QUAD, C_SYS, SI_BLUE, SI_RED, SI_AMBER, SI_PALE, SI_GREY,
                   DOUBLE_COLUMN, kcal_to_mha, CHEMICAL_ACCURACY_MHA, annotate_bars)

OUT = paths.FIGURES
W = DOUBLE_COLUMN
REGISTRY = {}


def figure(name):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def csv(name):
    return pd.read_csv(paths.values(name))


def finish(fig, name, relayout=None, dpi=st.DPI_COMBINATION, target_w=W):
    w, h = st.fit_width(fig, target_w, max_h=9.0, relayout=relayout)
    out = st.save(fig, OUT, name, dpi=dpi)
    print('  %-28s %5.2f x %4.2f in  %s' % (name, w, h, os.path.basename(out['tif'])))
    return out


# =========================================================== Figure 3 - circuit cost
@figure('fig3')
def fig3():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 27, 26, 28, 23, 22
    bar_w, y_bottom, y_top = 0.38, 1e3, 1.5e6
    df = csv('fig3_gates.csv')
    s = df[df.ansatz == 's-UCCD'].set_index('molecule')
    u = df[df.ansatz == 'UCCSD'].set_index('molecule')
    x = np.arange(len(ORDER))
    fig, (axl, axr), S = st.new_figure(16.5, 6.8, W, fs_tick, ncols=2)
    for ax, col, ylab in ((axl, 'cx_gates', 'number CX gates'),
                          (axr, 'total_gates', 'total number gates')):
        a = [int(s.loc[m, col]) for m in ORDER]
        b = [int(u.loc[m, col]) for m in ORDER]
        ax.set_yscale('log')
        ax.set_ylim(y_bottom, y_top)
        ax.bar(x - bar_w / 2, a, bar_w, color=C_SUCCD, edgecolor='black',
               linewidth=S(1.8), label='s-UCCD', zorder=3)
        ax.bar(x + bar_w / 2, b, bar_w, color=C_UCCSD, edgecolor='black',
               linewidth=S(1.8), label='UCCSD', zorder=3)
        annotate_bars(ax, x - bar_w / 2, a, 0.144, S, fs_annot, '{:d}', log=True)
        annotate_bars(ax, x + bar_w / 2, b, 0.144, S, fs_annot, '{:d}', log=True)
        ax.set_ylabel(ylab, fontsize=S(fs_label))
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
        ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.grid(True, axis='y', zorder=0)
        ax.set_axisbelow(True)
        ax.legend(loc='upper right', fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'fig3_circuit_cost', relayout=fig.tight_layout)


# =================================================== Figure 4 - matched operator sets
@figure('fig4')
def fig4():
    fs_tick, fs_mol, fs_label, fs_legend, fs_tag = 28, 27, 25, 24, 30
    y_bottom, y_top, x_left, x_right = -1.3, 2.75, 0, 78
    df = csv('fig4_sweep.csv')
    fig, axes, S = st.new_figure(20, 7, W, fs_tick, ncols=3, sharey=True)
    cols = {'a': ('dE_cm_kcalmol', r'(a) $c_m$'), 'b': ('dE_dm_kcalmol', r'(b) $d_m$'),
            'c': ('dE_tm_kcalmol', r'(c) $t_m$')}
    for ax, key in zip(axes, 'abc'):
        col, tag = cols[key]
        ax.axhline(0.0, color='black', linewidth=S(2.0), zorder=2)
        for m, lab in zip(ORDER, LABELS):
            d = df[df.molecule == m].sort_values('N')
            ax.plot(d['N'], kcal_to_mha(d[col]), marker='o', markersize=S(9.5),
                    linewidth=S(3.0), color=C_SYS[m], label=lab, zorder=3)
        ax.set_xlim(x_left, x_right)
        ax.set_ylim(y_bottom, y_top)
        ax.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.text(0.025, 0.035, tag, transform=ax.transAxes, fontsize=S(fs_tag),
                ha='left', va='bottom')
    axes[0].set_ylabel(r'$E_{s\mathrm{-CCD}}-\,E_{s\mathrm{-UCCD}}$ [mHa]',
                       fontsize=S(fs_label))
    axes[0].legend(loc='upper right', ncol=2, fontsize=S(fs_legend))

    # one centred abscissa label: the middle panel's label is centred in the figure
    axes[1].set_xlabel('Number of Configurations', fontsize=S(fs_label), labelpad=S(10))
    fig.tight_layout()
    return finish(fig, 'fig4_dE_matched_sets', relayout=fig.tight_layout)


# ============================================== Figure 5 - residual from the optimum
@figure('fig5')
def fig5():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 28, 27, 25, 24, 23
    bar_w, y_bottom, y_top, annot_pad = 0.26, 0.0, 3.5, 0.012
    df = csv('fig5_residual.csv')
    series = {k: df[df.source == k].set_index('molecule') for k in ('c_m', 'd_m', 't_m')}
    # The residual of eq 9 is non-negative by construction.  For the two systems
    # whose optimum p_m is an estimate (capture factor, see caption) the stored
    # residual can be marginally negative (C2H2: -0.015 and -0.074 mHa for d_m and
    # t_m); the figure shows the magnitude, as the submitted figure does.
    vals = {k: np.abs(kcal_to_mha([series[k].loc[m, 'residual_kcalmol'] for m in ORDER]))
            for k in series}
    x = np.arange(len(ORDER))
    pad = annot_pad * (y_top - y_bottom)
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_ylim(y_bottom, y_top)
    for off, k, c, lab in ((-bar_w, 'c_m', C_CM, r'$c_m$'), (0, 'd_m', C_DM, r'$d_m$'),
                           (bar_w, 't_m', C_TM, r'$t_m$')):
        ax.bar(x + off, vals[k], bar_w, color=c, edgecolor='black', linewidth=S(1.8),
               zorder=3, label=lab)
        annotate_bars(ax, x + off, vals[k], pad, S, fs_annot)
    ax.set_ylabel(r'$E_{s\mathrm{-UCCD}}(x_m)-\,E_{s\mathrm{-UCCD}}(p_m)$' '\n[mHa]',
                  fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=3, fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'fig5_residual_correlation', relayout=fig.tight_layout)


# ============================================ Figure 6 - quadruple correlation energy
@figure('fig6')
def fig6():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 28, 27, 25, 20, 23
    bar_w, y_bottom, y_top, annot_pad = 0.26, 0.0, 18.0, 0.012
    df = csv('fig6_quadruples.csv')
    names = {'t': '|E_sUCCD(t_m) - E_sCID|', 'd': '|E_sUCCD(d_m) - E_sCID|',
             'Q': '|E_sCI(D+Q) - E_sCID|'}
    vals = {k: kcal_to_mha([df[df.quantity == q].set_index('molecule').loc[m, 'energy_kcalmol']
                            for m in ORDER]) for k, q in names.items()}
    x = np.arange(len(ORDER))
    pad = annot_pad * (y_top - y_bottom)
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_ylim(y_bottom, y_top)
    for off, k, c, lab in (
            (-bar_w, 't', C_TM, r'$|E_{s\mathrm{-UCCD}}(t_m)-E_{s\mathrm{-CID}}|$'),
            (0, 'd', C_DM, r'$|E_{s\mathrm{-UCCD}}(d_m)-E_{s\mathrm{-CID}}|$'),
            (bar_w, 'Q', C_QUAD, r'$|E_{s\mathrm{-CI(D+Q)}}-E_{s\mathrm{-CID}}|$')):
        ax.bar(x + off, vals[k], bar_w, color=c, edgecolor='black', linewidth=S(1.8),
               zorder=3, label=lab)
        annotate_bars(ax, x + off, vals[k], pad, S, fs_annot)
    ax.set_ylabel('quadruple correlation energy \n[mHa]', fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'fig6_quadruple_correlation_energy', relayout=fig.tight_layout)


# ======================================================= Figure A1 - Enhanced CISD
@figure('figA1')
def figA1():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 28, 27, 25, 22, 23
    bar_w, y_bottom, y_top, annot_pad = 0.26, -3.5, 25.0, 0.012
    df = csv('figA1_enhanced_cisd.csv')
    if 'CISD+Davidson' not in set(df.method):
        raise SystemExit('figA1: run scripts/s2_system.py <system> fullcc and '
                         'scripts/export_values.py first (CISD+Davidson row missing)')
    vals = {k: kcal_to_mha([df[df.method == k].set_index('molecule').loc[m, 'dev_from_FCI_kcalmol']
                            for m in ORDER]) for k in ('CISD', 'CISD+Davidson', 'Enh(d_m)')}
    x = np.arange(len(ORDER))
    pad = annot_pad * (y_top - y_bottom)
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_ylim(y_bottom, y_top)
    for off, k, c, lab in ((-bar_w, 'CISD', C_UCCSD, 'CISD'),
                           (0, 'CISD+Davidson', C_TM, 'CISD + Davidson'),
                           (bar_w, 'Enh(d_m)', C_DM, 'Enhanced CISD, Eq. 11')):
        ax.bar(x + off, vals[k], bar_w, color=c, edgecolor='black', linewidth=S(1.8),
               zorder=3, label=lab)
        annotate_bars(ax, x + off, vals[k], pad, S, fs_annot, rotation=0)
    ax.axhline(0.0, color='black', linewidth=S(1.6), zorder=2)
    ax.axhline(CHEMICAL_ACCURACY_MHA, color='black', linestyle='--', linewidth=S(2.6),
               zorder=4)
    ax.set_ylabel(r'$E-E_{\mathrm{FCI}}$  [mHa]', fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=2, fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'figA1_enhanced_cisd', relayout=fig.tight_layout)


# ===================================================== Figure A2 - SQD deviation
@figure('figA2')
def figA2():
    fs_tick, fs_mol, fs_label, fs_legend = 28, 27, 25, 22
    bar_w, y_bottom, y_top = 0.14, 0.0, 22.0
    df = csv('figA2_sqd.csv')
    col = 'abs_dev_from_FCI_kcalmol'

    def v(p, sub):
        t = df[(df.parameters == p) & (df.subspace == sub)].set_index('molecule')
        return kcal_to_mha([t.loc[m, col] for m in ORDER])
    x = np.arange(len(ORDER))
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_ylim(y_bottom, y_top)
    spec = [(-2.5, 'c', 'SQD-D', C_CM, ''), (-1.5, 'd', 'SQD-D', C_DM, ''),
            (-0.5, 't', 'SQD-D', C_TM, ''), (0.5, 'c', 'SQD-SD', C_CM, '//'),
            (1.5, 'd', 'SQD-SD', C_DM, '//'), (2.5, 't', 'SQD-SD', C_TM, '//')]
    for off, p, sub, c, h in spec:
        ax.bar(x + off * bar_w, v(p, sub), bar_w, color=c, edgecolor='black',
               linewidth=S(1.5), zorder=3, hatch=h, label=r'$%s_m$, %s' % (p, sub))
    ax.set_ylabel(r'$|E_{\mathrm{SQD}}-E_{\mathrm{FCI}}|$' '\n[mHa]', fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=3, fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'figA2_sqd_deviation', relayout=fig.tight_layout)


# ================================================== Figure A3 - determinant counts
@figure('figA3')
def figA3():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 28, 27, 25, 22, 23
    bar_w, y_bottom, y_top = 0.26, 10, 6.0e5
    df = csv('figA3_determinants.csv')
    vals = {k: [int(df[df.series == k].set_index('molecule').loc[m, 'count']) for m in ORDER]
            for k in ('FCI space', 'SQD-D', 'SQD-SD')}
    x = np.arange(len(ORDER))
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_yscale('log')
    ax.set_ylim(y_bottom, y_top)
    for off, k, c, h in ((-bar_w, 'FCI space', C_UCCSD, ''), (0, 'SQD-D', C_DM, ''),
                         (bar_w, 'SQD-SD', C_TM, '//')):
        ax.bar(x + off, vals[k], bar_w, color=c, edgecolor='black', linewidth=S(1.8),
               label=k, hatch=h, zorder=3)
        annotate_bars(ax, x + off, vals[k], 0.144, S, fs_annot, '{:d}', log=True)
    ax.set_ylabel('Number of configurations', fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=8))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='upper right', ncol=2, fontsize=S(fs_legend))
    fig.tight_layout()
    return finish(fig, 'figA3_determinant_counts', relayout=fig.tight_layout)


# ============================================ Figure A4 - circuit energy against SQD
@figure('figA4')
def figA4():
    fs_tick, fs_mol, fs_label, fs_legend, fs_annot = 28, 27, 25, 22, 23
    bar_w, y_bottom, y_top, annot_pad = 0.26, 0.0, 20.0, 0.012
    df = csv('figA4_circuit_vs_sqd.csv')
    vals = {k: kcal_to_mha([df[df.series == k].set_index('molecule').loc[m, 'dev_from_FCI_kcalmol']
                            for m in ORDER]) for k in ('circuit s-UCCD(d_m)', 'SQD-D', 'SQD-SD')}
    x = np.arange(len(ORDER))
    pad = annot_pad * (y_top - y_bottom)
    fig, ax, S = st.new_figure(13.0, 6.8, W, fs_tick)
    ax.set_ylim(y_bottom, y_top)
    for off, k, c, h, lab in ((-bar_w, 'circuit s-UCCD(d_m)', C_DM, '', r'circuit, s-UCCD($d_m$)'),
                              (0, 'SQD-D', C_UCCSD, '', 'SQD-D'),
                              (bar_w, 'SQD-SD', C_TM, '//', 'SQD-SD')):
        ax.bar(x + off, vals[k], bar_w, color=c, edgecolor='black', linewidth=S(1.8),
               zorder=3, hatch=h, label=lab)
        annotate_bars(ax, x + off, vals[k], pad, S, fs_annot)
    ax.set_ylabel(r'$E-E_{\mathrm{FCI}}$' '\n[mHa]', fontsize=S(fs_label))
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS, fontsize=S(fs_mol))
    ax.grid(True, axis='y', zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=S(fs_legend), frameon=False)
    fig.tight_layout()
    return finish(fig, 'figA4_circuit_vs_sqd', relayout=fig.tight_layout)


# ================================================================ SI figures
def _si_style(S):
    plt.rcParams.update({'axes.grid': True, 'axes.grid.axis': 'y',
                         'grid.linestyle': ':', 'grid.linewidth': S(1.0),
                         'grid.color': '0.6', 'axes.axisbelow': True,
                         'legend.framealpha': 0.95})


@figure('figS1')
def figS1():
    """Validity domain of the closed form under bond stretching (Table S1)."""
    T = json.load(open(paths.values('si_tables.json')))
    rows = T['table2']
    c0_prod = [r['c0'] for r in T['table3']]
    fs = 20
    fig, (a1, a2), S = st.new_figure(16, 6.6, W, fs, ncols=2)
    _si_style(S)
    for sysname, col, mk in (('N2', SI_BLUE, 'o'), ('H8', SI_RED, 's')):
        r = sorted([p for p in rows if p['system'] == sysname], key=lambda p: -p['c0'])
        c0 = [p['c0'] for p in r]
        a1.plot(c0, [p['astar'] for p in r], mk + '-', color=col, ms=S(11), lw=S(2.4),
                label=r'%s: $\alpha^{*}$' % PRETTY[sysname])
        a1.plot(c0, [p['a_est'] for p in r], mk + '--', color=col, ms=S(11), lw=S(2.4),
                mfc='none', mew=S(2.0), label=r'%s: $\alpha_{\rm est}$' % PRETTY[sysname])
        a2.semilogy(c0, [max(kcal_to_mha(p['penalty']), 1e-4) for p in r], mk + '-',
                    color=col, ms=S(11), lw=S(2.4), label=PRETTY[sysname])
    for ax in (a1, a2):
        ax.axvspan(min(c0_prod), max(c0_prod), color='0.88', zorder=0)
        ax.set_xlabel(r'reference weight $c_0$')
        ax.invert_xaxis()
    a1.set_ylabel('scale factor')
    a1.legend(ncol=2, fontsize=S(17), loc='upper left')
    a2.axhline(CHEMICAL_ACCURACY_MHA, color='k', ls='--', lw=S(2.0))
    a2.set_ylabel('penalty / mHa')
    a2.legend(fontsize=S(17), loc='lower right')
    fig.tight_layout()
    return finish(fig, 'figS1_validity_domain', relayout=fig.tight_layout)


@figure('figS2')
def figS2():
    """Cost-free predictors of the optimal scale factor (Table S2)."""
    T = json.load(open(paths.values('si_tables.json')))
    P = {r['system']: r for r in T['table3']}
    fs = 20
    x = np.arange(len(ORDER))
    w = 0.36
    fig, (a1, a2), S = st.new_figure(16, 6.6, W, fs, ncols=2)
    _si_style(S)
    bar = dict(edgecolor='black', linewidth=S(1.6))
    a1.bar(x - w / 2, [P[m]['f_inv_c0'] for m in ORDER], w, color=SI_AMBER,
           label=r'$1/c_0$', **bar)
    a1.bar(x + w / 2, [P[m]['f_a_est'] for m in ORDER], w, color=SI_BLUE,
           label=r'$\alpha_{\rm est}$', **bar)
    a1.axhline(1.0, color='k', lw=S(2.0), ls='--')
    a1.set_xticks(x)
    a1.set_xticklabels(LABELS)
    a1.set_ylabel('fraction of ray energy recovered')
    a1.legend(loc='upper left', fontsize=S(17))
    astar = [P[m]['astar'] for m in ORDER]
    a2.plot(astar, astar, 'k--', lw=S(2.0))
    a2.plot(astar, [P[m]['a_est'] for m in ORDER], 'o', color=SI_BLUE, ms=S(14),
            label=r'$\alpha_{\rm est}$')
    a2.plot(astar, [P[m]['inv_c0'] for m in ORDER], 's', color=SI_RED, ms=S(14),
            mfc='none', mew=S(2.2), label=r'$1/c_0$')
    for m in ORDER:
        a2.annotate(PRETTY[m], (P[m]['astar'], P[m]['a_est']), fontsize=S(11),
                    xytext=(S(5), S(5)), textcoords='offset points')
    a2.set_xlabel(r'located optimum $\alpha^{*}$')
    a2.set_ylabel('prediction')
    a2.legend(loc='upper left', fontsize=S(17))
    fig.tight_layout()
    return finish(fig, 'figS2_predictors', relayout=fig.tight_layout)


@figure('figS3')
def figS3():
    """Parallelism of the parameterizations (Table S4)."""
    CMP = {os.path.basename(p)[5:-5]: json.load(open(p))
           for p in glob.glob(paths.si('comp_*.json'))}
    fs = 20
    x = np.arange(len(ORDER))
    w = 0.36
    fig, (a1, a2), S = st.new_figure(16, 6.6, W, fs, ncols=2)
    _si_style(S)
    bar = dict(edgecolor='black', linewidth=S(1.6))
    a1.bar(x - w / 2, [CMP[m]['theta_mp1'] for m in ORDER], w, color=SI_BLUE,
           label='Eq. 7 phases (what the circuit receives)', **bar)
    a1.bar(x + w / 2, [CMP[m]['theta_raw'] for m in ORDER], w, color=SI_RED,
           label='converged s-CCD phases', **bar)
    a1.set_xticks(x)
    a1.set_xticklabels(LABELS)
    a1.set_ylabel(r'$\theta=\arccos\,\langle c,t\rangle/\|c\|\|t\|$  / deg')
    a1.legend(loc='upper right', fontsize=S(16))
    for i, m in enumerate(ORDER):
        r = np.asarray(CMP[m]['ratio'])
        a2.vlines(i, r.min(), r.max(), color=SI_PALE, lw=S(16))
        a2.plot([i] * len(r), r, '|', color=SI_GREY, ms=S(14), mew=S(1.2))
        a2.plot(i, CMP[m]['norm_ratio'], 'o', color=SI_RED, ms=S(13), zorder=5)
    a2.plot([], [], 'o', color=SI_RED, ms=S(13), label=r'$\|t\|/\|c\|$')
    a2.set_xticks(x)
    a2.set_xticklabels(LABELS)
    a2.set_ylabel(r'$|t_i|/|c_i|$')
    a2.legend(loc='upper right', fontsize=S(17))
    fig.tight_layout()
    return finish(fig, 'figS3_parallelism', relayout=fig.tight_layout)


@figure('figS4')
def figS4():
    """The SQD noise condition, eq S1 (Table S7)."""
    T = json.load(open(paths.values('si_tables.json')))
    if T.get('table9_state', '').startswith('doubles'):
        print('  figS4: WARNING data/results/noise_singles.json missing; plotting the '
              'doubles-only condition, which is not what Table S7 reports')
    rows = {r['system']: r for r in T['table9']}
    f = T['fidelity']
    fs = 20
    x = np.arange(len(ORDER))
    fig, (a1, a2), S = st.new_figure(16, 6.6, W, fs, ncols=2)
    _si_style(S)
    bar = dict(edgecolor='black', linewidth=S(1.6))
    a1.bar(x, [rows[m]['D_wmin'] for m in ORDER], 0.5, color=SI_BLUE, **bar)
    a1.set_yscale('log')
    a1.set_xticks(x)
    a1.set_xticklabels(LABELS)
    a1.set_ylabel(r'$N_{\rm sec}\,w_{\min}$')
    ncx = [rows[m]['N_CX'] for m in ORDER]
    a2.bar(x, ncx, 0.5, color=SI_BLUE, label='Eq. S1 (sampled route)', **bar)
    n_direct = int(np.floor(np.log(1 - 0.05) / np.log(f)))
    a2.axhline(n_direct, color=SI_RED, ls='--', lw=S(2.2),
               label='Eq. 3 at 5%% (%d CX)' % n_direct)
    for i, v in enumerate(ncx):
        a2.text(i, v + max(ncx) * 0.02, '%d' % v, ha='center', fontsize=S(17))
    a2.set_xticks(x)
    a2.set_xticklabels(LABELS)
    a2.set_ylabel(r'admitted $N_{CX}$')
    a2.legend(loc='upper left', fontsize=S(17))
    fig.tight_layout()
    return finish(fig, 'figS4_sqd_condition', relayout=fig.tight_layout)


# ================================================================ main
def main(argv):
    if '--list' in argv:
        print('\n'.join(REGISTRY))
        return
    names = [a for a in argv if not a.startswith('-')] or list(REGISTRY)
    os.makedirs(OUT, exist_ok=True)
    print('figures ->', OUT)
    for n in names:
        if n not in REGISTRY:
            raise SystemExit('unknown figure %r; --list shows the names' % n)
        REGISTRY[n]()


if __name__ == '__main__':
    main(sys.argv[1:])
