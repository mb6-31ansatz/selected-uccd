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

"""The three schematic figures of the article, drawn with matplotlib.

    toc    the table-of-contents graphic (3.25 x 1.75 in)
    fig1   the algorithm workflow (single column, line art, 1200 dpi)
    fig2   the six molecular systems with their bond lengths, read from
           succd/geoms.py so that the figure cannot drift from the input

    python figures/schematics.py            all three
    python figures/schematics.py fig2       one of them
"""
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle, FancyArrowPatch, Arc

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, os.pardir))
sys.path.insert(0, HERE)
from succd import paths  # noqa: E402
from succd.geoms import SYSTEMS  # noqa: E402
import style as st  # noqa: E402

OUT = paths.FIGURES
NAVY = '#1f3b63'
LIGHT_BLUE = '#9dc3e6'
GREY = '#c9ced6'
PANEL = '#eef1f6'
AMBER = '#f4b41a'
DIM_BLUE = '#4472c4'


def _clean():
    plt.rcdefaults()
    plt.rcParams.update({'font.family': 'sans-serif',
                         'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
                         'mathtext.fontset': 'dejavusans',
                         'savefig.facecolor': 'white', 'figure.facecolor': 'white'})


def _blank(ax, xlim, ylim):
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.axis('off')


# ============================================================== TOC graphic
def toc():
    """Selected CI ranks and thresholds the doubles (left); each retained double
    becomes one parameterised block of the circuit (right); three take-aways."""
    _clean()
    W, H = st.TOC_W, st.TOC_H
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    _blank(ax, (0, 100), (0, 100 * H / W))
    T = 100 * H / W            # 53.8 units tall
    f = 1.0                    # base font scale (pt at print size)

    # ---- left: selected CI --------------------------------------------
    ax.text(16, T - 3.0, 'selected CI', ha='center', va='top', fontsize=7.5 * f,
            fontweight='bold', color=NAVY)
    x0, ytop = 5.0, T - 9.5
    heights = [25, 20, 7, 5, 3]
    bar_h, gap = 2.6, 1.0
    for i, h in enumerate(heights):
        y = ytop - i * (bar_h + gap) - bar_h
        ax.add_patch(Rectangle((x0, y), h, bar_h, facecolor=LIGHT_BLUE if i < 2 else GREY,
                               edgecolor='none'))
    ax.plot([x0, x0], [ytop - 5 * (bar_h + gap) - 0.3, ytop + 0.5], color=NAVY, lw=0.9)
    ax.text(x0 + heights[0] - 0.8, ytop - bar_h / 2, r'$|E_1|$', ha='right', va='center',
            fontsize=5.6 * f, color=NAVY)
    ax.text(x0 + heights[1] - 0.8, ytop - (bar_h + gap) - bar_h / 2, r'$|E_2|$',
            ha='right', va='center', fontsize=5.6 * f, color=NAVY)
    xd = x0 + 9.0
    ax.plot([xd, xd], [ytop - 5 * (bar_h + gap) - 1.5, ytop + 0.5], color=NAVY, lw=0.8,
            ls=(0, (3, 2)))
    ax.text(xd, ytop - 5 * (bar_h + gap) - 2.2, r'$\delta$', ha='center', va='top',
            fontsize=6.5 * f, color=NAVY)

    # ---- arrow -----------------------------------------------------------
    ax.add_patch(FancyArrowPatch((36, T - 19), (46, T - 19), arrowstyle='simple',
                                 mutation_scale=9, color=NAVY, lw=0))
    ax.text(41, T - 15.5, r'$\theta$', ha='center', va='bottom', fontsize=9 * f,
            color=AMBER, fontweight='bold')

    # ---- right: the circuit ----------------------------------------------
    ax.text(74, T - 3.0, 's-UCCD circuit', ha='center', va='top', fontsize=7.5 * f,
            fontweight='bold', color=NAVY)
    wires = [T - 10.5, T - 15.5, T - 20.5, T - 25.5]
    xl, xr = 50, 98
    for y in wires:
        ax.plot([xl, xr], [y, y], color=NAVY, lw=0.9)

    def cnot(xc, ctrl, targ):
        ax.plot([xc, xc], [wires[ctrl], wires[targ]], color=NAVY, lw=0.9)
        ax.add_patch(Circle((xc, wires[ctrl]), 0.9, color=NAVY, zorder=4))
        ax.add_patch(Circle((xc, wires[targ]), 1.4, facecolor='white', edgecolor=NAVY, lw=0.9,
                            zorder=4))
        ax.plot([xc - 1.4, xc + 1.4], [wires[targ]] * 2, color=NAVY, lw=0.9, zorder=5)
        ax.plot([xc, xc], [wires[targ] - 1.4, wires[targ] + 1.4], color=NAVY, lw=0.9, zorder=5)

    def block(xc, targ, label):
        ax.add_patch(FancyBboxPatch((xc - 3.2, wires[targ] - 1.9), 6.4, 3.8,
                                    boxstyle='round,pad=0.1,rounding_size=0.8',
                                    facecolor=AMBER, edgecolor=NAVY, lw=0.8, zorder=4))
        ax.text(xc, wires[targ], label, ha='center', va='center', fontsize=6.5 * f,
                color=NAVY, fontstyle='italic', zorder=5)

    cnot(56, 1, 0)
    block(62, 0, r'$\theta_1$')
    cnot(68, 1, 0)
    cnot(78, 2, 3)
    block(84, 3, r'$\theta_2$')
    cnot(90, 2, 3)
    ax.text(74, T - 29.0, 'one block per selected double', ha='center', va='top',
            fontsize=5.0 * f, color='#4d5a6b')

    # ---- bottom: three take-aways ---------------------------------------
    boxes = [(r'3–12$\times$', 'fewer\ntwo-qubit gates'),
             ('< 0.3 mHa', 'from the\nansatz optimum'),
             ('1', 'conventional\ndiagonalization')]
    bw, bh, by = 30.0, 15.5, 1.5
    for i, (big, small) in enumerate(boxes):
        bx = 2.5 + i * (bw + 2.5)
        ax.add_patch(FancyBboxPatch((bx, by), bw, bh, boxstyle='round,pad=0.1,rounding_size=1.5',
                                    facecolor=PANEL, edgecolor='none'))
        ax.text(bx + bw / 2, by + bh - 1.8, big, ha='center', va='top', fontsize=9.5 * f,
                fontweight='bold', color=NAVY)
        ax.text(bx + bw / 2, by + 1.6, small, ha='center', va='bottom', fontsize=5.0 * f,
                color='#3b4756', linespacing=1.15)
    out = st.save(fig, OUT, 'TOC_graphic', dpi=st.DPI_COMBINATION)
    print('  %-28s %5.2f x %4.2f in  %s' % ('TOC_graphic', W, H, os.path.basename(out['tif'])))
    return out


# ============================================================== Figure 1
def fig1():
    """The four-step workflow, as black-and-white line art."""
    _clean()
    W = st.SINGLE_COLUMN
    H = W * 1.25
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    _blank(ax, (0, 100), (0, 125))
    steps = ['1. Initialization', '2. Ansatz Expansion', '3. Ansatz Construction',
             '4. Retrieve Quantum Circuit\nrelated Outputs']
    bh, gap, bx, bw = 19.0, 12.5, 1.5, 97.0
    ytop = 123.5
    for i, label in enumerate(steps):
        y = ytop - i * (bh + gap) - bh
        ax.add_patch(Rectangle((bx, y), bw, bh, facecolor='white', edgecolor='black', lw=1.3))
        ax.text(bx + bw / 2, y + bh / 2, label, ha='center', va='center', fontsize=9.5,
                fontweight='bold', color=NAVY, linespacing=1.2)
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((50, y - 0.8), (50, y - gap + 0.8),
                                         arrowstyle='-|>', mutation_scale=14,
                                         color='black', lw=1.6))
    out = st.save(fig, OUT, 'fig1_workflow', dpi=st.DPI_LINE_ART)
    print('  %-28s %5.2f x %4.2f in  %s' % ('fig1_workflow', W, H, os.path.basename(out['tif'])))
    return out


# ============================================================== Figure 2
def _atoms(xyz):
    rows = [l.split() for l in xyz.strip().splitlines()]
    return [r[0] for r in rows], np.array([[float(v) for v in r[1:4]] for r in rows])


def _dist(p, i, j):
    return float(np.linalg.norm(p[i] - p[j]))


def _angle(p, i, j, k):
    """Angle i-j-k in degrees."""
    a, b = p[i] - p[j], p[k] - p[j]
    return math.degrees(math.acos(np.dot(a, b) / np.linalg.norm(a) / np.linalg.norm(b)))


def geometry_summary():
    """Bond lengths (Angstrom) and angles (degrees) of the six systems, from geoms.py."""
    g = {}
    lab, p = _atoms(SYSTEMS['CH2O'][0])
    iC = lab.index('C'); iO = lab.index('O'); iH = [i for i, l in enumerate(lab) if l == 'H']
    g['CH2O'] = dict(CO=_dist(p, iC, iO), CH=_dist(p, iC, iH[0]),
                     HCH=_angle(p, iH[0], iC, iH[1]), HCO=_angle(p, iH[0], iC, iO))
    lab, p = _atoms(SYSTEMS['C2H2'][0])
    iC = [i for i, l in enumerate(lab) if l == 'C']; iH = [i for i, l in enumerate(lab) if l == 'H']
    g['C2H2'] = dict(CC=_dist(p, iC[0], iC[1]),
                     CH=min(_dist(p, iC[0], h) for h in iH))
    lab, p = _atoms(SYSTEMS['N2'][0]); g['N2'] = dict(NN=_dist(p, 0, 1))
    lab, p = _atoms(SYSTEMS['CO'][0]); g['CO'] = dict(CO=_dist(p, 0, 1))
    for name in ('H6', 'H8'):
        lab, p = _atoms(SYSTEMS[name][0])
        z = np.sort(p[:, 2])
        g[name] = dict(spacings=[float(z[i + 1] - z[i]) for i in range(len(z) - 1)])
    return g


def fig2():
    """The six systems with their optimized bond lengths (MP2/def2-SVP)."""
    _clean()
    G = geometry_summary()
    W = st.DOUBLE_COLUMN
    H = W * 0.55
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0, 0, 1, 1])
    _blank(ax, (0, 100), (0, 55))
    fs_atom, fs_dim, fs_name, fs_formula = 10.5, 6.0, 8.0, 8.0
    bond_lw = 1.6

    # panel grid: top row 4 cells (width 25), bottom row 2 cells (width 50)
    ax.plot([0, 100], [27.5, 27.5], color='0.55', lw=0.6)
    for xg in (25, 50, 75):
        ax.plot([xg, xg], [27.5, 55], color='0.55', lw=0.6)
    ax.plot([50, 50], [0, 27.5], color='0.55', lw=0.6)

    def dim(x1, x2, y, text, dy=-1.9):
        ax.plot([x1, x2], [y, y], color=DIM_BLUE, lw=0.7)
        for x in (x1, x2):
            ax.plot([x, x], [y - 0.7, y + 0.7], color=DIM_BLUE, lw=0.7)
        ax.text((x1 + x2) / 2, y + dy, text, ha='center', va='top', fontsize=fs_dim,
                color=DIM_BLUE)

    def multi_bond(x1, x2, y, n, sep=0.75):
        offs = [(k - (n - 1) / 2) * sep for k in range(n)]
        for o in offs:
            ax.plot([x1, x2], [y + o, y + o], color='black', lw=bond_lw, solid_capstyle='butt')

    def caption(xc, name, formula):
        ax.text(xc, 33.5, name, ha='center', va='center', fontsize=fs_name, fontweight='bold')
        ax.text(xc, 29.8, formula, ha='center', va='center', fontsize=fs_formula)

    # ---- formaldehyde ----------------------------------------------------
    # H atoms placed at the actual H-C-O angle from the C=O axis, so the drawing
    # has the geometry it labels.
    cx, cy = 12.0, 45.0
    L_co, L_ch = 9.0, 8.0
    ox = cx + L_co
    th = math.radians(G['CH2O']['HCO'])
    h1 = (cx + L_ch * math.cos(th), cy + L_ch * math.sin(th))
    h2 = (cx + L_ch * math.cos(-th), cy + L_ch * math.sin(-th))
    ax.text(cx, cy, 'C', ha='center', va='center', fontsize=fs_atom)
    ax.text(ox, cy, 'O', ha='center', va='center', fontsize=fs_atom)
    for hx, hy in (h1, h2):
        ax.text(hx, hy, 'H', ha='center', va='center', fontsize=fs_atom)
        d = np.array([hx - cx, hy - cy]); d = d / np.linalg.norm(d)
        p1 = np.array([cx, cy]) + 1.9 * d; p2 = np.array([hx, hy]) - 1.7 * d
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color='black', lw=bond_lw)
    multi_bond(cx + 1.9, ox - 1.9, cy, 2)
    dim(cx + 1.9, ox - 1.9, cy - 7.4, '%.2f' % G['CH2O']['CO'])
    # C-H length: a dimension line parallel to the lower bond, offset outwards
    d = np.array([h2[0] - cx, h2[1] - cy]); d = d / np.linalg.norm(d)
    nrm = np.array([-d[1], d[0]])            # normal pointing below the bond
    a = np.array([cx, cy]) + 1.9 * d + 2.2 * nrm
    b = np.array([h2[0], h2[1]]) - 1.7 * d + 2.2 * nrm
    ax.plot([a[0], b[0]], [a[1], b[1]], color=DIM_BLUE, lw=0.7)
    m = (a + b) / 2 + 1.4 * nrm
    ax.text(m[0], m[1], '%.2f' % G['CH2O']['CH'], ha='center', va='center', fontsize=fs_dim,
            color=DIM_BLUE, rotation=math.degrees(math.atan2(d[1], d[0])) + 180)
    # the two angles
    hco = G['CH2O']['HCO']
    ax.add_patch(Arc((cx, cy), 7.0, 7.0, theta1=0, theta2=hco, color=DIM_BLUE, lw=0.7))
    ax.text(cx + 4.6 * math.cos(math.radians(hco / 2)), cy + 4.6 * math.sin(math.radians(hco / 2)),
            '%.1f' % hco, ha='left', va='bottom', fontsize=fs_dim, color=DIM_BLUE)
    ax.add_patch(Arc((cx, cy), 7.0, 7.0, theta1=hco, theta2=360 - hco, color=DIM_BLUE, lw=0.7))
    ax.text(cx - 4.6, cy, '%.1f' % G['CH2O']['HCH'], ha='right', va='center', fontsize=fs_dim,
            color=DIM_BLUE)
    caption(12.5, 'Formaldehyde', r'CH$_2$O')

    # ---- acetylene -------------------------------------------------------
    y = 46.0
    xs = [28.5, 34.5, 40.5, 46.5]
    for x, lab in zip(xs, 'HCCH'):
        ax.text(x, y, lab, ha='center', va='center', fontsize=fs_atom)
    ax.plot([xs[0] + 1.6, xs[1] - 1.6], [y, y], color='black', lw=bond_lw)
    multi_bond(xs[1] + 1.6, xs[2] - 1.6, y, 3)
    ax.plot([xs[2] + 1.6, xs[3] - 1.6], [y, y], color='black', lw=bond_lw)
    dim(xs[0], xs[1], y - 4.5, '%.2f' % G['C2H2']['CH'])
    dim(xs[1], xs[2], y - 4.5, '%.2f' % G['C2H2']['CC'])
    caption(37.5, 'Acetylene', r'C$_2$H$_2$')

    # ---- dinitrogen ------------------------------------------------------
    xa, xb = 57.5, 67.5
    ax.text(xa, y, 'N', ha='center', va='center', fontsize=fs_atom)
    ax.text(xb, y, 'N', ha='center', va='center', fontsize=fs_atom)
    ax.text(xa - 3.0, y, ':', ha='center', va='center', fontsize=fs_atom)
    ax.text(xb + 3.0, y, ':', ha='center', va='center', fontsize=fs_atom)
    multi_bond(xa + 1.7, xb - 1.7, y, 3)
    dim(xa, xb, y - 4.5, '%.2f' % G['N2']['NN'])
    caption(62.5, 'Dinitrogen', r'N$_2$')

    # ---- carbon monoxide -------------------------------------------------
    xa, xb = 82.5, 92.5
    ax.text(xa, y, 'C', ha='center', va='center', fontsize=fs_atom)
    ax.text(xb, y, 'O', ha='center', va='center', fontsize=fs_atom)
    ax.text(xa - 3.0, y, ':', ha='center', va='center', fontsize=fs_atom)
    ax.text(xb + 3.0, y, ':', ha='center', va='center', fontsize=fs_atom)
    multi_bond(xa + 1.7, xb - 1.7, y, 3)
    dim(xa, xb, y - 4.5, '%.2f' % G['CO']['CO'])
    caption(87.5, 'Carbon monoxide', 'CO')

    # ---- hydrogen chains -------------------------------------------------
    def dim_above(x1, x2, y, text):
        ax.plot([x1, x2], [y, y], color=DIM_BLUE, lw=0.7)
        for x in (x1, x2):
            ax.plot([x, x], [y - 0.7, y + 0.7], color=DIM_BLUE, lw=0.7)
        ax.text((x1 + x2) / 2, y + 1.2, text, ha='center', va='bottom', fontsize=fs_dim,
                color=DIM_BLUE)

    def chain2(name, x0, x1, yc, title, formula):
        sp = G[name]['spacings']
        n = len(sp) + 1
        scale = (x1 - x0) / sum(sp)
        xs = [x0]
        for s_ in sp:
            xs.append(xs[-1] + s_ * scale)
        for x in xs:
            ax.text(x, yc, 'H', ha='center', va='center', fontsize=fs_atom)
        for k in range(n - 1):
            ls = '-' if k % 2 == 0 else (0, (2.5, 2.0))
            ax.plot([xs[k] + 1.5, xs[k + 1] - 1.5], [yc, yc], color='black', lw=bond_lw, ls=ls)
        for k in range(n - 1):
            dim_above(xs[k], xs[k + 1], yc + 6.0, '%.2f' % sp[k])
        ax.text((x0 + x1) / 2, 7.0, title, ha='center', va='center', fontsize=fs_name,
                fontweight='bold')
        ax.text((x0 + x1) / 2, 3.3, formula, ha='center', va='center', fontsize=fs_formula)

    chain2('H6', 4.0, 46.0, 16.5, 'Hydrogen trimer', r'H$_6$')
    chain2('H8', 53.5, 96.5, 16.5, 'Hydrogen tetramer', r'H$_8$')

    out = st.save(fig, OUT, 'fig2_geometries', dpi=st.DPI_COMBINATION)
    print('  %-28s %5.2f x %4.2f in  %s' % ('fig2_geometries', W, H, os.path.basename(out['tif'])))
    return out


REGISTRY = {'toc': toc, 'fig1': fig1, 'fig2': fig2}

if __name__ == '__main__':
    names = [a for a in sys.argv[1:] if not a.startswith('-')] or list(REGISTRY)
    os.makedirs(OUT, exist_ok=True)
    print('schematics ->', OUT)
    for n in names:
        REGISTRY[n]()
