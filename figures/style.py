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

"""House style and export helpers shared by every figure of the article.

Journal requirements (ACS / JCTC, "Preparing Graphics")
-------------------------------------------------------
* single column  <= 3.33 in (240 pt) wide; double column 4.167-7.0 in wide
* maximum depth 9.167 in including the caption
* lettering >= 4.5 pt, lines >= 0.5 pt at final size
* TIFF; 300 dpi colour, 600 dpi combination (text + colour), 1200 dpi line art
* TOC graphic <= 3.25 x 1.75 in

Every figure is designed at the size it was drawn at in the notebooks
(``design_w`` inches wide, with the notebook's point sizes) and rendered scaled
so that its tight bounding box is ``target_w`` inches wide: all point sizes are
multiplied by ``s = target_w / design_w``.  The figure therefore looks exactly
like the notebook figure, at the physical size the journal prints it, and the
raster files carry the correct dpi tag.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DOUBLE_COLUMN = 7.0       # in
SINGLE_COLUMN = 3.33      # in
TOC_W, TOC_H = 3.25, 1.75  # in
DPI_COMBINATION = 600     # colour + text
DPI_LINE_ART = 1200       # black-and-white line art

# ---------------------------------------------------------------- palette
C_SUCCD = '#f4b41a'   # amber, the selected ansatz
C_UCCSD = '#9aa5b1'   # grey, the untruncated ansatz / reference spaces
C_CM, C_DM, C_TM = '#d6413b', '#f4b41a', '#4c8bc9'   # c_m red, d_m amber, t_m blue
C_QUAD = '#7fb069'    # green, explicit quadruple space
C_SYS = {'C2H2': '#4a7fc1', 'N2': '#f5a623', 'CH2O': '#a8d5e2',
         'CO': '#1e8b3a', 'H8': '#8b1a1a', 'H6': '#e8232a'}
SI_BLUE, SI_RED, SI_AMBER = '#2E75B6', '#D6392E', '#F2B233'
SI_PALE, SI_GREY = '#C8D4E0', '#7F8C8D'

ORDER = ['C2H2', 'N2', 'CH2O', 'CO', 'H8', 'H6']
LABELS = [r'$\mathrm{C_2H_2}$', r'$\mathrm{N_2}$', r'$\mathrm{CH_2O}$',
          r'$\mathrm{CO}$', r'$\mathrm{H_8}$', r'$\mathrm{H_6}$']
PRETTY = dict(zip(ORDER, LABELS))


class Scale:
    """Point-size scaler: ``S(28)`` is 28 pt at the design size, scaled."""

    def __init__(self, design_w, target_w):
        self.s = target_w / design_w

    def __call__(self, v):
        return v * self.s


def house_style(S, fs_tick):
    """The rcParams of the article figures, scaled by ``S``."""
    plt.rcdefaults()
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans'],
        'mathtext.fontset': 'dejavusans',
        'font.size': S(fs_tick),
        'xtick.labelsize': S(fs_tick), 'ytick.labelsize': S(fs_tick),
        'axes.linewidth': S(2.0),
        'xtick.major.size': S(9), 'ytick.major.size': S(9),
        'xtick.major.width': S(2.0), 'ytick.major.width': S(2.0),
        'xtick.major.pad': S(5), 'ytick.major.pad': S(5),
        'legend.framealpha': 1.0, 'legend.edgecolor': '0.75',
        'legend.borderpad': 0.45, 'legend.handlelength': 1.5,
        'legend.columnspacing': 1.1, 'legend.labelspacing': 0.35,
        'grid.linestyle': ':', 'grid.linewidth': S(1.4), 'grid.color': '0.78',
        'hatch.linewidth': S(2.2),
        'patch.linewidth': S(1.0),
        'savefig.bbox': 'tight', 'savefig.pad_inches': S(0.06),
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
    })


def new_figure(design_w, design_h, target_w, fs_tick=28, **kw):
    """A figure drawn at the design size and scaled to ``target_w`` inches."""
    S = Scale(design_w, target_w)
    house_style(S, fs_tick)
    fig, axes = plt.subplots(figsize=(design_w * S.s, design_h * S.s), **kw)
    return fig, axes, S


def _tight_width(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width + 2 * plt.rcParams['savefig.pad_inches'], \
        bb.height + 2 * plt.rcParams['savefig.pad_inches']


def fit_width(fig, target_w, max_h=None, n_iter=8, relayout=None):
    """Rescale the figure so that its tight bounding box is ``target_w`` wide.

    Point sizes are untouched; only the canvas grows or shrinks.  With
    ``max_h`` the height is capped as well.  ``relayout`` is a callable
    re-applying the figure's own layout (tight_layout with the right rect).
    The result never exceeds ``target_w`` (nor ``max_h``) by more than 0.2 %."""
    def resize(f):
        fw, fh = fig.get_size_inches()
        fig.set_size_inches(fw * f, fh * f)
        if relayout is not None:
            relayout()
    for _ in range(n_iter):
        w, h = _tight_width(fig)
        f = target_w / w
        if max_h is not None and h * f > max_h:
            f = max_h / h
        if abs(f - 1) < 2e-3:
            break
        resize(f)
    # never over the limit: shrink until the tight box fits
    for _ in range(6):
        w, h = _tight_width(fig)
        over = max(w / target_w, (h / max_h) if max_h else 0.0)
        if over <= 1.002:
            break
        resize(0.995 / over)
    return _tight_width(fig)


def save(fig, outdir, name, dpi=DPI_COMBINATION, pdf=True, png=True, tiff=True):
    """Write ``name``.png / .pdf / .tif.  The TIFF is RGB (no alpha), LZW
    compressed, with the resolution tag set, as the journal asks."""
    os.makedirs(outdir, exist_ok=True)
    paths = {}
    if png or tiff:
        p = os.path.join(outdir, name + '.png')
        fig.savefig(p, dpi=dpi)
        paths['png'] = p
    if pdf:
        p = os.path.join(outdir, name + '.pdf')
        fig.savefig(p)
        paths['pdf'] = p
    if tiff:
        from PIL import Image
        im = Image.open(paths['png'])
        if im.mode != 'RGB':
            bg = Image.new('RGB', im.size, 'white')
            if im.mode == 'RGBA':
                bg.paste(im, mask=im.split()[3])
            else:
                bg.paste(im.convert('RGB'))
            im = bg
        p = os.path.join(outdir, name + '.tif')
        im.save(p, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
        paths['tif'] = p
        if not png:
            os.remove(paths.pop('png'))
    plt.close(fig)
    return paths


def annotate_bars(ax, xs, vals, pad, S, fs, fmt='{:.1f}', log=False, rotation=90):
    """Print the value of every bar above it (below it when negative)."""
    for x, v in zip(xs, vals):
        if log:
            y, va = v * (1 + pad), 'bottom'
        elif v >= 0:
            y, va = v + pad, 'bottom'
        else:
            y, va = v - pad, 'top'
        ax.text(x, y, fmt.format(v), rotation=rotation, ha='center', va=va,
                fontsize=S(fs), clip_on=False)


def kcal_to_mha(x):
    """1 kcal mol^-1 = 1.593601 mHa (CODATA 2018)."""
    return np.asarray(x, dtype=float) * (1000.0 / 627.5094740631)


CHEMICAL_ACCURACY_MHA = 1000.0 / 627.5094740631   # 1 kcal/mol in mHa
