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

"""Verify SI Section 4 component by component.

Three things are checked that the summary statistics cannot settle:

  (i)   the entries of Tables 4 and 5, recomputed to more digits than they are printed;
  (ii)  the claim that "the larger departures fall on components of small magnitude",
        which is a statement about the JOINT distribution of |c_i| and the ratio and
        cannot be read off a min/max spread;
  (iii) the effect of the sign convention, since core.params imposes sgn(t_MP1) on t as
        well as on c, so the vector called "the s-CCD amplitudes" in the SI text is not
        in general the raw s-CCD amplitude vector.

Everything is computed with the shipped core / selected_cc_pipeline routines.
"""
import sys, json
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core
from succd import selected_cc_pipeline as scp

name = sys.argv[1]
C = core.build(name)
n = core.nsel(C)
P = core.params(C, n, False)
c, t = P['c'], P['t']
c0 = P['c0']

dbl = [tuple(map(int, x)) for x in C['dbl'][:n]]
_, _, t2 = scp.SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                            selected_singles=[], selected_doubles=dbl,
                            so_labels=C['lbl']).kernel()
t_raw = np.array([t2[i, j, a, b] for (i, j, a, b) in dbl], dtype=float)
s_mp1 = np.sign(C['dfr']['t_mp'].values[:n]).astype(float)

ac, at = np.abs(c), np.abs(t)
ratio = at / ac
mean = ratio.mean()
dev = np.abs(ratio - mean) / mean * 100.0           # per-component departure, %

# (ii) is the departure concentrated on the small components?
order = np.argsort(ac)                              # ascending magnitude
half = n // 2
dev_small = float(dev[order[:half]].mean())
dev_large = float(dev[order[half:]].mean())
rho = float(np.corrcoef(np.log(ac), dev)[0, 1])     # rank-free monotone indicator
top5 = order[::-1][:5]                              # five largest components
worst5 = np.argsort(dev)[::-1][:5]                  # five worst departures
frac_weight_worst = float((ac[worst5] ** 2).sum() / (ac ** 2).sum())

# (iii) sign convention
disagree = int((np.sign(t_raw) != s_mp1).sum())
cos_mp1 = float(c @ t / (np.linalg.norm(c) * np.linalg.norm(t)))
cos_raw = float(c @ t_raw / (np.linalg.norm(c) * np.linalg.norm(t_raw)))

out = dict(system=name, n=int(n), c0=float(c0),
           cos_mp1=cos_mp1, theta_mp1=float(np.degrees(np.arccos(min(1, cos_mp1)))),
           cos_raw=cos_raw, theta_raw=float(np.degrees(np.arccos(min(1, cos_raw)))),
           n_sign_disagree=disagree,
           ratio_mean=float(mean), spread_pct=float((ratio.max() - ratio.min()) / mean * 100),
           norm_ratio=float(np.linalg.norm(t) / np.linalg.norm(c)),
           alpha_est=float((2 - c0 ** 2) / c0), inv_c0=float(1 / c0),
           dev_small_half=dev_small, dev_large_half=dev_large,
           corr_logmag_dev=rho,
           weight_of_5_worst=frac_weight_worst,
           c_abs=[float(x) for x in ac], ratio=[float(x) for x in ratio])
paths.ensure(paths.SI)
json.dump(out, open(paths.si('comp_%s.json' % name), 'w'), indent=1)

nr = out['norm_ratio']
print('== %s  n=%d ==' % (name, n))
print('  Table 5 : ||t||/||c|| = %.6f   alpha_est = %.6f (%+.2f %%)   1/c0 = %.6f (%+.2f %%)'
      % (nr, out['alpha_est'], 100 * (out['alpha_est'] / nr - 1),
         out['inv_c0'], 100 * (out['inv_c0'] / nr - 1)))
print('  Table 4 : cos = %.8f  theta = %.4f deg   spread = %.2f %%'
      % (cos_mp1, out['theta_mp1'], out['spread_pct']))
print('  signs   : s-CCD disagrees with MP1 on %d of %d amplitudes' % (disagree, n))
print('            cos with RAW s-CCD signs = %.8f  theta = %.4f deg'
      % (cos_raw, out['theta_raw']))
print('  claim   : mean departure, smaller half = %.2f %%, larger half = %.2f %%'
      % (dev_small, dev_large))
print('            corr(log|c_i|, departure)   = %+.3f' % rho)
print('            the 5 worst-departing components carry %.2f %% of the doubles weight'
      % (100 * frac_weight_worst))
