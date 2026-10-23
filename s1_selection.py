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

"""Stage 1 - selection.

For every system, build the union context once and record the sorted PT2 selection
scores |E2_i|.  From those scores the map delta -> N follows exactly, for any delta,
without rebuilding anything.  The scores are stored so that every later stage and every
consistency check can be made against one single ranking.

Output: out/selection.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core

OUT = paths.RESULTS
paths.ensure(OUT)
SYSTEMS = ['H6', 'H8', 'N2', 'CO', 'CH2O', 'C2H2']
LABEL = {s: s for s in SYSTEMS}      # article label of each system key
DELTAS = [2e-4, 3e-4]


def run(names):
    f = os.path.join(OUT, 'selection.json')
    A = json.load(open(f)) if os.path.exists(f) else {}
    for name in names:
        t0 = time.time()
        C = core.build(name)
        score = np.sort(np.abs(C['dfr']['E2_i'].values))[::-1]
        rec = dict(system=name, label=LABEL[name], n_total=int(len(score)),
                   scores=[float(x) for x in score])
        for d in DELTAS:
            rec['N_at_%g' % d] = int((score > d).sum())
        # degeneracy structure at the new cut: is the shell split?
        Nnew = rec['N_at_0.0003']
        if 0 < Nnew < len(score):
            rec['score_last_kept'] = float(score[Nnew - 1])
            rec['score_first_dropped'] = float(score[Nnew])
            rec['gap_at_cut'] = float(score[Nnew - 1] - score[Nnew])
            # count members of the degenerate shell straddling the cut (1e-12 Ha tol)
            tol = 1e-12
            lo = Nnew - 1
            while lo > 0 and abs(score[lo - 1] - score[Nnew - 1]) < tol:
                lo -= 1
            hi = Nnew
            while hi < len(score) and abs(score[hi] - score[Nnew - 1]) < tol:
                hi += 1
            rec['shell_split'] = bool(hi > Nnew)
            rec['shell_lo'] = int(lo + 1)
            rec['shell_hi'] = int(hi)
        A[name] = rec
        json.dump(A, open(f, 'w'), indent=1)
        print('%-6s total=%3d  N(2e-4)=%3d  N(3e-4)=%3d  split=%s  (%.0fs)'
              % (name, rec['n_total'], rec['N_at_0.0002'], rec['N_at_0.0003'],
                 rec.get('shell_split'), time.time() - t0), flush=True)


if __name__ == '__main__':
    run(sys.argv[1:] or SYSTEMS)
