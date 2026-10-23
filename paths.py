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

"""Repository folders.

Every script and notebook resolves its input and output folders through this
module, so nothing depends on the current working directory.  The defaults
point into the repository; each can be overridden with an environment
variable, which is how a run can be redirected into a scratch folder without
touching the tracked data:

    SUCCD_ROOT         repository root (default: the parent of this package)
    SUCCD_RESULTS      raw stage outputs            data/results/
    SUCCD_VALUES       figure CSVs and SI tables    data/values/
    SUCCD_SI           SI component statistics      data/si/
    SUCCD_SHOT_SWEEP   measurement-budget sweep     data/shot_sweep/
    SUCCD_FIGURES      rendered figures             figures/output/
"""
import os

ROOT = os.path.abspath(os.environ.get(
    'SUCCD_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir)))

DATA = os.path.join(ROOT, 'data')
RESULTS = os.path.abspath(os.environ.get('SUCCD_RESULTS', os.path.join(DATA, 'results')))
VALUES = os.path.abspath(os.environ.get('SUCCD_VALUES', os.path.join(DATA, 'values')))
SI = os.path.abspath(os.environ.get('SUCCD_SI', os.path.join(DATA, 'si')))
SHOT_SWEEP = os.path.abspath(os.environ.get('SUCCD_SHOT_SWEEP',
                                            os.path.join(DATA, 'shot_sweep')))
FIGURES = os.path.abspath(os.environ.get('SUCCD_FIGURES',
                                         os.path.join(ROOT, 'figures', 'output')))
NOTEBOOKS = os.path.join(ROOT, 'notebooks')

# plotting order of the six systems, used by every figure of the article
ORDER = ['C2H2', 'N2', 'CH2O', 'CO', 'H8', 'H6']
PRETTY = {'H6': r'H$_6$', 'H8': r'H$_8$', 'N2': r'N$_2$', 'CO': 'CO',
          'CH2O': r'CH$_2$O', 'C2H2': r'C$_2$H$_2$'}


def ensure(*folders):
    """Create the given folders (default: all output folders) if missing."""
    for f in folders or (RESULTS, VALUES, SI, SHOT_SWEEP, FIGURES):
        os.makedirs(f, exist_ok=True)


def results(name):
    """Path of a file in the raw-results folder."""
    return os.path.join(RESULTS, name)


def values(name):
    """Path of a file in the figure-values folder."""
    return os.path.join(VALUES, name)


def si(name):
    """Path of a file in the SI-statistics folder."""
    return os.path.join(SI, name)


def shot_sweep(name):
    """Path of a file in the shot-sweep folder."""
    return os.path.join(SHOT_SWEEP, name)


if __name__ == '__main__':
    for k in ('ROOT', 'RESULTS', 'VALUES', 'SI', 'SHOT_SWEEP', 'FIGURES'):
        print('%-12s %s' % (k, globals()[k]))
