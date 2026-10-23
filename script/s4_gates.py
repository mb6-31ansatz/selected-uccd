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

"""Stage 4 - circuit cost (Figure 3, SI Tables 7 and 8).

Three circuits are built per system by one identical procedure - Hartree-Fock reference,
one PauliEvolutionGate per excitation generator in the order of Step 1, Jordan-Wigner
mapping - and transpiled to {cx, rz, sx, x} at optimisation level 1 with all-to-all
connectivity:

  s-UCCD   the N selected doubles, N returned by the delta selection
  UCCSD    every single and every double of the active space
  s-UCCSD  the N selected doubles plus every single, so that the cost of the singles
           block follows by subtraction

The comparison is therefore structural: one synthesis applied to different operator
pools.

Output: out/gates.json
"""
import json, os, sys, time, warnings
warnings.filterwarnings('ignore')
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), _os.pardir))
from succd import paths  # noqa: E402
from succd import core
from qiskit import transpile
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit_nature.second_q.mappers import JordanWignerMapper

OUT = paths.RESULTS
paths.ensure(OUT)
F = os.path.join(OUT, 'gates.json')
BASIS = ['cx', 'rz', 'sx', 'x']
OPT = 1
SEED = 7


def circ(C, ops):
    hf = HartreeFock(C['M'], C['npart'], JordanWignerMapper())
    qc = QuantumCircuit(hf.num_qubits)
    qc.compose(hf, inplace=True)
    for op in ops:
        qc.append(PauliEvolutionGate(op, time=0.1), range(hf.num_qubits))
    return qc


def counts(qc):
    t = transpile(qc, basis_gates=BASIS, optimization_level=OPT, seed_transpiler=SEED)
    d = dict(t.count_ops())
    return dict(nq=int(t.num_qubits), depth=int(t.depth()),
                cx=int(d.get('cx', 0)), total=int(sum(d.values())))


def run(name):
    t0 = time.time()
    C = core.build(name)
    # capture the COMPLETE operator pool before selection: select_doubles_by_pt2 prunes
    # C['mapped'] in place, so after it the "full" doubles list is the selected one.
    em = C['em_jw']
    mp_all = list(C['mapped'])
    sidx = list(C['singles_idx'])
    n_doubles_all = len(mp_all)
    n_singles_all = len(sidx)

    N = core.nsel(C)
    mp = C['mapped']
    ops_sel = [em[mp[k]] for k in range(N)]

    rec = dict(system=name, N=int(N), n_doubles_all=int(n_doubles_all),
               n_singles_all=int(n_singles_all))
    rec['s-UCCD'] = counts(circ(C, ops_sel))
    print('  %s s-UCCD  cx=%d total=%d' % (name, rec['s-UCCD']['cx'],
                                           rec['s-UCCD']['total']), flush=True)

    # full UCCSD: every mappable excitation of the active space, exactly as the
    # shipped gates.py builds it -- em_jw is indexed by the complete excitation list,
    # of which C['mapped'] selects only the doubles that survived pre-screening.
    ops_full = [o for o in em if o is not None]
    rec['n_params_UCCSD'] = len(ops_full)
    rec['UCCSD'] = counts(circ(C, ops_full))
    print('  %s UCCSD   cx=%d total=%d' % (name, rec['UCCSD']['cx'],
                                           rec['UCCSD']['total']), flush=True)

    # selected doubles + the complete singles block, for the subtraction of SI Table 7
    if n_singles_all:
        ops_sd = ops_sel + [em[sidx[k]] for k in range(n_singles_all)]
        rec['s-UCCD+S'] = counts(circ(C, ops_sd))
        rec['cx_singles_block'] = int(rec['s-UCCD+S']['cx'] - rec['s-UCCD']['cx'])
        rec['cx_per_single'] = (rec['cx_singles_block'] / n_singles_all
                                if n_singles_all else None)
        print('  %s +S      cx=%d  (singles block %d)'
              % (name, rec['s-UCCD+S']['cx'], rec['cx_singles_block']), flush=True)
    rec['cx_per_double'] = rec['s-UCCD']['cx'] / N
    rec['cx_ratio'] = rec['UCCSD']['cx'] / rec['s-UCCD']['cx']
    rec['total_ratio'] = rec['s-UCCD']['total'] / rec['UCCSD']['total']

    A = json.load(open(F)) if os.path.exists(F) else {}
    A[name] = rec
    json.dump(A, open(F, 'w'), indent=1)
    print('%s gates N=%d cx %d vs %d (x%.2f) per-double %.1f  (%.0fs)'
          % (name, N, rec['s-UCCD']['cx'], rec['UCCSD']['cx'], rec['cx_ratio'],
             rec['cx_per_double'], time.time() - t0), flush=True)


if __name__ == '__main__':
    for nm in sys.argv[1:]:
        run(nm)
