#!/usr/bin/env python3
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

"""Diagnose the environment before running the notebooks.

Run this first if a notebook fails at its import cell:

    python scripts/check_environment.py
"""
import importlib.metadata as md
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))

TESTED = {'qiskit': '2.5.2', 'qiskit-nature': '0.8.0', 'qiskit-algorithms': '0.4.0',
          'qiskit-aer': '0.17.2', 'pyscf': '2.14.0', 'numpy': '2.4.4',
          'scipy': '1.17.1', 'pandas': '3.0.2', 'matplotlib': '3.10.8'}

print(f'python {sys.version.split()[0]}\n')
print(f'{"package":22s}{"installed":14s}{"tested":14s}')
missing = []
for p, want in TESTED.items():
    try:
        got = md.version(p)
    except md.PackageNotFoundError:
        got = 'NOT INSTALLED'
        missing.append(p)
    flag = '' if got == want else '   <- differs'
    print(f'  {p:20s}{got:14s}{want:14s}{flag}')

print('\nimport chain used by the notebooks')
for label, code in (
        ('qiskit_algorithms', 'import qiskit_algorithms'),
        ('qiskit_nature drivers', 'from qiskit_nature.second_q.drivers import MethodType'),
        ('qiskit_aer', 'import qiskit_aer'),
        ('pyscf', 'from pyscf import gto, scf'),
        ('succd.selected_cc_pipeline', 'from succd import selected_cc_pipeline')):
    try:
        exec(compile(code, '<check>', 'exec'), {})
        print(f'  ok   {label}')
    except Exception as e:
        print(f'  FAIL {label}: {type(e).__name__}: {str(e)[:90]}')
        if 'AbstractOdeFunction' in str(e):
            print('       -> the qiskit-algorithms circular import. succd/__init__.py')
            print('          works around it; import that module before qiskit_nature, or')
            print('          add at the top of your own script:')
            print('          import qiskit_algorithms.time_evolvers.variational.solvers')

if missing:
    print(f'\nmissing: {", ".join(missing)}   ->  pip install -r requirements.txt')
