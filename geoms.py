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

H_chain8 = """  H   0.00000014047129      0.00000007179489     -2.49463655320066
  H   0.00000005471022      0.00000003978284     -1.74094036515667
  H   -0.00000000410940     -0.00000006874485      1.24863503981317
  H   -0.00000012507223     -0.00000010066620      2.00229939584808
  H   -0.00000019452856     -0.00000006736999      4.99770518795901
  H   -0.00000012436358      0.00000002198468      5.75136984133156
  H   0.00000006114635     -0.00000000235498      8.74093595148680
  H   0.00000019174591      0.00000010557361      9.49463150191870"""
H_chain6 = """  H   -0.00000013033398     -0.00000027656978     -1.63585537349176
  H   -0.00000002026762     -0.00000001049860     -0.88216228307366
  H   0.00000005145092      0.00000026737096      2.12317470217342
  H   0.00000019741506      0.00000028652252      2.87682413358394
  H   0.00000001191796      0.00000000445340      5.88216389396856
  H   -0.00000011018233     -0.00000027127849      6.63585492683949"""
CO = """  O   -1.07180221708728      0.00000000000000      0.00000000000000
  C   0.07180221708728      0.00000000000000      0.00000000000000"""
Acetylene = """  C   0.59610628651506      0.01277743826555      0.14122987213962
  C   -0.59608390436389     -0.01275984381194     -0.14117151138292
  H   1.63992701862499      0.03514052846642      0.38864357314535
  H   -1.63994940077616     -0.03505812292003     -0.38840193390205"""
N2 = """  N   0.00000000000000      0.00000000000000      0.03808486286764
  N   0.00000000000000      0.00000000000000      1.16191513713236"""
Formaldehyde = """  H   1.07578703937518     -0.14380969167056      1.05707279366277
  C   0.59552162304750      0.01521281882427      0.06858835087804
  H   1.27921938071923      0.20323734246116     -0.78595667392319
  O   -0.61132804314190     -0.01554046961487     -0.07040447061762"""
# name -> (xyz block in Angstrom, number of frozen core spatial orbitals)
SYSTEMS = {'H6': (H_chain6, 0), 'H8': (H_chain8, 0), 'N2': (N2, 2), 'CO': (CO, 2),
           'CH2O': (Formaldehyde, 2), 'C2H2': (Acetylene, 2)}
