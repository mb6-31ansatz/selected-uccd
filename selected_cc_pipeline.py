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

"""
selected_cc_pipeline.py
=======================

Self-contained "Selected CC(P)" + quantum-circuit (VQE / SQD) pipeline for
small molecules, written so that it depends on as FEW modules as possible.

What it computes, for the top-N double excitations of a molecule, is the dict

    save_all_full[N] = {
        'CID'                     : selected configuration-interaction-doubles E_corr,
        'ccd_e_corr'              : selected coupled-cluster-doubles E_corr,
        'circuit_e_corr_CID'      : <H> of the PauliEvolution ansatz with the
                                    CID amplitudes as (fixed) parameters,
        'vqe_e_corr'              : same ansatz, parameters variationally optimised,
        'sqd_e_corr'              : sample-based quantum diagonalisation, i.e. exact
                                    diagonalisation in the determinant subspace that
                                    the optimised ansatz populates,
        'circuit_e_corr_CCD'      : disabled placeholder branch (see note below),
        'circuit_e_corr_full_CCD' : disabled placeholder branch,
    }

------------------------------------------------------------------------------
MODULES NEEDED (minimal set)
------------------------------------------------------------------------------
  1. THIS file  ............ all the physics + classical CI/CCD + circuit/VQE/SQD.
                             Replaces the old proprietary stack
                             (element_creation_second_quantization_to_ci_MOD,
                              pyscf_problem, CI_matrix_creator, Iterative_CISDTQ,
                              and the original monolithic main script).
  2. pyscf_problem_extended.py ... ONLY used to build the qubit Hamiltonian,
                             its Z2-tapered form, and the JW/tapered excitation
                             Pauli strings.  (your module, unchanged)
  3. PySCFDriver_mod_vers.py ..... the modified driver that #2 imports.

Third-party: pyscf, qiskit, qiskit-nature, qiskit-aer, qiskit-algorithms, scipy.
Tested with qiskit 2.5.2 / nature 0.8.0 / aer 0.17.2 / algorithms 0.4.0, pyscf 2.14.0.

------------------------------------------------------------------------------
KEY DESIGN POINTS
------------------------------------------------------------------------------
* The CI matrix is built from scratch with Slater-Condon rules (Section 2),
  ported from the reference Fortran kernel `sci_ham.f90` and validated against
  PySCF CISD to ~1e-11.  No proprietary CI module is required.

* DOUBLES ARE RANKED BY THE FIRST-ORDER (MP1) COEFFICIENT

      c_i = <HF|H|D_i> / (E_HF - E_{D_i})   with the orbital-energy denominator
          = g_ijab / -(eps_a + eps_b - eps_i - eps_j)

  ranked by |c_i| descending.  (NOT by the 2nd-order energy g^2/D - that picks a
  slightly different subset.)  This is what makes the selected subset - and hence
  every downstream energy - match the reference values exactly.

* The two `circuit_e_corr_CCD` / `circuit_e_corr_full_CCD` entries are DISABLED
  branches in the original formulation (their internal circuit eigenvalue is 0),
  so they evaluate to  0 + core + repulsion - E_HF , a constant placeholder
  (~+30.9 Ha for N2/sto-6G), not a correlation energy.

* Circuit / VQE / SQD are made fast by pre-building each excitation generator as
  a sparse matrix and propagating the statevector with scipy.expm_multiply,
  instead of re-synthesising PauliEvolution gates on every energy evaluation
  (turns a ~30 s evaluation into ~0.05 s).
"""

import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse.linalg import expm_multiply

# qiskit / qiskit-nature (only the pieces we actually use)
from qiskit.quantum_info import Statevector

# ---------------------------------------------------------------------------
# qiskit-algorithms 0.4.0 contains a circular import that fires, on some
# installations, the first time qiskit-nature pulls it in:
#
#   abstract_ode_function -> ..var_qte_linear_solver -> solvers/__init__
#   solvers/__init__      -> var_qte_ode_solver      -> abstract_ode_function
#
# and the last edge finds abstract_ode_function only partially initialised:
#   ImportError: cannot import name 'AbstractOdeFunction' ...
#
# The cycle only breaks the import when abstract_ode_function is entered first.
# Importing the solvers subpackage up front makes solvers/__init__ run first,
# after which the same edges resolve. It is a no-op where the bug does not fire.
# ---------------------------------------------------------------------------
try:
    import qiskit_algorithms.time_evolvers.variational.solvers  # noqa: F401
except Exception:
    pass

from qiskit_nature.second_q.drivers import MethodType
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock

# pyscf
from pyscf import gto, scf, ao2mo, mcscf

# the ONE external module we still rely on (qubit Hamiltonian + excitations)
from . import pyscf_problem_extended as ppe


# =============================================================================
#  SECTION 1 - RHF spin-orbital integrals  +  Selected CCD (purely classical)
# =============================================================================
#
#  Everything here uses only pyscf + numpy.  It produces:
#    * the antisymmetrised spin-orbital integrals (Fock f, two-electron <pq||rs>)
#      in an "occupied-first" ordering with the frozen core removed,
#    * a per-double "isolated" scan giving each double's <HF|H|D>, its MP1
#      amplitude/energy, and its converged selected-CCD amplitude (hence sign),
#    * a Selected-CCD solver (singles can be switched off to give pure CCD).

def _make_tau(t1, t2):
    """tau_ij^ab = t2 + t_i^a t_j^b - t_i^b t_j^a  (CC effective doubles)."""
    return t2 + np.einsum("ia,jb->ijab", t1, t1) - np.einsum("ib,ja->ijab", t1, t1)


def _make_tau_tilde(t1, t2):
    """tau~_ij^ab = t2 + 1/2 (t_i^a t_j^b - t_i^b t_j^a)."""
    return t2 + 0.5 * (np.einsum("ia,jb->ijab", t1, t1)
                       - np.einsum("ib,ja->ijab", t1, t1))


class _DIIS:
    """Minimal Pulay DIIS accelerator for the CC amplitude iteration."""
    def __init__(self, size=6):
        self.size = size
        self._vecs = []
        self._errs = []

    def update(self, vec, err):
        self._vecs.append(vec.copy())
        self._errs.append(err.copy())
        if len(self._vecs) > self.size:
            self._vecs.pop(0)
            self._errs.pop(0)
        n = len(self._vecs)
        if n < 2:
            return vec
        # Solve the small DIIS least-squares problem B c = rhs.
        B = np.zeros((n + 1, n + 1))
        B[-1, :] = -1.0
        B[:, -1] = -1.0
        B[-1, -1] = 0.0
        for i in range(n):
            for j in range(n):
                B[i, j] = float(np.dot(self._errs[i], self._errs[j]))
        rhs = np.zeros(n + 1)
        rhs[-1] = -1.0
        try:
            c = np.linalg.solve(B, rhs)
        except np.linalg.LinAlgError:
            return vec
        return sum(c[i] * self._vecs[i] for i in range(n))


def _ccsd_update(f, g, t1, t2, nocc, nvirt):
    """One CCSD amplitude update (spin-orbital, antisymmetrised integrals).

    Standard Stanton-Gauss intermediates.  `f` is the spin-orbital Fock,
    `g[p,q,r,s] = <pq||rs>` the antisymmetrised two-electron integrals.
    Returns the updated (t1, t2)."""
    o = slice(0, nocc)
    v = slice(nocc, nocc + nvirt)
    ei = np.diag(f)[:nocc]
    ea = np.diag(f)[nocc:]
    tau = _make_tau(t1, t2)
    taut = _make_tau_tilde(t1, t2)

    # --- two-index intermediates F ---
    Fvv = f[v, v].copy()
    Fvv -= 0.5 * np.einsum("me,ma->ae", f[o, v], t1)
    Fvv += np.einsum("mafe,mf->ae", g[o, v, v, v], t1)
    Fvv -= 0.5 * np.einsum("mnaf,mnef->ae", taut, g[o, o, v, v])
    Foo = f[o, o].copy()
    Foo += 0.5 * np.einsum("me,ie->mi", f[o, v], t1)
    Foo += np.einsum("mnie,ne->mi", g[o, o, o, v], t1)
    Foo += 0.5 * np.einsum("inef,mnef->mi", taut, g[o, o, v, v])
    Fov = f[o, v] + np.einsum("nf,mnef->me", t1, g[o, o, v, v])
    Fvv[np.diag_indices(nvirt)] -= ea
    Foo[np.diag_indices(nocc)] -= ei

    # --- four-index intermediates W ---
    Woooo = g[o, o, o, o].copy()
    tmp = np.einsum("je,mnie->mnij", t1, g[o, o, o, v])
    Woooo += tmp - tmp.transpose(0, 1, 3, 2)
    Woooo += 0.25 * np.einsum("ijef,mnef->mnij", tau, g[o, o, v, v])
    Wvvvv = g[v, v, v, v].copy()
    tmp = np.einsum("mb,mafe->bafe", t1, g[o, v, v, v])
    Wvvvv -= tmp - tmp.transpose(1, 0, 2, 3)
    Wvvvv += 0.25 * np.einsum("mnab,mnef->abef", tau, g[o, o, v, v])
    Wovvo = g[o, v, v, o].copy()
    Wovvo += np.einsum("jf,mbef->mbej", t1, g[o, v, v, v])
    Wovvo -= np.einsum("nb,mnej->mbej", t1, g[o, o, v, o])
    Wovvo -= 0.5 * np.einsum("jnfb,mnef->mbej", t2, g[o, o, v, v])
    Wovvo -= np.einsum("jf,nb,mnef->mbej", t1, t1, g[o, o, v, v])

    # --- singles residual ---
    t1new = f[o, v].copy()
    t1new += np.einsum("ie,ae->ia", t1, Fvv)
    t1new -= np.einsum("ma,mi->ia", t1, Foo)
    t1new += np.einsum("imae,me->ia", t2, Fov)
    t1new -= np.einsum("nf,naif->ia", t1, g[o, v, o, v])
    t1new -= 0.5 * np.einsum("imef,maef->ia", t2, g[o, v, v, v])
    t1new -= 0.5 * np.einsum("mnae,mnie->ia", t2, g[o, o, o, v])

    # --- doubles residual ---
    Ftmp = Fvv - 0.5 * np.einsum("mb,me->be", t1, Fov)
    tmp = np.einsum("ijae,be->ijab", t2, Ftmp)
    t2new = tmp - tmp.transpose(0, 1, 3, 2)
    Ftmp2 = Foo + 0.5 * np.einsum("je,me->mj", t1, Fov)
    tmp = np.einsum("imab,mj->ijab", t2, Ftmp2)
    t2new -= tmp - tmp.transpose(1, 0, 2, 3)
    t2new += g[o, o, v, v]
    t2new += 0.5 * np.einsum("mnab,mnij->ijab", tau, Woooo)
    t2new += 0.5 * np.einsum("ijef,abef->ijab", tau, Wvvvv)
    tmp = np.einsum("imae,mbej->ijab", t2, Wovvo)
    tmp += np.einsum("ie,ma,mbje->ijab", t1, t1, g[o, v, o, v])
    tmp = (tmp - tmp.transpose(1, 0, 2, 3)
           - tmp.transpose(0, 1, 3, 2) + tmp.transpose(1, 0, 3, 2))
    t2new += tmp
    tmp = np.einsum("ie,jeba->ijab", t1, g[o, v, v, v])
    t2new += tmp - tmp.transpose(1, 0, 2, 3)
    tmp = np.einsum("ma,ijmb->ijab", t1, g[o, o, o, v])
    t2new -= tmp - tmp.transpose(0, 1, 3, 2)

    # --- divide by orbital-energy denominators (Moller-Plesset style) ---
    eia = ei[:, None] - ea[None, :]
    eijab = eia[:, None, :, None] + eia[None, :, None, :]
    t1new /= eia
    t2new /= eijab
    return t1new, t2new


def _ccsd_energy(f, g, t1, t2, nocc, nvirt):
    """CCSD correlation energy from current amplitudes."""
    o = slice(0, nocc)
    v = slice(nocc, nocc + nvirt)
    E = np.einsum("ia,ia->", f[o, v], t1)
    E += 0.25 * np.einsum("ijab,ijab->", g[o, o, v, v], t2)
    E += 0.5 * np.einsum("ijab,ia,jb->", g[o, o, v, v], t1, t1)
    return float(E)


class SelectedCCSD:
    """Coupled-cluster with an EXPLICIT list of allowed singles/doubles.

    Pass `selected_singles=[]` to get pure **Selected CCD**.  Amplitudes outside
    the selected set are projected to zero on every iteration."""

    def __init__(self, fock, g, nocc, nvirt,
                 selected_singles=None, selected_doubles=None,
                 n_singles=None, n_doubles=None,
                 max_iter=100, tol=1e-8, use_diis=True, diis_size=6,
                 verbose=False, so_labels=None):
        self.f = np.array(fock, float)
        self.g = np.array(g, float)
        self.nocc = nocc
        self.nvirt = nvirt
        self.max_iter = max_iter
        self.tol = tol
        self.use_diis = use_diis
        self.diis_size = diis_size
        self.verbose = verbose
        self.so_labels = so_labels
        o, v = slice(0, nocc), slice(nocc, nocc + nvirt)
        self._o, self._v = o, v
        eps_occ = np.diag(self.f)[o]
        eps_virt = np.diag(self.f)[v]
        self.D1 = (eps_virt[None, :] - eps_occ[:, None])
        self.D2 = (eps_virt[None, None, :, None] + eps_virt[None, None, None, :]
                   - eps_occ[:, None, None, None] - eps_occ[None, :, None, None])
        self.mask1, self.sel_singles = self._build_singles_mask(selected_singles, n_singles)
        self.mask2, self.sel_doubles = self._build_doubles_mask(selected_doubles, n_doubles)

    def _build_singles_mask(self, explicit, n):
        nocc, nvirt = self.nocc, self.nvirt
        mask = np.zeros((nocc, nvirt), bool)
        if explicit is not None:
            pairs = list(explicit)
        elif n is not None:
            score = np.abs(self.f[self._o, self._v] / self.D1)
            flat = score.ravel().argsort()[::-1][:min(n, score.size)]
            pairs = [(idx // nvirt, idx % nvirt) for idx in flat]
        else:
            pairs = [(i, a) for i in range(nocc) for a in range(nvirt)]
        for (i, a) in pairs:
            mask[i, a] = True
        return mask, pairs

    def _build_doubles_mask(self, explicit, n):
        nocc, nvirt = self.nocc, self.nvirt
        mask = np.zeros((nocc, nocc, nvirt, nvirt), bool)
        if explicit is not None:
            pairs = list(explicit)
        elif n is not None:
            o, v = self._o, self._v
            score = np.abs(self.g[o, o, v, v] / self.D2)
            flat = score.ravel().argsort()[::-1][:min(n, score.size)]
            pairs = []
            for idx in flat:
                ij, ab = divmod(idx, nvirt * nvirt)
                i, j = divmod(ij, nocc)
                a, b = divmod(ab, nvirt)
                pairs.append((i, j, a, b))
        else:
            pairs = [(i, j, a, b) for i in range(nocc) for j in range(nocc)
                     for a in range(nvirt) for b in range(nvirt)]
        # Each spatial double implies all four antisymmetric index permutations.
        for (i, j, a, b) in pairs:
            mask[i, j, a, b] = True
            mask[j, i, a, b] = True
            mask[i, j, b, a] = True
            mask[j, i, b, a] = True
        return mask, pairs

    def _apply_mask(self, t1, t2):
        return t1 * self.mask1, t2 * self.mask2

    def kernel(self):
        """Iterate the (selected) CC equations to convergence.
        Returns (E_corr, t1, t2)."""
        nocc, nvirt = self.nocc, self.nvirt
        f, g, o, v = self.f, self.g, self._o, self._v
        t1 = np.zeros((nocc, nvirt))
        t2 = -g[o, o, v, v] / self.D2          # MP1 guess
        t1, t2 = self._apply_mask(t1, t2)
        diis = _DIIS(self.diis_size) if self.use_diis else None
        E_corr = 0.0
        for it in range(1, self.max_iter + 1):
            t1n, t2n = _ccsd_update(f, g, t1, t2, nocc, nvirt)
            t1n, t2n = self._apply_mask(t1n, t2n)
            rms = float(np.sqrt((np.sum((t1n - t1) ** 2) + np.sum((t2n - t2) ** 2))
                                / (t1.size + t2.size)))
            if diis is not None:
                vec = np.concatenate([t1n.ravel(), t2n.ravel()])
                err = np.concatenate([(t1n - t1).ravel(), (t2n - t2).ravel()])
                vec = diis.update(vec, err)
                split = nocc * nvirt
                t1n = vec[:split].reshape(nocc, nvirt)
                t2n = vec[split:].reshape(nocc, nocc, nvirt, nvirt)
                t1n, t2n = self._apply_mask(t1n, t2n)
            t1, t2 = t1n, t2n
            E_corr = _ccsd_energy(f, g, t1, t2, nocc, nvirt)
            if rms < self.tol:
                break
        self.E_corr = E_corr
        self.t1 = t1
        self.t2 = t2
        return E_corr, t1, t2


def spinorb_label(idx, N):
    """Human-readable spin-orbital label: index<N -> alpha, else beta."""
    return f"{idx}\u03b1" if idx < N else f"{idx - N}\u03b2"


def build_spinorb_integrals_rhf(mol, mf, n_frozen_spatial=0, symmetry=True, verbose=False):
    """RHF -> antisymmetrised spin-orbital integrals, frozen core removed.

    Returns
    -------
    fock   : (nso,nso)            spin-orbital Fock (occupied-first ordering)
    g      : (nso,nso,nso,nso)    antisymmetrised <pq||rs>
    nocc   : int                  # active occupied spin-orbitals
    nvirt  : int                  # active virtual spin-orbitals
    N      : int                  # spatial MOs
    so_lbl : list[str]            label of every active spin-orbital
    mf     : the RHF object
    frozen_labels : list[str]
    """
    N = mf.mo_coeff.shape[1]
    nocc_spat = mol.nelectron // 2
    nvirt_spat = N - nocc_spat
    eps = mf.mo_energy

    # AO->MO two-electron integrals, to physicist ordering <pq|rs>.
    eri_4d = ao2mo.restore(1, ao2mo.kernel(mol, mf.mo_coeff), N)
    eri_phys = eri_4d.transpose(0, 2, 1, 3)

    # Spin-block up to antisymmetrised spin-orbital <pq||rs>.
    nso = 2 * N
    fock_so = np.zeros((nso, nso))
    fock_so[:N, :N] = np.diag(eps)
    fock_so[N:, N:] = np.diag(eps)
    g_so = np.zeros((nso, nso, nso, nso))
    sl = [slice(0, N), slice(N, nso)]
    for s1 in range(2):
        for s2 in range(2):
            pq = sl[s1], sl[s2]
            g_so[pq[0], pq[1], pq[0], pq[1]] += eri_phys                       # Coulomb
            g_so[pq[0], pq[1], pq[1], pq[0]] -= eri_phys.transpose(0, 1, 3, 2)  # exchange (same spin only)

    # Reorder to occupied-first (alpha-occ, beta-occ, alpha-virt, beta-virt).
    occ_orig = list(range(nocc_spat)) + list(range(N, N + nocc_spat))
    virt_orig = list(range(nocc_spat, N)) + list(range(N + nocc_spat, nso))
    perm = occ_orig + virt_orig
    fock_so = fock_so[np.ix_(perm, perm)]
    g_so = g_so[np.ix_(perm, perm, perm, perm)]
    nocc = 2 * nocc_spat
    nvirt = 2 * nvirt_spat
    so_labels = [spinorb_label(o, N) for o in perm]
    frozen_labels = []

    # Drop n_frozen_spatial core spatials (both spins).
    if n_frozen_spatial > 0:
        nf = n_frozen_spatial
        frozen_idx = list(range(nf)) + list(range(nocc_spat, nocc_spat + nf))
        frozen_labels = [so_labels[k] for k in frozen_idx]
        active_occ = [k for k in range(nocc) if k not in frozen_idx]
        active_idx = active_occ + list(range(nocc, nocc + nvirt))
        fock_so = fock_so[np.ix_(active_idx, active_idx)]
        g_so = g_so[np.ix_(active_idx, active_idx, active_idx, active_idx)]
        nocc = len(active_occ)
        so_labels = [so_labels[k] for k in active_idx]
    return fock_so, g_so, nocc, nvirt, N, so_labels, mf, frozen_labels


def scan_isolated_doubles(fock, g, nocc, nvirt, lbl, hf_config, skip_zero=True):
    """For every i<j -> a<b double, compute the quantities needed for ranking
    and for fixing the CCD sign.

    Columns of the returned DataFrame:
      i,j,a,b          spin-orbital indices (occupied i,j ; virtual a,b)
      g_ijab           <HF|H|D> = <ij||ab>     (the CI off-diagonal element)
      D                eps_a+eps_b-eps_i-eps_j (orbital-energy denominator)
      t_mp             MP1 amplitude  -g/D
      E2_i             MP2 energy contribution  -g^2/D
      t2_ccsd          converged 1-double CCD amplitude (its sign is THE sign)
      E_corr_isolated  that double's isolated CCD correlation energy
    """
    ei = np.diag(fock)[:nocc]
    ea = np.diag(fock)[nocc:]
    rows = []
    for i in range(nocc):
        for j in range(i + 1, nocc):
            for a in range(nvirt):
                for b in range(a + 1, nvirt):
                    g_ijab = float(g[i, j, nocc + a, nocc + b])
                    if skip_zero and abs(g_ijab) < 1e-12:
                        continue
                    D = float(ea[a] + ea[b] - ei[i] - ei[j])
                    if np.round(D, 6) == 0.0:
                        D = 1.0                      # guard a (rare) zero denominator
                    t_mp = -g_ijab / D
                    E2_i = -(g_ijab ** 2) / D
                    config = tuple(sorted((hf_config - {lbl[i], lbl[j]})
                                          | {lbl[nocc + a], lbl[nocc + b]}))
                    # one-double CCD: gives the converged amplitude -> its sign.
                    ccsd = SelectedCCSD(fock, g, nocc, nvirt,
                                        selected_singles=[], selected_doubles=[(i, j, a, b)],
                                        so_labels=lbl, verbose=False)
                    E_corr, _, t2_conv = ccsd.kernel()
                    # HF coefficient convention C_HF = -1  ->  sign_factor below.
                    t2_ccsd = int(np.sign(-1)) * float(t2_conv[i, j, a, b])
                    rows.append(dict(i=i, j=j, a=a, b=b,
                                     holes=(lbl[i], lbl[j]),
                                     particles=(lbl[nocc + a], lbl[nocc + b]),
                                     configuration=config,
                                     g_ijab=g_ijab, D=D, t_mp=t_mp, E2_i=E2_i,
                                     t2_ccsd=t2_ccsd, E_corr_isolated=float(E_corr)))
    df = pd.DataFrame(rows)
    for col in ("i", "j", "a", "b"):
        df[col] = df[col].astype(int)
    return df.reset_index(drop=True)


# =============================================================================
#  SECTION 2 - custom CI engine (Slater-Condon)   [replaces proprietary CI]
# =============================================================================
#
#  Determinants are represented as a pair of integer bitmasks (Ia, Ib) over the
#  *active spatial* orbitals: bit p set in Ia/Ib => spatial orbital p is occupied
#  by an alpha/beta electron.  Matrix elements use CHEMIST integrals (pq|rs)
#  ("eri") and the effective one-electron integrals h1 over the active space,
#  exactly as the reference Fortran kernel sci_ham.f90 does.

def _popcnt(x):
    return bin(x).count('1')


def _trailz(x):
    return (x & -x).bit_length() - 1


def _occ_list(s):
    out = []
    m = s
    while m:
        b = m & -m
        out.append(b.bit_length() - 1)
        m ^= b
    return out


def _two_orbs(mask):
    a = _trailz(mask)
    b = _trailz(mask & ~(1 << a))
    return a, b


def _parity_below(s, orb):
    return _popcnt(s & ((1 << orb) - 1))


def _rphase(bra, ket):
    """Fermionic phase to turn |ket> into |bra| within one spin string."""
    holes = ket & ~bra
    parts = bra & ~ket
    s = ket
    ph = 1
    h = holes
    while h:
        b = h & -h
        orb = b.bit_length() - 1
        ph *= (1 - 2 * (_parity_below(s, orb) & 1))
        s &= ~(1 << orb)
        h ^= b
    p = parts
    while p:
        b = p & -p
        orb = b.bit_length() - 1
        ph *= (1 - 2 * (_parity_below(s, orb) & 1))
        s |= (1 << orb)
        p ^= b
    return ph


def melem(Ia, Ib, Ja, Jb, h1, eri):
    """<I|H|J> between determinants I=(Ia,Ib), J=(Ja,Jb) via Slater-Condon.

    h1[p,q]        : effective one-electron integrals (active space)
    eri[p,q,r,s]   : chemist (pq|rs) two-electron integrals (active space)
    """
    dA = Ia ^ Ja
    dB = Ib ^ Jb
    ea = _popcnt(dA) // 2     # # of differing alpha spin-orbitals
    eb = _popcnt(dB) // 2     # # of differing beta  spin-orbitals
    if ea + eb > 2:
        return 0.0            # Slater-Condon: >2 differences => 0

    if ea == 0 and eb == 0:                      # ----- diagonal -----
        oa = _occ_list(Ia)
        ob = _occ_list(Ib)
        h = 0.0
        for p in oa:
            h += h1[p, p]
        for p in ob:
            h += h1[p, p]
        for k in oa:                              # alpha-alpha (Coulomb - exchange)
            for l in oa:
                h += 0.5 * (eri[k, k, l, l] - eri[k, l, l, k])
        for k in ob:                              # beta-beta
            for l in ob:
                h += 0.5 * (eri[k, k, l, l] - eri[k, l, l, k])
        for k in oa:                              # alpha-beta (Coulomb only)
            for l in ob:
                h += eri[k, k, l, l]
        return h

    if ea == 1 and eb == 0:                      # ----- single (alpha) -----
        p = _trailz(Ia & ~Ja)
        q = _trailz(Ja & ~Ia)
        ph = _rphase(Ia, Ja)
        h = h1[p, q]
        for r in _occ_list(Ia & Ja):             # common alpha
            h += eri[p, q, r, r] - eri[p, r, r, q]
        for r in _occ_list(Ib):                  # beta (Coulomb only)
            h += eri[p, q, r, r]
        return ph * h

    if ea == 0 and eb == 1:                      # ----- single (beta) -----
        p = _trailz(Ib & ~Jb)
        q = _trailz(Jb & ~Ib)
        ph = _rphase(Ib, Jb)
        h = h1[p, q]
        for r in _occ_list(Ib & Jb):
            h += eri[p, q, r, r] - eri[p, r, r, q]
        for r in _occ_list(Ia):
            h += eri[p, q, r, r]
        return ph * h

    if ea == 2 and eb == 0:                      # ----- double (alpha-alpha) -----
        p1, p2 = _two_orbs(Ia & ~Ja)
        q1, q2 = _two_orbs(Ja & ~Ia)
        ph = _rphase(Ia, Ja)
        return ph * (eri[p1, q2, p2, q1] - eri[p1, q1, p2, q2])

    if ea == 0 and eb == 2:                      # ----- double (beta-beta) -----
        p1, p2 = _two_orbs(Ib & ~Jb)
        q1, q2 = _two_orbs(Jb & ~Ib)
        ph = _rphase(Ib, Jb)
        return ph * (eri[p1, q2, p2, q1] - eri[p1, q1, p2, q2])

    # ----- double (alpha-beta) : Coulomb only, phases multiply -----
    pa = _trailz(Ia & ~Ja); qa = _trailz(Ja & ~Ia)
    pb = _trailz(Ib & ~Jb); qb = _trailz(Jb & ~Ib)
    return _rphase(Ia, Ja) * _rphase(Ib, Jb) * eri[pa, qa, pb, qb]


def ci_corr(dets, h1, eri):
    """Build H over the determinant list `dets` (dets[0] MUST be the HF
    reference) and diagonalise.  Returns (E_corr, eigvec) with the HF
    coefficient made positive and E_corr = lambda_min - <HF|H|HF>."""
    n = len(dets)
    H = np.zeros((n, n))
    for I in range(n):
        ia, ib = dets[I]
        for J in range(I, n):
            ja, jb = dets[J]
            v = melem(ia, ib, ja, jb, h1, eri)
            H[I, J] = v
            H[J, I] = v
    w, V = np.linalg.eigh(H)
    k = int(np.argmin(w))
    vec = V[:, k]
    if vec[0] < 0:
        vec = -vec
    return float(w[k] - H[0, 0]), vec


# =============================================================================
#  SECTION 3 - qiskit problem (Hamiltonian, tapering, excitations)
# =============================================================================

def build_qiskit_problem(molecule_xyz, basis, num_active_orbitals=3,
                         excitation_str=(1, 2), frozen_core='Y'):
    """Build the molecule object (qubit Hamiltonian, Z2-tapered Hamiltonian,
    JW/tapered excitation Pauli strings) via the user's pyscf_problem_extended.

    frozen_core : 'Y' freezes the chemical core, 'N' keeps all orbitals active.
    Returns the live `Molecule_simul_new_mapping` object (call it mol_A)."""
    mol_A = ppe.Molecule_simul_new_mapping(molecule_xyz, basis, 0, 0, MethodType.RHF)
    mol_A.problem_created(frozen_core=frozen_core,
                          active_space={'active': 'N',
                                        'num_active_orbitals': num_active_orbitals})
    mol_A.excitations_Dataframe(excitation_str=list(excitation_str))
    mol_A.problem_mapped_JW(list(excitation_str))
    mol_A.problem_reduction(reduction_in_place='inactive')
    mol_A.problem_parameters()
    return mol_A


def map_double_to_det(i, j, a, b, lbl, nocc, na, nb, ncore, M):
    """Convert a spin-orbital double (i,j -> a,b) into an active-space
    determinant (Ia, Ib) usable by the CI engine."""
    oa = set(range(na))
    ob = set(range(nb))
    for k in (i, j):                       # remove the two holes
        s = lbl[k]
        (oa if '\u03b1' in s else ob).discard(int(s[:-1]) - ncore)
    for k in (nocc + a, nocc + b):         # add the two particles
        s = lbl[k]
        (oa if '\u03b1' in s else ob).add(int(s[:-1]) - ncore)
    return (sum(1 << x for x in oa), sum(1 << x for x in ob))


def map_double_to_excitation_index(i, j, a, b, lbl, nocc, ncore, M, exc_lookup):
    """Find the qiskit excitation index whose (annihilation, creation) tuple
    matches this double.  qiskit spin-orbital indexing: alpha p -> p,
    beta p -> M + p (active indices)."""
    def qso(s):
        sp = int(s[:-1])
        return (sp - ncore) if ('\u03b1' in s) else M + (sp - ncore)
    ann = frozenset({qso(lbl[i]), qso(lbl[j])})
    cre = frozenset({qso(lbl[nocc + a]), qso(lbl[nocc + b])})
    return exc_lookup[(ann, cre)]


# =============================================================================
#  SECTION 4 - ranking by the MP1 (first-order) coefficient
# =============================================================================

def rank_doubles_pt2(df):
    """MODIFICATION 1.  Rank doubles by the first-order coefficient magnitude

         |c_i| = | <HF|H|D_i> / (E_HF - E_{D_i}) |
               = | g_ijab / -(eps_a+eps_b-eps_i-eps_j) |
               = | g_ijab | / D .

    Returns the DataFrame sorted most-important-first, plus the CCD sign column.
    """
    df = df.copy()
    df['pt2_coeff'] = np.abs(df['g_ijab'] / (-df['D']))   # |<HF|H|Di>/(E_HF-E_Di)|
    df['ccd_sign'] = np.sign(df['t2_ccsd']) * -1.0        # the shared sign vector
    return df.sort_values('pt2_coeff', ascending=False).reset_index(drop=True)


# =============================================================================
#  SECTION 5 - circuit energy / VQE / SQD  (fast, via expm_multiply)
# =============================================================================
#
#  Each excitation generator G_k = excitation_pauli_strings_map[idx] is a fixed
#  Hermitian operator; the gate is exp(-i t_k G_k).  We pre-build the sparse
#  matrices once and propagate the statevector with expm_multiply, so an energy
#  evaluation costs ~ms instead of re-synthesising 45 PauliEvolution gates.

def _generators(op_list, mapped, n):
    """Sparse  (-i G_k)  matrices for the first n selected excitations."""
    return [(-1j * op_list[mapped[k]].to_matrix(sparse=True)).tocsc()
            for k in range(n)]


def _propagate(psi0, gens, params):
    """Apply the product of exp(param_k * (-i G_k)) to |psi0>."""
    psi = psi0.copy()
    for k in range(len(params)):
        psi = expm_multiply(params[k] * gens[k], psi)
    return psi


def circuit_energy(params, psi0, gens, Hmat, shift):
    """<psi(params)| H |psi(params)>  +  shift   (shift = core+rep-E_HF)."""
    psi = _propagate(psi0, gens, params)
    return float(np.real(np.vdot(psi, Hmat @ psi))) + shift


def run_vqe(params0, psi0, gens, Hmat, shift, maxiter=120, ftol=1e-12, gtol=1e-9):
    """Variationally optimise the ansatz parameters (L-BFGS-B).
    `ftol`/`gtol` set the convergence threshold: loosen them (e.g. 1e-6) for a
    faster, approximate VQE; tighten for a fully converged one."""
    res = minimize(lambda p: circuit_energy(p, psi0, gens, Hmat, shift),
                   params0, method='L-BFGS-B',
                   options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol})
    return res.fun, res.x


# ---- matrix-free path (large systems): build the statevector by applying gates,
#      evaluate energy via SparsePauliOp.expectation_value; never form a 2^n x 2^n
#      operator.  Auto-selected when the qubit count is large (see build_union_context).

def _statevector_from_ops(C, ops, params, space):
    """Ansatz statevector = HartreeFock followed by exp(-i t_k G_k) gates, applied
    as a circuit (no operator matrices).  `ops` are SparsePauliOp generators."""
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import PauliEvolutionGate
    mapper = C['mol_A'].mapper_tap if space == 'tap' else JordanWignerMapper()
    hf = HartreeFock(C['M'], C['npart'], mapper)
    nq = hf.num_qubits
    qc = QuantumCircuit(nq)
    qc.compose(hf, inplace=True)
    for op, t in zip(ops, params):
        qc.append(PauliEvolutionGate(op, time=float(t)), range(nq))
    # Build the statevector by APPLYING gates (never densifying a 2^n operator).
    # Prefer Aer's C++ statevector simulator (fast for the long JW strings);
    # fall back to decomposing + qiskit Statevector if Aer is unavailable.
    try:
        from qiskit_aer import AerSimulator
        from qiskit import transpile
        qc2 = qc.copy(); qc2.save_statevector()
        sim = AerSimulator(method='statevector')
        res = sim.run(transpile(qc2, sim), shots=1).result()
        return np.asarray(res.get_statevector().data, dtype=complex)
    except Exception:
        return Statevector(qc.decompose(reps=2)).data.astype(complex)


def _propagate_ansatz(C, params, take, space):
    """Return the ansatz statevector for `params` (doubles then the `take` singles),
    in the tapered ('tap') or JW ('jw') space.  Dispatches on C['low_memory']:
    matrix-free circuit build for large systems, fast expm_multiply for small."""
    Nuse = len(params) - len(take)
    if C['low_memory']:
        idx_lst = ([C['mapped'][k] for k in range(Nuse)]
                   + [C['singles_idx'][t] for t in take])
        pool = C['em'] if space == 'tap' else C['em_jw']
        ops = [pool[i] for i in idx_lst]
        return _statevector_from_ops(C, ops, params, space)
    if space == 'tap':
        gens = [C['gens_tap'][k] for k in range(Nuse)] + [C['gens_tap_singles'][t] for t in take]
        psi0 = C['psi0']
    else:
        gens = [C['gens_jw'][k] for k in range(Nuse)] + [C['gens_jw_singles'][t] for t in take]
        psi0 = C['psi0_jw']
    return _propagate(psi0, gens, params)


def _energy_tap(C, psi, shift):
    """<psi|H_tapered|psi> + shift, matrix-free (expectation over Pauli terms) for
    large systems, dense matrix-vector for small ones."""
    if C['low_memory']:
        val = Statevector(psi).expectation_value(C['mol_A'].hamiltonian_tapering)
        return float(np.real(val)) + shift
    return float(np.real(np.vdot(psi, C['Hmat'] @ psi))) + shift


def run_sqd(params, psi0_jw, gens_jw, h1, eri, HFa, HFb, M, na, nb, threshold=1e-8):
    """Sample-based quantum diagonalisation.

    Propagate the ansatz in the (untapered) JW space, collect every basis state
    populated above `threshold`, turn each into an (Ia,Ib) determinant, then
    EXACTLY diagonalise H in that determinant subspace with the CI engine."""
    psi = _propagate(psi0_jw, gens_jw, params)
    probs = np.abs(psi) ** 2
    keep = np.where(probs > threshold)[0]
    seen = {(HFa, HFb)}
    dets = [(HFa, HFb)]
    for idx in keep:
        idx = int(idx)
        Ia = idx & ((1 << M) - 1)
        Ib = (idx >> M) & ((1 << M) - 1)
        if _popcnt(Ia) == na and _popcnt(Ib) == nb and (Ia, Ib) not in seen:
            seen.add((Ia, Ib))
            dets.append((Ia, Ib))
    E, _ = ci_corr(dets, h1, eri)
    return E, len(dets)


# =============================================================================
#  SECTION 6 - orchestrator
# =============================================================================

def run_selected_cc(molecule_xyz, basis='sto-6g', N=45, n_frozen=2,
                    num_active_orbitals=3, sqd_threshold=1e-8,
                    vqe_maxiter=120, verbose=True):
    """Run the whole pipeline for one molecule and one N.  Returns
    {N: {...the seven save_all_full energies...}} plus a details dict."""
    warnings.filterwarnings('ignore')

    # --- (a) qiskit problem: Hamiltonian, tapering, excitation strings ------
    mol_A = build_qiskit_problem(molecule_xyz, basis,
                                 num_active_orbitals=num_active_orbitals)
    M = mol_A.problem.num_spatial_orbitals
    npart = mol_A.problem.num_particles
    Htap = mol_A.hamiltonian_tapering
    core, rep = mol_A.core, mol_A.repulsion
    em = mol_A.excitation_pauli_strings_map        # tapered generators
    em_jw = mol_A.excitation_pauli_strings         # JW generators
    exc_lookup = {(frozenset(e[0]), frozenset(e[1])): k
                  for k, e in enumerate(mol_A.excitation_lst)}

    # --- (b) classical integrals (active-space, for the CI engine) ----------
    mol = gto.M(atom=molecule_xyz, basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol)
    mf.kernel()
    hf_e = mf.e_tot
    fock, g, nocc, nvirt, _, lbl, _, _ = build_spinorb_integrals_rhf(
        mol, mf, n_frozen_spatial=n_frozen)
    ncas = mol.nao - n_frozen
    nelecas = mol.nelectron - 2 * n_frozen
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1, _ = cas.get_h1eff()                          # effective 1e (frozen core folded)
    eri = ao2mo.restore(1, cas.get_h2eff(), ncas)    # chemist (pq|rs)
    na = nb = nelecas // 2
    HFa = (1 << na) - 1
    HFb = (1 << nb) - 1

    # --- (c) scan + MODIFICATION-1 ranking ----------------------------------
    df = scan_isolated_doubles(fock, g, nocc, nvirt, lbl,
                               frozenset(lbl[:nocc]), skip_zero=True)
    dfr = rank_doubles_pt2(df)
    n_avail = len(dfr)
    N = min(N, n_avail)
    rows = dfr.iloc[:N]

    # map each selected double -> excitation index AND -> CI determinant
    mapped, dets_order, dbl = [], [], []
    for r in rows.itertuples():
        i, j, a, b = int(r.i), int(r.j), int(r.a), int(r.b)
        mapped.append(map_double_to_excitation_index(i, j, a, b, lbl, nocc,
                                                      n_frozen, M, exc_lookup))
        dets_order.append(map_double_to_det(i, j, a, b, lbl, nocc, na, nb, n_frozen, M))
        dbl.append((i, j, a, b))
    ccd_signs = rows['ccd_sign'].values

    # --- (d) Selected CID and Selected CCD ----------------------------------
    CID, vec = ci_corr([(HFa, HFb)] + dets_order, h1, eri)
    ccd_e_corr, _, _ = SelectedCCSD(fock, g, nocc, nvirt,
                                    selected_singles=[], selected_doubles=dbl,
                                    so_labels=lbl).kernel()
    params_cid = np.abs(vec[1:]) * ccd_signs        # |coeff| * sign, gate order

    # --- (e) circuit / VQE (tapered space) ----------------------------------
    Hmat = Htap.to_matrix(sparse=True).tocsr()
    shift = core + rep - hf_e
    psi0 = Statevector(HartreeFock(M, npart, mol_A.mapper_tap)).data.astype(complex)
    gens = _generators(em, mapped, N)
    circuit_e_corr_CID = circuit_energy(params_cid, psi0, gens, Hmat, shift)
    vqe_e_corr, vqe_params = run_vqe(params_cid, psi0, gens, Hmat, shift,
                                     maxiter=vqe_maxiter)

    # --- (f) SQD (untapered JW space) ---------------------------------------
    psi0_jw = Statevector(HartreeFock(M, npart, JordanWignerMapper())).data.astype(complex)
    gens_jw = _generators(em_jw, mapped, N)
    sqd_e_corr, n_sqd = run_sqd(vqe_params, psi0_jw, gens_jw, h1, eri,
                                HFa, HFb, M, na, nb, threshold=sqd_threshold)

    # --- (g) disabled placeholder branches ----------------------------------
    placeholder = 0.0 + core + rep - hf_e

    save_all_full = {N: {
        'circuit_e_corr_CID': circuit_e_corr_CID,
        'circuit_e_corr_full_CCD': placeholder,
        'circuit_e_corr_CCD': placeholder,
        'vqe_e_corr': vqe_e_corr,
        'sqd_e_corr': sqd_e_corr,
        'ccd_e_corr': ccd_e_corr,
        'CID': CID,
    }}
    details = dict(mol_A=mol_A, mf=mf, dfr=dfr, mapped=mapped, dbl=dbl,
                   params_cid=params_cid, vqe_params=vqe_params,
                   n_available_doubles=n_avail, n_sqd_dets=n_sqd, hf_e=hf_e)
    if verbose:
        import pprint
        print(f"E(HF) = {hf_e:.10f}   available doubles = {n_avail}   N = {N}")
        print(f"SQD subspace size = {n_sqd} determinants (threshold {sqd_threshold:g})")
        pprint.pprint(save_all_full)
    return save_all_full, details


if __name__ == '__main__':
    N2 = ("N 0 0 0.03808486286764; "
          "N 0 0 1.16191513713236")
    run_selected_cc(N2, basis='sto-6g', N=45, n_frozen=2)


# =============================================================================
#  SECTION 7 - "union / summation" incremental-CI estimator
# =============================================================================
#
#  This reproduces the paper's  summation(index_inp, index_rotating)  and
#  fully_function_union(num_conf_lst, method)  on top of the validated CI engine
#  of Section 2.  Configurations are addressed by integer indices into a
#  `configs` registry whose entry 0 is the HF determinant (exactly like the
#  original "[0] + list(index_inp) + [i]" calls).

def ci_matrix(dets, h1, eri):
    """Dense CI Hamiltonian over a determinant list (dets[0] must be HF)."""
    n = len(dets)
    H = np.zeros((n, n))
    for I in range(n):
        ia, ib = dets[I]
        for J in range(I, n):
            ja, jb = dets[J]
            v = melem(ia, ib, ja, jb, h1, eri)
            H[I, J] = v
            H[J, I] = v
    return H


def _lambda_min(H):
    return float(np.linalg.eigvalsh(H)[0])


_POP16 = np.array([bin(i).count('1') for i in range(1 << 16)], dtype=np.int64)


def _popcount_arr(x):
    """Vectorised popcount for a numpy int array (orbital masks up to 32 bits)."""
    return _POP16[x & 0xFFFF] + _POP16[(x >> 16) & 0xFFFF]


def _ci_ground(dets, h1, eri, max_dense=1500):
    """Lowest eigenvalue of H over the determinant list `dets`.

    For small sets: dense `eigvalsh`.  For large sets (e.g. a full-variational SQD
    over thousands of configurations): build a SPARSE Hamiltonian -- only
    determinants differing by <=2 spin-orbitals couple (Slater-Condon), found by a
    vectorised popcount so we never scan pairs in pure Python beyond the coupled
    ones -- and get the ground state with a Lanczos solver.  Never forms a dense
    (n x n) matrix, so it stays within memory for large config sets."""
    n = len(dets)
    if n <= max_dense:
        return float(np.linalg.eigvalsh(ci_matrix(dets, h1, eri))[0])
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    ia = np.fromiter((d[0] for d in dets), dtype=np.int64, count=n)
    ib = np.fromiter((d[1] for d in dets), dtype=np.int64, count=n)
    rows = []; cols = []; vals = []
    for I in range(n):
        aI = int(ia[I]); bI = int(ib[I])
        rows.append(I); cols.append(I); vals.append(melem(aI, bI, aI, bI, h1, eri))
        if I + 1 < n:
            dd = _popcount_arr(ia[I] ^ ia[I + 1:]) + _popcount_arr(ib[I] ^ ib[I + 1:])
            for jj in np.nonzero(dd <= 4)[0]:          # differ by <=2 spin-orbitals
                J = I + 1 + int(jj)
                v = melem(aI, bI, int(ia[J]), int(ib[J]), h1, eri)
                if v != 0.0:
                    rows += [I, J]; cols += [J, I]; vals += [v, v]
    H = csr_matrix((vals, (rows, cols)), shape=(n, n))
    return float(eigsh(H, k=1, which='SA', return_eigenvectors=False)[0])


def summation(index_inp, index_rotating, configs, h1, eri, verbose=False):
    """CORE ROUTINE  (== paper's `summation`).

    Parameters
    ----------
    index_inp      : indices (into `configs`) of the N *base* configurations.
    index_rotating : indices of the configurations probed ONE AT A TIME.
    configs        : list; configs[0] = HF determinant, configs[k] = (Ia, Ib).

    Returns
    -------
    (S, lam0) with
        lam0 = lambda_min( CI over [HF] + index_inp )                 # "data_zero"
        S    = sum over i in index_rotating of
               [ lambda_min( CI over [HF] + index_inp + [i] ) - lam0 ]

    i.e. the additive sum of the per-configuration energy lowerings obtained by
    adding each rotating configuration to the N-configuration space.

    Speed: the (N+1)x(N+1) base matrix is built ONCE; every probe only borders it
    with a single extra row/column, so each increment costs one small eigvalsh
    plus N+1 matrix elements (no full rebuild)."""
    base = [configs[0]] + [configs[i] for i in index_inp]
    # Full-variational case (empty Pool M): only the base needs diagonalising.
    # Use the resilient (sparse when large) ground-state solver -> no dense S x S.
    if len(index_rotating) == 0:
        lam0 = _ci_ground(base, h1, eri)
        if verbose:
            print(f"    summation: base N={len(index_inp)}  rotating=0 "
                  f"(full variational)  lam0={lam0:.10f}")
        return 0.0, lam0
    Hb = ci_matrix(base, h1, eri)
    lam0 = _lambda_min(Hb)
    nb = len(base)
    S = 0.0
    for i in index_rotating:
        Ia, Ib = configs[i]
        border = np.empty(nb)
        for k in range(nb):
            bk = base[k]
            border[k] = melem(Ia, Ib, bk[0], bk[1], h1, eri)
        H = np.empty((nb + 1, nb + 1))
        H[:nb, :nb] = Hb
        H[:nb, nb] = border
        H[nb, :nb] = border
        H[nb, nb] = melem(Ia, Ib, Ia, Ib, h1, eri)
        S += _lambda_min(H) - lam0
    if verbose:
        print(f"    summation: base N={len(index_inp)}  rotating={len(index_rotating)}  "
              f"lam0={lam0:.10f}  S={S:.10f}")
    return S, lam0


def rank_pt2_external(base_dets, base_wf, base_energy, external, configs, h1, eri,
                      threshold=0.00005):
    """PT2 importance of each external configuration relative to the base CI
    wavefunction (analog of RANK_NEV_PT2 / simil_NEVPT2):

        c_mu = <Psi_base|H|mu> / (E_base - <mu|H|mu>)

    Returns a pandas Series of |c_mu| (index = config index) sorted descending,
    already filtered to |c_mu| > threshold."""
    imp = {}
    for mu in external:
        Ia, Ib = configs[mu]
        num = 0.0
        for k, d in enumerate(base_dets):
            num += base_wf[k] * melem(Ia, Ib, d[0], d[1], h1, eri)
        den = base_energy - melem(Ia, Ib, Ia, Ib, h1, eri)
        imp[mu] = (num / den) if den != 0.0 else 0.0
    s = abs(pd.Series(imp, dtype=float)).sort_values(ascending=False)
    return s[s > threshold]



def select_doubles_by_pt2(context, delta, order_by='energy', verbose=True):
    """Keep only the doubles whose contribution exceeds `delta`, most-important
    first, and prune/reorder the context IN PLACE so a subsequent
    run_sqd_union(context['n_doubles'], context)  (or any N <= it) uses exactly
    those doubles, in order.

    delta    : threshold. order_by='energy' -> compared to |E2_i| (the PT2
               second-order energy -g^2/D, in Hartree);
               order_by='coeff'  -> compared to the MP1 coefficient |g/(E_HF-E_D)|.
    order_by : 'energy' (default) or 'coeff'.
    Returns the number of doubles kept.
    """
    print('start selection')
    C = context
    dfr = C['dfr']
    score = (np.abs(dfr['E2_i'].values) if order_by == 'energy'
             else dfr['pt2_coeff'].values)
    keep = np.where(score > delta)[0]            # positions into the current arrays
    keep = keep[np.argsort(-score[keep])]        # order most-important first

    C['double_dets'] = [C['double_dets'][i] for i in keep]
    C['mapped']      = [C['mapped'][i]      for i in keep]
    C['dbl']         = [C['dbl'][i]         for i in keep]
    C['ccd_signs']   = np.asarray([C['ccd_signs'][i] for i in keep])
    if C.get('gens_tap') is not None:            # only present in the fast (matrix) mode
        C['gens_tap'] = [C['gens_tap'][i] for i in keep]
        C['gens_jw']  = [C['gens_jw'][i]  for i in keep]
    C['dfr'] = dfr.iloc[keep].reset_index(drop=True)
    C['n_doubles'] = int(len(keep))
    if verbose:
        print(f"select_doubles_by_pt2: kept {len(keep)}/{len(score)} doubles "
              f"({'|E2_i|' if order_by=='energy' else '|coeff|'} > {delta:g})")
    return int(len(keep))


def build_union_context(molecule_xyz, basis='sto-6g', n_frozen=2,
                        num_active_orbitals=3, low_memory=None, verbose=True):
    """Build (ONCE) every shared object the full union pipeline needs: the qiskit
    problem, the integrals, the MP1-ranked double determinants, and the sparse
    generator matrices for the tapered *and* JW ansatze (so `getting_save_all_dict`
    can slice `[:N]` and run circuit/VQE/SQD without rebuilding anything)."""
    warnings.filterwarnings('ignore')
    # qiskit freezes the chemical core when frozen_core='Y'; keep this consistent
    # with the classical n_frozen (n_frozen=0 -> no freezing on either side).
    mol_A = build_qiskit_problem(molecule_xyz, basis, num_active_orbitals=num_active_orbitals,
                                 frozen_core=('N' if n_frozen == 0 else 'Y'))
    M = mol_A.problem.num_spatial_orbitals
    npart = mol_A.problem.num_particles
    core, rep = getattr(mol_A, 'core', 0.0), mol_A.repulsion   # core=0 when frozen_core='N'
    exc_lookup = {(frozenset(e[0]), frozenset(e[1])): k
                  for k, e in enumerate(mol_A.excitation_lst)}

    mol = gto.M(atom=molecule_xyz, basis=basis, symmetry=True, verbose=0)
    mf = scf.RHF(mol); mf.max_cycle = 2000; mf.kernel(); hf_e = mf.e_tot
    # The number of spatial orbitals the qiskit problem actually froze; use THIS
    # for the classical side so the double->excitation map can never miss (which
    # is what caused KeyErrors when n_frozen disagreed with the qiskit core).
    n_frozen_eff = mol.nao - M
    if n_frozen_eff != n_frozen and verbose:
        print(f"note: qiskit froze {n_frozen_eff} core orbital(s); using that for the "
              f"classical side too (you passed n_frozen={n_frozen}).")
    n_frozen = n_frozen_eff
    fock, g, nocc, nvirt, _, lbl, _, _ = build_spinorb_integrals_rhf(mol, mf, n_frozen_spatial=n_frozen)
    ncas = mol.nao - n_frozen; nelecas = mol.nelectron - 2 * n_frozen
    cas = mcscf.CASCI(mf, ncas, nelecas)
    h1, _ = cas.get_h1eff(); eri = ao2mo.restore(1, cas.get_h2eff(), ncas)
    na = nb = nelecas // 2; HFa = (1 << na) - 1; HFb = (1 << nb) - 1
    HFdet = (HFa, HFb); shift = core + rep - hf_e

    df = scan_isolated_doubles(fock, g, nocc, nvirt, lbl, frozenset(lbl[:nocc]))
    dfr = rank_doubles_pt2(df)
    double_dets, mapped, dbl = [], [], []
    for r in dfr.itertuples():
        i, j, a, b = int(r.i), int(r.j), int(r.a), int(r.b)
        double_dets.append(map_double_to_det(i, j, a, b, lbl, nocc, na, nb, n_frozen, M))
        mapped.append(map_double_to_excitation_index(i, j, a, b, lbl, nocc, n_frozen, M, exc_lookup))
        dbl.append((i, j, a, b))
    n_doubles = len(double_dets)

    # pre-build sparse generators for tapered + JW spaces (all doubles, once)
    em = mol_A.excitation_pauli_strings_map
    em_jw = mol_A.excitation_pauli_strings
    # ---- singles: detect by rank, build their determinants (no matrices) ----
    singles_idx = [k for k, e in enumerate(mol_A.excitation_lst)
                   if len(e[0]) == 1 and em[k] is not None and em_jw[k] is not None]
    n_singles = len(singles_idx)
    single_dets = []
    for k in singles_idx:
        ann, cre = mol_A.excitation_lst[k]
        oa = set(range(na)); ob = set(range(nb))
        for so in ann:
            (oa if so < M else ob).discard(so if so < M else so - M)
        for so in cre:
            (oa if so < M else ob).add(so if so < M else so - M)
        single_dets.append((sum(1 << x for x in oa), sum(1 << x for x in ob)))

    # ---- representation: MATRIX-FREE for large qubit counts (avoid 2^n x 2^n
    #      operators, which is what broke the kernel for e.g. C2H2 all-electron) ----
    nq_jw = 2 * M
    if low_memory is None:
        low_memory = nq_jw > 16          # >16 qubits: 2^n x 2^n sparse operators get too big
    gens_tap = gens_jw = gens_tap_singles = gens_jw_singles = None
    Hmat = psi0 = psi0_jw = None
    if not low_memory:
        # fast path (small systems): pre-build sparse generator / Hamiltonian matrices
        gens_tap = [(-1j * em[mapped[k]].to_matrix(sparse=True)).tocsc() for k in range(n_doubles)]
        gens_jw = [(-1j * em_jw[mapped[k]].to_matrix(sparse=True)).tocsc() for k in range(n_doubles)]
        gens_tap_singles = [(-1j * em[k].to_matrix(sparse=True)).tocsc() for k in singles_idx]
        gens_jw_singles = [(-1j * em_jw[k].to_matrix(sparse=True)).tocsc() for k in singles_idx]
        Hmat = mol_A.hamiltonian_tapering.to_matrix(sparse=True).tocsr()
        psi0 = Statevector(HartreeFock(M, npart, mol_A.mapper_tap)).data.astype(complex)
        psi0_jw = Statevector(HartreeFock(M, npart, JordanWignerMapper())).data.astype(complex)

    C = dict(mol_A=mol_A, M=M, npart=npart, h1=h1, eri=eri, shift=shift, HFdet=HFdet,
             fock=fock, g=g, nocc=nocc, nvirt=nvirt, lbl=lbl, dbl=dbl,
             double_dets=double_dets, ccd_signs=dfr['ccd_sign'].values, mapped=mapped,
             gens_tap=gens_tap, gens_jw=gens_jw, Hmat=Hmat, psi0=psi0, psi0_jw=psi0_jw,
             na=na, nb=nb, n_doubles=n_doubles, hf_e=hf_e, dfr=dfr,
             singles_idx=singles_idx, n_singles=n_singles,
             gens_tap_singles=gens_tap_singles, gens_jw_singles=gens_jw_singles,
             single_dets=single_dets, low_memory=low_memory, em=em, em_jw=em_jw)
    if verbose:
        print(f"context: {n_doubles} doubles, {n_singles} singles, {nq_jw} JW qubits, "
              f"{'matrix-free (low-memory)' if low_memory else 'sparse-matrix (fast)'} mode")
    return C


def _pt2_select_singles(base_dets, base_wf, base_energy, single_dets, h1, eri, thresh=1e-8):
    """(A) Select single-excited configurations by their PT2 second-order energy
    relative to the base (doubles-CI) wavefunction:

        E2(s) = <Psi_base|H|s>^2 / (E_base - <s|H|s>)

    Keep every single with |E2(s)| > `thresh` (i.e. non-negligible; singles that
    are decoupled by symmetry give ~0 and are dropped).  Returns the kept indices
    into `single_dets`."""
    keep = []
    for si, d in enumerate(single_dets):
        num = 0.0
        for k, b in enumerate(base_dets):
            num += base_wf[k] * melem(d[0], d[1], b[0], b[1], h1, eri)
        den = base_energy - melem(d[0], d[1], d[0], d[1], h1, eri)
        e2 = (num * num / den) if den != 0.0 else 0.0
        if abs(e2) > thresh:
            keep.append(si)
    return keep


def getting_save_all_dict(N, C, threshold1=1e-3, threshold2=1e-8,
                          use_vqe=True, vqe_maxiter=120, vqe_tol=None, verbose=False,
                          include_singles=None, singles_init=0.05223385,
                          compute_energies=True):
    """FULL per-N pipeline (== the paper's `getting_save_all_dict`).

    Builds the top-N-doubles ansatz, evaluates the circuit energy, optionally
    optimises it (VQE), computes the Selected CCD and CID, then propagates the
    ansatz statevector in the JW space and reads off the SQD-populated
    configuration sets at BOTH thresholds (`configs1` = tighter, `configs2` =
    looser).  `sqd_e_corr` is the exact diagonalisation over `configs2`.

    VQE controls:
      use_vqe   : if False, skip the optimisation entirely and use the fixed CID
                  parameters for the statevector (fast; `vqe_e_corr` == circuit).
      vqe_maxiter : cap on L-BFGS-B iterations.
      vqe_tol   : convergence threshold (both ftol and gtol).  None -> tight
                  (1e-12).  Set e.g. 1e-6 for a fast, loosely-converged VQE.

    include_singles : None | 'all' | 'half' | int | 'pt2'
        'all'/'half'/int  -> append the first n singles with a fixed init parameter
                             (system-specific count, de-duplicated vs the doubles).
        'pt2' -> (A) PT2-select the singles that couple to the doubles-CI
                 wavefunction (keep |E2| > 1e-8), (B) build a CISD over the
                 selected doubles + those singles, and (C) take the ansatz
                 parameters from that CISD eigenvector: doubles keep their CCD
                 sign, singles all get sign +1.  A distinct methodology, testable
                 alongside the others.
        Faithful replacement for the original `list(np.arange(0,20,1))`, which
        appended the first 20 SINGLE-excitation operators to the ansatz.  Here
        the number of singles is read from the excitation list (system-specific,
        never hardcoded): 'all' -> every single, 'half' -> n_singles//2,
        int k -> the first k.  The chosen singles are set-differenced against the
        selected double indices, so an operator can NEVER enter the ansatz twice
        (this is the double-counting guard the hardcoded `20` lacked).  Each
        appended single starts from `singles_init` (0 gives a pure-doubles ansatz
        as the VQE initial point).  Default None reproduces the doubles-only
        results exactly."""
    Nuse = min(N, C['n_doubles'])
    base_dets = [C['HFdet']] + C['double_dets'][:Nuse]
    E_cid, vec = ci_corr(base_dets, C['h1'], C['eri'])

    # ---- build the ansatz parameters + the singles `take` list (mode-dependent) ----
    if include_singles == 'pt2':
        # (A) PT2-select singles vs the doubles-CI wavefunction (keep |E2| > 1e-8)
        lam0_D = _lambda_min(ci_matrix(base_dets, C['h1'], C['eri']))
        sel = set(C['mapped'][:Nuse])
        cand = [t for t in range(C['n_singles']) if C['singles_idx'][t] not in sel]
        keep = _pt2_select_singles(base_dets, vec, lam0_D,
                                   [C['single_dets'][t] for t in cand],
                                   C['h1'], C['eri'], thresh=1e-8)
        take = [cand[t] for t in keep]
        # (B) CISD over [HF] + selected doubles + PT2-selected singles
        cisd_dets = base_dets + [C['single_dets'][t] for t in take]
        E_cisd, vec_cisd = ci_corr(cisd_dets, C['h1'], C['eri'])
        # (C) signs: doubles keep their CCD sign, singles all get +1
        p_d = np.abs(vec_cisd[1:1 + Nuse]) * C['ccd_signs'][:Nuse]
        p_s = np.abs(vec_cisd[1 + Nuse:]) * 1.0
        params = np.concatenate([p_d, p_s])
        if verbose:
            print(f"  [singles='pt2'] {len(take)}/{len(cand)} singles kept (|E2|>1e-8); "
                  f"CISD(base) E_corr={E_cisd:.6f}; singles sign=+1")
    elif include_singles is not None:
        # ---- 'all' / 'half' / int : first-ns singles, params = fixed init ----
        params = np.abs(vec[1:]) * C['ccd_signs'][:Nuse]
        if include_singles == 'all':
            ns = C['n_singles']
        elif include_singles == 'half':
            ns = C['n_singles'] // 2
        else:
            ns = min(int(include_singles), C['n_singles'])
        sel = set(C['mapped'][:Nuse])
        take = [t for t in range(ns) if C['singles_idx'][t] not in sel]
        params = np.concatenate([params, np.full(len(take), float(singles_init))])
        if verbose:
            print(f"  + appended {len(take)} singles (of {ns} requested, "
                  f"n_singles={C['n_singles']}), de-duplicated vs the doubles")
    else:
        take = []
        params = np.abs(vec[1:]) * C['ccd_signs'][:Nuse]

    # ---- circuit energy + optional VQE (tapered space; matrix-free-aware) ----
    # `compute_energies=False` skips the tapered statevector entirely -- useful for
    # large systems when only the SQD configuration set is needed (run_sqd_union).
    if compute_energies:
        def _tap_energy(p):
            return _energy_tap(C, _propagate_ansatz(C, p, take, 'tap'), C['shift'])
        circuit_e_corr = _tap_energy(params)
        if use_vqe:
            _tol = 1e-12 if vqe_tol is None else float(vqe_tol)
            res = minimize(_tap_energy, params, method='L-BFGS-B',
                           options={'maxiter': vqe_maxiter, 'ftol': _tol, 'gtol': _tol})
            vqe_e_corr, vqe_params = float(res.fun), res.x
        else:
            vqe_e_corr, vqe_params = circuit_e_corr, params
    else:
        circuit_e_corr = None
        vqe_e_corr, vqe_params = None, params

    ccd_e_corr, _, _ = SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                    selected_singles=[], selected_doubles=C['dbl'][:Nuse],
                                    so_labels=C['lbl']).kernel()

    # ---- SQD: JW-space ansatz statevector -> populated configurations ----
    psi = _propagate_ansatz(C, vqe_params, take, 'jw')
    probs = np.abs(psi) ** 2
    M, na, nb = C['M'], C['na'], C['nb']

    def collect(thr):
        out = set()
        for idx in np.where(probs > thr)[0]:
            idx = int(idx)
            Ia = idx & ((1 << M) - 1); Ib = (idx >> M) & ((1 << M) - 1)
            if _popcnt(Ia) == na and _popcnt(Ib) == nb:
                out.add((Ia, Ib))
        return out

    configs2 = collect(threshold2)                      # looser / larger set
    configs1 = collect(threshold1)                      # tighter / smaller set
    big_list = [C['HFdet']] + [d for d in configs2 if d != C['HFdet']]
    hf_diag = melem(C['HFdet'][0], C['HFdet'][1], C['HFdet'][0], C['HFdet'][1], C['h1'], C['eri'])
    sqd_e_corr = _ci_ground(big_list, C['h1'], C['eri']) - hf_diag   # resilient (sparse if large)

    if verbose:
        ce = 'n/a' if circuit_e_corr is None else f"{circuit_e_corr:.6f}"
        ve = 'n/a' if vqe_e_corr is None else f"{vqe_e_corr:.6f}"
        print(f"  N={N}: CID={E_cid:.6f} CCD={ccd_e_corr:.6f} circ={ce} "
              f"vqe={ve} sqd={sqd_e_corr:.6f} "
              f"|configs1|={len(configs1)} |configs2|={len(configs2)}")
    return dict(CID=E_cid, ccd_e_corr=ccd_e_corr, circuit_e_corr_CID=circuit_e_corr,
                vqe_e_corr=vqe_e_corr, sqd_e_corr=sqd_e_corr,
                configs1=configs1, configs2=configs2,
                base_dets=base_dets, base_wf=vec, vqe_params=vqe_params,
                params=params, take=take)


def input_conf(save_all_N, doubles_dets, HFdet):
    """Build the per-N configuration registry and probing index sets from the
    SQD-populated sets (== the paper's `input_conf`).

    Returns (configs, doubles_idx, probing_D, small_idx, probing_small) where the
    *_idx / probing_* are index lists into `configs` (configs[0] = HF)."""
    big = save_all_N['configs2']
    small = save_all_N['configs1']
    dset = set(doubles_dets)
    configs = [HFdet] + list(doubles_dets)
    seen = set(configs)
    for d in sorted(big):
        if d not in seen:
            seen.add(d); configs.append(d)
    idx_of = {d: k for k, d in enumerate(configs)}
    doubles_idx = [idx_of[d] for d in doubles_dets]
    probing_D = [idx_of[d] for d in big if d != HFdet and d not in dset]
    small_idx = [idx_of[d] for d in small if d != HFdet and d in idx_of]
    probing_small = [idx_of[d] for d in big if d != HFdet and d not in small]
    return configs, doubles_idx, probing_D, small_idx, probing_small


def fully_function_union(num_conf_lst, method='D', context=None,
                         threshold1=1e-3, threshold2=1e-8, pt2_threshold=0.00005,
                         use_vqe=True, vqe_tol=None, include_singles=None,
                         singles_init=0.05223385, verbose=True):
    """FULL pipeline (== the paper's `fully_function_union`).

    For every N:
      1. `getting_save_all_dict(N)` runs circuit -> VQE -> SQD and returns the two
         SQD-populated configuration sets.
      2. `input_conf` turns them into the base doubles + probing index sets.
      3. `method='D'`: PT2-rank the probing_D set against the base CID wavefunction
         and move the important configs (|c|>pt2_threshold) into the base;
         otherwise use the small-threshold split.
      4. `summation(index_inp, index_rotating)` -> (S, lam0).
      5. value_full_{N}_CID = (S + lam0 + shift)*627.51  [kcal/mol]
         value_full_{N}_CCD = (S + ccd_e_corr)*627.51.
    """
    if context is None:
        raise ValueError("pass context=build_union_context(...)")
    HART2KCAL = 627.5094740631
    C = context; h1, eri, shift, HFdet = C['h1'], C['eri'], C['shift'], C['HFdet']
    save_all_full = {}
    for N in num_conf_lst:
        sd = getting_save_all_dict(N, C, threshold1=threshold1, threshold2=threshold2,
                                   use_vqe=use_vqe, vqe_tol=vqe_tol, verbose=verbose,
                                   include_singles=include_singles, singles_init=singles_init)
        save_all_full[N] = sd
        doubles_dets = sd['base_dets'][1:]
        configs, doubles_idx, probing_D, small_idx, probing_small = \
            input_conf(sd, doubles_dets, HFdet)

        base_dets = [HFdet] + [configs[k] for k in doubles_idx]
        _, base_wf = ci_corr(base_dets, h1, eri)
        base_raw = _lambda_min(ci_matrix(base_dets, h1, eri))

        if method == 'D':
            promoted = rank_pt2_external(base_dets, base_wf, base_raw, probing_D,
                                         configs, h1, eri, threshold=pt2_threshold)
            add = list(promoted.index)
            index_inp = doubles_idx + add
            index_rotating = [k for k in probing_D if k not in set(add)]
        else:
            index_inp = small_idx
            index_rotating = probing_small

        S, lam0 = summation(index_inp, index_rotating, configs, h1, eri, verbose=verbose)
        value_CID = (S + lam0 + shift) * HART2KCAL
        value_CCD = (S + sd['ccd_e_corr']) * HART2KCAL
        save_all_full[f'value_full_{N}_CID'] = value_CID
        save_all_full[f'value_full_{N}_CCD'] = value_CCD
        if verbose:
            print(f"  -> value_full_{N}_CID = {value_CID:.5f} kcal/mol   "
                  f"value_full_{N}_CCD = {value_CCD:.5f} kcal/mol\n")
    return save_all_full


def shots_for_floor(floor):
    """Number of measurement shots implied by an AMPLITUDE floor.

    `floor` cuts on |c|, so the smallest retained probability is floor**2 and a
    determinant sitting exactly at the floor needs ~1/floor**2 shots to be seen
    even once (and, by the coupon-collector bound, is missed ~37% of the time at
    exactly that budget).  Use this to label results honestly."""
    return 1.0 / float(floor) ** 2


def _populated_dets(psi, M, na, nb, floor, n_shots=None, seed=0):
    """Return {(Ia,Ib): |c|} for the configurations this run 'observes'.

    n_shots is None  -> DETERMINISTIC: every basis state with |c| > floor.
                        This is the noiseless infinite-shot ceiling, not SQD.
    n_shots is an int -> REAL SAMPLING: draw n_shots outcomes from |c|**2 and keep
                        the distinct ones.  `floor` is ignored; the shot budget is
                        the only knob, exactly as on hardware."""
    amp = np.abs(np.asarray(psi).ravel())
    if n_shots is None:
        idx_iter = np.where(amp > floor)[0]
    else:
        p = amp ** 2
        p = p / p.sum()
        rng = np.random.default_rng(seed)
        # sample the SET of observed outcomes exactly: P(seen) = 1-(1-p)^S
        pseen = -np.expm1(int(n_shots) * np.log1p(-np.clip(p, 0.0, 1.0 - 1e-18)))
        idx_iter = np.where(rng.random(p.size) < pseen)[0]
    det_amp = {}
    for idx in idx_iter:
        idx = int(idx)
        Ia = idx & ((1 << M) - 1); Ib = (idx >> M) & ((1 << M) - 1)
        if _popcnt(Ia) == na and _popcnt(Ib) == nb:
            det_amp[(Ia, Ib)] = amp[idx]
    return det_amp


def map_single_to_det(i, a, lbl, nocc, na, nb, ncore):
    """Spin-orbital single (i -> a) as an active-space determinant (Ia, Ib).
    Singles analogue of map_double_to_det."""
    oa = set(range(na)); ob = set(range(nb))
    s = lbl[i]
    (oa if '\u03b1' in s else ob).discard(int(s[:-1]) - ncore)
    s = lbl[nocc + a]
    (oa if '\u03b1' in s else ob).add(int(s[:-1]) - ncore)
    return (sum(1 << x for x in oa), sum(1 << x for x in ob))


def cc_ansatz_params(N, C, include_singles=None, verbose=False):
    """Ansatz parameters from Selected UCCD / UCCSD AMPLITUDES (t1, t2)
    instead of Selected CID / CISD eigenvector coefficients.

      include_singles=None   -> s-UCCD  : params = -t2[i,j,a,b] over selected doubles
      include_singles='pt2'  -> s-UCCSD : identical PT2 singles selection as the CI
                                path, then params = [-t2 doubles] + [-t1 singles]

    Sign convention verified empirically against the CI path: the CI parameters
    agree in sign with +t2 (0% agreement with -t2), and <Psi|H|Psi> is sensible
    only for +t2.  So params = +t2 (and +t1), i.e. the CI coefficient MAGNITUDE
    is replaced by the CC amplitude magnitude while the sign structure is kept.

    Returns (params, take, E_cc) where `take` indexes C['single_dets'] exactly as
    getting_save_all_dict does, so the result drops straight into
    _propagate_ansatz(C, params, take, 'jw').
    """
    Nuse = min(N, C['n_doubles'])
    dbl = [tuple(map(int, t)) for t in C['dbl'][:Nuse]]
    take, sing = [], []
    if include_singles == 'pt2':
        base_dets = [C['HFdet']] + C['double_dets'][:Nuse]
        _, vec = ci_corr(base_dets, C['h1'], C['eri'])
        lam0_D = _lambda_min(ci_matrix(base_dets, C['h1'], C['eri']))
        sel = set(C['mapped'][:Nuse])
        cand = [t for t in range(C['n_singles']) if C['singles_idx'][t] not in sel]
        keep = _pt2_select_singles(base_dets, vec, lam0_D,
                                   [C['single_dets'][t] for t in cand],
                                   C['h1'], C['eri'], thresh=1e-8)
        take = [cand[t] for t in keep]
        # map each selected single's determinant back to its (i, a) spin-orbital pair
        nocc, nvirt = C['nocc'], C['nvirt']
        # infer the frozen-core offset by reproducing a known double's determinant
        ncore = None
        for trial in range(0, 8):
            i0, j0, a0, b0 = dbl[0]
            if map_double_to_det(i0, j0, a0, b0, C['lbl'], nocc,
                                 C['na'], C['nb'], trial, C['M']) == C['double_dets'][0]:
                ncore = trial; break
        if ncore is None:
            raise RuntimeError("could not infer frozen-core offset from context")
        lut = {}
        for i in range(nocc):
            for a in range(nvirt):
                lut[map_single_to_det(i, a, C['lbl'], nocc, C['na'], C['nb'], ncore)] = (i, a)
        for t in take:
            d = C['single_dets'][t]
            if d not in lut:
                raise KeyError(f"single determinant {d} has no (i,a) preimage")
            sing.append(lut[d])

    E_cc, t1, t2 = SelectedCCSD(C['fock'], C['g'], C['nocc'], C['nvirt'],
                                selected_singles=sing, selected_doubles=dbl,
                                so_labels=C['lbl']).kernel()
    p_d = np.array([t2[i, j, a, b] for (i, j, a, b) in dbl], float)
    p_s = np.array([t1[i, a] for (i, a) in sing], float)
    params = np.concatenate([p_d, p_s]) if len(p_s) else p_d
    if verbose:
        tag = 's-UCCSD' if include_singles == 'pt2' else 's-UCCD'
        print(f"  [{tag}] {len(dbl)} doubles + {len(sing)} singles, "
              f"E_corr(sel-CC) = {E_cc:.6f} Ha; "
              f"|t2|max={np.abs(p_d).max():.3e}"
              + (f", |t1|max={np.abs(p_s).max():.3e}" if len(p_s) else ""))
    return params, take, float(E_cc)


def run_sqd_union(N, context, threshold_N=1e-2, floor=1e-8, use_vqe=False,
                  include_singles='all', singles_init=0.05223385,
                  full_variational=False, n_shots=None, seed=0, verbose=True):
    """EXPLICIT 5-step SQD-union estimator (Pool N Selected-CI + Pool M perturbation).

    This spells out the flow that is otherwise the `method != 'D'` branch of
    `fully_function_union`:

      1. FIND DOUBLES + SINGLES + their parameters.
         The ansatz is the selected doubles (parameters = |CID amplitude| * CCD
         sign) plus singles chosen by `include_singles` (see getting_save_all_dict;
         'pt2' uses PT2-selected singles + a CISD base + singles sign +1).  With
         `use_vqe` these are re-optimised; by default they are kept fixed.
      2. START SQD -> FIND ALL CONFIGURATIONS (populated down to `floor`).
      3. DIVIDE BY THRESHOLD -> POOL N (|amp| > threshold_N) and POOL M
         (floor < |amp| <= threshold_N).
      4. POOL N -> NEW SELECTED CI (exact diagonalisation -> lambda_0).
      5. POOL M -> PERTURBATION via `summation`.

    full_variational : if True, put EVERY populated configuration into Pool N and
        leave Pool M empty (Pool N Selected-CI over all configs, no perturbation).
        Equivalent to setting threshold_N = floor.  Returns E_corr = lambda_0+shift.

    Returns (E_corr, details) with E_corr = lambda_0 + shift + S  (Hartree)."""
    C = context
    h1, eri, shift, HF = C['h1'], C['eri'], C['shift'], C['HFdet']
    M, na, nb = C['M'], C['na'], C['nb']
    Nuse = min(N, C['n_doubles'])
    if full_variational:
        threshold_N = floor          # all populated configs -> Pool N, Pool M empty

    # --- step 1: build ansatz (doubles + singles) and its parameters ---
    sd = getting_save_all_dict(Nuse, C, use_vqe=use_vqe, include_singles=include_singles,
                               singles_init=singles_init, threshold2=floor,
                               compute_energies=use_vqe,   # skip circuit energy if not optimising
                               verbose=verbose)
    # --- step 2: SQD statevector -> populated configurations (matrix-free aware) ---
    psi = _propagate_ansatz(C, sd['vqe_params'], sd['take'], 'jw')
    det_amp = _populated_dets(psi, M, na, nb, floor, n_shots=n_shots, seed=seed)
    if verbose:
        if n_shots is None:
            print(f"  [deterministic |c|>{floor:g} -> smallest p kept = {floor**2:g}; "
                  f"equivalent shot budget ~{shots_for_floor(floor):.1e}]")
        else:
            print(f"  [sampled {int(n_shots):.1e} shots -> {len(det_amp)} distinct configurations]")

    # --- step 3: split into Pool N (strong) and Pool M (weak) ---
    amp_floor = floor if n_shots is None else 0.0
    poolN = [d for d, a in det_amp.items() if a > threshold_N and d != HF]
    poolM = [d for d, a in det_amp.items() if amp_floor < a <= threshold_N and d != HF]

    # --- steps 4 & 5: Selected CI over Pool N, perturbation over Pool M ---
    configs = [HF] + poolN + poolM
    ix = {d: k for k, d in enumerate(configs)}
    S, lam0 = summation([ix[d] for d in poolN], [ix[d] for d in poolM],
                        configs, h1, eri)
    E_selCI = lam0 + shift                 # Pool N Selected-CI correlation energy
    E_corr = E_selCI + S                    # + Pool M perturbation
    details = dict(poolN=len(poolN), poolM=len(poolM), lam0=lam0, S=S,
                   E_selCI=E_selCI, sqd_full=sd['sqd_e_corr'],
                   E_corr_kcal=E_corr * 627.5094740631)
    if verbose:
        print(f"N={N}: |PoolN|={len(poolN)} |PoolM|={len(poolM)}  "
              f"E_selCI(PoolN)={E_selCI:.8f}  +perturbation S={S:.8f}")
        print(f"  E_corr(union) = {E_corr:.11f} Ha = {E_corr*627.5094740631:.4f} kcal/mol"
              f"   (full SQD over all configs = {sd['sqd_e_corr']:.8f})")
    return E_corr, details
