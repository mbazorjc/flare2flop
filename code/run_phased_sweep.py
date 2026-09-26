#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_phased_sweep.py
===================
Parallel driver for the Paper 2 phased flare-to-compute sweep. Fans out
independent (facility x phase x CAPEX multiplier x discount rate) solves of
phased_compute_model, reconciles every one against analytical_benchmark, and
checkpoints so the job is resumable.

    python run_phased_sweep.py --smoke                  # 1 facility, 1 node
    python run_phased_sweep.py                          # full grid
    python run_phased_sweep.py --phases 3 4 --resume    # SMR phases, resume

-------------------------------------------------------------------------------
*** THE GUROBI LICENCE IS THE BINDING CONSTRAINT.  READ BEFORE SCALING UP. ***
-------------------------------------------------------------------------------
Paper 1's own sweep script carries the line

    MAX_JOBS=2    # Gurobi license limit -- DO NOT raise

and `rerun_failed.sh` exists because 50 cells died on "token.gurobi.com
unreachable". That history sets the design here.

The brief asked for "N workers x 4 threads" and a Slurm ARRAY job. Taken
literally those two things destroy each other: an array of 50 tasks means 50
processes racing for 2 licence sessions, 48 of which fail on token checkout --
which is precisely how the previous 50 dead cells were produced. Widening the
array makes the failure worse, not the throughput better.

This driver therefore enforces concurrency in two places:

  1. In-process, a ProcessPoolExecutor capped at --workers (default 2).
  2. Cluster-wide, a FILESYSTEM SEMAPHORE under --sem-dir. Every solve acquires
     a lease file before touching Gurobi and releases it after. Array tasks on
     different nodes see the same directory on shared storage, so total
     concurrency stays at --license-sessions no matter how wide the array is.
     Leases carry a heartbeat and are reclaimed if a holder dies, so a crashed
     task cannot deadlock the sweep.

Raise --license-sessions ONLY after confirming the entitlement with
`gurobi_cl --license` on a compute node. The default of 2 matches Paper 1.

Threads: the brief says 4; Paper 1 used GUROBI_THREADS=32. With only 2 sessions
live, 2 x 4 = 8 cores are busy on a node that probably has 64. Unless the
licence permits more sessions, prefer FEWER, FATTER solves: --threads 16 or 32
finishes the same grid sooner. --threads 4 is kept as the default because it is
what was asked for, and the trade-off is printed at startup.

-------------------------------------------------------------------------------
SCALE -- read this before submitting the full grid
-------------------------------------------------------------------------------
37 facilities x 4 phases x 3 CAPEX x 3 IR = 1,332 nodes. At the 3,600 s
per-solve limit and 2 concurrent sessions the worst case is 666 hours, about 28
days. At a more typical 20 minutes per solve it is roughly 9 days.

Two things cut that sharply:
  * --phases 3 4 halves the grid and covers every node where an SMR is actually
    built. Phases 1-2 have no reactor and no firm data-centre load, so their
    results vary with flare volume rather than with facility identity; running
    all 37 facilities through them reproduces the same answer 37 times.
  * --capex-mults 1.0 --discount-rates 0.12 runs the 37-node central case first.
    Confirm the science there before committing a fortnight of wall time. That
    is the baseline-first discipline run_sweep1.sh already recommends.

Nodes are NEVER interpolated. Technology selection is an integer decision that
changes across the grid, so every node is solved independently -- the same rule
Paper 1 states for its CAPEX x IR surface.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import shutil
import signal
import socket
import sys
import tempfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import analytical_benchmark as ab          # noqa: E402
import data_inputs as di                   # noqa: E402
import phased_compute_model as pcm         # noqa: E402

# Row schema. Order is the CSV column order and must not be reshuffled -- the
# post-processing layer and Paper 1's extract.py conventions both key on it.
ROW_COLUMNS = [
    "node_id", "facility_id", "country", "phase", "capex_mult", "discount_rate",
    "technology", "n_modules", "served_mwhe", "total_cost",
    "breakeven_unified", "breakeven_analytic", "co2_avoided_t", "profit",
    # provenance and diagnostics beyond the requested schema
    "breakeven_conv_A", "served_btc_mwhe", "served_comm_mwhe", "generation_mwhe",
    "gen_gas_mwhe", "demand_mwhe", "it_mw", "avail_quantiles", "gas_quantiles",
    "n_windows", "n_timelimit", "worst_rel_gap", "optimality",
    "rev_btc", "rev_dc", "rev_comm", "rev_carbon", "rev_penalty",
    "gate_F", "gate_rel_err", "solver", "source", "wall_s", "status", "error",
    # B-16: endogenous technology selection (Eq. D20) -- audit trail
    "tech_n_candidates", "tech_runner_up", "tech_cost_margin_pct", "tech_mode",
    # cost split -- required by uq_phased.py, which re-annualises AnnCap per
    # Monte-Carlo draw and must hold the other three terms fixed. Without the
    # split the UQ has to reconstruct AnnCap from the catalogue, which is a
    # second implementation of the same arithmetic and a place for them to
    # disagree silently.
    "cost_anncap", "cost_fixed", "cost_vom", "cost_startup",
    "overnight_capex", "nameplate_mw",
    # B-17: service check. A build that serves nothing is not a cheap build,
    # it is an infeasible one; these columns make that auditable per node.
    "unmet_mwhe", "served_frac", "tech_rejected_unserved",
    # D-17: Eq. D20 selects on minimum COST; the paper reports BREAKEVEN, which
    # nets revenue. Carbon revenue scales with generation, so a costlier design
    # can post a lower breakeven and the two criteria can disagree. These
    # columns make that an empirical question instead of an argument.
    "tech_be_best_name", "tech_be_best_value", "tech_criterion_gap",
    "tech_criterion_conflict",
]


# =============================================================================
# SECTION 1 -- CLUSTER-WIDE LICENCE SEMAPHORE
# =============================================================================

class LicenseSemaphore:
    """Filesystem lease semaphore, safe across nodes of an array job.

    Each acquire creates one lease file in `sem_dir` via O_CREAT|O_EXCL, which is
    atomic on POSIX and on the shared filesystems Slurm clusters use. A lease
    carries the holder's host, pid and a heartbeat timestamp. Leases older than
    `stale_s` without a heartbeat are reclaimed, so a task killed by the time
    limit cannot strand a slot for the rest of the sweep.

    This is what makes a Slurm ARRAY job safe against the 2-session licence cap.
    Without it, array width and licence entitlement are silently in conflict and
    the extra tasks die on token checkout.
    """

    def __init__(self, sem_dir: str, slots: int, stale_s: int = 7200,
                 poll_s: float = 5.0, timeout_s: Optional[float] = None):
        self.dir = sem_dir
        self.slots = max(1, int(slots))
        self.stale_s = stale_s
        self.poll_s = poll_s
        self.timeout_s = timeout_s
        os.makedirs(self.dir, exist_ok=True)
        self._held: Optional[str] = None

    def _reap(self) -> None:
        # NEVER let a filesystem error here kill the sweep. On shared storage an
        # unlink can fail for reasons that have nothing to do with this process:
        # a stale NFS handle, a lease owned by another user's array task, a
        # read-only remount. A licence semaphore that can crash a multi-day job
        # is a worse failure than a slot that takes a little longer to free.
        now = time.time()
        try:
            entries = os.listdir(self.dir)
        except OSError as e:
            sys.stderr.write(f"[sem] cannot list {self.dir}: {e}\n")
            return
        for fn in entries:
            if not fn.endswith(".lease"):
                continue
            p = os.path.join(self.dir, fn)
            try:
                age = now - os.path.getmtime(p)
                if age > self.stale_s:
                    os.unlink(p)
                    sys.stderr.write(f"[sem] reclaimed stale lease {fn} (age {age:.0f}s)\n")
            except FileNotFoundError:
                pass
            except OSError as e:
                sys.stderr.write(f"[sem] could not reclaim {fn}: {e}\n")

    def acquire(self) -> str:
        t0 = time.time()
        while True:
            self._reap()
            for i in range(self.slots):
                p = os.path.join(self.dir, f"slot{i}.lease")
                try:
                    fd = os.open(p, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                    with os.fdopen(fd, "w") as fh:
                        json.dump({"host": socket.gethostname(), "pid": os.getpid(),
                                   "t": time.time()}, fh)
                    self._held = p
                    return p
                except FileExistsError:
                    continue
                except OSError as e:               # permissions, quota, stale handle
                    sys.stderr.write(f"[sem] cannot create {p}: {e}\n")
                    continue
            if self.timeout_s is not None and time.time() - t0 > self.timeout_s:
                raise TimeoutError(
                    f"no Gurobi licence slot after {self.timeout_s:.0f}s "
                    f"({self.slots} slots in {self.dir})")
            time.sleep(self.poll_s * (0.5 + random.random()))   # jitter, avoid lockstep

    def heartbeat(self) -> None:
        if self._held:
            try:
                os.utime(self._held, None)
            except OSError:
                pass

    def release(self) -> None:
        """Free the slot. Must not raise -- a failed release that propagates
        would both lose the slot forever AND take down the worker.

        If the lease cannot be unlinked, backdate its mtime so the reaper treats
        it as stale on the next acquire. That turns a permanent slot leak into a
        delay bounded by `stale_s`. Only if BOTH fail is the slot genuinely lost,
        and that is logged loudly rather than swallowed.
        """
        p, self._held = self._held, None
        if not p:
            return
        try:
            os.unlink(p)
            return
        except FileNotFoundError:
            return
        except OSError as e:
            sys.stderr.write(f"[sem] unlink failed for {p}: {e}; backdating instead\n")
        try:
            os.utime(p, (0, 0))          # instantly stale -> reclaimed next acquire
        except OSError as e:
            sys.stderr.write(
                f"[sem] CANNOT RELEASE {p}: {e}. This slot is lost until the lease "
                f"is removed by hand or ages past stale_s={self.stale_s}s.\n")

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc):
        self.release()
        return False


MIQP_CAPABLE = ("gurobi", "gurobi_direct", "cplex", "cplex_direct", "scip")


LICENCE_FREE = ("appsi_highs", "highs", "cbc", "glpk", "scip")


def preflight_solver(preferred: str = "appsi_highs", verbose: bool = True) -> dict:
    """Find a usable solver and say plainly which, before burning wall time.

    Rewritten 2026-09-15. The previous version checked Gurobi and nothing else,
    so an environment with a perfectly good free solver installed was told it had
    no licence and refused to start.

    Solvers in LICENCE_FREE need no licence, no token server and no seat. When
    one of those is selected the concurrency semaphore is pointless and is
    switched off automatically -- workers can then equal cores.
    """
    out = {"solver": None, "licence_free": False, "checked": [], "detail": ""}
    try:
        import pyomo.environ as _pe
    except ImportError:
        out["detail"] = ("Pyomo is not installed. `pip install pyomo highspy` "
                         "gives you both the modelling layer and a free solver.")
        return out

    order = [preferred] + [s for s in
                           ("appsi_highs", "highs", "gurobi", "gurobi_direct",
                            "cplex", "cplex_direct", "cbc", "scip", "glpk")
                           if s != preferred]
    for name in order:
        try:
            o = _pe.SolverFactory(name)
            avail = o is not None and o.available(exception_flag=False)
        except Exception as e:                       # noqa: BLE001
            out["checked"].append(f"{name}: error ({type(e).__name__})")
            continue
        out["checked"].append(f"{name}: {'available' if avail else 'not available'}")
        if avail:
            # Gurobi can report "available" and still fail to start an
            # environment when the licence is bad -- the exact failure that
            # produced "Unauthorized access" on tanshi-Super-Server. Probe it.
            if "gurobi" in name:
                try:
                    import gurobipy as gp
                    env = gp.Env(empty=True); env.setParam("OutputFlag", 0)
                    env.start(); env.dispose()
                except Exception as e:               # noqa: BLE001
                    out["checked"][-1] += f"  (licence check FAILED: {e})"
                    continue
            out["solver"] = name
            out["licence_free"] = name in LICENCE_FREE
            out["detail"] = (f"using {name}" +
                             ("  — licence-free, no concurrency cap"
                              if out["licence_free"] else
                              "  — licensed solver, concurrency capped"))
            return out
    out["detail"] = ("no usable solver found. `pip install highspy` is the "
                     "shortest path — HiGHS needs no licence.")
    return out


# =============================================================================
# SECTION 2 -- NODE ENUMERATION
# =============================================================================

@dataclass(frozen=True)
class Node:
    facility_id: str
    country: str
    it_mw: float
    phase: int
    capex_mult: float
    discount_rate: float

    @property
    def node_id(self) -> str:
        """Stable id. Region prefix follows Paper 1's convention (KE/NGA/SA) so
        the output tree is greppable with the same patterns extract.py uses."""
        reg = {"Kenya": "KE", "Nigeria": "NGA", "South Africa": "SA"}.get(
            self.country, self.country[:3].upper())
        return (f"{reg}_ph{self.phase}_cap{self.capex_mult}_ir{self.discount_rate}"
                f"-fac{self.facility_id}")


def enumerate_nodes(phases: Sequence[int], capex_mults: Sequence[float],
                    discount_rates: Sequence[float], min_mw: float = 2.0,
                    limit: Optional[int] = None) -> List[Node]:
    """Facility list from data_inputs (the 37 above 2 MW) crossed with the grid."""
    fac = di.load_facilities(min_mw=min_mw, verbose=False)
    nodes = []
    for _, r in fac.iterrows():
        for ph in phases:
            for cm in capex_mults:
                for ir in discount_rates:
                    nodes.append(Node(str(r["FACILITY_ID"]), str(r["country"]),
                                      float(r["it_mw"]), int(ph), float(cm), float(ir)))
    nodes.sort(key=lambda n: (n.phase, -n.it_mw, n.capex_mult, n.discount_rate))
    return nodes[:limit] if limit else nodes


def select_reactor(it_mw: float, pue: float = 1.5) -> str:
    """Initial technology guess for the sizing pass.

    The MIQP re-selects endogenously (Eq. D20); this only seeds it. Picking the
    smallest catalogue design whose nameplate covers the load keeps the initial
    relaxation tight. NEVER invent a reactor -- every candidate comes from
    baseiaea.csv (hard convention #11).
    """
    cat = di.load_smr_catalogue().sort_values("Power in MWe")
    need = it_mw * pue
    fit = cat[cat["Power in MWe"] >= need]
    return str((fit.index[0] if len(fit) else cat.index[-1]))


def feasible_reactors(it_mw: float, pue: float = 1.5,
                      hard_module_cap: int = 20,
                      reliability_margin: float = 1.2) -> List[str]:
    """Candidate set for Eq. D20 technology selection.

    A design g is admissible iff its fleet at the module cap can meet peak firm
    demand with the reliability margin:

        cap_e[g] * min(MaxModules[g], hard_module_cap) >= margin * demand

    which is exactly Paper 1's capacity-adequacy constraint (Eq. D19) applied
    before the solve. Designs it excludes are infeasible in the MILP too, so
    pruning them changes no optimum -- it only avoids solving subproblems whose
    feasible set is empty.

    Every candidate comes from baseiaea.csv (hard convention #11). If no design
    covers the load even at the cap, the largest is returned, matching the
    clamp already applied by phased_compute_model.max_modules().
    """
    cat = di.load_smr_catalogue()
    need = it_mw * pue * reliability_margin
    out = []
    for name, r in cat.iterrows():
        cap = float(r["Power in MWe"])
        mx = min(int(r["Max Modules"]), hard_module_cap)
        if cap * mx >= need:
            out.append(str(name))
    if not out:
        out = [str(cat["Power in MWe"].idxmax())]
    return sorted(out, key=lambda n: float(cat.loc[n, "Power in MWe"]))


# =============================================================================
# SECTION 3 -- ONE NODE
# =============================================================================

def _blank_row(node: Node) -> Dict:
    r = {c: None for c in ROW_COLUMNS}
    r.update(node_id=node.node_id, facility_id=node.facility_id,
             country=node.country, phase=node.phase,
             capex_mult=node.capex_mult, discount_rate=node.discount_rate)
    return r


def solve_node(node: Node, opts: Dict) -> Dict:
    """Solve one grid node. Runs inside a worker process.

    Returns a plain dict so nothing solver-specific has to be pickled back.
    Every failure mode is caught and reported as a row with status != "ok";
    a worker must never take the pool down.
    """
    t0 = time.time()
    row = _blank_row(node)
    sem = None
    workdir = None
    try:
        # --- isolated scratch, per Paper 1's run_one() ---------------------
        # Concurrent solves in the same directory collide on Gurobi log files
        # and any scratch the model writes. The model only READS the catalogue
        # and facility data, but the isolation is cheap and removes the class
        # of bug entirely.
        workdir = tempfile.mkdtemp(prefix=f".wd_{node.node_id}_", dir=opts["scratch"])
        os.chdir(workdir)

        phases = pcm.PhaseConfig()
        sc = pcm.SolverConfig(solver=opts["solver"], mip_gap=opts["mip_gap"],
                              time_limit=opts["time_limit"], threads=opts["threads"],
                              tee=False)

        # phase -> a timestep inside that phase
        t_start = {1: 0, 2: phases.t2, 3: phases.t3, 4: phases.t4}[node.phase]

        # ==================================================================
        # B-16 / Eq. D20 -- ENDOGENOUS TECHNOLOGY SELECTION
        # ==================================================================
        # Paper 1 Eq. D20 selects one design with a one-hot binary vGenSel[g]
        # and switches that design's whole constraint block on with big-M. The
        # feasible set of that MILP is the DISJOINT UNION of the single-design
        # feasible sets -- no variable couples two designs, because exactly one
        # plant is built. Minimising over a union equals the minimum of the
        # per-set minima, so
        #
        #     min over the one-hot MILP  ==  min over g of (single-design solve)
        #
        # EXACTLY, not approximately. Enumerating the candidate set therefore
        # implements D20 without touching the verified single-design model, so
        # gate F and the Paper 1 regression are unchanged by construction.
        # Cost is one solve per admissible design instead of one per node.
        #
        # In phases where the phase economics build no reactor (smr_scale = 0)
        # there is no technology to select: the objective is provably invariant
        # to cfg.reactor -- verified 2026-09-17, total_cost identical to the
        # cent across eVinci / CAREM-25 / Rolls-Royce-SMR in Phase 1. Those
        # phases run one solve and report technology "none", instead of the
        # phantom reactor label the pre-B-16 sweep carried.
        smr_in_phase = pcm.PhaseConfig.PHASE_ECONOMICS[node.phase]["smr_scale"] > 0.0
        if smr_in_phase:
            candidates = feasible_reactors(node.it_mw)
            tech_mode = "D20_enumerated"
        else:
            candidates = [opts["reactor_for"][node.facility_id]]
            tech_mode = "no_smr_in_phase"

        def _mk(rname):
            # B-21: economic parameters come from the register bridge, so editing
            # data_inputs.csv actually reaches the solver. Node-level fields are
            # set AFTER the overrides so the grid axes always win -- capex_mult
            # and discount_rate are swept, never read from the CSV.
            return pcm.SiteConfig(
                **opts["site_overrides"],
                facility_id=node.facility_id,
                it_mw=node.it_mw,
                reactor=rname,
                capex_mult=node.capex_mult,
                discount_rate=node.discount_rate,
                subordinate_mining=opts["subordinate_mining"],
                quadratic=False,                   # never force the MIQP here
                ramp_penalty_mode=opts["ramp_penalty_mode"],
            )

        # --- licence lease: held only around the solves -------------------
        if opts["use_semaphore"]:
            sem = LicenseSemaphore(opts["sem_dir"], opts["license_sessions"],
                                   stale_s=opts["time_limit"] * 2 + 600,
                                   timeout_s=opts["sem_timeout"])
            sem.acquire()

        trials, failed = [], []
        for rname in candidates:
            c = _mk(rname)
            try:
                l_, _d, g_ = pcm.solve_site(
                    c, phases, sc, n_days=opts["n_days"], t_start=t_start,
                    force_reference=opts["force_reference"], verbose=False)
            except Exception as ex:                # noqa: BLE001
                # An individual design may be infeasible at this node. That is
                # vGenSel[g] = 0, not a node failure.
                failed.append(f"{rname}: {type(ex).__name__}")
                continue
            trials.append((float(l_.total_cost), rname, l_, g_, c))

        if sem:
            sem.release(); sem = None

        if not trials:
            raise RuntimeError("no admissible reactor solved; "
                               + "; ".join(failed[:6]))

        # ==================================================================
        # B-17 -- A TRIAL THAT DOES NOT SERVE THE LOAD IS NOT IN D20's
        #         FEASIBLE SET, SO IT CANNOT WIN ON COST
        # ==================================================================
        # Found 2026-09-17 in the v5 grid: 108 of 333 Phase-3 nodes selected
        # Aurora (1.5 MWe) with generation EXACTLY zero, served EXACTLY zero,
        # and full capital charged. Every non-Aurora row served correctly.
        #
        # The mechanism is a mismatch between two things that are both correct
        # on their own. `Ledger.total_cost` is the Eq. D3 aggregate -- capital,
        # fixed, VOM, start-up. The model carries NO value-of-lost-load term
        # (it relies on the sizing-pass adequacy constraint instead), so a
        # configuration whose unit commitment cannot follow the load simply
        # turns every module off: it pays no VOM and no start-up, and its D3
        # aggregate is therefore SMALLER than that of a plant which actually
        # runs. Ranking trials on that aggregate rewards exactly the build that
        # fails. The earlier sweeps never exposed this because technology was
        # pre-assigned by scale and the degenerate configuration was never
        # offered as a candidate.
        #
        # Eq. D20 selects within the feasible set of the full model, and that
        # set includes the load balance. So the fix is not a tie-break or a
        # penalty weight -- it is to state the feasible set correctly: trials
        # that leave firm load unserved are REJECTED, and the minimum cost is
        # taken over the remainder.
        #
        # Phases whose demand is supply-limited (dc_share < 1) have vUnmet
        # fixed to zero by construction (see B-15) and cannot fail this test,
        # which is correct -- there is no firm load there to fall short of.
        def _serves(l_):
            need = float(l_.served_dc) + float(l_.unmet)
            if need <= 0.0:
                return bool(smr_in_phase is False or l_.served_dc > 0.0)
            return float(l_.unmet) <= 0.005 * need and float(l_.served_dc) > 0.0

        firm_phase = pcm.PhaseConfig.PHASE_ECONOMICS[node.phase][
            "dc_share_of_available"] >= 1.0
        serving = [t for t in trials if _serves(t[2])] if firm_phase else trials
        rejected = len(trials) - len(serving)
        if firm_phase and not serving:
            # Nothing admissible can actually serve this facility. That is a
            # reportable finding, not a node to quietly fill with the cheapest
            # do-nothing build.
            trials.sort(key=lambda z: z[0])
            _c, _n, led, gate, cfg = trials[0]
            row.update(status="gate_fail", gate_F="FAIL",
                       tech_mode="D20_no_serving_candidate",
                       tech_n_candidates=len(trials),
                       tech_rejected_unserved=rejected,
                       unmet_mwhe=float(led.unmet), served_frac=0.0,
                       technology=led.technology, n_modules=led.modules,
                       wall_s=round(time.time() - t0, 2),
                       error=("no admissible design serves the firm load; "
                              f"{len(trials)} candidates all left load unmet"))
            return row

        # ---- D-17 instrumentation, before the cost sort decides anything ---
        # Compute what the selection WOULD have been under the reported metric,
        # so the divergence can be measured on the real grid rather than argued
        # about. This costs nothing: every trial is already solved.
        _be_pairs = []
        for _c, _n, _l, _g, _cf in serving:
            _b = _l.breakeven()
            if _b is not None and np.isfinite(_b):
                _be_pairs.append((float(_b), _n))
        _be_pairs.sort()
        be_best_name = _be_pairs[0][1] if _be_pairs else None
        be_best_val = _be_pairs[0][0] if _be_pairs else None

        serving.sort(key=lambda z: z[0])           # Eq. D20 objective: min cost
        best_cost, best_name, led, gate, cfg = serving[0]
        runner_up = serving[1][1] if len(serving) > 1 else None
        margin_pct = (round(100.0 * (serving[1][0] - best_cost) / best_cost, 4)
                      if len(serving) > 1 and best_cost > 0 else None)
        _need = float(led.served_dc) + float(led.unmet)

        be = led.breakeven()
        _gap = (float(be) - be_best_val
                if (be is not None and np.isfinite(be) and be_best_val is not None)
                else None)
        row.update(
            tech_be_best_name=be_best_name, tech_be_best_value=be_best_val,
            tech_criterion_gap=_gap,
            tech_criterion_conflict=(bool(be_best_name is not None
                                          and be_best_name != best_name)),
            unmet_mwhe=float(led.unmet),
            served_frac=(float(led.served_dc) / _need if _need > 0 else None),
            tech_rejected_unserved=rejected,
            tech_n_candidates=len(serving), tech_runner_up=runner_up,
            tech_cost_margin_pct=margin_pct, tech_mode=tech_mode,
            technology=(led.technology if smr_in_phase else "none"),
            n_modules=(led.modules if smr_in_phase else 0),
            served_mwhe=led.served_dc, total_cost=led.total_cost,
            breakeven_unified=be,
            breakeven_analytic=gate.get("analytic_be"),
            co2_avoided_t=led.co2_avoided_t,
            profit=led.profit(cfg.diesel_price),
            breakeven_conv_A=led.breakeven_total_served(),
            served_btc_mwhe=led.served_btc, served_comm_mwhe=led.served_comm,
            generation_mwhe=led.generation,
            gen_gas_mwhe=getattr(led, "gen_gas_mwhe", float("nan")),   # B-21
            n_windows=getattr(led, "n_windows", 0),                    # B-22
            n_timelimit=getattr(led, "n_timelimit", 0),                # B-22
            worst_rel_gap=getattr(led, "worst_rel_gap", 0.0),          # B-22
            optimality=("timelimit" if getattr(led, "n_timelimit", 0)
                        else "proven"),                                # B-22
            demand_mwhe=getattr(led, "demand_mwhe", float("nan")),     # B-21
            it_mw=node.it_mw,
            avail_quantiles=getattr(led, "avail_quantiles", ""),        # B-21
            gas_quantiles=getattr(led, "gas_quantiles", ""),            # B-21
            rev_btc=led.rev_btc, rev_dc=led.rev_dc, rev_comm=led.rev_comm,
            rev_carbon=led.rev_carbon, rev_penalty=led.rev_penalty,
            cost_anncap=led.anncap, cost_fixed=led.fixed, cost_vom=led.vom,
            cost_startup=led.startup,
            overnight_capex=led.overnight_capex_charged,
            nameplate_mw=led.nameplate_mw,
            gate_F="PASS" if gate["passed"] else "FAIL",
            gate_rel_err=gate.get("rel_err"),
            solver=opts["solver"], source=led.source,
            wall_s=round(time.time() - t0, 2),
            status="ok" if gate["passed"] else "gate_fail",
            error=None if gate["passed"] else json.dumps(
                [c for c in gate["checks"] if not c["ok"] and not c["skipped"]])[:900],
        )
    except Exception as e:                       # noqa: BLE001 -- workers never die
        row.update(status="error", wall_s=round(time.time() - t0, 2),
                   error=f"{type(e).__name__}: {e}"[:900],
                   solver=opts.get("solver"), gate_F="ERROR")
        row["_traceback"] = traceback.format_exc()[-2000:]
    finally:
        if sem:
            sem.release()
        try:
            os.chdir(opts["home"])
        except Exception:
            pass
        if workdir and os.path.isdir(workdir):
            shutil.rmtree(workdir, ignore_errors=True)
    return row


# =============================================================================
# SECTION 4 -- CHECKPOINT
# =============================================================================

class Checkpoint:
    """Resumable node ledger.

    Parquet is the primary format as requested. pyarrow is not guaranteed on
    every cluster python, so a CSV mirror is always written and is what the
    resume path falls back to. Losing resumability to a missing optional
    dependency would be a poor trade on a multi-day job.
    """

    def __init__(self, path: str, verbose: bool = True):
        self.parquet = path
        self.csv = os.path.splitext(path)[0] + ".csv"
        self.rows: List[Dict] = []
        self.verbose = verbose
        self._engine = None
        try:
            import pyarrow  # noqa: F401
            self._engine = "pyarrow"
        except ImportError:
            try:
                import fastparquet  # noqa: F401
                self._engine = "fastparquet"
            except ImportError:
                self._engine = None
                if verbose:
                    sys.stderr.write(
                        "[ckpt] neither pyarrow nor fastparquet found; "
                        "checkpointing to CSV only\n")

    def load_done(self) -> set:
        for p, rd in ((self.parquet, pd.read_parquet), (self.csv, pd.read_csv)):
            if os.path.exists(p):
                try:
                    df = rd(p)
                    self.rows = df.to_dict("records")
                    done = set(df.loc[df["status"] == "ok", "node_id"].astype(str))
                    if self.verbose:
                        print(f"  [resume] {len(df)} rows in {os.path.basename(p)}, "
                              f"{len(done)} completed OK")
                    return done
                except Exception as e:
                    sys.stderr.write(f"[ckpt] could not read {p}: {e}\n")
        return set()

    def add(self, row: Dict) -> None:
        self.rows.append(row)

    def flush(self) -> None:
        if not self.rows:
            return
        df = pd.DataFrame(self.rows)
        for c in ROW_COLUMNS:
            if c not in df.columns:
                df[c] = None
        df = df[[c for c in ROW_COLUMNS if c in df.columns]
                + [c for c in df.columns if c not in ROW_COLUMNS]]
        tmp = self.csv + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, self.csv)                 # atomic, survives a mid-write kill
        if self._engine:
            try:
                df.to_parquet(self.parquet, index=False, engine=self._engine)
            except Exception as e:
                sys.stderr.write(f"[ckpt] parquet write failed ({e}); CSV is current\n")


# =============================================================================
# SECTION 5 -- SWEEP
# =============================================================================

def shard(nodes: List[Node], task_id: int, n_tasks: int) -> List[Node]:
    """Round-robin shard for a Slurm array. Round-robin rather than contiguous
    so every task gets a mix of large and small facilities and the array
    finishes evenly instead of one task carrying all the hyperscale sites."""
    if n_tasks <= 1:
        return nodes
    return [n for i, n in enumerate(nodes) if i % n_tasks == task_id]


def run_sweep(args) -> int:
    home = os.getcwd()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "nodes"), exist_ok=True)
    scratch = args.scratch or tempfile.mkdtemp(prefix="phased_scratch_")
    os.makedirs(scratch, exist_ok=True)

    # ---- banner and preflight ------------------------------------------
    print("=" * 96)
    print("  Paper 2 phased sweep -- run_phased_sweep.py")
    print("=" * 96)
    print(f"  host {socket.gethostname()}  python {platform.python_version()}  "
          f"pid {os.getpid()}")
    print(f"  array task {args.task_id + 1}/{args.n_tasks}   outdir {args.outdir}")

    pf = preflight_solver(args.solver)
    print(f"  solver preflight : {pf['detail']}")
    for line in pf["checked"]:
        print(f"                     {line}")
    ok = pf["solver"] is not None
    if not ok and not args.allow_no_license:
        print("\n  REFUSING TO START. No usable solver. `pip install highspy` is the")
        print("  shortest path -- HiGHS needs no licence at all. Or pass")
        print("  --allow-no-license to run the solver-free reference path (valid")
        print("  accounting, but no technology or module selection).")
        return 2
    if ok:
        args.solver = pf["solver"]
        # A quadratic objective on a solver that cannot take one fails deep inside
        # the Pyomo interface with "expressions of degree None" -- which is simply
        # how the HiGHS interface reports "not linear", because it calls Pyomo's
        # repn with quadratic=False. Catch it here, where the message can say what
        # to do about it, instead of 1,332 identical tracebacks later.
        if args.ramp_penalty_mode == "quadratic" and \
                not any(k in args.solver for k in MIQP_CAPABLE):
            print(f"\n  NOTE: --ramp-penalty-mode quadratic needs an MIQP-capable solver")
            print(f"        ({', '.join(MIQP_CAPABLE)}), and this run has {args.solver}.")
            print(f"        Falling back to 'pwl', which represents the SAME quadratic")
            print(f"        penalty by tangent cuts and solves as a MILP. The Eq. D10")
            print(f"        formulation is unchanged; see ramp_penalty_diagnostics().")
            args.ramp_penalty_mode = "pwl"
        if pf["licence_free"] and not args.no_semaphore:
            # No seat cap, so the cluster-wide semaphore has nothing to protect.
            args.no_semaphore = True
            if args.workers <= 2:
                print(f"  NOTE: {pf['solver']} needs no licence, so the 2-session cap that "
                      f"shaped this\n        layer does not apply. Raise --workers to your "
                      f"core count; the\n        semaphore is disabled automatically.")
    force_ref = args.force_reference or (not ok)
    if force_ref:
        print("  MODE: solver-free reference dispatch (no technology/module selection)")

    if (not pf.get("licence_free")) and args.threads < 8 and args.workers * args.threads < 16:
        print(f"\n  NOTE: {args.workers} workers x {args.threads} threads = "
              f"{args.workers * args.threads} cores busy. Paper 1 used 32 threads per")
        print("  solve. With the licence capped at 2 sessions, fewer/fatter solves")
        print("  finish the same grid sooner -- consider --threads 16 or 32.")

    # ---- parameter provenance (B-21) -------------------------------------
    # Printed before a single solve, and written beside the results, so the run
    # log itself records which numbers rest on measurement and which still rest
    # on a placeholder. A reader should never have to diff two CSVs to find out
    # what a headline number was computed with.
    try:
        prov = di.override_provenance(args.param_set)
        os.makedirs(args.outdir, exist_ok=True)
        prov.to_csv(os.path.join(args.outdir, "parameter_provenance.csv"), index=False)
        chg = prov[prov["changed"]] if len(prov) else prov
        print(f"\n  parameter set    : {args.param_set}"
              f"   ({len(prov)} bridged, {len(chg)} differ from the v7 design case)")
        for _, r in chg.iterrows():
            print(f"      {r['siteconfig_field']:<22} {r['design_value']:>12,.4f}"
                  f"  ->{r['applied_value']:>12,.4f}   [{r['status']}]")
        still = prov[prov["status"].astype(str).str.upper().str.contains("PLACEHOLDER")]
        if len(still):
            print("      STILL PLACEHOLDER: "
                  + ", ".join(sorted(still["register_key"].astype(str))))
    except Exception as _ex:                                   # noqa: BLE001
        print(f"  parameter provenance unavailable: {type(_ex).__name__}: {_ex}")

    # ---- nodes ----------------------------------------------------------
    nodes = enumerate_nodes(args.phases, args.capex_mults, args.discount_rates,
                            limit=args.limit)
    total_all = len(nodes)
    nodes = shard(nodes, args.task_id, args.n_tasks)
    print(f"  ramp penalty     : {args.ramp_penalty_mode}"
          + ("  (quadratic shape, MILP form)" if args.ramp_penalty_mode == "pwl" else ""))
    print(f"\n  grid: {len(args.phases)} phase(s) x {len(args.capex_mults)} CAPEX x "
          f"{len(args.discount_rates)} IR")
    print(f"  {total_all} nodes total, {len(nodes)} on this task")

    est_h = len(nodes) * (args.time_limit / 3600.0) / max(1, args.workers)
    print(f"  worst-case wall time on this task: {est_h:,.0f} h "
          f"({est_h/24:,.1f} d) at the {args.time_limit}s per-solve limit")
    if est_h > args.walltime_hours and not args.smoke:
        print(f"  WARNING: that exceeds the {args.walltime_hours} h Slurm limit. The")
        print("  checkpoint makes this safe -- resubmit with --resume and it continues")
        print("  from where it stopped -- but widen the array or trim the grid.")

    ckpt = Checkpoint(os.path.join(args.outdir, "checkpoint_%02d.parquet" % args.task_id))
    done = ckpt.load_done() if args.resume else set()
    todo = [n for n in nodes if n.node_id not in done]
    print(f"  {len(todo)} to solve, {len(nodes) - len(todo)} already complete\n")
    if not todo:
        print("  nothing to do.")
        return 0

    reactor_for = {n.facility_id: select_reactor(n.it_mw) for n in todo}
    opts = dict(solver=args.solver, mip_gap=args.mip_gap, time_limit=args.time_limit,
                threads=args.threads, n_days=args.n_days, scratch=scratch, home=home,
                force_reference=force_ref, subordinate_mining=not args.no_subordination,
                ramp_penalty_mode=args.ramp_penalty_mode, reactor_for=reactor_for,
                use_semaphore=not args.no_semaphore and not force_ref,
                sem_dir=args.sem_dir, license_sessions=args.license_sessions,
                sem_timeout=args.sem_timeout,
                site_overrides=di.site_overrides(args.param_set),
                param_set=args.param_set)

    # ---- graceful stop ---------------------------------------------------
    state = {"stop": False}

    def _sig(signum, _frame):
        state["stop"] = True
        print(f"\n  signal {signum} received: no new nodes will be dispatched; "
              "in-flight solves finish and the checkpoint is written.")
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(s, _sig)
        except (ValueError, OSError):
            pass

    # ---- execute ---------------------------------------------------------
    t0 = time.time()
    n_ok = n_gate = n_err = 0
    gate_failures: List[Dict] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        pending = {}
        it = iter(todo)
        for _ in range(min(args.workers, len(todo))):
            n = next(it)
            pending[ex.submit(solve_node, n, opts)] = n
        completed = 0
        while pending:
            for fut in as_completed(list(pending), timeout=None):
                node = pending.pop(fut)
                try:
                    row = fut.result()
                except Exception as e:           # pool-level failure
                    row = _blank_row(node)
                    row.update(status="error", gate_F="ERROR",
                               error=f"pool: {type(e).__name__}: {e}"[:900])
                completed += 1

                tb = row.pop("_traceback", None)
                ckpt.add(row)
                # one CSV per node, as specified
                pd.DataFrame([{k: row.get(k) for k in ROW_COLUMNS}]).to_csv(
                    os.path.join(args.outdir, "nodes", f"{node.node_id}.csv"), index=False)

                st = row["status"]
                if st == "ok":
                    n_ok += 1
                elif st == "gate_fail":
                    n_gate += 1
                    gate_failures.append(row)
                else:
                    n_err += 1

                be = row.get("breakeven_unified")
                be_s = f"{be:9.2f}" if isinstance(be, (int, float)) and be == be else "      n/a"
                print(f"  [{completed:>5}/{len(todo)}] {st:<10} {node.node_id:<42} "
                      f"be {be_s}  {row.get('wall_s', 0):7.1f}s  "
                      f"gate {row.get('gate_F')}")
                if tb:
                    sys.stderr.write(f"--- traceback {node.node_id} ---\n{tb}\n")

                if completed % args.flush_every == 0:
                    ckpt.flush()

                # ---- the analytic-reconciliation halt ---------------------
                # "Log any failure; do not proceed." With a worker pool that
                # means: stop dispatching, let in-flight solves land, checkpoint,
                # exit non-zero. Killing mid-solve would lose finished work and
                # leak a licence lease.
                if st == "gate_fail" and args.halt_on_gate_fail:
                    print("\n  *** GATE F FAILED -- HALTING ***")
                    print(f"      node {node.node_id}")
                    print(f"      {row.get('error')}")
                    print("      No further nodes will be dispatched. Results so far are")
                    print("      checkpointed and resumable once the cause is fixed.")
                    state["stop"] = True

                if not state["stop"]:
                    try:
                        nx = next(it)
                        pending[ex.submit(solve_node, nx, opts)] = nx
                    except StopIteration:
                        pass
                break                            # re-enter as_completed on the new set

    ckpt.flush()
    dt = time.time() - t0

    # ---- summary ---------------------------------------------------------
    print("\n" + "=" * 96)
    print("  SWEEP SUMMARY")
    print("=" * 96)
    print(f"  solved {n_ok} ok, {n_gate} gate failures, {n_err} errors in "
          f"{dt/3600:.2f} h ({dt/max(1,n_ok+n_gate+n_err):.1f} s/node)")
    print(f"  checkpoint: {ckpt.csv}" + (f" and {ckpt.parquet}" if ckpt._engine else ""))

    df = pd.DataFrame(ckpt.rows)

    # ---- B-22 optimality audit -------------------------------------------
    # A sweep that silently mixes proven optima with abandoned incumbents is not
    # a sweep anyone should publish from. State the split before any headline
    # statistic is printed, not in a log file nobody opens.
    try:
        if len(df) and "optimality" in df.columns:
            n_tl = int((df["optimality"] == "timelimit").sum())
            if n_tl:
                g = pd.to_numeric(df["worst_rel_gap"], errors="coerce").fillna(0.0)
                print(f"\n  !! OPTIMALITY: {n_tl} of {len(df)} nodes hit the "
                      f"{args.time_limit}s per-solve limit and")
                print("     returned an INCUMBENT, not a proven optimum. Their cost "
                      "is an UPPER bound,")
                print("     so their breakeven is biased HIGH.")
                if (g > 0).any():
                    print(f"     worst reported gap {g.max():.2%}, "
                          f"median over affected nodes {g[g > 0].median():.2%}")
                else:
                    print("     (the solver reported no gap values)")
                print("     Re-run those nodes before reporting anything from them:")
                print("       python audit_optimality_b22.py <outdir> --drop")
                print("       python run_phased_sweep.py --outdir <same> --resume "
                      "--param-set register \\")
                print("           --workers $(( $(nproc) / 4 )) --threads 4")
            else:
                print(f"\n  optimality: all {len(df)} nodes proven to the "
                      f"{args.mip_gap:.0%} gap; none hit the time limit.")
    except Exception as ex:                                    # noqa: BLE001
        print(f"  optimality audit unavailable: {type(ex).__name__}: {ex}")

    if len(df) and "breakeven_unified" in df:
        good = df[(df["status"] == "ok") & df["breakeven_unified"].notna()]
        if len(good):
            print(f"\n  breakeven over {len(good)} solved nodes:")
            print(f"    median ${good['breakeven_unified'].median():,.2f}/MWhe   "
                  f"min ${good['breakeven_unified'].min():,.2f}   "
                  f"max ${good['breakeven_unified'].max():,.2f}")
            viable = (good["breakeven_unified"] < 300).sum()
            print(f"    below the $300 diesel reference: {viable}/{len(good)}")
            if "co2_avoided_t" in good:
                print(f"    CO2 avoided: {good['co2_avoided_t'].sum()/1e6:,.2f} Mt/yr "
                      "(sum over solved nodes; NOT a fleet total until the grid completes)")
    if gate_failures:
        print(f"\n  GATE F FAILURES ({len(gate_failures)}) -- results are NOT trustworthy:")
        for r in gate_failures[:10]:
            print(f"    {r['node_id']}: rel err {r.get('gate_rel_err')}")
    if args.scratch is None:
        shutil.rmtree(scratch, ignore_errors=True)

    if n_gate or n_err:
        return 1
    return 0


# =============================================================================
# SECTION 6 -- CLI
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_phased_sweep.py",
        description="Parallel (facility x phase x CAPEX x IR) sweep of the phased model.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("grid")
    g.add_argument("--phases", type=int, nargs="+", default=[1, 2, 3, 4])
    g.add_argument("--capex-mults", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    g.add_argument("--discount-rates", type=float, nargs="+", default=[0.05, 0.12, 0.18])
    g.add_argument("--limit", type=int, default=None, help="cap the node count (testing)")

    s = p.add_argument_group("solver and licence")
    s.add_argument("--solver", default="appsi_highs",
                   help="default HiGHS: free, no licence, no seat cap")
    s.add_argument("--workers", type=int, default=max(2, (os.cpu_count() or 4) // 4),
                   help="concurrent solves in THIS process. The old default of 2 was "
                        "the Gurobi seat limit; with HiGHS raise it to your core count.")
    s.add_argument("--threads", type=int, default=4, help="Gurobi Threads per solve")
    s.add_argument("--mip-gap", type=float, default=0.01)
    s.add_argument("--time-limit", type=int, default=3600, help="seconds per solve")
    s.add_argument("--license-sessions", type=int, default=2,
                   help="CLUSTER-WIDE concurrent Gurobi sessions. Raise only after "
                        "confirming with `gurobi_cl --license`.")
    s.add_argument("--sem-dir", default=None,
                   help="shared-filesystem directory for the licence semaphore")
    s.add_argument("--sem-timeout", type=float, default=None)
    s.add_argument("--no-semaphore", action="store_true")
    s.add_argument("--allow-no-license", action="store_true",
                   help="proceed on the reference path if no solver is available")
    s.add_argument("--force-reference", action="store_true")

    m = p.add_argument_group("model")
    m.add_argument("--param-set", default="register",
                   choices=["register", "design", "none"],
                   help="register: economic parameters come from data_inputs.csv "
                        "(measured / operator values, the v8 default). "
                        "design: the pre-v8 hardcoded defaults, which reproduce "
                        "the v7 grid exactly. none: SiteConfig defaults stand.")
    m.add_argument("--n-days", type=int, default=3, help="rolling days per node")
    m.add_argument("--no-subordination", action="store_true",
                   help="disable the mining subordination rule. Breaks the closed "
                        "form (Proposition 3) -- gate F will not be meaningful.")
    m.add_argument("--ramp-penalty-mode", default="pwl",
                   choices=["pwl", "quadratic", "l1", "none"],
                   help="Eq. D10 ramp penalty form. 'pwl' (default) keeps the "
                        "quadratic shape as tangent cuts and solves as a MILP on "
                        "any free solver. 'quadratic' is the true MIQP and needs "
                        "Gurobi, CPLEX or SCIP.")
    m.add_argument("--no-quadratic", action="store_true",
                   help="deprecated alias for --ramp-penalty-mode l1")

    r = p.add_argument_group("run")
    r.add_argument("--outdir", default="phased_sweep_results")
    r.add_argument("--scratch", default=None)
    r.add_argument("--resume", action="store_true")
    r.add_argument("--flush-every", type=int, default=5)
    r.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    r.add_argument("--n-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    r.add_argument("--walltime-hours", type=float, default=48.0)
    r.add_argument("--halt-on-gate-fail", dest="halt_on_gate_fail",
                   action="store_true", default=True)
    r.add_argument("--no-halt-on-gate-fail", dest="halt_on_gate_fail", action="store_false")
    r.add_argument("--smoke", action="store_true",
                   help="1 facility x 1 node, central case, for testing the harness")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = build_parser().parse_args(argv)
    if a.smoke:
        # Default the smoke test to PHASE 3, not phase 1. Phase 1 has no reactor
        # and no firm load, so its breakeven is undefined and gate F's F1 check
        # is skipped -- a smoke test that exercises the least of the machinery.
        # Phase 3 builds an SMR, serves firm load, and puts F1 through its paces.
        explicit = any(x.startswith("--phases") for x in (argv or sys.argv[1:]))
        a.phases = a.phases[:1] if explicit else [3]
        a.capex_mults, a.discount_rates, a.limit = [1.0], [0.12], 1
        a.n_days = min(a.n_days, 2)
        a.outdir = a.outdir + "_smoke"
        a.allow_no_license = True
        print("  SMOKE MODE: 1 facility, 1 node, central case\n")
    if a.sem_dir is None:
        a.sem_dir = os.path.join(os.path.abspath(a.outdir), ".gurobi_sem")
    if a.no_quadratic:
        print("  NOTE: --no-quadratic is deprecated; using --ramp-penalty-mode l1")
        a.ramp_penalty_mode = "l1"
    # The old worker-vs-seats warning printed BEFORE the preflight had decided
    # whether a semaphore was needed at all, so a licence-free run was told it
    # would be throttled to 2 when it would not be. run_sweep() reports the real
    # position after the preflight.
    return run_sweep(a)


if __name__ == "__main__":
    sys.exit(main())
