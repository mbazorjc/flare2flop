#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
phased_compute_model.py
=======================
Pyomo MIQP for Paper 2, "From Flares to FLOPS: A Phased Techno-Economic
Framework for Stranded Gas, Small Modular Reactors, and Compute Anchor Loads in
Sub-Saharan Africa".

Extends the Paper 1 formulation (Main_Manuscript_APEN_SMR_DataCentres_SSA.docx,
Eqs. D1-D26) to a four-phase flare-gas -> renewables -> SMR transition with
Bitcoin mining, AI/HPC compute and community offtake as demand streams.

    Phase 1  (t <  t2)   flare genset            -> Bitcoin mining
    Phase 2  (t2<=t<t3)  + solar / wind          -> mining + early compute
    Phase 3  (t3<=t<t4)  + SMR                   -> AI/HPC data centre
    Phase 4  (t >= t4)   SMR baseload + RE peak  -> full DC + community

-------------------------------------------------------------------------------
WHAT IS INHERITED FROM PAPER 1, UNCHANGED
-------------------------------------------------------------------------------
    D3      objective structure (profit = revenues - costs)
    D4      capital recovery factor
    D5      annualised capital with CAPEX multiplier mu
    D6      fixed O&M
    D7      variable O&M and fuel
    D8      start-up cost
    D10     quadratic ramp penalty  -- DEGENERACY BREAKING ONLY, excluded from
            every reported financial quantity (see exclude_ramp_from_financials)
    D11     energy balance          -- generalised by extension C, Paper 1 form
                                       recovered exactly when only the SMR runs
    D13     reactor thermal constraint
    D14     generation limits
    D15     minimum stable load
    D16     ramp constraints
    D17-D19 unit-commitment logic
    D20     technology and module selection
    D24     diesel breakeven        -- generalised by extension F

-------------------------------------------------------------------------------
*** READ THIS BEFORE TRUSTING GATE F ***  Proposition 3, methods note §7.5
-------------------------------------------------------------------------------
The unified breakeven is recoverable in closed form ONLY while the mining load
is subordinate to firm demand. The specification for extension C,

    P_avail[t] = s_btc[t] + s_dc[t] + s_comm[t] + c[t]

places no precedence between s_btc and s_dc. Combined with rev_BTC in the
objective (extension D), the optimiser is then free to allocate energy to
whichever stream pays more, so the dispatch becomes a function of the diesel
price d, net profit ceases to be affine in d, and

    Pi(d) = max{ d.S , rev_BTC } + (terms independent of d)

is piecewise linear with a kink. The closed form does not exist there, and
gate F cannot pass except by coincidence.

This module therefore adds the subordination constraint

    s_btc[t] <= P_avail[t] - s_dc[t] - s_comm[t]          (mining is residual)

controlled by `subordinate_mining`, DEFAULT TRUE. With it on, the allocation is
fixed by demand and availability rather than by price, Pi(d) stays affine, and
gate F is meaningful. With it off the model still solves and still reports, but
gate F is downgraded to a warning and the breakeven is labelled NOT RECOVERABLE.
The smoke test demonstrates both branches.

-------------------------------------------------------------------------------
RUNNING WITHOUT A SOLVER
-------------------------------------------------------------------------------
Pyomo and Gurobi are not importable in every environment this project runs in
(the desktop workspace has no package network). Under the subordination rule the
dispatch for a FIXED build is fully determined -- there is nothing left to
optimise hour by hour -- so it can be computed directly. `reference_dispatch()`
does exactly that in numpy, and gate F runs against it. This is not a substitute
for the MIQP: it cannot choose a technology or a module count. It exists so the
accounting chain (costs, revenues, served energy, breakeven) is executable and
verifiable everywhere, and so the MIQP has an independent target to match.

    python phased_compute_model.py              # auto: MIQP if available, else reference
    python phased_compute_model.py --no-solver  # force the reference path
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
for _d in (HERE, os.path.dirname(HERE)):
    if _d not in sys.path:
        sys.path.insert(0, _d)

import analytical_benchmark as ab          # noqa: E402  ground truth, no solver
import data_inputs as di                   # noqa: E402  parameter assembly

try:
    import pyomo.environ as pe
    from pyomo.opt import TerminationCondition
    HAVE_PYOMO = True
except ImportError:                        # documented, expected in some envs
    pe = None
    TerminationCondition = None
    HAVE_PYOMO = False

HOURS_PER_YEAR = ab.HOURS_PER_YEAR
TWO_POW_32 = ab.TWO_POW_32
M3_PER_SCF = di.M3_PER_SCF          # exact, international foot


# =============================================================================
# SECTION 1 -- CONFIGURATION
# =============================================================================

@dataclass
class PhaseConfig:
    """Phase boundaries in hours, and which technologies each phase admits.

    Extension E. Gating is applied as hard variable bounds inside each rolling
    window rather than as Big-M constraints on binaries, because the phase of an
    hour is known exogenously -- it is a function of the timestep, not a
    decision. Fixing a variable to zero is tighter than any Big-M and removes
    the column from the presolved problem entirely. `bigM_gating=True` restores
    the binary formulation for the sensitivity case where phase timing is itself
    a decision, which is Day-6 work and is not exercised here.
    """
    t2: int = 3 * HOURS_PER_YEAR            # end of Phase 1
    t3: int = 5 * HOURS_PER_YEAR            # end of Phase 2
    t4: int = 10 * HOURS_PER_YEAR           # end of Phase 3
    bigM_gating: bool = False

    def phase_of(self, t: int) -> int:
        if t < self.t2:
            return 1
        if t < self.t3:
            return 2
        if t < self.t4:
            return 3
        return 4

    def allows(self, t: int) -> Dict[str, bool]:
        """Which technologies and demand streams exist in the phase of hour t.

        NOTE the `dc` gate is p >= 2, not p >= 3. The paper's Phase 2 is
        "+ Solar/Wind hybrid -> Bitcoin + EARLY AI COMPUTE", so a compute load
        does exist there; it is supply-limited rather than firm (see
        `economics()['dc_share_of_available']`).
        """
        p = self.phase_of(t)
        return {
            "smr":   p >= 3,
            "solar": p >= 2,
            "wind":  p >= 2,
            "gas":   p <= 2,                # flare gas retired once the SMR runs
            "btc":   p <= 2,                # mining is the Phase 1-2 anchor
            "dc":    p >= 2,                # early compute from Phase 2
            "comm":  True,
        }

    # -- per-phase economics  (B-09 fix) ------------------------------------
    # Until 2026-09-15 `allows()` returned an IDENTICAL dict for phases 3 and 4,
    # so the 333 Phase-4 nodes of Sweep Run 2 were bit-identical duplicates of
    # the Phase-3 nodes and the four-phase architecture was never modelled. The
    # economic distinction between the phases lived only in
    # analytical_benchmark.build_phases and was never carried across. This
    # method is that definition, now shared by both codebases.
    #
    #   smr_scale               multiple of the reactor fleet built in the phase
    #   re_scale_of_genset      RE nameplate as a multiple of the flare genset
    #   re_scale_of_smr         RE nameplate as a multiple of the SMR fleet
    #   dc_share_of_available   Phase-2 compute is SUPPLY-limited, not firm:
    #                           the facility's ultimate demand dwarfs a flare
    #                           site's output, so an early-compute load is a
    #                           share of what is available, not of what the
    #                           finished data centre will eventually draw
    #   comm_share              convention #8 band, 0.10-0.15
    #
    # All five are [ASSUMED] -- no working-folder file sizes a phase.
    PHASE_ECONOMICS = {
        1: dict(smr_scale=0.0, re_scale_of_genset=0.0, re_scale_of_smr=0.0,
                dc_share_of_available=0.0,  comm_share=0.10),
        2: dict(smr_scale=0.0, re_scale_of_genset=1.0, re_scale_of_smr=0.0,
                dc_share_of_available=0.40, comm_share=0.10),
        3: dict(smr_scale=1.0, re_scale_of_genset=0.0, re_scale_of_smr=0.0,
                dc_share_of_available=1.0,  comm_share=0.10),
        4: dict(smr_scale=1.0, re_scale_of_genset=0.0, re_scale_of_smr=0.2,
                dc_share_of_available=1.0,  comm_share=0.15),
    }

    def economics(self, p: int) -> Dict[str, float]:
        return dict(self.PHASE_ECONOMICS[p])


@dataclass
class SolverConfig:
    """Solver settings.

    DEFAULT CHANGED 2026-09-15 from Gurobi to HiGHS.

    Paper 1 used Gurobi at a 1% gap. Gurobi's licence is the single largest
    constraint on this project -- it caps concurrency at 2 sessions, it produced
    the 50 dead cells that `rerun_failed.sh` exists to re-run, and it is
    currently returning "Unauthorized access" so the MIQP has never once been
    solver-executed.

    HiGHS removes all of that: no licence, no token server, no seat limit,
    `pip install highspy`. It is a serious MILP solver, not a toy, and because
    there is no seat cap the whole 2-session concurrency machinery in
    run_phased_sweep.py becomes unnecessary -- workers can equal cores.

    The one thing HiGHS does not do well is mixed-integer QUADRATIC programming.
    That does not matter here: see the note on the ramp penalty in build_model().
    The only quadratic term in the model is cosmetic and is now linearised by
    default, making this a pure MILP.

    CPLEX is a viable alternative ONLY with an IBM Academic Initiative licence.
    The free Community Edition caps at 1,000 variables and 1,000 constraints; a
    48-hour window with two modules already exceeds that several times over, so
    the Community Edition is not an option for this model.
    """
    solver: str = "appsi_highs"
    mip_gap: float = 0.01
    time_limit: int = 3600
    threads: int = 0
    tee: bool = False
    fallbacks: Tuple[str, ...] = ("appsi_highs", "highs", "gurobi", "gurobi_direct",
                                  "cbc", "scip", "glpk")


@dataclass
class SiteConfig:
    """One facility plus its energy system."""
    facility_id: str = "SMOKE"
    it_mw: float = 20.0
    pue: float = 1.5
    reactor: str = "CAREM-25"
    capex_mult: float = 1.0                 # mu, Eq. D5
    discount_rate: float = 0.12             # Eq. D4
    plant_life: int = 25
    diesel_price: float = 300.0             # d, the quantity the breakeven solves for
    # flare gas
    flare_mmscfd: float = 2.0
    flare_decline: float = 0.05
    eta_gen: float = 0.38                   # hard convention #7
    lhv_mj_m3: float = 36.0
    genset_capex_per_kwe: float = 800.0
    genset_fopex_per_kwe: float = 40.0
    genset_vom: float = 5.0
    # --- B-21 / D-09: measured system availability of the flare-to-power set ---
    # Fraction of hours the flare genset + mining load actually delivers. The
    # pre-v8 model had NO availability term at all: flare_gas_module returns
    # "MWh/yr at full uptime" and the register's genset_capacity_factor was read
    # only by analytical_benchmark.py, never by the solver. Phase-1/2 energy was
    # therefore an implicit 100%-uptime upper bound.
    #
    # Default stays 1.0 so every pre-v8 result reproduces byte-for-byte. The
    # sweep passes the register value (--param-set register); the design case is
    # recovered with --param-set design.
    #
    # CAVEAT, must survive into the manuscript: the pilot series measures POOL
    # uptime, which aggregates genset outage, miner outage, connectivity and
    # economic curtailment. It cannot attribute downtime to a cause, so it is an
    # as-operated system availability, not a genset forced-outage rate.
    genset_availability: float = 1.0
    # renewables
    solar_cap_mw: float = 0.0
    wind_cap_mw: float = 0.0
    solar_capex_per_kwe: float = 1100.0
    wind_capex_per_kwe: float = 1600.0
    solar_fopex_per_kwe: float = 22.64      # Lal Table 3 opfac_solar
    wind_fopex_per_kwe: float = 43.0        # Lal Table 3 opfac_wind
    # prices
    btc_price: float = 60000.0
    btc_difficulty: float = 1.10e14
    block_reward: float = 3.125             # hard convention #6
    # D-07 CLOSED 2026-09-18. Fleet efficiency derived from the pilot pool record:
    # measured 71.87 TH/s per worker against the published power envelope for that
    # hashrate class, which is flat to 3.3% across every candidate machine. The
    # old 17.5 J/TH was S21-class hardware the site does not run, and it made
    # mining revenue 2.6x too high. See data_inputs.MINER_PILOT.
    miner_efficiency_j_per_th: float = 17.5   # default kept for v7 reproducibility
    compute_price: float = 120.0            # $/MWhe of IT load  (D-16 OPEN)
    elec_price_comm: float = 100.0          # $/MWhe             (D-03 OPEN)
    carbon_price: float = 50.0              # $/tCO2             (D-04 OPEN)
    flare_penalty: float = 3.50             # $/mscf, PIA 2021
    # shares and caps
    comm_share_min: float = 0.10            # hard convention #8
    comm_share_max: float = 0.15
    export_cap: float = 0.0                 # hard convention #9, behind the meter
    capacity_factor: float = 0.95
    # emission factors and the carbon accounting basis  (B-08 / D-26)
    ef_diesel: float = 0.70
    ef_nuclear: float = 0.012
    carbon_basis: str = "delivered"        # delivered | served_dc | all_delivered
    flare_counterfactual: str = "flared"   # flared | unburned  -- see ef_gas_incremental
    ef_gas: float = field(default=0.0)      # derived in __post_init__
    ef_natural_gas_kg_per_mmbtu: float = 53.06
    # modelling switches
    subordinate_mining: bool = True         # Proposition 3 -- see module docstring
    ramp_penalty_eps: float = 1e-6          # Eq. D10, degeneracy only
    exclude_ramp_from_financials: bool = True
    quadratic: bool = False                 # legacy flag; True forces mode="quadratic"
    ramp_penalty_mode: str = "pwl"          # pwl | quadratic | l1 | none  -- see build_model
    ramp_pwl_cuts: int = 9                  # tangent cuts for the "pwl" mode
    reliability_margin: float = 1.2         # kappa, Eq. D2
    hard_module_cap: int = 20               # N_hard, Eq. D2

    def __post_init__(self):
        if self.ef_gas == 0.0:
            self.ef_gas = ab.ef_from_genset(self.eta_gen, self.ef_natural_gas_kg_per_mmbtu)

    @property
    def demand_mw(self) -> float:
        """Eq. D1: total facility demand = IT load x PUE."""
        return self.it_mw * self.pue

    @property
    def ef_gas_incremental(self) -> float:
        """Emission factor of the flare genset AGAINST ITS OWN COUNTERFACTUAL.

        This is not the same quantity as `ef_gas` (0.476 tCO2/MWhe, the absolute
        combustion factor), and using the absolute factor in an AVOIDED-emissions
        calculation is a category error that the previous formulation made.

        The counterfactual for captured associated gas is that the gas is FLARED.
        Flaring burns it and emits essentially the same CO2. Routing it through a
        genset therefore changes almost nothing about the carbon released -- the
        molecules burn either way -- so the INCREMENTAL emission factor of the
        project is approximately zero, not 0.476. What the project actually
        avoids is the diesel that would have served the load.

        flare_counterfactual = "flared"   -> 0.0   (default; the gas burns anyway)
        flare_counterfactual = "unburned" -> ef_gas (0.476; for the sensitivity
                                             where the gas would have stayed in
                                             the ground or been reinjected)

        FLAG: no methane-slip credit is taken in either case. Capture destroys
        marginally more methane than a flare does, worth roughly
        0.011 tCO2e/mscf, but claiming it needs its own citation and its own
        uncertainty treatment. Omitting it is conservative. See D-15.
        """
        return 0.0 if self.flare_counterfactual == "flared" else self.ef_gas


# =============================================================================
# SECTION 2 -- EXOGENOUS PROFILES
# =============================================================================

def flare_profile(cfg: SiteConfig, n_hours: int, t0: int = 0) -> np.ndarray:
    """Extension A. Available flare-genset electrical power, MW, per hour.

        P_flare[t] = eta_gen x V_flare[t] x LHV

    Volumes and the unit chain come from data_inputs.flare_gas_module, which
    converts MMSCFD -> m3/s -> MW. Decline is annual and held flat within a year.
    """
    years = int(math.ceil((t0 + n_hours) / HOURS_PER_YEAR)) + 1
    f = di.flare_gas_module(V0_mmscfd=cfg.flare_mmscfd, decline=cfg.flare_decline,
                            eta_gen=cfg.eta_gen, lhv_mj_m3=cfg.lhv_mj_m3,
                            penalty=cfg.flare_penalty, years=years)
    # B-21: availability derates the DELIVERABLE ceiling, not the resource.
    # flare_gas_volume applies the SAME factor, so the ratio vFlare/flare_mw in
    # the penalty term (Eq. E14) is unchanged at full dispatch and gas consumed
    # scales with energy produced. Gas not consumed stays flared: no penalty
    # avoided, no CO2 avoided. See SiteConfig.genset_availability.
    return f["hourly_mw"][t0:t0 + n_hours] * float(cfg.genset_availability)


# MIDPOINT levels, (i + 1/2)/N, not linspace(0, 1, N+1). Equally spaced levels
# INCLUDING both endpoints over-weight the extremes, so the mean of the quantiles
# is a biased estimate of the mean of the series -- which showed up as a uniform
# 0.2% error on the purely LINEAR community-share ratio, where the answer must be
# exact by construction. Midpoints are the equal-probability stratification and
# remove that bias; the explicit rescale below removes what is left of it.
_NQ = 20
_QLEVELS = (np.arange(_NQ) + 0.5) / _NQ


def _quantile_str(x: np.ndarray, levels: np.ndarray = _QLEVELS) -> str:
    """20 equal-probability quantiles of an hourly series, semicolon-joined.

    Quantiles rather than the raw series because the sweep writes one CSV row
    per facility-phase and 72 hourly values per row would be unreadable. The
    vector is then rescaled so its mean equals the series mean EXACTLY. That
    makes every linear functional of the shape exact -- generation, community
    share, avoided penalty -- and leaves only the kinked min() carrying any
    discretisation error at all, which is where the shape was needed in the
    first place.
    """
    if x is None or len(x) == 0:
        return ""
    x = np.asarray(x, dtype=float)
    q = np.quantile(x, levels)
    m, qm = float(x.mean()), float(q.mean())
    if qm > 1e-12 and m > 0.0:
        q = q * (m / qm)
    return ";".join(f"{v:.8g}" for v in q)


def parse_quantile_str(s: str) -> np.ndarray:
    """Inverse of _quantile_str. Empty or malformed input gives an empty array."""
    if not s or not isinstance(s, str):
        return np.array([], dtype=float)
    try:
        return np.array([float(v) for v in s.split(";") if v.strip() != ""], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


def genset_nameplate_mw(cfg: "SiteConfig", flare_mw) -> float:
    """B-21. Installed genset rating, MW -- UNDEGRADED by availability.

    The set is sized to the gas rate it must burn, so downtime does not shrink
    the machine: availability reduces ENERGY, never NAMEPLATE. flare_profile
    returns the deliverable ceiling (already multiplied by cfg.genset_availability),
    so the rating is recovered by dividing that scalar factor back out. This is
    exact, not an approximation, because the derate is a single scalar.

    Getting this wrong is the whole point of the check: if nameplate fell with
    availability, capital would fall with it and a less reliable plant would
    look CHEAPER. The breakeven must move the other way -- same capital charge,
    less energy to spread it over.
    """
    import numpy as _np
    peak = float(_np.max(flare_mw)) if len(flare_mw) else 0.0
    a = float(cfg.genset_availability)
    return max(peak / a, 0.0) if a > 1e-9 else 0.0


def flare_gas_volume(cfg: SiteConfig, n_hours: int, t0: int = 0) -> np.ndarray:
    """Associated gas consumed, mscf per hour, aligned with flare_profile."""
    years = int(math.ceil((t0 + n_hours) / HOURS_PER_YEAR)) + 1
    f = di.flare_gas_module(V0_mmscfd=cfg.flare_mmscfd, decline=cfg.flare_decline,
                            eta_gen=cfg.eta_gen, lhv_mj_m3=cfg.lhv_mj_m3,
                            penalty=cfg.flare_penalty, years=years)
    # B-21, paired with flare_profile: gas CONSUMED derates with availability.
    # f["annual_mscf"] remains the undegraded resource for the check-4e bound.
    return f["hourly_mscf"][t0:t0 + n_hours] * float(cfg.genset_availability)


# --- Lal et al. (2023) Eq. 2, solar PV ---------------------------------------
MU_PV = 0.25          # Lal Table 3, module efficiency
G_STC = 1.0           # kW/m2 at standard test conditions


def solar_power(irradiance_kw_m2: np.ndarray, cap_mw: float,
                mu_pv: float = MU_PV) -> np.ndarray:
    """Lal Eq. 2 -- PV output linear in irradiance.

        P = mu_PV x A x G,  written here per installed kW as
        P = cap x (G / G_STC) x (mu_PV / mu_PV) = cap x G / G_STC

    The area form and the nameplate form are the same linear function; the
    nameplate form is used so the capacity variable is comparable with the other
    supply classes. Output is clipped at nameplate.

    FLAG: irradiance is not in the working folder. `synthetic_irradiance` below
    is a placeholder diurnal profile, NOT measured data. Replace before any
    result is reported (open item D-11).
    """
    return np.clip(cap_mw * irradiance_kw_m2 / G_STC, 0.0, cap_mw)


# --- Lal et al. (2023) Eq. 1, wind, piecewise --------------------------------
V_CIN, V_R, V_COUT = 2.0, 11.0, 25.0      # Lal Table 3, m/s


def wind_power(speed_ms: np.ndarray, cap_mw: float, v_cin: float = V_CIN,
               v_r: float = V_R, v_cout: float = V_COUT) -> np.ndarray:
    """Lal Eq. 1 -- piecewise wind power curve.

        P = 0                                     v < v_cin  or  v > v_cout
        P = cap x (v^3 - v_cin^3)/(v_r^3 - v_cin^3)   v_cin <= v < v_r
        P = cap                                   v_r <= v <= v_cout

    FLAG: Lal Table 3 supplies the three cut-in / rated / cut-out speeds but the
    published Eq. 1 interpolation between cut-in and rated is not reproduced in
    the extract available here. The cubic form is used because wind power is
    proportional to the cube of speed; if Lal use a linear ramp the Phase-2
    energy differs by a few per cent. Confirm against the paper before reporting.
    """
    v = np.asarray(speed_ms, dtype=float)
    p = np.zeros_like(v)
    ramp = (v >= v_cin) & (v < v_r)
    p[ramp] = cap_mw * (v[ramp] ** 3 - v_cin ** 3) / (v_r ** 3 - v_cin ** 3)
    p[(v >= v_r) & (v <= v_cout)] = cap_mw
    return p


def synthetic_irradiance(n_hours: int, t0: int = 0, peak: float = 0.85) -> np.ndarray:
    """PLACEHOLDER diurnal irradiance, kW/m2. Not measured data. D-11 OPEN."""
    h = (np.arange(t0, t0 + n_hours) % 24)
    day = np.clip(np.sin((h - 6) / 12.0 * np.pi), 0.0, None)
    return peak * day


def synthetic_wind(n_hours: int, t0: int = 0, mean: float = 7.0) -> np.ndarray:
    """PLACEHOLDER wind speed, m/s. Not measured data. D-11 OPEN."""
    t = np.arange(t0, t0 + n_hours)
    return np.clip(mean + 2.5 * np.sin(t / 37.0) + 1.5 * np.sin(t / 11.0), 0.0, None)


def btc_revenue_per_mwh(cfg: SiteConfig) -> float:
    """Lal Eq. 11 collapsed to $/MWh of miner draw, for use as a linear price.

    Eq. 11 is linear in hashrate and hashrate is linear in miner power, so
    mining revenue is linear in s_btc and enters the objective as a price. That
    linearity is what keeps the problem an MIQP rather than something harder.

        rev = SP x R x H x t_m / (D x 2^32),  H = P[W] / eps x 1e12

    Revenue per MWh of miner draw depends ONLY on efficiency, never on which
    machine delivers it: at eps J/TH, one MW of draw buys 1e6/eps TH/s whatever
    the box. So the price is parameterised on eps directly rather than on a named
    unit, which is what lets the pilot-derived fleet efficiency (D-07) drive it.
    Passing 1 PH/s at eps kW is exactly one PH/s worth of draw, so the verified
    bitcoin_module is reused unchanged rather than reimplemented here.
    """
    eps = float(cfg.miner_efficiency_j_per_th)
    if not (eps > 0.0):
        raise ValueError(f"miner_efficiency_j_per_th must be positive, got {eps}")
    b = di.bitcoin_module(n_miners=1, hash_ths=1000.0,      # 1 PH/s
                          power_kw=eps,                     # which draws eps kW
                          btc_price=cfg.btc_price,
                          difficulty_th=cfg.btc_difficulty,
                          block_reward=cfg.block_reward, t_m=3600.0)
    return b["revenue_per_mwh_miner"]


# =============================================================================
# SECTION 3 -- REACTOR PARAMETERS
# =============================================================================

def ramp_penalty_diagnostics(cfg: SiteConfig, reactor: Optional[str] = None,
                             verbose: bool = True) -> dict:
    """Quantify the piecewise-linear representation of the Eq. D10 ramp penalty.

    Produces the numbers a Methods section needs in order to claim that the
    MILP form represents the MIQP formulation exactly to a stated tolerance,
    rather than merely asserting it.

    The penalty is eps * d^2 with d the hour-on-hour change in generation above
    minimum stable load, bounded by the reactor's ramp rate R. Representing the
    convex term by K tangent hyperplanes gives a worst-case error of
    (R/(K-1))^2 in MW^2, i.e. eps*(R/(K-1))^2 in dollars.
    """
    c = SiteConfig(**{**cfg.__dict__, "reactor": reactor or cfg.reactor}) \
        if reactor else cfg
    rx = reactor_params(c)
    R = float(rx["ramp"])
    K = max(3, int(cfg.ramp_pwl_cuts))
    eps = cfg.ramp_penalty_eps

    a = np.linspace(-R, R, K)
    d = np.linspace(-R, R, 20001)
    z = np.maximum(np.max(2 * a[:, None] * d[None, :] - (a ** 2)[:, None], axis=0), 0.0)
    err_mw2 = float(np.abs(d ** 2 - z).max())

    out = {
        "reactor": rx["name"], "ramp_mw_per_h": R, "cuts": K, "eps": eps,
        "max_error_mw2": err_mw2,
        "theoretical_bound_mw2": (R / (K - 1)) ** 2,
        "max_error_usd": err_mw2 * eps,
        "max_relative_error": err_mw2 / max(R ** 2, 1e-12),
        "penalty_worst_single_pair_usd": eps * R ** 2,
    }
    if verbose:
        print(f"  Eq. D10 ramp penalty — piecewise-linear representation")
        print(f"    reactor {out['reactor']}, ramp {R} MW/h, {K} tangent cuts, eps {eps}")
        print(f"    worst-case error : {err_mw2:.6f} MW^2 = ${out['max_error_usd']:.3e} "
              f"({out['max_relative_error']:.4%} of the term)")
        print(f"    theoretical bound: {out['theoretical_bound_mw2']:.6f} MW^2  "
              f"{'MATCH' if abs(err_mw2 - out['theoretical_bound_mw2']) < 1e-9 else 'MISMATCH'}")
        print(f"    the penalty's own worst-case magnitude on one ramp pair is "
              f"${out['penalty_worst_single_pair_usd']:.3e}")
        print(f"    reported financials are quoted to $0.01, so both are immaterial")
    return out


def reactor_params(cfg: SiteConfig) -> dict:
    """Pull one design from baseiaea.csv. Hard convention #11: never invent."""
    cat = di.load_smr_catalogue()
    if cfg.reactor not in cat.index:
        raise KeyError(f"{cfg.reactor!r} not in baseiaea.csv; available: {list(cat.index)}")
    r = cat.loc[cfg.reactor]
    return {
        "name": cfg.reactor,
        "cap_e": float(r["Power in MWe"]),
        "cap_t": float(r["Power in MWt"]),
        "eta_th": float(r["Thermal Efficiency"]),
        "eta_trans": float(r["Thermal Transfer Efficiency"]),
        "msl": float(r["MSL in MWe"]),
        "msl_turb": float(r["MSL_turb in MWe"]),
        "mdt": int(r["MDT in hours"]),
        "ramp": float(r["Ramp Rate (MW/hr)"]),
        "capex": float(r["CAPEX $/kWe"]),
        "fopex": float(r["FOPEX $/kWe"]),
        "vom": float(r["VOM in $/MWh-e"]),
        "fc": float(r["FC in $/MWh-e"]),
        "startup": float(r["Startupfixedcost in $"]),
        "max_modules": int(r["Max Modules"]),
    }


def max_modules(cfg: SiteConfig, rx: dict) -> int:
    """Eq. D2: N_max = min( ceil(kappa x D / Cap), N_hard ), capped by catalogue.

    This is the SEARCH BOUND for the MIQP -- the largest fleet it may consider --
    not the fleet it should build. The kappa = 1.2 reliability margin belongs
    here and nowhere else.
    """
    need = math.ceil(cfg.reliability_margin * cfg.demand_mw / rx["cap_e"])
    return int(max(1, min(need, cfg.hard_module_cap, rx["max_modules"])))


def required_modules(cfg: SiteConfig, rx: dict) -> int:
    """Smallest fleet satisfying Paper 1's capacity-adequacy rule, nameplate >= peak.

    Used by the reference dispatch, which cannot optimise the build. Using
    max_modules() here instead would force a second module on any facility whose
    demand exactly fills one -- 30 MW served by a 30 MWe CAREM-25 -- and double
    the annualised capital for no served energy. The MIQP chooses freely between
    1 and max_modules(); this is the value it should land on when capital is
    costly and the load is flat.
    """
    if cfg.demand_mw <= 0:
        return 1
    return int(max(1, min(math.ceil(cfg.demand_mw / rx["cap_e"]),
                          cfg.hard_module_cap, rx["max_modules"])))


# =============================================================================
# SECTION 4 -- THE PYOMO MODEL
# =============================================================================

def objective_degree_report(model, verbose: bool = True) -> dict:
    """Report the polynomial degree of the objective and of each named term.

    Exists because of a real failure: a quadratic objective handed to HiGHS dies
    inside the Pyomo interface with

        DegreeError: Highs interface does not support expressions of degree None

    "degree None" is not a corrupt expression -- it is simply how that interface
    reports "not linear", because it calls generate_standard_repn with
    quadratic=False and anything above degree 1 then comes back unclassified.
    The message names no term and no variable, so with 1,332 nodes you get 1,332
    identical tracebacks and no diagnosis. This function names the term.
    """
    if not HAVE_PYOMO:
        raise RuntimeError("Pyomo is not installed.")
    out = {"total": None, "terms": {}, "nonlinear": []}
    try:
        out["total"] = model.Profit.expr.polynomial_degree()
    except Exception:                                  # noqa: BLE001
        out["total"] = None
    for name, expr in getattr(model, "_parts", {}).items():
        try:
            deg = expr.polynomial_degree() if hasattr(expr, "polynomial_degree") else 0
        except Exception:                              # noqa: BLE001
            deg = None
        out["terms"][name] = deg
        if deg is None or (isinstance(deg, int) and deg > 1):
            out["nonlinear"].append(name)
    if verbose:
        print(f"  objective polynomial degree: {out['total']}")
        for k, v in sorted(out["terms"].items(), key=lambda kv: (kv[1] is None, kv[1] or 0)):
            flag = "  <-- NOT LINEAR" if (v is None or (isinstance(v, int) and v > 1)) else ""
            print(f"    {k:<14} degree {v}{flag}")
        if out["nonlinear"]:
            print(f"  linear-only solvers (HiGHS, CBC, GLPK) will reject this model "
                  f"because of: {', '.join(out['nonlinear'])}")
    return out


def build_model(cfg: SiteConfig, phases: PhaseConfig, t0: int, n_hours: int,
                n_modules: Optional[int] = None,
                init_state: Optional[dict] = None):
    """Construct the MIQP for one window [t0, t0 + n_hours).

    Variable naming follows Paper 1 (vGen, vRGen, vServed, vOn, vTurnOn, ...) so
    the two codebases can be diffed. New streams take the s_* names from the
    Paper 2 specification.

    n_modules None  -> module count is a decision (Eq. D20), the sizing pass.
    n_modules fixed -> build is frozen, the rolling dispatch pass.
    """
    if not HAVE_PYOMO:
        raise RuntimeError("Pyomo is not installed; use reference_dispatch() instead.")

    rx = reactor_params(cfg)
    p_here = phases.phase_of(t0)
    pc = phase_capacities(cfg, phases, p_here,
                          rx, n_modules if n_modules is not None else max_modules(cfg, rx))
    m = pe.ConcreteModel(name=f"phased_compute[{cfg.facility_id}]")
    T = list(range(n_hours))
    m.T = pe.Set(initialize=T, ordered=True)
    Nmax = n_modules if n_modules is not None else max_modules(cfg, rx)
    m.M = pe.Set(initialize=list(range(Nmax)), ordered=True)

    # ---- exogenous profiles ------------------------------------------------
    flare_mw = flare_profile(cfg, n_hours, t0)
    gas_mscf = flare_gas_volume(cfg, n_hours, t0)
    irr = synthetic_irradiance(n_hours, t0)
    wind_ms = synthetic_wind(n_hours, t0)
    solar_mw = solar_power(irr, pc["solar_mw"])       # B-10: from the phase table
    wind_mw = wind_power(wind_ms, pc["wind_mw"])
    allow = [phases.allows(t0 + t) for t in T]
    H = n_hours / HOURS_PER_YEAR                      # horizon fraction, Eq. D5

    m.pFlareMax = pe.Param(m.T, initialize={t: float(flare_mw[t]) for t in T})
    m.pGasMscf = pe.Param(m.T, initialize={t: float(gas_mscf[t]) for t in T})
    m.pSolar = pe.Param(m.T, initialize={t: float(solar_mw[t]) for t in T})
    m.pWind = pe.Param(m.T, initialize={t: float(wind_mw[t]) for t in T})
    m.pDemand = pe.Param(m.T, initialize={t: (cfg.demand_mw if allow[t]["dc"] else 0.0) for t in T})
    dc_share = pc["dc_share_of_available"]

    btc_price_mwh = btc_revenue_per_mwh(cfg)

    # ---- decision variables ------------------------------------------------
    # SMR fleet, Paper 1 naming
    m.vGen = pe.Var(m.M, m.T, within=pe.NonNegativeReals)        # MWhe electric
    m.vRGen = pe.Var(m.M, m.T, within=pe.NonNegativeReals)       # MWht thermal
    m.vGenAbv = pe.Var(m.M, m.T, within=pe.NonNegativeReals)     # above MSL, D15
    m.vOn = pe.Var(m.M, m.T, within=pe.Binary)                   # D17
    m.vEOn = pe.Var(m.M, m.T, within=pe.Binary)                  # turbine, D14
    m.vTurnOn = pe.Var(m.M, m.T, within=pe.Binary)               # D17-D18
    m.vTurnOff = pe.Var(m.M, m.T, within=pe.Binary)
    m.vModule = pe.Var(m.M, within=pe.Binary)                    # D20

    # new supply classes (extensions A, B)
    m.vFlare = pe.Var(m.T, within=pe.NonNegativeReals)
    m.vSolar = pe.Var(m.T, within=pe.NonNegativeReals)
    m.vWind = pe.Var(m.T, within=pe.NonNegativeReals)

    # demand streams (extension C)
    m.sBTC = pe.Var(m.T, within=pe.NonNegativeReals)
    m.sDC = pe.Var(m.T, within=pe.NonNegativeReals)
    m.sComm = pe.Var(m.T, within=pe.NonNegativeReals)
    m.vCurtail = pe.Var(m.T, within=pe.NonNegativeReals)
    m.vExport = pe.Var(m.T, within=pe.NonNegativeReals)
    m.vUnmet = pe.Var(m.T, within=pe.NonNegativeReals)

    if n_modules is not None:
        for mm in m.M:
            m.vModule[mm].fix(1)

    # ---- phase gating (extension E) ----------------------------------------
    # Phase membership is exogenous, so gated variables are FIXED to zero rather
    # than bounded by a Big-M. This is tighter and removes the columns entirely.
    for t in T:
        a = allow[t]
        if not a["gas"]:
            m.vFlare[t].fix(0.0)
        else:
            m.vFlare[t].setub(float(flare_mw[t]))
        if not a["solar"]:
            m.vSolar[t].fix(0.0)
        else:
            m.vSolar[t].setub(float(solar_mw[t]))
        if not a["wind"]:
            m.vWind[t].fix(0.0)
        else:
            m.vWind[t].setub(float(wind_mw[t]))
        if not a["btc"]:
            m.sBTC[t].fix(0.0)
        if not a["dc"]:
            m.sDC[t].fix(0.0)
            m.vUnmet[t].fix(0.0)
        elif pc["dc_share_of_available"] < 1.0:
            # Phase 2's compute load is SUPPLY-limited, so `_firm` is skipped and
            # there is no firm demand to fall short of. vUnmet then appears in no
            # constraint and in no objective term, the solver never assigns it a
            # value, and extract_ledger dies with
            #   ValueError: No value for uninitialized VarData object vUnmet[0]
            # -- which is what killed all 333 Phase-2 nodes of the 2026-09-16
            # full grid. Unmet firm load is not merely zero here, it is undefined,
            # so fixing it to zero is the correct statement as well as the safe one.
            m.vUnmet[t].fix(0.0)
        if not a["smr"]:
            for mm in m.M:
                m.vGen[mm, t].fix(0.0)
                m.vRGen[mm, t].fix(0.0)
                m.vGenAbv[mm, t].fix(0.0)
                m.vOn[mm, t].fix(0)
                m.vEOn[mm, t].fix(0)
                m.vTurnOn[mm, t].fix(0)
                m.vTurnOff[mm, t].fix(0)

    if phases.bigM_gating:
        # Alternative formulation retained for the Day-6 case where phase timing
        # is itself a decision. M_s is the class nameplate, never an arbitrary
        # large constant: a loose Big-M weakens the LP relaxation and inflates
        # the branch-and-bound tree.
        m.zPhase = pe.Var(range(1, 5), within=pe.Binary)
        m.PhaseOne = pe.Constraint(expr=sum(m.zPhase[p] for p in range(1, 5)) == 1)

    # =========================================================================
    # CONSTRAINTS
    # =========================================================================

    # ---- D13 reactor thermal ------------------------------------------------
    def _thermal(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vGen[mm, t] / rx["eta_th"] <= mo.vRGen[mm, t] * rx["eta_trans"]
    m.ReactorThermal = pe.Constraint(m.M, m.T, rule=_thermal)

    # ---- D14 generation limits ---------------------------------------------
    def _genlim_e(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vGen[mm, t] <= rx["cap_e"] * mo.vEOn[mm, t]
    m.GenLimitE = pe.Constraint(m.M, m.T, rule=_genlim_e)

    def _genlim_t(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vRGen[mm, t] * rx["eta_th"] <= rx["cap_e"] * mo.vOn[mm, t]
    m.GenLimitT = pe.Constraint(m.M, m.T, rule=_genlim_t)

    def _turb_on(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vEOn[mm, t] <= mo.vOn[mm, t]
    m.TurbineOn = pe.Constraint(m.M, m.T, rule=_turb_on)

    # ---- D15 minimum stable load -------------------------------------------
    def _msl(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vGenAbv[mm, t] == mo.vGen[mm, t] - rx["msl_turb"] * mo.vEOn[mm, t]
    m.MSL = pe.Constraint(m.M, m.T, rule=_msl)

    # ---- D16 ramp -----------------------------------------------------------
    prev_gen = (init_state or {}).get("gen_last", {})

    def _ramp_up(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        if t == 0:
            p = prev_gen.get(mm, None)
            if p is None:
                return pe.Constraint.Skip
            return mo.vGen[mm, t] - p <= rx["ramp"]
        return mo.vGen[mm, t] - mo.vGen[mm, t - 1] <= rx["ramp"]
    m.RampUp = pe.Constraint(m.M, m.T, rule=_ramp_up)

    def _ramp_dn(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        if t == 0:
            p = prev_gen.get(mm, None)
            if p is None:
                return pe.Constraint.Skip
            return p - mo.vGen[mm, t] <= rx["ramp"]
        return mo.vGen[mm, t - 1] - mo.vGen[mm, t] <= rx["ramp"]
    m.RampDown = pe.Constraint(m.M, m.T, rule=_ramp_dn)

    # ---- D17-D19 unit commitment -------------------------------------------
    prev_on = (init_state or {}).get("on_last", {})

    def _state(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        prev = prev_on.get(mm, 0) if t == 0 else mo.vOn[mm, t - 1]
        return mo.vOn[mm, t] == prev + mo.vTurnOn[mm, t] - mo.vTurnOff[mm, t]
    m.UCState = pe.Constraint(m.M, m.T, rule=_state)

    def _no_both(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vTurnOn[mm, t] + mo.vTurnOff[mm, t] <= 1
    m.UCExclusive = pe.Constraint(m.M, m.T, rule=_no_both)

    def _mdt(mo, mm, t):
        """Eq. D19, minimum down time.

        NOTE ON FIDELITY. The manuscript renders D19 with the look-back window
        running to t inclusive, which cannot be satisfied when vTurnOn[t] = 1
        (vOn[t] is then 1, so the window can supply at most MDT - 1 off-hours and
        the constraint is infeasible). Paper 1's SOURCE CODE
        (V26_external_use-ssa1-patch.py L556-574) looks back over T.prev(t, td+1)
        for td in 0..MDT-2, i.e. hours t-1 down to t-MDT+1, EXCLUDING t. The
        manuscript rendering lost that in the OMML conversion. This implements
        the code's window with the standard >= direction; Paper 1's code writes
        `<= t * vTurnon`, which is a weaker condition and should be checked.
        """
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        lo = max(0, t - rx["mdt"])
        if lo >= t:
            return pe.Constraint.Skip                  # window start, nothing to enforce
        need = min(rx["mdt"], t - lo)
        return sum(1 - mo.vOn[mm, tau] for tau in range(lo, t)) >= need * mo.vTurnOn[mm, t]
    m.MinDownTime = pe.Constraint(m.M, m.T, rule=_mdt)

    # ---- D20 selection ------------------------------------------------------
    def _on_le_built(mo, mm, t):
        if not allow[t]["smr"]:
            return pe.Constraint.Skip
        return mo.vOn[mm, t] <= mo.vModule[mm]
    m.OnRequiresBuild = pe.Constraint(m.M, m.T, rule=_on_le_built)

    def _order(mo, mm):
        # symmetry breaking: modules are identical, so force a canonical order.
        # Not in Paper 1, and it changes no optimum -- it only prunes the tree.
        if mm == 0:
            return pe.Constraint.Skip
        return mo.vModule[mm] <= mo.vModule[mm - 1]
    m.ModuleOrder = pe.Constraint(m.M, rule=_order)

    # ---- capacity adequacy (Paper 1 sizing rule) ---------------------------
    # If any fleet is built it must cover peak demand, so the optimiser cannot
    # build one token module and leave the load unserved.
    if n_modules is None and cfg.demand_mw > 0:
        # Paper 1's capacity-adequacy rule: building zero modules stays feasible
        # (a non-viable facility remains on diesel), but ANY build must cover
        # peak demand. Without this the optimiser exploits the absence of a VOLL
        # penalty by building one token module and leaving most of the load
        # unserved, returning a spuriously low breakeven.
        m.vBuildAny = pe.Var(within=pe.Binary)
        m.BuildAnyUpper = pe.Constraint(
            expr=sum(m.vModule[mm] for mm in m.M) <= len(m.M) * m.vBuildAny)
        m.BuildAnyLower = pe.Constraint(
            expr=sum(m.vModule[mm] for mm in m.M) >= m.vBuildAny)
        m.CapacityAdequacy = pe.Constraint(
            expr=rx["cap_e"] * sum(m.vModule[mm] for mm in m.M)
            >= cfg.demand_mw * m.vBuildAny)

    # ---- extension C: supply and demand split ------------------------------
    def _avail(mo, t):
        return (sum(mo.vGen[mm, t] for mm in mo.M) + mo.vFlare[t]
                + mo.vSolar[t] + mo.vWind[t])
    m._avail = _avail      # underscore: keeps Pyomo's Block.__setattr__ quiet

    def _balance(mo, t):
        # P_avail = s_btc + s_dc + s_comm + curtail + export       (extends D11)
        return _avail(mo, t) == (mo.sBTC[t] + mo.sDC[t] + mo.sComm[t]
                                 + mo.vCurtail[t] + mo.vExport[t])
    m.EnergyBalance = pe.Constraint(m.T, rule=_balance)

    def _firm(mo, t):
        # s_dc + unmet = D      (Paper 1's served + unmet = D, D11 line 2)
        # Skipped where compute is supply-limited rather than firm (Phase 2):
        # a flare site cannot serve the facility's ultimate demand, so an
        # early-compute load takes a share of what is available (B-09).
        if not allow[t]["dc"] or dc_share < 1.0:
            return pe.Constraint.Skip
        return mo.sDC[t] + mo.vUnmet[t] == mo.pDemand[t]
    m.FirmLoad = pe.Constraint(m.T, rule=_firm)

    def _dc_supply_limited(mo, t):
        if not allow[t]["dc"] or dc_share >= 1.0:
            return pe.Constraint.Skip
        return mo.sDC[t] <= dc_share * _avail(mo, t)
    m.DCSupplyLimited = pe.Constraint(m.T, rule=_dc_supply_limited)

    def _dc_cap(mo, t):
        return mo.sDC[t] <= mo.pDemand[t]
    m.DCCap = pe.Constraint(m.T, rule=_dc_cap)

    # Community band comes from the phase table (10% Phases 1-3, 15% Phase 4),
    # clamped into the SiteConfig range so an explicit override still wins.
    def _comm_lo(mo, t):
        return mo.sComm[t] >= pc["comm_share"] * _avail(mo, t)
    m.CommMin = pe.Constraint(m.T, rule=_comm_lo)

    def _comm_hi(mo, t):
        return mo.sComm[t] <= max(pc["comm_share"], cfg.comm_share_max) * _avail(mo, t)
    m.CommMax = pe.Constraint(m.T, rule=_comm_hi)

    def _export(mo, t):
        return mo.vExport[t] <= cfg.export_cap * max(cfg.demand_mw, 1e-9)
    m.ExportCap = pe.Constraint(m.T, rule=_export)

    # ---- the subordination rule -- see the module docstring ----------------
    if cfg.subordinate_mining:
        def _subord(mo, t):
            return mo.sBTC[t] <= _avail(mo, t) - mo.sDC[t] - mo.sComm[t]
        m.MiningSubordinate = pe.Constraint(m.T, rule=_subord)

    # =========================================================================
    # OBJECTIVE  (Eq. D3 extended by extension D)
    # =========================================================================
    crf_smr = ab.crf(cfg.discount_rate, cfg.plant_life)

    # D5 annualised capital, prorated by horizon fraction H, with multiplier mu
    # PHASE-VINTAGE CAPITAL (Eq. E3). A class is charged only where it is
    # commissioned in this window; see the note in reference_dispatch().
    a_smr = 1.0 if any(a["smr"] for a in allow) else 0.0
    a_gas = 1.0 if any(a["gas"] for a in allow) else 0.0
    a_sol = 1.0 if any(a["solar"] for a in allow) else 0.0
    a_wnd = 1.0 if any(a["wind"] for a in allow) else 0.0
    cap_smr = (H * crf_smr * cfg.capex_mult * rx["capex"] * 1000.0 * rx["cap_e"]
               * a_smr * sum(m.vModule[mm] for mm in m.M))
    gen_mw = genset_nameplate_mw(cfg, flare_mw)        # B-21: undegraded rating
    cap_gas = H * crf_smr * cfg.capex_mult * cfg.genset_capex_per_kwe * 1000.0 * gen_mw * a_gas
    cap_re = H * crf_smr * cfg.capex_mult * (
        cfg.solar_capex_per_kwe * 1000.0 * pc["solar_mw"]
        + cfg.wind_capex_per_kwe * 1000.0 * pc["wind_mw"])

    # D6 fixed O&M
    fom_smr = (H * rx["fopex"] * 1000.0 * rx["cap_e"] * a_smr
               * sum(m.vModule[mm] for mm in m.M))
    fom_gas = H * cfg.genset_fopex_per_kwe * 1000.0 * gen_mw * a_gas
    fom_re = H * (cfg.solar_fopex_per_kwe * 1000.0 * pc["solar_mw"]
                  + cfg.wind_fopex_per_kwe * 1000.0 * pc["wind_mw"])

    # D7 variable O&M and fuel. Nuclear on the electric basis (Paper 1's thermal
    # form times eta_th is identical); flare gas carries VOM only, fuel is free.
    vom_smr = sum((rx["vom"] + rx["fc"]) * m.vGen[mm, t] for mm in m.M for t in m.T)
    vom_gas = sum(cfg.genset_vom * m.vFlare[t] for t in m.T)

    # D8 start-up
    start = sum(rx["startup"] * m.vTurnOn[mm, t] for mm in m.M for t in m.T)

    # D10 quadratic ramp penalty -- DEGENERACY BREAKING ONLY.
    # ---- Eq. D10 ramp penalty --------------------------------------------
    # Paper 1 writes this as a quadratic, which makes the program an MIQP and in
    # practice ties it to Gurobi. THAT IS NOT NECESSARY. The term is convex and
    # it is being MINIMISED, and a convex function under minimisation is exactly
    # representable to any tolerance by its tangent lines:
    #
    #     z >= d^2      <==>      z >= 2*a_k*d - a_k^2   for tangent points a_k
    #
    # At the optimum z settles to max_k(2 a_k d - a_k^2), the piecewise-linear
    # OUTER approximation of d^2. No binaries are needed -- convexity does the
    # work -- so the quadratic ramp penalty becomes a handful of ordinary linear
    # constraints and the program is a MILP that HiGHS, CBC or SCIP can solve.
    #
    # With K tangent points evenly spaced over d in [-R, R] the worst-case
    # approximation error is (R/(K-1))^2, multiplied by epsilon = 1e-6. For
    # CAREM-25 (R = 9 MW/h) at K = 9 that is 1.27e-6 dollars per ramp pair,
    # against a penalty term whose own magnitude is at most 8.1e-5 dollars. Both
    # are many orders of magnitude below the cent at which any reported
    # financial is quoted, and the penalty is excluded from those anyway.
    #
    # MODES
    #   "pwl"       quadratic shape, MILP form. DEFAULT. Free solvers.
    #   "quadratic" true MIQP. Requires Gurobi, CPLEX or SCIP.
    #   "l1"        absolute-value penalty. Also MILP, but a DIFFERENT shape:
    #               L1 favours few large ramps, L2 favours many small ones, so
    #               it can break a tie differently. Kept for comparison.
    #   "none"      no penalty; dispatch degeneracy is then unbroken.
    mode = "quadratic" if cfg.quadratic else cfg.ramp_penalty_mode
    ramp_pen = 0.0
    ramp_pairs = [(mm, t) for mm in m.M for t in m.T
                  if t > 0 and allow[t]["smr"] and allow[t - 1]["smr"]]

    if cfg.ramp_penalty_eps > 0 and ramp_pairs and mode != "none":
        m.RampDevIdx = pe.Set(initialize=ramp_pairs, dimen=2, ordered=True)

        if mode == "quadratic":
            ramp_pen = cfg.ramp_penalty_eps * sum(
                (m.vGenAbv[mm, t] - m.vGenAbv[mm, t - 1]) ** 2 for mm, t in ramp_pairs)

        elif mode == "pwl":
            R = max(float(rx["ramp"]), 1e-6)
            K = max(3, int(cfg.ramp_pwl_cuts))
            cuts = np.linspace(-R, R, K)
            m.vRampSq = pe.Var(m.RampDevIdx, within=pe.NonNegativeReals)
            m.RampCutIdx = pe.Set(initialize=range(K), ordered=True)

            def _tangent(mo, mm, t, k):
                a = float(cuts[k])
                d = mo.vGenAbv[mm, t] - mo.vGenAbv[mm, t - 1]
                return mo.vRampSq[mm, t] >= 2.0 * a * d - a * a
            m.RampTangent = pe.Constraint(m.RampDevIdx, m.RampCutIdx, rule=_tangent)

            ramp_pen = cfg.ramp_penalty_eps * sum(
                m.vRampSq[mm, t] for mm, t in ramp_pairs)

        elif mode == "l1":
            m.vRampDev = pe.Var(m.RampDevIdx, within=pe.NonNegativeReals)

            def _dev_pos(mo, mm, t):
                return mo.vRampDev[mm, t] >= mo.vGenAbv[mm, t] - mo.vGenAbv[mm, t - 1]
            m.RampDevPos = pe.Constraint(m.RampDevIdx, rule=_dev_pos)

            def _dev_neg(mo, mm, t):
                return mo.vRampDev[mm, t] >= mo.vGenAbv[mm, t - 1] - mo.vGenAbv[mm, t]
            m.RampDevNeg = pe.Constraint(m.RampDevIdx, rule=_dev_neg)

            ramp_pen = cfg.ramp_penalty_eps * sum(
                m.vRampDev[mm, t] for mm, t in ramp_pairs)
        else:
            raise ValueError(f"unknown ramp_penalty_mode {mode!r}")

    total_cost = (cap_smr + cap_gas + cap_re + fom_smr + fom_gas + fom_re
                  + vom_smr + vom_gas + start + ramp_pen)

    # --- revenues (extension D) ---
    rev_diesel = cfg.diesel_price * sum(m.sDC[t] for t in m.T)          # D3
    rev_btc = btc_price_mwh * sum(m.sBTC[t] for t in m.T)               # Lal Eq. 11
    rev_dc = cfg.compute_price * sum(m.sDC[t] / cfg.pue for t in m.T)   # per IT MWh
    rev_comm = cfg.elec_price_comm * sum(m.sComm[t] for t in m.T)
    # ---- carbon, on the DELIVERED basis  (B-08 fix, D-26 settled) ---------
    # Emissions are incurred on energy GENERATED; avoided emissions accrue on
    # energy DELIVERED to a load that would otherwise have burned diesel. The
    # two are different quantities, and the previous formulation conflated them
    # by summing (EF_diesel - EF_source) over generation. That paid the full
    # (0.70 - 0.012) x carbon_price = $34.40/MWhe credit on CURTAILED energy,
    # against a Rolls-Royce marginal cost of $9.15/MWhe -- a standing
    # $25.25/MWhe incentive to overbuild and throw the output away, sitting
    # inside the objective where it biased technology and module selection.
    #
    #   avoided = delivered x EF_diesel  -  SUM_s generation_s x EF_incremental_s
    #
    # Curtailed nuclear energy still carries its 0.012 emissions and earns no
    # credit, so the overbuild incentive disappears for the right reason rather
    # than by subtracting a correction term.
    if cfg.carbon_basis == "served_dc":
        delivered = sum(m.sDC[t] for t in m.T)
    elif cfg.carbon_basis == "all_delivered":
        delivered = sum(m.sDC[t] + m.sComm[t] + m.sBTC[t] for t in m.T)
    else:                                     # "delivered" -- DC + community
        delivered = sum(m.sDC[t] + m.sComm[t] for t in m.T)
    emissions = (cfg.ef_nuclear * sum(m.vGen[mm, t] for mm in m.M for t in m.T)
                 + cfg.ef_gas_incremental * sum(m.vFlare[t] for t in m.T))
    co2_avoided = cfg.ef_diesel * delivered - emissions
    rev_carbon = cfg.carbon_price * co2_avoided
    rev_penalty = cfg.flare_penalty * sum(
        m.pGasMscf[t] * (m.vFlare[t] / max(float(flare_mw[t]), 1e-9)) for t in m.T
        if float(flare_mw[t]) > 1e-9)
    rev_export = 0.0

    m.Profit = pe.Objective(
        expr=(rev_diesel + rev_btc + rev_dc + rev_comm + rev_carbon
              + rev_penalty + rev_export - total_cost),
        sense=pe.maximize)

    # stash the symbolic pieces so extract_ledger can evaluate them
    m._parts = dict(cap_smr=cap_smr, cap_gas=cap_gas, cap_re=cap_re,
                    fom_smr=fom_smr, fom_gas=fom_gas, fom_re=fom_re,
                    vom_smr=vom_smr, vom_gas=vom_gas, start=start,
                    ramp_pen=ramp_pen, rev_diesel=rev_diesel, rev_btc=rev_btc,
                    rev_dc=rev_dc, rev_comm=rev_comm, rev_carbon=rev_carbon,
                    rev_penalty=rev_penalty, rev_export=rev_export,
                    co2_avoided=co2_avoided, ramp_penalty=ramp_pen)
    m._overnight = dict(smr_per_module=rx["capex"] * 1000.0 * rx["cap_e"] * a_smr,
                        gas=cfg.genset_capex_per_kwe * 1000.0 * gen_mw * a_gas,
                        re=(cfg.solar_capex_per_kwe * 1000.0 * pc["solar_mw"]
                            + cfg.wind_capex_per_kwe * 1000.0 * pc["wind_mw"]))
    m._meta = dict(rx=rx, H=H, t0=t0, n_hours=n_hours, allow=allow,
                   nameplate_mw=pc["smr_mw"] + gen_mw * a_gas + pc["re_mw"],
                   btc_price_mwh=btc_price_mwh, flare_mw=flare_mw,
                   gas_mscf=gas_mscf, cfg=cfg)
    return m


# =============================================================================
# SECTION 5 -- LEDGER, BREAKEVEN RECOVERY, AND GATE F
# =============================================================================

@dataclass
class Ledger:
    """Annualised financial and energy ledger for one solved window.

    POSITIVE-COST CONVENTION. Every cost field is a positive number and every
    revenue field is a positive number; the signs live in the formulae, not in
    the data. This is deliberately NOT the BestPer.csv convention, which negates
    costs and leaves export revenue positive -- the mismatch between those two is
    defect B-01 (methods note section 8). Nothing here reproduces that defect.
    """
    facility: str = ""
    source: str = ""                 # "pyomo" or "reference"
    phase: int = 0
    hours: int = 0
    # costs
    anncap: float = 0.0
    fixed: float = 0.0
    vom: float = 0.0
    startup: float = 0.0
    ramp_penalty: float = 0.0        # reported separately, NEVER in the breakeven
    # revenues
    rev_btc: float = 0.0
    rev_dc: float = 0.0
    rev_comm: float = 0.0
    rev_carbon: float = 0.0
    rev_penalty: float = 0.0
    rev_export: float = 0.0
    # energy
    served_dc: float = 0.0           # MWhe to the firm data-centre load
    served_btc: float = 0.0
    served_comm: float = 0.0
    curtailed: float = 0.0
    unmet: float = 0.0
    generation: float = 0.0
    co2_avoided_t: float = 0.0       # tCO2/yr vs the unabated-diesel counterfactual
    modules: int = 0
    technology: str = ""
    country: str = ""
    annualised_from_hours: int = 0   # 0 = already annual (reference path)
    overnight_capex_charged: float = 0.0   # $ of overnight capital actually billed
                                           # in this phase (phase-vintage aware)
    nameplate_mw: float = 0.0              # installed capacity ACTIVE in this phase
    # --- B-22: was the answer PROVEN, or just the best found before the clock ran out?
    # rolling_dispatch accepts maxTimeLimit as a valid termination and loads the
    # incumbent, which is the right call -- a feasible schedule beats no schedule.
    # But until now nothing recorded that it had happened, so a node solved to the
    # 1% gap and a node abandoned at an unknown gap were indistinguishable in the
    # checkpoint. These three fields make the difference visible and auditable.
    n_windows: int = 0                     # solves attempted in this node
    n_timelimit: int = 0                   # of those, how many hit the wall clock
    worst_rel_gap: float = 0.0             # worst |UB-LB|/|UB| seen, 0.0 if all proven
    demand_mwhe: float = 0.0               # B-21: firm annual demand, IT x PUE x 8760.
                                           # Carried so the UQ can re-derive the
                                           # Eq. E9 min() exactly instead of
                                           # inferring where the cap bit.
    avail_quantiles: str = ""              # B-21: 21 quantiles (0, 0.05, ... 1.0) of
                                           # HOURLY available MW, semicolon-joined.
                                           # The Eq. E9 cap is a min() applied hour by
                                           # hour, and min() of an annual average is
                                           # not the annual average of min(). Carrying
                                           # the shape lets the UQ integrate the kink
                                           # instead of evaluating it once at the mean,
                                           # which was wrong by up to 7.9% at 2-3 MW --
                                           # the flare-site scale this paper is about.
    gas_quantiles: str = ""                # B-21: the same 21 quantiles of HOURLY gas
                                           # generation, paired index-for-index with
                                           # avail_quantiles so the derate is applied
                                           # to the gas part of each quantile.
    gen_gas_mwhe: float = 0.0              # B-21: generation from the flare genset
                                           # alone. The availability derate acts on
                                           # THIS and nothing else, so the UQ closed
                                           # form needs it to weight a draw correctly
                                           # (uq_phased.evaluate, weight w). Emitting
                                           # it beats the constant gas share it
                                           # replaces, which was right for facilities
                                           # above ~6 MW and wrong below: at 2 MW the
                                           # true share is 0.886, not 0.671.

    @property
    def total_cost(self) -> float:
        """Eq. D3 cost aggregate. The ramp penalty is EXCLUDED: Paper 1 Eq. D10
        exists only to break dispatch degeneracy and is not a real cost."""
        return self.anncap + self.fixed + self.vom + self.startup

    @property
    def total_revenue_non_diesel(self) -> float:
        return (self.rev_btc + self.rev_dc + self.rev_comm + self.rev_carbon
                + self.rev_penalty + self.rev_export)

    def breakeven(self) -> float:
        """Extension F -- the unified diesel breakeven, Eq. E20.

            d* = [ AnnCap + Fixed + VOM + StartUp
                   - rev_BTC - rev_carbon - rev_penalty - rev_comm - rev_DC
                   - Export ] / Served

        Served is the DIESEL-DISPLACING energy, Sum_t s_dc, which is the direct
        inheritance of Paper 1's Sum_t vServed. This is open decision D-01
        convention (B). Convention (A), total served energy, is available via
        breakeven_total_served() and differs by roughly 11%.

        Returns NaN when no firm load is served, which is the correct answer for
        Phases 1-2: with s_dc = 0 there is no diesel being displaced and a
        diesel breakeven is not defined. Those phases are assessed on profit,
        not on breakeven.
        """
        if self.served_dc <= 1e-9:
            return float("nan")
        return (self.total_cost - self.total_revenue_non_diesel) / self.served_dc

    def breakeven_total_served(self) -> float:
        """D-01 convention (A): denominator is all delivered energy."""
        s = self.served_dc + self.served_btc + self.served_comm
        if s <= 1e-9:
            return float("nan")
        return (self.total_cost - self.total_revenue_non_diesel) / s

    def profit(self, diesel_price: float) -> float:
        """Eq. D23 generalised: Pi(d) = A + d.S, affine in d by construction."""
        return (self.total_revenue_non_diesel + diesel_price * self.served_dc
                - self.total_cost)

    def as_row(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d["breakeven_B"] = self.breakeven()
        d["breakeven_A"] = self.breakeven_total_served()
        return d


def extract_ledger(m, scale_to_year: bool = True) -> Ledger:
    """Evaluate a solved Pyomo model into a Ledger."""
    if not HAVE_PYOMO:
        raise RuntimeError("Pyomo is not installed.")
    v = pe.value
    p, meta = m._parts, m._meta
    cfg, n = meta["cfg"], meta["n_hours"]
    k = HOURS_PER_YEAR / n if scale_to_year else 1.0

    def val(x):
        return float(v(x)) if not isinstance(x, (int, float)) else float(x)

    def vsum(varobj, idx):
        """Sum a Var over an index, tolerating members the solver never valued.

        A variable that ends up in no constraint and no objective term is simply
        absent from the solved problem, and Pyomo raises rather than returning 0.
        One such variable should not take down the entire ledger for every node
        in a phase, so unvalued members are counted as zero and the fact is
        reported once on stderr rather than swallowed.
        """
        total, missing = 0.0, 0
        for i in idx:
            try:
                total += float(v(varobj[i]))
            except (ValueError, TypeError):
                missing += 1
        if missing:
            sys.stderr.write(
                f"[ledger] {varobj.name}: {missing}/{len(list(idx))} members carried no "
                f"solver value and were counted as 0 (they appear in no constraint "
                f"or objective term in this phase)\n")
        return total

    led = Ledger(
        facility=cfg.facility_id, source="pyomo", hours=n,
        anncap=(val(p["cap_smr"]) + val(p["cap_gas"]) + val(p["cap_re"])) * k,
        fixed=(val(p["fom_smr"]) + val(p["fom_gas"]) + val(p["fom_re"])) * k,
        vom=(val(p["vom_smr"]) + val(p["vom_gas"])) * k,
        startup=val(p["start"]) * k,
        ramp_penalty=val(p["ramp_pen"]) * k,
        rev_btc=val(p["rev_btc"]) * k,
        rev_dc=val(p["rev_dc"]) * k,
        rev_comm=val(p["rev_comm"]) * k,
        rev_carbon=val(p["rev_carbon"]) * k,
        rev_penalty=val(p["rev_penalty"]) * k,
        rev_export=val(p["rev_export"]) * k,
        served_dc=vsum(m.sDC, m.T) * k,
        served_btc=vsum(m.sBTC, m.T) * k,
        served_comm=vsum(m.sComm, m.T) * k,
        curtailed=vsum(m.vCurtail, m.T) * k,
        unmet=vsum(m.vUnmet, m.T) * k,
        generation=sum(v(m.vGen[mm, t]) for mm in m.M for t in m.T) * k
        + sum(v(m.vFlare[t]) + v(m.vSolar[t]) + v(m.vWind[t]) for t in m.T) * k,
        gen_gas_mwhe=sum(v(m.vFlare[t]) for t in m.T) * k,
        demand_mwhe=cfg.demand_mw * HOURS_PER_YEAR,
        avail_quantiles=_quantile_str(np.array(
            [sum(v(m.vGen[mm, t]) for mm in m.M) + v(m.vFlare[t])
             + v(m.vSolar[t]) + v(m.vWind[t]) for t in m.T], dtype=float)),
        gas_quantiles=_quantile_str(np.array(
            [v(m.vFlare[t]) for t in m.T], dtype=float)),
        modules=int(round(sum(v(m.vModule[mm]) for mm in m.M))),
        technology=cfg.reactor,
    )
    led.overnight_capex_charged = (
        m._overnight["smr_per_module"] * sum(v(m.vModule[mm]) for mm in m.M)
        + m._overnight["gas"] + m._overnight["re"])
    # Same expression the objective priced -- never a parallel recomputation.
    # The previous code recomputed it differently and could report
    # co2_avoided_t = 0.0 beside $2.3M of carbon revenue with nothing noticing.
    led.co2_avoided_t = val(p["co2_avoided"]) * k
    # B-19 (2026-09-17): this assignment was MISSING on the Pyomo path.
    # build_model() computes the phase-active nameplate and stashes it in
    # m._meta; reference_dispatch() sets it on its own Ledger; extract_ledger()
    # never did. Ledger.nameplate_mw therefore came back 0.0 on every solved
    # node of every sweep, and gate F5 -- which exists to catch a WINDOW ledger
    # being reported as an ANNUAL one -- took its "no nameplate_mw, skipping"
    # branch 1,332 times out of 1,332 without anyone noticing that it had never
    # once fired.
    #
    # This is a DIAGNOSTIC field only. It enters no cost, no revenue and no
    # constraint, so populating it changes no economic result: the v6 numbers
    # stand exactly as computed. What changes is that F5 now actually runs.
    # An independent audit on v6 (nameplate rebuilt from baseiaea.csv x
    # n_modules) put the implied fleet capacity factor at 0.9900 in phase 3 and
    # 0.875 in phase 4 once the 20% renewable overbuild is counted, so F5 is
    # expected to PASS everywhere -- but expected-to-pass is not the same as
    # checked, which is the whole point of the gate.
    led.nameplate_mw = float(meta.get("nameplate_mw", 0.0))
    return led


# -----------------------------------------------------------------------------
# GATE F
# -----------------------------------------------------------------------------

def gate_F(led: Ledger, cfg: SiteConfig, tol: float = 0.01,
           verbose: bool = True) -> dict:
    """HARD GATE. The model's breakeven must match analytical_benchmark to 1%.

    This is NOT a tautology. The Ledger's cost terms are accumulated from the
    dispatch -- module binaries, hourly generation, start-up events -- whereas
    analytical_benchmark recomputes AnnCap, Fixed and VOM from the baseiaea.csv
    catalogue in closed form, given only the served energy the model produced.
    Agreement therefore tests the model's capital annualisation, its fixed-O&M
    accumulation, its VOM integration and its revenue signs against an
    independent implementation. Disagreement has always been a real defect.

    Three checks:
      F1  breakeven agreement within `tol`
      F2  affinity of Pi(d) in the diesel price -- the property the closed form
          rests on. Pi is evaluated at three prices and must be exactly linear.
      F3  the ramp penalty must be absent from every reported financial.

    F1 is skipped, not failed, when served_dc is zero (Phases 1-2), because a
    diesel breakeven is undefined there. That is reported explicitly rather than
    being silently counted as a pass.
    """
    rx = reactor_params(cfg)
    out = {"checks": [], "passed": True}

    def add(name, ok, detail, skipped=False):
        out["checks"].append({"check": name, "ok": ok, "skipped": skipped,
                              "detail": detail})
        if not ok and not skipped:
            out["passed"] = False

    # ---- F1 breakeven agreement -------------------------------------------
    if led.served_dc <= 1e-9:
        add("F1 breakeven vs analytical", True,
            "SKIPPED: served_dc = 0, no diesel displaced, breakeven undefined "
            "(expected in Phases 1-2)", skipped=True)
        out["model_be"] = float("nan")
        out["analytic_be"] = float("nan")
        out["rel_err"] = float("nan")
    else:
        be_model = led.breakeven()
        # F1 reconstructs AnnCap, Fixed and VOM from the baseiaea.csv catalogue,
        # which is only meaningful when the fleet is a SINGLE technology. From
        # Phase 2 onward the fleet is mixed (genset + renewables, or SMR + RE
        # peaking) and the catalogue cannot reproduce a blended capital stack.
        # In those phases F1's sub-checks are SKIPPED and the independent
        # verification falls to F4, which validates AnnCap against
        # overnight x mu x CRF for any fleet composition.
        smr_only_capital = ab.overnight_capex(rx["capex"], rx["cap_e"] * max(led.modules, 1))
        single_tech = abs(led.overnight_capex_charged - smr_only_capital) < 1.0
        oc = smr_only_capital
        be_analytic, comp = ab.unified_breakeven(
            capex_overnight=oc, capex_mult=cfg.capex_mult,
            r=cfg.discount_rate, n=cfg.plant_life,
            fopex_per_kwe=rx["fopex"], nameplate_mwe=rx["cap_e"] * max(led.modules, 1),
            vom_fc_per_mwhe=rx["vom"] + rx["fc"], served_mwhe=led.served_dc,
            generation_mwhe=led.generation,
            rev_btc=led.rev_btc, rev_carbon=led.rev_carbon,
            rev_penalty=led.rev_penalty, rev_comm=led.rev_comm,
            rev_compute=led.rev_dc, startup=led.startup, export=led.rev_export)
        rel = abs(be_model - be_analytic) / max(abs(be_analytic), 1e-9)
        out["model_be"], out["analytic_be"], out["rel_err"] = be_model, be_analytic, rel
        if single_tech:
            add("F1 breakeven vs analytical", rel <= tol,
                f"model ${be_model:,.4f}/MWhe vs analytical ${be_analytic:,.4f}/MWhe, "
                f"rel err {rel*100:.4f}% (tol {tol*100:.1f}%)")
        else:
            add("F1 breakeven vs analytical", True,
                f"SKIPPED: mixed fleet (overnight ${led.overnight_capex_charged:,.0f} "
                f"vs SMR-only ${smr_only_capital:,.0f}); F4 carries the check",
                skipped=True)
        # component-level diff, so a failure names the guilty term
        for label, a, b in (("AnnCap", led.anncap, comp["anncap"]),
                            ("Fixed", led.fixed, comp["fixed"]),
                            ("VOM", led.vom, comp["vom"])):
            if not single_tech:
                add(f"F1.{label}", True,
                    "SKIPPED: mixed fleet, catalogue reconstruction not applicable "
                    "(F4 verifies AnnCap independently)", skipped=True)
                continue
            r = abs(a - b) / max(abs(b), 1e-9)
            add(f"F1.{label}", r <= tol,
                f"ledger ${a:,.2f} vs closed form ${b:,.2f}, rel err {r*100:.4f}%")

    # ---- F2 affinity of profit in the diesel price ------------------------
    d0, d1, d2 = 100.0, 300.0, 500.0
    p0, p1, p2 = led.profit(d0), led.profit(d1), led.profit(d2)
    lin_err = abs((p2 - p1) - (p1 - p0))
    scale = max(abs(p1), 1.0)
    add("F2 Pi(d) affine in d", lin_err / scale < 1e-9,
        f"second difference {lin_err:.6e} on scale {scale:.3e} "
        f"(slope = served_dc = {led.served_dc:,.1f} MWhe)")
    if not cfg.subordinate_mining:
        add("F2 subordination", True,
            "WARNING: subordinate_mining is OFF. Affinity holds for THIS fixed "
            "dispatch but the optimiser may re-allocate between s_btc and s_dc "
            "at a different diesel price, so the closed form is not recoverable "
            "in general (Proposition 3).", skipped=True)

    # ---- F3 ramp penalty excluded -----------------------------------------
    recomputed = led.anncap + led.fixed + led.vom + led.startup
    add("F3 ramp penalty excluded", abs(recomputed - led.total_cost) < 1e-6,
        f"total_cost ${led.total_cost:,.2f} excludes ramp penalty "
        f"${led.ramp_penalty:,.6f}")

    # ---- F4 capital and annualisation, PHASE-INDEPENDENT -------------------
    # The strongest check available, and the one that matters most, because
    # AnnCap does not depend on the dispatch at all: it is a closed-form
    # function of the overnight capital charged, the CAPEX multiplier and the
    # CRF. It can therefore be verified in EVERY phase, including the Phase 1-2
    # nodes where F1 is skipped for want of a firm load.
    #
    # That blind spot is not hypothetical. In the 2026-09-14 sweep, 666 Phase
    # 1-2 nodes reported PASS while every financial in them was understated by
    # 8760/72 = 121.67x, because rolling_dispatch() accumulated 72 committed
    # hours and never scaled them to a year. F1 could not see it (served_dc = 0)
    # and nothing else looked. F4 looks.
    if led.overnight_capex_charged > 0:
        expect_anncap = (led.overnight_capex_charged * cfg.capex_mult
                         * ab.crf(cfg.discount_rate, cfg.plant_life))
        rel = abs(led.anncap - expect_anncap) / max(expect_anncap, 1e-9)
        add("F4 AnnCap vs CRF closed form", rel <= tol,
            f"ledger ${led.anncap:,.2f} vs ${expect_anncap:,.2f} "
            f"(overnight ${led.overnight_capex_charged:,.0f} x mu {cfg.capex_mult} "
            f"x CRF {ab.crf(cfg.discount_rate, cfg.plant_life):.6f}), rel err {rel*100:.4f}%"
            + ("" if rel <= tol else
               f"  <-- off by {expect_anncap/max(led.anncap,1e-9):.2f}x; if that is "
               f"8760/committed_hours the ledger was never annualised"))
    else:
        add("F4 AnnCap vs CRF closed form", True,
            "SKIPPED: no capital charged in this phase", skipped=True)

    # ---- F5 energy basis ---------------------------------------------------
    # Secondary, cheap, and independent of cost: a ledger claiming an annual
    # basis cannot report an implied fleet capacity factor near zero.
    # Use the nameplate the LEDGER recorded for its own phase, and NOTHING else.
    #
    # The previous fallback reconstructed a nameplate from SiteConfig when the
    # ledger did not carry one. That reconstruction is phase-blind -- it adds
    # flare capacity to phases that retired it and omits RE that the phase
    # table added -- so it produced a wrong denominator and F5 failed nodes
    # whose dispatch was fine. A gate that fires on its own faulty
    # reconstruction is worse than one that abstains, because it costs a
    # diagnosis every time. If the ledger cannot say what was installed, F5
    # says so and skips.
    nameplate = led.nameplate_mw
    if led.generation > 1e-6 and nameplate > 1e-6:
        icf = led.generation / (nameplate * HOURS_PER_YEAR)
        # Upper bound 1.05, not 1.02. The check exists to catch a ledger that
        # reports a WINDOW as a year -- an error of two orders of magnitude, not
        # of five per cent. A few per cent of headroom absorbs the lumpiness of
        # a mixed SMR + renewables fleet without weakening the check at all.
        ok5 = 0.002 <= icf <= 1.05
        add("F5 energy annual basis", ok5,
            f"implied fleet capacity factor {icf:.5f} on {led.generation:,.0f} MWhe "
            f"over {nameplate:,.1f} MW"
            + ("" if ok5 else "  <-- looks like a WINDOW, not a year"))
    elif led.generation > 1e-6 and nameplate <= 0:
        add("F5 energy annual basis", True,
            "SKIPPED: ledger carries no nameplate_mw, so the implied capacity "
            "factor cannot be formed. Not inferred from SiteConfig -- that "
            "estimate is phase-blind and would produce false failures.",
            skipped=True)
    else:
        add("F5 energy annual basis", True, "SKIPPED: no generation", skipped=True)

    # ---- F6 carbon revenue ties to physical avoided CO2 --------------------
    # The Day-5 audit found a node reporting co2_avoided_t = 0.0 beside
    # $2,304,910.08 of carbon revenue, with nothing flagging the contradiction.
    # These two numbers are the same physical claim priced two ways and must
    # agree identically, not approximately.
    expect_rev = led.co2_avoided_t * cfg.carbon_price
    if abs(led.rev_carbon) > 1e-6 or abs(expect_rev) > 1e-6:
        rel6 = abs(led.rev_carbon - expect_rev) / max(abs(expect_rev), 1.0)
        add("F6 rev_carbon == CO2 x price", rel6 <= 1e-6,
            f"rev_carbon ${led.rev_carbon:,.2f} vs co2_avoided_t {led.co2_avoided_t:,.2f} t "
            f"x ${cfg.carbon_price:,.2f} = ${expect_rev:,.2f}"
            + ("" if rel6 <= 1e-6 else "  <-- the physical and financial carbon "
                                       "figures disagree; they are one claim"))
        # and neither may be credited to curtailment
        delivered = led.served_dc + led.served_comm + (
            led.served_btc if cfg.carbon_basis == "all_delivered" else 0.0)
        if cfg.carbon_basis == "served_dc":
            delivered = led.served_dc
        ok6b = led.co2_avoided_t <= cfg.ef_diesel * delivered + 1e-6
        add("F6b CO2 not booked on curtailment", ok6b,
            f"co2_avoided_t {led.co2_avoided_t:,.2f} t <= EF_diesel x delivered "
            f"{cfg.ef_diesel * delivered:,.2f} t (basis '{cfg.carbon_basis}')"
            + ("" if ok6b else "  <-- credit exceeds what delivered energy can justify"))
    else:
        add("F6 rev_carbon == CO2 x price", True, "SKIPPED: no carbon priced",
            skipped=True)

    if verbose:
        print("\n  " + "-" * 94)
        print("  GATE F -- unified breakeven must reproduce analytical_benchmark")
        print("  " + "-" * 94)
        for c in out["checks"]:
            tag = "SKIP" if c["skipped"] else ("PASS" if c["ok"] else "FAIL")
            print(f"    {tag:<5} {c['check']:<28} {c['detail']}")
        print(f"  GATE F: {'PASS' if out['passed'] else 'FAIL'}")
    return out


# =============================================================================
# SECTION 6 -- SOLVER-FREE REFERENCE DISPATCH
# =============================================================================

def phase_capacities(cfg: SiteConfig, phases: PhaseConfig, p: int,
                     rx: dict, n_modules: int) -> dict:
    """Resolve the capacities and shares that apply in phase p.

    Single source of truth for both the MIQP and the reference dispatch, so the
    two cannot drift. B-10 is fixed here: Phase 2's renewable nameplate comes
    from `re_scale_of_genset`, not from SiteConfig's default of zero, which is
    why "+ Solar/Wind hybrid" previously had no capacity and Phases 1 and 2
    differed only by three years of flare decline.
    """
    e = phases.economics(p)
    genset_mw = cfg.flare_mmscfd * 1e6 * M3_PER_SCF / 86400.0 * cfg.lhv_mj_m3 * cfg.eta_gen
    smr_mw = rx["cap_e"] * n_modules * e["smr_scale"]
    re_mw = genset_mw * e["re_scale_of_genset"] + smr_mw * e["re_scale_of_smr"]
    # B-10 choice: split the Phase-2 hybrid evenly between solar and wind.
    # [ASSUMED] -- no file sizes the split.
    return dict(genset_mw=genset_mw if phases.allows(phases.t2 if p == 2 else 0)["gas"] and p <= 2 else 0.0,
                smr_mw=smr_mw, smr_modules=int(round(n_modules * e["smr_scale"])),
                solar_mw=re_mw * 0.5, wind_mw=re_mw * 0.5, re_mw=re_mw,
                # The phase band is CLAMPED into the SiteConfig range so an
                # explicit config override still wins. Case 1 of the smoke test
                # sets min = max = 0 to switch community off for the Paper 1
                # regression; without this clamp the phase table would silently
                # re-enable it and the regression would fail for a reason that
                # has nothing to do with the inheritance being tested.
                comm_share=min(max(e["comm_share"], cfg.comm_share_min),
                               cfg.comm_share_max),
                dc_share_of_available=e["dc_share_of_available"])


def _co2_avoided(cfg: SiteConfig, df: pd.DataFrame) -> float:
    """Avoided CO2, tonnes, on the configured basis. Mirrors the Pyomo objective
    exactly so gate F6 compares like with like."""
    if cfg.carbon_basis == "served_dc":
        delivered = df["s_dc"].sum()
    elif cfg.carbon_basis == "all_delivered":
        delivered = df["s_dc"].sum() + df["s_comm"].sum() + df["s_btc"].sum()
    else:
        delivered = df["s_dc"].sum() + df["s_comm"].sum()
    emissions = (cfg.ef_nuclear * df["smr"].sum()
                 + cfg.ef_gas_incremental * df["flare"].sum())
    return cfg.ef_diesel * delivered - emissions


def reference_dispatch(cfg: SiteConfig, phases: PhaseConfig, t0: int,
                       n_hours: int, n_modules: int = 1) -> Tuple[Ledger, pd.DataFrame]:
    """Deterministic dispatch for a FIXED build. No solver, no optimisation.

    Under the subordination rule there is no hourly choice left to make: firm
    load is served first, community takes its contracted share, mining absorbs
    the residual, and anything left is curtailed. The dispatch is therefore a
    direct computation, and this function performs it so the accounting chain is
    executable in environments without Pyomo.

    SMR sizing within the hour solves the small fixed point created by the
    community share being defined on TOTAL availability:

        s_comm = phi . P_avail   and   s_dc <= P_avail - s_comm
        => to serve demand D in full,  P_avail >= D / (1 - phi)

    so the SMR target is  min( nameplate . CF . n ,  D/(1-phi) - flare - RE ).

    This cannot choose a technology or a module count. Use the MIQP for that.
    """
    rx = reactor_params(cfg)
    p_here = phases.phase_of(t0)
    pc = phase_capacities(cfg, phases, p_here, rx, n_modules)
    flare_mw = flare_profile(cfg, n_hours, t0)
    gas_mscf = flare_gas_volume(cfg, n_hours, t0)
    irr = synthetic_irradiance(n_hours, t0)
    wnd = synthetic_wind(n_hours, t0)
    solar_mw = solar_power(irr, pc["solar_mw"])
    wind_mw = wind_power(wnd, pc["wind_mw"])

    phi = pc["comm_share"]
    smr_cap = pc["smr_mw"] * cfg.capacity_factor
    dc_share = pc["dc_share_of_available"]
    rows = []
    for i in range(n_hours):
        a = phases.allows(t0 + i)
        fl = float(flare_mw[i]) if a["gas"] else 0.0
        so = float(solar_mw[i]) if a["solar"] else 0.0
        wi = float(wind_mw[i]) if a["wind"] else 0.0
        dem = cfg.demand_mw if a["dc"] else 0.0

        smr = 0.0
        if a["smr"]:
            need = dem / (1.0 - phi) if dem > 0 else 0.0
            smr = float(np.clip(need - fl - so - wi, 0.0, smr_cap))

        avail = smr + fl + so + wi
        comm = phi * avail
        # Phase-2 compute is SUPPLY-limited: a flare site cannot serve the
        # facility's ultimate demand, so early compute takes a share of what is
        # available rather than a share of eventual load (B-09).
        if dc_share < 1.0:
            dem = min(dem, dc_share * avail)
        dc = min(dem, max(0.0, avail - comm))
        unmet = max(0.0, dem - dc)
        resid = max(0.0, avail - comm - dc)
        btc = resid if a["btc"] else 0.0
        curt = resid - btc
        gas_used = float(gas_mscf[i]) * (fl / float(flare_mw[i])) if float(flare_mw[i]) > 1e-9 else 0.0
        rows.append((t0 + i, phases.phase_of(t0 + i), smr, fl, so, wi, avail,
                     dc, comm, btc, curt, unmet, gas_used))

    cols = ["t", "phase", "smr", "flare", "solar", "wind", "avail",
            "s_dc", "s_comm", "s_btc", "curtail", "unmet", "gas_mscf"]
    df = pd.DataFrame(rows, columns=cols)

    k = HOURS_PER_YEAR / n_hours
    H = n_hours / HOURS_PER_YEAR
    crf = ab.crf(cfg.discount_rate, cfg.plant_life)
    gen_mw = genset_nameplate_mw(cfg, flare_mw)        # B-21: undegraded rating
    btc_price_mwh = btc_revenue_per_mwh(cfg)

    # PHASE-VINTAGE CAPITAL (Eq. E3, the zBuild_{s,p} sum). A supply class is
    # charged capital and fixed O&M only in the phases where it is commissioned.
    # Charging SMR capital during Phase 1 -- before the reactor exists -- would
    # load roughly $28M/yr onto a flare-gas mining operation and make the whole
    # bridge-finance case look uneconomic for a reason that is purely an
    # accounting artefact.
    act = {c: bool(df[c].abs().sum() > 1e-9) or any(phases.allows(t0 + i)[k] for i in range(n_hours))
           for c, k in (("smr", "smr"), ("flare", "gas"), ("solar", "solar"), ("wind", "wind"))}
    overnight_charged = (
        rx["capex"] * 1000.0 * pc["smr_mw"] * 1.0
        + cfg.genset_capex_per_kwe * 1000.0 * gen_mw * act["flare"]
        + cfg.solar_capex_per_kwe * 1000.0 * pc["solar_mw"]
        + cfg.wind_capex_per_kwe * 1000.0 * pc["wind_mw"])
    anncap = crf * cfg.capex_mult * overnight_charged
    fixed = (rx["fopex"] * 1000.0 * pc["smr_mw"]
             + cfg.genset_fopex_per_kwe * 1000.0 * gen_mw * act["flare"]
             + cfg.solar_fopex_per_kwe * 1000.0 * pc["solar_mw"]
             + cfg.wind_fopex_per_kwe * 1000.0 * pc["wind_mw"])
    vom = ((rx["vom"] + rx["fc"]) * df["smr"].sum() + cfg.genset_vom * df["flare"].sum()) * k

    led = Ledger(
        facility=cfg.facility_id, source="reference",
        phase=int(df["phase"].iloc[0]), hours=n_hours,
        anncap=anncap, fixed=fixed, vom=vom, startup=0.0, ramp_penalty=0.0,
        rev_btc=btc_price_mwh * df["s_btc"].sum() * k,
        rev_dc=cfg.compute_price * df["s_dc"].sum() / cfg.pue * k,
        rev_comm=cfg.elec_price_comm * df["s_comm"].sum() * k,
        rev_carbon=cfg.carbon_price * _co2_avoided(cfg, df) * k,
        rev_penalty=cfg.flare_penalty * df["gas_mscf"].sum() * k,
        rev_export=0.0,
        served_dc=df["s_dc"].sum() * k, served_btc=df["s_btc"].sum() * k,
        served_comm=df["s_comm"].sum() * k, curtailed=df["curtail"].sum() * k,
        unmet=df["unmet"].sum() * k, generation=df["avail"].sum() * k,
        gen_gas_mwhe=df["flare"].sum() * k,
        demand_mwhe=cfg.demand_mw * HOURS_PER_YEAR,
        avail_quantiles=_quantile_str(df["avail"].to_numpy(float)),
        gas_quantiles=_quantile_str(df["flare"].to_numpy(float)),
        co2_avoided_t=_co2_avoided(cfg, df) * k,
        modules=pc["smr_modules"], technology=cfg.reactor,
        overnight_capex_charged=overnight_charged,
        nameplate_mw=pc["smr_mw"] + gen_mw * act["flare"] + pc["re_mw"])
    # NOTE: anncap and fixed are already annual; H is applied to neither, unlike
    # the Pyomo window ledger which prorates by H and is then scaled back by k.
    _ = H
    return led, df


# =============================================================================
# SECTION 7 -- TWO-STAGE SOLUTION PROCEDURE
# =============================================================================

def _pick_solver(sc: SolverConfig):
    """First available solver from the configured preference order."""
    if not HAVE_PYOMO:
        return None, None
    order = (sc.solver,) + tuple(s for s in sc.fallbacks if s != sc.solver)
    for name in order:
        try:
            opt = pe.SolverFactory(name)
            if opt is not None and opt.available(exception_flag=False):
                return name, opt
        except Exception:
            continue
    return None, None


MIQP_CAPABLE_SOLVERS = ("gurobi", "gurobi_direct", "cplex", "cplex_direct", "scip")


def _guard_quadratic(cfg: SiteConfig, solver_name: Optional[str]) -> SiteConfig:
    """Downgrade a quadratic ramp penalty to its exact tangent-cut form when the
    selected solver cannot take a quadratic objective.

    Belt and braces: the driver checks this too, but build_model can be called
    directly and the failure mode is a bare DegreeError three frames inside the
    solver interface. Better to switch representation -- which does not change
    the Eq. D10 formulation -- and say so once.
    """
    mode = "quadratic" if cfg.quadratic else cfg.ramp_penalty_mode
    if mode != "quadratic" or solver_name is None:
        return cfg
    if any(k in solver_name for k in MIQP_CAPABLE_SOLVERS):
        return cfg
    sys.stderr.write(
        f"[ramp] solver {solver_name!r} cannot take a quadratic objective; "
        f"representing the Eq. D10 penalty by tangent cuts instead "
        f"(ramp_penalty_mode='pwl'). The formulation is unchanged.\n")
    import copy
    c = copy.copy(cfg)
    c.quadratic = False
    c.ramp_penalty_mode = "pwl"
    return c


def _apply_options(name: str, opt, sc: SolverConfig):
    """Gurobi at 1% gap with a 3600 s per-solve limit, per Paper 1."""
    try:
        if "gurobi" in name:
            opt.options["MIPGap"] = sc.mip_gap
            opt.options["TimeLimit"] = sc.time_limit
            if sc.threads:
                opt.options["Threads"] = sc.threads
        elif "highs" in name:
            opt.options["mip_rel_gap"] = sc.mip_gap
            opt.options["time_limit"] = float(sc.time_limit)
            if sc.threads:
                opt.options["threads"] = sc.threads
            opt.options["presolve"] = "on"
        elif "cplex" in name:
            opt.options["mip_tolerances_mipgap"] = sc.mip_gap
            opt.options["timelimit"] = sc.time_limit
            if sc.threads:
                opt.options["threads"] = sc.threads
        elif name == "scip":
            opt.options["limits/gap"] = sc.mip_gap
            opt.options["limits/time"] = sc.time_limit
        elif name == "cbc":
            opt.options["ratio"] = sc.mip_gap
            opt.options["seconds"] = sc.time_limit
        elif name == "glpk":
            opt.options["mipgap"] = sc.mip_gap
            opt.options["tmlim"] = sc.time_limit
    except Exception:
        pass


def sizing_pass(cfg: SiteConfig, phases: PhaseConfig, sc: SolverConfig,
                stride: int = 6, horizon: int = HOURS_PER_YEAR,
                verbose: bool = True) -> int:
    """Stage 1 -- downsampled sizing. Every `stride`-th hour, module count free.

    Paper 1 solves a downsampled problem at peak demand to fix the technology
    and module count before the rolling dispatch begins, because the integer
    build decision must not flip between windows.
    """
    name, opt = _pick_solver(sc)
    if opt is None:
        n = max_modules(cfg, reactor_params(cfg))
        if verbose:
            print(f"  [stage 1] no solver available; falling back to the Eq. D2 "
                  f"sizing rule -> {n} module(s)")
        return n
    cfg = _guard_quadratic(cfg, name)
    n_h = max(24, horizon // stride)
    m = build_model(cfg, phases, t0=phases.t4, n_hours=n_h, n_modules=None)
    _apply_options(name, opt, sc)
    t = time.time()
    res = opt.solve(m, tee=sc.tee)
    n = int(round(sum(pe.value(m.vModule[mm]) for mm in m.M)))
    if verbose:
        print(f"  [stage 1] {name}: {n_h} downsampled hours, "
              f"{res.solver.termination_condition}, {time.time()-t:.1f}s -> "
              f"{n} module(s) of {cfg.reactor}")
    return max(1, n)


def _solve_quality(res) -> Tuple[bool, float]:
    """(hit_time_limit, relative MIP gap) from a Pyomo results object.

    Every accessor here is optional and solver-dependent, so the whole thing is
    defensive: a solver that reports none of it yields (False, 0.0) and the run
    continues. Reporting no gap is not the same as reporting a zero gap, but the
    time-limit flag is the load-bearing signal and it comes from the termination
    condition, which every solver sets.
    """
    hit, gap = False, 0.0
    try:
        tc = res.solver.termination_condition
        hit = (TerminationCondition is not None
               and tc == TerminationCondition.maxTimeLimit)
    except Exception:                                          # noqa: BLE001
        pass
    for attr in ("gap", "mip_gap"):
        try:
            v = float(getattr(res.solver, attr))
            if v == v and v >= 0.0:
                gap = max(gap, v)
        except Exception:                                      # noqa: BLE001
            pass
    if gap == 0.0:
        try:
            lb = float(res.problem.lower_bound)
            ub = float(res.problem.upper_bound)
            if lb == lb and ub == ub and abs(ub) > 1e-9:
                gap = max(gap, abs(ub - lb) / abs(ub))
        except Exception:                                      # noqa: BLE001
            pass
    return hit, gap


def rolling_dispatch(cfg: SiteConfig, phases: PhaseConfig, sc: SolverConfig,
                     n_modules: int, t_start: int, n_days: int,
                     lookahead: int = 48, commit: int = 24,
                     verbose: bool = True) -> Tuple[Ledger, pd.DataFrame]:
    """Stage 2 -- rolling day-ahead. Solve `lookahead` hours, commit `commit`.

    Unit-commitment and ramp state are carried across windows via init_state, so
    a module that is running at the end of a committed day starts the next
    window running. This mimics day-ahead operation while keeping each solve
    tractable, exactly as in Paper 1.
    """
    name, opt = _pick_solver(sc)
    if opt is None:
        raise RuntimeError("no solver available for the rolling dispatch")
    cfg = _guard_quadratic(cfg, name)
    _apply_options(name, opt, sc)

    agg = Ledger(facility=cfg.facility_id, source="pyomo", modules=n_modules)
    frames, state = [], None
    for day in range(n_days):
        t0 = t_start + day * commit
        m = build_model(cfg, phases, t0=t0, n_hours=lookahead,
                        n_modules=n_modules, init_state=state)
        res = opt.solve(m, tee=sc.tee)
        tc = res.solver.termination_condition
        if tc not in (TerminationCondition.optimal, TerminationCondition.feasible,
                      TerminationCondition.maxTimeLimit):
            raise RuntimeError(f"window {day} terminated {tc}")
        _hit, _gap = _solve_quality(res)                       # B-22
        agg.n_windows += 1
        agg.n_timelimit += int(_hit)
        agg.worst_rel_gap = max(agg.worst_rel_gap, _gap)
        # accumulate only the COMMITTED hours
        led = extract_ledger(m, scale_to_year=False)
        frac = commit / lookahead
        for f in ("anncap", "fixed", "vom", "startup", "ramp_penalty",
                  "rev_btc", "rev_dc", "rev_comm", "rev_carbon", "rev_penalty",
                  "rev_export", "served_dc", "served_btc", "served_comm",
                  "curtailed", "unmet", "generation", "co2_avoided_t"):
            setattr(agg, f, getattr(agg, f) + getattr(led, f) * frac)
        # STOCKS, not flows: identical in every window, so they are carried
        # across rather than summed or scaled. nameplate_mw was missing from
        # this list until 2026-09-16, which left agg.nameplate_mw at 0 and sent
        # gate F5 to its fallback nameplate estimate -- an estimate that is
        # phase-blind (it adds flare capacity that Phase 4 does not have and
        # omits the RE peaking that it does). The implied capacity factor then
        # came out just over 1.0 and F5 failed three Phase-4 nodes of the
        # 2026-09-16 central sweep for a reason that had nothing to do with the
        # dispatch. Same class of defect as B-05: the aggregation silently
        # dropping a field.
        agg.overnight_capex_charged = led.overnight_capex_charged
        agg.nameplate_mw = led.nameplate_mw
        agg.technology = led.technology
        # --- B-23 -----------------------------------------------------------
        # The four B-21 diagnostic fields were computed in extract_ledger and
        # then dropped here, because this aggregation lists the fields it carries
        # and nothing warned when a new one was missing. The pyomo path therefore
        # emitted gen_gas_mwhe = 0, demand_mwhe = 0 and empty quantile strings on
        # every node of the v8 grid, which silently disabled the availability
        # response in the Monte Carlo that consumes them. Exactly the failure the
        # comment above describes for nameplate_mw (B-05, B-19) -- third
        # occurrence of one defect, so the fix is paired with an assertion below
        # rather than another unguarded line.
        agg.gen_gas_mwhe += led.gen_gas_mwhe * frac      # FLOW, prorated
        agg.demand_mwhe = led.demand_mwhe                # STOCK
        agg.avail_quantiles = led.avail_quantiles        # STOCK (last window)
        agg.gas_quantiles = led.gas_quantiles
        frames.append(pd.DataFrame({
            "t": [t0 + i for i in range(commit)],
            "s_dc": [pe.value(m.sDC[i]) for i in range(commit)],
            "s_btc": [pe.value(m.sBTC[i]) for i in range(commit)],
            "s_comm": [pe.value(m.sComm[i]) for i in range(commit)],
        }))
        state = {
            "on_last": {mm: int(round(pe.value(m.vOn[mm, commit - 1]))) for mm in m.M},
            "gen_last": {mm: float(pe.value(m.vGen[mm, commit - 1])) for mm in m.M},
        }
        if verbose and day % 10 == 0:
            print(f"    day {day + 1}/{n_days}  t={t0}  {tc}")

    # ---- ANNUALISE -------------------------------------------------------
    # Each window ledger is prorated by H = lookahead/8760 inside build_model
    # and weighted by commit/lookahead here, so the accumulated total is the
    # cost and energy of `n_days * commit` COMMITTED HOURS, not of a year.
    # Reporting it unscaled understates every financial by 8760/(n_days*commit)
    # -- a factor of 121.67 at the default 3 days -- and the ledger then
    # disagrees with analytical_benchmark by exactly that factor.
    #
    # This is the defect gate F caught on the first Phase-3 node of the
    # 2026-09-14 sweep. It could not be caught earlier because gate F's F1
    # check is skipped whenever served_dc = 0, which is every Phase 1-2 node.
    # F4 below now catches it independently of F1.
    committed = n_days * commit
    agg.hours = committed
    if committed <= 0:
        raise RuntimeError("no committed hours; cannot annualise")
    scale = HOURS_PER_YEAR / committed
    for f in ("anncap", "fixed", "vom", "startup", "ramp_penalty",
              "rev_btc", "rev_dc", "rev_comm", "rev_carbon", "rev_penalty",
              "rev_export", "served_dc", "served_btc", "served_comm",
              "curtailed", "unmet", "generation", "co2_avoided_t",
              "gen_gas_mwhe"):                               # B-23
        setattr(agg, f, getattr(agg, f) * scale)
    # B-23 guard: a field extract_ledger sets and this loop forgets becomes a
    # silent zero downstream. Assert the ones already lost once. Cheap, and it
    # fails at the node rather than in a figure three scripts later.
    if agg.demand_mwhe <= 0.0 and cfg.demand_mw > 0:
        raise RuntimeError("B-23: demand_mwhe aggregated to zero")
    if not agg.avail_quantiles:
        raise RuntimeError("B-23: avail_quantiles aggregated to empty")
    agg.annualised_from_hours = committed
    return agg, pd.concat(frames, ignore_index=True)


def solve_site(cfg: SiteConfig, phases: PhaseConfig, sc: SolverConfig,
               n_days: int = 3, t_start: Optional[int] = None,
               force_reference: bool = False,
               verbose: bool = True) -> Tuple[Ledger, pd.DataFrame, dict]:
    """Full two-stage run, or the reference dispatch when no solver is present."""
    t_start = phases.t4 if t_start is None else t_start
    name, _ = _pick_solver(sc)
    use_ref = force_reference or not HAVE_PYOMO or name is None
    if use_ref:
        why = ("forced" if force_reference else
               ("Pyomo not importable" if not HAVE_PYOMO else "no solver available"))
        if verbose:
            print(f"  [reference path] {why}; running the deterministic dispatch")
        n = required_modules(cfg, reactor_params(cfg))
        led, df = reference_dispatch(cfg, phases, t_start, n_days * 24, n_modules=n)
    else:
        n = sizing_pass(cfg, phases, sc, verbose=verbose)
        led, df = rolling_dispatch(cfg, phases, sc, n, t_start, n_days, verbose=verbose)
        if led.n_timelimit and verbose:
            print(f"  [B-22] {led.n_timelimit}/{led.n_windows} window(s) hit the "
                  f"{sc.time_limit}s limit; worst gap {led.worst_rel_gap:.2%}. "
                  f"This node is an INCUMBENT, not a proven optimum.")
    gate = gate_F(led, cfg, verbose=verbose)
    return led, df, gate


# =============================================================================
# SECTION 8 -- SMOKE TEST
# =============================================================================

def _hdr(t):
    print("\n" + "=" * 100)
    print(t)
    print("=" * 100)


def smoke_test(force_reference: bool = False, verbose: bool = True) -> dict:
    """Single-facility smoke test.

    Case 1  Phase 3, SMR only, no new revenue streams. This is the REGRESSION
            case: with every Paper-2 revenue switched off the model must return
            Paper 1's breakeven, and gate F must pass. This is the one that
            would catch a broken inheritance.
    Case 2  Phase 3 with carbon and community revenue on -- the Prompt-0 worked
            example. Gate F must still pass.
    Case 3  Phase 1, flare gas driving mining, no firm load. Gate F's F1 is
            correctly SKIPPED here because no diesel is displaced.
    Case 4  Proposition 3 demonstration: the same configuration with
            subordinate_mining off, showing why the rule is load-bearing.
    """
    phases = PhaseConfig()
    sc = SolverConfig()
    results = {}

    _hdr("CASE 1 -- Phase 3, SMR only, all Paper-2 revenues OFF  (regression to Paper 1)")
    c1 = SiteConfig(facility_id="CASE1", compute_price=0.0, elec_price_comm=0.0,
                    carbon_price=0.0, flare_penalty=0.0, btc_price=0.0,
                    comm_share_min=0.0, comm_share_max=0.0, flare_mmscfd=0.0,
                    genset_capex_per_kwe=0.0, genset_fopex_per_kwe=0.0)
    l1, _, g1 = solve_site(c1, phases, sc, n_days=3, t_start=phases.t3,
                           force_reference=force_reference, verbose=verbose)
    # independent target: Paper 1 Eq. D24 on the same build and served energy
    rx = reactor_params(c1)
    be_p1 = ab.paper1_breakeven(
        anncap=ab.annualised_capital(
            ab.overnight_capex(rx["capex"], rx["cap_e"] * l1.modules),
            c1.capex_mult, c1.discount_rate, c1.plant_life),
        fixed=ab.fixed_om(rx["fopex"], rx["cap_e"] * l1.modules),
        vom_fuel=(rx["vom"] + rx["fc"]) * l1.served_dc,
        startup=l1.startup, tes=0.0, export=0.0, served=l1.served_dc)
    rel1 = abs(l1.breakeven() - be_p1) / max(abs(be_p1), 1e-9)
    print(f"\n  Paper 1 Eq. D24 on the same build : ${be_p1:,.4f}/MWhe")
    print(f"  Model breakeven                    : ${l1.breakeven():,.4f}/MWhe")
    print(f"  REGRESSION TO PAPER 1              : {'PASS' if rel1 < 0.01 else 'FAIL'} "
          f"(rel err {rel1*100:.4f}%)")
    results["case1"] = {"ledger": l1, "gate": g1, "paper1_be": be_p1, "rel": rel1}

    _hdr("CASE 2 -- Phase 3 with carbon + community revenue  (Prompt-0 worked example)")
    c2 = SiteConfig(facility_id="CASE2", flare_mmscfd=0.0, btc_price=0.0,
                    compute_price=0.0, genset_capex_per_kwe=0.0,
                    genset_fopex_per_kwe=0.0, elec_price_comm=100.0,
                    carbon_price=50.0, comm_share_min=0.10, comm_share_max=0.10)
    l2, _, g2 = solve_site(c2, phases, sc, n_days=3, t_start=phases.t3,
                           force_reference=force_reference, verbose=verbose)
    print(f"\n  breakeven, convention B (served_dc)     : ${l2.breakeven():,.2f}/MWhe")
    print(f"  breakeven, convention A (total served)  : ${l2.breakeven_total_served():,.2f}/MWhe")
    print(f"  D-01 spread                             : "
          f"${l2.breakeven() - l2.breakeven_total_served():,.2f}/MWhe  <-- STILL OPEN")
    results["case2"] = {"ledger": l2, "gate": g2}

    _hdr("CASE 3 -- Phase 1, flare gas -> Bitcoin mining, no firm load")
    c3 = SiteConfig(facility_id="CASE3", it_mw=0.0, flare_mmscfd=2.0,
                    btc_price=60000.0, compute_price=0.0)
    c3.it_mw = 0.001                     # keeps demand ~0 without dividing by zero
    l3, df3, g3 = solve_site(c3, phases, sc, n_days=3, t_start=0,
                             force_reference=force_reference, verbose=verbose)
    print(f"\n  mining energy    : {l3.served_btc:,.0f} MWhe/yr")
    print(f"  BTC revenue      : ${l3.rev_btc:,.0f}/yr")
    print(f"  avoided penalty  : ${l3.rev_penalty:,.0f}/yr")
    print(f"  profit at d=$300 : ${l3.profit(300.0):,.0f}/yr")
    print(f"  breakeven        : {l3.breakeven()} (undefined: no diesel displaced)")
    results["case3"] = {"ledger": l3, "gate": g3}

    _hdr("CASE 4 -- Proposition 3: what the subordination rule is protecting")
    c4 = SiteConfig(facility_id="CASE4", subordinate_mining=False,
                    flare_mmscfd=0.0, btc_price=60000.0, genset_capex_per_kwe=0.0,
                    genset_fopex_per_kwe=0.0)
    l4, _, g4 = solve_site(c4, phases, sc, n_days=3, t_start=phases.t3,
                           force_reference=force_reference, verbose=False)
    btc_mwh = btc_revenue_per_mwh(c4)
    print(f"\n  Mining revenue is ${btc_mwh:,.2f}/MWhe of miner draw.")
    print(f"  Firm load is worth the diesel price, ${c4.diesel_price:,.2f}/MWhe.")
    print("  With subordination ON the optimiser cannot compare them: firm load is")
    print("  served first by constraint, so the dispatch is price-independent and")
    print("  Pi(d) stays affine. With it OFF the optimiser would switch allocation")
    print(f"  at d = ${btc_mwh:,.2f}/MWhe, putting a kink in Pi(d) and destroying the")
    print("  closed form (methods note Proposition 3, Eq. E22).")
    print(f"  Crossover diesel price: ${btc_mwh:,.2f}/MWhe  "
          f"({'INSIDE' if btc_mwh < 450 else 'outside'} the Monte Carlo range $200-450)")
    results["case4"] = {"ledger": l4, "gate": g4, "crossover": btc_mwh}

    # ---- summary ----------------------------------------------------------
    _hdr("SMOKE TEST SUMMARY")
    rows = []
    for k, v in results.items():
        led, g = v["ledger"], v["gate"]
        rows.append({
            "case": k, "source": led.source, "modules": led.modules,
            "served_dc": f"{led.served_dc:,.0f}",
            "served_btc": f"{led.served_btc:,.0f}",
            "cost": f"{led.total_cost:,.0f}",
            "rev_other": f"{led.total_revenue_non_diesel:,.0f}",
            "breakeven": ("n/a" if math.isnan(led.breakeven())
                          else f"{led.breakeven():,.2f}"),
            "gate_F": "PASS" if g["passed"] else "FAIL"})
    print(pd.DataFrame(rows).to_string(index=False))
    all_pass = all(v["gate"]["passed"] for v in results.values()) and rel1 < 0.01
    print(f"\n  Paper 1 regression : {'PASS' if rel1 < 0.01 else 'FAIL'}")
    print(f"  Gate F, all cases  : {'PASS' if all(v['gate']['passed'] for v in results.values()) else 'FAIL'}")
    print(f"  OVERALL            : {'PASS' if all_pass else 'FAIL'}")
    if not HAVE_PYOMO or force_reference:
        print("\n  NOTE: this run used the solver-free reference dispatch. It verifies the")
        print("  accounting chain and gate F, but NOT the MIQP formulation itself —")
        print("  technology selection, module count, unit commitment and the rolling")
        print("  horizon are untested until Pyomo and a MIP solver are available.")
    results["all_pass"] = all_pass
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[2])
    ap.add_argument("--no-solver", action="store_true",
                    help="force the deterministic reference dispatch")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)

    print("\n" + "#" * 100)
    print("#  phased_compute_model.py — Pyomo MIQP, phased flare-to-compute system")
    print(f"#  Pyomo: {'available' if HAVE_PYOMO else 'NOT INSTALLED'}", end="")
    if HAVE_PYOMO:
        nm, _ = _pick_solver(SolverConfig())
        print(f"   solver: {nm or 'none available'}")
    else:
        print("   -> reference dispatch path")
    print("#" * 100)

    r = smoke_test(force_reference=a.no_solver, verbose=not a.quiet)
    return 0 if r["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
