#!/usr/bin/env python3
"""
Paper 2 -- Day 8: Monte Carlo uncertainty quantification + Sobol global sensitivity.

Extends the Paper 1 Monte Carlo pattern (corrected-postproc.py) and ADDS Sobol
global sensitivity analysis. Pure post-processing: no solver is called. Every
draw is evaluated by closed-form arithmetic on the ledger produced by the sweep,
which is why 20,000 draws x 37 facilities x 4 phases is seconds, not days.

    python uq_phased.py                         # defaults, reads the newest sweep
    python uq_phased.py --draws 20000 --sobol-n 4096
    python uq_phased.py --results phased_sweep_results_v4/checkpoint_00.csv
    python uq_phased.py --compute-basis gpu     # see FLAGGED ASSUMPTION 3

--------------------------------------------------------------------------------
THE CONSERVATIVE SIMPLIFICATION  (stated here and re-printed in the output log)
--------------------------------------------------------------------------------
Technology and module selection are FIXED at the central node (mu = 1.0,
r = 0.12) and are NOT re-optimised per draw. Dispatch is likewise frozen: the
served energy, the fixed O&M, the variable O&M and the start-up cost all keep
their central-node values. Only the annualised capital term is recomputed,
because only it depends on the sampled CAPEX multiplier and discount rate:

    AnnCap(mu, r) = Overnight * mu * CRF(r),      CRF(r) = r / (1 - (1+r)^-n)

This is CONSERVATIVE in a specific, checkable sense. The real optimiser, faced
with a draw of high mu and high r, would re-select a cheaper design and a
smaller fleet, which can only LOWER the cost it achieves. Freezing the central
build therefore over-states cost in exactly the draws that hurt, so the reported
P(viable) is a LOWER BOUND on the P(viable) of a fully re-optimised model. It is
not an unbiased estimate, and it must not be described as one.

--------------------------------------------------------------------------------
HARD CONVENTIONS (project instruction Step 3 -- never violated here)
--------------------------------------------------------------------------------
    CRF = r / (1 - (1+r)^-n), n = 25, central r = 12%
    diesel reference $300/MWhe (used as the mode of the diesel triangular)
    EF diesel 0.70, EF nuclear 0.012 tCO2/MWhe   (inherited in the ledger)
    PUE = 1.5                                     (inherited in the ledger)
    block reward R = 3.125 BTC                    (inherited in the ledger)
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import textwrap
import time
import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ------------------------------------------------------------------ constants
NYEARS = 25
DIESEL_REF = 300.0
CENTRAL_MU = 1.0
CENTRAL_R = 0.12
HOURS_PER_YEAR = 8760          # B-21, matches phased_compute_model

# Base values of the sampled parameters AS THEY STAND IN THE SOLVED LEDGER.
# Each sampled draw is applied as a RATIO against these, so they must match
# data_inputs.py exactly or every revenue term is silently mis-scaled.
BASE = {
    "carbon_price":   50.0,      # $/tCO2      data_inputs.py, D-04 OPEN
    "btc_price":      60000.0,   # $/BTC       data_inputs.py, D-06 OPEN
    "btc_difficulty": 1.10e14,   # -           data_inputs.py, D-06 OPEN
    "compute_price":  120.0,     # $/MWhe-IT   data_inputs.py, D-10 OPEN
    # B-21 additions. These are the values BAKED INTO the v8 grid, so a draw is
    # applied as the RATIO of its sample to the number below. Reading them from
    # the register rather than hardcoding keeps the evaluator honest if the CSV
    # changes again.
    "genset_availability": 0.5468,   # -        dailystats_2026-09-18.csv, MEASURED
    "community_tariff":    20.0,     # $/MWhe   operator 2026-09-18, $0.02/kWh
    "miner_efficiency":    46.10,    # J/TH     pilot-derived, D-07 CLOSED
}

try:                                                       # register wins if present
    import data_inputs as _di_base
    _ov = _di_base.site_overrides("register")
    BASE["genset_availability"] = float(_ov.get("genset_availability",
                                                BASE["genset_availability"]))
    BASE["community_tariff"] = float(_ov.get("elec_price_comm",
                                             BASE["community_tariff"]))
    BASE["carbon_price"] = float(_ov.get("carbon_price", BASE["carbon_price"]))
    BASE["btc_price"] = float(_ov.get("btc_price", BASE["btc_price"]))
    BASE["compute_price"] = float(_ov.get("compute_price", BASE["compute_price"]))
    BASE["miner_efficiency"] = float(_ov.get("miner_efficiency_j_per_th",
                                             BASE["miner_efficiency"]))
except Exception:                                          # noqa: BLE001
    pass

# Phase midpoint, in years from financial close. Growth and decline processes
# are evaluated here rather than at a phase boundary, so a phase is charged its
# average condition and not its best or worst hour.
#   Phase 1 yr 0-3 -> 1.5   Phase 2 yr 3-5 -> 4.0
#   Phase 3 yr 5-10 -> 7.5  Phase 4 yr 10+ -> 15.0   (25-yr horizon, midpoint)
PHASE_MIDYEAR = {1: 1.5, 2: 4.0, 3: 7.5, 4: 15.0}

# TEN parameters from v8. The brief specified eight; the two added here are the
# two the 2026-09-18 pilot data closed, and leaving them out of the Sobol design
# would have declared them unimportant by omission rather than by measurement.
PARAMS = ["capex_mult", "discount_rate", "diesel_benchmark", "btc_price",
          "difficulty_growth", "carbon_price", "compute_price", "flare_decline",
          "genset_availability", "community_tariff", "miner_efficiency"]

PARAM_LABEL = {
    "capex_mult":        "CAPEX multiplier",
    "discount_rate":     "Discount rate",
    "diesel_benchmark":  "Diesel benchmark",
    "btc_price":         "BTC spot price",
    "difficulty_growth": "Difficulty growth",
    "carbon_price":      "Carbon price",
    "compute_price":     "Compute price",
    "flare_decline":     "Flare decline",
    "genset_availability": "Genset availability",
    "community_tariff":  "Community tariff",
    "miner_efficiency":  "Miner efficiency",
}


# =============================================================================
# SECTION 0 -- INVERSE CDFs, NUMPY ONLY
# =============================================================================
# scipy is NOT a dependency. The cluster python that produced the sweep has it,
# but the analysis python may not, and a UQ script that cannot run where the
# results live is not much use. Every marginal below is inverted in closed form
# or by a documented approximation, and scipy -- when importable -- is used only
# to CHECK these, never to replace them.

_A = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
      1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
_B = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
      6.680131188771972e+01, -1.328068155288572e+01]
_C = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
      -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
_D = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
      3.754408661907416e+00]


def norm_ppf(p):
    """Standard normal quantile -- Acklam's rational approximation.

    Relative error below 1.15e-9 over the whole open interval, which is three
    orders finer than any Monte-Carlo error at 20,000 draws. Verified against
    scipy.stats.norm.ppf when scipy is present (see the self-test).
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-16, 1 - 1e-16)
    lo, hi = p < 0.02425, p > 1 - 0.02425
    mid = ~(lo | hi)
    x = np.empty_like(p)
    q = np.sqrt(-2 * np.log(p[lo]))
    x[lo] = ((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5])
             / ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1))
    q = np.sqrt(-2 * np.log(1 - p[hi]))
    x[hi] = -((((((_C[0]*q+_C[1])*q+_C[2])*q+_C[3])*q+_C[4])*q+_C[5])
              / ((((_D[0]*q+_D[1])*q+_D[2])*q+_D[3])*q+1))
    q = p[mid] - 0.5
    r = q * q
    x[mid] = ((((((_A[0]*r+_A[1])*r+_A[2])*r+_A[3])*r+_A[4])*r+_A[5])*q
              / (((((_B[0]*r+_B[1])*r+_B[2])*r+_B[3])*r+_B[4])*r+1))
    return x


def norm_cdf(x):
    """Phi(x) from the quantile function's own inverse relation is circular, so
    use the Zelen & Severo 26.2.17 rational form -- |error| < 7.5e-8, ample for
    computing truncation bounds."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))))
    c = 1.0 - (1.0 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x * x) * poly
    return np.where(x >= 0, c, 1.0 - c)


def _betacf(a, b, x, itmax=300, eps=3e-16):
    """Continued fraction for the incomplete beta (Lentz's method, NR 6.4)."""
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = np.ones_like(x)
    d = 1.0 - qab * x / qap
    d = np.where(np.abs(d) < 1e-300, 1e-300, d)
    d = 1.0 / d
    h = d.copy()
    for m in range(1, itmax + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d; d = np.where(np.abs(d) < 1e-300, 1e-300, d); d = 1.0 / d
        c = 1.0 + aa / c; c = np.where(np.abs(c) < 1e-300, 1e-300, c)
        h = h * d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d; d = np.where(np.abs(d) < 1e-300, 1e-300, d); d = 1.0 / d
        c = 1.0 + aa / c; c = np.where(np.abs(c) < 1e-300, 1e-300, c)
        delt = d * c
        h = h * delt
        if np.all(np.abs(delt - 1.0) < eps):
            break
    return h


def _lgamma(z):
    return np.vectorize(math.lgamma)(np.asarray(z, dtype=float))


def betainc(a, b, x):
    """Regularised incomplete beta I_x(a, b)."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    lnbt = np.where((x <= 0) | (x >= 1), 0.0,
                    _lgamma(a + b) - _lgamma(a) - _lgamma(b)
                    + a * np.log(np.where(x > 0, x, 1.0))
                    + b * np.log(np.where(x < 1, 1.0 - x, 1.0)))
    front = x < (a + 1.0) / (a + b + 2.0)
    out = np.empty_like(x)
    xf = np.where(front, x, 1.0 - x)
    cf = _betacf(np.where(front, a, b), np.where(front, b, a),
                 np.clip(xf, 1e-300, 1 - 1e-16))
    val = np.exp(lnbt) * cf / np.where(front, a, b)
    out = np.where(front, val, 1.0 - val)
    return np.clip(np.where(x <= 0, 0.0, np.where(x >= 1, 1.0, out)), 0.0, 1.0)


_BETA_CACHE: Dict[Tuple[float, float, int], Tuple[np.ndarray, np.ndarray]] = {}


def beta_ppf(p, a, b, nodes: int = 40001):
    """Beta quantile by monotone interpolation of its own CDF.

    The obvious implementation -- bisect on I_x(a, b) for every p -- costs
    (bisection steps) x (continued-fraction steps) x (number of draws). At a
    Saltelli design of N = 16,384 that is ~7e9 inner operations and the process
    is killed before it finishes; this is exactly what happened on the first
    run of this script. Since I_x(a, b) is smooth and strictly increasing, the
    same answer comes from evaluating it ONCE on a fixed x-grid and inverting
    by interpolation, which is O(nodes + draws) instead.

    At 40,001 nodes the interpolation error in x is below 1e-8 for the
    concentrations used here -- three orders below Monte-Carlo error at 20,000
    draws. The grid is cached per (a, b), so repeated calls cost only the
    np.interp.
    """
    key = (round(float(a), 10), round(float(b), 10), int(nodes))
    if key not in _BETA_CACHE:
        xg = np.linspace(0.0, 1.0, nodes)
        cg = betainc(a, b, xg)
        cg[0], cg[-1] = 0.0, 1.0
        cg = np.maximum.accumulate(cg)          # enforce monotonicity exactly
        _BETA_CACHE[key] = (cg, xg)
    cg, xg = _BETA_CACHE[key]
    return np.interp(np.clip(np.asarray(p, dtype=float), 0.0, 1.0), cg, xg)


def triang_ppf(p, lo, mode, hi):
    """Triangular quantile, closed form."""
    p = np.asarray(p, dtype=float)
    c = (mode - lo) / (hi - lo)
    left = p < c
    return np.where(left,
                    lo + np.sqrt(np.clip(p * (hi - lo) * (mode - lo), 0, None)),
                    hi - np.sqrt(np.clip((1 - p) * (hi - lo) * (hi - mode), 0, None)))


def rankdata(x):
    """Average ranks, ties shared -- the ranking Spearman needs."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    ranks[order] = np.arange(1, len(x) + 1, dtype=float)
    xs = x[order]
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1
    return ranks


def spearman(a, b):
    """Spearman's rho = Pearson on average ranks. Returns (rho, t-statistic).

    No p-value is reported. At 20,000 draws every |rho| above about 0.02 is
    'significant', so a p-value would carry no information -- the t statistic is
    given instead so the reader can see how far from noise each value is.
    """
    ra, rb = rankdata(a), rankdata(b)
    ra = ra - ra.mean(); rb = rb - rb.mean()
    den = np.sqrt((ra * ra).sum() * (rb * rb).sum())
    rho = float((ra * rb).sum() / den) if den > 0 else 0.0
    n = len(a)
    t = (rho * np.sqrt((n - 2) / max(1 - rho * rho, 1e-300))
         if abs(rho) < 1 else np.inf)
    return rho, float(t)


def selftest_ppf(log) -> None:
    """Check the numpy inverse CDFs against scipy when scipy is importable."""
    try:
        from scipy import stats as st
    except ImportError:
        log("  scipy absent -- numpy inverse CDFs used unchecked "
            "(Acklam |err| < 1.2e-9; beta by 80-step bisection)")
        return
    u = np.linspace(1e-6, 1 - 1e-6, 20001)
    e1 = float(np.max(np.abs(norm_ppf(u) - st.norm.ppf(u))))
    e2 = float(np.max(np.abs(beta_ppf(u, 5.6, 6.4) - st.beta.ppf(u, 5.6, 6.4))))
    e3 = float(np.max(np.abs(triang_ppf(u, 200, 300, 450)
                             - st.triang.ppf(u, 0.4, loc=200, scale=250))))
    log(f"  inverse-CDF self-test vs scipy: norm {e1:.2e}, beta {e2:.2e}, "
        f"triang {e3:.2e}")


def crf(r, n: int = NYEARS):
    r = np.asarray(r, dtype=float)
    return r / (1.0 - (1.0 + r) ** (-n))


# =============================================================================
# SECTION 1 -- THE EIGHT MARGINAL DISTRIBUTIONS
# =============================================================================
# Every distribution below is exactly as specified in the Day-8 brief. Where the
# brief gave a shape without every parameter the gap is filled from a project
# hard convention and FLAGGED, never invented silently.

def lognormal_from_mean_cv(mean: float, cv: float) -> Tuple[float, float]:
    """(mu, sigma) of the underlying normal for a lognormal with this mean and CV.

        CV^2 = exp(sigma^2) - 1        ->  sigma = sqrt(ln(1 + CV^2))
        mean = exp(mu + sigma^2 / 2)   ->  mu    = ln(mean) - sigma^2 / 2

    Using the MEAN (not the median) as the centre matters: at CV 0.35 the median
    of this lognormal is 0.943, not 1.0, so a median-centred parameterisation
    would quietly shift every CAPEX draw up by 6%.
    """
    sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
    mu = float(np.log(mean) - 0.5 * sigma * sigma)
    return mu, sigma


def beta_ab_for_mean(lo: float, hi: float, mean: float, conc: float = 12.0):
    """Beta(a, b) on [lo, hi] with the requested mean and a chosen concentration.

    conc = a + b sets the spread; 12 gives a clearly unimodal interior density
    on [0.05, 0.20] without pinning mass at the bounds.
    """
    frac = (mean - lo) / (hi - lo)
    return conc * frac, conc * (1.0 - frac)


# --- B-21: the measured availability marginal --------------------------------
# The pilot gives 718 daily uptime observations. Fitting a Beta to them and then
# sampling the Beta would throw away the shape of a distribution that is visibly
# not Beta -- it is left-skewed with a long tail toward zero and a hard ceiling
# at 1. The empirical inverse CDF samples the measurement itself, assuming only
# that the next day resembles the 718 observed ones. That assumption is stated,
# testable, and far weaker than a parametric shape.

_UPTIME_CACHE: Optional[np.ndarray] = None


def load_measured_uptime(path: str = "pilot_validation/pilot_daily_processed.csv"
                         ) -> Optional[np.ndarray]:
    """Sorted daily uptime fractions from the pilot, or None if absent."""
    global _UPTIME_CACHE
    if _UPTIME_CACHE is not None:
        return _UPTIME_CACHE
    try:
        d = pd.read_csv(path)
        col = next(c for c in d.columns if c.strip().lower() == "uptime")
        v = pd.to_numeric(d[col], errors="coerce").dropna().to_numpy(float)
        v = v[(v >= 0.0) & (v <= 1.0)]
        if v.size < 100:
            return None
        _UPTIME_CACHE = np.sort(v)
        return _UPTIME_CACHE
    except Exception:                                      # noqa: BLE001
        return None


# Mean published power across the ASIC class the pilot's measured per-worker
# hashrate places it in (M20S 3360, M31S 3344, M30S 3268, T19 3344, S19 3250 W).
# Power is flat to 3.3% across that class while hashrate spans 40%, so dividing
# this constant by the measured hashrate is how a fleet efficiency is recovered
# without knowing the fleet. See data_inputs.MINER_PILOT.
ASIC_CLASS_POWER_W = 3313.2

_EFF_CACHE: Optional[np.ndarray] = None


def load_measured_efficiency(path: str = "pilot_validation/pilot_daily_processed.csv"
                             ) -> Optional[np.ndarray]:
    """Sorted daily fleet efficiency, J/TH, implied by the pilot's own hashrate.

    The dispersion here is real and is the point: the fleet is a MIX, its
    composition drifted over the record (86 TH/s per worker in 2024, 67-72 in
    2025-26), and machines were added and retired. Sampling the empirical spread
    carries that heterogeneity into the result instead of pretending a mixed,
    undisclosed fleet has one nameplate number.
    """
    global _EFF_CACHE
    if _EFF_CACHE is not None:
        return _EFF_CACHE
    try:
        d = pd.read_csv(path)
        d.columns = [c.strip() for c in d.columns]
        hr = pd.to_numeric(d["Hashrate (PH/s)"], errors="coerce")
        up = pd.to_numeric(d["Uptime"], errors="coerce")
        wk = pd.to_numeric(d["Workers count"], errors="coerce")
        m = hr.notna() & up.notna() & wk.notna() & (wk > 0) & (up > 0.05) & (hr > 0)
        tpw = (hr[m] * 1000.0) / up[m] / wk[m]          # TH/s per worker, hashing
        tpw = tpw[(tpw > 5.0) & (tpw < 500.0)].to_numpy(float)
        if tpw.size < 100:
            return None
        _EFF_CACHE = np.sort(ASIC_CLASS_POWER_W / tpw)   # J/TH
        return _EFF_CACHE
    except Exception:                                    # noqa: BLE001
        return None


def empirical_ppf(u: np.ndarray, sorted_x: np.ndarray) -> np.ndarray:
    """Inverse CDF of an empirical sample, linearly interpolated between order
    statistics. u in [0,1] -> a draw with the sample's own distribution."""
    n = sorted_x.size
    pos = np.clip(u, 0.0, 1.0) * (n - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    w = pos - lo
    return sorted_x[lo] * (1.0 - w) + sorted_x[hi] * w


@dataclass
class Priors:
    """The eight marginals. Bounds are also what the Sobol sampler is given."""
    # 1. CAPEX multiplier ~ Lognormal, mean 1.0, CV 0.35
    capex_cv: float = 0.35
    capex_mean: float = 1.0
    # 2. discount rate ~ Beta on [0.05, 0.20], mean ~0.12
    r_lo: float = 0.05
    r_hi: float = 0.20
    r_mean: float = 0.12
    r_conc: float = 12.0
    # 3. diesel benchmark ~ Triangular(200, 450); mode = the $300 hard convention
    diesel_lo: float = 200.0
    diesel_mode: float = DIESEL_REF          # FLAGGED ASSUMPTION 1
    diesel_hi: float = 450.0
    # 4. BTC spot ~ Lognormal, mean 65000, CV 0.40
    btc_mean: float = 65000.0
    btc_cv: float = 0.40
    # 5. network difficulty growth ~ Normal(30%/yr, 10%)
    dif_mean: float = 0.30
    dif_sd: float = 0.10
    dif_trunc_lo: float = -0.50              # FLAGGED ASSUMPTION 2
    dif_trunc_hi: float = 1.50
    # 6. carbon price ~ Uniform(0, 100) $/tCO2
    carbon_lo: float = 0.0
    carbon_hi: float = 100.0
    # 7. compute price ~ Triangular(1.5, 4.0) $/GPU-hr; mode 2.5
    gpu_lo: float = 1.5
    gpu_mode: float = 2.5                    # FLAGGED ASSUMPTION 3a
    gpu_hi: float = 4.0
    gpu_kw: float = 1.2                      # FLAGGED ASSUMPTION 3b
    power_share: float = 0.10                # FLAGGED ASSUMPTION 3c
    compute_basis: str = "colo"              # "colo" | "gpu"
    # 8. flare decline ~ Uniform(0.03, 0.08) per year
    flare_lo: float = 0.03
    flare_hi: float = 0.08
    # 9. genset availability ~ the EMPIRICAL distribution of the 718 measured
    #    daily uptimes (pilot_daily_processed.csv). No parametric shape is
    #    imposed. If the file is absent the fallback is a Beta matched to the
    #    measured mean and variance, which is FLAGGED where it is used.
    avail_fallback_mean: float = 0.5468      # MEASURED, dailystats_2026-09-18
    avail_fallback_sd: float = 0.2000        # FLAGGED ASSUMPTION 4 (fallback only)
    # 10. community tariff ~ Triangular(20, 20, 100) $/MWhe.
    #    BOTH endpoints are traceable, which is why this shape and not a
    #    guessed band: 20 is the operator's measured $0.02/kWh rural mini-grid
    #    tariff and is also the floor, since the operator does not sell below
    #    the tariff it quoted; 100 is the register's prior cost-reflective
    #    figure, the value a NERC Band A urban tariff would imply. The mode sits
    #    on the measurement, so the central case is the measured case and the
    #    upside is a policy scenario, not a guess.
    comm_lo: float = 20.0                    # OPERATOR 2026-09-18
    comm_mode: float = 20.0                  # OPERATOR 2026-09-18
    comm_hi: float = 100.0                   # prior register value
    # 11. miner efficiency ~ the EMPIRICAL distribution of the fleet efficiency
    #     implied by the pilot's own per-worker hashrate. Fallback is a
    #     triangular over the three derivation routes, FLAGGED where used.
    eff_lo: float = 45.22                    # route B, Lal 3.25 kW
    eff_mode: float = 46.10                  # route A, adopted
    eff_hi: float = 46.79                    # route C, interpolated
    # (A constant Phase-2 gas share lived here briefly. It was removed once the
    #  allocation kink was found: see `reallocate`. Nothing should reintroduce
    #  a scalar weight for availability.)

    # -- derived -----------------------------------------------------------
    def gpu_to_mwh(self, gpu_price: np.ndarray) -> np.ndarray:
        """$/GPU-hr -> $/MWhe of IT load.

        FLAGGED ASSUMPTION 3 -- THIS IS DECISION D-10 AND IT IS NOT SETTLED.

        A $/GPU-hr price is the price of a COMPUTE SERVICE. It pays for the GPU
        capital, the network, the software stack and the operator's margin as
        well as the electricity. The facility modelled here sells POWER AND
        SPACE; it does not own the accelerators. Booking the whole GPU-hour
        price as facility revenue would credit the plant with rent on assets it
        did not buy.

            raw power-equivalent = gpu_price / gpu_kw * 1000   $/MWhe
            at gpu_kw = 1.2 :  $1.5/GPU-hr -> $1,250/MWhe
                               $4.0/GPU-hr -> $3,333/MWhe

        against a ledger base of $120/MWhe-IT: a 10x to 28x uplift. Applying it
        raw makes every phase trivially viable and the study worthless.

        So the default basis is "colo": the facility captures `power_share` of
        the GPU-hour price. At power_share = 0.10 the draw maps to $125-333/MWhe,
        which straddles the 2026 wholesale colocation band of roughly
        $110-205/MWhe ($80-150/kW-month at 730 h). That is the defensible
        reading and it is the default.

        --compute-basis gpu applies the raw conversion instead. It is provided
        so the magnitude of D-10 can be REPORTED, not so it can be used.
        """
        raw = gpu_price / max(self.gpu_kw, 1e-9) * 1000.0
        return raw if self.compute_basis == "gpu" else raw * self.power_share

    def bounds(self) -> List[List[float]]:
        """Saltelli sampling box. Uniform draws here are pushed through the
        inverse CDFs in `transform`, so the marginals are preserved exactly."""
        return [[0.0, 1.0]] * len(PARAMS)

    def transform(self, u: np.ndarray) -> Dict[str, np.ndarray]:
        """Map a (N, 8) matrix of U(0,1) to the eight marginals by inverse CDF.

        Inverse-CDF transport is what lets the SAME marginals serve both the
        Monte Carlo pass and the Saltelli design. Sampling the two passes from
        different generators would make their answers incomparable.
        """
        out = {}
        mu_c, sg_c = lognormal_from_mean_cv(self.capex_mean, self.capex_cv)
        out["capex_mult"] = np.exp(mu_c + sg_c * norm_ppf(u[:, 0]))

        a, b = beta_ab_for_mean(self.r_lo, self.r_hi, self.r_mean, self.r_conc)
        out["discount_rate"] = self.r_lo + (self.r_hi - self.r_lo) * beta_ppf(u[:, 1], a, b)

        c = (self.diesel_mode - self.diesel_lo) / (self.diesel_hi - self.diesel_lo)
        _ = c
        out["diesel_benchmark"] = triang_ppf(
            u[:, 2], self.diesel_lo, self.diesel_mode, self.diesel_hi)

        mu_b, sg_b = lognormal_from_mean_cv(self.btc_mean, self.btc_cv)
        out["btc_price"] = np.exp(mu_b + sg_b * norm_ppf(u[:, 3]))

        lo = float(norm_cdf((self.dif_trunc_lo - self.dif_mean) / self.dif_sd))
        hi = float(norm_cdf((self.dif_trunc_hi - self.dif_mean) / self.dif_sd))
        out["difficulty_growth"] = self.dif_mean + self.dif_sd * norm_ppf(
            lo + u[:, 4] * (hi - lo))

        out["carbon_price"] = self.carbon_lo + u[:, 5] * (self.carbon_hi - self.carbon_lo)

        out["compute_price"] = triang_ppf(
            u[:, 6], self.gpu_lo, self.gpu_mode, self.gpu_hi)

        out["flare_decline"] = self.flare_lo + u[:, 7] * (self.flare_hi - self.flare_lo)

        emp = load_measured_uptime()
        if emp is not None:
            out["genset_availability"] = empirical_ppf(u[:, 8], emp)
        else:
            m, sd = self.avail_fallback_mean, self.avail_fallback_sd
            # Beta matched on mean and variance; conc = m(1-m)/var - 1
            conc = max(m * (1.0 - m) / max(sd * sd, 1e-9) - 1.0, 0.1)
            out["genset_availability"] = np.clip(
                beta_ppf(u[:, 8], conc * m, conc * (1.0 - m)), 1e-4, 1.0)

        out["community_tariff"] = triang_ppf(
            u[:, 9], self.comm_lo, self.comm_mode, self.comm_hi)

        eff = load_measured_efficiency()
        if eff is not None:
            out["miner_efficiency"] = empirical_ppf(u[:, 10], eff)
        else:
            out["miner_efficiency"] = triang_ppf(
                u[:, 10], self.eff_lo, self.eff_mode, self.eff_hi)
        return out


# =============================================================================
# SECTION 2 -- THE CLOSED-FORM EVALUATOR
# =============================================================================

@dataclass
class BaseLedger:
    """One facility-phase at the central node -- everything a draw needs."""
    facility_id: str
    country: str
    phase: int
    technology: str
    n_modules: float
    served: float          # MWhe firm (Eq. E20 denominator, convention B)
    served_total: float    # MWhe firm + BTC + community (convention A)
    anncap: float
    fixed: float
    vom: float
    startup: float
    rev_btc: float
    rev_dc: float
    rev_comm: float
    rev_carbon: float
    rev_penalty: float
    be_central: float
    # B-21 -- what the availability draw acts on
    gen_total: float = 0.0       # MWhe available at the central availability
    gen_gas: float = 0.0         # MWhe of that from the flare genset
    served_btc: float = 0.0
    served_comm: float = 0.0
    demand: float = 0.0          # MWhe firm annual demand, IT x PUE x 8760
    avail_q: np.ndarray = None   # 21 quantiles of hourly available MW
    gas_q: np.ndarray = None     # 21 quantiles of hourly gas MW, paired


def _pq(v) -> np.ndarray:
    """Parse a semicolon-joined quantile string from the checkpoint."""
    if not isinstance(v, str) or not v.strip():
        return np.array([], dtype=float)
    try:
        return np.array([float(x) for x in v.split(";") if x.strip()], dtype=float)
    except ValueError:
        return np.array([], dtype=float)


def _reconstruct_anncap(row: pd.Series) -> float:
    """AnnCap when the sweep did not emit the cost split (pre-2026-09-17 runs).

    total_cost = AnnCap + Fixed + VOM + StartUp, and only AnnCap carries mu and
    CRF(r). At the central node mu = 1, so

        AnnCap_central = Overnight * CRF(0.12)

    If `overnight_capex` is present this is exact. If neither it nor the split
    is present the facility-phase is DROPPED rather than guessed -- a Monte
    Carlo built on a reconstructed capital base would be untraceable, which
    project convention 13 forbids.
    """
    for k in ("cost_anncap",):
        if k in row and pd.notna(row[k]):
            return float(row[k])
    if "overnight_capex" in row and pd.notna(row["overnight_capex"]):
        return float(row["overnight_capex"]) * float(crf(CENTRAL_R))
    # Third tier, for pre-B-16 checkpoints that carry neither: rebuild the
    # capital base from the catalogue and the ledger's own build.
    #     Overnight = CAPEX[g] * 1000 $/MW * cap_e[g] * n_modules
    # Only valid where the phase actually builds the reactor fleet
    # (smr_scale = 1), i.e. phases 3 and 4. Phases 1-2 capitalise gensets and
    # renewables, which baseiaea.csv does not describe, so they are dropped
    # rather than guessed.
    try:
        import data_inputs as _di
        cat = _di.load_smr_catalogue()
        tech = str(row.get("technology", ""))
        nmod = float(row.get("n_modules", 0) or 0)
        if int(row.get("phase", 0)) in (3, 4) and tech in cat.index and nmod > 0:
            r_ = cat.loc[tech]
            overnight = (float(r_["CAPEX $/kWe"]) * 1000.0
                         * float(r_["Power in MWe"]) * nmod)
            return overnight * float(crf(CENTRAL_R))
    except Exception:                          # noqa: BLE001
        pass
    return float("nan")


def load_base_ledgers(path: str, log) -> List[BaseLedger]:
    d = pd.read_csv(path)
    d = d[(d.status == "ok") & (d.capex_mult == CENTRAL_MU)
          & (d.discount_rate == CENTRAL_R)].copy()
    log(f"  central-node rows: {len(d)}  ({d.facility_id.nunique()} facilities x "
        f"{d.phase.nunique()} phases)")

    have_split = all(c in d.columns for c in
                     ("cost_anncap", "cost_fixed", "cost_vom", "cost_startup"))
    log(f"  cost split present in checkpoint: {have_split}"
        + ("" if have_split else "  -> reconstructing AnnCap from overnight_capex"))

    out, dropped, unserved = [], 0, []
    for _, r in d.iterrows():
        anncap = _reconstruct_anncap(r)
        if not np.isfinite(anncap):
            dropped += 1
            continue
        if have_split:
            fixed, vom, start = (float(r["cost_fixed"]), float(r["cost_vom"]),
                                 float(r["cost_startup"]))
        else:
            # The residual IS Fixed + VOM + StartUp. They are held constant
            # across draws, so only their SUM is ever needed.
            fixed = float(r["total_cost"]) - anncap
            vom = start = 0.0
        served = float(r["served_mwhe"]) if pd.notna(r["served_mwhe"]) else 0.0
        # Phases 3 and 4 carry firm demand. served == 0 there is not a phase
        # with no load -- it is a build that failed to serve one, and it must
        # not slide through as a quiet NaN breakeven. See defect B-17.
        if int(r["phase"]) in (3, 4) and served <= 0.0:
            unserved.append((str(r["facility_id"]), str(r.get("technology", "")),
                             float(r.get("unmet_mwhe", float("nan")))))
        out.append(BaseLedger(
            facility_id=str(r["facility_id"]), country=str(r["country"]),
            phase=int(r["phase"]), technology=str(r.get("technology", "")),
            n_modules=float(r.get("n_modules", 0) or 0),
            served=served,
            served_total=served + float(r.get("served_btc_mwhe", 0) or 0)
                                + float(r.get("served_comm_mwhe", 0) or 0),
            anncap=anncap, fixed=fixed, vom=vom, startup=start,
            rev_btc=float(r.get("rev_btc", 0) or 0),
            rev_dc=float(r.get("rev_dc", 0) or 0),
            rev_comm=float(r.get("rev_comm", 0) or 0),
            rev_carbon=float(r.get("rev_carbon", 0) or 0),
            rev_penalty=float(r.get("rev_penalty", 0) or 0),
            be_central=(float(r["breakeven_unified"])
                        if pd.notna(r["breakeven_unified"]) else float("nan")),
            gen_total=float(r.get("generation_mwhe", 0) or 0),
            gen_gas=float(r.get("gen_gas_mwhe", float("nan"))),
            served_btc=float(r.get("served_btc_mwhe", 0) or 0),
            served_comm=float(r.get("served_comm_mwhe", 0) or 0),
            demand=float(r.get("demand_mwhe", float("nan"))),
            avail_q=_pq(r.get("avail_quantiles", "")),
            gas_q=_pq(r.get("gas_quantiles", "")),
        ))
    if dropped:
        log(f"  DROPPED {dropped} facility-phases: no traceable capital base")
    if unserved:
        log("")
        log(f"  !! {len(unserved)} phase-3/4 facility-phases have served_mwhe = 0")
        log("     A firm-demand phase that serves nothing is defect B-17: the "
            "build was selected on")
        log("     the Eq. D3 cost aggregate, which a plant that never runs "
            "minimises by doing nothing.")
        log("     Their breakeven is undefined and every draw scores "
            "not-viable, so P(viable) below")
        log("     is depressed by exactly their share. RERUN THE SWEEP with the "
            "B-17 fix before")
        log("     quoting any number from this run.")
        for f, t, u in unserved[:12]:
            log(f"       {f:<32s} tech={t:<16s} unmet={u:,.0f} MWh")
        if len(unserved) > 12:
            log(f"       ... and {len(unserved) - 12} more")
        log("")
    return out


# =============================================================================
# B-21 -- EXACT RE-ALLOCATION UNDER AN AVAILABILITY DRAW
# =============================================================================
# A single multiplicative weight cannot represent an availability change in
# Phase 2, and the reason is worth stating because it was not obvious. Derating
# the genset shrinks AVAILABLE energy, and the Phase-2 allocation rule is not
# linear in available energy:
#
#     s_comm = comm_share x avail                     (Eq. E8, linear)
#     s_dc   = min(demand, dc_share x avail)          (Eq. E9, KINKED)
#     s_btc  = avail - s_comm - s_dc                  (residual)
#
# The min() is the kink. Where firm demand binds, a derate takes the whole cut
# out of mining; where it does not, the cut is shared. Empirically a constant
# weight is exact for facilities above ~6 MW and wrong by up to 32% at 2 MW --
# precisely the small sites this paper is about. So the allocation is recomputed
# instead of scaled, which is exact at every facility size and costs three
# vector operations.
#
# Phases 3 and 4 retire the genset, so avail' = avail and every ratio is 1.
# Phase 1 has no firm demand, so the rule collapses to a uniform derate and
# reproduces the naive weight exactly -- a useful check that this is a
# generalisation of the simple case and not a replacement for it.

PHASE_SHARES = {1: {"dc": 0.0, "comm": 0.10},
                2: {"dc": 0.40, "comm": 0.10},
                3: {"dc": 1.00, "comm": 0.10},
                4: {"dc": 1.00, "comm": 0.15}}
try:                                       # the model is the authority if present
    import phased_compute_model as _pcm_sh
    PHASE_SHARES = {int(k): {"dc": float(v["dc_share_of_available"]),
                             "comm": float(v["comm_share"])}
                    for k, v in _pcm_sh.PhaseConfig.PHASE_ECONOMICS.items()}
except Exception:                                              # noqa: BLE001
    pass


# Measured agreement between this closed form and the solver, from
# test_reallocation_b21.py on 2026-09-18. Reported in the run log so a reader of
# the Monte Carlo never has to take the surrogate on trust.
SURROGATE_CHECK = ("test_reallocation_b21.py, 2026-09-18: 102/110 stream ratios "
                   "exact to machine precision; 8 residuals, all Phase 2 at "
                   "2-4 MW, worst 1.6e-03 (0.16%)")


def reallocate(bl: BaseLedger, a: np.ndarray, a_central: float
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Allocation ratios (dc, btc, comm, gas) for an availability draw.

    `a` is the sampled availability, `a_central` the one baked into the ledger.
    Returns each stream's energy as a MULTIPLE of its central value, so the
    caller scales revenues without needing absolute quantities.

    The gas ratio is a / a_central always: the penalty credit and the avoided
    CO2 follow the gas actually burned, never the allocation of the resulting
    electricity.
    """
    ones = np.ones_like(a, dtype=float)
    gas_ratio = a / a_central

    gas0 = bl.gen_gas
    if not np.isfinite(gas0) or gas0 <= 0.0 or bl.gen_total <= 0.0:
        # No gas in this phase (3 and 4), or a checkpoint predating the
        # gen_gas_mwhe column. Both mean "availability cannot move this row",
        # which for phases 3-4 is exactly right and for an old phase-1/2
        # checkpoint is a FLAGGED degradation reported by load_base_ledgers.
        return ones, ones, ones, (gas_ratio if bl.phase in (1, 2) else ones)

    re0 = max(bl.gen_total - gas0, 0.0)          # non-gas supply, unaffected
    gas_und = gas0 / a_central                   # undegraded genset energy
    avail0 = bl.gen_total
    avail1 = gas_und * a + re0

    sh = PHASE_SHARES.get(bl.phase, {"dc": 0.0, "comm": 0.10})
    demand = bl.demand
    if not np.isfinite(demand) or demand <= 0.0:
        # No demand column (pre-v8 checkpoint). Fall back to the central served
        # energy, which is the demand whenever demand was the binding cap. This
        # is FLAGGED by load_base_ledgers rather than silently accepted.
        demand = bl.served if bl.served > 0 else 0.0

    # ---- the allocation, integrated over the HOURLY shape -------------------
    # The rule is applied to each quantile of hourly availability and averaged,
    # not applied once to the annual mean. Where the cap binds in every hour, or
    # in none, the two agree exactly; where it binds in some hours only, they do
    # not, and the second case is the 2-3 MW flare-site scale.
    aq, gq = bl.avail_q, bl.gas_q
    have_shape = (aq is not None and gq is not None
                  and len(aq) == len(gq) and len(aq) >= 2)
    dem_h = demand / HOURS_PER_YEAR              # firm demand, MW, flat

    def _alloc_h(av_h):
        """Eq. E8/E9 verbatim, on an hourly MW vector. Returns MW means."""
        comm = sh["comm"] * av_h
        dem = np.minimum(dem_h, sh["dc"] * av_h) if sh["dc"] < 1.0 else dem_h
        dc = np.minimum(dem, np.maximum(av_h - comm, 0.0))
        btc = np.maximum(av_h - comm - dc, 0.0)
        return dc.mean(-1), btc.mean(-1), comm.mean(-1)

    if have_shape:
        re_h = np.maximum(aq - gq, 0.0)                    # non-gas MW by quantile
        gas_und_h = gq / a_central                         # undegraded gas MW
        av1_h = gas_und_h[None, :] * np.asarray(a, float)[:, None] + re_h[None, :]
        dc1, btc1, comm1 = _alloc_h(av1_h)
        dc0, btc0, comm0 = _alloc_h(aq[None, :])
        dc0, btc0, comm0 = float(dc0[0]), float(btc0[0]), float(comm0[0])
    else:
        # Pre-v8 checkpoint: fall back to the annual aggregate. Exact wherever
        # the cap binds uniformly, and reported as degraded by load_base_ledgers.
        def _alloc(av):
            comm = sh["comm"] * av
            dem = np.minimum(demand, sh["dc"] * av) if sh["dc"] < 1.0 else demand
            dc = np.minimum(dem, np.maximum(av - comm, 0.0))
            btc = np.maximum(av - comm - dc, 0.0)
            return dc, btc, comm
        dc1, btc1, comm1 = _alloc(avail1)
        dc0, btc0, comm0 = _alloc(avail0)

    r_dc = dc1 / dc0 if dc0 > 0 else ones
    r_btc = btc1 / btc0 if btc0 > 0 else ones
    r_comm = comm1 / comm0 if comm0 > 0 else ones
    return r_dc, r_btc, r_comm, gas_ratio


def evaluate(bl: BaseLedger, p: Dict[str, np.ndarray], pri: Priors
             ) -> Dict[str, np.ndarray]:
    """Closed-form ledger for a vector of draws. Returns breakeven, profit, viable.

    COST -- only the capital term moves:
        AnnCap(mu, r) = AnnCap_0 * (mu / 1.0) * (CRF(r) / CRF(0.12))
    Fixed, VOM and StartUp keep their central values (the simplification above).

    REVENUE -- each term is rescaled by the RATIO of its sampled driver to the
    value already baked into the ledger:
        rev_btc     ~ SP_BIT / D            (Lal et al. Eq. 11)
        rev_carbon  ~ carbon_price
        rev_dc      ~ compute_price
        rev_penalty ~ (1 - delta)^tau       flare volume decays
        rev_comm    ~ tariff x r_comm        B-21, both drivers now sampled

    B-21 -- AVAILABILITY. The genset availability draw is NOT applied as a
    scalar weight. It shrinks available energy, and the Phase-2 allocation rule
    is kinked in available energy, so the streams are re-derived from the rule
    itself by `reallocate` above. That function returns one ratio per stream and
    is exact at every facility size; a scalar weight is not.

    Difficulty compounds to the phase midpoint tau:  D(tau) = D_0 (1 + g)^tau.
    In Phases 1 and 2 the miners burn flare gas, so the same decline that cuts
    the penalty credit also cuts mining energy; rev_btc is decayed there too.
    In Phases 3 and 4 mining is off (rev_btc = 0) and the term is inert.
    Costs are NOT decayed with the flare: that asymmetry is deliberate and
    conservative -- revenue falls, the cost of the plant does not.
    """
    tau = PHASE_MIDYEAR[bl.phase]

    anncap = bl.anncap * (p["capex_mult"] / CENTRAL_MU) * (
        crf(p["discount_rate"]) / float(crf(CENTRAL_R)))

    dif_factor = (1.0 + p["difficulty_growth"]) ** tau
    flare_factor = (1.0 - p["flare_decline"]) ** tau

    # --- B-21 availability, by exact re-allocation ---------------------------
    r_dc, r_btc, r_comm, r_gas = reallocate(
        bl, p["genset_availability"], BASE["genset_availability"])

    # Mining revenue per MWh of draw is inversely proportional to J/TH: at eps
    # J/TH one MW of draw buys 1e6/eps TH/s. So a less efficient fleet earns
    # proportionally less from the same energy, and the ledger's central value
    # is rescaled by the RATIO of efficiencies, central over sampled.
    eff_factor = BASE["miner_efficiency"] / p["miner_efficiency"]
    rev_btc = (bl.rev_btc * (p["btc_price"] / BASE["btc_price"])
               / dif_factor * r_btc * eff_factor)
    if bl.phase in (1, 2):
        rev_btc = rev_btc * flare_factor

    # Avoided CO2 is earned on energy DELIVERED, so it follows the delivered
    # total, which in the gas phases moves with the gas ratio.
    rev_carbon = (bl.rev_carbon * (p["carbon_price"] / BASE["carbon_price"])
                  * (r_gas if bl.phase in (1, 2) else 1.0))
    rev_dc = (bl.rev_dc * (pri.gpu_to_mwh(p["compute_price"])
                           / BASE["compute_price"]) * r_dc)
    rev_penalty = bl.rev_penalty * flare_factor * r_gas
    rev_comm = (bl.rev_comm * (p["community_tariff"] / BASE["community_tariff"])
                * r_comm)
    rev = rev_btc + rev_dc + rev_comm + rev_carbon + rev_penalty

    # VOM is charged per MWh GENERATED, so it follows the gas ratio in the gas
    # phases. Holding it fixed would be the conservative choice, but it would
    # also be wrong, and the term is small in Phases 1-2 where the fuel is free.
    vom_ratio = r_gas if bl.phase in (1, 2) else 1.0
    cost = anncap + bl.fixed + bl.vom * vom_ratio + bl.startup

    # Firm served energy in Phase 2 is supply-limited (Eq. E9), so it moves with
    # the re-allocated DC stream. In Phases 3-4 r_dc is identically 1.
    served = bl.served * r_dc

    # `served` is a vector now, because Phase-2 firm energy moves with the
    # availability draw. A draw that serves nothing has no defined breakeven and
    # must stay NaN rather than dividing by a near-zero and reporting a number
    # with the magnitude of a rounding error.
    served = np.broadcast_to(np.asarray(served, dtype=float), cost.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        be = np.where(served > 0.0, (cost - rev) / np.where(served > 0.0, served, 1.0),
                      np.nan)

    # Profit is defined in EVERY phase and is the only cross-phase robustness
    # metric available. Paper 1 Eq. D23 sign convention: avoided diesel is
    # revenue at the sampled benchmark.
    profit = rev + p["diesel_benchmark"] * served - cost
    viable = np.where(np.isfinite(be), be < p["diesel_benchmark"], profit > 0.0)
    return {"breakeven": be, "profit": profit, "viable": viable.astype(float),
            "cost": cost, "rev": rev}


# =============================================================================
# SECTION 3 -- SOBOL
# =============================================================================
# SALib is used when importable, exactly as the brief asks. A self-contained
# Saltelli + Jansen implementation is kept as a fallback so the script still
# runs on a cluster python without SALib, and so the two can be cross-checked
# against each other when both are present.

def saltelli_sample(n: int, d: int, rng, second_order: bool = True) -> np.ndarray:
    A = rng.random((n, d))
    B = rng.random((n, d))
    mats = [A, B]
    for i in range(d):                       # AB_i
        AB = A.copy(); AB[:, i] = B[:, i]; mats.append(AB)
    if second_order:
        for i in range(d):                   # BA_i
            BA = B.copy(); BA[:, i] = A[:, i]; mats.append(BA)
    return np.vstack(mats)


def sobol_indices(y: np.ndarray, n: int, d: int, second_order: bool = True) -> Dict:
    """First-order (Saltelli 2010) and total (Jansen 1999) indices.

        S1_i  = mean(B * (AB_i - A)) / Var(Y)
        ST_i  = mean((A - AB_i)^2) / (2 Var(Y))

    Jansen's total-index estimator is used because it stays well behaved when
    ST is near zero, where the naive 1 - mean(A*AB)/Var form goes negative.
    """
    yA = y[:n]
    yB = y[n:2 * n]
    var = np.var(np.concatenate([yA, yB]), ddof=1)
    S1 = np.zeros(d); ST = np.zeros(d)
    if var <= 0:
        return {"S1": S1, "ST": ST, "S2": np.full((d, d), np.nan), "var": 0.0}
    for i in range(d):
        yABi = y[(2 + i) * n:(3 + i) * n]
        S1[i] = np.mean(yB * (yABi - yA)) / var
        ST[i] = np.mean((yA - yABi) ** 2) / (2.0 * var)
    S2 = np.full((d, d), np.nan)
    if second_order:
        off = 2 + d
        for i in range(d):
            yBAi = y[(off + i) * n:(off + i + 1) * n]
            for j in range(i + 1, d):
                yABj = y[(2 + j) * n:(3 + j) * n]
                yBAj = y[(off + j) * n:(off + j + 1) * n]
                Vij = np.mean(yBAi * yABj - yA * yB) / var
                S2[i, j] = Vij - S1[i] - S1[j]
                _ = yBAj
    return {"S1": S1, "ST": ST, "S2": S2, "var": float(var)}


def sobol_bootstrap(y: np.ndarray, n: int, d: int, reps: int = 200,
                    seed: int = 0) -> Dict[str, np.ndarray]:
    """Bootstrap confidence intervals for S1 and ST.

    Sobol estimators are RATIOS OF SAMPLE MEANS. On a near-binary output whose
    probability sits close to 0 or 1 the denominator variance is small and the
    estimator becomes unstable: S1 can come back negative, and sum(S1) can
    leave [0, 1]. A negative S1 is never a real effect -- it is the estimator
    telling you N is too small for that term. Without an interval there is no
    way to tell a resolved index from noise, so the interval is not optional
    decoration here; it is what makes the table readable.

    Resampling is done on the ROW INDEX, so a bootstrap replicate keeps each
    (A_i, B_i, AB_i) triple together and the estimator stays valid.
    """
    rng = np.random.default_rng(seed)
    S1b = np.empty((reps, d)); STb = np.empty((reps, d))
    yA = y[:n]; yB = y[n:2 * n]
    ABs = [y[(2 + i) * n:(3 + i) * n] for i in range(d)]
    for r in range(reps):
        idx = rng.integers(0, n, n)
        a_, b_ = yA[idx], yB[idx]
        var = np.var(np.concatenate([a_, b_]), ddof=1)
        if var <= 0:
            S1b[r] = 0.0; STb[r] = 0.0; continue
        for i in range(d):
            ab = ABs[i][idx]
            S1b[r, i] = np.mean(b_ * (ab - a_)) / var
            STb[r, i] = np.mean((a_ - ab) ** 2) / (2.0 * var)
    return {"S1_lo": np.percentile(S1b, 2.5, axis=0),
            "S1_hi": np.percentile(S1b, 97.5, axis=0),
            "ST_lo": np.percentile(STb, 2.5, axis=0),
            "ST_hi": np.percentile(STb, 97.5, axis=0)}


def run_sobol(pri: Priors, bls: List[BaseLedger], n: int, seed: int, log
              ) -> Dict[str, Dict]:
    """Sobol indices on VIABILITY and on BREAKEVEN, pooled over facilities.

    Two outputs, deliberately:

      * viability is the decision-relevant output, but it is a Bernoulli, so its
        variance is bounded by 0.25 and the indices are noisier;
      * breakeven is continuous and gives cleaner indices, but the diesel
        benchmark CANNOT appear in it -- diesel enters only the comparison
        be < diesel. Diesel's S1 on breakeven is therefore ZERO BY
        CONSTRUCTION, and any non-zero value is estimator noise.

    Reporting both is what exposes that structure. Reporting only the continuous
    one would hide the single most important driver of the decision.
    """
    d = len(PARAMS)
    rng = np.random.default_rng(seed)
    U = saltelli_sample(n, d, rng, second_order=True)
    p = pri.transform(U)
    log(f"  Saltelli design: N={n}, d={d}, evaluations={U.shape[0]:,} "
        f"({2 + 2 * d} matrices)")

    out = {}
    for phase in sorted({b.phase for b in bls}):
        sel = [b for b in bls if b.phase == phase]
        via = np.zeros(U.shape[0]); be = np.zeros(U.shape[0]); nbe = 0
        for b in sel:
            e = evaluate(b, p, pri)
            via += e["viable"]
            if np.isfinite(e["breakeven"]).all():
                be += e["breakeven"]; nbe += 1
        via /= len(sel)                       # fraction of facilities viable
        res = {"viability": sobol_indices(via, n, d)}
        res["viability"].update(sobol_bootstrap(via, n, d, seed=seed + phase))
        if nbe:
            res["breakeven"] = sobol_indices(be / nbe, n, d)
            res["breakeven"].update(
                sobol_bootstrap(be / nbe, n, d, seed=seed + 100 + phase))
        # convergence: recompute on the first half of the design. A stable
        # index barely moves; one that halves or flips has not converged.
        half = n // 2
        for nm, series in (("viability", via),
                           ("breakeven", (be / nbe) if nbe else None)):
            if series is None or nm not in res:
                continue
            sh = np.concatenate([series[j * n:j * n + half]
                                 for j in range(2 + 2 * d)])
            res[nm]["ST_half"] = sobol_indices(sh, half, d)["ST"]
        out[phase] = res

    # cross-check against SALib when it is installed, exactly as briefed
    try:
        from SALib.analyze import sobol as salib_sobol
        prob = {"num_vars": d, "names": PARAMS, "bounds": pri.bounds()}
        sel = [b for b in bls if b.phase == 3]
        via = np.mean([evaluate(b, p, pri)["viable"] for b in sel], axis=0)
        sa = salib_sobol.analyze(prob, via, calc_second_order=True,
                                 print_to_console=False)
        dmax = float(np.max(np.abs(np.array(sa["ST"]) - out[3]["viability"]["ST"])))
        log(f"  SALib cross-check (phase 3, ST): max |difference| = {dmax:.2e}")
        out["_salib"] = {k: np.asarray(v) for k, v in sa.items()
                         if k in ("S1", "ST", "S1_conf", "ST_conf")}
    except ImportError:
        log("  SALib not importable -- built-in Saltelli/Jansen estimator used. "
            "`pip install SALib` enables the cross-check.")
    except Exception as e:                    # noqa: BLE001
        log(f"  SALib cross-check skipped: {type(e).__name__}: {e}")
    return out


# =============================================================================
# SECTION 4 -- MONTE CARLO
# =============================================================================

def run_mc(pri: Priors, bls: List[BaseLedger], draws: int, seed: int, log
           ) -> pd.DataFrame:
    """20,000 draws per facility-phase. Returns per-facility-phase summary rows
    plus the per-draw arrays needed for the threshold curves."""
    rng = np.random.default_rng(seed)
    rows = []
    diesel_grid = np.arange(100.0, 501.0, 5.0)
    for b in bls:
        u = rng.random((draws, len(PARAMS)))
        p = pri.transform(u)
        e = evaluate(b, p, pri)
        be, pr, vi = e["breakeven"], e["profit"], e["viable"]
        # P(viable) as a function of a DETERMINISTIC diesel benchmark: the
        # threshold curve. The sampled diesel is held out here on purpose --
        # the curve answers "at a diesel price of X, what is the chance this
        # facility beats it", which is the question an investor asks.
        if np.isfinite(be).all():
            curve = (be[:, None] < diesel_grid[None, :]).mean(axis=0)
        else:
            # No firm load, so viability cannot depend on the diesel benchmark:
            # the curve is flat at P(profit > 0). It must still have ONE ENTRY
            # PER GRID POINT. The first version wrote
            #     (pr[:, None] > 0).mean(axis=0) * 0 + (pr > 0).mean()
            # which reduces over the draw axis of a (draws, 1) array and yields
            # a length-1 curve, so np.vstack later failed with "array at index 0
            # has size 81 and the array at index 2 has size 1". Broadcasting a
            # scalar into the right shape is the whole fix.
            curve = np.full(diesel_grid.shape, float((pr > 0).mean()))
        assert curve.shape == diesel_grid.shape, (
            f"threshold curve for {b.facility_id} phase {b.phase} has shape "
            f"{curve.shape}, expected {diesel_grid.shape}")
        # Phase 1 has no firm load, so `be` is all-NaN by construction and
        # every nan-aggregate below warns. That warning is expected and says
        # nothing a reader needs, so it is silenced HERE and nowhere else --
        # a blanket filter would also hide the all-NaN slices that mean B-17.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            be_mean = np.nanmean(be) if np.isfinite(be).any() else np.nan
            be_p05, be_p50, be_p95 = (np.nanpercentile(be, q)
                                      if np.isfinite(be).any() else np.nan
                                      for q in (5, 50, 95))
            be_sd = np.nanstd(be) if np.isfinite(be).any() else np.nan
        rows.append(dict(
            facility_id=b.facility_id, country=b.country, phase=b.phase,
            technology=b.technology, n_modules=b.n_modules,
            be_central=b.be_central,
            be_mean=be_mean, be_p05=be_p05, be_p50=be_p50, be_p95=be_p95,
            be_sd=be_sd,
            profit_mean=pr.mean(), profit_sd=pr.std(),
            profit_cv=(pr.std() / abs(pr.mean()) if pr.mean() else np.nan),
            p_profit_pos=float((pr > 0).mean()),
            p_viable=float(vi.mean()),
            curve=json.dumps([round(float(x), 5) for x in curve]),
        ))
    log(f"  Monte Carlo: {draws:,} draws x {len(bls)} facility-phases "
        f"= {draws * len(bls):,} evaluations")
    df = pd.DataFrame(rows)
    df.attrs["diesel_grid"] = diesel_grid
    return df


def spearman_table(pri: Priors, bls: List[BaseLedger], draws: int, seed: int
                   ) -> pd.DataFrame:
    """Paper 1's sensitivity method: Spearman rank correlation on the MC sample.

    This is what Sobol is being compared AGAINST. Spearman measures monotone
    association one parameter at a time; it cannot see an interaction, and it
    cannot apportion variance. The comparison in the output log is the point.
    """
    rng = np.random.default_rng(seed + 1)
    rows = []
    for phase in sorted({b.phase for b in bls}):
        sel = [b for b in bls if b.phase == phase]
        u = rng.random((draws, len(PARAMS)))
        p = pri.transform(u)
        via = np.mean([evaluate(b, p, pri)["viable"] for b in sel], axis=0)
        for i, k in enumerate(PARAMS):
            rho, tstat = spearman(p[k], via)
            rows.append(dict(phase=phase, parameter=k, spearman_rho=rho,
                             abs_rho=abs(rho), t_stat=tstat, n=draws))
    return pd.DataFrame(rows)


# =============================================================================
# SECTION 5 -- FIGURES
# =============================================================================
# Palette and rules follow the same validated set used by make_viability_figures.py.
# Categorical slots 1-3 (blue / orange / aqua) validate all-pairs in light mode
# (worst CVD dE 9.2, worst normal-vision dE 24.0). Aqua sits below 3:1 contrast
# on the surface, so every line is ALSO direct-labelled and dash-coded.

C_BLUE, C_ORANGE, C_AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8985"
SURFACE, GRAY_MID = "#fcfcfb", "#f0efec"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#2a78d6", "#256abf",
             "#1c5cab", "#184f95", "#0d366b"]
COUNTRY_STYLE = {                       # entity -> colour, never rank -> colour
    "Nigeria":      (C_BLUE,   "-"),
    "Kenya":        (C_ORANGE, (0, (5, 2))),
    "South Africa": (C_AQUA,   (0, (1.5, 1.5))),
}


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 600,
        "font.family": "DejaVu Sans", "font.size": 8.5,
        "axes.edgecolor": INK3, "axes.linewidth": 0.6,
        "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK2, "ytick.color": INK2,
        "axes.facecolor": SURFACE, "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    return plt


def _head(ax, title, sub):
    nl = sub.count("\n") + 1
    ax.text(0, 1.045 + 0.042 * nl, title, transform=ax.transAxes, fontsize=10,
            fontweight="bold", color=INK, va="bottom", ha="left")
    ax.text(0, 1.028, sub, transform=ax.transAxes, fontsize=7.5, color=INK2,
            va="bottom", ha="left", linespacing=1.45)


def make_figures(mc: pd.DataFrame, sob: Dict, spear: pd.DataFrame,
                 pri: Priors, draws: int, log, stem: str = "fig_uq_phased",
                 publication: bool = False):
    plt = _mpl()
    grid = mc.attrs["diesel_grid"]
    fig = plt.figure(figsize=(11.0, 9.8))
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.28,
                          left=0.075, right=0.965, top=0.875, bottom=0.075)
    axA, axB = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    axC, axD = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    # ---- A: risk-adjusted viability threshold curves, by country (Phase 3)
    p3 = mc[mc.phase == 3]
    for _ci, (ctry, (col, ls)) in enumerate(COUNTRY_STYLE.items()):
        s = p3[p3.country == ctry]
        if not len(s):
            continue
        curves = np.vstack([np.array(json.loads(c)) for c in s.curve])
        med = curves.mean(axis=0)
        axA.plot(grid, med, color=col, ls=ls, lw=2.0, label=ctry, zorder=3)
        axA.fill_between(grid, np.percentile(curves, 25, axis=0),
                         np.percentile(curves, 75, axis=0), color=col,
                         alpha=0.10, lw=0, zorder=2)
        jref = int(np.argmin(np.abs(grid - DIESEL_REF)))
        axA.plot([DIESEL_REF], [med[jref]], "o", ms=5, mfc=col, mec="white",
                 mew=1.1, zorder=5)
        # The three curves are within ~0.05 of each other at $300, so the
        # callouts are staggered vertically and given leader lines rather than
        # left to overlap.
        dy = (14, -2, -18)[_ci]
        axA.annotate(f"{ctry}  {med[jref]:.3f}", (DIESEL_REF, med[jref]),
                     textcoords="offset points", xytext=(14, dy), fontsize=7.3,
                     color=col, va="center", ha="left",
                     arrowprops=dict(arrowstyle="-", color=col, lw=0.6,
                                     shrinkA=0, shrinkB=2))
    axA.axvline(DIESEL_REF, color=INK3, lw=0.9, ls=":", zorder=1)
    axA.text(DIESEL_REF - 5, 0.03, "$300 reference", fontsize=7.2, color=INK2,
             ha="right")
    axA.axhline(0.5, color=GRAY_MID, lw=1.2, zorder=1)
    axA.set_xlabel("Diesel benchmark ($/MWhe)")
    axA.set_ylabel("P(viable)")
    axA.set_ylim(0, 1.02); axA.set_xlim(grid[0], grid[-1])
    axA.grid(color="#e8e7e4", lw=0.5); axA.set_axisbelow(True)
    axA.legend(loc="lower right", fontsize=7.6, labelcolor=INK2, handlelength=2.6,
               bbox_to_anchor=(1.0, 0.02))
    _head(axA, "A   Risk-adjusted viability threshold",
          f"Mean over facilities of P(breakeven < diesel), {draws:,} draws each. "
          "Phase 3.\nBand = inter-quartile range across facilities in that country.")

    # ---- B: breakeven distributions by country and phase
    # Phase 2 sits near -$225/MWhe while phases 3-4 straddle zero. On one axis
    # the phase-2 boxes collapse to flat lines at the floor and squash the
    # phases that carry the headline. A broken axis is the honest fix: two
    # stacked panels over the same categories, each with its own scale, and the
    # break stated in the subtitle so nobody reads it as one continuous range.
    phases = [p for p in (2, 3, 4) if (mc.phase == p).any()]
    hi_ph = [p for p in phases if p != 2]
    lo_ph = [p for p in phases if p == 2]
    ctrys = [c for c in COUNTRY_STYLE if (mc.country == c).any()]
    # The figure already created a single axes in this grid cell. Splitting the
    # cell means that placeholder must GO -- leaving it in place kept an empty
    # 0-1 axes underneath the new pair, which is where the stray "0.0 ... 1.0"
    # ticks came from on both axes.
    axB.remove()
    sub_gs = gs[0, 1].subgridspec(2, 1, height_ratios=[2.4, 1.0], hspace=0.12)
    axB = fig.add_subplot(sub_gs[0])
    # NOT sharex: a shared axis plus an axes-transform break mark let the
    # decoration's 0-1 coordinates into the DATA limits and overwrote the
    # country ticks with 0.0-1.0. The two panels are aligned by setting the
    # same explicit xlim instead, which cannot leak.
    axBlo = fig.add_subplot(sub_gs[1]) if lo_ph else None

    def _boxes(ax, want):
        pos, data, cols, ticks = [], [], [], []
        for gi, ctry in enumerate(ctrys):
            for pi, ph in enumerate(phases):
                if ph not in want:
                    continue
                sl = mc[(mc.country == ctry) & (mc.phase == ph)]
                if not len(sl):
                    continue
                pos.append(gi * (len(phases) + 1) + pi)
                data.append(sl.be_p50.values)
                cols.append(BLUE_RAMP[2 + 2 * pi])
            ticks.append((gi * (len(phases) + 1) + (len(phases) - 1) / 2, ctry))
        if not data:
            return ticks
        bp = ax.boxplot(data, positions=pos, widths=0.72, patch_artist=True,
                        medianprops=dict(color=INK, lw=1.4),
                        whiskerprops=dict(color=INK3, lw=0.8),
                        capprops=dict(color=INK3, lw=0.8),
                        flierprops=dict(marker="o", ms=3, mfc="none",
                                        mec=INK3, mew=0.6))
        for patch, c in zip(bp["boxes"], cols):
            patch.set_facecolor(c); patch.set_edgecolor(SURFACE)
            patch.set_linewidth(1.6)
        ax.grid(axis="y", color="#e8e7e4", lw=0.5); ax.set_axisbelow(True)
        return ticks

    ticks = _boxes(axB, hi_ph)
    axB.axhline(DIESEL_REF, color=INK3, lw=0.9, ls=":")
    axB.text(ticks[-1][0] + 1.6, DIESEL_REF, " $300", fontsize=7.2,
             color=INK2, va="center")
    axB.axhline(0, color=GRAY_MID, lw=1.0)
    axB.set_ylabel("Median breakeven per facility ($/MWhe)")
    axB.legend(handles=[plt.Rectangle((0, 0), 1, 1,
                                      fc=BLUE_RAMP[2 + 2 * phases.index(p)],
                                      ec=SURFACE, label=f"Phase {p}")
                        for p in phases],
               loc="upper left", fontsize=7.6, labelcolor=INK2, ncol=len(phases),
               bbox_to_anchor=(0.0, 1.0))
    xlo = -0.8
    xhi = (len(ctrys) - 1) * (len(phases) + 1) + len(phases) - 1 + 1.2
    axB.set_xlim(xlo, xhi)
    if axBlo is not None:
        _boxes(axBlo, lo_ph)
        axBlo.set_xlim(xlo, xhi)
        axBlo.set_xticks([t[0] for t in ticks])
        axBlo.set_xticklabels([t[1] for t in ticks])
        axBlo.set_ylabel("Phase 2", fontsize=8, color=INK2)
        axBlo.spines["top"].set_visible(True)
        axBlo.spines["top"].set_color(INK3)
        axBlo.spines["top"].set_linestyle((0, (2, 3)))
        axB.set_xticks([])
        axB.spines["bottom"].set_visible(False)
    else:
        axB.set_xticks([t[0] for t in ticks])
        axB.set_xticklabels([t[1] for t in ticks])
    _head(axB, "B   Breakeven distribution by country and phase",
          "Box = across facilities; each facility's value is its Monte-Carlo "
          "median.\nAxis is BROKEN: Phase 2 has its own scale. Phase 1 is absent "
          "\u2014 no firm load.")

    # ---- C: Sobol S1 vs ST, phase 3 viability, with Spearman overlay
    S1 = sob[3]["viability"]["S1"]; ST = sob[3]["viability"]["ST"]
    order = np.argsort(-ST)
    y = np.arange(len(PARAMS))
    axC.barh(y + 0.19, ST[order], height=0.36, color=C_BLUE, label="Sobol $S_T$ (total)")
    axC.barh(y - 0.19, S1[order], height=0.36, color=C_ORANGE, label="Sobol $S_1$ (first-order)")
    sp = spear[spear.phase == 3].set_index("parameter")
    xmax = max(0.35, float(ST.max()) * 1.30)
    for k, yy in zip([PARAMS[i] for i in order], y):
        axC.text(xmax * 0.995, yy, f"|$\\rho$| {sp.loc[k,'abs_rho']:.2f}",
                 fontsize=7.0, color=INK2, va="center", ha="right")
    axC.text(xmax * 0.995, -0.85, "Spearman", fontsize=7.0, color=INK3,
             va="center", ha="right", style="italic")
    axC.set_yticks(y, [PARAM_LABEL[PARAMS[i]] for i in order], fontsize=7.6)
    axC.invert_yaxis()
    axC.set_xlabel("Sobol index on viability (Phase 3)")
    axC.set_xlim(0, xmax)
    axC.grid(axis="x", color="#e8e7e4", lw=0.5); axC.set_axisbelow(True)
    axC.legend(loc="lower left", fontsize=7.6, labelcolor=INK2,
               bbox_to_anchor=(0.18, 0.04))
    if publication:   # journal styling: no internal project labels
        _head(axC, "C   Sobol' indices and Spearman rank correlation",
              "$S_T-S_1$ is variance carried by interaction, which a rank "
              "correlation cannot detect.\n$|\\rho|$: Spearman rank correlation, "
              "the measure used in the single-phase study.")
    else:
        _head(axC, "C   Sobol global sensitivity vs Paper 1's Spearman",
              "$S_T-S_1$ is variance carried by interaction — invisible to a rank "
              "correlation.\n$|\\rho|$ is the Spearman statistic Paper 1 would have reported.")

    # ---- D: phase robustness under joint uncertainty
    ph_all = sorted(mc.phase.unique())
    ppos = np.array([float(p) for p in ph_all])
    pv = [mc[mc.phase == p].p_profit_pos.mean() for p in ph_all]
    cv = [np.nanmedian(np.abs(mc[mc.phase == p].profit_cv)) for p in ph_all]
    axD.bar(ppos - 0.17, pv, width=0.34, color=C_BLUE, label="P(profit > 0)")
    for x, v in zip(ppos - 0.17, pv):
        axD.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=7.4, color=INK2)
    axD2 = axD.twinx()          # NOT a dual-scale line chart: two different
    axD2.bar(ppos + 0.17, cv, width=0.34, color=C_ORANGE,   # bar families, each
             label="CV of annual profit")                   # with its own axis
    for x, v in zip(ppos + 0.17, cv):
        axD2.text(x, v * 1.03, f"{v:.2f}", ha="center", fontsize=7.4, color=INK2)
    axD.set_ylim(0, 1.28); axD2.set_ylim(0, max(cv) * 1.45)
    axD.set_xticks(ppos, [f"Phase {p}" for p in ph_all])
    axD.set_ylabel("P(profit > 0)", color=C_BLUE)
    axD2.set_ylabel("CV of annual profit (lower = more robust)", color=C_ORANGE)
    axD2.spines["top"].set_visible(False)
    axD.grid(axis="y", color="#e8e7e4", lw=0.5); axD.set_axisbelow(True)
    axD.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=C_BLUE, label="P(profit > 0)"),
                        plt.Rectangle((0, 0), 1, 1, fc=C_ORANGE, label="CV of profit")],
               loc="upper center", fontsize=7.6, labelcolor=INK2, ncol=2,
               bbox_to_anchor=(0.5, 1.02))
    _head(axD, "D   Robustness to joint uncertainty, by phase",
          "Profit is the only metric defined in all four phases, so it is the "
          "cross-phase\ncomparator. Breakeven is not — Phase 1 has no firm load.")

    if not publication:
      fig.suptitle("Monte Carlo uncertainty and Sobol global sensitivity of the phased model",
                 x=0.075, ha="left", fontsize=12.5, fontweight="bold", y=0.985)
      fig.text(0.075, 0.960,
             f"Paper 2 — From Flares to FLOPS  |  {draws:,} draws per facility-phase  |  "
             f"{len(PARAMS)} uncertain parameters  |  build FROZEN at the central node",
             fontsize=8, color=INK2, ha="left")
    for ext in ("png", "pdf"):
        fig.savefig(f"{stem}.{ext}", dpi=600, bbox_inches="tight")
    log(f"  wrote {stem}.png / .pdf  (600 dpi)")
    for ax, tag in [(axA, "A"), (axB, "B"), (axC, "C"), (axD, "D")]:
        bb = ax.get_tightbbox(fig.canvas.get_renderer()).transformed(
            fig.dpi_scale_trans.inverted())
        for e in ("png", "pdf"):
            fig.savefig(f"{stem}_panel{tag}.{e}", dpi=600,
                        bbox_inches=bb.expanded(1.06, 1.05))
    log(f"  wrote {stem}_panel[A-D].png / .pdf")


# =============================================================================
# SECTION 6 -- MAIN
# =============================================================================

def newest_checkpoint() -> Optional[str]:
    cands = []
    for d in sorted(os.listdir("."), reverse=True):
        p = os.path.join(d, "checkpoint_00.csv")
        if d.startswith("phased_sweep_results") and os.path.isfile(p):
            cands.append(p)
    return cands[0] if cands else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Monte Carlo UQ + Sobol global sensitivity for the phased model")
    ap.add_argument("--results", default=None,
                    help="checkpoint CSV (default: newest phased_sweep_results*/)")
    ap.add_argument("--draws", type=int, default=20000)
    ap.add_argument("--sobol-n", type=int, default=4096,
                    help="Saltelli base N; evaluations = N*(2d+2) = N*18")
    ap.add_argument("--seed", type=int, default=20260917)
    ap.add_argument("--compute-basis", choices=("colo", "gpu"), default="colo",
                    help="see FLAGGED ASSUMPTION 3; 'gpu' is for reporting D-10, "
                         "not for headline results")
    ap.add_argument("--gpu-kw", type=float, default=1.2)
    ap.add_argument("--power-share", type=float, default=0.10)
    ap.add_argument("--outdir", default="uq_results")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--publication", action="store_true",
                    help="journal styling of the figure: no suptitle or project labels "
                         "(numerical outputs are unchanged)")
    a = ap.parse_args(argv)

    LOG: List[str] = []

    def log(s=""):
        print(s, flush=True); LOG.append(str(s))

    t0 = time.time()
    res = a.results or newest_checkpoint()
    if not res or not os.path.isfile(res):
        log("No sweep checkpoint found. Run run_phased_sweep.py first.")
        return 2
    os.makedirs(a.outdir, exist_ok=True)

    log("=" * 86)
    log("  PAPER 2 -- MONTE CARLO UQ + SOBOL GLOBAL SENSITIVITY")
    log("=" * 86)
    log(f"  source        : {res}")
    log(f"  draws         : {a.draws:,} per facility-phase")
    log(f"  Sobol base N  : {a.sobol_n:,}")
    log(f"  seed          : {a.seed}")
    log(f"  compute basis : {a.compute_basis}")
    log("")
    log("  " + "-" * 82)
    log("  CONSERVATIVE SIMPLIFICATION -- STATED AS REQUIRED")
    log("  " + "-" * 82)
    log(textwrap.indent(textwrap.fill(
        "Technology and module selection are FIXED at the central node "
        "(mu = 1.0, r = 12%) and are not re-optimised per draw. Dispatch is "
        "frozen: served energy, fixed O&M, variable O&M and start-up cost keep "
        "their central-node values. Only the annualised capital term is "
        "recomputed, via AnnCap = Overnight * mu * CRF(r), because only it "
        "depends on the sampled CAPEX multiplier and discount rate.", 82), "  "))
    log("")
    log(textwrap.indent(textwrap.fill(
        "This biases the result in a known direction. A real optimiser handed a "
        "draw of high mu and high r would switch to a cheaper design and a "
        "smaller fleet, which can only lower the cost it achieves. Freezing the "
        "central build therefore OVER-STATES cost in precisely the draws that "
        "hurt, so every P(viable) below is a LOWER BOUND on the probability a "
        "fully re-optimised model would report. It is not an unbiased estimate "
        "and must not be reported as one.", 82), "  "))
    log("")
    log(textwrap.indent(textwrap.fill(
        "DISPATCH IS NOT FROZEN AGAINST THE AVAILABILITY DRAW (B-21). Genset "
        "availability changes how much energy exists, and the Eq. E9 compute "
        "cap is a min(), so a scalar rescaling of the frozen dispatch would be "
        "wrong -- by up to 32% at a 2 MW site. The four demand streams are "
        "therefore re-derived from the allocation rule itself, integrated over "
        "the hourly availability shape carried in the checkpoint. Agreement "
        "with the solver: " + SURROGATE_CHECK.split(': ', 1)[1] + ".", 82), "  "))
    log("")

    # B-21 / D-07: two marginals are sampled from the pilot record itself. If
    # that file is not beside the scripts the run silently falls back to a
    # parametric shape, which would quietly turn two MEASURED distributions into
    # two assumed ones. Say so loudly instead.
    _up, _ef = load_measured_uptime(), load_measured_efficiency()
    log("  " + "-" * 82)
    log("  EMPIRICAL MARGINALS FROM THE PILOT RECORD")
    log("  " + "-" * 82)
    for _nm, _v, _fb in (("genset_availability", _up, "Beta matched on mean and sd"),
                         ("miner_efficiency", _ef, "Triangular over the three routes")):
        if _v is None:
            log(f"    {_nm:<22} !! pilot_validation/pilot_daily_processed.csv NOT FOUND")
            log(f"    {'':<22}    falling back to {_fb} -- FLAGGED, not measured")
        else:
            log(f"    {_nm:<22} empirical, n = {_v.size}   "
                f"p10 {np.percentile(_v, 10):.4g}  median {np.median(_v):.4g}  "
                f"p90 {np.percentile(_v, 90):.4g}")
    log("")

    pri = Priors(compute_basis=a.compute_basis, gpu_kw=a.gpu_kw,
                 power_share=a.power_share)
    log("  " + "-" * 82)
    log("  FLAGGED ASSUMPTIONS (not traceable to a project file)")
    log("  " + "-" * 82)
    log(f"   1. Diesel triangular mode set to ${pri.diesel_mode:.0f}/MWhe. The brief "
        "gave bounds\n      (200, 450) but no mode; the project hard convention #2 "
        "supplies it.")
    log(f"   2. Difficulty growth Normal({pri.dif_mean:.0%}, {pri.dif_sd:.0%}) truncated "
        f"to [{pri.dif_trunc_lo:+.0%}, {pri.dif_trunc_hi:+.0%}] so that (1+g)^tau "
        "cannot go\n      negative. Truncation is symmetric in probability, not in value.")
    log(f"   3. Compute price is drawn in $/GPU-hr and must be converted to $/MWhe-IT.")
    log(f"      3a. Triangular mode {pri.gpu_mode} $/GPU-hr (brief gave bounds only).")
    log(f"      3b. {pri.gpu_kw} kW per accelerator at system level.")
    log(f"      3c. power_share = {pri.power_share:.2f} -- the facility sells POWER AND "
        "SPACE,\n          not GPU-hours, so it captures only the power component.")
    lo_m = float(pri.gpu_to_mwh(np.array([pri.gpu_lo]))[0])
    hi_m = float(pri.gpu_to_mwh(np.array([pri.gpu_hi]))[0])
    log(f"      -> sampled range maps to ${lo_m:,.0f}-${hi_m:,.0f}/MWhe-IT "
        f"against a ledger base of ${BASE['compute_price']:.0f}.")
    if a.compute_basis == "gpu":
        log("      !! BASIS 'gpu': the RAW conversion is in force. This credits the "
            "plant with\n         rent on accelerators it does not own. Reporting "
            "mode only. D-10 OPEN.")
    elif not (80.0 <= lo_m <= 260.0 and 80.0 <= hi_m <= 400.0):
        log("      !! the mapped range sits outside the 2026 wholesale colocation "
            "band of\n         roughly $110-205/MWhe. Revisit power_share before "
            "quoting these numbers.")
    else:
        log("      OK: straddles the 2026 wholesale colocation band (~$110-205/MWhe).")
    log(f"   4. Phase midpoints for compounding: {PHASE_MIDYEAR}")
    log(f"   5. Community tariff is NOT among the eight sampled parameters, so "
        "rev_comm is\n      held at its central value. D-03 remains OPEN and "
        "unquantified by this run.")
    log("")

    selftest_ppf(log)
    log("")
    log("  " + "-" * 82)
    log("  BASE LEDGERS")
    log("  " + "-" * 82)
    bls = load_base_ledgers(res, log)
    if not bls:
        log("  no usable base ledgers -- abort")
        return 3
    techs = sorted({b.technology for b in bls if b.phase in (3, 4)})
    log(f"  technologies frozen in for phases 3-4: {techs}")
    log("")

    log("  " + "-" * 82)
    log("  MONTE CARLO")
    log("  " + "-" * 82)
    mc = run_mc(pri, bls, a.draws, a.seed, log)
    grid = mc.attrs["diesel_grid"]
    mc.to_csv(os.path.join(a.outdir, "mc_summary.csv"), index=False)
    for ph in sorted(mc.phase.unique()):
        s = mc[mc.phase == ph]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            _p50, _p05, _p95 = (np.nanmedian(s.be_p50), np.nanmedian(s.be_p05),
                                np.nanmedian(s.be_p95))
        log(f"   phase {ph}: P(viable) mean {s.p_viable.mean():.4f}   "
            f"P(profit>0) mean {s.p_profit_pos.mean():.4f}   "
            f"median breakeven P50 {_p50:>9.2f}   "
            f"P05 {_p05:>9.2f}  P95 {_p95:>9.2f}")
    log("")
    log("   P(viable) by country, phase 3:")
    for c, s in mc[mc.phase == 3].groupby("country"):
        curves = np.vstack([np.array(json.loads(x)) for x in s.curve])
        m = curves.mean(axis=0)
        j = int(np.argmin(np.abs(m - 0.5)))
        log(f"     {c:<14s} n={len(s):2d}  P(viable)={s.p_viable.mean():.4f}  "
            f"P=0.5 crossing at ${grid[j]:.0f}/MWhe  "
            f"P(viable at $300)={m[int(np.argmin(np.abs(grid - DIESEL_REF)))]:.4f}")
    log("")

    log("  " + "-" * 82)
    log("  SOBOL GLOBAL SENSITIVITY")
    log("  " + "-" * 82)
    sob = run_sobol(pri, bls, a.sobol_n, a.seed, log)
    spear = spearman_table(pri, bls, min(a.draws, 20000), a.seed)
    spear.to_csv(os.path.join(a.outdir, "spearman.csv"), index=False)

    rows = []
    for ph, res_ph in sob.items():
        if not isinstance(ph, int):
            continue
        for out_name, idx in res_ph.items():
            for i, k in enumerate(PARAMS):
                rows.append(dict(phase=ph, output=out_name, parameter=k,
                                 S1=idx["S1"][i], ST=idx["ST"][i],
                                 interaction=idx["ST"][i] - idx["S1"][i]))
    sdf = pd.DataFrame(rows)
    sdf.to_csv(os.path.join(a.outdir, "sobol_indices.csv"), index=False)

    for ph in sorted({r["phase"] for r in rows}):
        for out_name in ("viability", "breakeven"):
            s = sdf[(sdf.phase == ph) & (sdf.output == out_name)]
            if not len(s):
                continue
            log(f"   phase {ph}, output = {out_name}")
            log(f"     {'parameter':<20s} {'S1':>8s} {'ST':>8s} "
                f"{'ST 95% CI':>17s} {'drift':>7s} {'ST-S1':>8s} "
                f"{'|rho|':>7s} {'Sob':>4s} {'Spr':>4s}  resolved")
            s = s.sort_values("ST", ascending=False)
            sp = spear[spear.phase == ph].sort_values("abs_rho", ascending=False)
            sp_rank = {r.parameter: i + 1 for i, r in enumerate(sp.itertuples())}
            for i, r in enumerate(s.itertuples()):
                rho = float(spear[(spear.phase == ph)
                                  & (spear.parameter == r.parameter)].abs_rho.iloc[0])
                k = PARAMS.index(r.parameter)
                ix = sob[ph][out_name]
                lo, hi = float(ix["ST_lo"][k]), float(ix["ST_hi"][k])
                drift = (abs(float(ix["ST_half"][k]) - r.ST)
                         if "ST_half" in ix else float("nan"))
                # "yes"   the interval excludes zero and is tight relative to ST
                # "zero"   the interval is entirely inside the noise floor
                # "NOISE"  neither -- do not report this index as a finding
                width = (hi - lo) / max(abs(r.ST), 1e-12)
                resolved = ("zero" if hi < 0.02 else
                            "yes" if (lo > 0.02 and width < 0.25) else "NOISE")
                log(f"     {PARAM_LABEL[r.parameter]:<20s} {r.S1:8.4f} {r.ST:8.4f} "
                    f"[{lo:7.4f},{hi:7.4f}] {drift:7.4f} "
                    f"{r.interaction:8.4f} {rho:7.3f} {i+1:4d} "
                    f"{sp_rank[r.parameter]:4d}  {resolved}")
            log("")

    # ---- what Sobol saw that Spearman could not -------------------------
    log("  " + "-" * 82)
    log("  INTERACTIONS SOBOL REVEALS THAT SPEARMAN CANNOT")
    log("  " + "-" * 82)
    log(textwrap.indent(textwrap.fill(
        "Spearman's rho measures monotone association between ONE input and the "
        "output. It is structurally blind to interaction: if two parameters "
        "matter only jointly, each can show a near-zero rho while together "
        "carrying most of the variance. ST - S1 is exactly that missing "
        "quantity, and S2 names the pair.", 82), "  "))
    log("")
    for ph in sorted({r["phase"] for r in rows}):
        v = sob[ph].get("viability")
        if v is None:
            continue
        S2 = v["S2"]
        pairs = [(S2[i, j], PARAMS[i], PARAMS[j])
                 for i in range(len(PARAMS)) for j in range(i + 1, len(PARAMS))
                 if np.isfinite(S2[i, j])]
        pairs.sort(key=lambda z: -abs(z[0]))
        sS1, sST = float(np.sum(v["S1"])), float(np.sum(v["ST"]))
        log(f"   phase {ph}: sum(S1) = {sS1:.4f}   sum(ST) = {sST:.4f}   "
            f"sum(ST) - sum(S1) = {sST - sS1:.4f}")
        if not (-0.05 <= sS1 <= 1.05) or sST < sS1:
            log("     !! sum(S1) outside [0, 1] or sum(ST) < sum(S1): the "
                "estimator has NOT converged at this N.\n"
                "        Raise --sobol-n before reading the second-order terms "
                "below; they are the\n        noisiest quantity in the whole "
                "analysis and fail first.")
        else:
            log("     sanity: 0 <= sum(S1) <= 1 <= sum(ST) holds -- additive "
                "share and interaction share are consistent.")
        for val, i, j in pairs[:4]:
            log(f"     S2[{PARAM_LABEL[i]} x {PARAM_LABEL[j]}] = {val:+.4f}")
        log("")

    # ---- phase robustness ------------------------------------------------
    log("  " + "-" * 82)
    log("  PHASE ROBUSTNESS UNDER JOINT UNCERTAINTY")
    log("  " + "-" * 82)
    log(f"     {'phase':>5s} {'P(profit>0)':>12s} {'CV(profit)':>11s} "
        f"{'P(viable)':>10s} {'be P05':>10s} {'be P95':>10s} {'be spread':>10s}")
    for ph in sorted(mc.phase.unique()):
        s = mc[mc.phase == ph]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            p05, p95 = np.nanmedian(s.be_p05), np.nanmedian(s.be_p95)
        log(f"     {ph:5d} {s.p_profit_pos.mean():12.4f} "
            f"{np.nanmedian(np.abs(s.profit_cv)):11.3f} {s.p_viable.mean():10.4f} "
            f"{p05:10.2f} {p95:10.2f} {p95 - p05:10.2f}")
    log("")
    r1 = mc[mc.phase == 1]; r3 = mc[mc.phase == 3]
    if len(r1) and len(r3):
        c1 = float(np.nanmedian(np.abs(r1.profit_cv)))
        c3 = float(np.nanmedian(np.abs(r3.profit_cv)))
        verdict = ("MORE robust" if c1 < c3 else "LESS robust")
        log(textwrap.indent(textwrap.fill(
            f"Phase 1 (flare gas + Bitcoin) is {verdict} than Phase 3 (SMR + data "
            f"centre) to joint uncertainty: CV of annual profit {c1:.3f} vs "
            f"{c3:.3f}, and P(profit > 0) {r1.p_profit_pos.mean():.4f} vs "
            f"{r3.p_profit_pos.mean():.4f}. Breakeven cannot serve as the "
            "comparator because Phase 1 has no firm load and therefore no Eq. E20 "
            "denominator.", 82), "  "))
    log("")

    if not a.no_figures:
        log("  " + "-" * 82)
        log("  FIGURES")
        log("  " + "-" * 82)
        try:
            make_figures(mc, sob, spear, pri, a.draws, log,
                         stem=os.path.join(a.outdir, "fig_uq_phased"),
                         publication=a.publication)
        except Exception as e:                # noqa: BLE001
            log(f"  figure generation failed: {type(e).__name__}: {e}")

    log("")
    log(f"  wrote {a.outdir}/mc_summary.csv, sobol_indices.csv, spearman.csv")
    log(f"  elapsed {time.time() - t0:.1f}s")
    with open(os.path.join(a.outdir, "uq_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    print(f"\n  log written to {a.outdir}/uq_log.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
