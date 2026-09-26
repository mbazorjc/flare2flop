#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analytical_benchmark.py
=======================
GROUND-TRUTH ENGINE for Paper 2, "From Flares to FLOPS: A Phased Techno-Economic
Framework for Stranded Gas, Small Modular Reactors, and Compute Anchor Loads in
Sub-Saharan Africa".

PURE PYTHON. NO SOLVER. NO OPTIMISATION. Every number this file produces is a
closed-form arithmetic expression that can be reproduced with a pocket calculator.
That is the whole point: this module is the Level-1 analytic reconstruction that
hard convention #10 requires BEFORE any Pyomo output is trusted.

Dependencies: numpy, pandas (stdlib otherwise). Nothing else, by design.

-------------------------------------------------------------------------------
LINEAGE
-------------------------------------------------------------------------------
Paper 1 : Main_Manuscript_APEN_SMR_DataCentres_SSA.docx
          Eq. D23 (profit, affine in diesel price d)
          Eq. D24 (breakeven, d* at Pi = 0)
Paper 1 code : V26_external_use-ssa1-patch.py  (ledger sign convention, L1210-1222)
               corrected-postproc.py           (post-processed breakeven, L64-66)
Reactors : baseiaea.csv (IAEA ARIS 2024). NEVER invent reactor parameters.
Facilities : ADC.csv. NEVER invent facility data.
Bitcoin : Lal, Zhu & You (2023), Eq. 11 and Table 3.
Parameters : data_inputs.csv (Day-2 master parameter file).

-------------------------------------------------------------------------------
THE TWO EQUATIONS THIS FILE IMPLEMENTS
-------------------------------------------------------------------------------
Paper 1, Eq. D23 (positive-cost convention; all C_* are POSITIVE costs):

    Pi(d) = R_export + d*S - ( C_cap + C_fom + C_vom+fuel + C_start + C_TES )

Paper 1, Eq. D24 (set Pi = 0):

    d* = ( C_cap + C_fom + C_vom+fuel + C_start + C_TES - R_export ) / S     (D24)

Paper 2, unified breakeven (NOVEL; this file's core):

    d*_unified = [ AnnCap + Fixed_OM + VOM_fuel + StartUp
                   - rev_BTC - rev_carbon - rev_penalty - rev_comm
                   - rev_compute - R_export ] / Served                       (U1)

WHY THE EXTENSION IS LEGITIMATE. Every new revenue term is a price-independent
additive offset: none is a function of the diesel price d. Pi(d) therefore stays
affine in d, so the breakeven remains recoverable in closed form exactly as in
Paper 1. If a future revenue stream DOES depend on d, this derivation breaks and
the closed form must be re-derived. Flag it if that ever happens.

REGRESSION GATE V1: set rev_BTC = rev_carbon = rev_penalty = rev_comm =
rev_compute = 0 and (U1) must return Eq. D24 EXACTLY -- to the cent, not to
+/-2%. test_regression_to_paper1() enforces this.

-------------------------------------------------------------------------------
KNOWN DEFECT B-01 (inherited from Paper 1 tooling) -- see section 5
-------------------------------------------------------------------------------
corrected-postproc.py computes

    num = -(ACC_csv + Fixed_csv + SUC_csv - EP_csv)

The BestPer.csv ledger (V26_external_use-ssa1-patch.py L1210-1222) stores costs
NEGATED and export revenue POSITIVE:

    ACC_csv = -|C_cap| ; FOMC_csv = -|C_fom| ; SUC_csv = -|C_start| ;
    EP_csv  = +|R_export|

so that expression expands to |A| + |F| + |S| + |E| -- export revenue is ADDED to
the cost numerator instead of subtracted. The defect is DORMANT in Paper 1 because
V26 L424 sets ExportCap = 0 for LoadType='DataCenter', forcing vExport = 0 and
EP_csv = 0; Paper 1's published numbers are unaffected. It goes LIVE in Paper 2
Phases 2 and 4, which introduce community and grid offtake. This module implements
the CORRECT sign (export subtracted) and paper1_breakeven_from_ledger() reproduces
the buggy expression side by side so the divergence is visible, not silent.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
# Files are looked up in this module's own directory first, then its parent, so
# the module works whether it sits beside baseiaea.csv in the working folder or
# one level down in paper2_analytics/.
SEARCH_PATH = (HERE, os.path.dirname(HERE), os.getcwd())


def _find(filename: str):
    """First existing copy of `filename` along SEARCH_PATH, else None."""
    for d in SEARCH_PATH:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None
SECONDS_PER_YEAR = 365 * 24 * 3600          # 31,536,000 s (non-leap)
HOURS_PER_YEAR = 8760
TWO_POW_32 = 2 ** 32                        # 4,294,967,296  (Lal Eq. 11 denominator)
MMBTU_PER_MWH = 3.412141633                 # thermal conversion


# =============================================================================
# SECTION 1 -- PARAMETER LOADING
# =============================================================================
# Resolution order:
#   1. `import data_inputs` (a Python module, if Day 2 produces one)
#   2. data_inputs.csv in this directory (the Day-2 master parameter file)
#   3. embedded EMBEDDED_DEFAULTS, with a loud warning
#
# Every parameter carries a STATUS:
#   FILE        -- read from a file in the working folder. Trustworthy.
#   CONVENTION  -- a project hard convention (STEP 3). Trustworthy by fiat.
#   PLACEHOLDER -- NOT traceable to any file. Hard convention #13 requires this
#                  be flagged in every output. Day-2 collection item.
# -----------------------------------------------------------------------------

EMBEDDED_DEFAULTS: Dict[str, tuple] = {
    # key                 value        unit            status         source
    "discount_rate":      (0.12,       "-",            "CONVENTION",  "Convention #1"),
    "plant_life_years":   (25,         "yr",           "CONVENTION",  "Convention #1"),
    "diesel_reference":   (300.0,      "$/MWhe",       "CONVENTION",  "Convention #2"),
    "ef_diesel":          (0.70,       "tCO2/MWhe",    "FILE",        "corrected-postproc.py L31"),
    "ef_nuclear":         (0.012,      "tCO2/MWhe",    "FILE",        "corrected-postproc.py L31"),
    "pue":                (1.5,        "-",            "CONVENTION",  "Convention #4"),
    "flare_penalty":      (3.50,       "$/mscf",       "CONVENTION",  "Convention #5"),
    "btc_block_reward":   (3.125,      "BTC/block",    "CONVENTION",  "Convention #6"),
    "genset_efficiency":  (0.38,       "-",            "CONVENTION",  "Convention #7"),
    "community_share":    (0.10,       "-",            "CONVENTION",  "Convention #8"),
    "export_cap":         (0.0,        "-",            "CONVENTION",  "Convention #9"),
    "carbon_price":       (50.0,       "$/tCO2",       "PLACEHOLDER", "D-04 OPEN"),
    "community_tariff":   (100.0,      "$/MWhe",       "PLACEHOLDER", "D-03 OPEN"),
    "btc_spot_price":     (60000.0,    "$/BTC",        "PLACEHOLDER", "D-06 OPEN"),
    "btc_difficulty":     (1.10e14,    "-",            "PLACEHOLDER", "D-06 OPEN"),
    "miner_efficiency":   (17.5,       "J/TH",         "PLACEHOLDER", "D-07 OPEN"),
    "gas_hhv":            (1037.0,     "Btu/scf",      "PLACEHOLDER", "D-08 OPEN"),
    "ef_natural_gas":     (53.06,      "kgCO2/MMBtu",  "PLACEHOLDER", "D-08 OPEN"),
    "genset_capex":       (800.0,      "$/kWe",        "PLACEHOLDER", "D-09 OPEN"),
    "genset_fopex":       (40.0,       "$/kWe-yr",     "PLACEHOLDER", "D-09 OPEN"),
    "genset_vom_fc":      (5.0,        "$/MWhe",       "PLACEHOLDER", "D-09 OPEN"),
    "compute_price":      (120.0,      "$/MWhe-IT",    "PLACEHOLDER", "D-10 OPEN"),
    "re_capex":           (1100.0,     "$/kWe",        "PLACEHOLDER", "D-11 OPEN"),
    "re_fopex":           (22.64,      "$/kW-yr",      "FILE",        "Lal Table 3 opfac_solar"),
    "re_capacity_factor": (0.20,       "-",            "PLACEHOLDER", "D-11 OPEN"),
    "capacity_factor":    (0.95,       "-",            "PLACEHOLDER", "Prompt-0 brief"),
    "genset_capacity_factor": (0.90,   "-",            "PLACEHOLDER", "D-09 OPEN"),
    "site_2mw_capacity":  (2.0,        "MWe",          "FILE",        "Barrels deck"),
}


class Params:
    """Parameter store that never loses provenance.

    P['carbon_price']      -> the float
    P.status('carbon_price') -> 'PLACEHOLDER'
    P.placeholders()       -> every untraceable key currently in play
    """

    def __init__(self, table: pd.DataFrame, origin: str):
        self.table = table
        self.origin = origin

    # -- construction ---------------------------------------------------------
    @classmethod
    def load(cls, path: Optional[str] = None) -> "Params":
        # (1) a Python module, if Day 2 ships one
        try:
            sys.path.insert(0, HERE)
            import data_inputs as _di  # type: ignore
            if hasattr(_di, "as_dataframe"):
                return cls(_di.as_dataframe(), "data_inputs.py")
            if hasattr(_di, "PARAMS"):
                rows = [
                    {"key": k, "value": v[0], "unit": v[1], "status": v[2], "source": v[3]}
                    for k, v in _di.PARAMS.items()
                ]
                return cls(pd.DataFrame(rows).set_index("key"), "data_inputs.py")
        except ImportError:
            pass
        # (2) the Day-2 CSV
        csv = path or _find("data_inputs.csv")
        if csv and os.path.exists(csv):
            df = pd.read_csv(csv, comment="#")
            df = df.dropna(subset=["key"])
            df["value"] = pd.to_numeric(df["value"], errors="coerce")
            return cls(df.set_index("key"), os.path.relpath(csv))
        # (3) embedded fallback
        sys.stderr.write(
            "\n!! data_inputs not found. Falling back to EMBEDDED_DEFAULTS.\n"
            "!! Numbers below are NOT reproducible from the working folder.\n\n"
        )
        rows = [
            {"key": k, "value": v[0], "unit": v[1], "status": v[2], "source": v[3]}
            for k, v in EMBEDDED_DEFAULTS.items()
        ]
        return cls(pd.DataFrame(rows).set_index("key"), "EMBEDDED_DEFAULTS")

    # -- access ---------------------------------------------------------------
    def __getitem__(self, key: str) -> float:
        if key not in self.table.index:
            if key in EMBEDDED_DEFAULTS:
                return float(EMBEDDED_DEFAULTS[key][0])
            raise KeyError(f"parameter {key!r} not in {self.origin} and no embedded default")
        return float(self.table.loc[key, "value"])

    def status(self, key: str) -> str:
        if key in self.table.index:
            return str(self.table.loc[key, "status"])
        return EMBEDDED_DEFAULTS.get(key, (None, None, "UNKNOWN", ""))[2]

    def source(self, key: str) -> str:
        if key in self.table.index:
            return str(self.table.loc[key, "source"])
        return EMBEDDED_DEFAULTS.get(key, (None, None, "", "UNKNOWN"))[3]

    def placeholders(self) -> pd.DataFrame:
        t = self.table
        return t[t["status"].astype(str).str.upper() == "PLACEHOLDER"]


def load_reactors(path: Optional[str] = None) -> pd.DataFrame:
    """baseiaea.csv -> DataFrame indexed by design name.

    Hard convention #11: reactor parameters come from this file or nowhere.
    """
    p = path or _find("baseiaea.csv")
    if not p or not os.path.exists(p):
        raise FileNotFoundError(
            f"baseiaea.csv not found on {SEARCH_PATH}. Convention #11 forbids inventing "
            "reactor parameters, so this module refuses to proceed."
        )
    return pd.read_csv(p).set_index("Sites")


# =============================================================================
# SECTION 2 -- PRIMITIVE CLOSED FORMS
# =============================================================================

def crf(r: float, n: int) -> float:
    """Capital recovery factor. Hard convention #1.

        CRF = r / ( 1 - (1+r)^-n )

    Hand-check at r=0.12, n=25:
        1.12^25       = 17.00006
        1.12^-25      = 0.0588233
        1 - 0.0588233 = 0.9411767
        0.12 / 0.9411767 = 0.1275000
    """
    if r == 0:
        return 1.0 / n                      # limiting case, undiscounted
    return r / (1.0 - (1.0 + r) ** (-n))


def annualised_capital(capex_overnight: float, capex_mult: float, r: float, n: int) -> float:
    """AnnCap = CAPEX_overnight x CAPEX_mult x CRF(r, n).   ADAPTED (Paper 1)."""
    return capex_overnight * capex_mult * crf(r, n)


def overnight_capex(capex_per_kwe: float, nameplate_mwe: float) -> float:
    """Overnight CAPEX in $ from a $/kWe catalogue rate. 1 MWe = 1000 kWe."""
    return capex_per_kwe * nameplate_mwe * 1000.0


def fixed_om(fopex_per_kwe: float, nameplate_mwe: float) -> float:
    """Fixed O&M in $/yr from a $/kWe-yr catalogue rate.

    NOTE ON THE BRIEF'S SIGNATURE. The specification wrote
        fixed = fopex_per_mwe * capex_basis_mwe
    with `capex_basis_mwe` not among the function's arguments. baseiaea.csv
    reports FOPEX in $/kWe-yr, so the correct expression needs the x1000 unit
    conversion and an explicit nameplate argument. Both are supplied here.
    Omitting the x1000 understates Fixed_OM by three orders of magnitude
    ($5,400 instead of $5,400,000 in the Prompt-0 example).
    """
    return fopex_per_kwe * nameplate_mwe * 1000.0


def served_energy(nameplate_mwe: float, capacity_factor: float,
                  demand_mw: Optional[float] = None) -> float:
    """Annual delivered energy, MWhe/yr.

        Served = min( nameplate x CF , demand ) x 8760

    Behind-the-meter (convention #9) the fleet cannot sell surplus, so served
    energy is capped by demand. Paper 1's capacity-adequacy rule additionally
    requires nameplate >= peak demand whenever a build occurs.
    """
    supply = nameplate_mwe * capacity_factor
    if demand_mw is not None:
        supply = min(supply, demand_mw)
    return supply * HOURS_PER_YEAR


# =============================================================================
# SECTION 3 -- THE CORE: UNIFIED BREAKEVEN
# =============================================================================

def unified_breakeven(
    capex_overnight: float,
    capex_mult: float,
    r: float,
    n: int,
    fopex_per_kwe: float,
    nameplate_mwe: float,
    vom_fc_per_mwhe: float,
    served_mwhe: float,
    rev_btc: float = 0.0,
    generation_mwhe: float = None,
    rev_carbon: float = 0.0,
    rev_penalty: float = 0.0,
    rev_comm: float = 0.0,
    rev_compute: float = 0.0,
    startup: float = 0.0,
    export: float = 0.0,
):
    """Unified diesel breakeven, $/MWhe.   Eq. (U1), NOVEL.

        d* = [ AnnCap + Fixed_OM + VOM_fuel + StartUp
               - rev_BTC - rev_carbon - rev_penalty - rev_comm - rev_compute
               - export ] / Served

    Positive-cost convention throughout: every cost argument is a POSITIVE
    number and every revenue argument is a POSITIVE number. The signs in the
    expression, not in the inputs, do the work. See section 5 for the mapping
    to the BestPer.csv ledger, which negates costs.

    Two deviations from the brief's signature, both deliberate:

      * `nameplate_mwe` added, and `fopex_per_mwe` renamed `fopex_per_kwe`.
        baseiaea.csv reports FOPEX in $/kWe-yr; the brief's pseudocode
        referenced an undefined `capex_basis_mwe`. See fixed_om().

      * `startup` added. Paper 1's Eq. D24 carries C_start explicitly. It is
        ~0 for a baseload SMR (defect B-02) but is NOT negligible for the
        Phase 1-2 cycling gensets and curtailable mining load, where it must
        be carried. Defaulting it to 0 preserves the brief's behaviour.

    `export` is SUBTRACTED, matching manuscript Eq. D24. This is the correct
    sign and it differs from the literal expression in corrected-postproc.py
    (defect B-01). Under convention #9 export = 0 in the baseline, so the two
    agree numerically there and diverge only in Phases 2 and 4.

    Returns (breakeven_$/MWhe, components_dict).
    """
    if served_mwhe <= 0:
        raise ValueError("served_mwhe must be positive; a zero-energy build has no breakeven")

    anncap = annualised_capital(capex_overnight, capex_mult, r, n)
    fixed = fixed_om(fopex_per_kwe, nameplate_mwe)
    # VOM and fuel are incurred on every megawatt-hour GENERATED, not on the
    # subset delivered to the diesel-displacing load. The two coincide only when
    # served energy is the whole output -- true in Paper 1, false as soon as
    # community offtake or mining takes a share. `generation_mwhe` defaults to
    # `served_mwhe` so existing callers and gate V1 are unaffected.
    gen_basis = served_mwhe if generation_mwhe is None else generation_mwhe
    vom = vom_fc_per_mwhe * gen_basis

    total_cost = anncap + fixed + vom + startup
    total_revenue = rev_btc + rev_carbon + rev_penalty + rev_comm + rev_compute + export
    numerator = total_cost - total_revenue

    components = dict(
        crf=crf(r, n),
        anncap=anncap,
        fixed=fixed,
        vom=vom,
        generation_mwhe=gen_basis,
        startup=startup,
        total_cost=total_cost,
        rev_btc=rev_btc,
        rev_carbon=rev_carbon,
        rev_penalty=rev_penalty,
        rev_comm=rev_comm,
        rev_compute=rev_compute,
        export=export,
        total_revenue=total_revenue,
        numerator=numerator,
        served_mwhe=served_mwhe,
    )
    return numerator / served_mwhe, components


# =============================================================================
# SECTION 4 -- REVENUE-STREAM CLOSED FORMS
# =============================================================================

def btc_revenue(spot_price: float, block_reward: float, hashrate_hs: float,
                seconds: float, difficulty: float) -> float:
    """Bitcoin mining revenue.   ADAPTED from Lal, Zhu & You (2023) Eq. 11.

        rev = SP_BIT x R x H x t_m / ( D x 2^32 )

    Units: SP_BIT [$/BTC], R [BTC/block], H [hashes/s], t_m [s], D [-].
    R = 3.125 post-2024 halving (convention #6). Lal's Table 3 value of 6.25
    is pre-halving and must NOT be used.

    Dimensional check: H x t_m = hashes tried; D x 2^32 = expected hashes per
    block; the ratio is blocks won; x R x SP_BIT = dollars.
    """
    return spot_price * block_reward * hashrate_hs * seconds / (difficulty * TWO_POW_32)


def hashrate_from_power(power_mw: float, efficiency_j_per_th: float) -> float:
    """Miner hashrate [H/s] from electrical draw and ASIC efficiency.

        H = P[W] / eff[J/TH] x 1e12 H/TH

    efficiency_j_per_th is a PLACEHOLDER (D-07): Lal Table 3 gives a per-miner
    power of 3.25 kW but no hashrate, so the J/TH figure is not file-traceable.
    """
    return (power_mw * 1e6) / efficiency_j_per_th * 1e12


def carbon_revenue(served_mwhe: float, ef_displaced: float, ef_source: float,
                   carbon_price: float) -> float:
    """Carbon credit.   NOVEL.

        rev_carbon = Served x ( EF_displaced - EF_source ) x Price_carbon

    EF_displaced defaults to diesel (0.70 tCO2/MWhe) throughout Paper 2 because
    the counterfactual everywhere is unabated diesel self-generation, exactly as
    in Paper 1's abatement supply curve.
    """
    return served_mwhe * (ef_displaced - ef_source) * carbon_price


def penalty_revenue(volume_mscf: float, penalty_per_mscf: float) -> float:
    """Avoided flare penalty.   NOVEL. Convention #5 (PIA 2021, $3.50/mscf).

        rev_penalty = V_captured x 3.50
    """
    return volume_mscf * penalty_per_mscf


def flare_volume_mscf(electric_mwhe: float, genset_efficiency: float,
                      gas_hhv_btu_per_scf: float) -> float:
    """Associated gas consumed to produce a given electrical output, in mscf.

        E_thermal[MWh] = E_electric / eta                        (convention #7)
        V[mscf]        = E_thermal x 3.4121 MMBtu/MWh / (HHV/1000) MMBtu/mscf

    gas_hhv is a PLACEHOLDER (D-08): no working-folder file reports the heating
    value of Niger Delta associated gas.
    """
    thermal_mwh = electric_mwhe / genset_efficiency
    mmbtu = thermal_mwh * MMBTU_PER_MWH
    mmbtu_per_mscf = gas_hhv_btu_per_scf / 1000.0
    return mmbtu / mmbtu_per_mscf


def ef_from_genset(genset_efficiency: float, ef_gas_kg_per_mmbtu: float) -> float:
    """Emission factor of a gas genset, tCO2/MWhe.   NOVEL.

        EF = (3.4121 / eta) MMBtu/MWhe x EF_gas kg/MMBtu / 1000

    At eta = 0.38 and 53.06 kgCO2/MMBtu this is 0.476 tCO2/MWhe, i.e. about
    two-thirds of diesel. Burning captured flare gas is therefore a PARTIAL
    abatement, not a zero-carbon one -- the paper must say so.

    FLAG: no methane-slip credit is taken. Routine flaring destroys ~98% of the
    methane, so the incremental CH4 benefit of capture is second-order and is
    NOT claimed here. If the paper later claims it, it needs its own citation
    and its own uncertainty treatment.
    """
    return (MMBTU_PER_MWH / genset_efficiency) * ef_gas_kg_per_mmbtu / 1000.0


def community_revenue(served_mwhe: float, community_share: float, tariff: float) -> float:
    """Community power sales.   NOVEL.

        E_comm     = phi_comm x Served          (convention #8, 10-15%)
        rev_comm   = E_comm x Price_elec

    DEVIATION FROM THE BRIEF, deliberate. The brief writes
        rev_comm = P_comm x Price_elec x 8760
    which equals phi x nameplate x 8760 x price, i.e. 10% of NAMEPLATE-HOURS,
    not 10% of GENERATION. That conflicts with convention #8 whenever CF < 1.
    Open decision D-02. The generation-based form is used here; the literal
    form is reproduced in community_revenue_literal() so the gap is visible.
    """
    return community_share * served_mwhe * tariff


def community_revenue_literal(nameplate_mwe: float, community_share: float,
                              tariff: float) -> float:
    """The brief's literal form, kept only so D-02 can be quantified."""
    return community_share * nameplate_mwe * tariff * HOURS_PER_YEAR


def compute_revenue(served_mwhe: float, pue: float, price_per_mwhe_it: float,
                    compute_share: float = 1.0) -> float:
    """AI/HPC compute sales.   NOVEL.

        E_IT      = Served x compute_share / PUE                  (convention #4)
        rev_compute = E_IT x Price_compute

    Served energy is FACILITY energy; compute is sold per IT megawatt-hour, so
    the PUE divisor is required. Omitting it overstates revenue by 50% at
    PUE = 1.5. Price_compute is a PLACEHOLDER (D-10).
    """
    return served_mwhe * compute_share / pue * price_per_mwhe_it


# =============================================================================
# SECTION 5 -- PAPER 1 REFERENCE IMPLEMENTATIONS AND THE LEDGER SIGN MAPPING
# =============================================================================

def paper1_breakeven(anncap: float, fixed: float, vom_fuel: float = 0.0,
                     startup: float = 0.0, tes: float = 0.0,
                     export: float = 0.0, served: float = 1.0) -> float:
    """Paper 1 Eq. D24, positive-cost convention. The regression target.

        d* = ( C_cap + C_fom + C_vom+fuel + C_start + C_TES - R_export ) / S
    """
    return (anncap + fixed + vom_fuel + startup + tes - export) / served


def paper1_breakeven_from_ledger(acc_csv: float, fomc_csv: float, suc_csv: float,
                                 ep_csv: float, served: float,
                                 buggy: bool = False) -> float:
    """Recompute Paper 1's breakeven from raw BestPer.csv columns.

    Ledger convention (V26_external_use-ssa1-patch.py L1210-1222):
        ACC_csv = -|C_cap| ; FOMC_csv = -|C_fom| ; SUC_csv = -|C_start|
        EP_csv  = +|R_export|          <-- NOT negated

    buggy=False -> the CORRECT expression, export subtracted (Eq. D24).
    buggy=True  -> the literal expression in corrected-postproc.py L64-66,
                   which adds export revenue to the cost numerator (defect B-01).
                   Reproduced here only so the divergence can be measured.
    """
    if buggy:
        return -(acc_csv + fomc_csv + suc_csv - ep_csv) / served
    return (-acc_csv - fomc_csv - suc_csv - ep_csv) / served


# =============================================================================
# SECTION 6 -- REGRESSION TESTS (the V1 verification gate)
# =============================================================================

def test_regression_to_paper1(verbose: bool = True) -> bool:
    """VERIFICATION GATE V1.

    With all Paper-2 revenue streams zero, unified_breakeven() must equal
    Paper 1's Eq. D24 EXACTLY -- to the cent, not to +/-2%. Any residual is a
    model defect. Four assertions:

      1. Zero-revenue collapse, export = 0  -> equals D24 exactly.
      2. Zero-revenue collapse, export > 0  -> still equals D24 exactly, and
         the breakeven FALLS, because export revenue offsets cost.
      3. Ledger round-trip: the corrected sign reproduces (1) from raw
         BestPer.csv columns.
      4. Defect B-01 is dormant at export = 0 and live at export > 0, by the
         exact amount 2 x export / served.
    """
    P = Params.load()
    reactors = load_reactors()
    rx = reactors.loc["CAREM-25"]

    mwe = float(rx["Power in MWe"])
    oc = overnight_capex(float(rx["CAPEX $/kWe"]), mwe)
    fop = float(rx["FOPEX $/kWe"])
    vomfc = float(rx["VOM in $/MWh-e"]) + float(rx["FC in $/MWh-e"])
    r, n = P["discount_rate"], int(P["plant_life_years"])
    served = served_energy(mwe, P["capacity_factor"])
    startup = 0.0

    ok = True
    lines = []

    # -- 1. zero-revenue collapse, no export ---------------------------------
    be_u, c = unified_breakeven(oc, 1.0, r, n, fop, mwe, vomfc, served, startup=startup)
    be_p1 = paper1_breakeven(c["anncap"], c["fixed"], c["vom"], startup, 0.0, 0.0, served)
    d1 = abs(be_u - be_p1)
    ok &= d1 < 1e-9
    lines.append(f"  [1] collapse, export=0       unified={be_u:12.6f}  D24={be_p1:12.6f}  |d|={d1:.2e}  {'PASS' if d1 < 1e-9 else 'FAIL'}")

    # -- 2. zero-revenue collapse WITH export --------------------------------
    exp = 1_500_000.0
    be_ue, ce = unified_breakeven(oc, 1.0, r, n, fop, mwe, vomfc, served,
                                  startup=startup, export=exp)
    be_p1e = paper1_breakeven(ce["anncap"], ce["fixed"], ce["vom"], startup, 0.0, exp, served)
    d2 = abs(be_ue - be_p1e)
    ok &= d2 < 1e-9 and be_ue < be_u
    lines.append(f"  [2] collapse, export>0       unified={be_ue:12.6f}  D24={be_p1e:12.6f}  |d|={d2:.2e}  {'PASS' if d2 < 1e-9 and be_ue < be_u else 'FAIL'}")

    # -- 3. ledger round-trip, corrected sign --------------------------------
    acc_csv, fomc_csv, suc_csv = -c["anncap"], -c["fixed"], -startup
    be_led = paper1_breakeven_from_ledger(acc_csv, fomc_csv, suc_csv, 0.0, served, buggy=False)
    be_novom = paper1_breakeven(c["anncap"], c["fixed"], 0.0, startup, 0.0, 0.0, served)
    d3 = abs(be_led - be_novom)
    ok &= d3 < 1e-9
    lines.append(f"  [3] ledger round-trip        ledger={be_led:12.6f}  D24={be_novom:12.6f}  |d|={d3:.2e}  {'PASS' if d3 < 1e-9 else 'FAIL'}")

    # -- 4. defect B-01: dormant at export=0, live at export>0 ---------------
    good0 = paper1_breakeven_from_ledger(acc_csv, fomc_csv, suc_csv, 0.0, served, buggy=False)
    bad0 = paper1_breakeven_from_ledger(acc_csv, fomc_csv, suc_csv, 0.0, served, buggy=True)
    goodE = paper1_breakeven_from_ledger(acc_csv, fomc_csv, suc_csv, exp, served, buggy=False)
    badE = paper1_breakeven_from_ledger(acc_csv, fomc_csv, suc_csv, exp, served, buggy=True)
    dormant = abs(good0 - bad0) < 1e-9
    live = abs((badE - goodE) - 2.0 * exp / served) < 1e-6
    ok &= dormant and live
    lines.append(f"  [4] B-01 dormant @ export=0  good={good0:12.6f}  buggy={bad0:12.6f}  {'PASS' if dormant else 'FAIL'}")
    lines.append(f"      B-01 live    @ export>0  good={goodE:12.6f}  buggy={badE:12.6f}  error=+${badE - goodE:,.2f}/MWhe (= 2E/S)  {'PASS' if live else 'FAIL'}")

    if verbose:
        print("\n" + "=" * 78)
        print("VERIFICATION GATE V1 -- regression of Eq. (U1) to Paper 1 Eq. D24")
        print("=" * 78)
        print("\n".join(lines))
        print(f"\n  GATE V1: {'PASS' if ok else 'FAIL'}")
    assert ok, "V1 regression gate FAILED -- Eq. (U1) does not collapse to Eq. D24"
    return ok


# =============================================================================
# SECTION 7 -- PROMPT-0 WORKED EXAMPLE
# =============================================================================

HAND_VALUE_NO_REVENUE = 143.84      # $/MWhe, hand-computed in D1_unified_breakeven_derivation.md
HAND_TOLERANCE = 5.00               # $/MWhe


def worked_example(P: Optional[Params] = None, reactors: Optional[pd.DataFrame] = None,
                   verbose: bool = True):
    """The Prompt-0 example: 20 MW IT, PUE 1.5, 1 x CAREM-25, r=12%, n=25, CF=0.95.

    Hand-computed reference (see D1_unified_breakeven_derivation.md):
        CRF     = 0.127500
        AnnCap  = $27,539,993/yr
        Fixed   = $5,400,000/yr
        Served  = 249,660 MWhe/yr
        (e) no-revenue breakeven = $143.84/MWhe
        (g) after carbon+community = $99.44/MWhe
    """
    P = P or Params.load()
    reactors = reactors if reactors is not None else load_reactors()
    rx = reactors.loc["CAREM-25"]

    it_load = 20.0
    pue = P["pue"]
    demand = it_load * pue                              # 30.0 MW
    mwe = float(rx["Power in MWe"])                     # 30.0 MWe, one module
    capex_rate = float(rx["CAPEX $/kWe"])
    fopex_rate = float(rx["FOPEX $/kWe"])
    vom = float(rx["VOM in $/MWh-e"])
    fc = float(rx["FC in $/MWh-e"])
    r, n = P["discount_rate"], int(P["plant_life_years"])
    cf = P["capacity_factor"]

    oc = overnight_capex(capex_rate, mwe)
    served = served_energy(mwe, cf, demand_mw=demand)

    # (e) no new revenue streams
    be_e, ce = unified_breakeven(oc, 1.0, r, n, fopex_rate, mwe, vom + fc, served)

    # (f) carbon
    rev_c = carbon_revenue(served, P["ef_diesel"], P["ef_nuclear"], P["carbon_price"])
    # (g) + community
    rev_k = community_revenue(served, P["community_share"], P["community_tariff"])
    rev_k_lit = community_revenue_literal(mwe, P["community_share"], P["community_tariff"])

    be_g, cg = unified_breakeven(oc, 1.0, r, n, fopex_rate, mwe, vom + fc, served,
                                 rev_carbon=rev_c, rev_comm=rev_k)
    served_dc = served - P["community_share"] * served
    be_g_b = cg["numerator"] / served_dc                       # D-01 convention (B)
    be_g_lit = (ce["numerator"] - rev_c - rev_k_lit) / served  # D-02 literal form

    rows = [
        (1, "IT load", "", 20.0, "MW", "brief"),
        (2, "PUE", "pue", pue, "-", f"{P.status('pue')}: {P.source('pue')}"),
        (3, "Facility demand", "D", demand, "MW", "1 x 2"),
        (4, "Annual demand", "", demand * HOURS_PER_YEAR, "MWhe/yr", "3 x 8760"),
        (5, "Reactor", "", float("nan"), "CAREM-25 x1", "baseiaea.csv"),
        (6, "Nameplate", "P", mwe, "MWe", "FILE: baseiaea.csv"),
        (7, "CAPEX rate", "", capex_rate, "$/kWe", "FILE: baseiaea.csv"),
        (8, "FOPEX rate", "", fopex_rate, "$/kWe-yr", "FILE: baseiaea.csv"),
        (9, "VOM", "", vom, "$/MWh-e", "FILE: baseiaea.csv"),
        (10, "Fuel cost", "", fc, "$/MWh-e", "FILE: baseiaea.csv"),
        (11, "Discount rate", "r", r, "-", f"{P.status('discount_rate')}"),
        (12, "Plant life", "n", n, "yr", f"{P.status('plant_life_years')}"),
        (13, "CRF", "CRF", ce["crf"], "1/yr", "r/(1-(1+r)^-n)"),
        (14, "Overnight CAPEX", "", oc, "$", "7 x 6 x 1000"),
        (15, "AnnCap", "", ce["anncap"], "$/yr", "14 x 13"),
        (16, "Fixed O&M", "", ce["fixed"], "$/yr", "8 x 6 x 1000"),
        (17, "Capacity factor", "CF", cf, "-", f"{P.status('capacity_factor')}"),
        (18, "Served energy", "S", served, "MWhe/yr", "min(6x17, 3) x 8760"),
        (19, "Serve ratio", "", served / (demand * HOURS_PER_YEAR), "-", "18 / 4"),
        (20, "VOM + fuel", "", ce["vom"], "$/yr", "(9+10) x 18"),
        (21, "Total annualised cost", "", ce["total_cost"], "$/yr", "15+16+20"),
        (22, "(e) BREAKEVEN, no new revenue", "", be_e, "$/MWhe", "21 / 18"),
        (23, "  - capital component", "", ce["anncap"] / served, "$/MWhe", "15 / 18"),
        (24, "  - fixed O&M component", "", ce["fixed"] / served, "$/MWhe", "16 / 18"),
        (25, "  - VOM + fuel component", "", vom + fc, "$/MWhe", "20 / 18"),
        (26, "EF diesel", "", P["ef_diesel"], "tCO2/MWhe", f"{P.status('ef_diesel')}"),
        (27, "EF nuclear", "", P["ef_nuclear"], "tCO2/MWhe", f"{P.status('ef_nuclear')}"),
        (28, "CO2 avoided", "", served * (P["ef_diesel"] - P["ef_nuclear"]), "tCO2/yr", "18 x (26-27)"),
        (29, "Carbon price", "", P["carbon_price"], "$/tCO2", f"{P.status('carbon_price')}: {P.source('carbon_price')}"),
        (30, "(f) rev_carbon", "", rev_c, "$/yr", "28 x 29"),
        (31, "  - per served MWh", "", rev_c / served, "$/MWhe", "30 / 18"),
        (32, "Community share", "phi", P["community_share"], "-", f"{P.status('community_share')}"),
        (33, "Community energy", "E_comm", P["community_share"] * served, "MWhe/yr", "32 x 18"),
        (34, "Community tariff", "", P["community_tariff"], "$/MWhe", f"{P.status('community_tariff')}: {P.source('community_tariff')}"),
        (35, "rev_comm", "", rev_k, "$/yr", "33 x 34"),
        (36, "  - per served MWh", "", rev_k / served, "$/MWhe", "35 / 18"),
        (37, "rev_BTC / rev_penalty", "", 0.0, "$/yr", "zero in Phase-3 config"),
        (38, "Net numerator", "", cg["numerator"], "$/yr", "21-30-35"),
        (39, "(g-A) BREAKEVEN_UNIFIED  [D-01 conv. A]", "", be_g, "$/MWhe", "38 / 18"),
        (40, "Served_DC", "", served_dc, "MWhe/yr", "18 - 33"),
        (41, "(g-B) BREAKEVEN_UNIFIED  [D-01 conv. B]", "", be_g_b, "$/MWhe", "38 / 40"),
        (42, "(g) literal rev_comm form [D-02]", "", be_g_lit, "$/MWhe", "brief's P_comm x 8760"),
        (43, "Reduction (e) -> (g-A)", "", be_e - be_g, "$/MWhe", "22 - 39"),
        (44, "Diesel reference", "", P["diesel_reference"], "$/MWhe", f"{P.status('diesel_reference')}"),
        (45, "Margin vs diesel at (g-A)", "", P["diesel_reference"] - be_g, "$/MWhe", "44 - 39"),
    ]
    table = pd.DataFrame(rows, columns=["#", "Quantity", "Symbol", "Value", "Unit", "Source"])

    if verbose:
        print("\n" + "=" * 78)
        print("PROMPT-0 WORKED EXAMPLE -- 20 MW IT / PUE 1.5 / 1 x CAREM-25 / r=12% / CF=0.95")
        print("=" * 78)
        for _, row in table.iterrows():
            v = row["Value"]
            vs = "" if (isinstance(v, float) and np.isnan(v)) else (
                f"{v:>16,.4f}" if abs(v) < 1000 else f"{v:>16,.0f}")
            print(f"  {row['#']:>2}  {row['Quantity']:<42} {vs}  {row['Unit']:<12} {row['Source']}")

    # -- assertion against the hand value ------------------------------------
    delta = abs(be_e - HAND_VALUE_NO_REVENUE)
    assert delta < HAND_TOLERANCE, (
        f"(e) breakeven {be_e:.2f} deviates from hand value "
        f"{HAND_VALUE_NO_REVENUE:.2f} by ${delta:.2f} (> ${HAND_TOLERANCE:.2f})"
    )

    if verbose:
        print("\n" + "-" * 78)
        print(f"  HAND-CHECK  (e) computed ${be_e:.2f}/MWhe  vs hand ${HAND_VALUE_NO_REVENUE:.2f}/MWhe"
              f"  |delta| = ${delta:.2f}  (tol ${HAND_TOLERANCE:.2f})   PASS")
        print(f"  REDUCTION   (e) ${be_e:.2f}  ->  (g-A) ${be_g:.2f}/MWhe   "
              f"= -${be_e - be_g:.2f}/MWhe ({(be_e - be_g) / be_e * 100:.1f}%)")
        print(f"  D-01 SPREAD (g-A) ${be_g:.2f}  vs  (g-B) ${be_g_b:.2f}/MWhe   "
              f"= ${be_g_b - be_g:.2f}/MWhe ({(be_g_b - be_g) / be_g * 100:.1f}%)  <-- OPEN DECISION")
        print(f"  D-02 SPREAD (g-A) ${be_g:.2f}  vs  literal ${be_g_lit:.2f}/MWhe  "
              f"= ${be_g_lit - be_g:.2f}/MWhe                <-- OPEN DECISION")
        print("-" * 78)

    return table, dict(be_no_revenue=be_e, be_unified_A=be_g, be_unified_B=be_g_b,
                       be_unified_literal=be_g_lit, served=served, components=cg)


# =============================================================================
# SECTION 8 -- PHASE LOOP
# =============================================================================

@dataclass
class PhaseSpec:
    """One phase of the 4-phase transition. Everything needed for Eq. (U1)."""
    phase: int
    years: str
    source: str
    demand_label: str
    nameplate_mwe: float
    capacity_factor: float
    capex_per_kwe: float
    fopex_per_kwe: float
    vom_fc_per_mwhe: float
    ef_source: float                    # tCO2/MWhe of the generating asset
    life_years: int = 25
    miner_share: float = 0.0            # fraction of served energy to mining
    compute_share: float = 0.0          # fraction of served energy to AI/HPC
    community: bool = False
    flare_capture: bool = False
    gas_share: float = 0.0              # fraction of served energy that is gas-fired
    notes: str = ""
    placeholders: Sequence[str] = field(default_factory=tuple)


def build_phases(P: Params, reactors: pd.DataFrame) -> list:
    """The four phases of the transition model.

    Phase 2 is included for completeness but is NOT part of the brief's
    requested table; it is printed with its revenue set so the progression
    reads continuously.

    SIZING ANCHOR. Phases 1-2 are anchored on the 2 MW operational site in
    southern Nigeria reported in Barrels to Bitcoin-Delta4.pdf -- the only
    real-world capacity figure in the working folder. Phases 3-4 step up to the
    CAREM-25 module (30 MWe, baseiaea.csv) matching the Prompt-0 facility.
    FLAG: the step from 2 MW to 30 MWe is a modelling choice, not a datum.
    """
    rx = reactors.loc["CAREM-25"]
    smr_mwe = float(rx["Power in MWe"])
    smr_capex = float(rx["CAPEX $/kWe"])
    smr_fopex = float(rx["FOPEX $/kWe"])
    smr_vomfc = float(rx["VOM in $/MWh-e"]) + float(rx["FC in $/MWh-e"])
    ef_gen = ef_from_genset(P["genset_efficiency"], P["ef_natural_gas"])

    return [
        PhaseSpec(
            phase=1, years="0-3", source="Flare genset", demand_label="Bitcoin mining",
            nameplate_mwe=P["site_2mw_capacity"],
            capacity_factor=P["genset_capacity_factor"],
            capex_per_kwe=P["genset_capex"], fopex_per_kwe=P["genset_fopex"],
            vom_fc_per_mwhe=P["genset_vom_fc"], ef_source=ef_gen,
            miner_share=1.0, flare_capture=True, gas_share=1.0,
            notes="rev = BTC + penalty + carbon(gas vs diesel)",
            placeholders=("genset_capex", "genset_fopex", "genset_vom_fc",
                          "genset_capacity_factor", "ef_natural_gas", "gas_hhv",
                          "btc_spot_price", "btc_difficulty", "miner_efficiency",
                          "carbon_price"),
        ),
        PhaseSpec(
            phase=2, years="3-5", source="Flare genset + solar PV",
            demand_label="Bitcoin + early AI compute",
            nameplate_mwe=P["site_2mw_capacity"] * 2.0,
            capacity_factor=0.5 * (P["genset_capacity_factor"] + P["re_capacity_factor"]),
            capex_per_kwe=0.5 * (P["genset_capex"] + P["re_capex"]),
            fopex_per_kwe=0.5 * (P["genset_fopex"] + P["re_fopex"]),
            vom_fc_per_mwhe=0.5 * P["genset_vom_fc"], ef_source=0.5 * ef_gen,
            miner_share=0.6, compute_share=0.4, flare_capture=True, gas_share=0.5,
            notes="rev = BTC + penalty + carbon + early compute",
            placeholders=("genset_capex", "re_capex", "re_capacity_factor",
                          "compute_price", "btc_spot_price", "btc_difficulty",
                          "miner_efficiency", "carbon_price"),
        ),
        PhaseSpec(
            phase=3, years="5-10", source="SMR (CAREM-25)", demand_label="AI/HPC data centre",
            nameplate_mwe=smr_mwe, capacity_factor=P["capacity_factor"],
            capex_per_kwe=smr_capex, fopex_per_kwe=smr_fopex,
            vom_fc_per_mwhe=smr_vomfc, ef_source=P["ef_nuclear"],
            community=True,
            notes="rev = carbon(nuclear) + community",
            placeholders=("capacity_factor", "carbon_price", "community_tariff"),
        ),
        PhaseSpec(
            phase=4, years="10+", source="SMR baseload + RE peaking",
            demand_label="Full DC + community",
            nameplate_mwe=smr_mwe * 1.2, capacity_factor=P["capacity_factor"],
            capex_per_kwe=(smr_capex * smr_mwe + P["re_capex"] * smr_mwe * 0.2) / (smr_mwe * 1.2),
            fopex_per_kwe=(smr_fopex * smr_mwe + P["re_fopex"] * smr_mwe * 0.2) / (smr_mwe * 1.2),
            vom_fc_per_mwhe=smr_vomfc, ef_source=P["ef_nuclear"] * (1 / 1.2),
            compute_share=0.85, community=True,
            notes="rev = carbon + community + DC compute",
            placeholders=("capacity_factor", "carbon_price", "community_tariff",
                          "compute_price", "re_capex"),
        ),
    ]


def evaluate_phase(spec: PhaseSpec, P: Params) -> dict:
    """Apply Eq. (U1) to one phase with its own energy source and revenue set."""
    served = served_energy(spec.nameplate_mwe, spec.capacity_factor)
    oc = overnight_capex(spec.capex_per_kwe, spec.nameplate_mwe)

    # -- rev_BTC : Lal Eq. 11 ------------------------------------------------
    rev_btc = 0.0
    if spec.miner_share > 0:
        miner_mw = spec.nameplate_mwe * spec.capacity_factor * spec.miner_share
        H = hashrate_from_power(miner_mw, P["miner_efficiency"])
        rev_btc = btc_revenue(P["btc_spot_price"], P["btc_block_reward"], H,
                              SECONDS_PER_YEAR, P["btc_difficulty"])

    # -- rev_penalty : convention #5 ----------------------------------------
    rev_penalty, volume = 0.0, 0.0
    if spec.flare_capture and spec.gas_share > 0:
        # Only the GAS-FIRED share of served energy consumes associated gas, so
        # only that share earns the avoided flare penalty. Phase 1 is all gas;
        # Phase 2 is a 50/50 genset/PV blend (spec.gas_share).
        gas_elec = served * spec.gas_share
        volume = flare_volume_mscf(gas_elec, P["genset_efficiency"], P["gas_hhv"])
        rev_penalty = penalty_revenue(volume, P["flare_penalty"])

    # -- rev_carbon : counterfactual is unabated diesel everywhere ----------
    rev_carbon = carbon_revenue(served, P["ef_diesel"], spec.ef_source, P["carbon_price"])

    # -- rev_comm : convention #8 -------------------------------------------
    rev_comm = community_revenue(served, P["community_share"], P["community_tariff"]) \
        if spec.community else 0.0

    # -- rev_compute : convention #4 ----------------------------------------
    rev_compute = compute_revenue(served, P["pue"], P["compute_price"], spec.compute_share) \
        if spec.compute_share > 0 else 0.0

    be, comp = unified_breakeven(
        oc, 1.0, P["discount_rate"], spec.life_years,
        spec.fopex_per_kwe, spec.nameplate_mwe, spec.vom_fc_per_mwhe, served,
        rev_btc=rev_btc, rev_carbon=rev_carbon, rev_penalty=rev_penalty,
        rev_comm=rev_comm, rev_compute=rev_compute,
    )
    be_gross = comp["total_cost"] / served

    return dict(
        phase=spec.phase, years=spec.years, source=spec.source,
        demand=spec.demand_label, mwe=spec.nameplate_mwe, cf=spec.capacity_factor,
        served=served, anncap=comp["anncap"], fixed=comp["fixed"], vom=comp["vom"],
        cost_total=comp["total_cost"], rev_btc=rev_btc, rev_carbon=rev_carbon,
        rev_penalty=rev_penalty, rev_comm=rev_comm, rev_compute=rev_compute,
        rev_total=comp["total_revenue"], flare_mscf=volume,
        be_gross=be_gross, be_unified=be,
        # THREE-WAY STATUS, not a binary. A NEGATIVE breakeven is not merely
        # "viable": it means the non-electricity revenue streams alone more than
        # cover annualised cost, so the project clears with ZERO diesel
        # displacement. That is the bridge-finance claim of Paper 2 in its
        # strongest form and must be reported as its own category, because
        # collapsing it into "viable" hides the mechanism the paper is about.
        status=("REVENUE-SUFFICIENT" if be < 0 else
                ("VIABLE" if be < P["diesel_reference"] else "NOT VIABLE")),
        viable=be < P["diesel_reference"],
        n_placeholders=len(spec.placeholders), notes=spec.notes,
    )


def phase_table(P: Optional[Params] = None, reactors: Optional[pd.DataFrame] = None,
                verbose: bool = True) -> pd.DataFrame:
    """Per-phase unified breakeven across the 4-phase transition."""
    P = P or Params.load()
    reactors = reactors if reactors is not None else load_reactors()
    rows = [evaluate_phase(s, P) for s in build_phases(P, reactors)]
    df = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 78)
        print("PHASE LOOP -- unified breakeven by transition phase")
        print("=" * 78)
        print("!! Phases 1, 2 and 4 rest largely on PLACEHOLDER parameters (Day-2 items).")
        print("!! Treat their breakevens as STRUCTURAL DEMONSTRATIONS, not results.\n")
        hdr = (f"  {'Ph':<3}{'Years':<7}{'Source':<26}{'MWe':>7}{'Served':>11}"
               f"{'Cost $/MWhe':>13}{'Rev $/MWhe':>12}{'BREAKEVEN':>12}{'STATUS':>20}{'PH':>4}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for _, x in df.iterrows():
            print(f"  {x['phase']:<3}{x['years']:<7}{x['source'][:25]:<26}{x['mwe']:>7.1f}"
                  f"{x['served']:>11,.0f}{x['be_gross']:>13,.2f}"
                  f"{x['rev_total'] / x['served']:>12,.2f}{x['be_unified']:>12,.2f}"
                  f"{x['status']:>20}{x['n_placeholders']:>4}")
        print("\n  Revenue decomposition ($/MWhe of served energy):")
        sub = (f"  {'Ph':<3}{'BTC':>11}{'carbon':>11}{'penalty':>11}"
               f"{'community':>11}{'compute':>11}{'TOTAL':>11}")
        print(sub)
        print("  " + "-" * (len(sub) - 2))
        for _, x in df.iterrows():
            s = x["served"]
            print(f"  {x['phase']:<3}{x['rev_btc'] / s:>11,.2f}{x['rev_carbon'] / s:>11,.2f}"
                  f"{x['rev_penalty'] / s:>11,.2f}{x['rev_comm'] / s:>11,.2f}"
                  f"{x['rev_compute'] / s:>11,.2f}{x['rev_total'] / s:>11,.2f}")
        if (df["be_unified"] < 0).any():
            print("\n  NOTE ON NEGATIVE BREAKEVENS. A breakeven below zero means the")
            print("  non-electricity revenue streams alone exceed annualised cost: the")
            print("  phase clears with NO diesel displacement at all. Economically this is")
            print("  the bridge-finance result. It is ALSO the point at which the closed")
            print("  form stops being the binding constraint -- a negative breakeven says")
            print("  nothing about whether the mining load can absorb the generation hour")
            print("  by hour, which only the Day-4/5 Pyomo dispatch can test. Report these")
            print("  as an upper bound on attractiveness, never as a standalone result.\n")
        print("\n  Revenue set by phase:")
        for s in build_phases(P, reactors):
            print(f"    Phase {s.phase} ({s.years:>4}): {s.notes}")
    return df


# =============================================================================
# SECTION 9 -- ENTRY POINT
# =============================================================================

def provenance_report(P: Params, verbose: bool = True) -> pd.DataFrame:
    """Hard convention #13: every untraceable assumption must be flagged."""
    ph = P.placeholders()
    if verbose:
        print("\n" + "=" * 78)
        print(f"PROVENANCE -- parameters loaded from: {P.origin}")
        print("=" * 78)
        counts = P.table["status"].astype(str).str.upper().value_counts()
        for k, v in counts.items():
            print(f"  {k:<14} {v:>3}")
        print(f"\n  {len(ph)} PLACEHOLDER parameters are NOT traceable to any working-folder file")
        print("  (hard convention #13). Each is a Day-2 collection item:\n")
        for key, row in ph.iterrows():
            print(f"    {key:<26} {row['value']:>14}  {str(row['unit']):<14} {row['source']}")
    return ph


def main() -> int:
    print("\n" + "#" * 78)
    print("#  analytical_benchmark.py -- Paper 2 ground-truth engine")
    print("#  PURE CLOSED FORM. NO SOLVER. Every figure hand-reproducible.")
    print("#" * 78)

    P = Params.load()
    reactors = load_reactors()

    provenance_report(P)
    test_regression_to_paper1()
    worked_example(P, reactors)
    phase_table(P, reactors)

    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print("  V1 regression gate ......... PASS  (Eq. U1 collapses to Eq. D24 exactly)")
    print("  Prompt-0 hand-check ........ PASS  (within $5 of $143.84/MWhe)")
    print("  Defect B-01 ................ reproduced and quantified (2E/S)")
    print("  Open decisions ............. D-01 denominator, D-02 rev_comm form")
    print("  Next ....................... Day 2: replace every PLACEHOLDER above")
    print("=" * 78 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
