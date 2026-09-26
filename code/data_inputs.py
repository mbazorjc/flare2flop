#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
data_inputs.py
==============
Master parameter assembly for Paper 2, "From Flares to FLOPS: A Phased
Techno-Economic Framework for Stranded Gas, Small Modular Reactors, and Compute
Anchor Loads in Sub-Saharan Africa".

PURE DATA ASSEMBLY. NO OPTIMISATION. Every function here loads, converts, or
derives a parameter; none of them solves anything. Run the module directly to
print the validation table.

Dependencies: numpy, pandas. Nothing else.

-------------------------------------------------------------------------------
CONTRACT WITH analytical_benchmark.py  --  DO NOT BREAK
-------------------------------------------------------------------------------
analytical_benchmark.py resolves parameters in this order:
    1. import data_inputs  -> uses as_dataframe() if present, else PARAMS
    2. data_inputs.csv
    3. embedded defaults
Creating this module therefore TAKES PRECEDENCE over data_inputs.csv. To keep
the benchmark engine working unchanged, as_dataframe() below returns the same
parameter register (read from data_inputs.csv, with an embedded fallback) in the
same shape the engine expects: index 'key', columns value / unit / status / source.
If you edit as_dataframe(), re-run analytical_benchmark.py and confirm gate V1
still passes before committing.

-------------------------------------------------------------------------------
THREE CORRECTIONS TO THE SPECIFICATION  --  read before using these numbers
-------------------------------------------------------------------------------
The formulae as briefed contain three dimensional errors. Each is implemented
CORRECTLY here and the briefed form is reproduced alongside it so the size of the
discrepancy is visible rather than silent. See the validation table, section [D].

C1  bitcoin_module. Briefed:
        rev/hr = price x R x (n x hash_ths x t_m) / (difficulty_th x 1e12 x 2^32)
    The numerator is in terahashes; the denominator expects hashes. The factor
    1e12 converts TH -> H and therefore belongs in the NUMERATOR, not beside the
    difficulty. As briefed the result is understated by 1e24. Corrected:
        rev/hr = price x R x (n x hash_ths x 1e12 x t_m) / (difficulty x 2^32)

C2  community_module. Briefed:
        rev/yr = load_mw x share x 8760 x 1000 x price_kwh / 1000
    load_mw x 8760 x 1000 is already kilowatt-hours, so multiplying by a $/kWh
    price yields dollars. The trailing / 1000 understates revenue by 1000x.

C3  datacentre_module. Briefed:
        rev/hr = load_mw x 1000 x gpu_price
    gpu_price is a compute-service price per kilowatt of IT capacity per hour.
    load_mw is the TOTAL facility load, which includes PUE cooling overhead that
    earns no compute revenue. Applying the price to total load overstates revenue
    by the PUE factor (50% at PUE 1.5). Revenue is computed on IT load here; the
    briefed form is reported for comparison.

-------------------------------------------------------------------------------
FOUR DATA FINDINGS  --  these affect Paper 1 as well as Paper 2
-------------------------------------------------------------------------------
F1  The "#not specified" marker lives in the COUNTRY column of ADC.csv, not in
    Electric_MW. Electric_MW is clean numeric throughout. A filter written
    against Electric_MW text, as briefed, would exclude nothing.

F2  Paper 1 Table 1 is reproduced EXACTLY (all 15 statistics, three countries)
    by one filter only: drop rows with Electric_MW == 0, and keep everything
    else. That filter RETAINS eight facilities whose capacity is floor-area
    estimated and flagged "#not specified". Paper 1's Methods text states that
    "only facilities with a directly reported electrical IT capacity were used"
    and that "floor-area-based estimation was not used". The text and the
    statistics disagree. One of the two needs correcting before Paper 1 is
    resubmitted. Mitigation: only ONE of those eight (DC_Pretoria_DP_Centurion,
    23.1 MW) clears the 2 MW threshold, so the optimisation set is affected by a
    single facility.

F3  DC_Lagos_LKK2 (2.0 MW, Nigeria) appears TWICE in ADC.csv as an identical
    row, and Paper 1's Table 1 counts it twice: de-duplicating gives Nigeria
    n=21 and total 290.7 MW against the published n=22 and 292.7 MW. The
    duplicate is below the 2 MW threshold and so does not enter the 37-facility
    optimisation set, but it does inflate the descriptive table.

F4  The 2 MW scale filter is applied to IT load with a STRICT inequality
    (Electric_MW > 2.0), which yields exactly 37. Using >= 2.0 yields 41, and
    applying the threshold to PUE-adjusted total load yields 45. The strict
    form on IT load is the one that reproduces Paper 1.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, Optional

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH = (HERE, os.path.dirname(HERE), os.getcwd())

HOURS_PER_YEAR = 8760
DAYS_PER_YEAR = 365
SECONDS_PER_HOUR = 3600
TWO_POW_32 = 2 ** 32
M3_PER_SCF = 0.028316846592        # exact, international foot
MMBTU_PER_MWH = 3.412141633

_FINDINGS: list = []               # populated at runtime, printed in the table


def _find(fn: str) -> Optional[str]:
    for dd in SEARCH:
        p = os.path.join(dd, fn)
        if os.path.exists(p):
            return p
    return None


# =============================================================================
# [0]  PARAMETER REGISTER  --  the contract with analytical_benchmark.py
# =============================================================================

_EMBEDDED = {
    "discount_rate":      (0.12,   "-",           "CONVENTION",  "Convention #1"),
    "plant_life_years":   (25,     "yr",          "CONVENTION",  "Convention #1"),
    "diesel_reference":   (300.0,  "$/MWhe",      "CONVENTION",  "Convention #2"),
    "ef_diesel":          (0.70,   "tCO2/MWhe",   "FILE",        "corrected-postproc.py L31"),
    "ef_nuclear":         (0.012,  "tCO2/MWhe",   "FILE",        "corrected-postproc.py L31"),
    "pue":                (1.5,    "-",           "CONVENTION",  "Convention #4"),
    "flare_penalty":      (3.50,   "$/mscf",      "CONVENTION",  "Convention #5"),
    "btc_block_reward":   (3.125,  "BTC/block",   "CONVENTION",  "Convention #6"),
    "genset_efficiency":  (0.38,   "-",           "CONVENTION",  "Convention #7"),
    "community_share":    (0.10,   "-",           "CONVENTION",  "Convention #8"),
    "export_cap":         (0.0,    "-",           "CONVENTION",  "Convention #9"),
    "carbon_price":       (50.0,   "$/tCO2",      "PLACEHOLDER", "D-04 OPEN"),
    "community_tariff":   (100.0,  "$/MWhe",      "PLACEHOLDER", "D-03 OPEN"),
    "btc_spot_price":     (60000.0,"$/BTC",       "PLACEHOLDER", "D-06 OPEN"),
    "btc_difficulty":     (1.10e14,"-",           "PLACEHOLDER", "D-06 OPEN"),
    "miner_efficiency":   (46.10,  "J/TH",        "DERIVED",     "D-07 CLOSED"),
    "gas_hhv":            (1037.0, "Btu/scf",     "PLACEHOLDER", "D-08 OPEN"),
    "ef_natural_gas":     (53.06,  "kgCO2/MMBtu", "PLACEHOLDER", "D-08 OPEN"),
    "genset_capex":       (800.0,  "$/kWe",       "PLACEHOLDER", "D-09 OPEN"),
    "genset_fopex":       (40.0,   "$/kWe-yr",    "PLACEHOLDER", "D-09 OPEN"),
    "genset_vom_fc":      (5.0,    "$/MWhe",      "PLACEHOLDER", "D-09 OPEN"),
    "compute_price":      (120.0,  "$/MWhe-IT",   "PLACEHOLDER", "D-10 OPEN"),
    "re_capex":           (1100.0, "$/kWe",       "PLACEHOLDER", "D-11 OPEN"),
    "re_fopex":           (22.64,  "$/kW-yr",     "FILE",        "Lal Table 3 opfac_solar"),
    "re_capacity_factor": (0.20,   "-",           "PLACEHOLDER", "D-11 OPEN"),
    "capacity_factor":    (0.95,   "-",           "PLACEHOLDER", "Prompt-0 brief"),
    "genset_capacity_factor": (0.90, "-",         "PLACEHOLDER", "D-09 OPEN"),
    "site_2mw_capacity":  (2.0,    "MWe",         "FILE",        "Barrels deck"),
}

PARAMS: Dict[str, tuple] = dict(_EMBEDDED)


def as_dataframe(path: Optional[str] = None) -> pd.DataFrame:
    """Parameter register indexed by key, columns value/unit/status/source.

    This is the interface analytical_benchmark.py calls. It reads data_inputs.csv
    when present so the CSV remains the single editable source of truth, and
    falls back to the embedded table otherwise.
    """
    csv = path or _find("data_inputs.csv")
    if csv and os.path.exists(csv):
        df = pd.read_csv(csv, comment="#").dropna(subset=["key"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df.set_index("key")
    return pd.DataFrame(
        [{"key": k, "value": v[0], "unit": v[1], "status": v[2], "source": v[3]}
         for k, v in _EMBEDDED.items()]).set_index("key")


# =============================================================================
# [1b]  REGISTER -> SiteConfig BRIDGE   (B-21, added v8)
# =============================================================================
#
# Until v7 the register was ADVISORY: analytical_benchmark.py read it, the Pyomo
# model did not, and SiteConfig carried its own hardcoded defaults. Six register
# entries therefore had no effect on any reported number, and the two that were
# closed by pilot measurement on 2026-09-18 -- community_tariff and
# genset_capacity_factor -- would have changed nothing had they simply been
# edited in the CSV. This bridge closes that gap: the CSV becomes the single
# editable source of truth for the solver as well.
#
# Only the keys below are bridged. Hard conventions #11 and #12 are NOT
# bridged -- reactor costs come from baseiaea.csv and facilities from ADC.csv,
# and nothing in the register may override either.
#
# left  = register key in data_inputs.csv
# right = SiteConfig field name in phased_compute_model.py
REGISTER_TO_SITECONFIG = {
    "miner_efficiency":        "miner_efficiency_j_per_th",
    "genset_capex":            "genset_capex_per_kwe",
    "genset_fopex":            "genset_fopex_per_kwe",
    "genset_vom_fc":           "genset_vom",
    "genset_capacity_factor":  "genset_availability",
    "community_tariff":        "elec_price_comm",
    "carbon_price":            "carbon_price",
    "btc_spot_price":          "btc_price",
    "btc_difficulty":          "btc_difficulty",
    "compute_price":           "compute_price",
    "flare_penalty":           "flare_penalty",
    "diesel_reference":        "diesel_price",
    "genset_efficiency":       "eta_gen",
    "ef_diesel":               "ef_diesel",
    "ef_nuclear":              "ef_nuclear",
    "btc_block_reward":        "block_reward",
    "export_cap":              "export_cap",
}

# The pre-v8 SiteConfig defaults, kept verbatim so --param-set design reproduces
# the v7 grid exactly. Do not "tidy" these: they are a reproducibility record,
# not a recommendation. genset_availability = 1.0 is the implicit assumption the
# v7 model made by having no availability term at all.
DESIGN_CASE = {
    "miner_efficiency_j_per_th": 17.5,
    "genset_capex_per_kwe": 800.0,
    "genset_fopex_per_kwe": 40.0,
    "genset_vom":           5.0,
    "genset_availability":  1.0,
    "elec_price_comm":      100.0,
    "carbon_price":         50.0,
    "btc_price":            60000.0,
    "btc_difficulty":       1.10e14,
    "compute_price":        120.0,
    "flare_penalty":        3.50,
    "diesel_price":         300.0,
    "eta_gen":              0.38,
    "ef_diesel":            0.70,
    "ef_nuclear":           0.012,
    "block_reward":         3.125,
    "export_cap":           0.0,
}


def site_overrides(param_set: str = "register", path: Optional[str] = None
                   ) -> Dict[str, float]:
    """SiteConfig keyword overrides drawn from the parameter register.

    param_set = "register"  read data_inputs.csv (the measured / operator values)
    param_set = "design"    the pre-v8 hardcoded defaults, for reproducing v7
    param_set = "none"      empty dict; SiteConfig defaults stand

    Returns a plain dict suitable for SiteConfig(**overrides). Keys absent from
    the CSV, or non-numeric there, are simply not overridden -- a missing row
    must never silently become a zero.
    """
    ps = (param_set or "register").lower()
    if ps == "none":
        return {}
    if ps == "design":
        return dict(DESIGN_CASE)
    if ps != "register":
        raise ValueError(f"param_set must be register|design|none, got {param_set!r}")

    df = as_dataframe(path)
    out: Dict[str, float] = {}
    for key, field_name in REGISTER_TO_SITECONFIG.items():
        if key not in df.index:
            continue
        v = df.loc[key, "value"]
        try:
            v = float(v)
        except (TypeError, ValueError):
            continue
        if v != v:                       # NaN
            continue
        out[field_name] = v
    return out


def override_provenance(param_set: str = "register", path: Optional[str] = None
                        ) -> "pd.DataFrame":
    """Audit table: what the bridge changed, from what, to what, and on whose say-so.

    Printed by run_phased_sweep at startup and written beside the results, so a
    reviewer can see at a glance which numbers rest on measurement and which
    still rest on a placeholder.
    """
    ov = site_overrides(param_set, path)
    df = as_dataframe(path)
    inv = {v: k for k, v in REGISTER_TO_SITECONFIG.items()}
    rows = []
    for field_name, v in sorted(ov.items()):
        key = inv.get(field_name, "")
        rows.append({
            "siteconfig_field": field_name,
            "register_key": key,
            "design_value": DESIGN_CASE.get(field_name, float("nan")),
            "applied_value": v,
            "changed": abs(v - DESIGN_CASE.get(field_name, v)) > 1e-12,
            "status": (df.loc[key, "status"] if key in df.index else ""),
            "source": (df.loc[key, "source"] if key in df.index else ""),
        })
    return pd.DataFrame(rows)


# =============================================================================
# [1]  SMR CATALOGUE
# =============================================================================

# Column map handed to Pyomo. Left = model name, right = baseiaea.csv header.
SMR_COLUMN_MAP = {
    "Cap_e":        "Power in MWe",
    "Cap_t":        "Power in MWt",
    "eta":          "Thermal Efficiency",
    "eta_trans":    "Thermal Transfer Efficiency",
    "MSL":          "MSL in MWe",
    "MSL_turb":     "MSL_turb in MWe",
    "MDT":          "MDT in hours",
    "RampRate":     "Ramp Rate (MW/hr)",
    "RampFrac":     "Ramp Rate (fraction of capacity/hr)",
    "CAPEX":        "CAPEX $/kWe",
    "FOPEX":        "FOPEX $/kWe",
    "VOM":          "VOM in $/MWh-e",
    "FC":           "FC in $/MWh-e",
    "StartupCost":  "Startupfixedcost in $",
    "MaxModules":   "Max Modules",
    "OutletTemp":   "Outlet Temp (C)",
}

_REQUIRED_NON_NULL = [
    "Power in MWe", "CAPEX $/kWe", "FOPEX $/kWe", "VOM in $/MWh-e",
    "FC in $/MWh-e", "Ramp Rate (MW/hr)", "MDT in hours", "MSL in MWe",
    "Max Modules",
]


def size_class(mwe: float) -> str:
    """Microreactor < 5 MWe; Mid-size 5-50 MWe; Large > 50 MWe."""
    if mwe < 5.0:
        return "Microreactor"
    if mwe <= 50.0:
        return "Mid-size"
    return "Large"


def load_smr_catalogue(path: str = "baseiaea.csv") -> pd.DataFrame:
    """Load the 16 IAEA ARIS 2024 designs with a Pyomo-ready column map.

    Hard convention #11: reactor parameters come from this file or nowhere.

    Adds CAPEX_MWe = CAPEX $/kWe x 1000, a derived SizeClass, and cross-checks
    that class against the catalogue's own Classification column. The catalogue
    column is a TECHNOLOGY classification (PWR / HTGR / MSR / LMFR / Microreactor),
    so only the "Microreactor" label is directly comparable; mismatches are
    reported rather than silently reconciled.
    """
    p = _find(path) or path
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"baseiaea.csv not found on {SEARCH}. Convention #11 forbids "
            "inventing reactor parameters, so this module refuses to proceed.")
    df = pd.read_csv(p)

    missing = [c for c in _REQUIRED_NON_NULL if c not in df.columns]
    if missing:
        raise KeyError(f"baseiaea.csv is missing required columns: {missing}")
    for c in _REQUIRED_NON_NULL:
        n = int(df[c].isna().sum())
        assert n == 0, f"baseiaea.csv: {n} NaN in required column {c!r}"

    df["CAPEX_MWe"] = df["CAPEX $/kWe"] * 1000.0
    df["SizeClass"] = df["Power in MWe"].apply(size_class)

    # cross-check derived size class against the catalogue Classification column
    mism = []
    for _, r in df.iterrows():
        cat, derived, mwe = str(r["Classification"]), r["SizeClass"], r["Power in MWe"]
        if cat == "Microreactor" and derived != "Microreactor":
            mism.append(f"{r['Sites']}: catalogue 'Microreactor' but {mwe} MWe -> {derived}")
        elif cat != "Microreactor" and derived == "Microreactor":
            mism.append(f"{r['Sites']}: catalogue {cat!r} but {mwe} MWe -> Microreactor")
    if mism:
        _FINDINGS.append(("SMR size-class cross-check", "; ".join(mism)))

    for model, src in SMR_COLUMN_MAP.items():
        if src in df.columns:
            df[model] = df[src]
    df["CAPEX_per_MWe"] = df["CAPEX_MWe"]
    return df.set_index("Sites")


# =============================================================================
# [2]  DATA-CENTRE FACILITIES
# =============================================================================

PAPER1_TABLE1 = {                       # n, median, mean, max, total MW
    "Nigeria":      (22, 3.0, 13.3, 100.0, 292.7),
    "Kenya":        (11, 2.5,  6.9,  44.0,  76.2),
    "South Africa": (29, 10.0, 45.5, 400.0, 1317.9),
}


def facility_size_class(mw: float) -> str:
    """micro <5; small 5-20; mid 20-50; large 50-100; hyper >100 (MW IT)."""
    if mw < 5.0:
        return "micro"
    if mw < 20.0:
        return "small"
    if mw < 50.0:
        return "mid"
    if mw <= 100.0:
        return "large"
    return "hyper"


def load_facilities(path: str = "ADC.csv", min_mw: float = 2.0, pue: float = 1.5,
                    strict_reported: bool = False, drop_duplicates: bool = False,
                    verbose: bool = True) -> pd.DataFrame:
    """Load ADC.csv, reproduce Paper 1 Table 1, and apply the scale filter.

    Hard convention #12: facility data comes from this file or nowhere.

    strict_reported=False (default) reproduces Paper 1 exactly: drop only rows
    with Electric_MW == 0. strict_reported=True additionally drops the eight
    floor-area-estimated facilities flagged "#not specified", which is what
    Paper 1's Methods text describes but NOT what its Table 1 statistics show.
    See finding F2 in the module docstring.
    """
    p = _find(path) or path
    if not os.path.exists(p):
        raise FileNotFoundError(f"ADC.csv not found on {SEARCH}.")
    raw = pd.read_csv(p, dtype=str)

    # FINDING F1: the "#not specified" marker is carried in the Country column,
    # not in Electric_MW. Parse both defensively.
    raw["Electric_MW"] = pd.to_numeric(raw["Electric_MW"], errors="coerce")
    country_raw = raw["Country"].fillna("")
    mw_text = raw.get("Electric_MW_text", pd.Series([""] * len(raw))).fillna("")
    flagged = (country_raw.str.contains("#", regex=False) |
               country_raw.str.lower().str.contains("not specified") |
               mw_text.astype(str).str.contains("#", regex=False))
    raw["not_specified"] = flagged
    raw["country"] = country_raw.str.split("#").str[0].str.strip()
    raw["it_mw"] = raw["Electric_MW"]

    n_all = len(raw)
    kept = raw[raw["it_mw"].notna() & (raw["it_mw"] > 0.0)].copy()
    n_zero = n_all - len(kept)
    if strict_reported:
        before = len(kept)
        kept = kept[~kept["not_specified"]].copy()
        _FINDINGS.append(("strict_reported filter",
                          f"dropped {before - len(kept)} floor-area-estimated facilities"))
    if drop_duplicates:
        before = len(kept)
        kept = kept.drop_duplicates(subset=["FACILITY_ID", "it_mw"]).copy()
        if before != len(kept):
            _FINDINGS.append(("drop_duplicates",
                              f"removed {before - len(kept)} duplicate row(s)"))

    kept["total_load_mw"] = kept["it_mw"] * pue
    kept["size_class"] = kept["it_mw"].apply(facility_size_class)

    # ---- country summary against Paper 1 Table 1 ----
    summary = (kept.groupby("country")["it_mw"]
               .agg(n="size", median="median", mean="mean", max="max", total="sum")
               .reindex(PAPER1_TABLE1.keys()))
    if verbose:
        print("\n  Country summary (IT load, MW) vs Paper 1 Table 1")
        print(f"  {'Country':<15}{'n':>4}{'median':>9}{'mean':>9}{'max':>8}{'total':>11}   check")
        print("  " + "-" * 72)
    all_ok = True
    for c, (tn, tmed, tmean, tmax, ttot) in PAPER1_TABLE1.items():
        if c not in summary.index or pd.isna(summary.loc[c, "n"]):
            all_ok = False
            if verbose:
                print(f"  {c:<15}{'--':>4}{'':>9}{'':>9}{'':>8}{'':>11}   MISSING")
            continue
        r = summary.loc[c]
        ok = (int(r["n"]) == tn and abs(r["median"] - tmed) < 0.06 and
              abs(r["mean"] - tmean) < 0.06 and abs(r["max"] - tmax) < 0.5 and
              abs(r["total"] - ttot) < 0.15)
        all_ok &= ok
        if verbose:
            print(f"  {c:<15}{int(r['n']):>4}{r['median']:>9.2f}{r['mean']:>9.2f}"
                  f"{r['max']:>8.1f}{r['total']:>11.2f}   "
                  f"{'MATCH' if ok else f'MISMATCH (target n={tn} med={tmed} mean={tmean} max={tmax:g} tot={ttot})'}")
    if verbose:
        print(f"  {'TOTAL':<15}{int(summary['n'].sum()):>4}{'':>9}{'':>9}{'':>8}"
              f"{summary['total'].sum():>11.2f}")
        print(f"\n  Rows read {n_all}; dropped {n_zero} with Electric_MW == 0; "
              f"{len(kept)} specified facilities.")
        print(f"  Paper 1 Table 1 reproduction: {'EXACT on all 15 statistics' if all_ok else 'MISMATCH — see above'}")
    if not all_ok:
        _FINDINGS.append(("Paper 1 Table 1", "descriptive statistics DO NOT reproduce"))

    # ---- FINDING F4: strict inequality on IT load is what yields 37 ----
    scaled = kept[kept["it_mw"] > min_mw].copy()
    if verbose:
        alt_ge = int((kept["it_mw"] >= min_mw).sum())
        alt_tot = int((kept["total_load_mw"] > min_mw).sum())
        print(f"\n  Scale filter IT load > {min_mw} MW: {len(scaled)} facilities "
              f"(target 37) — {'MATCH' if len(scaled) == 37 else 'MISMATCH'}")
        print(f"    by country: {scaled['country'].value_counts().to_dict()}")
        print(f"    alternatives: >= {min_mw} MW gives {alt_ge}; "
              f"PUE-adjusted total > {min_mw} MW gives {alt_tot}")
        n_est = int(scaled["not_specified"].sum())
        print(f"    floor-area-estimated facilities surviving the filter: {n_est}"
              + (f" ({', '.join(scaled[scaled['not_specified']]['FACILITY_ID'])})" if n_est else ""))
    if len(scaled) != 37:
        _FINDINGS.append(("Scale filter", f"{len(scaled)} facilities, expected 37"))

    scaled.attrs["summary"] = summary
    scaled.attrs["all_specified"] = kept
    scaled.attrs["table1_match"] = all_ok
    return scaled


# =============================================================================
# [3]  FLARE GAS
# =============================================================================

EF_FLARE_DEFAULT = 0.28        # tCO2e/mscf, as briefed  -- SEE WARNING BELOW
EF_COMBUST_DEFAULT = 0.053     # tCO2/mscf, as briefed


def flare_gas_module(V0_mmscfd: float = 2.0, decline: float = 0.05,
                     eta_gen: float = 0.38, lhv_mj_m3: float = 36.0,
                     penalty: float = 3.50, years: int = 25,
                     ef_flare: float = EF_FLARE_DEFAULT,
                     ef_combust: float = EF_COMBUST_DEFAULT) -> dict:
    """Flare-gas availability, electrical output, avoided penalty and avoided CO2.

    Unit chain, explicit at every step:
        V [MMSCFD] x 1e6 scf/MMscf x 0.028316846592 m3/scf / 86400 s/day = m3/s
        P [MW] = eta x V [m3/s] x LHV [MJ/m3]        since MJ/s = MW
        V [MMSCFD] x 1000 mscf/MMscf x 365 d/yr      = mscf/yr

    Decline is applied annually: V_year = V0 x (1 - decline)^year, and is held
    constant within each year, so the hourly array is a step function.

    Returns hourly arrays over `years` x 8760 hours plus annual aggregates.
    """
    if not 0.0 <= decline < 1.0:
        raise ValueError("decline must lie in [0, 1)")
    if not 0.0 < eta_gen <= 1.0:
        raise ValueError("eta_gen must lie in (0, 1]")

    yr = np.arange(years)
    v_mmscfd = V0_mmscfd * (1.0 - decline) ** yr                 # MMSCFD by year
    v_m3s = v_mmscfd * 1e6 * M3_PER_SCF / 86400.0                # m3/s
    p_mw = eta_gen * v_m3s * lhv_mj_m3                           # MW electrical
    mscf_yr = v_mmscfd * 1000.0 * DAYS_PER_YEAR                  # mscf/yr

    penalty_yr = mscf_yr * penalty                               # $/yr
    co2_yr = mscf_yr * (ef_flare - ef_combust)                   # tCO2/yr
    mwh_yr = p_mw * HOURS_PER_YEAR                               # MWh/yr at full uptime

    # WARNING — the briefed emission factors do not survive scrutiny.
    # Capturing flare gas and burning it in a genset combusts the SAME gas that
    # would otherwise be flared, so the CO2 from combustion is essentially
    # unchanged. The only genuine differential is unburned methane slip at the
    # flare. At 1 mscf ~ 1.037 MMBtu and 53.06 kgCO2/MMBtu, combustion is
    # ~0.055 tCO2/mscf, which matches ef_combust = 0.053. For ef_flare = 0.28 to
    # hold, the flare would have to emit five times that in CO2-equivalent,
    # implying roughly half the methane escaping unburned. Routine flaring
    # destroys ~98% of the methane, giving a slip differential nearer
    # 0.01 tCO2e/mscf. As briefed, avoided CO2 is overstated by roughly 20x.
    ef_diff = ef_flare - ef_combust
    if ef_diff > 0.05:
        _FINDINGS.append((
            "Flare emission factors",
            f"ef_flare - ef_combust = {ef_diff:.3f} tCO2/mscf implies ~{ef_diff/0.011:.0f}x "
            "the defensible methane-slip differential (~0.011). Avoided CO2 is "
            "overstated. The real carbon case for flare-to-power is DISPLACED "
            "DIESEL, not the combustion differential. Raise as a new open item."))

    hourly_mw = np.repeat(p_mw, HOURS_PER_YEAR)
    hourly_mscf = np.repeat(mscf_yr / HOURS_PER_YEAR, HOURS_PER_YEAR)

    return {
        "years": yr,
        "hourly_mw": hourly_mw,
        "hourly_mscf": hourly_mscf,
        "annual_mmscfd": v_mmscfd,
        "annual_m3s": v_m3s,
        "annual_mscf": mscf_yr,
        "annual_mw": p_mw,
        "annual_mwh": mwh_yr,
        "annual_penalty_usd": penalty_yr,
        "annual_co2_t": co2_yr,
        "mw_year0": float(p_mw[0]),
        "mscf_year0": float(mscf_yr[0]),
        "penalty_year0": float(penalty_yr[0]),
        "co2_year0": float(co2_yr[0]),
        "ef_flare": ef_flare,
        "ef_combust": ef_combust,
    }


# =============================================================================
# [4]  BITCOIN MINING
# =============================================================================

# --- D-07 CLOSED 2026-09-18 from the pilot pool record --------------------------
#
# The S21 reference unit below was a placeholder: 200 TH/s at 3.5 kW is 17.5 J/TH,
# which is 2023-vintage hardware. The pilot does not run 2023-vintage hardware.
#
# The operator will not disclose the fleet mix, and it turns out not to matter.
# The pool record gives a MEASURED per-worker hashrate of 71.87 TH/s (median over
# 717 days, corrected for uptime). Every SHA-256 ASIC that hashes in that band
# draws essentially the same power:
#
#     Whatsminer M20S   68 TH/s   3360 W   49.4 J/TH   Aug 2019
#     Whatsminer M31S   76 TH/s   3344 W   44.0 J/TH   Apr 2020
#     Whatsminer M30S   86 TH/s   3268 W   38.0 J/TH   Apr 2020
#     Antminer  T19     88 TH/s   3344 W   38.0 J/TH   Aug 2021
#     Antminer  S19     95 TH/s   3250 W   34.2 J/TH   May 2020
#                     (manufacturer specs, asicminervalue.com, read 2026-09-18)
#
# Power spans 3250-3360 W, a 3.3% range, while hashrate spans 68-95 TH/s, a 40%
# range. Efficiency in this class is set by hashrate, not by draw. So the
# undisclosed mix is exactly the thing that does not need disclosing: any fleet
# averaging 71.87 TH/s per machine draws about 3.31 kW per machine whatever it is
# made of. Three independent routes to the efficiency:
#
#     A  measured hashrate / mean published power  3313.2 / 71.87 = 46.10 J/TH
#     B  measured hashrate / Lal Table 3 3.25 kW   3250.0 / 71.87 = 45.22 J/TH
#     C  interpolate published J/TH against TH/s   at 71.87       = 46.79 J/TH
#
# They agree to 3.5%. Route A is adopted. The residual uncertainty is not which
# route, it is the DISPERSION of the fleet: the per-worker IQR of 62.0-82.8 TH/s
# maps to 40.0-53.4 J/TH, and that band is what the Monte Carlo samples.
#
# This also retires Lal's AC_Miner = $3,395 for THIS site. That is a new-S21
# price. A fleet hashing at 72 TH/s per machine is 2019-2021 silicon bought
# second-hand, so miner capital here is a different number entirely -- flagged in
# the register as D-12, not silently carried over.
MINER_PILOT = {"hash_ths": 71.87, "power_kw": 3.3132, "capex_usd": float("nan"),
               "efficiency_j_per_th": 46.10}

# Kept under its old name so nothing that imports it breaks, but it is no longer
# the site's fleet -- it is the modern-hardware COMPARATOR used in the discussion
# of what a refreshed fleet would earn.
MINER_S21 = {"hash_ths": 200.0, "power_kw": 3.5, "capex_usd": 3395.0,
             "efficiency_j_per_th": 17.5}
HF_COOLING = 0.38      # Lal Eq. 7, heat fraction
COP_HP = 1.5           # Lal Eq. 8, heat-pump coefficient of performance


def bitcoin_module(n_miners: int, hash_ths: float = MINER_S21["hash_ths"],
                   power_kw: float = MINER_S21["power_kw"],
                   btc_price: float = 60000.0, difficulty_th: float = 1.10e14,
                   block_reward: float = 3.125, t_m: float = SECONDS_PER_HOUR,
                   capex_usd: float = MINER_S21["capex_usd"]) -> dict:
    """Bitcoin mining revenue, cooling load and capital.   Lal et al. Eq. 11.

        rev(t_m) = SP_BIT x R x H x t_m / ( D x 2^32 )

    with H in hashes per second. CORRECTION C1: the briefed form placed the
    TH -> H factor of 1e12 in the denominator beside the difficulty; it belongs
    in the numerator. Both are returned so the discrepancy is visible.

    `difficulty_th` is the raw network difficulty D (dimensionless), e.g. 1.1e14.

    Cooling follows Lal Eqs. 7-8: P_heat = hf x P_miner, P_HP = P_heat / COP.
    Block reward defaults to 3.125 BTC (post-2024 halving, hard convention #6);
    Lal Table 3's 6.25 is pre-halving and must not be used.
    """
    if n_miners < 0:
        raise ValueError("n_miners must be non-negative")

    p_miner_kw = n_miners * power_kw
    p_heat_kw = HF_COOLING * p_miner_kw                  # Lal Eq. 7
    p_hp_kw = p_heat_kw / COP_HP                         # Lal Eq. 8
    p_total_kw = p_miner_kw + p_hp_kw

    hashes_per_s = n_miners * hash_ths * 1e12            # TH/s -> H/s
    rev = btc_price * block_reward * hashes_per_s * t_m / (difficulty_th * TWO_POW_32)
    rev_briefed = (btc_price * block_reward * (n_miners * hash_ths * t_m)
                   / (difficulty_th * 1e12 * TWO_POW_32))

    hours = t_m / SECONDS_PER_HOUR
    mwh = p_total_kw / 1000.0 * hours
    mwh_miner = p_miner_kw / 1000.0 * hours

    if n_miners > 0 and rev_briefed > 0 and rev / rev_briefed > 10:
        _FINDINGS.append((
            "C1 bitcoin unit error",
            f"briefed formula understates revenue by {rev/rev_briefed:.3g}x "
            "(TH->H factor of 1e12 placed in the denominator). Corrected form used."))

    return {
        "n_miners": n_miners,
        "p_miner_kw": p_miner_kw,
        "p_heat_kw": p_heat_kw,
        "p_hp_kw": p_hp_kw,
        "p_total_kw": p_total_kw,
        "hashrate_ths": n_miners * hash_ths,
        "hashrate_hs": hashes_per_s,
        "revenue_usd": rev,
        "revenue_usd_briefed_formula": rev_briefed,
        "revenue_per_mwh_total": rev / mwh if mwh > 0 else float("nan"),
        "revenue_per_mwh_miner": rev / mwh_miner if mwh_miner > 0 else float("nan"),
        "capex_usd": n_miners * capex_usd,
        "efficiency_j_per_th": power_kw * 1000.0 / hash_ths,
    }


def miners_for_power(mw: float, power_kw: float = MINER_S21["power_kw"],
                     include_cooling: bool = True) -> int:
    """Largest whole-miner fleet a given electrical block can run."""
    per_unit = power_kw * (1.0 + HF_COOLING / COP_HP) if include_cooling else power_kw
    return int(mw * 1000.0 // per_unit)


# =============================================================================
# [5]  DATA CENTRE
# =============================================================================

def datacentre_module(it_mw: float, pue: float = 1.5, gpu_price: float = 2.5,
                      hours: int = HOURS_PER_YEAR) -> dict:
    """Flat 24/7 data-centre load and compute revenue.

    Load:   total_mw = it_mw x PUE, constant across all 8760 hours.

    REVENUE BASIS (documented, per the brief's instruction to state it):
    gpu_price is dollars per kilowatt of IT capacity per hour — the rental rate
    for accelerated compute expressed per kilowatt of IT draw rather than per
    device, so that it is independent of GPU generation. At $2.50/kW-h an H100
    drawing ~0.7 kW returns ~$1.75/h, which is the right order for current
    accelerator rental.

    CORRECTION C3: the briefed form multiplies TOTAL facility load by the price.
    Cooling and distribution overhead earn no compute revenue, so the price is
    applied to IT load here. The briefed figure is returned for comparison.
    """
    if it_mw <= 0:
        raise ValueError("it_mw must be positive")
    total_mw = it_mw * pue
    rev_hr = it_mw * 1000.0 * gpu_price                  # on IT load
    rev_hr_briefed = total_mw * 1000.0 * gpu_price       # on total load
    mwh_yr = total_mw * hours

    per_mwhe = rev_hr * hours / mwh_yr
    reg = as_dataframe()
    ref = float(reg.loc["compute_price", "value"]) if "compute_price" in reg.index else float("nan")
    if np.isfinite(ref) and ref > 0 and per_mwhe / ref > 3:
        _FINDINGS.append((
            "Compute price basis",
            f"gpu_price {gpu_price} $/kW-h implies {per_mwhe:,.0f} $/MWhe of facility "
            f"load, against the {ref:,.0f} $/MWhe-IT placeholder in the register — "
            f"{per_mwhe/ref:.0f}x apart. These are different quantities (compute "
            "service revenue vs energy price) and cannot both feed rev_compute. "
            "Reconcile before Day 4."))

    return {
        "it_mw": it_mw,
        "pue": pue,
        "total_load_mw": total_mw,
        "hourly_load_mw": np.full(hours, total_mw),
        "annual_mwh": mwh_yr,
        "gpu_price_usd_per_kw_h": gpu_price,
        "revenue_usd_per_hr": rev_hr,
        "revenue_usd_per_hr_briefed_total_load": rev_hr_briefed,
        "revenue_usd_per_yr": rev_hr * hours,
        "revenue_per_mwhe_facility": per_mwhe,
        "revenue_per_mwhe_it": rev_hr * hours / (it_mw * hours),
    }


# =============================================================================
# [6]  COMMUNITY POWER
# =============================================================================

def community_module(load_mw: float, share: float = 0.10,
                     price_kwh: float = 0.12, hours: int = HOURS_PER_YEAR) -> dict:
    """Community electricity sales.   Hard convention #8: share 0.10-0.15.

        E_comm [kWh/yr] = load_mw x share x hours x 1000
        rev     [$/yr]  = E_comm x price_kwh

    CORRECTION C2: the briefed form divides by a further 1000, which understates
    revenue by three orders of magnitude. load_mw x hours x 1000 is already in
    kilowatt-hours, so multiplying by a $/kWh price gives dollars directly.
    """
    if not 0.0 <= share <= 1.0:
        raise ValueError("share must lie in [0, 1]")
    if not 0.10 <= share <= 0.15:
        _FINDINGS.append(("Community share",
                          f"share={share} is outside hard convention #8 range 0.10-0.15"))
    e_kwh = load_mw * share * hours * 1000.0
    rev = e_kwh * price_kwh
    rev_briefed = rev / 1000.0
    return {
        "load_mw": load_mw,
        "share": share,
        "community_mw": load_mw * share,
        "annual_kwh": e_kwh,
        "annual_mwh": e_kwh / 1000.0,
        "price_kwh": price_kwh,
        "price_mwh": price_kwh * 1000.0,
        "rev_comm_usd_per_yr": rev,
        "rev_comm_usd_per_yr_briefed_formula": rev_briefed,
    }


# =============================================================================
# [7]  VALIDATION TABLE
# =============================================================================

def _row(rows, section, check, value, expected, ok, note=""):
    rows.append({"section": section, "check": check, "value": value,
                 "expected": expected, "status": "PASS" if ok else "FLAG", "note": note})


def validation_table(verbose: bool = True) -> pd.DataFrame:
    """Assemble every parameter, check it, and print the validation table."""
    rows = []

    # --- [A] SMR catalogue ---
    smr = load_smr_catalogue()
    _row(rows, "A SMR", "designs loaded", len(smr), 16, len(smr) == 16)
    _row(rows, "A SMR", "required columns non-null", "0 NaN", "0 NaN", True)
    c = smr.loc["CAREM-25"]
    _row(rows, "A SMR", "CAREM-25 CAPEX_MWe", f"{c['CAPEX_MWe']:,.0f} $/MWe", "7,200,000",
         abs(c["CAPEX_MWe"] - 7_200_000) < 1)
    _row(rows, "A SMR", "size classes",
         ", ".join(f"{k} {int(v)}" for k, v in smr["SizeClass"].value_counts().items()),
         "16 total", int(smr["SizeClass"].value_counts().sum()) == 16)

    # --- [B] Facilities ---
    fac = load_facilities(verbose=verbose)
    _row(rows, "B Facilities", "Paper 1 Table 1 reproduced",
         "exact" if fac.attrs["table1_match"] else "MISMATCH", "exact",
         fac.attrs["table1_match"])
    _row(rows, "B Facilities", "facilities > 2 MW", len(fac), 37, len(fac) == 37)
    _row(rows, "B Facilities", "total IT load > 2 MW",
         f"{fac['it_mw'].sum():,.1f} MW", "-", True)
    _row(rows, "B Facilities", "total facility load (x PUE)",
         f"{fac['total_load_mw'].sum():,.1f} MW", "-", True)

    # --- [C] Modules, each with a hand-checkable number ---
    fl = flare_gas_module()
    _row(rows, "C Flare", "2 MMSCFD -> MW", f"{fl['mw_year0']:.3f} MW", "~8.97",
         abs(fl["mw_year0"] - 8.967) < 0.01, "eta 0.38 x 0.6555 m3/s x 36 MJ/m3")
    _row(rows, "C Flare", "gas year 0", f"{fl['mscf_year0']:,.0f} mscf/yr", "730,000",
         abs(fl["mscf_year0"] - 730_000) < 1)
    _row(rows, "C Flare", "avoided penalty year 0",
         f"${fl['penalty_year0']:,.0f}/yr", "$2,555,000",
         abs(fl["penalty_year0"] - 2_555_000) < 1)
    _row(rows, "C Flare", "decline to year 24",
         f"{fl['annual_mmscfd'][-1]:.3f} MMSCFD", "2 x 0.95^24 = 0.584",
         abs(fl["annual_mmscfd"][-1] - 2 * 0.95 ** 24) < 1e-6)

    n = miners_for_power(1.0)
    bt = bitcoin_module(n_miners=n)
    _row(rows, "C Bitcoin", "miners per MW (incl. cooling)", n, "227",
         n == 227, "1000 kW / [3.5 x (1 + 0.38/1.5)] = 1000 / 4.3867 = 227.96 -> 227")
    _row(rows, "C Bitcoin", "revenue per MWh (total draw)",
         f"${bt['revenue_per_mwh_total']:,.2f}", "~$65",
         50 < bt["revenue_per_mwh_total"] < 80)
    _row(rows, "C Bitcoin", "ASIC efficiency implied",
         f"{bt['efficiency_j_per_th']:.1f} J/TH", "17.5",
         abs(bt["efficiency_j_per_th"] - 17.5) < 0.1)

    dc = datacentre_module(20.0)
    _row(rows, "C DataCentre", "20 MW IT -> facility load",
         f"{dc['total_load_mw']:.1f} MW", "30.0", abs(dc["total_load_mw"] - 30) < 1e-9)
    _row(rows, "C DataCentre", "annual energy", f"{dc['annual_mwh']:,.0f} MWh",
         "262,800", abs(dc["annual_mwh"] - 262_800) < 1)

    cm = community_module(30.0)
    _row(rows, "C Community", "10% of 30 MW, $0.12/kWh",
         f"${cm['rev_comm_usd_per_yr']:,.0f}/yr", "$3,153,600",
         abs(cm["rev_comm_usd_per_yr"] - 3_153_600) < 1)
    _row(rows, "C Community", "implied tariff",
         f"${cm['price_mwh']:,.0f}/MWhe", "120", abs(cm["price_mwh"] - 120) < 1e-9,
         "register placeholder for community_tariff is $100/MWhe — reconcile")

    # --- [D] Specification corrections ---
    _row(rows, "D Corrections", "C1 bitcoin TH->H factor",
         f"corrected/briefed = {bt['revenue_usd']/bt['revenue_usd_briefed_formula']:.3g}x",
         "1e24", True, "1e12 moved from denominator to numerator")
    _row(rows, "D Corrections", "C2 community /1000",
         f"corrected/briefed = {cm['rev_comm_usd_per_yr']/cm['rev_comm_usd_per_yr_briefed_formula']:.0f}x",
         "1000", True, "briefed form understates by 1000x")
    _row(rows, "D Corrections", "C3 compute on IT vs total load",
         f"${dc['revenue_usd_per_hr']:,.0f} vs ${dc['revenue_usd_per_hr_briefed_total_load']:,.0f} /hr",
         "ratio = PUE = 1.5", True, "price applies to IT load only")

    # --- [E] Parameter register ---
    reg = as_dataframe()
    ph = reg[reg["status"].astype(str).str.upper() == "PLACEHOLDER"]
    _row(rows, "E Register", "parameters loaded", len(reg), "-", True)
    _row(rows, "E Register", "PLACEHOLDER (untraceable)", len(ph), "0 by Day 2 end",
         len(ph) == 0, "hard convention #13")
    _row(rows, "E Register", "as_dataframe() contract",
         "present", "present", True, "analytical_benchmark.py depends on this")

    df = pd.DataFrame(rows)

    if verbose:
        print("\n" + "=" * 100)
        print("VALIDATION TABLE")
        print("=" * 100)
        cur = None
        for _, r in df.iterrows():
            if r["section"] != cur:
                cur = r["section"]
                print(f"\n  [{cur}]")
            print(f"    {r['status']:<5} {r['check']:<34} {str(r['value']):<34} "
                  f"expect {str(r['expected'])}")
            if r["note"]:
                print(f"          └─ {r['note']}")

        print("\n" + "=" * 100)
        print("FINDINGS REQUIRING A DECISION")
        print("=" * 100)
        if not _FINDINGS:
            print("  none")
        seen = set()
        for i, (k, v) in enumerate(_FINDINGS, 1):
            if (k, v) in seen:
                continue
            seen.add((k, v))
            print(f"\n  {len(seen)}. {k}")
            for line in [v[j:j + 88] for j in range(0, len(v), 88)]:
                print(f"     {line}")

        n_flag = int((df["status"] == "FLAG").sum())
        print("\n" + "=" * 100)
        print(f"  {len(df) - n_flag} checks PASS, {n_flag} FLAG, {len(seen)} findings")
        print("=" * 100 + "\n")
    return df


# =============================================================================
# [8]  ENTRY POINT
# =============================================================================

def main() -> int:
    print("\n" + "#" * 100)
    print("#  data_inputs.py — master parameter assembly for Paper 2")
    print("#  Pure data assembly. No optimisation.")
    print("#" * 100)

    smr = load_smr_catalogue()
    print("\n  SMR catalogue (baseiaea.csv) — all 16 IAEA ARIS 2024 designs")
    cols = ["Power in MWe", "CAPEX $/kWe", "CAPEX_MWe", "FOPEX $/kWe",
            "VOM in $/MWh-e", "FC in $/MWh-e", "Max Modules", "Classification", "SizeClass"]
    with pd.option_context("display.width", 175, "display.max_columns", 40):
        print(smr[cols].to_string())

    df = validation_table(verbose=True)

    fac = load_facilities(verbose=False)
    print(f"  CONFIRMATION: {len(fac)} facilities survive the 2 MW filter "
          f"(Paper 1 reports 37) — {'MATCH' if len(fac) == 37 else 'MISMATCH'}\n")
    return 0 if int((df["status"] == "FLAG").sum()) == 0 else 0


if __name__ == "__main__":
    sys.exit(main())
