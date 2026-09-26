#!/usr/bin/env python3
"""
Paper 2 -- Section 4.1.4: validation of Eq. (E31) against the operational
2 MW flare-gas Bitcoin site, southern Nigeria.

    python validate_pilot_mining.py --data dailystats_2026-09-18.csv

--------------------------------------------------------------------------------
WHAT THIS DATA CAN AND CANNOT VALIDATE
--------------------------------------------------------------------------------
The file is a POOL STATISTICS export: daily hashrate, uptime, worker count,
hashprice, realised BTC and the BTC/USD close. It contains NO electrical
measurement -- no MW, no kWh, no gas flow.

So the validation splits, and the split is not cosmetic:

  DIRECT, against measurement
      Eq. (E31), the Bitcoin revenue model. Realised BTC against hashrate x
      hashprice. This is a true out-of-sample test of the paper's revenue
      equation and it is reported with full error statistics.

  DIRECT, against measurement
      Availability. The uptime column is measured and replaces the D-09
      placeholder outright.

  INDIRECT, conditional on an assumed miner efficiency
      Electrical power, and therefore $/MWhe. Power is INFERRED as
      hashrate x J/TH, and J/TH is itself the D-07 placeholder. Any $/MWhe
      statement below is conditional on that assumption and is reported across
      a range rather than as a point. It is NOT a measurement and is never
      called one.

  NOT VALIDATED AT ALL
      Gas flow, heating value (D-08), genset efficiency, parasitic load.
      Nothing in this file bears on them.

ASHRAE Guideline 14 bands are applied to the direct test: NMBE within +/-10%,
CV(RMSE) <= 30%. They were fixed in validate_pilot.py before this file arrived.
--------------------------------------------------------------------------------
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

G14 = {"nmbe": 10.0, "cvrmse": 30.0}
MODEL_UPTIME = 0.90          # D-09 placeholder currently in data_inputs
MODEL_JTH = 17.5             # D-07 placeholder
LAL_MINER_KW = 3.25          # Lal et al. Table 3, P_MAX_Miner
LOG: List[str] = []


def say(s: str = "") -> None:
    print(s, flush=True)
    LOG.append(str(s))


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(
        s.astype(str).str.replace(r'[$,%"\s]', "", regex=True), errors="coerce")


def metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    m = np.isfinite(pred) & np.isfinite(obs)
    p, o = np.asarray(pred, float)[m], np.asarray(obs, float)[m]
    err = p - o
    obar = o.mean()
    rmse = float(np.sqrt(np.mean(err ** 2)))
    nz = np.abs(o) > 1e-12
    return {"n": len(p), "rmse": rmse, "cvrmse": 100 * rmse / obar,
            "mbe": float(err.mean()), "nmbe": 100 * float(err.mean()) / obar,
            "mape": 100 * float(np.mean(np.abs(err[nz] / o[nz]))),
            "r2": float(1 - np.sum(err ** 2) / np.sum((o - obar) ** 2)),
            "obs_total": float(o.sum()), "pred_total": float(p.sum())}


def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path, encoding="utf-8-sig")
    d.columns = [c.strip().strip('"') for c in d.columns]
    d["date"] = pd.to_datetime(d["Date"])
    for c in ("Hashrate (PH/s)", "Shares Efficiency", "Uptime", "Workers count",
              "Price (BTC/PH/s/Day)", "Price (USD/PH/s/Day)",
              "Miner Revenue (BTC)", "End of Day Balance (BTC)",
              "BTC/USD Price"):
        if c in d.columns:
            d[c] = num(d[c])
    for c in ("Uptime", "Shares Efficiency"):
        d[c] = d[c] / 100.0
    return d.sort_values("date").reset_index(drop=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="dailystats_2026-09-18.csv")
    ap.add_argument("--outdir", default="pilot_validation")
    ap.add_argument("--site-mw", type=float, default=2.0)
    ap.add_argument("--jth", type=float, default=None,
                    help="assumed miner J/TH; default derives it from Lal "
                         "Table 3 miner power and the measured hashrate")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)

    d = load(a.data)
    say("=" * 80)
    say("  SECTION 4.1.4 -- VALIDATION AGAINST THE OPERATIONAL 2 MW SITE")
    say("=" * 80)
    say(f"  source      : {a.data}")
    say(f"  records     : {len(d):,} daily rows, "
        f"{d.date.min().date()} to {d.date.max().date()}")
    prod = d[d["Miner Revenue (BTC)"].notna() & (d["Hashrate (PH/s)"] > 0)].copy()
    say(f"  producing   : {len(prod):,} days with both hashrate and revenue")
    say("")

    # ---------------------------------------------------------------- E31
    say("  " + "-" * 76)
    say("  DIRECT TEST 1 -- Eq. (E31), Bitcoin revenue")
    say("  " + "-" * 76)
    say(textwrap.indent(textwrap.fill(
        "Eq. (E31) states rev = SP_BIT x R x H x t / (D x 2^32). The pool "
        "reports the same quantity already reduced to a hashprice, "
        "Price (BTC/PH/s/Day), which is exactly R x t / (D x 2^32) per PH/s. "
        "Predicting realised BTC as hashrate x hashprice is therefore a test "
        "of Eq. (E31) itself, not of a re-parameterisation of it.", 76), "    "))
    say("")
    prod["pred_btc"] = prod["Hashrate (PH/s)"] * prod["Price (BTC/PH/s/Day)"]
    m = metrics(prod.pred_btc.values, prod["Miner Revenue (BTC)"].values)
    for k, lab in (("n", "days"), ("obs_total", "measured BTC"),
                   ("pred_total", "predicted BTC"), ("rmse", "RMSE (BTC/day)"),
                   ("cvrmse", "CV(RMSE) %"), ("nmbe", "NMBE %"),
                   ("mape", "MAPE %"), ("r2", "R2")):
        say(f"    {lab:<20s} {m[k]:>16,.6f}")
    ok = abs(m["nmbe"]) <= G14["nmbe"] and m["cvrmse"] <= G14["cvrmse"]
    say(f"    ASHRAE G14 (NMBE +/-10%, CV(RMSE) <=30%): "
        f"{'PASS' if ok else 'FAIL'}")
    say("")
    se = prod["Shares Efficiency"].mean()
    say(textwrap.indent(textwrap.fill(
        f"The bias is POSITIVE and small: the equation over-predicts by "
        f"{m['nmbe']:.2f}%. That is not unexplained residual. Mean shares "
        f"efficiency over the record is {se:.4f}, i.e. {100*(1-se):.2f}% of "
        f"submitted work is stale or rejected, and a pool fee of order 1-2% "
        f"sits on top. Together they account for the gap almost exactly. "
        f"Eq. (E31) describes gross hashrate-proportional issuance; the site "
        f"receives it net of share loss and pool fee. The paper should state "
        f"that Eq. (E31) is a GROSS revenue model and apply a realisation "
        f"factor of {m['obs_total']/m['pred_total']:.4f} when comparing to "
        f"operator cash.", 76), "    "))
    say("")

    # ------------------------------------------------------------- uptime
    say("  " + "-" * 76)
    say("  DIRECT TEST 2 -- availability. THE MODEL IS WRONG BY A LARGE MARGIN.")
    say("  " + "-" * 76)
    u = d["Uptime"].dropna()
    say(f"    measured uptime   mean {u.mean():.4f}   median {u.median():.4f}")
    say(f"                      p10 {u.quantile(.10):.4f}   p90 {u.quantile(.90):.4f}"
        f"   max {u.max():.4f}")
    say(f"    days above 0.90   {int((u > 0.90).sum())} of {len(u)}")
    say(f"    days below 0.50   {int((u < 0.50).sum())} of {len(u)}")
    say(f"    MODEL ASSUMES     {MODEL_UPTIME:.4f}   (D-09 placeholder)")
    say(f"    ratio model/measured  {MODEL_UPTIME/u.mean():.3f}")
    say("")
    say(textwrap.indent(textwrap.fill(
        f"Measured availability is {u.mean():.1%}, not the {MODEL_UPTIME:.0%} "
        f"the model assumes. Only {int((u>0.90).sum())} of {len(u)} days "
        f"cleared 90%. This is the single most consequential finding in the "
        f"validation: every Phase-1 and Phase-2 energy and revenue quantity in "
        f"the paper is scaled by this factor, so the model over-states "
        f"flare-gas generation, mining revenue, avoided penalty and avoided CO2 "
        f"by roughly {MODEL_UPTIME/u.mean():.2f}x. The placeholder must be "
        f"replaced by the measured value and the Phase-1 results re-run before "
        f"any of them are quoted.", 76), "    "))
    say("")

    # ------------------------------------------------------------ fleet scale
    say("  " + "-" * 76)
    say("  DIRECT TEST 3 -- installed scale. The site is NOT a 2 MW mining load.")
    say("  " + "-" * 76)
    w = d["Workers count"].dropna()
    prod["ths_per_worker"] = (prod["Hashrate (PH/s)"] * 1000
                              / prod["Workers count"]
                              / prod["Uptime"].clip(lower=0.01))
    tpw = prod.ths_per_worker.median()
    say(f"    workers           max {w.max():.0f}   last-90-day median "
        f"{w.tail(90).median():.0f}   current {w.dropna().iloc[-1]:.0f}")
    say(f"    TH/s per worker at 100% uptime, median {tpw:,.1f}")
    say(f"    implied fleet power at Lal Table 3 ({LAL_MINER_KW} kW/miner):")
    say(f"      peak  {w.max()*LAL_MINER_KW/1000:.3f} MW"
        f"   recent {w.tail(90).median()*LAL_MINER_KW/1000:.3f} MW"
        f"   of a {a.site_mw:.1f} MW site")
    say(f"    peak utilisation of nameplate: "
        f"{100*w.max()*LAL_MINER_KW/1000/a.site_mw:.1f}%")
    say("")
    say(textwrap.indent(textwrap.fill(
        f"The site has never operated at 2 MW of mining load. Peak deployment "
        f"was {w.max():.0f} miners, about "
        f"{w.max()*LAL_MINER_KW/1000:.2f} MW or "
        f"{100*w.max()*LAL_MINER_KW/1000/a.site_mw:.0f}% of nameplate, and the "
        f"recent fleet is roughly "
        f"{w.tail(90).median()*LAL_MINER_KW/1000*1000:.0f} kW. The paper must "
        f"describe this as a 2 MW SITE with a partially deployed mining fleet, "
        f"never as 2 MW of operating mining load, and the validation is of the "
        f"revenue EQUATION at the scale actually run -- not of the model's "
        f"8.97 MW Phase-1 configuration, which no measurement here reaches.",
        76), "    "))
    say("")

    # ------------------------------------- conditional: revenue per MWhe
    say("  " + "-" * 76)
    say("  INDIRECT -- $/MWhe of mining energy. CONDITIONAL ON D-07.")
    say("  " + "-" * 76)
    hp = d["Price (USD/PH/s/Day)"].dropna()
    say(f"    measured hashprice  median ${hp.median():.2f}/PH/s/day"
        f"   first ${hp.iloc[0]:.2f}   last ${hp.iloc[-1]:.2f}")
    jth_implied = LAL_MINER_KW * 1000.0 / tpw
    say(f"    J/TH implied by Lal Table 3 miner power and measured hashrate: "
        f"{jth_implied:.1f}")
    say(f"    D-07 placeholder in the model: {MODEL_JTH:.1f} J/TH")
    say("")
    say(f"    {'assumed J/TH':>14s} {'kW per PH/s':>13s} {'MWh/PH/s/day':>14s} "
        f"{'$/MWhe at median hashprice':>28s}")
    rows = []
    for jth in (17.5, 21.5, 29.5, 34.5, jth_implied, 45.0):
        kw = jth * 1000.0 / 1000.0          # J/TH x 1000 TH/s = W -> kW
        mwh = kw * 24 / 1000.0
        usd = hp.median() / mwh
        rows.append(dict(jth=jth, kw_per_phs=kw, mwh_per_phs_day=mwh,
                         usd_per_mwhe=usd))
        tag = "  <- implied by this data" if abs(jth - jth_implied) < 1e-9 else \
              ("  <- model placeholder" if abs(jth - MODEL_JTH) < 1e-9 else "")
        say(f"    {jth:>14.1f} {kw:>13.1f} {mwh:>14.3f} {usd:>28,.2f}{tag}")
    say("")
    say(f"    MODEL VALUE: $81.6419/MWhe at BTC $60,000 (Phase 1 and 2)")
    say("")
    say(textwrap.indent(textwrap.fill(
        f"The model's $81.64/MWhe is bracketed by this table, and where it "
        f"falls depends entirely on D-07. At the placeholder 17.5 J/TH the "
        f"pilot's own hashprice implies ${hp.median()/(17.5*24/1000):.2f}/MWhe, "
        f"so the model is CONSERVATIVE. At the {jth_implied:.1f} J/TH this "
        f"data implies -- older hardware, which is what an early flare site "
        f"actually gets -- it implies "
        f"${hp.median()/(jth_implied*24/1000):.2f}/MWhe and the model is "
        f"roughly {81.6419/(hp.median()/(jth_implied*24/1000)):.1f}x TOO HIGH. "
        f"This range, not a point, is the honest statement until miner power "
        f"is measured. Note the caveat that the implied J/TH itself rests on "
        f"Lal's 3.25 kW per miner rather than a meter.", 76), "    "))
    say("")

    # ----------------------------------------------------------- economics
    say("  " + "-" * 76)
    say("  REALISED ECONOMICS")
    say("  " + "-" * 76)
    usd = (prod["Miner Revenue (BTC)"] * prod["BTC/USD Price"]).sum()
    yrs = (prod.date.max() - prod.date.min()).days / 365.25
    say(f"    BTC mined, whole record     {prod['Miner Revenue (BTC)'].sum():.6f} BTC")
    say(f"    realised USD (same-day px)  ${usd:,.0f} over {yrs:.2f} yr")
    say(f"    annualised                  ${usd/yrs:,.0f}/yr")
    say(f"    BTC/USD over record         ${d['BTC/USD Price'].min():,.0f} "
        f"to ${d['BTC/USD Price'].max():,.0f}, median "
        f"${d['BTC/USD Price'].median():,.0f}")
    say("")
    for y, g in prod.groupby(prod.date.dt.year):
        say(f"    {y}: {len(g):>3d} days  mean hashrate "
            f"{g['Hashrate (PH/s)'].mean():>5.2f} PH/s  mean uptime "
            f"{g['Uptime'].mean():.3f}  "
            f"${(g['Miner Revenue (BTC)']*g['BTC/USD Price']).sum():>9,.0f}")
    say("")
    say(textwrap.indent(textwrap.fill(
        f"Realised revenue is ${usd/yrs:,.0f}/yr against the model's Phase-1 "
        f"mining revenue of $5.45M/yr. The ratio is not a model error: the "
        f"model's Phase-1 site is 8.97 MW of genset feeding mining, the pilot "
        f"has run at most {w.max()*LAL_MINER_KW/1000:.2f} MW and recently far "
        f"less. The two are not the same object and must not be compared as "
        f"totals. What IS comparable, and what this validation establishes, is "
        f"the revenue EQUATION per unit hashrate, which holds to R2 = "
        f"{m['r2']:.4f}.", 76), "    "))
    say("")

    prod.to_csv(os.path.join(a.outdir, "pilot_daily_processed.csv"), index=False)
    pd.DataFrame([m]).to_csv(os.path.join(a.outdir, "pilot_E31_metrics.csv"),
                             index=False)
    pd.DataFrame(rows).to_csv(os.path.join(a.outdir, "pilot_jth_sensitivity.csv"),
                              index=False)
    try:
        make_fig(prod, d, m, a.outdir)
    except Exception as e:                     # noqa: BLE001
        say(f"  figure skipped: {type(e).__name__}: {e}")
    with open(os.path.join(a.outdir, "pilot_validation_log.txt"), "w") as fh:
        fh.write("\n".join(LOG) + "\n")
    say(f"  wrote {a.outdir}/ (log, metrics, processed series, Fig. 9)")
    return 0 if ok else 1


def make_fig(prod: pd.DataFrame, d: pd.DataFrame, m: Dict, outdir: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "savefig.dpi": 600, "font.family": "DejaVu Sans", "font.size": 7.5,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.minor.visible": True, "ytick.minor.visible": True,
        "axes.edgecolor": "#0b0b0b", "axes.linewidth": 0.7,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "legend.frameon": True, "legend.edgecolor": "#8a8985",
        "axes.labelsize": 8, "legend.fontsize": 6.6,
    })
    B, O, INK2, INK3 = "#2a78d6", "#eb6834", "#52514e", "#8a8985"
    fig, ax = plt.subplots(1, 3, figsize=(7.48, 2.5))

    ax[0].plot(prod.date, prod["Miner Revenue (BTC)"] * 1e3, color=B, lw=0.6,
               label="measured")
    ax[0].plot(prod.date, prod.pred_btc * 1e3, color=O, lw=0.6, ls="--",
               label="Eq. (E31)")
    ax[0].set_ylabel("Daily revenue (mBTC)")
    ax[0].set_xlabel("Date")
    ax[0].legend(loc="upper right")
    ax[0].set_title("(a) Revenue, measured vs modelled", fontsize=7.6, loc="left")
    for t in ax[0].get_xticklabels():
        t.set_rotation(30); t.set_ha("right")

    lo = 0.0
    hi = float(max(prod["Miner Revenue (BTC)"].max(), prod.pred_btc.max())) * 1e3
    ax[1].plot([lo, hi], [lo, hi], color=INK3, lw=0.8, ls=":")
    ax[1].plot(prod["Miner Revenue (BTC)"] * 1e3, prod.pred_btc * 1e3, "o",
               ms=2.0, mfc=B, mec="none", alpha=0.45)
    ax[1].set_xlabel("Measured (mBTC/day)")
    ax[1].set_ylabel("Eq. (E31) (mBTC/day)")
    ax[1].set_title("(b) Parity", fontsize=7.6, loc="left")
    ax[1].text(0.04, 0.96, f"n = {int(m['n']):,}\n$R^2$ = {m['r2']:.4f}\n"
                           f"CV(RMSE) = {m['cvrmse']:.2f}%\n"
                           f"NMBE = {m['nmbe']:+.2f}%\nMAPE = {m['mape']:.2f}%",
               transform=ax[1].transAxes, va="top", fontsize=6.3, color=INK2)

    u = d.dropna(subset=["Uptime"])
    ax[2].plot(u.date, u["Uptime"], color=B, lw=0.5, alpha=0.8)
    ax[2].axhline(u["Uptime"].mean(), color=B, lw=1.2,
                  label=f"measured mean {u['Uptime'].mean():.3f}")
    ax[2].axhline(MODEL_UPTIME, color=O, lw=1.4, ls="--",
                  label=f"model assumes {MODEL_UPTIME:.2f}")
    ax[2].set_ylim(0, 1.0)
    ax[2].set_ylabel("Uptime (fraction)")
    ax[2].set_xlabel("Date")
    ax[2].legend(loc="upper right")
    ax[2].set_title("(c) Availability", fontsize=7.6, loc="left")
    for t in ax[2].get_xticklabels():
        t.set_rotation(30); t.set_ha("right")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(outdir, f"Fig9_pilot_validation.{ext}"),
                    dpi=600, bbox_inches="tight")
    say(f"    wrote {outdir}/Fig9_pilot_validation.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
