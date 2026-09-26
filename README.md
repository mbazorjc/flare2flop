# flare2flop
codes for analysing compute bridge finance for SMR in SSA

The model follows one site through four phases:

| Phase | Years | Supply | Load | Revenue streams |
|---|---|---|---|---|
| 1 | 0–3 | associated-gas gensets | bitcoin mining + community | avoided flaring charge, mining, carbon, community |
| 2 | 3–5 | + PV–wind hybrid | + early AI compute | as Phase 1 + compute |
| 3 | 5–10 | small modular reactor (SMR) | AI/HPC data centre + community | compute, carbon, community |
| 4 | 10+ | SMR + 20 % renewable overbuild | data centre + community (15 %) | as Phase 3 |

Viability is reported as a **unified diesel breakeven** that reduces algebraically to the single-revenue
SMR formulation of the authors' earlier study (SSRN, https://doi.org/10.2139/ssrn.7372621) when the
added streams are zero. Timing of the SMR is valued with a Cox–Ross–Rubinstein real-options lattice,
and uncertainty with Monte Carlo (20,000 draws) and Sobol' indices. The mining revenue equation is
validated against 718 days of pool records from a 2 MW flare-to-mining site in southern Nigeria.

---

To run:

pip install numpy pandas matplotlib

python verify_phased.py  

python validate_pilot_mining.py 

python uq_phased.py           

