"""
LDI liquidity stress test for a stylised UK defined-benefit pension fund
=======================================================================
Liabilities : Impavido (2011/2013), "Conducting Stress Tests of Defined Benefit Pension Plans"
              (IMF Guide to Stress Testing, Ch. 12) - data, assumptions and template logic replicated.
Liquidity   : Cont and Schaanning (2017), "Fire Sales, Indirect Contagion and Systemic Stress
              Testing", Section 2 - single-fund version (no cross-institution contagion).
Calibration : FCA (2023), Bank of England (2023), TPR (2022, 2023), Pinter (2023),
              Bandera and Stevens (2024) - see Chapter 4, Table 4.4.

Run:  python ldi_stress_test.py        (creates tables in ./outputs and charts in ./outputs/figures)
"""

# =====================================================================================
# 0. SETUP
# =====================================================================================
import os
from dataclasses import dataclass, replace
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# =====================================================================================
# 0b. PARAMETERS - edit here only; every other section reads these names
# =====================================================================================
MODEL_VERSION = "v1.4 (working version, 2026-10-20; UK funding-ratio range, buffer formula and map, assumptions table added)"
# ---- starting position ----
HEDGE = 0.85                     # liability-matching exposure / liability value (Bandera and Stevens, 2024)
START_FR = 1.00                  # starting funding ratio (assets / liabilities)
START_2022, PEAK_2022, START_2026 = "2022-09-22", "2022-09-27", "2026-09-30"   # any date in the Bank file (2016 to 30 Sep 2026)
# ---- shocks and standards ----
PARALLEL_SHOCKS_BP = (160, 250, 400)   # parallel rises tested (FCA 160bp; FPC minimum 250bp)
STANDARD_BP = 250                      # minimum resilience the 2026 fund is sized to (Bank of England, 2023)
HEADROOM_2022_BP = 150                 # pre-crisis headroom of the 2022 fund (TPR)
FALL_BP = -100                         # fall used for the solvency test (largest five-day fall since 2016: -99bp)
# ---- composition of hedging assets (PPF Purple Book: gilts and index-linked as a share of total assets) ----
BONDS_2022, GILTS_OF_BONDS_2022 = 0.716, 0.478 + 0.220   # 2022 edition (31 March 2022): bonds 71.6%; index-linked 47.8% + government fixed interest 22.0%
BONDS_2025, GILTS_OF_BONDS_2025 = 0.706, 0.444 + 0.208   # bonds 70.6%; index-linked 44.4% + government fixed interest 20.8%
# ---- market and trading costs ----
DEPTH_EQUITY, DEPTH_GILT = 4.0, 6.0    # market depth D in multiples of fund assets (placeholder; varied in sensitivities)
ALPHA = 0.5                            # implementation shortfall weight (Cont and Schaanning, 2017)
TC_EQUITY, TC_GILT = 0.005, 0.0007     # equity cost 0.5% (proxy); gilt cost 7bp in stress (Pinter, 2023)
SALE_RULE = "pecking"                  # "pecking" (cash, equities, gilts) or "proportional"
ILLIQUIDITY_MULTIPLES = (0.5, 1.0, 2.0, 4.0)   # sensitivity: depth divided by these
GILT_COST_SENSITIVITY = (0.0007, 0.0019)       # sensitivity: 7bp and 19bp
# ---- trade-off tables ----
CASH_BUDGETS = (0.0275, 0.05, 0.077, 0.10, 0.15, 0.20)   # includes the 2.75% and 7.7% benchmarks of Bandera and Stevens
HEDGES_SHOWN = (0.80, 0.85, 0.90, 1.00)
PROFILES = {"Cautious": (0.077, 1.00), "Balanced": (0.10, 0.85), "Growth-seeking": (0.15, 0.80)}   # (cash budget, hedge ratio) shown in charts
# ---- Purple Book 2025 context used in the hedging-asset sensitivity and profile guide ----
NEG_CASH_2022, ANNUITIES_2022, CORP_OF_BONDS_2022 = 0.088, 0.068, 0.302   # 2022: negative cash, annuities, all corporate bonds (IG not separable)
ASSETS_2025_BN = 1068.1                # Purple Book 2025 total assets (GBP bn), used to scale the 2025 overlay to pounds
ANNUITIES_2025 = 0.129                 # annuity (buy-in) holdings, share of assets
NEG_CASH_2025 = 0.079                  # negative cash = repo / swap financing of LDI (weighted average -7.9%)
IG_CORP_OF_BONDS_2025 = 0.235          # investment-grade corporate bonds as a share of bonds (0.4 + 12.2 + 10.9 per cent)
PURPLE_GROUPS = {"Closed to new benefit accrual": (0.088, 0.763, 0.184),     # (equities, bonds, annuities) as shares of assets
                 "Closed to new members": (0.137, 0.753, 0.055),
                 "Open": (0.421, 0.497, 0.005)}
ROBUST_FR = (0.80, 0.9433, 1.10, 1.25) # UK-justified starting funding ratios: 80% (low, as on a buy-out basis in 2022), 94.3% (Impavido plan: 3,773/4,000), 110%, 125% (Purple Book s179, 2025)
ROBUST_INFLATION = (0.025, 0.045)
ROBUST_STANDARDS_BP = (160, 400)
TORNADO_FR = (0.80, 1.25)              # starting funding ratios tested in the sensitivity (tornado) chart
TORNADO_HEDGE = (0.80, 0.90)           # hedge ratios tested
TORNADO_INFLATION = (0.025, 0.045)     # indexation assumptions tested
# ---- calibration targets (GBP bn) ----
HEDGED_LIAB_BN, SALES_BN, BOE_BN = 1400.0, 36.0, 19.3        # TPR (2022); Pinter (2023); Bandera and Stevens (2024)
PINTER_OIS_BN, PINTER_REPO_BN, PINTER_INFL_BN = 185.0, 135.0, 62.0    # Pinter (2023): net LDI-PI positions, 22 Sep 2022
# (Impavido's own settings - ages, accrual, 3.5% indexation - sit in Section 1; changing them breaks the replication.)

PALETTE = {"blue": "#2F5D8C", "light": "#9DB7DB", "red": "#C0504D", "green": "#6AA84F",
           "grey": "#7F7F7F", "gold": "#E3A72F", "dark": "#1F2D3D"}
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlesize": 12, "axes.labelsize": 10, "figure.dpi": 150})


# =====================================================================================
# 1. DATA - Impavido's real DB plan (IMF template Model.xlsx: sheets Input and BAL)
# =====================================================================================
def load_plan(path="data/impavido_model_input.csv"):
    """One row per age 0-125: mortality q'x(m), termination q'x(t), merit scale, members (AN, RN),
    wages (AW) and pensions (RB) per member in the cohort, as in the template's Input sheet."""
    return pd.read_csv(path).fillna(0.0)


PLAN = load_plan()
ASSETS_BAL = pd.read_csv("data/impavido_assets_BAL.csv")        # the plan's asset portfolio (sheet BAL)
ENTRY_AGE, RET_AGE, MAX_AGE = 20, 55, 115                       # template inputs D12, D13, D14
ACCRUAL, PRODUCTIVITY = 0.01, 0.01
INFL_WAGE, INFL_ANNUITY = 0.035, 0.035
TRUE_LIAB, TRUE_ASSETS = 4_000_000_000, 3_773_148_938.877245   # actuary's liability; plan assets (BAL)
WAGE_TO_RETIREMENT = True     # chapter variant: wages projected to age r (matches published Tables 12.4-12.6);
                              # False = workbook formula (wages projected to the last working year)


# =====================================================================================
# 2. LIABILITY MODEL - projected benefit obligation, constant dollar (exact template logic)
# =====================================================================================
def liability_cashflows(pa=INFL_ANNUITY, pw=INFL_WAGE, wage_to_ret=WAGE_TO_RETIREMENT, plan=PLAN):
    """Expected nominal benefit payments by year k (annuity-due: first payment at t = 0 for retirees).
    AL-R sheet : retiree cohort liability = members x pension x indexed annuity.
    Brx/AL-A   : active cohort liability = (x-y)/(r-y) x accrued pension at r x survival to r x annuity.
    Discounting a payment at time k at a flat yield y gives exactly the workbook's value."""
    q_m, q_t, ms = plan.q_m.values, plan.q_t.values, plan.ms.values
    AN, RN, AW, RB = plan.AN.values, plan.RN.values, plan.AW.values, plan.RB.values
    n = len(plan)
    y, r = ENTRY_AGE, RET_AGE
    pT = 1 - (q_m * (1 - q_t / 2) + q_t * (1 - q_m / 2))                # composite survival, Input!M
    infl = (1 + pa) ** np.arange(n + 1)

    def surv(x0):                                                         # sP_x: probability of surviving s years
        sp = np.ones(n + 1)
        for s in range(1, n + 1):
            idx = x0 + s
            sp[s] = sp[s - 1] * (0.0 if idx > MAX_AGE else pT[idx - 1])
        return sp

    cf = np.zeros(2 * n + 2)
    for x in range(r, MAX_AGE + 1):                                      # retired cohorts
        cf[: n + 1] += RN[x] * RB[x] * infl * surv(x)
    stream_r = infl * surv(r)                                             # annuity stream at retirement
    horizon = 1 if wage_to_ret else 0
    g = (1 + pw) * (1 + PRODUCTIVITY)
    for x in range(y, r):                                                 # active cohorts
        if AN[x] == 0:
            continue
        wf = ms[r - 1] / ms[x] * g ** (r - 1 - x + horizon)
        accrued = ACCRUAL * (r - y) * wf * AN[x] * AW[x]
        prod = np.prod([pT[x + s - 1] for s in range(1, n) if x + s < r])  # template's survival-to-r product
        w = (x - y) / (r - y) * accrued * prod
        start = r - x
        cf[start: start + n + 1] += w * stream_r
    return cf


def pv(cf, y):
    """Present value at a flat annual yield y."""
    return float(np.sum(cf / (1 + y) ** np.arange(len(cf))))


def mod_duration(cf, y, h=1e-4):
    return -(pv(cf, y + h) - pv(cf, y - h)) / (2 * h) / pv(cf, y)


# ---- replication check against Impavido's published funding ratios ------------------
FR_PLAN = 100 * TRUE_ASSETS / TRUE_LIAB            # 94.33: plan assets / actuary's liability
PUB_RATE = {0.09: 94.33, 0.08: 80.76, 0.07: 68.33, 0.06: 57.11, 0.05: 47.08, 0.04: 38.25}        # Table 12.4
PUB_PW = {-0.015: 101.97, -0.010: 99.40, -0.005: 96.86, 0.0: 94.33, 0.005: 91.83, 0.010: 89.35, 0.015: 86.90}  # 12.6
PUB_PA = {-0.015: 109.56, -0.010: 104.39, -0.005: 99.31, 0.0: 94.33, 0.005: 89.44, 0.010: 84.67, 0.015: 80.00}  # 12.6


def replication_table():
    base_cf = liability_cashflows()
    base = pv(base_cf, 0.09)
    rows = []
    for rate, pub in PUB_RATE.items():
        rows.append(("Discount rate", f"{rate:.0%}", pub, FR_PLAN * base / pv(base_cf, rate)))
    for d, pub in PUB_PW.items():
        rows.append(("Wage inflation", f"{d*1e4:+.0f}bp", pub, FR_PLAN * base / pv(liability_cashflows(pw=INFL_WAGE + d), 0.09)))
    for d, pub in PUB_PA.items():
        rows.append(("Annuity inflation", f"{d*1e4:+.0f}bp", pub, FR_PLAN * base / pv(liability_cashflows(pa=INFL_ANNUITY + d), 0.09)))
    t = pd.DataFrame(rows, columns=["Shock", "Value", "Impavido (published)", "This model"])
    t["Difference"] = t["This model"] - t["Impavido (published)"]
    return t


def workbook_check():
    """Workbook-exact variant: should reproduce the template's cells AL-R!D1, AL-A!D1 and AL-TOT!A6."""
    cf = liability_cashflows(wage_to_ret=False)
    return pv(cf, 0.09)          # compare with 3,968,302,344.18 in AL-TOT!A6


# ---- replication of Impavido Table 12.3: asset-class shocks on the plan's real portfolio --------
PUB_T123 = {   # published funding ratios (%): shock -> [Government, Financial, Real, Bonds, Stocks, FX, Total]
    -0.25: [89.74, 89.99, 87.47, 82.12, 83.06, 88.10, 70.75], -0.20: [90.66, 90.86, 88.84, 84.56, 85.31, 89.35, 75.46],
    -0.15: [91.57, 91.73, 90.21, 87.00, 87.56, 90.59, 80.18], -0.10: [92.49, 92.59, 91.58, 89.44, 89.82, 91.84, 84.90],
    -0.05: [93.41, 93.46, 92.96, 91.89, 92.07, 93.08, 89.61], 0.0: [94.33] * 7,
    0.05: [95.25, 95.20, 95.70, 96.77, 96.58, 95.57, 99.05], 0.10: [96.17, 96.06, 97.07, 99.21, 98.84, 96.82, 103.76],
    0.15: [97.08, 96.93, 98.44, 101.65, 101.09, 98.07, 108.48], 0.20: [98.00, 97.80, 99.82, 104.10, 103.35, 99.31, 113.19],
    0.25: [98.92, 98.67, 101.19, 106.54, 105.60, 100.56, 117.91]}
T123_COLS = ["Government", "Financial", "Real", "Bonds", "Stocks", "FX", "Total"]


def asset_class_shares():
    """Shares of plan assets by Table 12.3 class, from the BAL sheet (values in thousands of LCU)."""
    b = ASSETS_BAL
    tot = TRUE_ASSETS / 1000
    dom = b[(b.level == 2) & (b.group == "DOMESTIC INVESTMENTS")]
    gov = dom[dom.item.str.contains("Government")].value.sum()
    fin = dom[dom.item.str.contains("Financial")].value.sum()
    real = dom[dom.item.str.contains("Nonfinancial")].value.sum()
    transit = b[(b.level == 1) & (b.group.str.contains("TRANSIT"))].value.sum()
    foreign = b[(b.level == 1) & (b.group.str.contains("FOREIGN"))].value.sum() + transit
    stocks = b[(b.level == 3) & b.item.str.contains("Stock|stock|shares|Shares|Mutual funds|Privatization", regex=True)].value.sum()
    return dict(zip(T123_COLS, [gov / tot, fin / tot, real / tot, 1 - stocks / tot - transit / tot, stocks / tot, foreign / tot, 1.0]))


def asset_shock_table():
    sh = asset_class_shares()
    rows = []
    for shock, pub in PUB_T123.items():
        for c, p in zip(T123_COLS, pub):
            rows.append(dict(Shock=shock, Class=c, Share=sh[c], Published=p, Model=FR_PLAN * (1 + shock * sh[c])))
    t = pd.DataFrame(rows)
    t["Difference"] = t.Model - t.Published
    return t


# =====================================================================================
# 2b. YIELD CURVES - Bank of England nominal spot curve (daily, 2016-2026)
# =====================================================================================
def load_boe_curves(path="data/boe_spot_curve_2016_2026.csv"):
    """Spot yields (%, continuously compounded as published by the Bank) for maturities 0.5-40 years."""
    d = pd.read_csv(path, index_col=0, parse_dates=True)
    d.columns = d.columns.astype(float)
    return d


BOE = load_boe_curves()
CF_LEN = 254                                  # length of the liability cash-flow vector (years 0..253)


def curve_array(date, n=CF_LEN):
    """Continuously compounded spot yield for each payment year k = 0..n-1 (annual points are published nodes;
    held flat beyond 40 years, as Impavido's template holds its last node flat)."""
    row = BOE.loc[date] / 100.0
    k = np.arange(n)
    y = np.interp(np.clip(k, 0.5, 40.0), row.index.values, row.values)
    return y


CURVES = {START_2022: curve_array(START_2022), START_2026: curve_array(START_2026),
          "flat4": np.full(CF_LEN, np.log(1.04))}                 # flat 4% annual = legacy placeholder curve
MOVE_2022 = curve_array(PEAK_2022) - curve_array(START_2022)       # actual 22 -> 27 Sept 2022 change, by maturity


def pv_curve(cf, ycc):
    """Present value with continuously compounded spot yields: sum cf_k exp(-y_k k)."""
    return float(np.sum(cf * np.exp(-np.asarray(ycc)[: len(cf)] * np.arange(len(cf)))))


def dur_curve(cf, ycc, h=1e-4):
    return -(pv_curve(cf, ycc + h) - pv_curve(cf, ycc - h)) / (2 * h) / pv_curve(cf, ycc)


# =====================================================================================
# 3. FUND - stylised UK portfolio and liability-hedging overlay
# =====================================================================================
@dataclass(frozen=True)
class Fund:
    fr0: float = START_FR        # starting funding ratio: assets / liabilities
    curve_id: str = START_2026   # starting gilt curve (key of CURVES)
    cash: float = 0.02           # cash buffer, share of assets
    equity: float = 0.15         # return-seeking assets, share of assets
    hedge_ratio: float = HEDGE   # liability-matching exposure / liability value (Bandera and Stevens, 2024)
    liab_scale: float = 100.0    # liabilities normalised to 100 at the starting curve


def build_fund(f: Fund, cf=None):
    """Scale liabilities to 100. Gilts are the remainder of assets; the overlay (swap / repo-financed gilts)
    tops hedging exposure up to the target hedge ratio. Hedging assets are liability-matching: their value moves
    with liabilities, lam(curve) = L(curve) / L(start curve)."""
    cf = liability_cashflows() if cf is None else cf
    base = CURVES[f.curve_id]
    cf = cf * (f.liab_scale / pv_curve(cf, base))
    L0, A0 = f.liab_scale, f.fr0 * f.liab_scale
    cash, eq = f.cash * A0, f.equity * A0
    gilts = A0 - cash - eq
    if gilts < -1e-9:
        raise ValueError("cash + equity exceed total assets")
    overlay = max(0.0, f.hedge_ratio * L0 - gilts)
    lam = lambda ycc: pv_curve(cf, ycc) / L0
    return dict(cf=cf, base=base, L0=L0, A0=A0, cash=cash, equity=eq, gilts=gilts, overlay=overlay,
                hedge_achieved=(gilts + overlay) / L0, dur_L=dur_curve(cf, base), lam=lam)


def make_fund(curve_id, gilt_share, cash, fr0=START_FR, hedge=HEDGE):
    """Fund whose hedging assets (gilts and index-linked gilts) are a fixed share of assets and whose cash buffer is
    `cash`; everything else is return-seeking (equities, corporate bonds, other). The overlay tops hedging up to `hedge`."""
    return Fund(fr0=fr0, curve_id=curve_id, cash=cash, equity=1.0 - gilt_share - cash, hedge_ratio=hedge)


def parallel_add_on(lam, ycc, target, lo=-0.05, hi=0.50):
    """Parallel add-on to the curve at which the liability-matching price factor equals `target`."""
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if lam(ycc + mid) < target else (mid, hi)
    return (lo + hi) / 2


# =====================================================================================
# 4. SHOCK - the 2022 curve move, or a parallel rise / fall
# =====================================================================================
@dataclass(frozen=True)
class Shock:
    dy_bp: float = 250.0         # parallel move in basis points (continuously compounded yield)
    equity: float = 0.0          # optional equity price change (not used in the base case)
    path: str = "parallel"       # "parallel" or "actual2022" (Bank curve move 22 -> 27 Sept 2022, by maturity)


def shock_vector(s: Shock, n=CF_LEN):
    return MOVE_2022[:n] if s.path == "actual2022" else np.full(n, s.dy_bp / 1e4)


# =====================================================================================
# 5. LIQUIDITY - Cont and Schaanning fire-sale cascade, single fund
# =====================================================================================
@dataclass(frozen=True)
class Market:
    depth_equity: float = DEPTH_EQUITY   # market depth D, in multiples of fund assets: Psi(q) = q / D   (their eq. 11)
    depth_gilt: float = DEPTH_GILT       # stress parameter: not observable; varied in the sensitivity tables
    alpha: float = ALPHA         # implementation shortfall weight (their eq. 15)
    tc_equity: float = TC_EQUITY # equity trading cost 0.5% (UK stamp-duty rate, used as a cost proxy)
    tc_gilt: float = TC_GILT     # gilt trading cost: +7bp in stress (Pinter, 2023)
    rule: str = SALE_RULE        # "pecking": cash -> equities -> gilts ; "proportional": pro-rata (their eq. 8-10)
    max_rounds: int = 25


def _net(q, depth, alpha, tc):
    """Net proceeds from selling q at current prices: q (1 - alpha q/D - tc)   [Cont-Schaanning eq. 15]."""
    return q * (1 - alpha * q / depth - tc)


def _raise_cash(need, hold, depth, alpha, tc):
    """Gross volume q (<= hold) whose net proceeds equal `need`; solves the quadratic in q."""
    if need <= 0 or hold <= 0:
        return 0.0, 0.0
    a, b = alpha / depth, 1 - tc
    disc = b * b - 4 * a * need
    q = hold if disc < 0 else min(hold, (b - np.sqrt(disc)) / (2 * a))
    return q, _net(q, depth, alpha, tc)


def run_cascade(f: Fund, s: Shock, m: Market = Market(), cf=None, record=False):
    """Apply the shock, meet the collateral call from cash, then sell assets round by round until the call is met
    or marketable assets are exhausted (Cont and Schaanning, Section 2)."""
    p = build_fund(f, cf)
    A0, L0, lam, N, base = p["A0"], p["L0"], p["lam"], p["overlay"], p["base"]
    y = base + shock_vector(s, len(base))                  # current curve (shock plus any price-impact add-on)
    pe = 1 + s.equity
    units_g, units_e = p["gilts"], p["equity"]              # holdings in base-price units
    cash, posted = p["cash"], 0.0
    D_e, D_g = m.depth_equity * A0, m.depth_gilt * A0
    liab = lambda yy: pv_curve(p["cf"], yy)

    def state():
        pg = lam(y)
        v_over = N * (pg - 1)
        return pg, v_over, cash + units_e * pe + units_g * pg + v_over + posted

    pg, v_over, assets = state()
    out = dict(L0=L0, A0=A0, FR0=A0 / L0, overlay=N, hedge=p["hedge_achieved"], dur_L=p["dur_L"],
               FR_shock=assets / liab(y), call0=max(0.0, -v_over))
    log, sold_e, sold_g, loss_mtm, loss_is, cost, status, k = [], 0.0, 0.0, 0.0, 0.0, 0.0, "No forced sales", 0
    for k in range(1, m.max_rounds + 1):
        pg, v_over, _ = state()
        call = max(0.0, -v_over) - posted                  # collateral still to be posted
        pay = min(cash, max(call, 0.0))                    # 1) cash buffer first
        cash -= pay
        posted += pay
        short = call - pay
        if short <= 1e-4:
            k -= 1
            break
        status = "Forced sales"
        val_e, val_g = units_e * pe, units_g * pg
        if m.rule == "pecking":                            # 2a) cash -> equities -> gilts
            q_e, net_e = _raise_cash(short, val_e, D_e, m.alpha, m.tc_equity)
            q_g, net_g = _raise_cash(short - net_e, val_g, D_g, m.alpha, m.tc_gilt)
        else:                                              # 2b) pro-rata deleveraging (their eq. 8-10)
            lo, hi = 0.0, 1.0
            for _ in range(60):
                mid = (lo + hi) / 2
                tot = _net(val_e * mid, D_e, m.alpha, m.tc_equity) + _net(val_g * mid, D_g, m.alpha, m.tc_gilt)
                lo, hi = (mid, hi) if tot < short else (lo, mid)
            q_e, q_g = val_e * hi, val_g * hi
            net_e, net_g = _net(q_e, D_e, m.alpha, m.tc_equity), _net(q_g, D_g, m.alpha, m.tc_gilt)
        psi_e, psi_g = q_e / D_e, q_g / D_g                # market impact Psi(q) = q / D   (eq. 11)
        loss_mtm += (val_e - q_e) * psi_e + (val_g - q_g) * psi_g      # mark-to-market loss on what is kept (eq. 14)
        loss_is += m.alpha * (q_e * psi_e + q_g * psi_g)               # implementation shortfall (eq. 15)
        cost += q_e * m.tc_equity + q_g * m.tc_gilt
        units_e -= q_e / pe
        units_g -= q_g / pg
        cash += net_e + net_g
        pe *= 1 - psi_e
        if psi_g > 0:                                      # gilt price impact = yield rise, which also revalues L
            y = y + parallel_add_on(lam, y, pg * (1 - psi_g))
        pay2 = min(cash, short)
        cash -= pay2
        posted += pay2
        sold_e += q_e
        sold_g += q_g
        pg2, v2, a2 = state()
        if record:
            log.append(dict(round=k, call=call, from_cash=pay, shortfall=short, sold_equity=q_e, sold_gilts=q_g,
                            extra_yield_bp=float(np.mean(y - base - shock_vector(s, len(base))) * 1e4),
                            funding_ratio=100 * a2 / liab(y)))
        if units_e * pe + units_g * pg2 <= 1e-9 and (max(0.0, -v2) - posted) > 1e-9:
            status = "Illiquid: assets exhausted"
            break
    pg, v_over, assets = state()
    unmet = max(0.0, max(0.0, -v_over) - posted)
    if unmet > 1e-3 and status == "Forced sales":
        status = "Illiquid: collateral unmet"
    out.update(extra_yield_bp=float(np.mean(y - base - shock_vector(s, len(base))) * 1e4), rounds=k,
               sold_equity=sold_e, sold_gilts=sold_g, sold_total_pct=100 * (sold_e + sold_g) / A0,
               loss_mtm=loss_mtm, loss_shortfall=loss_is, costs=cost,
               fire_sale_loss_pct=100 * (loss_mtm + loss_is + cost) / A0, FR_final=assets / liab(y),
               unmet=unmet, status=status)
    if record:
        out["rounds_table"] = pd.DataFrame(log)
    return out


def bp_for_lambda(lam, base, target):
    """Parallel rise (bp) at which the liability-matching price factor falls to `target`."""
    return parallel_add_on(lam, base, target) * 1e4


def resilience_bp(f: Fund, cf=None):
    """Largest parallel yield rise the cash buffer alone can absorb (no forced sales)."""
    p = build_fund(f, cf)
    if p["overlay"] <= 1e-9:
        return np.inf
    k = p["cash"] / p["overlay"]
    return np.inf if k >= 1 else bp_for_lambda(p["lam"], p["base"], 1 - k)


def buffer_for_bp(f: Fund, bp, cf=None):
    """Cash buffer (share of assets) that absorbs a parallel `bp` rise without forced sales.
    Closed form: cash = K (1 - lam) / lam, with K = hedge exposure not covered by non-cash assets."""
    p = build_fund(replace(f, cash=0.0), cf)
    lam_bp = p["lam"](p["base"] + bp / 1e4)
    K = max(0.0, f.hedge_ratio * p["L0"] - p["A0"] + p["equity"])
    cash = K * (1 - lam_bp) / lam_bp
    return cash / p["A0"] if cash <= p["A0"] - p["equity"] else np.nan


# ---- validation of the deleveraging logic against Cont and Schaanning (2017), Section 2 and Figure 1 -------
def cs_sales_fraction(eps, lam=25.0, lam_max=33.0, illiquid=0.20, lam_b=None, targeting=False):
    """Share of marketable assets sold after a loss `eps` on illiquid assets (their eq. 8-10).
    Balance sheet normalised to assets = 1: capital C = 1/lam, illiquid = `illiquid`, marketable = 1 - illiquid."""
    C, pi = 1 / lam, 1 - illiquid
    lam_b = lam_max if lam_b is None else lam_b
    loss = illiquid * eps
    C0 = max(C - loss, 0.0)
    if targeting:                                           # leverage targeting: restore lam_b for any shock
        g = (pi + illiquid * (1 - eps) - lam_b * C0) / pi
        return float(min(max(g, 0.0), 1.0))
    lev = (pi + illiquid * (1 - eps)) / C0 if C0 > 0 else np.inf
    if lev <= lam_max:
        return 0.0
    return float(min((lam_b - 1) * loss / pi + C * (lam - lam_b) / pi, 1.0))


def selftest():
    """Checks that the formulas used in the cascade reproduce Cont and Schaanning's closed forms."""
    out = []
    def check(name, ok):
        out.append((name, "pass" if ok else "FAIL"))
        assert ok, name
    check("no sales below the 5% threshold (Fig. 1, solid line)", cs_sales_fraction(0.049) == 0.0)
    check("20% of marketable assets sold at a 7.5% loss", abs(cs_sales_fraction(0.075) - 0.20) < 1e-9)
    check("40% sold at 10%", abs(cs_sales_fraction(0.10) - 0.40) < 1e-9)
    check("all sold from a 17.5% loss", abs(cs_sales_fraction(0.175) - 1.0) < 1e-9)
    check("leverage targeting is linear from zero (Gamma = 6 eps)", abs(cs_sales_fraction(0.05, lam_b=25, targeting=True) - 0.30) < 1e-9)
    check("a safety buffer (0.95 lam_max) makes the response jump at the threshold",
          cs_sales_fraction(0.0501, lam_b=0.95 * 33) > 0.05)
    # market impact and implementation shortfall (their eq. 11 and 15): net = q (1 - alpha q/D - tc)
    q, D, a_, tc = 3.0, 20.0, 0.5, 0.001
    check("net proceeds formula", abs(_net(q, D, a_, tc) - q * (1 - a_ * q / D - tc)) < 1e-12)
    qq, nn = _raise_cash(2.0, 10.0, D, a_, tc)
    check("raise-cash solver inverts the proceeds formula", abs(nn - 2.0) < 1e-9)
    # cascade terminates and conserves value: no forced sales inside the safety zone
    f = make_fund(START_2026, 0.5, 0.25)
    check("no forced sales when the buffer exceeds the call", run_cascade(f, Shock(100))["sold_total_pct"] == 0.0)
    r = run_cascade(make_fund(START_2026, 0.5, 0.0), Shock(250), Market(rule="proportional"))
    check("pro-rata cascade ends in finitely many rounds", r["rounds"] < Market().max_rounds)
    return pd.DataFrame(out, columns=["Test", "Result"])


# =====================================================================================
# 6. SCENARIOS - back-test on 2022, recommendation on today's curve, portfolio trade-off
# =====================================================================================
# Hedging assets as a share of total assets, from the parameter block (Purple Book compositions).
# Gross gilts and index-linked include exposure bought with borrowed money (shown in the Purple Book as negative cash).
GILT_GROSS_2022, GILT_GROSS_2026 = BONDS_2022 * GILTS_OF_BONDS_2022, BONDS_2025 * GILTS_OF_BONDS_2025
# Model definition (C in the Hedge_definition sheet): own-funded hedging assets = gross gilts and index-linked, less the
# borrowing that financed part of them (negative cash), plus annuities (buy-ins), which hedge the pensions and never need collateral.
G22 = GILT_GROSS_2022 - NEG_CASH_2022 + ANNUITIES_2022
G26 = GILT_GROSS_2026 - NEG_CASH_2025 + ANNUITIES_2025
SHOCKS = {"2022 curve move (22-27 Sep)": Shock(path="actual2022"),
          **{f"Parallel +{b}bp": Shock(b) for b in PARALLEL_SHOCKS_BP}}


def typical_fund(curve_id, bp, cf=None, gilt_share=None):
    """Typical fund: hedging gilts fixed at the Purple Book share, overlay up to the 85% hedge, and a cash buffer that
    gives `bp` of headroom (the rest of assets is return-seeking)."""
    g = G22 if curve_id == START_2022 else G26 if gilt_share is None else gilt_share
    p = build_fund(make_fund(curve_id, g, 0.0), cf)
    cash = p["overlay"] * (1 - p["lam"](p["base"] + bp / 1e4)) / p["A0"]
    return make_fund(curve_id, g, cash)


def typical_2022(cf=None):
    """Typical 2022 fund: cash buffer = TPR's pre-crisis headroom (150bp)."""
    return typical_fund(START_2022, HEADROOM_2022_BP, cf)


def typical_2026(cf=None):
    """Typical 2026 fund: cash buffer sized to the 250bp minimum standard."""
    return typical_fund(START_2026, STANDARD_BP, cf)


def effective_parallel_bp(shock: Shock, f: Fund, cf=None):
    """Parallel rise with the same effect on liability value as the given shock."""
    p = build_fund(f, cf)
    return bp_for_lambda(p["lam"], p["base"], p["lam"](p["base"] + shock_vector(shock, len(p["base"]))))


def run_table(f, shocks=SHOCKS, market=Market(), cf=None, label=""):
    rows = []
    for name, sh in shocks.items():
        r = run_cascade(f, sh, market, cf)
        rows.append(dict(Fund=label, Shock=name, Equivalent_parallel_bp=effective_parallel_bp(sh, f, cf),
                         Cash_pct=100 * f.cash, Overlay_pct=100 * r["overlay"] / r["A0"],
                         Collateral_call_pct=100 * r["call0"] / r["A0"], Forced_sales_pct=r["sold_total_pct"],
                         Fire_sale_loss_pct=r["fire_sale_loss_pct"], FR_after_shock_pct=100 * r["FR_shock"],
                         FR_final_pct=100 * r["FR_final"], Status=r["status"]))
    return pd.DataFrame(rows)


def backtest_2022(cf=None):
    """(a) Typical 2022 fund under three shocks, forced sales scaled to the hedged-liability stock.
    (b) Reverse calibration: the cash buffer (and headroom in bp) implied by Pinter's 36bn of net sales."""
    f = typical_2022(cf)
    t = run_table(f, {k: SHOCKS[k] for k in list(SHOCKS)[:3]}, cf=cf, label=f"Typical 2022 fund ({HEADROOM_2022_BP}bp headroom)")
    t["Implied_sector_sales_bn"] = t.Forced_sales_pct / 100 * HEDGED_LIAB_BN
    target = 100 * SALES_BN / HEDGED_LIAB_BN
    rows = []
    for name in (list(SHOCKS)[1], list(SHOCKS)[0]):         # first parallel shock (160bp) and the actual curve move
        sh = SHOCKS[name]
        sales = lambda c: run_cascade(make_fund(START_2022, G22, c), sh, cf=cf)["sold_total_pct"]
        if sales(0.0) < target:
            rows.append(dict(Shock=name, Implied_cash_pct=np.nan, Implied_headroom_bp=np.nan))
            continue
        lo, hi = 0.0, f.cash
        for _ in range(50):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if sales(mid) > target else (lo, mid)
        c = (lo + hi) / 2
        rows.append(dict(Shock=name, Implied_cash_pct=100 * c, Implied_headroom_bp=resilience_bp(make_fund(START_2022, G22, c), cf)))
    return t, pd.DataFrame(rows), f


def tradeoff_grid(cf=None, curve=START_2026, bp=STANDARD_BP, fr0=START_FR, hedges=None, equities=None):
    """Cash buffer needed for the minimum standard, and the funding ratio after a fall, by equity share and hedge ratio."""
    hedges = np.round(np.arange(0.60, 1.001, 0.01), 2) if hedges is None else hedges
    equities = np.round(np.arange(0.0, 0.501, 0.01), 2) if equities is None else equities
    rows = []
    for h in hedges:
        for e in equities:
            f0 = Fund(curve_id=curve, fr0=fr0, equity=e, hedge_ratio=h, cash=0.0)
            c = buffer_for_bp(f0, bp, cf)
            if not np.isfinite(c) or e + c > 1.0:
                rows.append(dict(hedge=h, equity=e, cash=np.nan))
                continue
            f = replace(f0, cash=c)
            p = build_fund(f, cf)
            if p["overlay"] <= 1e-9 and h < p["gilts"] / p["L0"] - 1e-9:   # target below physical gilts: not distinct
                rows.append(dict(hedge=h, equity=e, cash=np.nan))
                continue
            fr_down = run_cascade(f, Shock(FALL_BP), cf=cf)["FR_shock"]
            rows.append(dict(hedge=h, equity=e, cash=c, gilts=1 - e - c, overlay=p["overlay"] / p["A0"],
                             hedge_achieved=p["hedge_achieved"], fr_after_fall=fr_down))
    return pd.DataFrame(rows)


def equity_capacity(grid, budgets=CASH_BUDGETS, hedges=HEDGES_SHOWN):
    """Largest return-seeking share a fund can hold at each hedge ratio if the cash buffer for the 250bp standard
    may not exceed the given budget. The budget is a trustee choice, not an assumption."""
    rows = []
    for b in budgets:
        for h in hedges:
            g = grid[(grid.hedge == h) & grid.cash.notna() & (grid.cash <= b + 1e-9)]
            if g.empty:
                rows.append(dict(Cash_budget=b, Hedge=h, Max_return_seeking=np.nan))
                continue
            top = g.loc[g.equity.idxmax()]
            rows.append(dict(Cash_budget=b, Hedge=h, Max_return_seeking=top.equity, Cash_needed=top.cash, Gilts=top.gilts,
                             Overlay=top.overlay, FR_after_fall=top.fr_after_fall))
    return pd.DataFrame(rows)


def buffer_sensitivity(cf=None):
    """Cash needed (% of assets) at 160bp and 250bp: by starting curve, funding ratio and hedging-gilt share."""
    rows = []
    for cid in (START_2022, START_2026):
        dur = build_fund(make_fund(cid, G22, 0.0), cf)["dur_L"]
        for fr0 in sorted(set((START_FR,) + ROBUST_FR)):
            for gname, g in ((f"2022 mix ({100 * G22:.1f}% hedging assets)", G22), (f"2025 mix ({100 * G26:.1f}% hedging assets)", G26)):
                for bp in (160, 250):
                    p = build_fund(make_fund(cid, g, 0.0, fr0=fr0), cf)
                    cash = p["overlay"] * (1 - p["lam"](p["base"] + bp / 1e4)) / p["A0"]
                    rows.append(dict(Start_curve=cid, Liability_duration=dur, Start_funding_ratio=fr0, Asset_mix=gname,
                                     Shock_bp=bp, Overlay_pct=100 * p["overlay"] / p["A0"], Cash_needed_pct=100 * cash))
    return pd.DataFrame(rows)


def cascade_sensitivity(cf=None, shock=250):
    """Robustness of forced sales and losses to the least observable parameters: typical 2026 mix with a thin 2% buffer."""
    f = make_fund(START_2026, G26, 0.02)
    rows = []
    for rule in ("pecking", "proportional"):
        for mult in ILLIQUIDITY_MULTIPLES:
            for tcg in GILT_COST_SENSITIVITY:
                m = Market(depth_equity=DEPTH_EQUITY / mult, depth_gilt=DEPTH_GILT / mult, rule=rule, tc_gilt=tcg)
                r = run_cascade(f, Shock(shock), m, cf)
                rows.append(dict(Rule=rule, Illiquidity_multiple=mult, Gilt_cost_bp=1e4 * tcg,
                                 Forced_sales_pct=r["sold_total_pct"], Fire_sale_loss_pct=r["fire_sale_loss_pct"],
                                 Extra_yield_bp=r["extra_yield_bp"], FR_final_pct=100 * r["FR_final"], Rounds=r["rounds"]))
    return pd.DataFrame(rows)


def curve_facts(cf=None):
    cf = liability_cashflows() if cf is None else cf
    rows = []
    for cid in (START_2022, START_2026):
        row = BOE.loc[cid]
        p = build_fund(Fund(curve_id=cid), cf)
        rows.append(dict(Date=cid, **{f"{m:g}y": row[m] for m in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0)},
                         Liability_duration=p["dur_L"]))
    mv = (BOE.loc[PEAK_2022] - BOE.loc[START_2022]) * 100
    rows.append(dict(Date="Change 22-27 Sep 2022 (bp)", **{f"{m:g}y": mv[m] for m in (1.0, 5.0, 10.0, 20.0, 30.0, 40.0)}))
    return pd.DataFrame(rows)


def legacy_regression(cf=None):
    """New engine on a flat 4% curve reproduces the previous version (flat curve, 30% equities, 2% cash)."""
    import json
    old = json.load(open("legacy_numbers.json")) if os.path.exists("legacy_numbers.json") else None
    if old is None:
        return pd.DataFrame()
    f = Fund(fr0=1.0, curve_id="flat4", equity=0.30, hedge_ratio=0.85, cash=0.02)
    cc_bp = 1e4 * (np.log(1.065) - np.log(1.04))              # annual 4% -> 6.5% expressed in continuous yields
    m = Market(depth_equity=4.0, depth_gilt=6.0, alpha=0.5, tc_equity=0.005, tc_gilt=0.001, rule="pecking")   # legacy settings
    r = run_cascade(f, Shock(cc_bp), m, cf)
    rows = [("Collateral call (per 100)", old["pecking_250"]["call0"], r["call0"]),
            ("Forced sales (% assets)", old["pecking_250"]["sold_total_pct"], r["sold_total_pct"]),
            ("Final funding ratio", old["pecking_250"]["FR_final"], r["FR_final"]),
            ("Cash for 250bp (share of assets)", old["buffer250"],
             buffer_for_bp(replace(f, cash=0.0), cc_bp, cf))]
    t = pd.DataFrame(rows, columns=["Quantity", "Previous version", "New engine"])
    t["Difference"] = t["New engine"] - t["Previous version"]
    return t


def inflation_sensitivity():
    """Impavido's 3.5% indexation assumption is inherited; vary it (wage and pension inflation together)."""
    rows = []
    for pi in (0.025, 0.035, 0.045):
        cfi = liability_cashflows(pa=pi, pw=pi)
        for cid, bp in ((START_2022, HEADROOM_2022_BP), (START_2026, STANDARD_BP)):
            f = typical_fund(cid, bp, cfi)
            p = build_fund(f, cfi)
            rows.append(dict(Inflation=pi, Start_curve=cid, Liability_duration=p["dur_L"],
                             Overlay_pct=100 * p["overlay"] / p["A0"], Cash_pct=100 * f.cash,
                             Cash_for_250bp_pct=100 * buffer_for_bp(replace(f, cash=0.0), 250, cfi)))
    return pd.DataFrame(rows)


def overlay_check(cf=None):
    """Compare the model's overlay with Pinter's net LDI-PI swap and repo positions (22 Sep 2022)."""
    f = typical_2022(cf)
    p = build_fund(f, cf)
    total_assets_bn = HEDGED_LIAB_BN / HEDGE            # liabilities of GBP 1.4tn are hedged at 85%; assets = liabilities
    rows = [("Model overlay (swaps + repo-financed gilts)", p["overlay"] / p["A0"] * total_assets_bn),
            ("Pinter: net OIS receive-fixed + gilt repo borrowing", PINTER_OIS_BN + PINTER_REPO_BN),
            ("Pinter: the above plus inflation swaps", PINTER_OIS_BN + PINTER_REPO_BN + PINTER_INFL_BN)]
    return pd.DataFrame(rows, columns=["Measure", "GBP bn"])


def parameter_register():
    R = [
        ("Liability data and method", "Impavido plan; PBO constant dollar", "Sourced: Impavido (2011/2013) and IMF Model.xlsx; replication RMSE 0.03pp"),
        ("Starting curves", f"BoE nominal spot, {START_2022} and {START_2026}", "Sourced: Bank of England; continuously compounded (Bank yield-curve FAQ)"),
        ("Shock: 2022 move", f"Bank curve change {START_2022} to {PEAK_2022} (about +90 to +113bp)", "Derived from Bank of England data"),
        ("Shock: severe rise", ", ".join(f"+{b}bp" for b in PARALLEL_SHOCKS_BP), "Sourced: FCA (2023) 160bp; Bank of England (2023) 250bp minimum resilience"),
        ("Shock: fall", f"{FALL_BP}bp", "Derived: largest five-day fall in the 30-year spot yield since 2016 (-99bp); Impavido uses +/-150bp"),
        ("Hedge ratio", f"{HEDGE:.0%} of liability value", "Sourced: Bandera and Stevens (2024)"),
        ("Hedging assets (own-funded)", f"{100 * G22:.1f}% of assets (2022); {100 * G26:.1f}% (2026)", "Derived from the PPF Purple Book 2025 (weighted averages; 2022 and 2025 rows): gilts and index-linked (bonds 71.6% / 70.6% of assets; gilts and index-linked 69.8% / 65.2% of bonds), less negative cash as a proxy for the borrowing behind them (8.8% / 7.9%), plus annuities (6.8% / 12.9%)"),
        ("Hedging-asset definition sensitivity", "Four definitions, A to D", "Hedge_definition sheet: A gross gilts and index-linked; B less repo-type borrowing; C plus annuities (base); D plus corporate bonds (PPF-style)"),
        ("Return-seeking assets", "Remainder after gilts and cash (equities, corporate bonds, other)", "Derived; Purple Book equities alone (19%, 15%) would leave the fund unlevered, contradicting 2022 evidence"),
        ("Cash buffer, typical 2022 fund", f"Sized to {HEADROOM_2022_BP}bp headroom", "Sourced: TPR pre-crisis headroom (arrangements reviewed)"),
        ("Cash buffer, typical 2026 fund", f"Sized to {STANDARD_BP}bp", "Sourced: Bank of England (2023)"),
        ("Starting funding ratio", f"{START_FR:.0%}", "Judgement: neutral baseline (Purple Book s179: 103% in 2021, 125% in 2025); 125% as sensitivity"),
        ("Hedged-liability stock", f"GBP {HEDGED_LIAB_BN / 1000:.1f}tn (end-2021)", "Sourced: TPR (2022)"),
        ("Observed forced sales", f"GBP {SALES_BN:.0f}bn net gilt sales; BoE purchases GBP {BOE_BN}bn", "Sourced: Pinter (2023); Bandera and Stevens (2024)"),
        ("Implementation shortfall weight", f"{ALPHA}", "Sourced: Cont and Schaanning (2017), eq. 15"),
        ("Market impact", "Linear, Psi(q) = q / D", "Sourced: Cont and Schaanning (2017), eq. 11"),
        ("Market depth D", f"Equities {DEPTH_EQUITY:g}x, gilts {DEPTH_GILT:g}x fund assets", "PLACEHOLDER: not observable; varied 0.5x to 4x. Read as the whole sector (GBP 1.65tn), the gilt value is the same order as Cont and Schaanning's calm-market UK sovereign depth (their Table 4, whose units look like EUR trillions); depth in a stress is lower"),
        ("Gilt trading cost", f"{1e4 * TC_GILT:.0f}bp ({1e4 * GILT_COST_SENSITIVITY[-1]:.0f}bp sensitivity)", "Sourced: Pinter (2023): stress increases in trading costs"),
        ("Equity trading cost", f"{TC_EQUITY:.1%}", "Chosen proxy equal to the UK stamp-duty rate (stamp duty falls on purchases); flagged"),
        ("Sale rule", f"{SALE_RULE}: cash, then equities, then gilts; pro-rata as check", "Proposal; Cont and Schaanning (2017) eq. 8-10"),
        ("Payment timing", "Annuity due, as in the template", "Sourced: Impavido workbook"),
    ]
    return pd.DataFrame(R, columns=["Parameter", "Value", "Source or status"])


def allocation_profiles(grid):
    """The three allocations shown in the charts: for each (cash budget, hedge ratio) the largest return-seeking share
    that survives the 250bp standard without forced sales (taken from the trade-off grid)."""
    rows = []
    for name, (budget, hedge) in PROFILES.items():
        r = equity_capacity(grid, budgets=(budget,), hedges=(hedge,)).iloc[0]
        rows.append(dict(Profile=name, Cash_budget=budget, Hedge=hedge, Cash=r.Cash_needed, Gilts=r.Gilts,
                         Return_seeking=r.Max_return_seeking, Overlay=r.Overlay, FR_after_fall=r.FR_after_fall))
    return pd.DataFrame(rows)


def tornado_table(cf=None):
    """Cash needed (% of assets) for the 250bp standard by the typical 2026 fund, changing one assumption at a time."""
    def cash_needed(curve=START_2026, g=G26, fr0=START_FR, hedge=HEDGE, infl=INFL_ANNUITY):
        c = liability_cashflows(pa=infl, pw=infl) if infl != INFL_ANNUITY else cf
        p = build_fund(make_fund(curve, g, 0.0, fr0=fr0, hedge=hedge), c)
        return 100 * p["overlay"] * (1 - p["lam"](p["base"] + STANDARD_BP / 1e4)) / p["A0"]
    base = cash_needed()
    rows = [("Starting curve (2022 instead of 2026)", START_2022, START_2022, cash_needed(curve=START_2022), cash_needed(curve=START_2022)),
            ("Starting funding ratio", TORNADO_FR[0], TORNADO_FR[1], cash_needed(fr0=TORNADO_FR[0]), cash_needed(fr0=TORNADO_FR[1])),
            ("Overlay size (hedge ratio +/- 5 points; same as gilt share -/+ 5)", TORNADO_HEDGE[0], TORNADO_HEDGE[1],
             cash_needed(hedge=TORNADO_HEDGE[0]), cash_needed(hedge=TORNADO_HEDGE[1])),
            ("Indexation assumption", TORNADO_INFLATION[0], TORNADO_INFLATION[1],
             cash_needed(infl=TORNADO_INFLATION[0]), cash_needed(infl=TORNADO_INFLATION[1]))]
    t = pd.DataFrame(rows, columns=["Assumption", "Alt_1", "Alt_2", "Cash_at_alt_1_pct", "Cash_at_alt_2_pct"])
    t.insert(1, "Base_cash_pct", base)
    return t


def robustness_profiles(cf=None):
    """The three allocations re-derived under alternative settings (one change at a time).
    Each profile is a (cash budget, hedge ratio) pair; the table gives the largest return-seeking share that meets the standard."""
    scen = [("Base (2026 curve, 100% funded, 3.5% indexation, 250bp)", dict()),
            ("2022 curve", dict(curve=START_2022)),
            *[(f"Starting funding ratio {fr:.1%}".replace(".0%", "%"), dict(fr0=fr)) for fr in ROBUST_FR],
            (f"Indexation {ROBUST_INFLATION[0]:.1%}", dict(infl=ROBUST_INFLATION[0])),
            (f"Indexation {ROBUST_INFLATION[1]:.1%}", dict(infl=ROBUST_INFLATION[1])),
            (f"Standard {ROBUST_STANDARDS_BP[0]}bp", dict(bp=ROBUST_STANDARDS_BP[0])),
            (f"Standard {ROBUST_STANDARDS_BP[1]}bp", dict(bp=ROBUST_STANDARDS_BP[1]))]
    hedges = np.array(sorted({h for _, h in PROFILES.values()}))
    equities = np.round(np.arange(0.0, 0.801, 0.01), 2)
    rows = []
    for name, kw in scen:
        infl = kw.get("infl")
        cf_i = liability_cashflows(pa=infl, pw=infl) if infl else cf
        g = tradeoff_grid(cf_i, curve=kw.get("curve", START_2026), bp=kw.get("bp", STANDARD_BP), fr0=kw.get("fr0", START_FR),
                          hedges=hedges, equities=equities)
        for pname, (budget, hedge) in PROFILES.items():
            r = equity_capacity(g, budgets=(budget,), hedges=(hedge,)).iloc[0]
            rows.append(dict(Scenario=name, Profile=pname, Hedge=hedge, Cash_budget=budget, Cash_needed=r.get("Cash_needed", np.nan),
                             Return_seeking=r.get("Max_return_seeking", np.nan), Overlay=r.get("Overlay", np.nan),
                             FR_after_fall=r.get("FR_after_fall", np.nan)))   # NaN = no allocation meets the standard within the cash budget
    t = pd.DataFrame(rows)
    order = {p: i for i, p in enumerate(PROFILES)}
    ok = t.pivot(index="Scenario", columns="Profile", values="Return_seeking").fillna(-1.0)   # infeasible profile counts as lowest
    seq = list(PROFILES)
    t["Ordering_kept"] = t.Scenario.map(lambda sc: bool(all(ok.loc[sc, seq[i]] <= ok.loc[sc, seq[i + 1]] + 1e-9 for i in range(len(seq) - 1))))
    return t


def hedge_definition_table(cf=None):
    """What counts as a hedging asset drives the overlay, and the cash buffer is proportional to the overlay.
    Built from Purple Book shares: gilts and index-linked (gross of repo financing), negative cash, annuities and corporate bonds.
    2022 composition (31 March 2022) is paired with the 2022 curve (to compare with Pinter's September 2022 positions); 2025 with the 2026 curve."""
    sets = [("2022 data, 2022 curve", START_2022, GILT_GROSS_2022, NEG_CASH_2022, ANNUITIES_2022, BONDS_2022 * CORP_OF_BONDS_2022, "all corporate (upper bound)",
             HEDGED_LIAB_BN / HEDGE),        # sector assets implied by TPR's GBP 1.4tn hedged at 85%
            ("2025 data, 2026 curve", START_2026, GILT_GROSS_2026, NEG_CASH_2025, ANNUITIES_2025, BONDS_2025 * IG_CORP_OF_BONDS_2025, "investment-grade corporate",
             ASSETS_2025_BN)]                # Purple Book 2025 universe assets
    rows = []
    for label, curve, g0, neg, ann, corp, corp_label, total_assets_bn in sets:
        options = [("A. Gilts + index-linked, gross", g0),
                   ("B. A less repo financing (physically funded gilts)", g0 - neg),
                   ("C. B plus annuities (model base)", g0 - neg + ann),
                   (f"D. C plus {corp_label} bonds", g0 - neg + ann + corp)]
        for name, g in options:
            p = build_fund(make_fund(curve, g, 0.0), cf)
            ov = p["overlay"] / p["A0"]
            cash = ov * (1 - p["lam"](p["base"] + STANDARD_BP / 1e4))
            is22 = curve == START_2022
            rows.append(dict(Data=label, Definition=name, Hedging_assets_pct=100 * g, Overlay_pct=100 * ov,
                             Cash_for_250bp_pct=100 * cash, Scale_assets_bn=total_assets_bn, Implied_sector_overlay_bn=ov * total_assets_bn,
                             Pinter_net_bn_low=PINTER_OIS_BN + PINTER_REPO_BN if is22 else np.nan,
                             Pinter_net_bn_high=PINTER_OIS_BN + PINTER_REPO_BN + PINTER_INFL_BN if is22 else np.nan))
    return pd.DataFrame(rows)


GUIDE = {"Cautious": ("Closed to new benefit accrual", "Mature scheme near buy-out, weaker sponsor covenant, little tolerance for a funding shortfall"),
         "Balanced": ("Closed to new members", "Scheme still accruing benefits or mid-maturity, adequate covenant"),
         "Growth-seeking": ("Open", "Open scheme or strong covenant that can carry more risk and tie up more collateral")}


def profile_guide(prof):
    """Maps each profile to the scheme type it suits (judgement) with the Purple Book 2025 allocation of that scheme group."""
    rows = []
    for _, r in prof.iterrows():
        grp, why = GUIDE.get(r.Profile, ("", ""))
        eq, bonds, ann = PURPLE_GROUPS.get(grp, (np.nan,) * 3)
        rows.append(dict(Profile=r.Profile, Hedge=r.Hedge, Cash=r.Cash, Gilts=r.Gilts, Return_seeking=r.Return_seeking,
                         Overlay=r.Overlay, FR_after_fall=r.FR_after_fall, Suits_scheme_group=grp, Rationale_judgement=why,
                         PB2025_equities=eq, PB2025_bonds=bonds, PB2025_annuities=ann))
    return pd.DataFrame(rows)


def buffer_formula_table(cf=None):
    """Rule of thumb: buffer ~ overlay x (1 - exp(-k D dy)). The exact fall in hedge value is compared with the formula over
    liability durations (indexation 0% to 4.5%), both curves and shocks of 100 to 400bp; k is fitted by least squares."""
    rows = []
    for cid in (START_2022, START_2026):
        base = CURVES[cid]
        for pi in (0.0, 0.01, 0.02, 0.025, INFL_ANNUITY, 0.045):
            c = cf if (pi == INFL_ANNUITY and cf is not None) else liability_cashflows(pa=pi, pw=pi)
            D = dur_curve(c, base)
            for bp in (100, 160, 250, 400):
                rows.append(dict(Curve=cid, Indexation=pi, Duration=D, Shock_bp=bp, Exact_fall=1 - pv_curve(c, base + bp / 1e4) / pv_curve(c, base)))
    t = pd.DataFrame(rows)
    x = (t.Duration * t.Shock_bp / 1e4).values
    k = float((x * (-np.log(1 - t.Exact_fall.values))).sum() / (x ** 2).sum())
    t["Formula_fall"] = 1 - np.exp(-k * x)
    t["Error_points"] = 100 * (t.Formula_fall - t.Exact_fall)
    return t, k


def buffer_map_points(cf=None):
    """Model points for the buffer map: (label, duration, overlay % of assets, cash for the standard %)."""
    pts = []
    for label, cid, f in (("Typical 2022 fund", START_2022, typical_2022(cf)), ("Typical 2026 fund (base C)", START_2026, typical_2026(cf))):
        p = build_fund(f, cf)
        ov = 100 * p["overlay"] / p["A0"]
        pts.append((label, p["dur_L"], ov, ov * (1 - p["lam"](p["base"] + STANDARD_BP / 1e4))))
    hd = hedge_definition_table(cf)
    dur26 = build_fund(make_fund(START_2026, G26, 0.0), cf)["dur_L"]
    for _, r in hd[hd.Data.str.startswith("2025")].iterrows():
        tag = r.Definition[0]
        if tag != "C":
            pts.append((f"Definition {tag}", dur26, r.Overlay_pct, r.Cash_for_250bp_pct))
    return pts


def assumptions_table():
    """Thesis-ready table: each assumption, its basis category, source or rationale, and where it is tested."""
    R = [
        ("Liability cash flows (members, wages, pensions, mortality, termination; retirement 55, accrual 1%)", "Impavido plan", "Impavido",
         "Impavido (2011/2013) and the IMF workbook; replicated exactly", "Indexation only (Inflation_sensitivity); cash-flow shape not varied"),
        ("Indexation of pensions and wages 3.5%", "3.5%", "Impavido",
         "Impavido's template; inside the UK range (Purple Book: fixed 3%, RPI capped at 5%, CPI capped at 2.5% are common)", "Inflation_sensitivity, Robustness: 2.5% to 4.5%"),
        ("Valuation method: projected benefit obligation, constant dollar; annuity-due payments", "As template", "Impavido", "Impavido workbook", "Replication check"),
        ("Starting curves", f"BoE nominal spot, {START_2022} and {START_2026}", "Literature or official",
         "Bank of England; continuously compounded (Bank yield-curve FAQ)", "Buffer_sensitivity, Robustness (2022 curve)"),
        ("Shocks", "2022 curve move; +160bp; +250bp; +400bp", "Literature or official",
         "FCA (2023) 160bp; Bank of England (2023) 250bp minimum; TPR (2024) 300 to 400bp buffers; the 22 to 27 Sep 2022 Bank curve move", "Robustness (160bp, 400bp standards)"),
        ("Fall for the solvency test", f"{FALL_BP}bp", "Rational calibration",
         "Largest five-day fall in the 30-year spot yield since 2016 was -99bp (Bank data); Impavido uses +/-150bp", "Reported as an outcome, not a constraint"),
        ("Hedge ratio", f"{HEDGE:.0%} of liability value", "Literature or official", "Bandera and Stevens (2024)", "Tornado and trade-off tables: 80% to 100%"),
        ("Hedging assets (own-funded gilts and index-linked, plus annuities)", f"{100 * G22:.1f}% (2022), {100 * G26:.1f}% (2026) of assets", "Rational calibration",
         "PPF Purple Book 2025 (2022 and 2025 rows): gilts and index-linked, less negative cash as a proxy for borrowing, plus annuities", "Hedge_definition: four definitions"),
        ("Cash buffer of the typical fund", f"150bp (2022), {STANDARD_BP}bp (2026)", "Literature or official", "TPR pre-crisis headroom; FPC minimum resilience", "Back-test implied headroom; equity-capacity tables"),
        ("Starting funding ratio", f"{START_FR:.0%} (UK range 80% to 125%; Impavido 94.3%)", "Rational calibration",
         "Neutral base. Purple Book: s179 103% (2021) to 125% (2025); estimated full buy-out 74% (2021) to 96% (2025)", "Robustness: 80%, 94.3%, 110%, 125%; Tornado; Buffer_sensitivity"),
        ("Price impact and implementation shortfall", f"Linear, weight {ALPHA}", "Literature or official", "Cont and Schaanning (2017), eq. 11 and 15", "Cascade_sensitivity (pro-rata rule)"),
        ("Gilt trading cost", f"{1e4 * TC_GILT:.0f}bp ({1e4 * GILT_COST_SENSITIVITY[-1]:.0f}bp sensitivity)", "Literature or official", "Pinter (2023): increases in trading costs in stress", "Cascade_sensitivity"),
        ("Equity trading cost", f"{TC_EQUITY:.1%}", "Rational calibration",
         "Deliberately conservative stress allowance: the UK stamp-duty rate (a buyer's tax); FCA quoted spread is about 8bp. Affects about 0.04% of assets", "Immaterial to forced-sales volume"),
        ("Market depth", f"Gilts {DEPTH_GILT:g}x, equities {DEPTH_EQUITY:g}x fund assets", "Placeholder, tested",
         "Not observable; read as the sector, the gilt value is the same order as Cont and Schaanning's calm-market UK sovereign depth", "Cascade_sensitivity: 0.5x to 4x"),
        ("Back-test scale and targets", "GBP 1.4tn hedged; GBP 36bn net sales", "Literature or official", "TPR (2022); Pinter (2023); Bandera and Stevens (2024)", "Plausibility check only"),
        ("Overlay as one liability-matching exposure", "No haircuts, margin, rollover or financing cost", "Scope", "Reduced-form scope of the thesis (Chapter 1)", "Stated limitation"),
        ("Single-step shock; no recapitalisation; no returns", "Instant 250bp-type rise", "Scope", "Thesis scope; time dimension tested by bounding (forced-sales volume changes by 3% or less)", "Stated limitation"),
    ]
    return pd.DataFrame(R, columns=["Assumption", "Value", "Basis", "Source or rationale", "Where tested"])


# =====================================================================================
# 7. OUTPUT - tables and charts (slide-ready)
# =====================================================================================
def save(fig, name):
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight", dpi=200)
    plt.close(fig)


def chart_replication(tab):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4))
    for a, shock in zip(ax, ["Discount rate", "Wage inflation", "Annuity inflation"]):
        t = tab[tab.Shock == shock]
        a.plot(range(len(t)), t["Impavido (published)"], "o-", color=PALETTE["dark"], label="Impavido (published)")
        a.plot(range(len(t)), t["This model"], "s--", color=PALETTE["red"], label="This model")
        a.set_xticks(range(len(t)))
        a.set_xticklabels(t["Value"], fontsize=7, rotation=45)
        a.set_title(shock)
        a.set_ylabel("Funding ratio (%)")
    ax[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Replication of Impavido's stress tests (differences below 0.06 points)", y=1.04, fontweight="bold")
    save(fig, "fig1_replication.png")


def chart_curves():
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    mats = BOE.columns.values
    ax[0].plot(mats, BOE.loc[START_2022].values, color=PALETTE["blue"], lw=2.4, label="22 Sep 2022 (before shock)")
    ax[0].plot(mats, BOE.loc[PEAK_2022].values, color=PALETTE["red"], lw=2.4, label="27 Sep 2022 (peak)")
    ax[0].plot(mats, BOE.loc[START_2026].values, color=PALETTE["green"], lw=2.4, label="30 Sep 2026 (latest)")
    ax[0].set_xlabel("Maturity (years)")
    ax[0].set_ylabel("Spot yield (%)")
    ax[0].set_title("Bank of England nominal gilt spot curves")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].bar(mats, (BOE.loc[PEAK_2022] - BOE.loc[START_2022]).values * 100, width=0.4, color=PALETTE["red"])
    ax[1].set_xlabel("Maturity (years)")
    ax[1].set_ylabel("Change (bp)")
    ax[1].set_title("The 2022 move, 22 to 27 September: close to parallel")
    save(fig, "fig2_curves.png")


def chart_solvency_vs_liquidity(cf):
    shocks = np.arange(0, 451, 25)
    hedged = make_fund(START_2026, G26, 0.02, hedge=1.0)
    under = make_fund(START_2026, G26, 0.02, hedge=0.5)
    g = [100 * run_cascade(under, Shock(d), cf=cf)["FR_shock"] for d in shocks]
    h = [100 * run_cascade(hedged, Shock(d), cf=cf)["FR_shock"] for d in shocks]
    rr = [run_cascade(hedged, Shock(d), cf=cf) for d in shocks]
    call = [100 * r["call0"] / r["A0"] for r in rr]
    sold = [r["sold_total_pct"] for r in rr]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.1), sharex=True)
    ax[0].plot(shocks, g, color=PALETTE["green"], lw=2.6, label="Under-hedged (50% hedged)")
    ax[0].plot(shocks, h, color=PALETTE["blue"], lw=2.6, label="Fully hedged (100%)")
    ax[0].set_ylabel("Funding ratio after shock (%)")
    ax[0].set_title("Solvency: funding ratio")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    ax[1].fill_between(shocks, 0, call, color=PALETTE["light"], alpha=0.7, label="Collateral call")
    ax[1].plot(shocks, sold, color=PALETTE["red"], lw=2.6, label="Forced sales (2% cash buffer)")
    ax[1].set_ylabel("% of assets")
    ax[1].set_title("Liquidity: fully hedged fund")
    ax[1].legend(frameon=False, fontsize=8, loc="upper left")
    for a_ in ax:
        a_.set_xlabel("Parallel rise in gilt yields (bp), 2026 curve")
        lo_, hi_ = a_.get_ylim()
        for x, lab, ha in [(160, "FCA: 160bp", "right"), (250, "Minimum: 250bp", "left")]:
            a_.axvline(x, color=PALETTE["grey"], ls=":", lw=1)
            a_.text(x + (-4 if ha == "right" else 4), lo_ + (hi_ - lo_) * 0.55, lab, fontsize=8, color=PALETTE["grey"],
                    ha=ha, rotation=90, va="center")
    fig.suptitle("The same yield rise improves solvency but creates a liquidity problem", y=1.03, fontweight="bold")
    save(fig, "fig3_solvency_vs_liquidity.png")


def chart_portfolio_map(grid):
    piv = lambda col: grid.pivot(index="hedge", columns="equity", values=col)
    cash, frd = piv("cash"), piv("fr_after_fall")
    X, Y = np.meshgrid(cash.columns.values * 100, cash.index.values * 100)
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.contourf(X, Y, 100 * np.ma.masked_invalid(cash.values), levels=np.arange(0, 36, 2.5), cmap="Blues")
    fig.colorbar(im, ax=ax).set_label("Cash buffer needed for 250bp (% of assets)")
    cs = ax.contour(X, Y, 100 * np.ma.masked_invalid(frd.values), levels=[92, 94, 96, 98, 100], colors=PALETTE["red"],
                    linewidths=1.2, linestyles="--")
    ax.clabel(cs, fmt="FR %d%%", fontsize=7.5)
    t26 = typical_2026()
    ax.plot(t26.equity * 100, t26.hedge_ratio * 100, "*", ms=16, color=PALETTE["gold"], mec="black", zorder=6)
    ax.annotate(f"Typical 2026 fund\n(85% hedged, {100 * t26.equity:.0f}% return-seeking,\n{100 * t26.cash:.1f}% cash)", (t26.equity * 100, t26.hedge_ratio * 100),
                xytext=(27, 68), fontsize=8.5, bbox=dict(boxstyle="round", fc="white", ec="black"),
                arrowprops=dict(arrowstyle="-", color="black"), zorder=7)
    ax.set_xlabel("Return-seeking share of assets (%)")
    ax.set_ylabel("Hedge ratio (% of liability value)")
    ax.set_title("Portfolio trade-off on the 30 Sep 2026 curve (red dashed: funding ratio after a 100bp fall)", fontsize=10)
    save(fig, "fig4_portfolio_map.png")


def chart_equity_capacity(cap):
    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    for h, col in zip(sorted(cap.Hedge.unique()), [PALETTE["light"], PALETTE["blue"], PALETTE["gold"], PALETTE["dark"], PALETTE["red"]]):
        t = cap[cap.Hedge == h]
        ax.plot(100 * t.Cash_budget, 100 * t.Max_return_seeking, "o-", lw=2.4, color=col, label=f"Hedge ratio {h:.0%}")
    for x, lab in [(2.75, "2.75%: halves LDI effect"), (7.7, "7.7%: closes spread")]:
        ax.axvline(x, color=PALETTE["grey"], ls=":", lw=1)
        ax.text(x + 0.2, 4, lab, fontsize=7.5, color=PALETTE["grey"], rotation=90, va="bottom")
    ax.set_xlabel("Cash budget for the 250bp standard (% of assets)")
    ax.set_ylabel("Largest return-seeking share (%)")
    ax.set_title("What a cash buffer buys: return-seeking capacity by hedge ratio (2026 curve)")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.text(20, 1.5, "Benchmarks: Bandera and Stevens (2024); search grid capped at 50%", fontsize=7.5, color=PALETTE["grey"], ha="right")
    save(fig, "fig5_equity_capacity.png")


def chart_backtest(bt, implied):
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    labels = list(bt.Shock) + ["Observed: net LDI-PI\ngilt sales (Pinter)", "Observed: Bank of England\npurchases"]
    vals = list(bt.Implied_sector_sales_bn) + [SALES_BN, BOE_BN]
    cols = [PALETTE["blue"]] * len(bt) + [PALETTE["red"], PALETTE["grey"]]
    ax.bar(range(len(vals)), vals, color=cols)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}", ha="center", fontsize=8)
    import textwrap
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels([textwrap.fill(l.replace("\n", " "), 18) for l in labels], fontsize=7.5)
    ax.set_ylabel("GBP bn")
    ax.set_title("2022 back-test: model-implied forced sales vs observed (typical 2022 fund)")
    save(fig, "fig6_backtest.png")


def chart_cascade(cf):
    f = make_fund(START_2022, G22, 0.02)
    r = run_cascade(f, Shock(250), Market(rule="proportional"), cf, record=True)
    t = r["rounds_table"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 3.9), gridspec_kw=dict(wspace=0.35))
    p = build_fund(f, cf)
    parts = [("Cash buffer", p["cash"], PALETTE["green"]), ("Equity sales", r["sold_equity"], PALETTE["gold"]),
             ("Gilt sales", r["sold_gilts"], PALETTE["red"])]
    left = 0
    for name, v, col in parts:
        ax[0].barh(0, v, left=left, color=col, label=f"{name}: {v:.1f}")
        left += v
    ax[0].axvline(r["call0"], color=PALETTE["dark"], ls="--")
    ax[0].text(r["call0"], 0.55, f"Initial call {r['call0']:.1f}", ha="center", fontsize=8)
    ax[0].set_ylim(-0.6, 0.8)
    ax[0].set_yticks([])
    ax[0].set_xlabel("Cash raised (per 100 of liabilities)")
    ax[0].set_title("Sources of cash (+250bp, 2% buffer)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3)
    if len(t):
        ax[1].bar(t["round"], t["sold_equity"] + t["sold_gilts"], color=PALETTE["blue"])
        ax[1].set_xlabel("Deleveraging round")
        ax[1].set_ylabel("Forced sales (per 100 of liabilities)")
        ax[1].set_title("Fire-sale cascade by round (pro-rata rule)")
    save(fig, "fig7_cascade.png")


def chart_cs_figure1():
    eps = np.linspace(0, 0.25, 501)
    fig, ax = plt.subplots(figsize=(7.4, 4.3))
    ax.plot(eps * 100, [100 * cs_sales_fraction(e, lam_b=25, targeting=True) for e in eps], ls="--", color=PALETTE["grey"],
            label="Leverage targeting (earlier literature)")
    ax.plot(eps * 100, [100 * cs_sales_fraction(e) for e in eps], color=PALETTE["blue"], lw=2.6, label="Constraint, target = limit")
    ax.plot(eps * 100, [100 * cs_sales_fraction(e, lam_b=0.95 * 33) for e in eps], ".", ms=3, color=PALETTE["red"],
            label="Constraint, 5% safety buffer")
    ax.axvline(5, color="black", ls=":", lw=1)
    ax.text(5.3, 90, "5% threshold", fontsize=8)
    ax.set_xlabel("Loss on illiquid assets (%)")
    ax.set_ylabel("Marketable assets sold (%)")
    ax.set_title("Replication of Cont and Schaanning (2017), Figure 1")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig8_cs_figure1.png")


def chart_pipeline(cf):
    f = typical_2022(cf)
    p = build_fund(f, cf)
    r = run_cascade(make_fund(START_2022, G22, 0.02), Shock(250), Market(), cf)
    lam = p["lam"](p["base"] + 0.025)
    steps = [
        (17, 45, "1  LIABILITIES", "Impavido plan data\\n(members, wages, pensions)\\n-> benefit cash flows\\n-> discounted on the gilt curve",
         f"L = 100\\nduration {p['dur_L']:.1f} years", "#D9E2F3"),
        (50, 45, "2  FUND", f"Assets = 100: cash {p['cash']:.0f} | return-seeking {p['equity']:.0f}\\ngilts {p['gilts']:.0f} + overlay {p['overlay']:.0f}",
         "hedge ratio 85%\\nof liability value", "#D9E2F3"),
        (83, 45, "3  SHOCK", "Gilt curve +250bp\\nliabilities, gilts and overlay\\nrevalued", f"liabilities {100 * lam:.1f}\\noverlay {p['overlay'] * (lam - 1):.1f}", "#F4CCCC"),
        (83, 17, "4  COLLATERAL CALL", "Overlay loss must be paid\\nin cash, now\\nbuffer pays first",
         f"call {r['call0']:.1f}\\ncash 2.0 -> shortfall {r['call0'] - 2.0:.1f}", "#F4CCCC"),
        (50, 17, "5  LIQUIDITY CASCADE", "Cont & Schaanning: sell assets,\\nprice impact q/D, shortfall alpha,\\nrepeat until the call is met",
         f"sells {r['sold_equity'] + r['sold_gilts']:.1f} of assets", "#FCE5CD"),
        (17, 17, "6  RESULTS", "Funding ratio, forced sales,\\nfire-sale loss, solvency vs\\nilliquidity",
         f"FR {100 * r['FR_shock']:.1f} -> {100 * r['FR_final']:.1f}\\nforced sales {r['sold_total_pct']:.1f}% of assets", "#D9EAD3"),
    ]
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(11, 6.2))
    ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")
    W, H = 29, 21
    for x, y, t, body, res, fc in steps:
        ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.3,rounding_size=1.5", fc=fc, ec="#444", lw=1))
        ax.text(x, y + H / 2 - 2.2, t, ha="center", va="center", fontsize=9.5, fontweight="bold")
        ax.text(x, y + 0.8, body.replace("\\n", "\n"), ha="center", va="center", fontsize=7.8)
        ax.text(x, y - H / 2 + 3.0, res.replace("\\n", "\n"), ha="center", va="center", fontsize=8, fontweight="bold", color="#1F2D3D")
    arr = lambda a, b: ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=14, lw=1.5, color="#333"))
    arr((31.8, 45), (35.2, 45)); arr((64.8, 45), (68.2, 45)); arr((83, 34.2), (83, 27.8)); arr((68.2, 17), (64.8, 17)); arr((35.2, 17), (31.8, 17))
    ax.text(50, 3.2, "Worked example: typical 2022 fund with a thin 2% buffer, 22 Sep 2022 curve, +250bp. Repeated for every scenario.",
            ha="center", fontsize=9, style="italic", color="#333")
    ax.text(50, 59.5, "Model pipeline", ha="center", fontsize=13, fontweight="bold")
    save(fig, "fig0_pipeline.png")


# ---- slide and appendix charts for the portfolio-allocation results -----------------------------
ALLOC_COLORS = {"Cash": PALETTE["green"], "Gilts": PALETTE["blue"], "Return-seeking": PALETTE["gold"]}


def chart_allocation_donuts(prof):
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.6))
    for ax, (_, r) in zip(axes, prof.iterrows()):
        vals = [100 * r.Cash, 100 * r.Gilts, 100 * r.Return_seeking]
        ax.pie(vals, colors=list(ALLOC_COLORS.values()), startangle=90, counterclock=False,
               wedgeprops=dict(width=0.38, edgecolor="white"), autopct=lambda p: f"{p:.1f}%" if p > 3 else "",
               pctdistance=0.81, textprops=dict(fontsize=8.5, color="white", fontweight="bold"))
        ax.text(0, 0, f"{r.Hedge:.0%} hedged\nFunding ratio\nafter fall {100 * r.FR_after_fall:.1f}%", ha="center", va="center", fontsize=8.5)
        ax.set_title(r.Profile, fontweight="bold", fontsize=12)
        ax.text(0, -1.32, f"plus overlay: {100 * r.Overlay:.0f}% of assets", ha="center", fontsize=9, style="italic")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in ALLOC_COLORS.values()]
    fig.legend(handles, list(ALLOC_COLORS), loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("Three allocations that meet the 250bp standard (30 Sep 2026 curve)", fontweight="bold", y=1.0)
    fig.subplots_adjust(bottom=0.12, top=0.86)
    save(fig, "fig9_allocation_donuts.png")


def chart_allocation_bars(prof):
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    x = np.arange(len(prof))
    bottom = np.zeros(len(prof))
    for col, name in (("Cash", "Cash"), ("Gilts", "Gilts"), ("Return_seeking", "Return-seeking")):
        v = 100 * prof[col].values
        ax.bar(x, v, 0.5, bottom=bottom, color=ALLOC_COLORS[name], label=name)
        for i, (vv, bb) in enumerate(zip(v, bottom)):
            ax.text(i, bb + vv / 2, f"{vv:.1f}%", ha="center", va="center", fontsize=8.5, color="white", fontweight="bold")
        bottom += v
    ov = 100 * prof["Overlay"].values
    ax.bar(x + 0.0, ov, 0.5, bottom=100, color="white", edgecolor=PALETTE["grey"], hatch="//", label="Overlay (extra hedging exposure)")
    for i, o in enumerate(ov):
        ax.text(i, 100 + o / 2, f"{o:.1f}%", ha="center", va="center", fontsize=8.5)
        ax.text(i, 100 + o + 3, f"FR after fall {100 * prof.FR_after_fall[i]:.1f}%", ha="center", fontsize=8.5, color=PALETTE["dark"])
    ax.axhline(100, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.Profile}\n{r.Hedge:.0%} hedged" for _, r in prof.iterrows()])
    ax.set_ylabel("% of assets")
    ax.set_ylim(0, 185)
    ax.set_title("Asset mix and overlay by profile (above 100% = overlay exposure)")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    save(fig, "fig10_allocation_stacked.png")


def chart_tornado(t):
    t = t.copy()
    base = t.Base_cash_pct.iloc[0]
    t["lo"] = t[["Cash_at_alt_1_pct", "Cash_at_alt_2_pct"]].min(axis=1)
    t["hi"] = t[["Cash_at_alt_1_pct", "Cash_at_alt_2_pct"]].max(axis=1)
    t["range"] = t.hi - t.lo
    t = t.sort_values("range")
    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    for i, (_, r) in enumerate(t.iterrows()):
        ax.barh(i, r.lo - base, left=base, color=PALETTE["green"], height=0.55) if r.lo < base - 1e-9 else None
        ax.barh(i, r.hi - base, left=base, color=PALETTE["red"], height=0.55) if r.hi > base + 1e-9 else None
        if r.lo < base - 1e-9:
            ax.text(r.lo - 0.1, i, f"{r.lo:.1f}", ha="right", va="center", fontsize=8)
        if r.hi > base + 1e-9:
            ax.text(r.hi + 0.1, i, f"{r.hi:.1f}", ha="left", va="center", fontsize=8)
    ax.axvline(base, color="black", lw=1)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels(t.Assumption, fontsize=8.5)
    ax.set_xlabel("Cash needed for the 250bp standard (% of assets)")
    ax.set_title(f"What moves the buffer? Typical 2026 fund, base {base:.1f}%")
    ax.set_xlim(t.lo.min() - 1.8, t.hi.max() + 1.8)
    ax.text(0.99, 0.03, "green: less cash needed | red: more cash needed", transform=ax.transAxes, ha="right", fontsize=8, color=PALETTE["grey"])
    save(fig, "fig11_tornado.png")


def chart_tradeoff_slide(cap, prof):
    fig, ax = plt.subplots(figsize=(8, 4.6))
    cols = {0.80: PALETTE["light"], 0.85: PALETTE["blue"], 0.90: PALETTE["gold"], 1.00: PALETTE["dark"]}
    ends = []
    for h in sorted(cap.Hedge.unique()):
        t = cap[cap.Hedge == h]
        ax.plot(100 * t.Cash_budget, 100 * t.Max_return_seeking, "-", lw=2.6, color=cols.get(h, PALETTE["grey"]))
        ends.append([h, 100 * t.Cash_budget.iloc[-1] + 0.3, 100 * t.Max_return_seeking.iloc[-1]])
    ends.sort(key=lambda e: (e[2], -e[0]))
    for i in range(1, len(ends)):                      # keep end labels apart where lines meet
        ends[i][2] = max(ends[i][2], ends[i - 1][2] + 3.0)
    for h, x_, y_ in ends:
        ax.text(x_, y_, f"{h:.0%} hedged", va="center", fontsize=9, color=cols.get(h, PALETTE["grey"]), fontweight="bold")
    ax.set_ylim(0, 58)
    ax.text(1.8, 55.5, "(search grid capped at 50%)", ha="left", fontsize=7.5, color=PALETTE["grey"])
    for _, r in prof.iterrows():
        ax.plot(100 * r.Cash_budget, 100 * r.Return_seeking, "o", ms=10, color="white", mec="black", mew=1.6, zorder=5)
        ax.annotate(r.Profile, (100 * r.Cash_budget, 100 * r.Return_seeking), xytext=(6, -14), textcoords="offset points", fontsize=8.5,
                    bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9), zorder=6)
    ax.set_xlabel("Cash held for the 250bp standard (% of assets)")
    ax.set_ylabel("Largest return-seeking share (%)")
    ax.set_xlim(1.5, 23)
    ax.set_title("More growth assets need more cash; more hedging leaves less room for growth")
    save(fig, "fig12_tradeoff_slide.png")


def chart_answer_table(guide):
    fig, ax = plt.subplots(figsize=(12.5, 3.8))
    ax.axis("off")
    cols = ["Profile", "Hedge", "Cash", "Gilts", "Return-\nseeking", "Overlay", "Funding ratio\nafter 100bp fall", "Suits (judgement)", "Purple Book group:\nequities / annuities"]
    short = {"Closed to new benefit accrual": "Mature, closed to accrual", "Closed to new members": "Closed to new members", "Open": "Open schemes"}
    cells = []
    for _, r in guide.iterrows():
        cells.append([r.Profile, f"{r.Hedge:.0%}", f"{100 * r.Cash:.1f}%", f"{100 * r.Gilts:.1f}%", f"{100 * r.Return_seeking:.0f}%",
                      f"{100 * r.Overlay:.0f}%", f"{100 * r.FR_after_fall:.1f}%", short.get(r.Suits_scheme_group, r.Suits_scheme_group),
                      f"{100 * r.PB2025_equities:.1f}% / {100 * r.PB2025_annuities:.1f}%"])
    widths = [0.12, 0.06, 0.07, 0.07, 0.08, 0.07, 0.12, 0.2, 0.17]
    tb = ax.table(cellText=cells, colLabels=cols, colWidths=widths, loc="center", cellLoc="center")
    tb.auto_set_font_size(False)
    tb.set_fontsize(9)
    tb.scale(1, 2.4)
    for (i, j), c in tb.get_celld().items():
        if i == 0:
            c.set_facecolor(PALETTE["dark"]); c.set_text_props(color="white", fontweight="bold")
        else:
            c.set_facecolor(["#F2F2F2", "white", "#F2F2F2"][(i - 1) % 3])
    ax.set_title("The answer on one page: allocations that meet the 250bp standard (30 Sep 2026 curve)", fontweight="bold", fontsize=11, y=0.98)
    ax.text(0.5, 0.0, "Weights are % of assets; overlay is extra hedging exposure. The scheme group is a judgement; the Purple Book columns show equities and annuities only,\nso they are not like-for-like with return-seeking assets.",
            ha="center", fontsize=7.5, color=PALETTE["grey"], transform=ax.transAxes)
    save(fig, "fig13_answer_table.png")


def chart_hedge_waterfall():
    """How the model's hedging assets and overlay are built from the Purple Book 2025 numbers (per 100 of assets)."""
    gross, neg, ann = 100 * GILT_GROSS_2026, 100 * NEG_CASH_2025, 100 * ANNUITIES_2025
    own = gross - neg + ann
    over = 100 * HEDGE - own
    p = build_fund(make_fund(START_2026, G26, 0.0))
    fall = 1 - p["lam"](p["base"] + STANDARD_BP / 1e4)
    cash = over * fall
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    labels = ["Gilts and\nindex-linked\n(gross)", "less: borrowing\nbehind them\n(negative cash)", "plus:\nannuities", "= own-funded\nhedging assets", "Overlay needed\n(85 minus own-\nfunded)", "Hedge target\n(85% of\nliabilities)"]
    ax.bar(0, gross, color=PALETTE["blue"])
    ax.bar(1, neg, bottom=gross - neg, color=PALETTE["red"])
    ax.bar(2, ann, bottom=gross - neg, color=PALETTE["green"])
    ax.bar(3, own, color=PALETTE["dark"])
    ax.bar(4, over, bottom=own, color="white", edgecolor=PALETTE["grey"], hatch="//")
    ax.bar(5, own, color=PALETTE["dark"]); ax.bar(5, over, bottom=own, color="white", edgecolor=PALETTE["grey"], hatch="//")
    for x, v, y in [(0, f"{gross:.1f}", gross / 2), (1, f"-{neg:.1f}", gross - neg / 2), (2, f"+{ann:.1f}", gross - neg + ann / 2), (3, f"{own:.1f}", own / 2),
                    (4, f"{over:.1f}", own + over / 2), (5, f"{100 * HEDGE:.0f}", 100 * HEDGE + 2)]:
        ax.text(x, y, v, ha="center", va="center", fontsize=10, fontweight="bold", color="white" if x in (0, 1, 3) else "black")
    ax.set_xticks(range(6))
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("per 100 of the fund's assets")
    ax.set_ylim(0, 105)
    ax.set_title("From the Purple Book to the overlay (2025 data)")
    ax.text(-0.35, 100, f"Cash buffer for a 250bp rise\n= overlay x {100 * fall:.1f}% fall in hedge value\n= {over:.1f} x {fall:.3f} = {cash:.1f}", ha="left", va="top", fontsize=9.5,
            bbox=dict(boxstyle="round", fc="#FFF8E1", ec=PALETTE["gold"]))
    save(fig, "fig14_hedging_waterfall.png")


def chart_buffer_map(k, pts):
    D, O = np.meshgrid(np.linspace(10, 26, 161), np.linspace(10, 50, 161))
    Z = O * (1 - np.exp(-k * D * STANDARD_BP / 1e4))
    fig, ax = plt.subplots(figsize=(8.8, 5.4))
    im = ax.contourf(D, O, Z, levels=np.arange(0, 27, 2), cmap="Blues")
    fig.colorbar(im, ax=ax).set_label(f"Cash buffer for a {STANDARD_BP}bp rise (% of assets)")
    cs = ax.contour(D, O, Z, levels=[5, 10, 15, 20], colors="black", linewidths=0.9)
    ax.clabel(cs, fmt="%d%%", fontsize=8)
    ax.axvline(12, color=PALETTE["red"], ls="--", lw=1.4)
    ax.text(12.15, 48, "Purple Book s179\nsensitivity: about 12 years\n(PPF-level benefits)", fontsize=7.5, color=PALETTE["red"], va="top")
    for label, dur, ov, buf in pts:
        star = label.startswith("Typical 2026")
        ax.plot(dur, ov, "*" if star else "o", ms=15 if star else 8, color=PALETTE["gold"] if star else "white", mec="black", zorder=5)
        ax.annotate(f"{label}\n{buf:.1f}%", (dur, ov), xytext=(6, -28) if label.startswith("Typical 2022") else (6, 5), textcoords="offset points", fontsize=7.5,
                    bbox=dict(boxstyle="round", fc="white", ec="grey", alpha=0.9), zorder=6)
    ax.set_xlabel("Liability duration (years)")
    ax.set_ylabel("Overlay (% of assets)")
    ax.set_title(f"Buffer map: buffer = overlay x (1 - exp(-{k:.2f} x duration x {STANDARD_BP / 1e4:.3f}))", fontsize=10.5)
    save(fig, "fig15_buffer_map.png")


def main():
    cf = liability_cashflows()
    rep = replication_table()
    t123 = asset_shock_table()
    st = selftest()
    facts = curve_facts(cf)
    bt, implied, f22 = backtest_2022(cf)
    f26 = typical_2026(cf)
    rec = run_table(f26, SHOCKS, cf=cf, label="Typical 2026 fund (250bp buffer)")
    thin = run_table(make_fund(START_2026, G26, 0.02), SHOCKS, cf=cf, label="2026 mix, thin 2% buffer")
    grid = tradeoff_grid(cf)
    cap = equity_capacity(grid)
    bufs = buffer_sensitivity(cf)
    csens = cascade_sensitivity(cf)
    reg = legacy_regression(cf)
    par = parameter_register()
    infl = inflation_sensitivity()
    ovl = overlay_check(cf)
    prof = allocation_profiles(grid)
    torn = tornado_table(cf)
    rob = robustness_profiles(cf)
    hdef = hedge_definition_table(cf)
    guide = profile_guide(prof)
    bform, kfit = buffer_formula_table(cf)
    assum = assumptions_table()
    tables = {"Replication": rep, "Table_12_3": t123, "Curve_facts": facts, "Backtest_2022": bt, "Implied_headroom": implied,
              "Typical_2026": rec, "Thin_buffer_2026": thin, "Equity_capacity": cap, "Buffer_sensitivity": bufs,
              "Cascade_sensitivity": csens, "Inflation_sensitivity": infl, "Overlay_check": ovl, "Allocation_profiles": prof, "Robustness": rob, "Hedge_definition": hdef, "Profile_guide": guide, "Buffer_formula": bform, "Assumptions": assum, "Tornado": torn, "Legacy_regression": reg, "Selftests": st, "Parameters": par}
    with pd.ExcelWriter(os.path.join(OUT, "results.xlsx")) as xw:
        for k, v in tables.items():
            v.to_excel(xw, sheet_name=k[:31], index=False)
        grid.to_excel(xw, sheet_name="Tradeoff_grid", index=False)
    chart_replication(rep); chart_curves(); chart_solvency_vs_liquidity(cf); chart_portfolio_map(grid)
    chart_equity_capacity(cap); chart_backtest(bt, implied); chart_cascade(cf); chart_cs_figure1(); chart_pipeline(cf)
    chart_allocation_donuts(prof); chart_allocation_bars(prof); chart_tornado(torn); chart_tradeoff_slide(cap, prof); chart_answer_table(guide); chart_hedge_waterfall(); chart_buffer_map(kfit, buffer_map_points(cf))
    pd.set_option("display.width", 250)
    print("Replication RMSE %.3f max %.3f | Table 12.3 max diff %.3f | workbook total AL %.2f" %
          (np.sqrt((rep.Difference ** 2).mean()), rep.Difference.abs().max(), t123.Difference.abs().max(), workbook_check()))
    print(st.to_string(index=False))
    print("\nLEGACY REGRESSION\n", reg.round(4).to_string(index=False))
    print("\nCURVE FACTS\n", facts.round(2).to_string(index=False))
    print("\nTYPICAL 2022: cash %.2f%%, return-seeking %.1f%%, gilts %.1f%%" % (100 * f22.cash, 100 * f22.equity, 100 * (1 - f22.cash - f22.equity)))
    print(bt.round(2).to_string(index=False)); print(implied.round(2).to_string(index=False))
    print("\nTYPICAL 2026: cash %.2f%%, return-seeking %.1f%%, gilts %.1f%%" % (100 * f26.cash, 100 * f26.equity, 100 * (1 - f26.cash - f26.equity))); print(rec.round(2).to_string(index=False))
    print("\nTHIN BUFFER 2026\n", thin.round(2).to_string(index=False))
    print("\nEQUITY CAPACITY\n", cap.round(3).to_string(index=False))
    print("\nBUFFER SENSITIVITY\n", bufs.round(2).to_string(index=False))
    print("\nCASCADE SENSITIVITY\n", csens.round(2).to_string(index=False))
    print("\nINFLATION SENSITIVITY\n", infl.round(2).to_string(index=False))
    print("\nOVERLAY CHECK\n", ovl.round(0).to_string(index=False))
    print("\nROBUSTNESS (return-seeking share by profile and scenario)\n", rob.pivot(index="Scenario", columns="Profile", values="Return_seeking").round(3).to_string())
    print("Ordering kept in every scenario:", bool(rob.Ordering_kept.all()))
    print("\nHEDGING-ASSET DEFINITION\n", hdef.round(1).to_string(index=False))
    print("\nBUFFER FORMULA: k = %.3f; error range %.2f to %.2f points of hedge value (mean abs %.2f)" % (kfit, bform.Error_points.min(), bform.Error_points.max(), bform.Error_points.abs().mean()))
    print("\nMODEL VERSION:", MODEL_VERSION)


if __name__ == "__main__":
    main()
