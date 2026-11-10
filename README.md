# LDI liquidity stress test

Python code and input data for the MSc thesis *How Much Liquidity Is Enough? Cash Buffers and Portfolio Allocation in
Liability-Driven Investment, Calibrated to the 2022 UK Gilt Market Crisis* (Syed Hassan Raza Zaidi, University of Siena).

The model takes a stylised UK defined-benefit pension fund, applies a rise in gilt yields, and tests whether the fund can
meet the collateral call on its leveraged hedge without forced asset sales. It replicates Impavido (2011) for the
liabilities and uses the forced-sale cascade of Cont and Schaanning (2017). It then reports the cash buffer needed and the
trade-off between hedging, return-seeking assets and cash.

## Contents
| File | Description |
|---|---|
| `ldi_stress_test.py` | The whole model. All parameters are in one block at the top (Section 0b). |
| `data/impavido_model_input.csv` | Liability data of the plan in Impavido (2011) (sheet Input of the IMF template). |
| `data/impavido_assets_BAL.csv` | The plan's asset portfolio (sheet BAL of the IMF template). |
| `data/boe_spot_curve_2016_2026.csv` | Bank of England nominal spot curve, 2016 to 2026. |
| `legacy_numbers.json` | Benchmark values used by the regression test. |

## How to run
```
pip install -r requirements.txt
python ldi_stress_test.py
```
Run it from this folder, because the script reads its inputs by relative path. There are no random numbers, so every run
gives the same results. The script writes the results workbook `outputs/results.xlsx` and the charts in `outputs/figures/`.

## Sources of the data
- Impavido, G. (2011), *Conducting Stress Tests of Defined Benefit Pension Plans*, IMF Guide to Stress Testing, Ch. 12.
- Bank of England, nominal spot yield curve data.
- Pension Protection Fund, *The Purple Book* (asset allocations, entered as parameters in the script).
