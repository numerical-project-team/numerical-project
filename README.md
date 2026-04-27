# SABR Simulation Paper Replication

This workspace contains a Python replication scaffold for:

Choi, Hu, Kwok, "Efficient and accurate simulation of the stochastic-alpha-beta-rho model"

The current scaffold focuses on reproducing the paper's core Monte Carlo machinery:

- Algorithm 1: shifted-lognormal sampling of conditional integrated variance
- Algorithm 2: martingale-preserving CEV approximation for the conditional forward
- Algorithm 3: exact sampling of a CEV transition via a shifted-Poisson mixture gamma law
- Algorithm 4: full SABR terminal simulation over a time grid

## Presentation overview

The paper studies efficient Monte Carlo simulation of the SABR model

```math
dF_t = \sigma_t F_t^\beta\,dW_t,\qquad
\frac{d\sigma_t}{\sigma_t} = \nu\,dZ_t,\qquad
dW_t\,dZ_t = \rho\,dt.
```

The volatility step is sampled exactly:

```math
\sigma_{t+h}
=
\sigma_t
\exp\left(\nu\sqrt{h}\,X_\sigma-\frac{1}{2}\nu^2h\right),
\qquad X_\sigma\sim N(0,1).
```

The first difficult quantity is the normalized conditional average variance

```math
I_t^h
=
\frac{1}{\sigma_t^2h}
\int_t^{t+h}\sigma_s^2\,ds
\;\Bigg|\;\sigma_{t+h}.
```

Algorithm 1 approximates this with a shifted lognormal random variable:

```math
I_t^h
\approx
\mu
\left[
\frac{1}{6}
+
\frac{5}{6}
\exp\left(aX-\frac{1}{2}a^2\right)
\right],
\qquad X\sim N(0,1),
```

where

```math
\mu=\mathbb{E}[I_t^h\mid\sigma_{t+h}],\qquad
a=\sqrt{\log\left(1+\frac{36}{25}v^2\right)},\qquad
v=
\frac{\sqrt{\mathrm{Var}(I_t^h\mid\sigma_{t+h})}}
{\mathbb{E}[I_t^h\mid\sigma_{t+h}]}.
```

The second difficult quantity is the conditional forward price. For `0 < beta < 1`, Algorithm 2 uses a martingale-preserving CEV approximation:

```math
F_{t+h}\mid\sigma_{t+h},I_t^h
\approx
\mathrm{CEV}_\beta
\left(
\bar F_t^h,
(\rho^*)^2\sigma_t^2hI_t^h
\right),
\qquad
\rho^*=\sqrt{1-\rho^2},
```

with conditional mean

```math
\bar F_t^h
=
F_t
\exp\left(
\frac{\rho(\sigma_{t+h}-\sigma_t)}{\nu F_t^{1-\beta}}
-
\frac{\rho^2\sigma_t^2hI_t^h}{2F_t^{2(1-\beta)}}
\right).
```

This choice is designed so that the forward process remains close to a martingale:

```math
\mathbb{E}[F_{t+h}]\approx F_t.
```

Algorithm 3 then samples the CEV transition exactly using a Gamma-Poisson-Gamma construction. Algorithm 4 combines the steps: exact volatility step, shifted-lognormal average variance, martingale-preserving CEV approximation, exact CEV sampling, and repetition over the full time grid.

## Files

- `sabr_replicate.py`: core implementation
- `run_experiments.py`: CLI entrypoint for tables, figures, and validation
- `notebooks/paper_reproduction_walkthrough.ipynb`: presentation-oriented walkthrough with formulas, sanity checks, paper tables, figure datasets, and validation
- `notebooks/replication_sanity_checks.ipynb`: notebook with model-level sanity checks and validation summaries
- `requirements.txt`: lightweight dependency list for the replication environment

## What is implemented

- Exact volatility stepping
- PyFeng-backed conditional moment formulas for the normalized integrated variance
- PyFeng-backed shifted-lognormal parameter fitting with fixed shift `lambda = 5/6`
- Exact CEV sampling for `0 < beta < 1`
- Special handling for `beta = 1` and `|rho| = 1`
- European call pricing from Monte Carlo samples
- Table 3 parameter presets as named cases
- Direct experiment entrypoints for Tables 1, 2, 4, 5, 6, 7
- Figure 1 moment-comparison dataset
- Figure 2 / Table 7 convergence dataset
- Figure 3 comparison dataset between the paper scheme and Islah's approximation
- PyFeng analytic approximation rows where the package already provides them, plus paper-reference rows for the remaining baselines
- A 2D SABR PDE / finite-difference benchmark solver in `(F, log sigma)` coordinates
- CLI support for switching benchmark sources with `--benchmark-source paper|fdm|mc|none`
- A regression test covering the `rho = 1` Islah edge case
- A notebook with extra sanity checks outside the paper tables:
  1. `nu = 0` CEV limit
  2. `beta = 1, nu = 0` Black-Scholes limit
  3. martingale checks across maturities
  4. `|rho| = 1` Islah stability
  5. quick validation summary

## What is not implemented yet

- Direct reimplementation of competing baselines such as Euler, low-bias, PSE, Hagan, ZC Map, or Hyb ZC Map
- Variance reduction and full performance tuning

## Current status

Paper-scale validation with the built-in PDE/FDM benchmark currently reports:

- overall conclusion: `Implementation consistent with paper`
- replication conclusion: `Replication likely successful`
- martingale test: noise-dominated with no evidence of systematic drift
- Table 2: `PASS`
- Table 1: `WARNING`, with a small residual discrepancy (max relative error about `0.43%`)

This means the main replication target now looks successful, while Table 1 still shows a mild benchmark-level gap worth documenting.

## Run

Install the small dependency set first:

```powershell
python -m pip install -r .\requirements.txt
```

Then run the CLI:

```powershell
python .\run_experiments.py --experiment table1 --paper-scale
python .\run_experiments.py --experiment table1 --paper-scale --benchmark-source fdm
python .\run_experiments.py --experiment table4 --paper-scale
python .\run_experiments.py --experiment table7 --n-paths 20000 --repeats 3 --benchmark-source fdm
python .\run_experiments.py --experiment figure3 --n-paths 50000 --repeats 5 --benchmark-source fdm
python .\run_experiments.py --experiment validate --quick --benchmark-source fdm
```

You can also save any tabular output:

```powershell
python .\run_experiments.py --experiment table7 --output-csv .\outputs\table7.csv
```

Current paper-scale reproducibility commands:

```powershell
python .\run_experiments.py --experiment table1 --paper-scale --benchmark-source fdm --output-csv .\outputs\table1_paper_scale_fdm.csv
python .\run_experiments.py --experiment table2 --paper-scale --benchmark-source fdm --output-csv .\outputs\table2_paper_scale_fdm.csv
python .\run_experiments.py --experiment validate --paper-scale --benchmark-source fdm --output-csv .\outputs\validation
pytest -q .\tests\test_islah_rho1.py
```

For an explanation-oriented walkthrough rather than a test log, open:

```text
.\notebooks\paper_reproduction_walkthrough.ipynb
```

For a shorter sanity-check-only notebook, open:

```text
.\notebooks\replication_sanity_checks.ipynb
```

## Notes

- The paper's formulas were transcribed from the PDF and implemented directly.
- We now rely on [PyFENG](https://pyfeng.readthedocs.io/en/latest/) for building blocks it already exposes well: conditional average-variance moments, shifted-lognormal moment fitting, and several analytic SABR approximation models.
- The project directly reproduces the paper's proposed simulation method. Several competing baseline rows are included from PyFeng or paper-reference values rather than being fully reimplemented from scratch.
- `--benchmark-source paper` uses the tabulated paper benchmarks when they are available.
- `--benchmark-source fdm` recomputes benchmark prices with the built-in PDE/FDM solver.
- `table7` / `figure3` fall back to the internal high-resolution Monte Carlo benchmark unless `--benchmark-source fdm` is requested.
- This is now a stronger reproduction scaffold with direct table/figure entrypoints and a built-in PDE benchmark, but it is still not a finished paper-grade reproduction package.
- The saved validation CSVs are written as `validation_table1_df.csv`, `validation_table2_df.csv`, and `validation_martingale_df.csv` inside the output directory.
- The fastest path from here is:
  1. replace paper-reference baseline rows with actual baseline implementations,
  2. tune variance reduction and runtime for paper-scale sweeps,
  3. compare the new PDE benchmarks against an independent reference implementation.
