from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time
from typing import Callable, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd

try:
    import pyfeng as pf
except ImportError:  # pragma: no cover - exercised indirectly through README fallback instructions
    pf = None

EPS = 1e-14
PDF_FLOOR = 1e-300

BenchmarkProvider = Callable[[Mapping[str, float]], Optional[float]]


def dummy_benchmark(row: Mapping[str, float]) -> None:
    """Placeholder benchmark provider for validation workflows."""
    del row
    return None


table1_benchmark = {
    (-0.75, 0.2): 0.07910,
    (-0.5, 0.2): 0.07942,
    (-0.25, 0.2): 0.07969,
    (-0.75, 0.4): 0.07860,
    (-0.75, 0.6): 0.07811,
}


table2_benchmark = {
    (1.0, 0.2, 0.4): 0.07989,
    (1.0, 0.2, 0.6): 0.08002,
    (1.0, 0.2, 0.8): 0.08017,
    (1.0, 0.4, 0.8): 0.08044,
    (1.0, 0.8, 0.8): 0.08043,
    (0.75, 0.2, 0.4): 0.07998,
    (0.75, 0.2, 0.6): 0.08008,
    (0.75, 0.2, 0.8): 0.08018,
    (0.75, 0.4, 0.8): 0.08083,
    (0.75, 0.8, 0.8): 0.08276,
    (0.0, 0.2, 0.4): 0.07996,
    (0.0, 0.2, 0.6): 0.07994,
    (0.0, 0.2, 0.8): 0.07992,
    (0.0, 0.4, 0.8): 0.08068,
    (0.0, 0.8, 0.8): 0.08355,
}


TABLE4_FDM = np.array([0.84255, 0.68906, 0.40646, 0.28502, 0.18304, 0.05343, 0.01096], dtype=float)
TABLE5_FDM = np.array([0.82886, 0.66959, 0.39772, 0.29118, 0.20690, 0.10018, 0.05014], dtype=float)
TABLE6_FDM = np.array([0.04559, 0.04141, 0.03942, 0.03750, 0.03390, 0.03061], dtype=float)

TABLE4_ANALYTIC_REFERENCE = {
    "Hagan": np.array([22.35, 23.64, 17.94, 13.81, 9.38, 2.54, 0.82], dtype=float),
    "ZC Map": np.array([0.37, 0.51, 1.39, 2.29, 3.20, 4.02, 2.66], dtype=float),
    "Hyb ZC Map": np.array([4.64, 5.81, 4.07, 2.29, 0.43, -1.26, -0.30], dtype=float),
}

TABLE5_ANALYTIC_REFERENCE = {
    "Hagan": np.array([11.65, 15.94, 16.69, 14.67, 12.18, 8.56, 6.88], dtype=float),
    "ZC Map": np.array([-1.56, -1.57, 0.56, 2.37, 4.02, 5.77, 5.33], dtype=float),
    "Hyb ZC Map": np.array([3.07, 4.53, 3.86, 2.37, 1.00, -0.10, 0.52], dtype=float),
}

PYFENG_ANALYTIC_MODELS = (
    ("Hagan", "SabrHagan2002"),
    ("Choi-Wu P", "SabrChoiWu2021P"),
    ("Choi-Wu H", "SabrChoiWu2021H"),
)

TABLE6_BASELINE_REFERENCE = {
    ("Euler", 1.0 / 400.0): {"bias_x1e3": np.array([1.6, 1.5, 1.5, 1.4, 1.3, 1.2], dtype=float), "time_sec": 4.12},
    ("Euler", 1.0 / 800.0): {"bias_x1e3": np.array([0.7, 0.6, 0.5, 0.5, 0.4, 0.3], dtype=float), "time_sec": 8.26},
    ("Euler", 1.0 / 1600.0): {"bias_x1e3": np.array([-0.3, -0.3, -0.3, -0.3, -0.3, -0.3], dtype=float), "time_sec": 16.2},
    ("low-bias", 0.25): {"bias_x1e3": np.array([0.5, 0.5, 0.5, 0.4, 0.4, 0.4], dtype=float), "time_sec": 6.54},
    ("low-bias", 0.125): {"bias_x1e3": np.array([0.4, 0.4, 0.4, 0.3, 0.3, 0.2], dtype=float), "time_sec": 14.7},
    ("PSE", 1.0): {"bias_x1e3": np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=float), "time_sec": 8.19},
}

TABLE7_PAPER_REFERENCE = [
    {"n_paths": 160_000, "step": 1.0, "rms_error_x1e3": 3.27, "time_sec": 0.53},
    {"n_paths": 320_000, "step": 0.5, "rms_error_x1e3": 1.94, "time_sec": 2.27},
    {"n_paths": 640_000, "step": 0.25, "rms_error_x1e3": 1.21, "time_sec": 9.66},
    {"n_paths": 1_280_000, "step": 0.125, "rms_error_x1e3": 0.86, "time_sec": 41.35},
    {"n_paths": 2_560_000, "step": 0.0625, "rms_error_x1e3": 0.59, "time_sec": 279.53},
]

def _safe_ratio(num: np.ndarray, den: np.ndarray, fallback: float | np.ndarray) -> np.ndarray:
    """Elementwise ratio with a configurable fallback when the denominator is tiny."""
    out = np.empty_like(num, dtype=float)
    mask = np.abs(den) > PDF_FLOOR
    out[mask] = num[mask] / den[mask]
    out[~mask] = fallback if np.isscalar(fallback) else np.asarray(fallback)[~mask]
    return out


def _correlated_drift_term(
    sigma_t: np.ndarray,
    sigma_next: np.ndarray,
    nu: float,
    scale: np.ndarray,
    rho: float,
) -> np.ndarray:
    """Stable version of the correlated drift term in Eqs. (7), (15b), and (16b).

    When `nu -> 0`, the volatility path becomes deterministic so `sigma_next - sigma_t`
    vanishes. The limiting value of this term is therefore zero and should not be
    evaluated as `0 / 0`.
    """
    if abs(nu) < EPS:
        return np.zeros_like(scale, dtype=float)
    return rho * (sigma_next - sigma_t) / (nu * scale)


def _sln_w_from_skewness(skewness: np.ndarray) -> np.ndarray:
    """Recover `w = exp(sigma^2) - 1` from skewness using the cubic in Proposition 1."""
    s2 = np.asarray(skewness, dtype=float) ** 2
    x = 2.0 * np.cosh(np.arccosh(1.0 + 0.5 * s2) / 3.0)
    return np.maximum(x - 2.0, 0.0)


def _lognormal_shape_stats_from_w(w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return CV, skewness, and ex-kurtosis for a unit-mean lognormal with variance ratio `w`."""
    w = np.asarray(w, dtype=float)
    cv = np.sqrt(np.maximum(w, 0.0))
    skewness = (w + 3.0) * cv
    ex_kurtosis = w**4 + 6.0 * w**3 + 15.0 * w**2 + 16.0 * w
    return cv, skewness, ex_kurtosis


def _resolve_benchmark(
    benchmark: BenchmarkProvider | Mapping[object, float] | None,
    row: Mapping[str, float],
    keys: Sequence[str],
) -> float | None:
    """Resolve a benchmark value from either a callable provider or a keyed mapping."""
    if benchmark is None:
        return None
    if callable(benchmark):
        return benchmark(row)
    key = tuple(row[k] for k in keys)
    return benchmark.get(key)


def _martingale_conclusion_from_zscores(z_scores: Sequence[float]) -> str:
    """Classify martingale deviations as sampling noise or systematic bias."""
    z = np.abs(np.asarray(z_scores, dtype=float))
    z = z[np.isfinite(z)]
    if z.size == 0:
        return "Monte Carlo noise dominated"
    if np.all(z < 2.0):
        return "Monte Carlo noise dominated"
    if np.sum(z > 3.0) >= max(3, z.size // 2):
        return "Evidence of systematic bias"
    return "Monte Carlo noise dominated"


def _validation_label(score: int) -> str:
    if score >= 2:
        return "PASS"
    if score == 1:
        return "WARNING"
    return "FAIL"


@dataclass(frozen=True)
class SABRParams:
    f0: float
    sigma0: float
    nu: float
    rho: float
    beta: float

    @property
    def beta_star(self) -> float:
        return 1.0 - self.beta

    @property
    def rho_star(self) -> float:
        return math.sqrt(max(0.0, 1.0 - self.rho * self.rho))


@dataclass(frozen=True)
class MonteCarloConfig:
    maturity: float
    step: float
    n_paths: int
    seed: int = 12345

    @property
    def n_steps(self) -> int:
        return int(round(self.maturity / self.step))


@dataclass(frozen=True)
class FDMConfig:
    n_f: int = 180
    n_y: int = 80
    n_t: int | None = None
    n_t_per_year: int = 240
    min_n_t: int = 240
    theta: float = 0.5
    f_max: float | None = None
    y_span: float | None = None
    sigma_floor: float = 1e-4

    def resolved_n_t(self, maturity: float) -> int:
        if self.n_t is not None:
            return int(self.n_t)
        return max(self.min_n_t, int(math.ceil(self.n_t_per_year * maturity)))


def _bounded_divide(x: np.ndarray) -> np.ndarray:
    """Keep denominators away from zero inside the batched tridiagonal solver."""
    x = np.asarray(x, dtype=float)
    signs = np.where(x >= 0.0, 1.0, -1.0)
    return np.where(np.abs(x) > PDF_FLOOR, x, signs * PDF_FLOOR)


def _solve_tridiagonal_batch(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
) -> np.ndarray:
    """Solve a batch of tridiagonal systems with the Thomas algorithm.

    Each input has shape `(batch, n)` where `n` is the system size.
    """
    lower = np.asarray(lower, dtype=float)
    diag = np.asarray(diag, dtype=float)
    upper = np.asarray(upper, dtype=float)
    rhs = np.asarray(rhs, dtype=float)

    if rhs.ndim != 2:
        raise ValueError("rhs must be a 2D array with shape (batch, n)")
    batch, n = rhs.shape
    if n == 0:
        return rhs.copy()

    c_prime = np.zeros_like(rhs)
    d_prime = np.zeros_like(rhs)
    solution = np.zeros_like(rhs)

    denom0 = _bounded_divide(diag[:, 0])
    if n > 1:
        c_prime[:, 0] = upper[:, 0] / denom0
    d_prime[:, 0] = rhs[:, 0] / denom0

    for idx in range(1, n):
        denom = _bounded_divide(diag[:, idx] - lower[:, idx] * c_prime[:, idx - 1])
        if idx < n - 1:
            c_prime[:, idx] = upper[:, idx] / denom
        d_prime[:, idx] = (rhs[:, idx] - lower[:, idx] * d_prime[:, idx - 1]) / denom

    solution[:, -1] = d_prime[:, -1]
    for idx in range(n - 2, -1, -1):
        solution[:, idx] = d_prime[:, idx] - c_prime[:, idx] * solution[:, idx + 1]
    return solution


def _default_fmax(params: SABRParams, maturity: float, max_strike: float) -> float:
    """Heuristic upper F boundary for the PDE grid."""
    base_level = max(params.f0, max_strike, 1e-6)
    effective_scale = params.sigma0 * (base_level ** max(params.beta, 0.0)) * math.sqrt(max(maturity, EPS))
    effective_scale *= 1.0 + 0.5 * params.nu * math.sqrt(max(maturity, 0.0))
    candidate = max_strike + 12.0 * effective_scale + 2.0 * params.f0
    return max(4.0 * max(params.f0, max_strike), candidate, max_strike + 1.0)


def _default_y_span(params: SABRParams, maturity: float) -> float:
    """Choose a log-volatility span wide enough for the SABR benchmark cases."""
    return min(max(2.5, 4.0 * params.nu * math.sqrt(max(maturity, EPS))), 4.5)


def _prepare_fdm_environment(
    params: SABRParams,
    maturity: float,
    strikes: Sequence[float],
    config: FDMConfig | None = None,
) -> dict[str, object]:
    """Build the grids and operator coefficients used by the SABR PDE solver."""
    cfg = FDMConfig() if config is None else config
    max_strike = float(max(strikes))
    f_max = _default_fmax(params, maturity, max_strike) if cfg.f_max is None else float(cfg.f_max)
    y_span = _default_y_span(params, maturity) if cfg.y_span is None else float(cfg.y_span)

    y_center = math.log(max(params.sigma0, cfg.sigma_floor))
    y_min = math.log(max(cfg.sigma_floor, math.exp(y_center - y_span)))
    y_max = y_center + y_span

    f_grid = np.linspace(0.0, f_max, cfg.n_f + 1)
    y_grid = np.linspace(y_min, y_max, cfg.n_y + 1)
    sigma_grid = np.exp(y_grid)
    d_f = float(f_grid[1] - f_grid[0])
    d_y = float(y_grid[1] - y_grid[0])
    n_t = cfg.resolved_n_t(maturity)

    f_interior = np.maximum(f_grid[1:-1], 0.0)[None, :]
    sigma_interior = sigma_grid[1:-1][:, None]
    a_f = 0.5 * sigma_interior * sigma_interior * (f_interior ** (2.0 * params.beta))
    a_cross = params.rho * params.nu * sigma_interior * (f_interior ** params.beta)

    a_y_lower = 0.5 * params.nu * params.nu / (d_y * d_y) + 0.25 * params.nu * params.nu / d_y
    a_y_diag = -params.nu * params.nu / (d_y * d_y)
    a_y_upper = 0.5 * params.nu * params.nu / (d_y * d_y) - 0.25 * params.nu * params.nu / d_y

    return {
        "config": cfg,
        "f_grid": f_grid,
        "y_grid": y_grid,
        "sigma_grid": sigma_grid,
        "d_f": d_f,
        "d_y": d_y,
        "n_t": n_t,
        "theta": float(cfg.theta),
        "a_f": a_f,
        "a_cross": a_cross,
        "a_y_lower": float(a_y_lower),
        "a_y_diag": float(a_y_diag),
        "a_y_upper": float(a_y_upper),
        "maturity": float(maturity),
    }


def _apply_pde_boundaries(surface: np.ndarray, f_grid: np.ndarray, strike: float) -> None:
    """Apply call-option boundary conditions on the `(y, F)` PDE grid in place."""
    intrinsic = np.maximum(f_grid - strike, 0.0)
    surface[:, 0] = 0.0
    surface[:, -1] = float(np.maximum(f_grid[-1] - strike, 0.0))
    surface[0, :] = intrinsic
    if surface.shape[0] >= 2:
        surface[-1, :] = surface[-2, :]
    surface[0, 0] = 0.0
    surface[0, -1] = float(np.maximum(f_grid[-1] - strike, 0.0))
    surface[-1, 0] = 0.0
    surface[-1, -1] = float(np.maximum(f_grid[-1] - strike, 0.0))


def _apply_f_operator(surface: np.ndarray, a_f: np.ndarray, d_f: float) -> np.ndarray:
    out = np.zeros_like(surface)
    out[1:-1, 1:-1] = a_f * (
        surface[1:-1, 2:] - 2.0 * surface[1:-1, 1:-1] + surface[1:-1, :-2]
    ) / (d_f * d_f)
    return out


def _apply_y_operator(
    surface: np.ndarray,
    a_y_lower: float,
    a_y_diag: float,
    a_y_upper: float,
) -> np.ndarray:
    out = np.zeros_like(surface)
    out[1:-1, 1:-1] = (
        a_y_lower * surface[:-2, 1:-1]
        + a_y_diag * surface[1:-1, 1:-1]
        + a_y_upper * surface[2:, 1:-1]
    )
    return out


def _apply_cross_operator(surface: np.ndarray, a_cross: np.ndarray, d_f: float, d_y: float) -> np.ndarray:
    out = np.zeros_like(surface)
    out[1:-1, 1:-1] = a_cross * (
        surface[2:, 2:] - surface[2:, :-2] - surface[:-2, 2:] + surface[:-2, :-2]
    ) / (4.0 * d_f * d_y)
    return out


def _solve_f_implicit(rhs: np.ndarray, stage_surface: np.ndarray, a_f: np.ndarray, dt_theta: float, d_f: float) -> np.ndarray:
    alpha = dt_theta * a_f / (d_f * d_f)
    lower = -alpha.copy()
    diag = 1.0 + 2.0 * alpha
    upper = -alpha.copy()
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0

    rhs_adj = rhs.copy()
    rhs_adj[:, 0] += alpha[:, 0] * stage_surface[1:-1, 0]
    rhs_adj[:, -1] += alpha[:, -1] * stage_surface[1:-1, -1]
    return _solve_tridiagonal_batch(lower, diag, upper, rhs_adj)


def _solve_y_implicit(
    rhs: np.ndarray,
    stage_surface: np.ndarray,
    a_y_lower: float,
    a_y_diag: float,
    a_y_upper: float,
    dt_theta: float,
) -> np.ndarray:
    rhs_t = rhs.T.copy()
    batch, n = rhs_t.shape
    lower = np.full((batch, n), -dt_theta * a_y_lower, dtype=float)
    diag = np.full((batch, n), 1.0 - dt_theta * a_y_diag, dtype=float)
    upper = np.full((batch, n), -dt_theta * a_y_upper, dtype=float)
    lower[:, 0] = 0.0
    upper[:, -1] = 0.0

    rhs_t[:, 0] += dt_theta * a_y_lower * stage_surface[0, 1:-1]
    rhs_t[:, -1] += dt_theta * a_y_upper * stage_surface[-1, 1:-1]
    return _solve_tridiagonal_batch(lower, diag, upper, rhs_t).T


def _bilinear_interpolate(
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    values: np.ndarray,
    x0: float,
    y0: float,
) -> float:
    """Bilinear interpolation on a tensor grid."""
    ix = int(np.clip(np.searchsorted(x_grid, x0) - 1, 0, len(x_grid) - 2))
    iy = int(np.clip(np.searchsorted(y_grid, y0) - 1, 0, len(y_grid) - 2))

    x1, x2 = float(x_grid[ix]), float(x_grid[ix + 1])
    y1, y2 = float(y_grid[iy]), float(y_grid[iy + 1])
    q11 = float(values[iy, ix])
    q12 = float(values[iy + 1, ix])
    q21 = float(values[iy, ix + 1])
    q22 = float(values[iy + 1, ix + 1])

    if abs(x2 - x1) < EPS or abs(y2 - y1) < EPS:
        return q11

    wx = (x0 - x1) / (x2 - x1)
    wy = (y0 - y1) / (y2 - y1)
    return (
        (1.0 - wx) * (1.0 - wy) * q11
        + (1.0 - wx) * wy * q12
        + wx * (1.0 - wy) * q21
        + wx * wy * q22
    )


def finite_difference_call_prices(
    params: SABRParams,
    maturity: float,
    strikes: Sequence[float],
    config: FDMConfig | None = None,
) -> pd.DataFrame:
    """Price European calls with a 2D SABR finite-difference benchmark solver.

    The PDE is solved in `(F, y)` coordinates with `y = log(sigma)` using a
    Douglas-style ADI splitting with the mixed derivative kept explicit. The
    returned prices are intended as PDE/FDM
    benchmarks rather than a production-grade calibration engine.
    """
    if maturity <= 0.0:
        strikes_arr = np.asarray(strikes, dtype=float)
        return pd.DataFrame(
            {
                "strike": strikes_arr,
                "fdm_price": np.maximum(params.f0 - strikes_arr, 0.0),
                "maturity": maturity,
            }
        )

    env = _prepare_fdm_environment(params, maturity, strikes, config=config)
    f_grid = env["f_grid"]
    y_grid = env["y_grid"]
    d_f = float(env["d_f"])
    d_y = float(env["d_y"])
    n_t = int(env["n_t"])
    theta = float(env["theta"])
    a_f = env["a_f"]
    a_cross = env["a_cross"]
    a_y_lower = float(env["a_y_lower"])
    a_y_diag = float(env["a_y_diag"])
    a_y_upper = float(env["a_y_upper"])
    dt = float(maturity) / n_t

    prices = []
    y0 = math.log(max(params.sigma0, env["config"].sigma_floor))

    for strike in strikes:
        payoff = np.maximum(f_grid - float(strike), 0.0)
        surface = np.tile(payoff[None, :], (len(y_grid), 1))
        _apply_pde_boundaries(surface, f_grid, float(strike))

        for _ in range(n_t):
            _apply_pde_boundaries(surface, f_grid, float(strike))
            a0_surface = _apply_cross_operator(surface, a_cross, d_f, d_y)
            a1_surface = _apply_f_operator(surface, a_f, d_f)
            a2_surface = _apply_y_operator(surface, a_y_lower, a_y_diag, a_y_upper)

            y0_surface = surface + dt * (a0_surface + a1_surface + a2_surface)
            _apply_pde_boundaries(y0_surface, f_grid, float(strike))

            rhs1 = y0_surface[1:-1, 1:-1] - theta * dt * a1_surface[1:-1, 1:-1]
            y1_interior = _solve_f_implicit(rhs1, y0_surface, a_f, theta * dt, d_f)
            y1_surface = y0_surface.copy()
            y1_surface[1:-1, 1:-1] = y1_interior
            _apply_pde_boundaries(y1_surface, f_grid, float(strike))

            rhs2 = y1_surface[1:-1, 1:-1] - theta * dt * a2_surface[1:-1, 1:-1]
            y2_interior = _solve_y_implicit(
                rhs2,
                y1_surface,
                a_y_lower,
                a_y_diag,
                a_y_upper,
                theta * dt,
            )
            surface = y1_surface.copy()
            surface[1:-1, 1:-1] = y2_interior
            _apply_pde_boundaries(surface, f_grid, float(strike))

        price = _bilinear_interpolate(f_grid, y_grid, surface, params.f0, y0)
        prices.append({"strike": float(strike), "fdm_price": float(price), "maturity": float(maturity)})

    return pd.DataFrame(prices)


def finite_difference_call_price(
    params: SABRParams,
    maturity: float,
    strike: float,
    config: FDMConfig | None = None,
) -> float:
    """Single-strike convenience wrapper around `finite_difference_call_prices`."""
    return float(finite_difference_call_prices(params, maturity, [strike], config=config)["fdm_price"].iloc[0])


def fdm_benchmark_prices(
    params: SABRParams,
    maturity: float,
    strikes: Sequence[float],
    config: FDMConfig | None = None,
) -> dict[float, float]:
    """Return a simple `strike -> FDM price` mapping for experiment helpers."""
    frame = finite_difference_call_prices(params, maturity, strikes, config=config)
    return {float(row.strike): float(row.fdm_price) for row in frame.itertuples(index=False)}


def build_table1_fdm_benchmark(
    strike: float = 1.0,
    maturity: float = 1.0,
    f0: float = 1.0,
    sigma0: float = 0.2,
    config: FDMConfig | None = None,
) -> dict[tuple[float, float], float]:
    """Compute the Table 1 `(rho, nu) -> FDM price` benchmark map with the PDE solver."""
    benchmark = {}
    cases = table1_default_cases().drop_duplicates(subset=["rho", "nu"])
    for row in cases.itertuples(index=False):
        params = SABRParams(f0=f0, sigma0=sigma0, nu=float(row.nu), rho=float(row.rho), beta=1.0)
        benchmark[(float(row.rho), float(row.nu))] = finite_difference_call_price(
            params,
            maturity=maturity,
            strike=strike,
            config=config,
        )
    return benchmark


def build_table2_fdm_benchmark(
    strike: float = 1.0,
    maturity: float = 1.0,
    f0: float = 1.0,
    sigma0: float = 0.2,
    config: FDMConfig | None = None,
) -> dict[tuple[float, float, float], float]:
    """Compute the Table 2 `(rho, nu, beta) -> FDM price` benchmark map with the PDE solver."""
    benchmark = {}
    cases = table2_default_cases().drop_duplicates(subset=["rho", "nu", "beta"])
    for row in cases.itertuples(index=False):
        params = SABRParams(
            f0=f0,
            sigma0=sigma0,
            nu=float(row.nu),
            rho=float(row.rho),
            beta=float(row.beta),
        )
        benchmark[(float(row.rho), float(row.nu), float(row.beta))] = finite_difference_call_price(
            params,
            maturity=maturity,
            strike=strike,
            config=config,
        )
    return benchmark


def sample_sigma_next(
    sigma_t: np.ndarray, nu: float, h: float, rng: np.random.Generator
) -> np.ndarray:
    """Exact volatility step from Eq. (2)."""
    if abs(nu) < EPS or h <= 0.0:
        return sigma_t.copy()
    z = rng.standard_normal(size=sigma_t.shape)
    return sigma_t * np.exp(nu * math.sqrt(h) * z - 0.5 * nu * nu * h)


def conditional_integrated_variance_moments(
    sigma_t: np.ndarray,
    sigma_next: np.ndarray,
    nu: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the first four raw moments of `I_t^h`.

    We now delegate the raw-moment evaluation to `pyfeng.SabrMcTimeDisc`, which
    already implements the conditional average-variance moment formulas used in
    SABR literature. This keeps the project focused on the paper-specific
    conditional forward / CEV machinery rather than reimplementing the same
    average-variance building blocks.
    """
    _require_pyfeng()
    sigma_t = np.asarray(sigma_t, dtype=float)
    sigma_next = np.asarray(sigma_next, dtype=float)
    hat_nu = nu * math.sqrt(h)

    if abs(hat_nu) < EPS:
        ones = np.ones_like(sigma_t, dtype=float)
        return ones, ones, ones, ones

    z_hat = np.log(sigma_next / sigma_t) / hat_nu
    mu1, mu2_raw, mu3_raw, mu4_raw = pf.SabrMcTimeDisc.cond_avgvar_mvsk(
        abs(hat_nu),
        z_hat,
        mnc=True,
    )
    return (
        np.asarray(mu1, dtype=float),
        np.asarray(mu2_raw, dtype=float),
        np.asarray(mu3_raw, dtype=float),
        np.asarray(mu4_raw, dtype=float),
    )


def raw_moments_to_central_stats(
    mu1: np.ndarray,
    mu2_raw: np.ndarray,
    mu3_raw: np.ndarray,
    mu4_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert raw moments into mean, variance, std, CV, skewness, and ex-kurtosis.

    Implements the statistics defined in Remark 5 after Proposition 2. This helper
    is public and notebook-friendly for Figure 1 style moment comparisons.
    """
    mean = np.asarray(mu1, dtype=float)
    var = np.maximum(mu2_raw - mean * mean, 0.0)
    std = np.sqrt(var)
    cv = _safe_ratio(std, np.maximum(mean, PDF_FLOOR), fallback=0.0)

    centered3 = mu3_raw - 3.0 * mean * mu2_raw + 2.0 * mean**3
    centered4 = mu4_raw - 4.0 * mean * mu3_raw + 6.0 * mean * mean * mu2_raw - 3.0 * mean**4

    skewness = _safe_ratio(centered3, np.maximum(var * std, PDF_FLOOR), fallback=0.0)
    ex_kurtosis = _safe_ratio(centered4, np.maximum(var * var, PDF_FLOOR), fallback=3.0) - 3.0
    return mean, var, std, cv, skewness, ex_kurtosis


def moment_statistics_from_raw(
    mu1: np.ndarray,
    mu2_raw: np.ndarray,
    mu3_raw: np.ndarray,
    mu4_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Backward-compatible thin wrapper over `raw_moments_to_central_stats`."""
    _, var, _, cv, skewness, ex_kurtosis = raw_moments_to_central_stats(
        mu1, mu2_raw, mu3_raw, mu4_raw
    )
    return var, cv, skewness, ex_kurtosis


def sample_conditional_integrated_variance(
    sigma_t: np.ndarray,
    sigma_next: np.ndarray,
    nu: float,
    h: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample `I_t^h` with the shifted-lognormal approximation in Algorithm 1 / Eq. (9)."""
    _require_pyfeng()
    sigma_t = np.asarray(sigma_t, dtype=float)
    sigma_next = np.asarray(sigma_next, dtype=float)
    hat_nu = nu * math.sqrt(h)

    if abs(hat_nu) < EPS:
        return np.ones_like(sigma_t, dtype=float)

    z_hat = np.log(sigma_next / sigma_t) / hat_nu
    mu1, sigma_sln, lambda_sln = pf.SabrMcTimeDisc.cond_avgvar_lnshift_params(
        abs(hat_nu),
        z_hat,
        ratio=5.0 / 6.0,
    )
    x = rng.standard_normal(size=sigma_t.shape)
    return np.asarray(mu1, dtype=float) * (
        (1.0 - lambda_sln) + lambda_sln * np.exp(sigma_sln * x - 0.5 * sigma_sln * sigma_sln)
    )


def sample_cev_exact(
    f_bar: np.ndarray,
    variance_scale: np.ndarray,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact CEV transition sampler from Algorithm 3.

    This implements the shifted-Poisson mixture gamma construction summarized in
    Proposition 3 and Algorithm 3, and is reused inside Algorithm 4.
    """
    beta_star = 1.0 - beta
    if beta_star <= 0.0:
        raise ValueError("Exact CEV sampler requires 0 < beta < 1.")

    out = np.zeros_like(f_bar, dtype=float)
    # When the variance scale is negligible relative to the current level, the
    # CEV transition is effectively deterministic and should be returned as f_bar.
    small_variance = variance_scale <= 1e-12 * np.maximum(f_bar * f_bar, 1.0)
    out[small_variance] = np.maximum(f_bar[small_variance], 0.0)

    alive = (f_bar > 0.0) & (variance_scale > 0.0) & (~small_variance)
    if not np.any(alive):
        return np.where(variance_scale <= 0.0, np.maximum(f_bar, 0.0), out)

    alpha = 1.0 / (2.0 * beta_star)
    z0 = f_bar[alive] ** (2.0 * beta_star) / (beta_star * beta_star * variance_scale[alive])

    x = rng.gamma(shape=alpha, scale=1.0, size=z0.shape)
    survive = x < 0.5 * z0

    result = np.zeros_like(z0)
    if np.any(survive):
        lam = 0.5 * z0[survive] - x[survive]
        huge_lambda = lam > 1e8

        if np.any(~huge_lambda):
            pois = rng.poisson(lam[~huge_lambda])
            zt = 2.0 * rng.gamma(shape=pois + 1.0, scale=1.0)
            result[np.flatnonzero(survive)[~huge_lambda]] = (
                beta_star * beta_star * variance_scale[alive][survive][~huge_lambda] * zt
            ) ** (1.0 / (2.0 * beta_star))

        if np.any(huge_lambda):
            # Prevent Poisson overflow in the near-deterministic regime.
            result[np.flatnonzero(survive)[huge_lambda]] = np.maximum(
                f_bar[alive][survive][huge_lambda], 0.0
            )

    out[alive] = result
    out[variance_scale <= 0.0] = np.maximum(f_bar[variance_scale <= 0.0], 0.0)
    return out


def _simulate_terminal_forward_zero_vov(
    params: SABRParams,
    mc: MonteCarloConfig,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact terminal sampler for the `nu = 0` degeneracy.

    When the vol-of-vol vanishes, the SABR model reduces to a time-homogeneous
    CEV model, and the terminal law no longer depends on `rho`. Handling this
    branch explicitly avoids leaking the generic correlated-SABR approximation
    into a regime where an exact transition is already available.
    """
    if abs(params.beta) < EPS:
        raise NotImplementedError("The beta=0 normal-SABR case is not included in this starter.")

    if mc.maturity <= 0.0:
        return np.full(mc.n_paths, params.f0, dtype=float)

    variance_scale = np.full(mc.n_paths, params.sigma0 * params.sigma0 * mc.maturity, dtype=float)
    initial_forward = np.full(mc.n_paths, params.f0, dtype=float)

    if params.beta_star < 1e-8:
        z = rng.standard_normal(size=mc.n_paths)
        total_var = variance_scale
        return initial_forward * np.exp(np.sqrt(total_var) * z - 0.5 * total_var)

    return sample_cev_exact(initial_forward, variance_scale, params.beta, rng)


def _simulate_terminal_forward_scheme(
    params: SABRParams,
    mc: MonteCarloConfig,
    use_islah: bool = False,
) -> np.ndarray:
    """Terminal-price simulator for the SABR model under either the paper scheme or Islah's scheme."""
    rng = np.random.default_rng(mc.seed)
    if abs(params.nu) < EPS:
        return _simulate_terminal_forward_zero_vov(params, mc, rng)

    f = np.full(mc.n_paths, params.f0, dtype=float)
    sigma = np.full(mc.n_paths, params.sigma0, dtype=float)

    beta_star = params.beta_star
    rho_star_sq = max(0.0, 1.0 - params.rho * params.rho)

    for _ in range(mc.n_steps):
        absorbed = f <= 0.0
        sigma_next = sample_sigma_next(sigma, params.nu, mc.step, rng)
        sigma[absorbed] = sigma_next[absorbed]
        if np.all(absorbed):
            continue

        if abs(params.beta) < EPS:
            raise NotImplementedError(
                "The beta=0 normal-SABR case is not included in this starter."
            )

        sigma_alive = sigma[~absorbed]
        sigma_next_alive = sigma_next[~absorbed]
        f_alive = f[~absorbed]
        ih = sample_conditional_integrated_variance(
            sigma_alive, sigma_next_alive, params.nu, mc.step, rng
        )

        if beta_star < 1e-8:
            drift = _correlated_drift_term(
                sigma_alive,
                sigma_next_alive,
                params.nu,
                np.ones_like(f_alive),
                params.rho,
            )
            mean = f_alive * np.exp(
                drift - 0.5 * params.rho * params.rho * sigma_alive * sigma_alive * mc.step * ih
            )
            z = rng.standard_normal(size=mean.shape)
            total_var = rho_star_sq * sigma_alive * sigma_alive * mc.step * ih
            f[~absorbed] = mean * np.exp(np.sqrt(total_var) * z - 0.5 * total_var)
            sigma = sigma_next
            continue

        f_pow = np.maximum(f_alive, PDF_FLOOR) ** beta_star
        drift = _correlated_drift_term(
            sigma_alive, sigma_next_alive, params.nu, f_pow, params.rho
        )
        variance_scale = rho_star_sq * sigma_alive * sigma_alive * mc.step * ih

        if use_islah:
            beta_prime = params.beta / (1.0 - beta_star * params.rho * params.rho)
            beta_prime_star = 1.0 - beta_prime
            islah_shift = beta_star * _correlated_drift_term(
                sigma_alive,
                sigma_next_alive,
                params.nu,
                np.ones_like(f_alive),
                params.rho,
            )
            if abs(beta_prime_star) < 1e-12:
                # At |rho| = 1 we have beta_prime = 1, so the Islah transform
                # becomes deterministic and the generic power-law formulas hit
                # their removable singularity at beta_prime_star = 0.
                f_next = np.maximum(f_pow + islah_shift, 0.0) ** (1.0 / beta_star)
            else:
                transformed_mean = np.abs(
                    (beta_prime_star / beta_star) * (f_pow + islah_shift)
                ) ** (1.0 / beta_prime_star)
                if rho_star_sq < EPS:
                    y = transformed_mean
                else:
                    y = sample_cev_exact(transformed_mean, variance_scale, beta_prime, rng)
                f_next = (
                    (beta_star / beta_prime_star) * np.maximum(y, 0.0) ** beta_prime_star
                ) ** (1.0 / beta_star)
        else:
            f_bar = f_alive * np.exp(
                drift
                - 0.5 * params.rho * params.rho * sigma_alive * sigma_alive * mc.step * ih
                / np.maximum(f_pow * f_pow, PDF_FLOOR)
            )
            f_bar = np.maximum(f_bar, 0.0)

            if rho_star_sq < EPS:
                f_next = f_bar
            else:
                f_next = sample_cev_exact(f_bar, variance_scale, params.beta, rng)

        f[~absorbed] = f_next
        sigma = sigma_next

    return f


def simulate_terminal_forward(
    params: SABRParams,
    mc: MonteCarloConfig,
) -> np.ndarray:
    """Terminal-price simulator for the paper's martingale-preserving CEV approximation."""
    return _simulate_terminal_forward_scheme(params, mc, use_islah=False)


def simulate_terminal_forward_islah(
    params: SABRParams,
    mc: MonteCarloConfig,
) -> np.ndarray:
    """Terminal-price simulator using the Appendix B Islah-style conditional approximation."""
    return _simulate_terminal_forward_scheme(params, mc, use_islah=True)


def european_call_price(samples: np.ndarray, strike: float) -> float:
    """Plain Monte Carlo European call price used in the paper's numerical tables."""
    return float(np.mean(np.maximum(samples - strike, 0.0)))


def price_many_strikes(
    params: SABRParams,
    mc: MonteCarloConfig,
    strikes: Sequence[float],
    terminal_samples: np.ndarray | None = None,
) -> pd.DataFrame:
    """Price a batch of European calls for one simulation run.

    Inputs:
    - params: SABR model parameters.
    - mc: Monte Carlo configuration for this run.
    - strikes: iterable of strike values.
    - terminal_samples: optional precomputed `F_T` samples.

    Output:
    - DataFrame with columns `strike`, `price`, `mean_terminal`, `n_paths`, `step`, `maturity`.
    """
    terminal = simulate_terminal_forward(params, mc) if terminal_samples is None else terminal_samples
    rows = []
    mean_terminal = float(np.mean(terminal))
    for strike in strikes:
        rows.append(
            {
                "strike": float(strike),
                "price": european_call_price(terminal, float(strike)),
                "mean_terminal": mean_terminal,
                "n_paths": mc.n_paths,
                "step": mc.step,
                "maturity": mc.maturity,
            }
        )
    return pd.DataFrame(rows)


def repeated_pricing(
    params: SABRParams,
    mc: MonteCarloConfig,
    strikes: Sequence[float],
    n_repeats: int = 50,
    seed0: int | None = None,
    benchmark_prices: Mapping[float, float] | None = None,
    simulator: Callable[[SABRParams, MonteCarloConfig], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Repeat pricing runs and aggregate Monte Carlo error statistics.

    Inputs:
    - params: SABR model parameters.
    - mc: base Monte Carlo configuration.
    - strikes: strike grid.
    - n_repeats: number of independent runs.
    - seed0: optional seed offset. Uses `mc.seed` when omitted.
    - benchmark_prices: optional mapping `strike -> benchmark price`.
    - simulator: optional terminal simulator. Defaults to the paper's main scheme.

    Output:
    - DataFrame with per-strike aggregated columns:
      `strike`, `mean_price`, `stdev_price`, `stderr_price`, `bias`,
      `benchmark_price`, `n_repeats`, `n_paths`, `step`, `maturity`,
      plus `runtime_sec_mean`.
    """
    if seed0 is None:
        seed0 = mc.seed
    if simulator is None:
        simulator = simulate_terminal_forward

    per_run_frames = []
    runtimes = []
    for i in range(n_repeats):
        mc_i = replace(mc, seed=seed0 + i)
        t0 = time.perf_counter()
        terminal = simulator(params, mc_i)
        runtimes.append(time.perf_counter() - t0)
        frame = price_many_strikes(params, mc_i, strikes, terminal_samples=terminal)
        frame["repeat"] = i
        per_run_frames.append(frame)

    all_runs = pd.concat(per_run_frames, ignore_index=True)
    grouped = all_runs.groupby("strike", as_index=False).agg(
        mean_price=("price", "mean"),
        stdev_price=("price", "std"),
        mean_terminal=("mean_terminal", "mean"),
        stdev_terminal_estimator=("mean_terminal", "std"),
    )
    grouped["stdev_price"] = grouped["stdev_price"].fillna(0.0)
    grouped["stdev_terminal_estimator"] = grouped["stdev_terminal_estimator"].fillna(0.0)
    grouped["stderr_price"] = grouped["stdev_price"] / math.sqrt(max(n_repeats, 1))
    grouped["stderr_terminal_estimator"] = grouped["stdev_terminal_estimator"] / math.sqrt(max(n_repeats, 1))
    grouped["n_repeats"] = n_repeats
    grouped["n_paths"] = mc.n_paths
    grouped["step"] = mc.step
    grouped["maturity"] = mc.maturity
    grouped["runtime_sec_mean"] = float(np.mean(runtimes))

    if benchmark_prices is not None:
        grouped["benchmark_price"] = grouped["strike"].map(benchmark_prices)
        grouped["bias"] = grouped["mean_price"] - grouped["benchmark_price"]
        grouped["relative_error"] = grouped["bias"] / grouped["benchmark_price"]
    else:
        grouped["benchmark_price"] = np.nan
        grouped["bias"] = np.nan
        grouped["relative_error"] = np.nan
    return grouped


def martingale_test(
    params: SABRParams,
    maturities: Sequence[float],
    step: float,
    n_paths: int = 100_000,
    seed0: int = 12345,
    simulator: Callable[[SABRParams, MonteCarloConfig], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run the martingale test `E[F_T] ≈ F_0` across maturities.

    Inputs:
    - params: SABR model parameters.
    - maturities: sequence of maturities to test.
    - step: time step used for each maturity.
    - n_paths: number of Monte Carlo paths per maturity.
    - seed0: seed offset.
    - simulator: optional terminal simulator. Defaults to the paper's main scheme.

    Output:
    - DataFrame with `maturity`, `mean_terminal`, `std_terminal`, `stderr_terminal`,
      `martingale_error`, `z_score`, `conclusion`, and `runtime_sec`.
    """
    rows = []
    if simulator is None:
        simulator = simulate_terminal_forward
    for i, maturity in enumerate(maturities):
        mc = MonteCarloConfig(maturity=float(maturity), step=step, n_paths=n_paths, seed=seed0 + i)
        t0 = time.perf_counter()
        terminal = simulator(params, mc)
        runtime_sec = time.perf_counter() - t0
        mean_terminal = float(np.mean(terminal))
        std_terminal = float(np.std(terminal, ddof=1))
        stderr_terminal = float(std_terminal / math.sqrt(len(terminal)))
        martingale_error = float(mean_terminal - params.f0)
        rows.append(
            {
                "maturity": float(maturity),
                "mean_terminal": mean_terminal,
                "std_terminal": std_terminal,
                "stderr_terminal": stderr_terminal,
                "martingale_error": martingale_error,
                "z_score": martingale_error / stderr_terminal if stderr_terminal > 0.0 else np.nan,
                "runtime_sec": runtime_sec,
                "step": step,
                "n_paths": n_paths,
            }
        )
    out = pd.DataFrame(rows)
    conclusion = _martingale_conclusion_from_zscores(out["z_score"].to_numpy())
    out["conclusion"] = conclusion
    return out


def summarize_martingale(df: pd.DataFrame) -> dict[str, float | str]:
    """Aggregate martingale z-scores into a compact validation summary."""
    abs_z = df["z_score"].abs()
    return {
        "mean_abs_z": float(abs_z.mean()),
        "max_abs_z": float(abs_z.max()),
        "conclusion": str(df["conclusion"].iloc[0]) if len(df) else "Monte Carlo noise dominated",
    }


def runtime_benchmark(
    params: SABRParams,
    configs: Sequence[MonteCarloConfig],
    strike: float | None = None,
) -> pd.DataFrame:
    """Benchmark runtime across a list of Monte Carlo configurations.

    Inputs:
    - params: SABR model parameters.
    - configs: sequence of MonteCarloConfig objects.
    - strike: optional strike to price alongside runtime.

    Output:
    - DataFrame with one row per config and columns including
      `maturity`, `step`, `n_paths`, `runtime_sec`, `mean_terminal`,
      and optional `price`.
    """
    rows = []
    for mc in configs:
        t0 = time.perf_counter()
        terminal = simulate_terminal_forward(params, mc)
        runtime_sec = time.perf_counter() - t0
        row = {
            "maturity": mc.maturity,
            "step": mc.step,
            "n_paths": mc.n_paths,
            "seed": mc.seed,
            "runtime_sec": runtime_sec,
            "mean_terminal": float(np.mean(terminal)),
        }
        if strike is not None:
            row["strike"] = float(strike)
            row["price"] = european_call_price(terminal, float(strike))
        rows.append(row)
    return pd.DataFrame(rows)


def figure1_moment_comparison(
    hat_nu: float = 0.4,
    z_hat_grid: Sequence[float] | None = None,
    sigma_t: float = 1.0,
) -> pd.DataFrame:
    """Build the Figure 1 moment-comparison dataset.

    Inputs:
    - hat_nu: paper's `\\hat\\nu = nu * sqrt(h)`.
    - z_hat_grid: optional grid for `\\hat Z`. Defaults to `np.linspace(-4, 4, 161)`.
    - sigma_t: initial volatility level used to reconstruct `sigma_{t+h}` from `\\hat Z`.

    Output:
    - DataFrame with exact and approximation-implied skewness / ex-kurtosis columns:
      `exact_*`, `ln_*`, `sln_fixed_*`, `sln_exact_*`.
    """
    if z_hat_grid is None:
        z_hat_grid = np.linspace(-4.0, 4.0, 161)

    z_hat = np.asarray(z_hat_grid, dtype=float)
    sigma_t_arr = np.full_like(z_hat, float(sigma_t))
    sigma_next = sigma_t_arr * np.exp(hat_nu * z_hat)

    mu1, mu2_raw, mu3_raw, mu4_raw = conditional_integrated_variance_moments(
        sigma_t_arr, sigma_next, nu=float(hat_nu), h=1.0
    )
    _, _, _, cv_exact, skew_exact, exk_exact = raw_moments_to_central_stats(
        mu1, mu2_raw, mu3_raw, mu4_raw
    )

    # LN approximation: fit variance only.
    w_ln = cv_exact**2
    _, skew_ln, exk_ln = _lognormal_shape_stats_from_w(w_ln)

    # SLN with fixed lambda = 5/6: fit variance through sigma.
    lambda_fixed = 5.0 / 6.0
    w_sln_fixed = cv_exact**2 / (lambda_fixed**2)
    _, skew_sln_fixed, exk_sln_fixed = _lognormal_shape_stats_from_w(w_sln_fixed)

    # SLN with exact lambda: fit sigma to exact skewness, then lambda from CV.
    w_sln_exact = _sln_w_from_skewness(skew_exact)
    sqrt_w_exact = np.sqrt(np.maximum(w_sln_exact, 0.0))
    lambda_exact = _safe_ratio(cv_exact, np.maximum(sqrt_w_exact, PDF_FLOOR), fallback=0.0)
    _, skew_sln_exact, exk_sln_exact = _lognormal_shape_stats_from_w(w_sln_exact)

    return pd.DataFrame(
        {
            "z_hat": z_hat,
            "hat_nu": hat_nu,
            "exact_cv": cv_exact,
            "exact_skewness": skew_exact,
            "exact_ex_kurtosis": exk_exact,
            "ln_skewness": skew_ln,
            "ln_ex_kurtosis": exk_ln,
            "sln_fixed_lambda": lambda_fixed,
            "sln_fixed_skewness": skew_sln_fixed,
            "sln_fixed_ex_kurtosis": exk_sln_fixed,
            "sln_exact_lambda": lambda_exact,
            "sln_exact_skewness": skew_sln_exact,
            "sln_exact_ex_kurtosis": exk_sln_exact,
        }
    )


def table1_default_cases() -> pd.DataFrame:
    """Return the paper's Table 1 parameter grid."""
    return pd.DataFrame(
        [
            {"rho": -0.75, "nu": 0.2, "step": 1.0},
            {"rho": -0.75, "nu": 0.2, "step": 0.5},
            {"rho": -0.75, "nu": 0.2, "step": 0.25},
            {"rho": -0.5, "nu": 0.2, "step": 1.0},
            {"rho": -0.25, "nu": 0.2, "step": 1.0},
            {"rho": -0.75, "nu": 0.4, "step": 1.0},
            {"rho": -0.75, "nu": 0.6, "step": 1.0},
        ]
    )


def run_table1_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 50,
    seed0: int = 12345,
    strike: float = 1.0,
    maturity: float = 1.0,
    f0: float = 1.0,
    sigma0: float = 0.2,
    benchmark: BenchmarkProvider | Mapping[object, float] | None = table1_benchmark,
) -> pd.DataFrame:
    """Reproduce the Table 1 experiment scaffold for `beta = 1`.

    Inputs:
    - n_paths, n_repeats, seed0: Monte Carlo controls.
    - strike, maturity, f0, sigma0: base market/model inputs from the paper.
    - benchmark: optional callable or mapping for FDM prices.
      Mapping keys should be `(rho, nu)` as in the paper because the benchmark
      does not vary with `step`.

    Output:
    - DataFrame with per-case Monte Carlo summary statistics and optional relative error.
    """
    rows = []
    for idx, row in table1_default_cases().iterrows():
        params = SABRParams(f0=f0, sigma0=sigma0, nu=float(row["nu"]), rho=float(row["rho"]), beta=1.0)
        mc = MonteCarloConfig(
            maturity=maturity,
            step=float(row["step"]),
            n_paths=n_paths,
            seed=seed0 + idx * n_repeats,
        )
        summary = repeated_pricing(
            params=params,
            mc=mc,
            strikes=[strike],
            n_repeats=n_repeats,
            seed0=seed0 + idx * n_repeats,
            benchmark_prices={
                strike: _resolve_benchmark(benchmark, {"rho": row["rho"], "nu": row["nu"]}, ("rho", "nu"))
            }
            if benchmark is not None
            else None,
        )
        out = summary.iloc[0].to_dict()
        out.update({"rho": float(row["rho"]), "nu": float(row["nu"]), "beta": 1.0})
        rows.append(out)
    return pd.DataFrame(rows)


def table2_default_cases() -> pd.DataFrame:
    """Return the paper's Table 2 parameter grid."""
    rows = []
    for rho in [1.0, 0.75, 0.0]:
        for step in [1.0, 0.5, 0.25]:
            rows.extend(
                [
                    {"rho": rho, "nu": 0.2, "beta": 0.4, "step": step},
                    {"rho": rho, "nu": 0.2, "beta": 0.6, "step": step},
                    {"rho": rho, "nu": 0.2, "beta": 0.8, "step": step},
                    {"rho": rho, "nu": 0.4, "beta": 0.8, "step": step},
                    {"rho": rho, "nu": 0.8, "beta": 0.8, "step": step},
                ]
            )
    return pd.DataFrame(rows)


def run_table2_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 50,
    seed0: int = 12345,
    strike: float = 1.0,
    maturity: float = 1.0,
    f0: float = 1.0,
    sigma0: float = 0.2,
    benchmark: BenchmarkProvider | Mapping[object, float] | None = table2_benchmark,
) -> pd.DataFrame:
    """Reproduce the Table 2 experiment scaffold for `rho = 1, 0.75, 0`.

    Inputs:
    - n_paths, n_repeats, seed0: Monte Carlo controls.
    - strike, maturity, f0, sigma0: base market/model inputs from the paper.
    - benchmark: optional callable or mapping for FDM prices.
      Mapping keys should be `(rho, nu, beta)` because the benchmark does not vary with `step`.

    Output:
    - DataFrame with per-case Monte Carlo summary statistics and optional relative error.
      This is the scaffold for the paper's error-decomposition table.
    """
    rows = []
    cases = table2_default_cases()
    for idx, row in cases.iterrows():
        params = SABRParams(
            f0=f0,
            sigma0=sigma0,
            nu=float(row["nu"]),
            rho=float(row["rho"]),
            beta=float(row["beta"]),
        )
        mc = MonteCarloConfig(
            maturity=maturity,
            step=float(row["step"]),
            n_paths=n_paths,
            seed=seed0 + idx * n_repeats,
        )
        benchmark_value = _resolve_benchmark(
            benchmark,
            {"rho": row["rho"], "nu": row["nu"], "beta": row["beta"]},
            ("rho", "nu", "beta"),
        )
        summary = repeated_pricing(
            params=params,
            mc=mc,
            strikes=[strike],
            n_repeats=n_repeats,
            seed0=seed0 + idx * n_repeats,
            benchmark_prices={strike: benchmark_value} if benchmark_value is not None else None,
        )
        out = summary.iloc[0].to_dict()
        out.update(
            {
                "rho": float(row["rho"]),
                "nu": float(row["nu"]),
                "beta": float(row["beta"]),
            }
        )
        rows.append(out)
    return pd.DataFrame(rows)


def _case_params(case_name: str, maturity: float | None = None) -> tuple[SABRParams, float]:
    """Construct SABR parameters for one of the paper's named Table 3 cases."""
    case = case_table_3()[case_name]
    maturity_value = float(case["maturity"] if maturity is None else maturity)
    params = SABRParams(
        f0=float(case["f0"]),
        sigma0=float(case["sigma0"]),
        nu=float(case["nu"]),
        rho=float(case["rho"]),
        beta=float(case["beta"]),
    )
    return params, maturity_value


def _pyfeng_is_available() -> bool:
    """Return whether pyfeng is available for analytic SABR / CEV reference calculations."""
    return pf is not None


def _require_pyfeng() -> None:
    """Raise a clear error if pyfeng-dependent functionality is used without the dependency."""
    if pf is None:
        raise ImportError(
            "pyfeng is required for this functionality. Install `pyfeng` and `scipy` first."
        )


def _pyfeng_analytic_rows(
    table_name: str,
    case_name: str,
    params: SABRParams,
    maturity: float,
    strike_ratios: Sequence[float],
    strikes: Sequence[float],
    benchmark_prices: Mapping[float, float] | None,
) -> pd.DataFrame:
    """Generate analytic SABR reference rows using pyfeng's built-in approximation models."""
    if pf is None:
        raise ImportError("pyfeng is not installed; analytic rows cannot be generated from the package.")

    rows = []
    strikes_arr = np.asarray(strikes, dtype=float)
    benchmark_prices = {} if benchmark_prices is None else benchmark_prices
    benchmark_arr = np.asarray(
        [benchmark_prices.get(float(strike), np.nan) for strike in strikes_arr],
        dtype=float,
    )

    for label, class_name in PYFENG_ANALYTIC_MODELS:
        model_cls = getattr(pf, class_name)
        model = model_cls(
            sigma=params.sigma0,
            vov=params.nu,
            rho=params.rho,
            beta=params.beta,
            is_fwd=True,
        )
        prices = np.asarray(model.price(strikes_arr, params.f0, maturity), dtype=float)
        biases = prices - benchmark_arr
        for ratio, strike, benchmark_price, mean_price, bias in zip(
            strike_ratios,
            strikes_arr,
            benchmark_arr,
            prices,
            biases,
        ):
            rows.append(
                {
                    "table": table_name,
                    "case": case_name,
                    "model": label,
                    "source": "pyfeng",
                    "strike_ratio": float(ratio),
                    "strike": float(strike),
                    "benchmark_price": float(benchmark_price),
                    "mean_price": float(mean_price),
                    "bias": float(bias),
                    "bias_x1e3": float(1e3 * bias),
                    "stdev_price": np.nan,
                    "stderr_price": np.nan,
                    "runtime_sec_mean": np.nan,
                    "step": np.nan,
                    "maturity": float(maturity),
                    "n_paths": np.nan,
                    "n_repeats": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _analytic_reference_rows(
    table_name: str,
    case_name: str,
    params: SABRParams,
    maturity: float,
    strike_ratios: Sequence[float],
    strikes: Sequence[float],
    benchmark_prices: Mapping[float, float] | None,
    paper_reference_biases_x1e3: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Combine package-backed analytic rows with paper-only rows we still cannot compute."""
    frames = []
    if _pyfeng_is_available():
        frames.append(
            _pyfeng_analytic_rows(
                table_name=table_name,
                case_name=case_name,
                params=params,
                maturity=maturity,
                strike_ratios=strike_ratios,
                strikes=strikes,
                benchmark_prices=benchmark_prices,
            )
        )

    residual_reference = dict(paper_reference_biases_x1e3)
    if _pyfeng_is_available():
        residual_reference.pop("Hagan", None)

    if residual_reference:
        if benchmark_prices is None:
            benchmark_seq = np.full(len(strikes), np.nan, dtype=float)
        else:
            benchmark_seq = [benchmark_prices.get(float(strike), np.nan) for strike in strikes]
        frames.append(
            _reference_bias_rows(
                table_name=table_name,
                case_name=case_name,
                params=params,
                strike_ratios=strike_ratios,
                benchmark_prices=benchmark_seq,
                reference_biases_x1e3=residual_reference,
            )
        )

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _reference_bias_rows(
    table_name: str,
    case_name: str,
    params: SABRParams,
    strike_ratios: Sequence[float],
    benchmark_prices: Sequence[float],
    reference_biases_x1e3: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Convert paper-reported reference biases into a DataFrame aligned with our Monte Carlo outputs."""
    strikes = params.f0 * np.asarray(strike_ratios, dtype=float)
    rows = []
    for model, biases_x1e3 in reference_biases_x1e3.items():
        for ratio, strike, benchmark_price, bias_x1e3 in zip(
            strike_ratios, strikes, benchmark_prices, biases_x1e3
        ):
            bias = 1e-3 * float(bias_x1e3)
            rows.append(
                {
                    "table": table_name,
                    "case": case_name,
                    "model": model,
                    "source": "paper_reference",
                    "strike_ratio": float(ratio),
                    "strike": float(strike),
                    "benchmark_price": float(benchmark_price),
                    "mean_price": float(benchmark_price + bias),
                    "bias": bias,
                    "bias_x1e3": float(bias_x1e3),
                    "stdev_price": np.nan,
                    "stderr_price": np.nan,
                    "runtime_sec_mean": np.nan,
                    "step": np.nan,
                    "maturity": np.nan,
                    "n_paths": np.nan,
                    "n_repeats": np.nan,
                }
            )
    return pd.DataFrame(rows)


def _paper_scale_setups(
    base_setups: Sequence[Mapping[str, float]],
    n_paths_override: int | None = None,
) -> list[dict[str, float]]:
    """Optionally scale a list of paper-reference Monte Carlo setups to a different path budget."""
    setups = [dict(item) for item in base_setups]
    if n_paths_override is None or not setups:
        return setups

    base_paths = int(setups[0]["n_paths"])
    scale = float(n_paths_override) / float(base_paths)
    for item in setups:
        item["n_paths"] = max(1_000, int(round(item["n_paths"] * scale)))
    return setups


def run_table4_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 50,
    seed0: int = 12345,
    benchmark_prices: Mapping[float, float] | None = None,
) -> pd.DataFrame:
    """Reproduce Table 4 for Case I, together with the paper's analytic reference biases."""
    params, maturity = _case_params("Case I")
    strike_ratios = [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
    strikes = [params.f0 * x for x in strike_ratios]
    if benchmark_prices is None:
        benchmark_prices = {strike: price for strike, price in zip(strikes, TABLE4_FDM)}
    rows = []

    for idx, step in enumerate([1.0, 0.25, 0.0625]):
        mc = MonteCarloConfig(
            maturity=maturity,
            step=step,
            n_paths=n_paths,
            seed=seed0 + idx * n_repeats,
        )
        summary = repeated_pricing(
            params=params,
            mc=mc,
            strikes=strikes,
            n_repeats=n_repeats,
            seed0=seed0 + idx * n_repeats,
            benchmark_prices=benchmark_prices,
        )
        summary["table"] = "Table 4"
        summary["case"] = "Case I"
        summary["model"] = "Our method"
        summary["source"] = "simulated"
        summary["bias_x1e3"] = 1e3 * summary["bias"]
        summary["strike_ratio"] = summary["strike"] / params.f0
        rows.append(summary)

    rows.append(
        _analytic_reference_rows(
            table_name="Table 4",
            case_name="Case I",
            params=params,
            maturity=maturity,
            strike_ratios=strike_ratios,
            strikes=strikes,
            benchmark_prices=benchmark_prices,
            paper_reference_biases_x1e3=TABLE4_ANALYTIC_REFERENCE,
        )
    )
    return pd.concat(rows, ignore_index=True)


def run_table5_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 50,
    seed0: int = 12345,
    benchmark_prices: Mapping[float, float] | None = None,
) -> pd.DataFrame:
    """Reproduce Table 5 for Case II, together with the paper's analytic reference biases."""
    params, maturity = _case_params("Case II")
    strike_ratios = [0.2, 0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
    strikes = [params.f0 * x for x in strike_ratios]
    if benchmark_prices is None:
        benchmark_prices = {strike: price for strike, price in zip(strikes, TABLE5_FDM)}
    rows = []

    for idx, step in enumerate([1.0, 0.25, 0.0625]):
        mc = MonteCarloConfig(
            maturity=maturity,
            step=step,
            n_paths=n_paths,
            seed=seed0 + idx * n_repeats,
        )
        summary = repeated_pricing(
            params=params,
            mc=mc,
            strikes=strikes,
            n_repeats=n_repeats,
            seed0=seed0 + idx * n_repeats,
            benchmark_prices=benchmark_prices,
        )
        summary["table"] = "Table 5"
        summary["case"] = "Case II"
        summary["model"] = "Our method"
        summary["source"] = "simulated"
        summary["bias_x1e3"] = 1e3 * summary["bias"]
        summary["strike_ratio"] = summary["strike"] / params.f0
        rows.append(summary)

    rows.append(
        _analytic_reference_rows(
            table_name="Table 5",
            case_name="Case II",
            params=params,
            maturity=maturity,
            strike_ratios=strike_ratios,
            strikes=strikes,
            benchmark_prices=benchmark_prices,
            paper_reference_biases_x1e3=TABLE5_ANALYTIC_REFERENCE,
        )
    )
    return pd.concat(rows, ignore_index=True)


def run_table6_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 50,
    seed0: int = 12345,
    benchmark_prices: Mapping[float, float] | None = None,
) -> pd.DataFrame:
    """Reproduce Table 6 for Case III and append the paper's competing-baseline reference rows."""
    params, maturity = _case_params("Case III")
    strike_ratios = [0.4, 0.8, 1.0, 1.2, 1.6, 2.0]
    strikes = [params.f0 * x for x in strike_ratios]
    if benchmark_prices is None:
        benchmark_prices = {strike: price for strike, price in zip(strikes, TABLE6_FDM)}

    mc = MonteCarloConfig(maturity=maturity, step=1.0, n_paths=n_paths, seed=seed0)
    summary = repeated_pricing(
        params=params,
        mc=mc,
        strikes=strikes,
        n_repeats=n_repeats,
        seed0=seed0,
        benchmark_prices=benchmark_prices,
    )
    summary["table"] = "Table 6"
    summary["case"] = "Case III"
    summary["model"] = "Our method"
    summary["source"] = "simulated"
    summary["bias_x1e3"] = 1e3 * summary["bias"]
    summary["strike_ratio"] = summary["strike"] / params.f0

    rows = [summary]
    for (model, step), ref in TABLE6_BASELINE_REFERENCE.items():
        for ratio, strike, benchmark_price, bias_x1e3 in zip(
            strike_ratios, strikes, TABLE6_FDM, ref["bias_x1e3"]
        ):
            bias = 1e-3 * float(bias_x1e3)
            rows.append(
                pd.DataFrame(
                    [
                        {
                            "table": "Table 6",
                            "case": "Case III",
                            "model": model,
                            "source": "paper_reference",
                            "strike_ratio": float(ratio),
                            "strike": float(strike),
                            "benchmark_price": float(benchmark_price),
                            "mean_price": float(benchmark_price + bias),
                            "bias": bias,
                            "bias_x1e3": float(bias_x1e3),
                            "stdev_price": np.nan,
                            "stderr_price": np.nan,
                            "runtime_sec_mean": float(ref["time_sec"]),
                            "step": float(step),
                            "maturity": maturity,
                            "n_paths": np.nan,
                            "n_repeats": np.nan,
                        }
                    ]
                )
            )
    return pd.concat(rows, ignore_index=True)


def run_table7_experiment(
    n_paths_base: int | None = None,
    n_repeats: int = 5,
    seed0: int = 12345,
    benchmark_step: float = 0.03125,
    benchmark_n_paths: int | None = None,
    benchmark_repeats: int = 3,
    benchmark_source: str = "mc",
    fdm_config: FDMConfig | None = None,
) -> pd.DataFrame:
    """Reproduce the Table 7 / Figure 2 convergence study.

    `benchmark_source="mc"` uses the existing high-resolution Monte Carlo benchmark,
    while `benchmark_source="fdm"` switches the ATM reference price to the PDE/FDM solver.
    """
    params, maturity = _case_params("Case IV")
    setups = _paper_scale_setups(TABLE7_PAPER_REFERENCE, n_paths_override=n_paths_base)
    strike = params.f0
    if benchmark_source == "fdm":
        benchmark_price = finite_difference_call_price(
            params,
            maturity=maturity,
            strike=strike,
            config=fdm_config,
        )
        benchmark_note = "PDE/FDM benchmark"
    else:
        if benchmark_n_paths is None:
            benchmark_n_paths = max(20_000, 2 * int(setups[0]["n_paths"]))
        benchmark_summary = repeated_pricing(
            params=params,
            mc=MonteCarloConfig(
                maturity=maturity,
                step=benchmark_step,
                n_paths=benchmark_n_paths,
                seed=seed0 + 9_999,
            ),
            strikes=[strike],
            n_repeats=benchmark_repeats,
            seed0=seed0 + 9_999,
        )
        benchmark_price = float(benchmark_summary["mean_price"].iloc[0])
        benchmark_note = f"internal high-resolution MC (h={benchmark_step:g})"

    rows = []
    for idx, setup in enumerate(setups):
        mc = MonteCarloConfig(
            maturity=maturity,
            step=float(setup["step"]),
            n_paths=int(setup["n_paths"]),
            seed=seed0 + 100 * idx,
        )
        summary = repeated_pricing(
            params=params,
            mc=mc,
            strikes=[strike],
            n_repeats=n_repeats,
            seed0=seed0 + 100 * idx,
            benchmark_prices={strike: benchmark_price},
        )
        row = summary.iloc[0].to_dict()
        row.update(
            {
                "table": "Table 7",
                "case": "Case IV",
                "model": "Our method",
                "source": "simulated",
                "benchmark_note": benchmark_note,
                "rms_error": math.sqrt(float(row["bias"]) ** 2 + float(row["stdev_price"]) ** 2),
            }
        )
        row["rms_error_x1e3"] = 1e3 * row["rms_error"]
        rows.append(row)

    for setup in TABLE7_PAPER_REFERENCE:
        rows.append(
            {
                "table": "Table 7",
                "case": "Case IV",
                "model": "Our method (paper reference)",
                "source": "paper_reference",
                "benchmark_note": "paper FDM benchmark",
                "strike": strike,
                "mean_price": np.nan,
                "stdev_price": np.nan,
                "stderr_price": np.nan,
                "benchmark_price": np.nan,
                "bias": np.nan,
                "relative_error": np.nan,
                "mean_terminal": np.nan,
                "stdev_terminal_estimator": np.nan,
                "stderr_terminal_estimator": np.nan,
                "n_repeats": np.nan,
                "n_paths": int(setup["n_paths"]),
                "step": float(setup["step"]),
                "maturity": maturity,
                "runtime_sec_mean": float(setup["time_sec"]),
                "rms_error": 1e-3 * float(setup["rms_error_x1e3"]),
                "rms_error_x1e3": float(setup["rms_error_x1e3"]),
            }
        )
    return pd.DataFrame(rows)


def run_figure3_experiment(
    n_paths: int = 100_000,
    n_repeats: int = 2,
    seed0: int = 12345,
    benchmark_step: float = 0.25,
    benchmark_n_paths: int | None = None,
    benchmark_repeats: int = 2,
    benchmark_source: str = "mc",
    fdm_config: FDMConfig | None = None,
) -> pd.DataFrame:
    """Build the Figure 3 comparison dataset between the paper scheme and Islah's approximation.

    `benchmark_source="mc"` uses the previous high-resolution Monte Carlo ATM benchmark.
    `benchmark_source="fdm"` switches the ATM option benchmark to the PDE/FDM solver.
    """
    base_case = case_table_3()["Case V"]
    strike = float(base_case["f0"])
    rows = []

    for maturity in range(1, 11):
        params, _ = _case_params("Case V", maturity=float(maturity))
        if benchmark_source == "fdm":
            benchmark_price = finite_difference_call_price(
                params,
                maturity=float(maturity),
                strike=strike,
                config=fdm_config,
            )
            benchmark_note = "PDE/FDM benchmark"
        else:
            if benchmark_n_paths is None:
                benchmark_n = max(20_000, 2 * n_paths)
            else:
                benchmark_n = benchmark_n_paths

            benchmark_summary = repeated_pricing(
                params=params,
                mc=MonteCarloConfig(
                    maturity=float(maturity),
                    step=benchmark_step,
                    n_paths=benchmark_n,
                    seed=seed0 + 50_000 + maturity,
                ),
                strikes=[strike],
                n_repeats=benchmark_repeats,
                seed0=seed0 + 50_000 + maturity,
            )
            benchmark_price = float(benchmark_summary["mean_price"].iloc[0])
            benchmark_note = f"internal high-resolution MC (h={benchmark_step:g})"

        configs = [
            ("Our method", 1.0, simulate_terminal_forward),
            ("Our method", 0.5, simulate_terminal_forward),
            ("Islah", 1.0, simulate_terminal_forward_islah),
            ("Islah", 0.5, simulate_terminal_forward_islah),
        ]
        for offset, (model, step, simulator) in enumerate(configs):
            summary = repeated_pricing(
                params=params,
                mc=MonteCarloConfig(
                    maturity=float(maturity),
                    step=step,
                    n_paths=n_paths,
                    seed=seed0 + 1_000 * maturity + offset * n_repeats,
                ),
                strikes=[strike],
                n_repeats=n_repeats,
                seed0=seed0 + 1_000 * maturity + offset * n_repeats,
                benchmark_prices={strike: benchmark_price},
                simulator=simulator,
            )
            row = summary.iloc[0].to_dict()
            row.update(
                {
                    "figure": "Figure 3",
                    "case": "Case V",
                    "model": model,
                    "step": step,
                    "maturity": float(maturity),
                    "benchmark_note": benchmark_note,
                }
            )
            row["forward_error"] = float(row["mean_terminal"] - params.f0)
            row["option_error"] = float(row["mean_price"] - benchmark_price)
            rows.append(row)

    return pd.DataFrame(rows)


def figure2_runtime_tradeoff(
    n_paths_base: int | None = None,
    n_repeats: int = 5,
    seed0: int = 12345,
    benchmark_source: str = "mc",
    fdm_config: FDMConfig | None = None,
) -> pd.DataFrame:
    """Alias for the Table 7 convergence dataset used in Figure 2."""
    return run_table7_experiment(
        n_paths_base=n_paths_base,
        n_repeats=n_repeats,
        seed0=seed0,
        benchmark_source=benchmark_source,
        fdm_config=fdm_config,
    )


def case_table_3() -> dict[str, dict[str, float]]:
    """Parameter presets copied from Table 3 of the paper."""
    return {
        "Case I": {"f0": 1.0, "sigma0": 0.25, "nu": 0.3, "rho": -0.8, "beta": 0.3, "maturity": 10.0},
        "Case II": {"f0": 1.0, "sigma0": 0.25, "nu": 0.3, "rho": -0.5, "beta": 0.6, "maturity": 10.0},
        "Case III": {"f0": 0.05, "sigma0": 0.4, "nu": 0.6, "rho": 0.0, "beta": 0.3, "maturity": 1.0},
        "Case IV": {"f0": 1.1, "sigma0": 0.4, "nu": 0.8, "rho": -0.3, "beta": 0.3, "maturity": 4.0},
        "Case V": {"f0": 1.1, "sigma0": 0.3, "nu": 0.5, "rho": -0.8, "beta": 0.4, "maturity": 10.0},
    }


def summarize_prices(
    params: SABRParams,
    mc: MonteCarloConfig,
    strikes: Iterable[float],
) -> list[tuple[float, float]]:
    """Convenience wrapper for paper-style strike sweeps."""
    terminal = simulate_terminal_forward(params, mc)
    return [(float(k), european_call_price(terminal, float(k))) for k in strikes]


def _count_significant_bias_worsening(
    grp: pd.DataFrame,
    bias_col: str = "bias",
    stderr_col: str = "stderr_price",
) -> tuple[int, int, float]:
    """Count statistically significant increases in absolute bias as step decreases."""
    grp = grp.sort_values("step", ascending=False)
    abs_bias = grp[bias_col].abs().to_numpy(dtype=float)
    stderr = grp[stderr_col].to_numpy(dtype=float)

    material = 0
    severe = 0
    max_z = 0.0
    for idx in range(len(grp) - 1):
        if not (
            np.isfinite(abs_bias[idx])
            and np.isfinite(abs_bias[idx + 1])
            and np.isfinite(stderr[idx])
            and np.isfinite(stderr[idx + 1])
        ):
            continue
        diff = abs_bias[idx + 1] - abs_bias[idx]
        combined_stderr = math.hypot(stderr[idx], stderr[idx + 1])
        if combined_stderr <= PDF_FLOOR:
            z_score = math.inf if diff > 0.0 else 0.0
        else:
            z_score = diff / combined_stderr
        max_z = max(max_z, float(z_score))
        if z_score > 2.0:
            material += 1
        if z_score > 3.0:
            severe += 1
    return material, severe, max_z


def validate_table1(df: pd.DataFrame) -> tuple[str, str]:
    """Validate Table 1 style results against paper-level expectations."""
    rel = df["relative_error"].dropna().abs() if "relative_error" in df.columns else pd.Series(dtype=float)
    severity_score = 2
    messages = []

    if rel.empty:
        severity_score = min(severity_score, 1)
        messages.append("No benchmark provided; relative-error thresholds were skipped.")
    else:
        max_rel_pct = 100.0 * float(rel.max())
        avg_rel_pct = 100.0 * float(rel.mean())
        if max_rel_pct < 0.3:
            messages.append(
                f"All relative errors are below 0.3% (max {max_rel_pct:.4f}%, avg {avg_rel_pct:.4f}%)."
            )
        elif max_rel_pct <= 1.0:
            severity_score = min(severity_score, 1)
            messages.append(
                f"Relative errors stay within 1% but exceed 0.3% (max {max_rel_pct:.4f}%, avg {avg_rel_pct:.4f}%)."
            )
        else:
            severity_score = 0
            messages.append(
                f"Relative errors exceed 1% (max {max_rel_pct:.4f}%, avg {avg_rel_pct:.4f}%)."
            )

    material_increase_count = 0
    severe_increase_count = 0
    max_worsening_z = 0.0
    for (rho, nu), grp in df.groupby(["rho", "nu"]):
        material, severe, worst_z = _count_significant_bias_worsening(grp)
        material_increase_count += material
        severe_increase_count += severe
        max_worsening_z = max(max_worsening_z, worst_z)
    if severe_increase_count >= 2:
        severity_score = min(severity_score, 0)
        messages.append(
            "Step-size trend shows repeated statistically significant bias worsening as step decreases "
            f"(worst increase {max_worsening_z:.2f} standard errors)."
        )
    elif material_increase_count >= 1:
        severity_score = min(severity_score, 1)
        messages.append(
            "Step-size trend is broadly consistent; at least one bias increase exceeds Monte Carlo noise "
            f"(worst increase {max_worsening_z:.2f} standard errors)."
        )
    else:
        messages.append(
            "Step-size trend is broadly consistent with decreasing bias once Monte Carlo standard error is taken into account."
        )

    return _validation_label(severity_score), " ".join(messages)


def validate_table2(df: pd.DataFrame) -> tuple[str, str]:
    """Validate Table 2 style error decomposition trends."""
    rel = df["relative_error"].abs() if "relative_error" in df.columns else pd.Series(np.nan, index=df.index)
    has_benchmark = rel.notna().any()
    severity_score = 2
    messages = []

    if not has_benchmark:
        severity_score = 1
        messages.append("No benchmark provided; trend checks based on relative_error were skipped.")
        return _validation_label(severity_score), " ".join(messages)

    rho_means = df.assign(abs_rel=rel).groupby("rho", as_index=True)["abs_rel"].mean()
    rho_means_pct = {rho: 100.0 * float(val) for rho, val in rho_means.items()}
    if 0.0 in rho_means.index and rho_means.loc[0.0] == rho_means.min():
        messages.append(
            "Average relative error by rho: "
            + ", ".join(f"rho={rho:g}: {val:.4f}%" for rho, val in rho_means_pct.items())
            + ". rho=0 has the smallest average error."
        )
    else:
        severity_score = 0
        messages.append(
            "Average relative error by rho: "
            + ", ".join(f"rho={rho:g}: {val:.4f}%" for rho, val in rho_means_pct.items())
            + ". rho=0 does not have the smallest average error."
        )

    rho_order = [0.0, 0.75, 1.0]
    if all(r in rho_means.index for r in rho_order) and rho_means.loc[0.0] <= rho_means.loc[0.75] <= rho_means.loc[1.0]:
        messages.append("Average error increases with rho.")
    else:
        severity_score = min(severity_score, 1)
        messages.append("Average error does not increase monotonically with rho.")

    material_increase_count = 0
    severe_increase_count = 0
    max_worsening_z = 0.0
    for (rho, nu, beta), grp in df.assign(abs_rel=rel).groupby(["rho", "nu", "beta"]):
        material, severe, worst_z = _count_significant_bias_worsening(grp)
        material_increase_count += material
        severe_increase_count += severe
        max_worsening_z = max(max_worsening_z, worst_z)
    if severe_increase_count >= 3:
        severity_score = min(severity_score, 0)
        messages.append(
            "Step-size trend shows repeated statistically significant bias worsening as step decreases "
            f"(worst increase {max_worsening_z:.2f} standard errors)."
        )
    elif material_increase_count >= 1:
        severity_score = min(severity_score, 1)
        messages.append(
            "Step-size trend is broadly consistent across rho/nu/beta groups; only a few bias increases rise above Monte Carlo noise "
            f"(worst increase {max_worsening_z:.2f} standard errors)."
        )
    else:
        messages.append(
            "Step-size trend is broadly consistent with decreasing bias across rho/nu/beta groups once Monte Carlo standard error is taken into account."
        )

    return _validation_label(severity_score), " ".join(messages)


def run_full_validation(
    table1_benchmark: BenchmarkProvider | Mapping[object, float] | None = table1_benchmark,
    table2_benchmark: BenchmarkProvider | Mapping[object, float] | None = table2_benchmark,
    n_paths_table1: int = 100_000,
    n_repeats_table1: int = 50,
    n_paths_table2: int = 100_000,
    n_repeats_table2: int = 50,
    martingale_n_paths: int = 100_000,
    quick_mode: bool = False,
) -> dict[str, object]:
    """Run the validation layer on top of the current replication scaffold."""
    if quick_mode:
        n_paths_table1 = 5_000
        n_repeats_table1 = 3
        n_paths_table2 = 3_000
        n_repeats_table2 = 2
        martingale_n_paths = 15_000

    case_v = case_table_3()["Case V"]
    martingale_params = SABRParams(
        f0=case_v["f0"],
        sigma0=case_v["sigma0"],
        nu=case_v["nu"],
        rho=case_v["rho"],
        beta=case_v["beta"],
    )

    martingale_df = martingale_test(
        martingale_params,
        maturities=list(range(1, 11)),
        step=1.0,
        n_paths=martingale_n_paths,
        seed0=101,
    )
    martingale_summary = summarize_martingale(martingale_df)

    table1_df = run_table1_experiment(
        n_paths=n_paths_table1,
        n_repeats=n_repeats_table1,
        benchmark=table1_benchmark,
    )
    table1_status, table1_expl = validate_table1(table1_df)

    table2_df = run_table2_experiment(
        n_paths=n_paths_table2,
        n_repeats=n_repeats_table2,
        benchmark=table2_benchmark,
    )
    table2_status, table2_expl = validate_table2(table2_df)

    overall = "Implementation consistent with paper"
    if "FAIL" in (table1_status, table2_status) or martingale_summary["conclusion"] != "Monte Carlo noise dominated":
        overall = "Potential issues detected"
    replication_line = (
        "Replication likely successful"
        if martingale_summary["conclusion"] == "Monte Carlo noise dominated"
        and table1_status in {"PASS", "WARNING"}
        and table2_status in {"PASS", "WARNING"}
        else "Potential systematic issues remain"
    )
    if quick_mode and ("FAIL" in (table1_status, table2_status)):
        replication_line = (
            "Result may be dominated by Monte Carlo noise; rerun at paper scale before diagnosing systematic issues."
        )
    scale_label = "quick smoke test" if quick_mode else "paper-scale validation"

    print("======== SABR REPLICATION VALIDATION ========")
    print(f"Run mode: {scale_label}")
    print()
    print("Martingale test:")
    print(
        f"mean |z| = {martingale_summary['mean_abs_z']:.3f}, "
        f"max |z| = {martingale_summary['max_abs_z']:.3f}, "
        f"conclusion = {martingale_summary['conclusion']}"
    )
    if table1_benchmark is None or table2_benchmark is None:
        print("WARNING: No benchmark provided -> validation is structural only")
    print()
    print("Table 1:")
    print(table1_status)
    print(table1_expl)
    print()
    print("Table 2:")
    print(table2_status)
    print(table2_expl)
    print()
    print("Conclusion:")
    print(f"- {overall}")
    print(f"- {replication_line}")

    return {
        "martingale_df": martingale_df,
        "martingale_summary": martingale_summary,
        "table1_df": table1_df,
        "table1_status": table1_status,
        "table1_explanation": table1_expl,
        "table2_df": table2_df,
        "table2_status": table2_status,
        "table2_explanation": table2_expl,
        "overall_conclusion": overall,
        "replication_conclusion": replication_line,
        "quick_mode": quick_mode,
        "run_mode": scale_label,
    }


def rerun_paper_scale_validation() -> dict[str, object]:
    """Convenience helper for a full paper-scale validation run."""
    return run_full_validation(quick_mode=False)
