"""Beaucage Unified Fit for SAXS data.

Edit FIT_CONFIG below, then run:
    python beaucage_unified_fit.py --file sample.dat
    python beaucage_unified_fit.py --folder data_folder
Files may contain q, I, and optionally sigma (whitespace or comma separated).
Levels are ordered from smallest scatterer (high q) to largest (low q), as in Irena.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.special import erf


@dataclass
class Parameter:
    value: float
    fit: bool = True
    lower: float = 0.0
    upper: float = np.inf


@dataclass
class Level:
    G: Parameter
    Rg: Parameter
    B: Parameter
    P: Parameter
    # Irena uses k=1.06 for mass-fractal levels and k=1 for other levels.
    # Keep this choice fixed during fitting so the model stays continuous in P.
    mass_fractal: bool = False
    cutoff: str = "none"  # "none", "previous", or "custom"
    Rgco: Parameter = field(default_factory=lambda: Parameter(0.0, False))
    correlations: bool = False
    eta: Parameter = field(default_factory=lambda: Parameter(100.0, False, 0.0))
    pack: Parameter = field(default_factory=lambda: Parameter(0.0, False, 0.0, 5.92))


@dataclass
class Config:
    background: Parameter
    levels: list[Level]
    q_min: float | None = None
    q_max: float | None = None
    residual: str = "log"  # "log" or "sigma"; sigma falls back to log if absent
    max_nfev: int = 30000


# USER SETTINGS: start with the smallest/high-q population. Add levels toward low q.
# Set fit=False to hold a parameter fixed. Bounds apply only to parameters being fitted.
# For hierarchical populations, set the larger level cutoff="previous".
# For independent populations, leave cutoff="none".
FIT_CONFIG = Config(
    background=Parameter(4.0, True, 0.0, 20.0),
    levels=[
        # Level 1: smaller structure / shoulder at intermediate-to-high q.
        Level(
            G=Parameter(40.0, True, 1.0, 500.0),
            Rg=Parameter(40.0, True, 5.0, 150.0),
            B=Parameter(1e-4, True, 1e-12, 1.0),
            P=Parameter(4.0, True, 1.0, 5.0),
            mass_fractal=False,
            cutoff="none",
        ),

        # Level 2: unresolved large structure that produces the low-q power law.
        # G=0 and a very large fixed Rg suppress its unobserved Guinier region.
        Level(
            G=Parameter(0.0, False),
            Rg=Parameter(10000.0, False),
            B=Parameter(1e-6, True, 1e-12, 1.0),
            P=Parameter(4.0, True, 1.0, 5.0),
            mass_fractal=False,
            cutoff="none",
        ),

        # OPTIONAL LEVEL 3: remove the leading "#" characters to enable it.
        #Level(
        #    G=Parameter(1000.0, True, 1e-12, 1e9),
        #    Rg=Parameter(300.0, True, 100.0, 3000.0),
        #    B=Parameter(1e-4, True, 1e-12, 10.0),
        #    P=Parameter(2.5, True, 1.0, 4.0),
        #    mass_fractal=True,
        #    cutoff="previous",
        #),
    ],
    q_min=0.0025,
    q_max=0.15,
    residual="log",
)


def spherical_amplitude(x: np.ndarray) -> np.ndarray:
    """3(sin(x)-x cos(x))/x^3, with its finite limit at zero."""
    out = np.empty_like(x, dtype=float)
    small = np.abs(x) < 1e-3
    out[small] = 1 - x[small] ** 2 / 10 + x[small] ** 4 / 280
    z = x[~small]
    out[~small] = 3 * (np.sin(z) - z * np.cos(z)) / z**3
    return out


def model(q: np.ndarray, config: Config) -> tuple[np.ndarray, list[np.ndarray]]:
    q = np.asarray(q, float)
    if np.any(q <= 0) or not np.all(np.isfinite(q)):
        raise ValueError("q must be positive and finite")
    total = np.full_like(q, config.background.value, dtype=float)
    components = []
    for i, level in enumerate(config.levels):
        G, Rg, B, P = (getattr(level, name).value for name in ("G", "Rg", "B", "P"))
        if min(G, Rg, B, P) < 0 or Rg == 0:
            raise ValueError(f"Level {i+1}: G, B, P must be nonnegative and Rg positive")
        if level.cutoff == "none":
            rgco = 0.0
        elif level.cutoff == "previous" and i > 0:
            rgco = config.levels[i - 1].Rg.value
        elif level.cutoff == "custom":
            rgco = level.Rgco.value
        else:
            raise ValueError(f"Level {i+1}: invalid cutoff {level.cutoff!r}")
        k = 1.06 if level.mass_fractal else 1.0
        transition = erf(k * q * Rg / np.sqrt(6))
        power = B * np.exp(-(q * rgco) ** 2 / 3) * (transition**3 / q) ** P
        component = G * np.exp(-(q * Rg) ** 2 / 3) + power
        if level.correlations:
            if level.eta.value <= 0 or level.pack.value < 0:
                raise ValueError(f"Level {i+1}: eta must be positive; pack nonnegative")
            denominator = 1 + level.pack.value * spherical_amplitude(q * level.eta.value)
            if np.any(denominator <= 0):
                raise ValueError(f"Level {i+1}: correlation denominator is nonpositive")
            component = component / denominator
        total += component
        components.append(component)
    return total, components


def load_data(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    rows = []
    with path.open(encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith(("#", "//", ";")):
                continue
            fields = line.replace(",", " ").split()
            try:
                numbers = [float(item) for item in fields[:3]]
            except ValueError:
                continue  # text header
            if len(numbers) >= 2:
                rows.append(numbers)
    if not rows:
        raise ValueError("No numeric q, I rows")
    q = np.array([row[0] for row in rows])
    intensity = np.array([row[1] for row in rows])
    good = np.isfinite(q) & np.isfinite(intensity) & (q > 0) & (intensity > 0)
    q, intensity = q[good], intensity[good]
    if len(q) < 10:
        raise ValueError("Need at least 10 positive finite q/I points")
    sigma = None
    if all(len(row) >= 3 for row in rows):
        sigma_all = np.array([row[2] for row in rows])
        sigma = sigma_all[good]
        if not np.all(np.isfinite(sigma) & (sigma > 0)):
            sigma = None
    order = np.argsort(q)
    return q[order], intensity[order], None if sigma is None else sigma[order]


def parameters(config: Config) -> list[tuple[str, Parameter]]:
    items = [("background", config.background)]
    for i, level in enumerate(config.levels, 1):
        items.extend((f"{name}{i}", getattr(level, name)) for name in ("G", "Rg", "B", "P"))
        if level.cutoff == "custom":
            items.append((f"Rgco{i}", level.Rgco))
        if level.correlations:
            items.extend((f"{name}{i}", getattr(level, name)) for name in ("eta", "pack"))
    return items


def fit(q: np.ndarray, intensity: np.ndarray, sigma: np.ndarray | None,
        config: Config) -> dict:
    mask = np.ones(len(q), dtype=bool)
    if config.q_min is not None:
        mask &= q >= config.q_min
    if config.q_max is not None:
        mask &= q <= config.q_max
    qfit, ifit = q[mask], intensity[mask]
    sfit = None if sigma is None else sigma[mask]
    entries = parameters(config)
    active = [(name, param) for name, param in entries if param.fit]
    if len(qfit) <= len(active) + 2:
        raise ValueError("Fit range has too few points for the free parameters")
    if not active:
        raise ValueError("No parameters selected to fit")
    for name, param in entries:
        if not np.isfinite(param.value) or param.value < 0:
            raise ValueError(f"{name}: starting value must be finite and nonnegative")
        if param.fit and not (param.lower <= param.value <= param.upper):
            raise ValueError(f"{name}: starting value is outside bounds")
        if param.fit and param.lower >= param.upper:
            raise ValueError(f"{name}: lower bound must be smaller than upper")
    x0 = np.array([param.value for _, param in active])
    lower = np.array([param.lower for _, param in active])
    upper = np.array([param.upper for _, param in active])
    use_sigma = config.residual == "sigma" and sfit is not None
    if config.residual not in ("log", "sigma"):
        raise ValueError("residual must be 'log' or 'sigma'")

    def residual(values):
        for (_, param), value in zip(active, values):
            param.value = value
        predicted, _ = model(qfit, config)
        if not np.all(np.isfinite(predicted)) or np.any(predicted <= 0):
            return np.full(len(qfit), 1e100)
        return ((predicted - ifit) / sfit if use_sigma
                else np.log(predicted) - np.log(ifit))

    result = least_squares(residual, x0, bounds=(lower, upper),
                           max_nfev=config.max_nfev, x_scale="jac")
    residual(result.x)  # leave fitted values in config
    dof = len(qfit) - len(active)
    score = float(np.sum(result.fun**2) / dof)
    covariance = np.linalg.pinv(result.jac.T @ result.jac) * score
    errors = np.sqrt(np.maximum(np.diag(covariance), 0))
    output = {"success": bool(result.success), "message": result.message,
              "points": len(qfit), "residual": "sigma" if use_sigma else "log",
              "reduced_chi2": score if use_sigma else np.nan,
              "mean_squared_log_residual": score if not use_sigma else np.nan}
    output.update({name: param.value for name, param in entries})
    output.update({f"{name}_err": error for (name, _), error in zip(active, errors)})
    return output


def save_outputs(path: Path, q: np.ndarray, intensity: np.ndarray, config: Config,
                 out_dir: Path) -> None:
    predicted, components = model(q, config)
    stem = path.stem
    with (out_dir / f"{stem}_fit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q", "intensity", "fit", "residual"] +
                        [f"level{i}" for i in range(1, len(components) + 1)])
        for row in zip(q, intensity, predicted, (intensity-predicted)/intensity,
                       *components):
            writer.writerow(row)
    fig, (ax, rx) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                 gridspec_kw={"height_ratios": [3, 1]})
    ax.loglog(q, intensity, ".", label="Data")
    ax.loglog(q, predicted, "-", label="Unified fit")
    for i, component in enumerate(components, 1):
        ax.loglog(q, np.maximum(component, np.finfo(float).tiny),
                  "--", label=f"Level {i}")
    # Set the visible intensity range from the data and total fit only. Individual
    # level curves can become many orders of magnitude smaller outside the q range
    # where they contribute; letting those tails autoscale the axis obscures the
    # data without adding useful information. The curves are still calculated in
    # full and are available in the per-sample CSV.
    visible = np.r_[intensity, predicted]
    visible = visible[np.isfinite(visible) & (visible > 0)]
    if visible.size:
        ax.set_ylim(0.5 * visible.min(), 2.0 * visible.max())
    ax.set_ylabel("I(q)")
    ax.legend()
    ax.grid(alpha=.25)
    rx.semilogx(q, (intensity-predicted)/intensity, ".")
    rx.axhline(0, color="k", lw=.8)
    rx.set(xlabel=r"q ($\AA^{-1}$)", ylabel="(data-fit)/data")
    rx.grid(alpha=.25)
    fig.suptitle(path.name)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}_fit.png", dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", type=Path, help="One SAXS data file")
    group.add_argument("--folder", type=Path, help="Batch folder")
    parser.add_argument("--out-dir", type=Path, default=Path("beaucage_fits"))
    args = parser.parse_args()
    paths = ([args.file] if args.file else
             sorted(p for p in args.folder.iterdir()
                    if p.suffix.lower() in (".dat", ".txt", ".csv")))
    if not paths:
        parser.error("No data files found")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    import copy
    for path in paths:
        sample_config = copy.deepcopy(FIT_CONFIG)  # each batch fit starts at user values
        try:
            q, intensity, sigma = load_data(path)
            result = fit(q, intensity, sigma, sample_config)
            save_outputs(path, q, intensity, sample_config, args.out_dir)
            print(f"{path.name}: {result['message']}")
        except Exception as exc:
            result = {"success": False, "message": str(exc)}
            print(f"{path.name}: failed: {exc}")
        rows.append({"filename": path.name, **result})
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with (args.out_dir / "fit_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return 0 if all(row["success"] for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
