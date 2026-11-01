"""Load and validate the project configuration from ``configs/*.yaml``."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


@dataclass(frozen=True)
class Paths:
    raw: Path
    interim: Path
    processed: Path
    reports: Path


@dataclass(frozen=True)
class WalkForward:
    test_start: date
    refit_frequency: str
    neural_refit_frequency: str
    validation_fraction: float
    embargo_days: int


@dataclass(frozen=True)
class Costs:
    spread_pips: dict[str, float]
    pip_size: dict[str, float]
    stress_multiplier: float

    def spread_in_price(self, pair: str) -> float:
        """Full bid-ask spread of ``pair`` in price units."""
        return self.spread_pips[pair] * self.pip_size[pair]


@dataclass(frozen=True)
class Config:
    seed: int
    paths: Paths
    pairs: tuple[str, ...]
    context_instruments: tuple[str, ...]
    extra_pairs: tuple[str, ...]
    daily_only: dict[str, str]  # pair -> FRED series of its daily noon rates
    start: date
    end: date
    timezone: str
    day_close_hour: int
    bar_minutes: int
    walk_forward: WalkForward
    costs: Costs
    fred_series: dict[str, str]
    policy_rates: dict[str, str]


def load_config(path: str | Path | None = None) -> Config:
    """Read a YAML config, resolve paths against the repo root and validate it."""
    path = Path(path) if path is not None else DEFAULT_CONFIG
    raw = yaml.safe_load(path.read_text())

    paths = Paths(**{name: REPO_ROOT / rel for name, rel in raw["paths"].items()})
    cfg = Config(
        seed=int(raw["seed"]),
        paths=paths,
        pairs=tuple(raw["pairs"]),
        context_instruments=tuple(raw["context_instruments"]),
        extra_pairs=tuple(raw.get("extra_pairs", ())),
        daily_only=dict(raw.get("daily_only") or {}),
        start=raw["period"]["start"],
        end=raw["period"]["end"],
        timezone=raw["trading_day"]["timezone"],
        day_close_hour=int(raw["trading_day"]["close_hour"]),
        bar_minutes=int(raw["bars"]["minutes"]),
        walk_forward=WalkForward(**raw["walk_forward"]),
        costs=Costs(**raw["costs"]),
        fred_series=dict(raw["fred_series"]),
        policy_rates=dict(raw["policy_rates"]),
    )
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    if not cfg.start < cfg.walk_forward.test_start < cfg.end:
        raise ValueError("expected period.start < walk_forward.test_start < period.end")
    if not 0 < cfg.walk_forward.validation_fraction < 1:
        raise ValueError("walk_forward.validation_fraction must be between 0 and 1")
    if cfg.walk_forward.embargo_days < 0:
        raise ValueError("walk_forward.embargo_days must not be negative")
    if 1440 % cfg.bar_minutes:
        raise ValueError("bars.minutes must divide a 24-hour day")
    for pair in (*cfg.pairs, *cfg.extra_pairs):
        if pair not in cfg.costs.spread_pips or pair not in cfg.costs.pip_size:
            raise ValueError(f"missing spread or pip size for {pair}")
    if set(cfg.extra_pairs) & set(cfg.pairs):
        raise ValueError("extra_pairs must not repeat a core pair")
    if not set(cfg.daily_only) <= set(cfg.extra_pairs):
        raise ValueError("daily_only pairs must be listed in extra_pairs")
