from pathlib import Path

import pytest
import yaml

from fxml.config import DEFAULT_CONFIG, REPO_ROOT, load_config


def test_default_config_loads():
    cfg = load_config()
    assert len(cfg.pairs) == 7
    assert cfg.start < cfg.walk_forward.test_start < cfg.end
    assert cfg.bar_minutes == 5


def test_every_pair_has_costs_and_h10_series():
    cfg = load_config()
    h10_pairs = {name.removeprefix("h10_") for name in cfg.fred_series.values()}
    for pair in cfg.pairs:
        assert cfg.costs.spread_in_price(pair) > 0
        assert pair in h10_pairs


def test_paths_resolve_inside_repo():
    cfg = load_config()
    for path in vars(cfg.paths).values():
        assert path.is_relative_to(REPO_ROOT)


def test_invalid_dates_are_rejected(tmp_path: Path):
    raw = yaml.safe_load(DEFAULT_CONFIG.read_text())
    raw["walk_forward"]["test_start"] = raw["period"]["start"]
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match="test_start"):
        load_config(bad)
