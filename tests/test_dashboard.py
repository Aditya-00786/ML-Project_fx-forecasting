import importlib.util
import json
import re

import pandas as pd
import pytest

from fxml.config import REPO_ROOT
from fxml.dashboard import data as dd
from fxml.dashboard.page import render_page
from fxml.dashboard.render import block, component_css, glyph, scorecard_table, tokens

DATA_FILE = REPO_ROOT / "reports" / "dashboard" / "data.json"
needs_data = pytest.mark.skipif(not DATA_FILE.exists(), reason="dashboard data not built")


def test_verdict_codes_follow_the_diebold_mariano_sign_and_p_value():
    assert dd.verdict(-3.1, 0.002) == 2  # lower loss, significant
    assert dd.verdict(-0.5, 0.6) == 1
    assert dd.verdict(0.5, 0.6) == -1
    assert dd.verdict(4.0, 0.0001) == -2
    assert dd.verdict(None, None) == 0 and dd.verdict(float("nan"), 0.1) == 0


def test_rounding_and_labels():
    assert dd.rounded(0.123456) == 0.1235 and dd.rounded(-1234.5678) == -1235.0
    assert dd.rounded(float("nan")) is None and dd.rounded(0) == 0.0
    assert dd.model_label("lightgbm_experts") == "LightGBM · GMM experts"
    assert dd.model_group("baseline_har") == "Benchmarks"
    assert dd.model_group("ridge_hmm") == "Regime-aware" and dd.model_group("lstm") == "Neural"
    ordered = sorted(["mlp", "lightgbm_hmm", "ridge", "baseline_garch"], key=dd.model_sort_key)
    assert ordered == ["baseline_garch", "ridge", "lightgbm_hmm", "mlp"]


def toy_table():
    return pd.DataFrame({
        "pair": ["EURUSD", "EURUSD", "USDJPY", "USDJPY", "average", "average"],
        "model": ["baseline_har", "lightgbm"] * 3,
        "days": [100.0] * 6, "qlike": [0.2, 0.18, 0.3, 0.33, 0.25, 0.255],
        "dm_vs_baseline_har": [None, -2.5, None, 1.0, None, None],
        "dm_p_vs_baseline_har": [None, 0.01, None, 0.3, None, None],
    })


def test_scorecard_counts_and_cells():
    card = dd.scorecard(toy_table(), "volatility", ["EURUSD", "USDJPY"])
    assert [c["id"] for c in card["columns"]] == ["baseline_har", "lightgbm"]
    cells = {(c["pair"], c["model"]): c for c in card["cells"]}
    assert cells[("EURUSD", "lightgbm")]["v"] == 2 and cells[("USDJPY", "lightgbm")]["v"] == -1
    summary = {s["model"]: s for s in card["summary"]}
    assert summary["lightgbm"]["better"] == 1 and summary["lightgbm"]["worse"] == 0


def test_scorecard_table_marks_every_cell_and_leaves_out_the_reference():
    data = {"meta": {"tasks": dd.TASKS}}
    card = dd.scorecard(toy_table(), "volatility", ["EURUSD", "USDJPY"])
    html = scorecard_table([block(data, "volatility", "volatility", card)], ["EURUSD", "USDJPY"],
                           {"EURUSD": "EUR/USD", "USDJPY": "USD/JPY"}, card_id="t",
                           caption="toy", labels={})
    assert html.count('role="gridcell"') == 2  # HAR-RV is the reference, not a column
    assert "LightGBM on EUR/USD: significantly better than HAR-RV" in html
    assert 'class="sc-cell v2"' in html and 'class="sc-cell vn1"' in html
    assert "#g-up-fill" in html and "<span class=\"gl" not in html
    inline = scorecard_table([block(data, "volatility", "volatility", card)], ["EURUSD"],
                             {"EURUSD": "EUR/USD"}, card_id="t", caption="toy", labels={},
                             inline=True)
    assert "<svg" not in inline and 'class="gl gl-up-fill"' in inline  # for hosts that strip SVG
    assert glyph(0) != glyph(2)


def test_theme_tokens_and_embeddable_styles():
    light, dark = tokens("light"), tokens("dark")
    assert light["course"] == "#3f2bb0" and dark["course"] == "#d6c9ff"
    assert all(v.startswith("#") or v.startswith("rgba") for v in (light["ink"], dark["ink"]))
    css = component_css("dark")
    assert ".sc-cell" in css and "\nbody {" not in css and "prefers-color-scheme" not in css


def test_cumulative_advantage_rises_when_the_model_has_lower_loss():
    dates = pd.bdate_range("2020-01-01", periods=60)
    rows = []
    for model, error in (("baseline_zero", 0.02), ("good", 0.01)):
        rows += [{"task": "returns", "pair": "EURUSD", "model": model, "date": d, "actual": 0.0,
                  "forecast": error} for d in dates]
    preds = pd.DataFrame(rows)
    months = dd.test_months(preds)
    path = dd.cumulative_advantage(preds, "returns", months)["EURUSD"]["good"]
    assert len(path) == len(months) and all(b > a for a, b in zip(path, path[1:], strict=False))


@needs_data
def test_page_renders_from_the_saved_data():
    data = json.loads(DATA_FILE.read_text())
    page = render_page(data)
    assert page.startswith("<!doctype html>") and "$" not in re.sub(r"<script.*?</script>", "",
                                                                    page, flags=re.S)
    payload = re.search(r'<script type="application/json" id="dashboard-data">(.*?)</script>',
                        page, re.S).group(1)
    assert json.loads(payload.replace("<\\/", "</"))["meta"] == data["meta"]
    for anchor in ("view-verdicts", "view-explorer", "view-present", "ev-verdicts", "sc-master",
                   "tpl-present-volatility", "bt-chart", "regime-chart"):
        assert f'id="{anchor}"' in page
    assert len(data["questions"]) == 5
    assert all(q["evidence"] and q["answer"] for q in data["questions"])
    assert not re.search(r"/home/|/Users/", page)  # no machine paths in the shared page


@needs_data
@pytest.mark.skipif(importlib.util.find_spec("streamlit") is None, reason="streamlit not installed")
def test_streamlit_app_runs_the_verdicts_page():
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_file(str(REPO_ROOT / "dashboard" / "app.py"), default_timeout=120).run()
    assert not app.exception
    assert app.title[0].value == json.loads(DATA_FILE.read_text())["meta"]["title"]
