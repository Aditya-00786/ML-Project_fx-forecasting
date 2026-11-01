"""The static dashboard page: one self-contained HTML file with the styles, the script and the
collected results inlined, so it opens offline from the repository.

The page skeleton lives in ``assets/page.html``; this module renders the parts that depend on
the results (scorecards, research questions, controls) and fills it in.
"""

from __future__ import annotations

import json
from html import escape
from importlib.resources import files
from string import Template

from fxml.dashboard.render import (
    SPRITE,
    block,
    legend,
    question_html,
    scorecard_table,
    stylesheet,
)

VERDICT_MODELS = {  # the master scorecard's columns: a representative model from each family
    "volatility": ["baseline_garch", "baseline_ewma", "ridge", "random_forest", "lightgbm",
                   "lightgbm_experts", "lightgbm_hmm", "mlp", "lstm", "transformer"],
    "direction": ["logistic_l2", "naive_bayes", "svm_rbf", "random_forest", "lightgbm",
                  "lightgbm_hmm", "mlp", "lstm", "transformer"],
    "returns": ["baseline_arima", "ridge", "lightgbm", "mlp", "lstm", "transformer"],
}
HORIZON_MODELS = {
    "volatility": ["baseline_naive", "baseline_ewma", "baseline_garch", "ridge", "lightgbm"],
    "direction": ["logistic_l2", "lightgbm"],
}
SECTIONS = [
    ("ex-scorecards", "Scorecards"), ("ex-horizons", "Horizons"), ("ex-regimes", "Regimes"),
    ("ex-backtest", "Backtest"), ("ex-var", "Value at Risk"), ("ex-learning", "Learning curves"),
    ("ex-calibration", "Calibration"), ("ex-features", "Features"), ("ex-extra", "Extra pairs"),
    ("ex-data", "Data"),
]
PROMPT = ("Select any triangle to see the test behind it: the loss gap, the Diebold–Mariano "
          "statistic and how the advantage built up month by month.")
GROUP_OPTIONS = [("all", "All"), ("Benchmarks", "Benchmarks"), ("Classical", "Classical"),
                 ("Regime-aware", "Regime-aware"), ("Neural", "Neural")]
SIZING_OPTIONS = [("unit", "Unit size"), ("vt_lightgbm", "Vol target: LightGBM"),
                  ("vt_har", "Vol target: HAR-RV"), ("all", "All")]


def horizon_card(data: dict, task: str, h: int) -> dict:
    hz = data["horizons"][task]
    cells = [c for c in hz["cells"] if c["h"] == h]
    summary = []
    for column in hz["columns"]:
        mine = [c for c in cells if c["model"] == column["id"]]
        summary.append({"model": column["id"], "better": sum(c["v"] == 2 for c in mine),
                        "worse": sum(c["v"] == -2 for c in mine)})
    return {"columns": hz["columns"], "cells": cells, "summary": summary}


def horizon_blocks(data: dict, task: str) -> list[dict]:
    return [block(data, task, f"h{h}:{task}", horizon_card(data, task, h),
                  models=HORIZON_MODELS[task], title="Next day" if h == 1 else f"Next {h} days")
            for h in data["horizons"][task]["horizons"]]


def _labels(data: dict) -> dict[str, str]:
    labels = {}
    for card in data["scorecards"].values():
        labels |= {c["id"]: c["label"] for c in card["columns"]}
    return labels


def _seg(element_id: str, label: str, options: list[tuple[str, str]], active: str) -> str:
    buttons = "".join(
        f'<button type="button" data-value="{value}" '
        f'aria-pressed="{"true" if value == active else "false"}">{escape(text)}</button>'
        for value, text in options)
    return (f'<div><label id="{element_id}-label">{escape(label)}</label>'
            f'<div class="seg" id="{element_id}" role="group" '
            f'aria-labelledby="{element_id}-label">{buttons}</div></div>')


def _cards(attribute: str, evidence: str, tables: dict[str, str], first: str) -> str:
    return "".join(
        f'<div class="sc-wrap" {attribute}="{key}" data-evidence="{evidence}"'
        f'{"" if key == first else " hidden"}>{table}</div>' for key, table in tables.items())


def _extra(data: dict, labels: dict[str, str]) -> tuple[str, str]:
    """The extra pairs' explorer section and its presentation slide template."""
    extra = data.get("extra")
    if not extra:
        return "", ""
    tasks, pl = data["meta"]["tasks"], data["meta"]["pair_labels"]
    tables = {t: scorecard_table([block(data, t, f"x:{t}", extra["scorecards"][t])],
                                 extra["pairs"], pl, card_id=f"sc-x-{t}", labels=labels,
                                 summary_rows=False,
                                 caption=f"{tasks[t]['label']} on the extra pairs")
              for t in tasks}
    daily = ", ".join(pl[p] for p in extra["daily_only"])
    controls = _seg("x-task", "Task", [(t, tasks[t]["label"]) for t in tasks], "volatility")
    section = (
        '<section class="panel" id="ex-extra" aria-labelledby="ex-extra-h">'
        '<h2 id="ex-extra-h">Extra pairs</h2>'
        '<p class="lede">EUR/GBP, a cross with no dollar leg, and USD/INR, a managed currency, '
        "run through the same walk-forward tests as the core pairs but stay out of every core "
        f"table and average. {escape(daily)} has no free intraday data: its prices are the "
        "Federal Reserve's noon rates, so it has no volatility task, lacks the intraday "
        "features, and takes every input dated at the 17:00 close from the day before.</p>"
        f'<div class="controls">{controls}</div>'
        + _cards("data-x-card", "ev-extra", tables, "volatility") + legend()
        + f'<div class="evidence empty" id="ev-extra" aria-live="polite">{PROMPT}</div>'
        "<h3>What the extra pairs show</h3>"
        '<ul class="findings">' + "".join(f"<li>{escape(t)}</li>"
                                          for t in extra["summary"]["sentences"]) + "</ul>"
        + (f'<p class="sc-note">{escape(daily)}\'s price is one indicative quote taken after '
           "the onshore market has closed, so some of its predictability may be slow "
           "adjustment in a thin quote rather than a tradable edge.</p>"
           if extra["summary"]["daily_only_beats_chance"] else "")
        + '<h3>Strategies on the extra pairs, net of costs</h3><div id="x-backtest"></div>'
        "</section>")
    slide = scorecard_table(
        [block(data, "volatility", "x:volatility", extra["scorecards"]["volatility"],
               models=["ridge", "random_forest", "lightgbm", "mlp"]),
         block(data, "direction", "x:direction", extra["scorecards"]["direction"])],
        extra["pairs"], pl, card_id="tpl-x", labels=labels, summary_rows=False,
        caption="Extra pairs")
    template = f'<template id="tpl-present-extra"><div class="sc-wrap">{slide}</div></template>'
    return section, template


def render_page(data: dict) -> str:
    meta = data["meta"]
    pairs, labels, pl, tasks = meta["pairs"], _labels(data), meta["pair_labels"], meta["tasks"]
    scorecards = data["scorecards"]

    def table(blocks, card_id, caption, **kw):
        return scorecard_table(blocks, pairs, pl, card_id=card_id, labels=labels,
                               caption=caption, **kw)

    master = table([block(data, t, t, scorecards[t], models=VERDICT_MODELS[t]) for t in tasks],
                   "sc-master", "Each model against its task's reference, pair by pair")
    task_tables = {t: table([block(data, t, t, scorecards[t])], f"sc-{t}",
                            f"{tasks[t]['label']}: every model against its reference")
                   for t in tasks}
    horizon_tables = {t: table(horizon_blocks(data, t), f"sc-hz-{t}",
                               f"{tasks[t]['label']} by forecast horizon")
                      for t in ("volatility", "direction")}
    extra_section, extra_template = _extra(data, labels)
    templates = (
        '<template id="tpl-present-volatility"><div class="sc-wrap">'
        + table([block(data, "volatility", "volatility", scorecards["volatility"],
                       models=VERDICT_MODELS["volatility"])], "tpl-vol", "Volatility")
        + legend() + '</div></template><template id="tpl-present-direction"><div class="sc-wrap">'
        + table([block(data, t, t, scorecards[t], models=VERDICT_MODELS[t])
                 for t in ("direction", "returns")], "tpl-dir", "Direction and returns")
        + legend() + '</div></template><template id="tpl-present-horizons"><div class="sc-wrap">'
        + table(horizon_blocks(data, "volatility"), "tpl-hz", "Volatility by horizon")
        + "</div></template>" + extra_template)
    n_extra = len(meta["extra_pairs"])
    task_options = [(t, tasks[t]["label"]) for t in tasks]

    fragments = {
        "title": escape(meta["title"]), "course": escape(meta["course"]),
        "test_start": meta["test_start"], "test_end": meta["test_end"],
        "first_test_year": meta["test_start"][:4], "refit": escape(meta["refit"]),
        "alpha": f"{meta['alpha']:.0%}",
        "pair_note": f"{len(pairs)} majors" + (f" + {n_extra} extra" if n_extra else ""),
        "data_to": meta["data_to"],
        "style": stylesheet(), "sprite": SPRITE, "legend": legend(), "prompt": PROMPT,
        "master": master,
        "questions": "".join(question_html(q) for q in data["questions"]),
        "index": "".join(f'<a href="#{i}">{escape(t)}</a>' for i, t in SECTIONS
                         if i != "ex-extra" or extra_section),
        "task_seg": _seg("ex-task", "Task", task_options, "volatility"),
        "group_seg": _seg("ex-group", "Models", GROUP_OPTIONS, "all"),
        "task_cards": _cards("data-task-card", "ev-explorer", task_tables, "volatility"),
        "horizon_seg": _seg("hz-task", "Task", task_options[:2], "volatility"),
        "horizon_cards": _cards("data-hz-card", "ev-horizons", horizon_tables, "volatility"),
        "sizing_seg": _seg("bt-sizing", "Sizing", SIZING_OPTIONS, "unit"),
        "lc_target_seg": _seg("lc-target", "Target", [("high_volatility", "High-volatility day"),
                                                      ("direction", "Up day")], "high_volatility"),
        "lc_metric_seg": _seg("lc-metric", "Measure", [("roc_auc", "AUC"), ("log_loss", "Log loss"),
                                                       ("ece", "Calibration error")], "roc_auc"),
        "shap_seg": _seg("shap-task", "Task", task_options[:2], "volatility"),
        "extra_section": extra_section,
        "templates": templates,
        "payload": json.dumps(data, separators=(",", ":"), allow_nan=False).replace("</", "<\\/"),
        "script": files("fxml.dashboard").joinpath("assets/dashboard.js").read_text(),
    }
    skeleton = files("fxml.dashboard").joinpath("assets/page.html").read_text()
    return Template(skeleton).substitute(fragments)
