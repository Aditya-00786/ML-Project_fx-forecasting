"""HTML for the dashboard's scorecards, shared by the static page and the Streamlit app.

A scorecard is a table: pairs down the side, models across the top in groups, and one triangle
per cell (see ``fxml.dashboard.data`` for the verdict codes). The triangles are drawn from an
SVG sprite defined once per page, so each cell costs a few bytes.
"""

from __future__ import annotations

import re
from html import escape
from importlib.resources import files

VERDICT_WORDS = {
    2: "significantly better than",
    1: "better than, but not significantly,",
    -1: "worse than, but not significantly,",
    -2: "significantly worse than",
}
GLYPH_IDS = {2: "g-up-fill", 1: "g-up", -1: "g-down", -2: "g-down-fill"}

_HOLLOW = 'fill="none" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"'
_SYMBOLS = {
    "g-up-fill": '<path d="M6 1.6 11 10.4H1z" fill="currentColor"/>',
    "g-up": f'<path d="M6 2.9 9.8 9.65H2.2z" {_HOLLOW}/>',
    "g-down": f'<path d="M2.2 2.35H9.8L6 9.1z" {_HOLLOW}/>',
    "g-down-fill": '<path d="M1 1.6H11L6 10.4z" fill="currentColor"/>',
    "g-none": '<path d="M3.5 6H8.5" stroke="currentColor" stroke-width="1.2"/>',
}
SPRITE = ('<svg class="sc-sprite" aria-hidden="true" width="0" height="0" focusable="false">'
          + "".join(f'<symbol id="{i}" viewBox="0 0 12 12">{p}</symbol>'
                    for i, p in _SYMBOLS.items())
          + "</svg>")


def stylesheet() -> str:
    """The dashboard's tokens and component styles (one file for both forms)."""
    return files("fxml.dashboard").joinpath("assets/dashboard.css").read_text()


_SECTION = "/* " + "-" * 64
_DARK_ROOT = re.compile(r':root\[data-theme="dark"\] \{.*?\n\}\n', re.S)


def tokens(theme: str = "light") -> dict[str, str]:
    """The colour tokens of one theme, as ``{name: value}`` without the leading dashes."""
    css = stylesheet()
    light = css[css.index(":root {"):css.index("}", css.index(":root {"))]
    values = dict(re.findall(r"--([\w-]+):\s*([^;]+);", light))
    if theme == "dark":
        values |= dict(re.findall(r"--([\w-]+):\s*([^;]+);", _DARK_ROOT.search(css).group(0)))
    values = {k: v.strip() for k, v in values.items()}
    for key, value in values.items():  # resolve references such as var(--n10)
        while value.startswith("var(--"):
            value = values[value[6:-1]]
        values[key] = value
    return values


def component_css(theme: str = "auto") -> str:
    """Only the scorecard, legend and research-question styles, for embedding in a host page
    (the Streamlit app) without touching its own layout.

    With ``theme="auto"`` every colour is a ``light-dark()`` pair, so the browser picks the
    host's current scheme itself and follows a theme switch at once, with no rerun; "light"
    or "dark" pins one theme."""
    css = stylesheet()
    start, end = css.index(f"{_SECTION} verdicts view */"), css.index(f"{_SECTION} evidence */")
    if theme == "auto":
        light, dark = tokens("light"), tokens("dark")
        pairs = {k: v if v == dark[k] else f"light-dark({v}, {dark[k]})" for k, v in light.items()}
    else:
        pairs = tokens(theme)
    root = ":root {" + "".join(f"--{k}: {v};" for k, v in pairs.items()) + "}"
    return root + ".sr-only{position:absolute;width:1px;height:1px;overflow:hidden;" \
        "clip:rect(0 0 0 0);white-space:nowrap}" + css[start:end]


def verdict_class(code: int) -> str:
    """CSS class of a verdict code: v2, v1, v0, vn1, vn2."""
    return f"v{code}" if code >= 0 else f"vn{-code}"


def glyph(code: int, inline: bool = False) -> str:
    """A cell's triangle: a reference to the page's SVG sprite, or, where a host page strips
    SVG (Streamlit), the same shape drawn in CSS (``inline``)."""
    symbol = GLYPH_IDS.get(code, "g-none")
    if inline:
        return f'<span class="gl gl-{symbol[2:]}" aria-hidden="true"></span>'
    return f'<svg class="g" aria-hidden="true"><use href="#{symbol}"/></svg>'


def describe(model_label: str, pair_label: str, reference_label: str, code: int) -> str:
    if code == 0:
        return f"{model_label} on {pair_label}: no test"
    return f"{model_label} on {pair_label}: {VERDICT_WORDS[code]} {reference_label}"


def scorecard_table(
    blocks: list[dict], pairs: list[str], pair_labels: dict[str, str], *, card_id: str,
    caption: str, labels: dict[str, str], summary_rows: bool = True, inline: bool = False,
) -> str:
    """One scorecard table.

    ``blocks`` are column groups side by side, each ``{"key", "title", "subtitle", "task",
    "reference_label", "columns": [{"id", "label", "group"}], "cells": {(pair, model): cell},
    "summary": {model: {"better", "worse"}}}``. A block's ``key`` tags its cells so the page
    can find the evidence behind them.
    """
    head_top = ['<th class="sc-corner" scope="col"><span class="sr-only">Pair</span></th>']
    head_models = ['<th class="sc-corner" scope="col"></th>']
    for b, block in enumerate(blocks):
        span = len(block["columns"])
        head_top.append(
            f'<th class="sc-block{" sc-block-first" if b else ""}" scope="colgroup" '
            f'colspan="{span}"><span class="sc-block-title">{escape(block["title"])}</span>'
            f'<span class="sc-block-sub">{escape(block["subtitle"])}</span></th>')
        previous_group = None
        for i, column in enumerate(block["columns"]):
            classes = ["sc-col"]
            if i == 0 and b:
                classes.append("sc-edge")
            elif column["group"] != previous_group and i:
                classes.append("sc-group")
            previous_group = column["group"]
            head_models.append(
                f'<th class="{" ".join(classes)}" scope="col" data-block="{block["key"]}" '
                f'data-model="{column["id"]}" data-group="{column["group"]}" '
                f'title="{escape(column["label"])} ({escape(column["group"])})">'
                f'<span>{escape(column["label"])}</span></th>')
    rows = []
    for pair in pairs:
        cells = [f'<th class="sc-pair" scope="row">{escape(pair_labels[pair])}</th>']
        for b, block in enumerate(blocks):
            previous_group = None
            for i, column in enumerate(block["columns"]):
                cell = block["cells"].get((pair, column["id"]))
                code = cell["v"] if cell else 0
                classes = ["sc-cell", verdict_class(code)]
                if i == 0 and b:
                    classes.append("sc-edge")
                elif column["group"] != previous_group and i:
                    classes.append("sc-group")
                previous_group = column["group"]
                label = describe(labels.get(column["id"], column["label"]), pair_labels[pair],
                                 block["reference_label"], code) if cell else \
                    f"{column['label']} on {pair_labels[pair]}: not run"
                cells.append(
                    f'<td class="{" ".join(classes)}" role="gridcell" tabindex="-1" '
                    f'data-block="{block["key"]}" data-pair="{pair}" '
                    f'data-model="{column["id"]}" data-group="{column["group"]}" '
                    f'aria-label="{escape(label)}" '
                    f'title="{escape(label)}">{glyph(code if cell else 99, inline)}</td>')
        rows.append(f'<tr>{"".join(cells)}</tr>')
    foot = ""
    if summary_rows:
        lines = []
        for kind, word in (("better", f"Significantly better, of {len(pairs)}"),
                           ("worse", f"Significantly worse, of {len(pairs)}")):
            cells = [f'<th class="sc-pair sc-count-label" scope="row">{word}</th>']
            for b, block in enumerate(blocks):
                previous_group = None
                for i, column in enumerate(block["columns"]):
                    count = block["summary"].get(column["id"], {}).get(kind, 0)
                    classes = ["sc-count", f"sc-{kind}" if count else "sc-zero"]
                    if i == 0 and b:
                        classes.append("sc-edge")
                    elif column["group"] != previous_group and i:
                        classes.append("sc-group")
                    previous_group = column["group"]
                    cells.append(f'<td class="{" ".join(classes)}" data-group="{column["group"]}">'
                                 f'{count if count else "·"}</td>')
            lines.append(f'<tr>{"".join(cells)}</tr>')
        foot = f'<tfoot>{"".join(lines)}</tfoot>'
    return (f'<table class="sc" id="{card_id}" role="grid" aria-label="{escape(caption)}">'
            f'<caption class="sr-only">{escape(caption)}</caption>'
            f'<thead><tr class="sc-head-top">{"".join(head_top)}</tr>'
            f'<tr class="sc-head-models">{"".join(head_models)}</tr></thead>'
            f'<tbody>{"".join(rows)}</tbody>{foot}</table>')


def block(data: dict, task: str, key: str, card: dict, *, models: list[str] | None = None,
          title: str | None = None) -> dict:
    """A scorecard block for one task from a collected scorecard (``data["scorecards"][task]``
    or an extra pairs one), optionally limited to ``models`` in that order."""
    spec = data["meta"]["tasks"][task]
    reference = spec["reference"]
    columns = [c for c in card["columns"] if c["id"] != reference]
    if models is not None:
        by_id = {c["id"]: c for c in columns}
        columns = [by_id[m] for m in models if m in by_id]
    reference_label = next((c["label"] for c in card["columns"] if c["id"] == reference),
                           reference)
    loss = {"volatility": "QLIKE", "direction": "Brier score", "returns": "squared error"}[task]
    return {
        "key": key, "task": task, "title": title or spec["label"],
        "subtitle": f"{loss} vs {reference_label}", "reference_label": reference_label,
        "columns": columns,
        "cells": {(c["pair"], c["model"]): c for c in card["cells"]},
        "summary": {s["model"]: s for s in card["summary"]},
    }


def question_html(q: dict, goto: bool = True) -> str:
    """One research question as a row that unfolds: its verdict and one-line result always in
    view, the full answer, tests and source tables inside."""
    evidence = "".join(f'<li><span>{escape(e["label"])}: {escape(e["test"])}</span>'
                       f'<span class="path">{escape(e["source"])}</span></li>'
                       for e in q["evidence"])
    button = ""
    if goto:
        view = q.get("view", {})
        target = (f"ex-scorecards:{view['task']}" if "task" in view else
                  {"backtest": "ex-backtest", "learning": "ex-learning"}[view["section"]])
        button = (f'<button type="button" class="quiet" data-goto="{target}">'
                  "Open in the Explorer</button>")
    return (f'<details class="rq"><summary><span class="rq-id">{q["id"]}</span>'
            f'<span class="rq-verdict">{escape(q["verdict"])}</span>'
            f'<span class="rq-short">{escape(q["short"])}</span></summary>'
            f'<div class="rq-body"><p class="rq-question">{escape(q["question"])}</p>'
            f'<p class="rq-answer">{escape(q["answer"])}</p>'
            f'<ul class="rq-evidence">{evidence}</ul>{button}</div></details>')


def legend(inline: bool = False) -> str:
    items = [(2, "significantly better"), (1, "better, not significant"),
             (-1, "worse, not significant"), (-2, "significantly worse")]
    parts = "".join(f'<li class="{verdict_class(c)}">{glyph(c, inline)}<span>{w}</span></li>'
                    for c, w in items)
    return f'<ul class="sc-legend">{parts}</ul>'
