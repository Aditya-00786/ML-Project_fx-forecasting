"""The results dashboard as a Streamlit app: the same verdicts, explorer and presentation as
reports/dashboard.html, with Streamlit's own controls for picking any cell, strategy or chart.

Reads reports/dashboard/data.json, written by scripts/build_dashboard.py.

    uv run --group dashboard streamlit run dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from fxml.dashboard.page import HORIZON_MODELS, VERDICT_MODELS, horizon_blocks
from fxml.dashboard.render import (
    block,
    component_css,
    legend,
    question_html,
    scorecard_table,
)

DATA = Path(__file__).resolve().parents[1] / "reports" / "dashboard" / "data.json"
LOSS = {"volatility": "QLIKE", "direction": "Brier score", "returns": "squared error"}
VERDICT = {2: "significantly better than", 1: "better than, but not significantly,",
           -1: "worse than, but not significantly,", -2: "significantly worse than",
           0: "not tested against"}
GROUPS = ["Benchmarks", "Classical", "Regime-aware", "Neural"]

st.set_page_config(page_title="FX forecasting results", layout="wide")


@st.cache_data
def load(path: str, stamp: float) -> dict:
    return json.loads(Path(path).read_text())


if not DATA.exists():
    st.error("No dashboard data yet. Run `uv run python scripts/build_dashboard.py` first, then "
             "reload this page.")
    st.stop()
D = load(str(DATA), DATA.stat().st_mtime)
META, TASKS, PL = D["meta"], D["meta"]["tasks"], D["meta"]["pair_labels"]
# Colours never depend on which theme the script thinks is active: Streamlit reports a theme
# switch only on the next rerun, so the styles below resolve light or dark in the browser and
# the charts use colours that read on both grounds (axes and grids follow Streamlit's theme).
CHART = {"neutral": "#7d7c76", "course": "#7b68ee", "ochre": "#b07d10", "better": "#4a8fe0",
         "band": "rgba(128, 128, 128, 0.13)", "others": "rgba(128, 128, 128, 0.45)"}
st.html(f"<style>{component_css('auto')} .rq-verdict{{margin:0}} "
        f".slide-title{{font-size:2.4rem;font-weight:650;line-height:1.1;letter-spacing:-0.02em;"
        "margin:0 0 .4rem} .slide-take{font-size:1.25rem;color:var(--ink-2);max-width:60ch}"
        "</style>")
LABELS = {c["id"]: c["label"] for card in D["scorecards"].values() for c in card["columns"]}
for card in D["horizons"].values():
    LABELS |= {c["id"]: c["label"] for c in card["columns"]}


def label(model: str) -> str:
    return LABELS.get(model, model)


# ------------------------------------------------------------ chart helpers


def styled(chart: alt.Chart, height: int = 260) -> alt.Chart:
    """One look for every chart: thin marks and quiet axes, whose colours follow Streamlit's
    theme so they switch with it."""
    return (chart.properties(height=height, background="transparent")
            .configure_view(stroke=None)
            .configure_axis(labelFontSize=12, titleFontSize=12, titleFontWeight="normal")
            .configure_legend(orient="top"))


def stress_layer(start: str | None = None) -> alt.Chart:
    rows = pd.DataFrame(META["stress"])
    if start:
        rows = rows[rows["end"] >= start].assign(start=lambda r: r["start"].clip(lower=start))
    return alt.Chart(rows).mark_rect(color=CHART["band"]).encode(
        x="start:T", x2="end:T", tooltip=[alt.Tooltip("name:N", title="Stress episode")])


def months(values: list[str]) -> pd.DatetimeIndex:
    return pd.to_datetime(values)


def p_text(p) -> str:
    if p is None:
        return "–"
    return "< 0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


# ------------------------------------------------------------ scorecards and evidence


def scorecard(blocks: list[dict], pairs: list[str], caption: str, summary: bool = True) -> None:
    st.html('<div class="sc-wrap">' + scorecard_table(
        blocks, pairs, PL, card_id="sc", labels=LABELS, caption=caption, summary_rows=summary,
        inline=True) + "</div>" + legend(inline=True))


def lookup(kind: str, task: str, pair: str, model: str) -> dict:
    if kind == "extra":
        card, trace_months = D["extra"]["scorecards"][task], D["extra"]["months"]
        trace = D["extra"]["traces"][task].get(pair, {}).get(model)
        source = f"reports/tables/extra_pairs_{task}.csv"
    else:
        card, trace_months = D["scorecards"][task], D["months"]
        trace = D["traces"][task].get(pair, {}).get(model)
        source = f"reports/tables/ml_{task}.csv"
    cells = {(c["pair"], c["model"]): c for c in card["cells"]}
    reference = TASKS[task]["reference"]
    return {"cell": cells.get((pair, model)), "ref": cells.get((pair, reference)),
            "reference": reference, "trace": trace, "months": trace_months, "source": source,
            "models": [c["id"] for c in card["columns"] if c["id"] != reference]}


def evidence(kind: str, key: str, pairs: list[str], task: str | None = None) -> None:
    """Pick a cell (task, pair, model) and show the test behind it."""
    st.markdown("##### The test behind a cell")
    cols = st.columns(3)
    if task is None:
        available = [t for t in TASKS if kind != "extra" or D["extra"]["scorecards"][t]["cells"]]
        task = cols[0].selectbox("Task", available, format_func=lambda t: TASKS[t]["label"],
                                 key=f"{key}-task")
    else:
        cols[0].selectbox("Task", [task], format_func=lambda t: TASKS[t]["label"],
                          key=f"{key}-task", disabled=True)
    card_pairs = sorted({c["pair"] for c in (D["extra"]["scorecards"][task] if kind == "extra"
                                             else D["scorecards"][task])["cells"]})
    pair = cols[1].selectbox("Pair", [p for p in pairs if p in card_pairs],
                             format_func=lambda p: PL[p], key=f"{key}-pair")
    options = lookup(kind, task, pair, TASKS[task]["reference"])["models"]
    default = options.index("lightgbm") if "lightgbm" in options else 0
    model = cols[2].selectbox("Model", options, index=default, format_func=label,
                              key=f"{key}-model")
    info = lookup(kind, task, pair, model)
    cell, ref = info["cell"], info["ref"]
    if not cell:
        st.info(f"{label(model)} was not run on {PL[pair]}.")
        return
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        value, base = cell["value"], ref["value"] if ref else None
        if task == "volatility" and value is not None and base:
            gap = value / base - 1
            head = (f"QLIKE {value:.4f} against {label(info['reference'])}'s {base:.4f} "
                    f"({abs(gap):.1%} {'lower' if gap < 0 else 'higher'})")
        elif task == "direction":
            head = f"AUC {value:.3f} against {base:.3f} for {label(info['reference'])}"
        else:
            head = f"Out-of-sample R² {value:.4f} (a zero forecast scores 0)"
        st.markdown(f"**{label(model)}** on {PL[pair]}: {head}. On {LOSS[task]}, it is "
                    f"{VERDICT[cell['v']]} {label(info['reference'])}.")
        rows = [("Diebold–Mariano statistic", f"{cell['dm']:.2f}"), ("p-value", p_text(cell["p"]))]
        if cell.get("pt_p") is not None:
            rows.append(("Pesaran–Timmermann p (up/down calls)", p_text(cell["pt_p"])))
        if cell.get("days"):
            rows.append(("Test days", f"{cell['days']:,}"))
        st.dataframe(pd.DataFrame(rows, columns=["Test", "Value"]), hide_index=True,
                     width="stretch")
        st.caption(f"Source: `{info['source']}`")
    with right:
        if info["trace"]:
            frame = pd.DataFrame({"month": months(info["months"]), "advantage": info["trace"]})
            line = alt.Chart(frame).mark_line(color=CHART["course"], strokeWidth=2.5).encode(
                x=alt.X("month:T", title=None),
                y=alt.Y("advantage:Q", title=f"Running {LOSS[task]} advantage"),
                tooltip=[alt.Tooltip("month:T", format="%Y-%m"),
                         alt.Tooltip("advantage:Q", format=".3f")])
            zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(
                color=CHART["neutral"]).encode(y="y:Q")
            area = alt.Chart(frame).mark_area(color=CHART["course"], opacity=0.1).encode(
                x="month:T", y="advantage:Q")
            st.altair_chart(styled(stress_layer(info["months"][0]) + area + zero + line, 220),
                            width="stretch")
            st.caption(f"Cumulative {LOSS[task]} of {label(info['reference'])} minus "
                       f"{label(model)}, month by month: rising means {label(model)} is pulling "
                       "ahead. Shaded: stress episodes.")


def regime_chart(height: int) -> alt.Chart:
    r = D["regimes"]["timeline"]
    frame = pd.DataFrame({"week": pd.to_datetime(r["dates"]), "Gaussian mixture": r["gmm"],
                          "Hidden Markov": r["hmm"]}).melt("week", var_name="model",
                                                           value_name="p")
    lines = alt.Chart(frame).mark_line(strokeWidth=1.6).encode(
        x=alt.X("week:T", title=None),
        y=alt.Y("p:Q", title="Probability of the turbulent regime", axis=alt.Axis(format="%")),
        color=alt.Color("model:N", title=None, scale=alt.Scale(
            domain=["Gaussian mixture", "Hidden Markov"],
            range=[CHART["neutral"], CHART["ochre"]])),
        tooltip=["model:N", alt.Tooltip("week:T"), alt.Tooltip("p:Q", format=".0%")])
    return styled(stress_layer() + lines, height)


def equity_chart(key: str, sizing: str, height: int) -> alt.Chart:
    """Every strategy at one sizing in grey, with ``key`` traced in the selection colour."""
    b = D["backtest"]
    curves = pd.DataFrame({k: v for k, v in b["curves"].items() if k.endswith(f"|{sizing}")})
    curves["month"] = months(b["months"])
    frame = curves.melt("month", var_name="strategy", value_name="cumulative")
    others = alt.Chart(frame[frame["strategy"] != key]).mark_line(
        color=CHART["others"], strokeWidth=1).encode(x=alt.X("month:T", title=None),
                                             y=alt.Y("cumulative:Q", title="Cumulative net return",
                                                     axis=alt.Axis(format="%")),
                                             detail="strategy:N")
    mine = alt.Chart(frame[frame["strategy"] == key]).mark_line(
        color=CHART["course"], strokeWidth=2.5).encode(
        x="month:T", y="cumulative:Q",
        tooltip=[alt.Tooltip("month:T", format="%Y-%m"),
                 alt.Tooltip("cumulative:Q", format=".1%")])
    return styled(stress_layer() + others + mine, height)


def learning_chart(target: str, metric: str, height: int) -> alt.Chart:
    rows = pd.DataFrame(D["learning"]["rows"])
    rows = rows[rows["target"] == target]
    rows["window"] = rows["size"].map(lambda s: "full" if s == 0 else str(s))
    order = ["21", "63", "126", "252", "504", "1008", "full"]
    rows["model"] = rows["model"].map({"logistic": "Logistic regression",
                                       "naive_bayes": "Naive Bayes"})
    chart = alt.Chart(rows).mark_line(point=True, strokeWidth=2).encode(
        x=alt.X("window:O", sort=order, title="Training days", axis=alt.Axis(labelAngle=0)),
        y=alt.Y(f"{metric}:Q", title=None, scale=alt.Scale(zero=False)),
        color=alt.Color("model:N", title=None, scale=alt.Scale(
            domain=["Logistic regression", "Naive Bayes"],
            range=[CHART["neutral"], CHART["ochre"]])),
        tooltip=["model:N", "window:O", alt.Tooltip(f"{metric}:Q", format=".3f")])
    return styled(chart, height)


# ------------------------------------------------------------ pages


def verdicts() -> None:
    st.title(META["title"])
    st.caption(f"{META['course']} · test period {META['test_start']} to {META['test_end']} · "
               f"walk-forward, refitted {META['refit']} · Diebold–Mariano at "
               f"{META['alpha']:.0%}")
    st.subheader("Every model against its benchmark, pair by pair")
    st.caption("Each triangle is one model on one pair over the test period, judged by a "
               "Diebold–Mariano test of its daily loss against the task's reference. Hover "
               "a cell for its verdict.")
    scorecard([block(D, t, t, D["scorecards"][t], models=VERDICT_MODELS[t]) for t in TASKS],
              META["pairs"], "Every model against its benchmark")
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        evidence("core", "verdicts", META["pairs"])
    with right:
        st.subheader("Five research questions")
        st.caption("Open a question for its answer, tests and source tables.")
        st.html("".join(question_html(q, goto=False) for q in D["questions"]))


def explorer() -> None:
    st.title("Explorer")
    tabs = st.tabs(["Scorecards", "Horizons", "Regimes", "Backtest", "Value at Risk",
                    "Learning curves", "Calibration", "Features", "Extra pairs", "Data"])
    with tabs[0]:
        c1, c2 = st.columns([1, 2])
        task = c1.segmented_control("Task", list(TASKS), default="volatility",
                                    format_func=lambda t: TASKS[t]["label"], key="ex-task")
        task = task or "volatility"
        groups = c2.pills("Model families", GROUPS, selection_mode="multi", default=GROUPS,
                          key="ex-groups")
        card = D["scorecards"][task]
        models = [c["id"] for c in card["columns"] if c["group"] in (groups or GROUPS)]
        scorecard([block(D, task, task, card, models=models)], META["pairs"], TASKS[task]["label"])
        evidence("core", "explorer", META["pairs"], task)
    with tabs[1]:
        task = st.segmented_control("Task", ["volatility", "direction"], default="volatility",
                                    format_func=lambda t: TASKS[t]["label"], key="hz-task")
        task = task or "volatility"
        st.caption("Forecasts of the average variance and the summed return over the next 5 and "
                   "22 trading days; the tests allow for overlapping forecasts.")
        scorecard(horizon_blocks(D, task), META["pairs"], "By horizon")
        st.caption(f"Models shown: {', '.join(label(m) for m in HORIZON_MODELS[task])}.")
    with tabs[2]:
        st.altair_chart(regime_chart(260), width="stretch")
        summary = pd.DataFrame(D["regimes"]["summary"])
        summary["method"] = summary["method"].map({"gmm": "Gaussian mixture",
                                                   "hmm": "Hidden Markov"})
        summary["regime"] = summary["regime"].map({0: "calm", 1: "turbulent"})
        st.dataframe(summary, hide_index=True, width="stretch")
    with tabs[3]:
        backtest()
    with tabs[4]:
        var = pd.DataFrame(D["backtest"]["var"])
        var["model"] = var["model"].map(label)
        st.caption("99% one-day Value at Risk; a forecast passes when exceedances match the 1% "
                   "target (Kupiec) and do not cluster (Christoffersen).")
        st.dataframe(var, hide_index=True, width="stretch", column_config={
            "rate": st.column_config.NumberColumn("Exceedance rate", format="percent")})
    with tabs[5]:
        learning_curves()
    with tabs[6]:
        cal = pd.DataFrame(D["calibration"]).dropna(subset=["calibration_slope"])
        cal = cal[~cal["model"].str.startswith("baseline_")]
        cal = cal.assign(lo=cal["calibration_slope"] - 1.96 * cal["calibration_slope_se"],
                         hi=cal["calibration_slope"] + 1.96 * cal["calibration_slope_se"])
        y = alt.Y("label:N", title=None, sort=alt.EncodingSortField("calibration_slope"))
        bars = alt.Chart(cal).mark_rule(color=CHART["neutral"]).encode(
            y=y, x=alt.X("lo:Q", title="Calibration slope (1 = well calibrated)"), x2="hi:Q")
        dots = alt.Chart(cal).mark_point(filled=True, color=CHART["neutral"], size=50).encode(
            y=y, x="calibration_slope:Q",
            tooltip=["label:N", alt.Tooltip("calibration_slope:Q", format=".2f"),
                     alt.Tooltip("calibration_slope_se:Q", format=".2f", title="standard error"),
                     alt.Tooltip("ece:Q", format=".3f", title="calibration error")])
        ideal = alt.Chart(pd.DataFrame({"x": [1]})).mark_rule(
            color=CHART["better"], strokeDash=[4, 3]).encode(x="x:Q")
        st.altair_chart(styled(bars + dots + ideal, 22 * len(cal) + 40),
                        width="stretch")
    with tabs[7]:
        task = st.segmented_control("Task", ["volatility", "direction"], default="volatility",
                                    format_func=lambda t: TASKS[t]["label"], key="shap-task")
        shap = pd.DataFrame(D["shap"][task or "volatility"])
        chart = alt.Chart(shap).mark_bar(color=CHART["neutral"], height=14).encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("mean_abs_shap:Q", title="Mean absolute SHAP value"),
            tooltip=["feature:N", "group:N", alt.Tooltip("mean_abs_shap:Q", format=".4f")])
        st.altair_chart(styled(chart, 22 * len(shap) + 30), width="stretch")
    with tabs[8]:
        extra_pairs()
    with tabs[9]:
        st.dataframe(pd.DataFrame(D["data"]["rows"]), hide_index=True, width="stretch")


def backtest() -> None:
    b = D["backtest"]
    sizing = st.segmented_control("Sizing", list(b["sizing_labels"]), default="unit",
                                  format_func=lambda s: b["sizing_labels"][s], key="bt-sizing")
    sizing = sizing or "unit"
    table = pd.DataFrame(b["portfolio"])
    table = table[(table["sizing"] == sizing) & (table["signal"] != "flat")]
    table = table.sort_values("sharpe", ascending=False).reset_index(drop=True)
    table.insert(0, "strategy", table["signal"].map(
        lambda s: b["strategy_labels"].get(s.removesuffix("_confident"), s)
        + (" (confident)" if s.endswith("_confident") else "")))
    shown = table[["strategy", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown",
                   "p_value", "sharpe_2x_costs"]]
    st.caption("Equal-risk portfolio of the seven pairs, net of costs. Select a row to trace it.")
    picked = st.dataframe(shown, hide_index=True, width="stretch", on_select="rerun",
                          selection_mode="single-row", key="bt-table", column_config={
                              "ann_return": st.column_config.NumberColumn("Return a year",
                                                                          format="percent"),
                              "ann_vol": st.column_config.NumberColumn("Volatility",
                                                                       format="percent"),
                              "max_drawdown": st.column_config.NumberColumn("Max drawdown",
                                                                            format="percent")})
    rows = picked.selection.rows if picked and picked.selection else []
    chosen = table.iloc[rows[0]] if rows else table.iloc[0]
    st.altair_chart(equity_chart(f"{chosen['signal']}|{sizing}", sizing, 280),
                    width="stretch")
    st.caption(f"Violet: {chosen['strategy']}. Grey: every other strategy at this sizing.")
    parts = pd.DataFrame(b["breakdown"])
    parts = parts[(parts["signal"] == chosen["signal"]) & (parts["sizing"] == sizing)]
    if not parts.empty:
        st.markdown(f"##### {chosen['strategy']} by regime, stress and subperiod")
        st.dataframe(parts.drop(columns=["signal", "sizing"]), hide_index=True,
                     width="stretch")


def learning_curves() -> None:
    c1, c2 = st.columns(2)
    target = c1.segmented_control("Target", ["high_volatility", "direction"],
                                  default="high_volatility", key="lc-target",
                                  format_func=lambda t: {"high_volatility": "High-volatility day",
                                                         "direction": "Up day"}[t])
    metric = c2.segmented_control("Measure", ["roc_auc", "log_loss", "ece"], default="roc_auc",
                                  key="lc-metric",
                                  format_func=lambda m: {"roc_auc": "AUC", "log_loss": "Log loss",
                                                         "ece": "Calibration error"}[m])
    st.altair_chart(learning_chart(target or "high_volatility", metric or "roc_auc", 280),
                    width="stretch")


def extra_pairs() -> None:
    extra = D.get("extra")
    if not extra:
        st.info("No extra pairs in this build.")
        return
    st.caption("EUR/GBP and USD/INR run through the same walk-forward tests but stay out of every "
               "core table. USD/INR has daily noon rates only: no volatility task, no intraday "
               "features, and inputs dated at the 17:00 close taken from the day before.")
    for sentence in extra["summary"]["sentences"]:
        st.markdown(f"- {sentence}")
    task = st.segmented_control("Task", list(TASKS), default="direction", key="x-task",
                                format_func=lambda t: TASKS[t]["label"]) or "direction"
    scorecard([block(D, task, f"x:{task}", extra["scorecards"][task])], extra["pairs"],
              "Extra pairs", summary=False)
    evidence("extra", "extra", extra["pairs"], task)
    st.markdown("##### Strategies, net of costs")
    st.dataframe(pd.DataFrame(extra["backtest"]), hide_index=True, width="stretch")


def present() -> None:
    q = {x["id"]: x for x in D["questions"]}
    slides = [
        (q["RQ1"]["headline"], q["RQ1"]["answer"],
         lambda: scorecard([block(D, "volatility", "volatility", D["scorecards"]["volatility"],
                                  models=VERDICT_MODELS["volatility"])],
                           META["pairs"], "Volatility")),
        (q["RQ2"]["headline"], q["RQ2"]["answer"],
         lambda: scorecard([block(D, t, t, D["scorecards"][t], models=VERDICT_MODELS[t])
                            for t in ("direction", "returns")], META["pairs"], "Direction")),
        (q["RQ1"]["horizon_headline"], q["RQ1"]["answer"].split(". ")[-1],
         lambda: scorecard(horizon_blocks(D, "volatility"), META["pairs"], "Horizons")),
        (q["RQ3"]["headline"], q["RQ3"]["answer"],
         lambda: st.altair_chart(regime_chart(420), width="stretch")),
        (q["RQ4"]["headline"], q["RQ4"]["answer"],
         lambda: st.altair_chart(equity_chart("carry|unit", "unit", 420),
                                 width="stretch")),
        (q["RQ5"]["headline"], q["RQ5"]["answer"],
         lambda: st.altair_chart(learning_chart("high_volatility", "roc_auc", 420),
                                 width="stretch")),
    ]
    i = st.session_state.setdefault("slide", 0)
    c1, c2, c3 = st.columns([1, 6, 1])
    if c1.button("Previous", width="stretch", disabled=i == 0):
        st.session_state.slide = i - 1
        st.rerun()
    c2.progress((i + 1) / len(slides), text=f"{i + 1} / {len(slides)}")
    if c3.button("Next", width="stretch", disabled=i == len(slides) - 1):
        st.session_state.slide = i + 1
        st.rerun()
    title, takeaway, body = slides[i]
    st.html(f'<p class="slide-title">{title}</p><p class="slide-take">{takeaway}</p>')
    body()


st.navigation([st.Page(verdicts, title="Verdicts", default=True),
               st.Page(explorer, title="Explorer"),
               st.Page(present, title="Present")]).run()
