/* Results dashboard: scorecard interaction, evidence, charts, explorer sections and slides.
   Reads the collected results from the JSON block the build script embeds; draws everything
   as SVG with no external library, so the page works offline. */
(() => {
  "use strict";

  const D = JSON.parse(document.getElementById("dashboard-data").textContent);
  const T = D.meta.tasks;
  const PL = D.meta.pair_labels;
  const NS = "http://www.w3.org/2000/svg";
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => [...r.querySelectorAll(s)];

  // ------------------------------------------------------------ labels and formatting

  const LABEL = {};
  const addColumns = (cols) => cols.forEach((c) => { LABEL[c.id] = c.label; });
  Object.values(D.scorecards).forEach((sc) => addColumns(sc.columns));
  Object.values(D.horizons).forEach((sc) => addColumns(sc.columns));
  if (D.extra) Object.values(D.extra.scorecards).forEach((sc) => addColumns(sc.columns));
  const label = (m) => LABEL[m] || m;
  const minus = (s) => s.replace(/^-/, "−");
  const fmt = {
    num: (v, d = 3) => (v == null ? "–" : minus(Number(v).toFixed(d))),
    pct: (v, d = 1) => (v == null ? "–" : minus((v * 100).toFixed(d)) + "%"),
    p: (v) => (v == null ? "–" : v < 0.001 ? "< 0.001" : v.toFixed(v < 0.01 ? 3 : 2)),
    int: (v) => (v == null ? "–" : Math.round(v).toLocaleString("en")),
  };
  const VERDICT = {
    2: "significantly better than", 1: "better than, but not significantly,",
    "-1": "worse than, but not significantly,", "-2": "significantly worse than", 0: "not tested against",
  };
  const LOSS = { volatility: "QLIKE", direction: "Brier score", returns: "squared error" };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ------------------------------------------------------------ theme

  const root = document.documentElement;
  const themeButton = $(".theme");
  const THEMES = ["auto", "light", "dark"];
  const THEME_NAMES = { auto: "Theme: system", light: "Theme: light", dark: "Theme: dark" };
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch (e) { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
  };
  function applyTheme(t) {
    if (t === "auto") root.removeAttribute("data-theme"); else root.dataset.theme = t;
    themeButton.textContent = THEME_NAMES[t];
    themeButton.dataset.theme = t;
    store.set("fx-theme", t);
    redrawAll();
  }
  themeButton.addEventListener("click", () => {
    const next = THEMES[(THEMES.indexOf(themeButton.dataset.theme || "auto") + 1) % 3];
    applyTheme(next);
  });

  // ------------------------------------------------------------ tooltip

  const tip = $(".tip");
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const w = tip.offsetWidth, h = tip.offsetHeight;
    let left = x + 14, top = y + 14;
    if (left + w > innerWidth - 8) left = x - w - 14;
    if (top + h > innerHeight - 8) top = y - h - 14;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };

  // ------------------------------------------------------------ chart kit

  const charts = new Set();
  function redrawAll() { charts.forEach((c) => c.draw()); }
  const resize = new ResizeObserver((entries) => {
    entries.forEach((e) => { const c = e.target.__chart; if (c && e.contentRect.width !== c.width) c.draw(); });
  });

  function el(tag, attrs, parent) {
    const node = document.createElementNS(NS, tag);
    for (const k in attrs) if (attrs[k] != null) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }
  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) { hi = lo + 1; }
    const span = hi - lo, step0 = span / n, mag = 10 ** Math.floor(Math.log10(step0));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => span / s <= n) || 10 * mag;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toFixed(10));
    return out;
  }
  const parseDate = (s) => (s.length === 7 ? new Date(`${s}-15T00:00:00`) : new Date(`${s}T00:00:00`));

  function chart(host, opts) {
    const c = { host, opts, width: 0, draw: null };
    host.classList.add("chart");
    host.__chart = c;
    c.draw = () => drawLine(c);
    charts.add(c);
    resize.observe(host);
    c.draw();
    return c;
  }

  function drawLine(c) {
    const { host, opts } = c;
    const width = host.clientWidth || 600;
    c.width = width;
    const height = opts.height || 220;
    const named = opts.series.filter((s) => s.label);
    const big = !!host.closest(".slide");  // projector type on slides
    const m = { top: opts.yLabel ? (big ? 34 : 26) : 12, right: named.length ? (opts.labelRoom || 96) * (big ? 1.4 : 1) : 16,
      bottom: big ? 34 : 26, left: (opts.left || 52) * (big ? 1.35 : 1) };
    const iw = Math.max(40, width - m.left - m.right), ih = height - m.top - m.bottom;
    host.innerHTML = "";
    const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": opts.aria || "" }, host);
    const xs = opts.x;
    const categorical = !!opts.categories;
    const x0 = categorical ? 0 : xs[0].getTime(), x1 = categorical ? xs.length - 1 : xs[xs.length - 1].getTime();
    const sx = (i) => m.left + (categorical ? (xs.length === 1 ? iw / 2 : (i / (xs.length - 1)) * iw)
      : ((xs[i].getTime() - x0) / (x1 - x0)) * iw);
    const sxDate = (d) => m.left + ((d.getTime() - x0) / (x1 - x0)) * iw;
    let lo = Infinity, hi = -Infinity;
    opts.series.forEach((s) => s.values.forEach((v) => { if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }));
    if (opts.zero) { lo = Math.min(lo, 0); hi = Math.max(hi, 0); }
    if (opts.yMin != null) lo = opts.yMin;
    if (opts.yMax != null) hi = opts.yMax;
    if (!isFinite(lo)) { lo = 0; hi = 1; }
    const pad = (hi - lo) * 0.06 || 1;
    if (opts.yMin == null) lo -= pad;
    if (opts.yMax == null) hi += pad;
    const ticks = niceTicks(lo, hi, opts.yTicks || 4);
    lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]);
    const sy = (v) => m.top + ih - ((v - lo) / (hi - lo)) * ih;
    const step = ticks.length > 1 ? ticks[1] - ticks[0] : 1;
    const decimals = Math.max(0, Math.min(4, -Math.floor(Math.log10(step) + 1e-9)));
    const yf = opts.yFormat || ((v) => fmt.num(v, decimals));

    if (opts.shade && !categorical) {
      const g = el("g", {}, svg);
      (opts.shade.periods || []).forEach((p) => {
        const a = parseDate(p.start), b = parseDate(p.end);
        if (b.getTime() < x0 || a.getTime() > x1) return;
        const xa = sxDate(new Date(Math.max(a, x0))), xb = sxDate(new Date(Math.min(b, x1)));
        el("rect", { class: "stress", x: xa, y: m.top, width: Math.max(1.5, xb - xa), height: ih }, g);
        if (opts.shade.labels && xb - xa >= 0) {
          const t = el("text", { class: "stress-label", x: xa + 2, y: m.top + 10 }, g);
          t.textContent = p.short;
        }
      });
    }
    const grid = el("g", { class: "grid" }, svg);
    const axis = el("g", { class: "axis" }, svg);
    ticks.forEach((t) => {
      el("line", { x1: m.left, x2: m.left + iw, y1: sy(t), y2: sy(t) }, grid);
      const tx = el("text", { x: m.left - 8, y: sy(t) + 3.5, "text-anchor": "end" }, axis);
      tx.textContent = yf(t);
    });
    if (categorical) {
      xs.forEach((cat, i) => {
        const tx = el("text", { x: sx(i), y: m.top + ih + (big ? 24 : 17), "text-anchor": "middle" }, axis);
        tx.textContent = cat;
      });
    } else {
      const y0 = xs[0].getFullYear(), y1 = xs[xs.length - 1].getFullYear();
      const every = Math.max(1, Math.ceil((y1 - y0 + 1) / Math.max(2, Math.floor(iw / 58))));
      for (let y = y0 + 1; y <= y1; y++) {
        if ((y - y0) % every) continue;
        const d = new Date(`${y}-01-01T00:00:00`);
        if (d.getTime() < x0 || d.getTime() > x1) continue;
        const tx = el("text", { x: sxDate(d), y: m.top + ih + (big ? 24 : 17), "text-anchor": "middle" }, axis);
        tx.textContent = y;
      }
    }
    el("line", { x1: m.left, x2: m.left + iw, y1: m.top + ih, y2: m.top + ih }, axis);
    if (opts.zero) el("line", { class: "zero", x1: m.left, x2: m.left + iw, y1: sy(0), y2: sy(0) }, svg);
    if (opts.yLabel) {
      const t = el("text", { x: 0, y: big ? 18 : 12, class: "ylabel" }, svg);
      t.textContent = opts.yLabel;
    }

    const pathOf = (values) => {
      let d = "", pen = false;
      values.forEach((v, i) => {
        if (v == null) { pen = false; return; }
        d += `${pen ? "L" : "M"}${sx(i).toFixed(1)} ${sy(v).toFixed(1)}`;
        pen = true;
      });
      return d;
    };
    if (opts.diverging && opts.series[0]) {
      const s = opts.series[0];
      const pts = s.values.map((v, i) => (v == null ? null : [sx(i), sy(v)])).filter(Boolean);
      if (pts.length > 1) {
        const id = `clip${Math.random().toString(36).slice(2, 8)}`;
        const defs = el("defs", {}, svg);
        const up = el("clipPath", { id: `${id}u` }, defs);
        el("rect", { x: m.left, y: m.top, width: iw, height: Math.max(0, sy(0) - m.top) }, up);
        const dn = el("clipPath", { id: `${id}d` }, defs);
        el("rect", { x: m.left, y: sy(0), width: iw, height: Math.max(0, m.top + ih - sy(0)) }, dn);
        const area = `M${pts[0][0]} ${sy(0)}` + pts.map((p) => `L${p[0]} ${p[1]}`).join("") + `L${pts[pts.length - 1][0]} ${sy(0)}Z`;
        el("path", { d: area, class: "area-up", "clip-path": `url(#${id}u)` }, svg);
        el("path", { d: area, class: "area-down", "clip-path": `url(#${id}d)` }, svg);
      }
    }
    const order = [...opts.series].sort((a, b) => (a.label ? 1 : 0) - (b.label ? 1 : 0));
    order.forEach((s) => el("path", { d: pathOf(s.values), class: `line ${s.cls || "ink"}` }, svg));
    // direct labels at the line ends, nudged apart
    const ends = named.map((s) => {
      let i = s.values.length - 1;
      while (i > 0 && s.values[i] == null) i--;
      return { s, y: sy(s.values[i] ?? 0), x: sx(i) };
    }).sort((a, b) => a.y - b.y);
    const gap = big ? 20 : 13;
    for (let i = 1; i < ends.length; i++) if (ends[i].y - ends[i - 1].y < gap) ends[i].y = ends[i - 1].y + gap;
    ends.forEach((e) => {
      const t = el("text", { x: e.x + 6, y: e.y + (big ? 6 : 4), class: `label ${e.s.cls || "ink"}` }, svg);
      t.textContent = e.s.label;
    });

    // hover crosshair
    const cross = el("line", { class: "cross", y1: m.top, y2: m.top + ih, visibility: "hidden" }, svg);
    const DOT = { ink: "ink", b: "series-b", course: "course", dim: "n6" };
    const dots = named.map((s) => el("circle", { r: 3.5, class: "dot", visibility: "hidden", fill: `var(--${DOT[s.cls] || "ink"})` }, svg));
    const hit = el("rect", { class: "hit", x: m.left, y: m.top, width: iw, height: ih }, svg);
    const nearest = (px) => {
      if (categorical) return Math.max(0, Math.min(xs.length - 1, Math.round(((px - m.left) / iw) * (xs.length - 1))));
      const t = x0 + ((px - m.left) / iw) * (x1 - x0);
      let lo2 = 0, hi2 = xs.length - 1;
      while (hi2 - lo2 > 1) { const mid = (lo2 + hi2) >> 1; if (xs[mid].getTime() < t) lo2 = mid; else hi2 = mid; }
      return Math.abs(xs[lo2] - t) < Math.abs(xs[hi2] - t) ? lo2 : hi2;
    };
    hit.addEventListener("pointermove", (ev) => {
      const r = svg.getBoundingClientRect();
      const i = nearest(((ev.clientX - r.left) / r.width) * width);
      cross.setAttribute("x1", sx(i)); cross.setAttribute("x2", sx(i)); cross.setAttribute("visibility", "visible");
      named.forEach((s, k) => {
        const v = s.values[i];
        dots[k].setAttribute("visibility", v == null ? "hidden" : "visible");
        if (v != null) { dots[k].setAttribute("cx", sx(i)); dots[k].setAttribute("cy", sy(v)); }
      });
      const head = categorical ? xs[i] : (opts.dateLabel ? opts.dateLabel(xs[i]) : xs[i].toISOString().slice(0, 7));
      const rows = named.map((s) => `<div>${esc(s.label)}: <strong>${s.values[i] == null ? "–" : yf(s.values[i])}</strong></div>`).join("");
      showTip(`<div class="muted">${esc(head)}</div>${rows}`, ev.clientX, ev.clientY);
    });
    hit.addEventListener("pointerleave", () => {
      cross.setAttribute("visibility", "hidden"); dots.forEach((d) => d.setAttribute("visibility", "hidden")); hideTip();
    });
  }

  function barChart(host, rows, opts = {}) {
    const c = { host, width: 0 };
    host.classList.add("chart");
    host.__chart = c;
    c.draw = () => {
      const width = host.clientWidth || 500; c.width = width;
      const rowH = opts.rowHeight || 22, labelW = opts.labelWidth || 170;
      const height = rows.length * rowH + 24;
      const iw = width - labelW - 64;
      const hi = Math.max(...rows.map((r) => r.value)) || 1;
      host.innerHTML = "";
      const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": opts.aria || "" }, host);
      rows.forEach((r, i) => {
        const y = i * rowH + 4;
        const t = el("text", { x: labelW - 10, y: y + rowH / 2 + 2, "text-anchor": "end" }, svg);
        t.textContent = r.label;
        const w = (r.value / hi) * iw;
        const bar = el("rect", { x: labelW, y: y + 4, width: Math.max(1, w), height: rowH - 9, class: `bar ${r.cls || ""}` }, svg);
        const v = el("text", { x: labelW + w + 6, y: y + rowH / 2 + 2 }, svg);
        v.textContent = (opts.format || ((x) => fmt.num(x, 3)))(r.value);
        bar.addEventListener("pointermove", (ev) => showTip(`<strong>${esc(r.label)}</strong><div>${esc(r.note || "")}</div>`, ev.clientX, ev.clientY));
        bar.addEventListener("pointerleave", hideTip);
      });
    };
    charts.add(c); resize.observe(host); c.draw();
  }

  function dotWhisker(host, rows, opts = {}) {
    const c = { host, width: 0 };
    host.classList.add("chart");
    host.__chart = c;
    c.draw = () => {
      const width = host.clientWidth || 500; c.width = width;
      const rowH = 22, labelW = 170, height = rows.length * rowH + 34;
      const iw = width - labelW - 24;
      const lo = Math.min(-0.5, ...rows.map((r) => r.lo)), hi = Math.max(1.25, ...rows.map((r) => r.hi));
      const sx = (v) => labelW + ((Math.max(lo, Math.min(hi, v)) - lo) / (hi - lo)) * iw;
      host.innerHTML = "";
      const svg = el("svg", { viewBox: `0 0 ${width} ${height}`, height, role: "img", "aria-label": opts.aria || "" }, host);
      const grid = el("g", { class: "grid" }, svg);
      niceTicks(lo, hi, 6).forEach((t) => {
        el("line", { x1: sx(t), x2: sx(t), y1: 0, y2: rows.length * rowH }, grid);
        const tx = el("text", { x: sx(t), y: rows.length * rowH + 16, "text-anchor": "middle" }, svg);
        tx.textContent = minus(String(t));
      });
      el("line", { class: "zero", x1: sx(0), x2: sx(0), y1: 0, y2: rows.length * rowH }, svg);
      el("line", { class: "ideal", x1: sx(1), x2: sx(1), y1: 0, y2: rows.length * rowH }, svg);
      const ideal = el("text", { x: sx(1) + 4, y: rows.length * rowH + 30 }, svg);
      ideal.textContent = "1 = well calibrated";
      rows.forEach((r, i) => {
        const y = i * rowH + rowH / 2;
        const t = el("text", { x: labelW - 10, y: y + 4, "text-anchor": "end" }, svg);
        t.textContent = r.label;
        el("line", { class: "whisker", x1: sx(r.lo), x2: sx(r.hi), y1: y, y2: y }, svg);
        const dot = el("circle", { cx: sx(r.value), cy: y, r: 4, fill: "var(--ink)", class: "dot" }, svg);
        dot.addEventListener("pointermove", (ev) => showTip(`<strong>${esc(r.label)}</strong><div>slope ${fmt.num(r.value, 2)} ± ${fmt.num(r.se * 1.96, 2)}</div>`, ev.clientX, ev.clientY));
        dot.addEventListener("pointerleave", hideTip);
      });
    };
    charts.add(c); resize.observe(host); c.draw();
  }

  // ------------------------------------------------------------ evidence behind a cell

  function lookup(block, pair, model) {
    let task, cells, months = null, trace = null, kind, h = 1, source;
    if (block.startsWith("x:")) {
      task = block.slice(2); kind = "extra";
      cells = D.extra.scorecards[task].cells; months = D.extra.months;
      trace = D.extra.traces[task]?.[pair]?.[model] || null;
      source = `reports/tables/extra_pairs_${task}.csv`;
    } else if (block.startsWith("h")) {
      const [hh, t] = block.slice(1).split(":"); task = t; h = +hh; kind = "horizon";
      cells = D.horizons[t].cells.filter((c) => c.h === h);
      source = h === 1 ? `reports/tables/ml_${task}.csv` : `reports/tables/horizons_${task}.csv`;
    } else {
      task = block; kind = "core";
      cells = D.scorecards[task].cells; months = D.months;
      trace = D.traces[task]?.[pair]?.[model] || null;
      source = `reports/tables/ml_${task}.csv`;
    }
    const ref = T[task].reference;
    return {
      task, kind, h, months, trace, source, ref,
      cell: cells.find((c) => c.pair === pair && c.model === model),
      refCell: cells.find((c) => c.pair === pair && c.model === ref),
    };
  }

  function sentence(info, model, pair) {
    const { task, cell, refCell, ref } = info;
    const spec = T[task];
    if (!cell) return `${label(model)} was not run on ${PL[pair]}.`;
    const v = cell.value, r = refCell ? refCell.value : null;
    let gap = "";
    if (task === "volatility" && v != null && r) {
      const g = v / r - 1;
      gap = ` (${fmt.pct(Math.abs(g))} ${g < 0 ? "lower" : "higher"})`;
    }
    const metric = task === "volatility" ? `QLIKE ${fmt.num(v, 4)} against ${label(ref)}'s ${fmt.num(r, 4)}${gap}`
      : task === "direction" ? `AUC ${fmt.num(v, 3)} against ${fmt.num(r, 3)} for ${label(ref)}`
        : `out-of-sample R² ${fmt.num(v, 4)} (a zero forecast scores 0)`;
    const word = VERDICT[cell.v];
    return `${metric}. On ${LOSS[task]}, ${label(model)} is ${word} ${label(ref)}${cell.v ? ` (Diebold–Mariano ${fmt.num(cell.dm, 2)}, p ${cell.p < 0.001 ? "< 0.001" : "= " + fmt.p(cell.p)})` : ""}.`;
  }

  function renderEvidence(target, block, pair, model) {
    const box = document.getElementById(target);
    if (!box) return;
    const info = lookup(block, pair, model);
    const { task, cell } = info;
    box.classList.remove("empty", "reveal");
    void box.offsetWidth;
    box.classList.add("reveal");
    const hz = info.kind === "horizon" ? ` · ${info.h === 1 ? "next day" : `next ${info.h} days`}` : "";
    const stats = cell ? [
      [`${T[task].metric_label}, ${label(model)}`, task === "volatility" ? fmt.num(cell.value, 4) : fmt.num(cell.value, task === "returns" ? 4 : 3)],
      [`${T[task].metric_label}, ${label(info.ref)}`, info.refCell ? fmt.num(info.refCell.value, task === "volatility" ? 4 : 3) : "–"],
      ["Diebold–Mariano statistic", fmt.num(cell.dm, 2)],
      ["p-value", fmt.p(cell.p)],
      ...(cell.pt_p != null ? [["Pesaran–Timmermann p (up/down calls)", fmt.p(cell.pt_p)]] : []),
      ...(cell.days ? [["Test days", fmt.int(cell.days)]] : []),
    ] : [];
    box.innerHTML = `
      <div class="ev-text">
        <p class="ev-title"><span class="ev-model">${esc(label(model))}</span> on ${esc(PL[pair])}${esc(hz)}</p>
        <p class="ev-sentence">${esc(sentence(info, model, pair))}</p>
        <dl class="ev-stats">${stats.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("")}</dl>
        <p class="ev-source">Source <span class="path">${esc(info.source)}</span></p>
      </div>
      <div class="ev-chart">${info.trace ? `
        <h3>Running advantage over ${esc(label(info.ref))}</h3>
        <p>Cumulative ${esc(LOSS[task])} of ${esc(label(info.ref))} minus ${esc(label(model))}, summed month by month. Rising means ${esc(label(model))} is pulling ahead; shaded bands are stress episodes.</p>
        <div class="ev-plot"></div>` : `<p>${info.kind === "horizon" ? "Longer horizons report the test over the whole period only." : ""}</p>`}
      </div>`;
    if (info.trace) {
      chart($(".ev-plot", box), {
        x: info.months.map(parseDate), height: 190,
        series: [{ values: info.trace, cls: "course", label: label(model) }],
        zero: true, diverging: true, shade: { periods: D.meta.stress }, labelRoom: 110,
        yLabel: `Cumulative ${LOSS[task]} advantage`,
        aria: `Running loss advantage of ${label(model)} over ${label(info.ref)} on ${PL[pair]}`,
      });
    }
  }

  // ------------------------------------------------------------ scorecard interaction

  function cellsOf(table, block, model) {
    return $$(`[data-block="${block}"][data-model="${model}"]`, table);
  }
  function clearCross(table) { $$(".xh", table).forEach((n) => n.classList.remove("xh")); }
  function crosshair(td) {
    const table = td.closest("table");
    clearCross(table);
    $$("td.sc-cell", td.parentElement).forEach((n) => n.classList.add("xh"));
    cellsOf(table, td.dataset.block, td.dataset.model).forEach((n) => n.classList.add("xh"));
  }
  function select(td, focus = true) {
    const table = td.closest("table");
    $$(".sel", table).forEach((n) => n.classList.remove("sel"));
    $$(".sel-col", table).forEach((n) => n.classList.remove("sel-col"));
    td.classList.add("sel");
    $$("th", table).filter((th) => th.dataset.block === td.dataset.block && th.dataset.model === td.dataset.model)
      .forEach((th) => th.classList.add("sel-col"));
    $$("td.sc-cell", table).forEach((n) => n.setAttribute("tabindex", "-1"));
    td.setAttribute("tabindex", "0");
    if (focus) td.focus({ preventScroll: true });
    const wrap = td.closest("[data-evidence]");
    if (wrap) renderEvidence(wrap.dataset.evidence, td.dataset.block, td.dataset.pair, td.dataset.model);
  }
  function tipFor(td) {
    const info = lookup(td.dataset.block, td.dataset.pair, td.dataset.model);
    const c = info.cell;
    if (!c) return `<strong>${esc(label(td.dataset.model))}</strong> · ${esc(PL[td.dataset.pair])}<div class="muted">not run</div>`;
    const metric = `${T[info.task].metric_label} ${fmt.num(c.value, info.task === "volatility" ? 4 : 3)}`;
    return `<strong>${esc(label(td.dataset.model))}</strong> · ${esc(PL[td.dataset.pair])}<div>${esc(metric)}</div><div class="muted">DM ${fmt.num(c.dm, 2)}, p ${fmt.p(c.p)} · select for the evidence</div>`;
  }

  document.addEventListener("pointerover", (ev) => {
    const td = ev.target.closest && ev.target.closest("td.sc-cell");
    if (!td) return;
    crosshair(td);
  });
  document.addEventListener("pointermove", (ev) => {
    const td = ev.target.closest && ev.target.closest("td.sc-cell");
    if (td) showTip(tipFor(td), ev.clientX, ev.clientY);
  });
  document.addEventListener("pointerout", (ev) => {
    const td = ev.target.closest && ev.target.closest("td.sc-cell");
    if (td && !td.contains(ev.relatedTarget)) { clearCross(td.closest("table")); hideTip(); }
  });
  document.addEventListener("click", (ev) => {
    const td = ev.target.closest && ev.target.closest("td.sc-cell");
    if (td) select(td);
  });
  document.addEventListener("keydown", (ev) => {
    const td = document.activeElement && document.activeElement.closest && document.activeElement.closest("td.sc-cell");
    if (!td) return;
    const row = td.parentElement, table = td.closest("table");
    const visible = (n) => n.offsetParent !== null;
    const rowCells = $$("td.sc-cell", row).filter(visible);
    const i = rowCells.indexOf(td);
    let next = null;
    if (ev.key === "ArrowRight") next = rowCells[i + 1];
    else if (ev.key === "ArrowLeft") next = rowCells[i - 1];
    else if (ev.key === "ArrowDown" || ev.key === "ArrowUp") {
      const rows = $$("tbody tr", table);
      const r = rows[rows.indexOf(row) + (ev.key === "ArrowDown" ? 1 : -1)];
      if (r) next = $(`td.sc-cell[data-block="${td.dataset.block}"][data-model="${td.dataset.model}"]`, r);
    } else if (ev.key === "Home") next = rowCells[0];
    else if (ev.key === "End") next = rowCells[rowCells.length - 1];
    else if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); select(td); return; }
    if (next) { ev.preventDefault(); select(next); crosshair(next); hideTip(); }
  });
  // every table starts with one reachable cell
  $$("table.sc").forEach((t) => { const f = $("td.sc-cell", t); if (f) f.setAttribute("tabindex", "0"); });

  // ------------------------------------------------------------ views

  const tabs = $$('.views [role="tab"]');
  function showView(name) {
    tabs.forEach((t) => {
      const on = t.dataset.view === name;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (name === "present") present.open(); else present.close();
    store.set("fx-view", name === "present" ? "verdicts" : name);
    hideTip();
    requestAnimationFrame(redrawAll);
  }
  tabs.forEach((t) => t.addEventListener("click", () => showView(t.dataset.view)));
  document.addEventListener("keydown", (ev) => {
    if (ev.target.closest && ev.target.closest("input, select, textarea")) return;
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    if (!$("#view-present").hidden) return;
    const n = { 1: "verdicts", 2: "explorer", 3: "present" }[ev.key];
    if (n) showView(n);
  });

  // RQ buttons jump into the explorer
  $$("[data-goto]").forEach((b) => b.addEventListener("click", () => {
    showView("explorer");
    const [section, task] = b.dataset.goto.split(":");
    if (task) setTask(task);
    document.getElementById(section).scrollIntoView({ behavior: "smooth", block: "start" });
  }));

  // ------------------------------------------------------------ explorer: scorecards

  const segs = (host, onPick) => {
    $$("button", host).forEach((b) => b.addEventListener("click", () => {
      $$("button", host).forEach((x) => x.setAttribute("aria-pressed", x === b ? "true" : "false"));
      onPick(b.dataset.value);
    }));
  };
  const GROUPS = ["Benchmarks", "Classical", "Regime-aware", "Neural"];
  let currentTask = "volatility";
  function setTask(task) {
    currentTask = task;
    $$("#ex-task button").forEach((b) => b.setAttribute("aria-pressed", b.dataset.value === task ? "true" : "false"));
    $$("[data-task-card]").forEach((w) => { w.hidden = w.dataset.taskCard !== task; });
    const box = $("#ev-explorer");
    box.className = "evidence empty";
    box.textContent = "Select any triangle to see the test behind it: the loss gap, the Diebold–Mariano statistic and how the advantage built up month by month.";
    applyGroup();
  }
  let currentGroup = "all";
  function applyGroup() {
    $$("[data-task-card]").forEach((w) => {
      GROUPS.forEach((g) => w.classList.toggle(`sc-hide-${g}`, currentGroup !== "all" && g !== currentGroup));
      const table = $("table", w);
      const th = $(".sc-head-top th.sc-block", table) || $$(".sc-head-top th", table)[1];
      if (th) th.colSpan = $$("th.sc-col", table).filter((n) => getComputedStyle(n).display !== "none").length || 1;
    });
  }
  segs($("#ex-task"), setTask);
  segs($("#ex-group"), (g) => { currentGroup = g; applyGroup(); });
  segs($("#hz-task"), (t) => { $$("[data-hz-card]").forEach((w) => { w.hidden = w.dataset.hzCard !== t; }); });
  segs($("#x-task"), (t) => { $$("[data-x-card]").forEach((w) => { w.hidden = w.dataset.xCard !== t; }); });

  // ------------------------------------------------------------ explorer: regimes

  function regimeChart(host, height = 210) {
    const r = D.regimes.timeline;
    chart(host, {
      x: r.dates.map(parseDate), height, yMin: 0, yMax: 1,
      series: [{ values: r.gmm, cls: "ink", label: "Gaussian mixture" }, { values: r.hmm, cls: "b", label: "Hidden Markov" }],
      shade: { periods: [{ start: r.dates[0], end: r.first_refit, short: "in sample" }, ...D.meta.stress], labels: true },
      yFormat: (v) => fmt.pct(v, 0), labelRoom: 112, dateLabel: (d) => `week of ${d.toISOString().slice(0, 10)}`,
      yLabel: "Probability of the turbulent regime",
      aria: "Weekly probability of the turbulent regime from the Gaussian mixture and the hidden Markov model",
    });
  }
  regimeChart($("#regime-chart"));
  $("#regime-table").innerHTML = table(
    ["Model", "Regime", "Share of days", "Mean spell (days)", "Stays next day", "Next-day vol (a year)", "Stress days caught"],
    D.regimes.summary.map((r) => [r.method === "gmm" ? "Gaussian mixture" : "Hidden Markov", r.regime === 0 ? "calm" : "turbulent",
      fmt.pct(r.share_of_days, 0), fmt.num(r.mean_spell_days, 0), fmt.pct(r.stay_probability, 1), `${fmt.num(r.next_day_vol_pct, 1)}%`, fmt.pct(r.stress_capture, 0)]),
    [2, 3, 4, 5, 6]);

  function table(head, rows, numeric = [], opts = {}) {
    const th = head.map((h, i) => `<th class="${numeric.includes(i) ? "num" : ""}" scope="col">${esc(h)}</th>`).join("");
    const body = rows.map((r, k) => `<tr${opts.rowAttr ? opts.rowAttr(k) : ""}>${r.map((c, i) => `<td class="${numeric.includes(i) ? "num" : ""}${opts.cellClass ? " " + (opts.cellClass(k, i) || "") : ""}">${c}</td>`).join("")}</tr>`).join("");
    return `<div class="dt-wrap"><table class="dt"><thead><tr>${th}</tr></thead><tbody>${body}</tbody></table></div>`;
  }

  // ------------------------------------------------------------ explorer: backtest

  const B = D.backtest;
  const stratName = (signal, sizing) => {
    const base = signal.replace(/_confident$/, "");
    const s = (B.strategy_labels[base] || base) + (signal.endsWith("_confident") ? " (confident)" : "");
    return sizing === "unit" ? s : `${s}, ${B.sizing_labels[sizing].replace("Vol target: ", "vol target ")}`;
  };
  let sizingFilter = "unit";
  let chosenStrategy = "carry|unit";
  const bmonths = B.months.map(parseDate);
  function equityChart(host, highlight, height = 260, keep = ["long|unit"]) {
    const keys = Object.keys(B.curves).filter((k) => sizingFilter === "all" || k.endsWith(`|${sizingFilter}`) || k === highlight);
    const series = keys.filter((k) => k !== highlight && !keep.includes(k)).map((k) => ({ values: B.curves[k], cls: "ghost" }));
    keep.filter((k) => k !== highlight && B.curves[k]).forEach((k) => series.push({ values: B.curves[k], cls: "dim", label: stratName(...k.split("|")) }));
    series.push({ values: B.curves[highlight], cls: "course", label: stratName(...highlight.split("|")) });
    chart(host, {
      x: bmonths, height, zero: true, series, shade: { periods: D.meta.stress },
      yFormat: (v) => fmt.pct(v, 0), labelRoom: 150, yLabel: "Cumulative net return",
      aria: "Cumulative net return of the equal-risk portfolio for each strategy",
    });
  }
  function renderBacktest() {
    const rows = B.portfolio.filter((r) => r.signal !== "flat" && (sizingFilter === "all" || r.sizing === sizingFilter))
      .sort((a, b) => (b.sharpe ?? -9) - (a.sharpe ?? -9));
    if (!rows.find((r) => `${r.signal}|${r.sizing}` === chosenStrategy)) chosenStrategy = `${rows[0].signal}|${rows[0].sizing}`;
    $("#bt-table").innerHTML = table(
      ["Strategy", "Return a year", "Volatility", "Sharpe", "Sortino", "Max drawdown", "p-value", "Sharpe at 2× costs"],
      rows.map((r) => [esc(stratName(r.signal, r.sizing)), fmt.pct(r.ann_return), fmt.pct(r.ann_vol), fmt.num(r.sharpe, 2), fmt.num(r.sortino, 2),
        fmt.pct(r.max_drawdown, 0), fmt.p(r.p_value), fmt.num(r.sharpe_2x_costs, 2)]),
      [1, 2, 3, 4, 5, 6, 7],
      { rowAttr: (k) => ` class="pick${`${rows[k].signal}|${rows[k].sizing}` === chosenStrategy ? " sel" : ""}" data-key="${rows[k].signal}|${rows[k].sizing}" tabindex="0"`,
        cellClass: (k, i) => (i === 3 ? (rows[k].sharpe > 0 ? "pos" : "neg") : "") });
    $$("#bt-table tr.pick").forEach((tr) => {
      const pick = () => { chosenStrategy = tr.dataset.key; renderBacktest(); };
      tr.addEventListener("click", pick);
      tr.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); } });
    });
    equityChart($("#bt-chart"), chosenStrategy);
    const [sig, siz] = chosenStrategy.split("|");
    const parts = B.breakdown.filter((r) => r.signal === sig && r.sizing === siz);
    $("#bt-breakdown-title").textContent = `${stratName(sig, siz)} by regime, stress and subperiod`;
    $("#bt-breakdown").innerHTML = parts.length ? table(["Split", "Bucket", "Days", "Return a year", "Sharpe", "Max drawdown"],
      parts.map((r) => [esc(r.dimension), esc(r.bucket), fmt.int(r.days), fmt.pct(r.ann_return), fmt.num(r.sharpe, 2), fmt.pct(r.max_drawdown, 0)]),
      [2, 3, 4, 5]) : `<p class="sc-note">Breakdowns are saved for unit-size and LightGBM vol-targeted strategies; pick one of those.</p>`;
  }
  segs($("#bt-sizing"), (v) => { sizingFilter = v; renderBacktest(); });
  renderBacktest();
  $("#bt-targeting").innerHTML = table(["Sizing", "Realised volatility", "Tracking error of the 10% target", "Quarters above 15%"],
    B.targeting.map((r) => [esc(B.sizing_labels[r.sizing] || r.sizing), fmt.pct(r.mean_realised_vol), fmt.pct(r.tracking), fmt.pct(r.quarters_above_1_5x, 0)]),
    [1, 2, 3]);

  // ------------------------------------------------------------ explorer: value at risk

  const varRows = [...B.var].sort((a, b) => (a.method + a.side + a.model).localeCompare(b.method + b.side + b.model));
  $("#var-table").innerHTML = table(["Variance forecast", "Method", "Position", "Exceedance rate (target 1%)", "Pairs passing Kupiec", "Pairs passing Christoffersen"],
    varRows.map((r) => [esc(label(r.model)), r.method === "fhs" ? "Filtered historical" : "Normal", esc(r.side), fmt.pct(r.rate, 2),
      `${r.kupiec_pass} of ${r.pairs}`, `${r.christoffersen_pass} of ${r.pairs}`]),
    [3, 4, 5], { cellClass: (k, i) => (i === 4 ? (varRows[k].kupiec_pass >= varRows[k].pairs - 1 ? "pos" : varRows[k].kupiec_pass <= 2 ? "neg" : "") : "") });

  // ------------------------------------------------------------ explorer: learning curves

  const L = D.learning.rows;
  let lcTarget = "high_volatility", lcMetric = "roc_auc";
  const LC_METRIC = { roc_auc: ["AUC", (v) => fmt.num(v, 3)], log_loss: ["Log loss", (v) => fmt.num(v, 3)], ece: ["Calibration error", (v) => fmt.num(v, 3)] };
  function learningChart(host, height = 240, target = lcTarget, metric = lcMetric) {
    const sizes = [21, 63, 126, 252, 504, 1008, 0];
    const cats = sizes.map((s) => (s ? `${s}` : "full"));
    const val = (model, s) => { const r = L.find((x) => x.target === target && x.model === model && x.size === s); return r ? r[metric] : null; };
    chart(host, {
      x: cats, categories: true, height,
      series: [{ values: sizes.map((s) => val("logistic", s)), cls: "ink", label: "Logistic regression" },
        { values: sizes.map((s) => val("naive_bayes", s)), cls: "b", label: "Naive Bayes" }],
      yFormat: LC_METRIC[metric][1], labelRoom: 130, yLabel: LC_METRIC[metric][0],
      aria: `${LC_METRIC[metric][0]} against training days for logistic regression and Naive Bayes`,
    });
  }
  segs($("#lc-target"), (v) => { lcTarget = v; learningChart($("#lc-chart")); });
  segs($("#lc-metric"), (v) => { lcMetric = v; learningChart($("#lc-chart")); });
  learningChart($("#lc-chart"));

  // ------------------------------------------------------------ explorer: calibration and features

  const cal = D.calibration.filter((r) => r.calibration_slope != null && !r.model.startsWith("baseline_"))
    .sort((a, b) => a.calibration_slope - b.calibration_slope);
  dotWhisker($("#cal-chart"), cal.map((r) => ({ label: r.label, value: r.calibration_slope, se: r.calibration_slope_se,
    lo: r.calibration_slope - 1.96 * r.calibration_slope_se, hi: r.calibration_slope + 1.96 * r.calibration_slope_se })),
  { aria: "Calibration slope of each direction model with a 95% interval" });
  let shapTask = "volatility";
  const shapChart = () => barChart($("#shap-chart"), D.shap[shapTask].map((r) => ({ label: r.feature, value: r.mean_abs_shap, note: `group: ${r.group}` })),
    { aria: "Mean absolute SHAP value of the top LightGBM features", format: (v) => fmt.num(v, 4) });
  segs($("#shap-task"), (v) => { shapTask = v; shapChart(); });
  shapChart();

  // ------------------------------------------------------------ explorer: extra pairs and data

  if (D.extra) {
    const xb = D.extra.backtest.filter((r) => r.signal !== "flat").sort((a, b) => (a.pair + a.sizing).localeCompare(b.pair + b.sizing) || b.sharpe - a.sharpe);
    $("#x-backtest").innerHTML = table(["Pair", "Strategy", "Return a year", "Sharpe", "Sortino", "Max drawdown", "p-value"],
      xb.map((r) => [esc(PL[r.pair]), esc(stratName(r.signal, r.sizing)), fmt.pct(r.ann_return), fmt.num(r.sharpe, 2), fmt.num(r.sortino, 2), fmt.pct(r.max_drawdown, 0), fmt.p(r.p_value)]),
      [2, 3, 4, 5, 6], { cellClass: (k, i) => (i === 3 ? (xb[k].sharpe > 0 ? "pos" : "neg") : "") });
  }
  $("#data-table").innerHTML = table(["Pair", "Set", "From", "To", "Days present", "Days with an hour gap", "Noon return correlation with H.10", "In daylight-saving gap weeks"],
    D.data.rows.map((r) => [esc(PL[r.pair] || r.pair), r.kind === "extra" ? "extra" : "core", esc(r.first), esc(r.last), `${fmt.num(r.coverage_pct, 1)}%`,
      fmt.int(r.partial_days), r.return_corr == null ? "daily rates only" : fmt.num(r.return_corr, 4), r.return_corr_clock_gap == null ? "–" : fmt.num(r.return_corr_clock_gap, 4)]),
    [4, 5, 6, 7]);

  // index highlighting while scrolling the explorer
  const links = $$(".index a");
  const spy = new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (e.isIntersecting) links.forEach((a) => a.setAttribute("aria-current", a.getAttribute("href") === `#${e.target.id}` ? "true" : "false"));
    });
  }, { rootMargin: "-20% 0px -70% 0px" });
  $$(".panel").forEach((p) => spy.observe(p));

  // ------------------------------------------------------------ present

  const present = (() => {
    const host = $("#view-present");
    const stage = $(".slide", host);
    const counter = $(".count", host);
    const bar = $(".progress i", host);
    const q = Object.fromEntries(D.questions.map((x) => [x.id, x]));
    const months = D.months.map(parseDate);
    const lgbTraces = Object.entries(D.traces.volatility).map(([pair, m]) => ({ pair, values: m.lightgbm }));
    const avgTrace = months.map((_, i) => { const v = lgbTraces.map((t) => t.values?.[i]).filter((x) => x != null); return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null; });
    const fromTemplate = (id) => () => { const t = document.getElementById(id); return t.content.cloneNode(true); };
    const SLIDES = [
      { title: "What is predictable in G10 FX, when, and is it worth money?",
        takeaway: `Seven major currency pairs and two extra ones, tested month by month from ${D.meta.test_start} to ${D.meta.test_end}: every model against a benchmark, every claim with a test.`,
        stage: () => { const ul = document.createElement("ul"); ul.className = "slide-verdicts"; ul.innerHTML = D.questions.map((x) => `<li><b>${x.id}</b><span>${esc(x.question)}</span></li>`).join(""); return ul; } },
      { title: q.RQ1.headline, takeaway: q.RQ1.answer.split(". ").slice(0, 1).join(". ") + ".", stage: fromTemplate("tpl-present-volatility") },
      { title: `${label("lightgbm")} ${avgTrace.filter((v) => v != null).at(-1) > 0 ? "pulls ahead of HAR-RV month after month" : "falls behind HAR-RV"}`, takeaway: "Running QLIKE advantage over HAR-RV: each grey line is one pair, the highlighted line their average.",
        stage: (d, fit) => chart(d, { x: months, height: fit(), zero: true, shade: { periods: D.meta.stress, labels: true },
          series: [...lgbTraces.map((t) => ({ values: t.values, cls: "dim" })), { values: avgTrace, cls: "course", label: "Average of 7 pairs" }], labelRoom: 170,
          yLabel: "Cumulative QLIKE advantage over HAR-RV" }) },
      { title: q.RQ2.headline, takeaway: q.RQ2.answer, stage: fromTemplate("tpl-present-direction") },
      { title: q.RQ3.headline, takeaway: q.RQ3.answer, stage: (d, fit) => regimeChart(d, fit()) },
      { title: q.RQ4.headline, takeaway: q.RQ4.answer.split(". ").slice(0, 2).join(". ") + ".",
        stage: (d, fit) => { const keep = sizingFilter; sizingFilter = "unit"; equityChart(d, "carry|unit", fit()); sizingFilter = keep; } },
      { title: q.RQ1.horizon_headline, takeaway: q.RQ1.answer.split(". ").slice(-1)[0], stage: fromTemplate("tpl-present-horizons") },
      { title: q.RQ5.headline, takeaway: q.RQ5.answer, stage: (d, fit) => learningChart(d, fit(), "high_volatility", "roc_auc") },
      ...(D.extra ? [{ title: "Beyond the seven majors", takeaway: D.extra.summary.headline, stage: fromTemplate("tpl-present-extra") }] : []),
      { title: "Five answers", takeaway: "", stage: () => { const ul = document.createElement("ul"); ul.className = "slide-verdicts"; ul.innerHTML = D.questions.map((x) => `<li><b>${x.id}</b><div><strong>${esc(x.verdict)}.</strong> <span>${esc(x.answer)}</span></div></li>`).join(""); return ul; } },
    ];
    let index = 0, isOpen = false;
    function render() {
      const s = SLIDES[index];
      charts.forEach((c) => { if (stage.contains(c.host)) { charts.delete(c); resize.unobserve(c.host); } });
      stage.classList.remove("enter"); void stage.offsetWidth; stage.classList.add("enter");
      stage.innerHTML = `<h2>${esc(s.title).replace(/HAR-RV/g, "HAR&#8209;RV")}</h2><p class="takeaway">${esc(s.takeaway)}</p><div class="stage"></div>`;
      const box = $(".stage", stage);
      const fit = () => Math.max(260, box.clientHeight - 12);
      if (s.stage.length) {  // a chart: draw it once the slide has its size
        const d = document.createElement("div");
        box.appendChild(d);
        requestAnimationFrame(() => s.stage(d, fit));
      } else box.appendChild(s.stage());
      counter.textContent = `${index + 1} / ${SLIDES.length}`;
      bar.style.transform = `scaleX(${(index + 1) / SLIDES.length})`;
    }
    function go(d) { index = Math.max(0, Math.min(SLIDES.length - 1, index + d)); render(); }
    document.addEventListener("keydown", (ev) => {
      if (!isOpen) return;
      if (["ArrowRight", "PageDown", " "].includes(ev.key)) { ev.preventDefault(); go(1); }
      else if (["ArrowLeft", "PageUp"].includes(ev.key)) { ev.preventDefault(); go(-1); }
      else if (ev.key === "Home") { index = 0; render(); }
      else if (ev.key === "End") { index = SLIDES.length - 1; render(); }
      else if (ev.key === "Escape") showView("verdicts");
    });
    $(".prev", host).addEventListener("click", () => go(-1));
    $(".next", host).addEventListener("click", () => go(1));
    $(".leave", host).addEventListener("click", () => showView("verdicts"));
    return {
      open() { if (!isOpen) { isOpen = true; render(); } },
      close() { isOpen = false; },
    };
  })();

  // ------------------------------------------------------------ start

  applyTheme(store.get("fx-theme") || "auto");
  const startView = store.get("fx-view");
  showView(["verdicts", "explorer"].includes(startView) ? startView : "verdicts");
  setTask("volatility");
})();
