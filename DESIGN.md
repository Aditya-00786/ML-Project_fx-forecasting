---
name: FX forecasting results
description: A forecast verification bulletin; every result is a verdict against a reference forecast.
colors:
  bulletin-paper: "#fcfcfb"
  raised-paper: "#f5f5f2"
  hover-wash: "#ecebe6"
  hairline: "#e1e0d9"
  control-stroke: "#c3c2b7"
  group-rule: "#a6a59c"
  bar-grey: "#898781"
  ink-quiet: "#6c6b66"
  ink-secondary: "#52514e"
  ink: "#0b0b0b"
  stress-band: "#efeee9"
  better: "#2a78d6"
  better-text: "#1f68c4"
  better-wash: "rgba(42, 120, 214, 0.12)"
  worse: "#e34948"
  worse-text: "#c22f2e"
  worse-wash: "rgba(227, 73, 72, 0.12)"
  course-violet: "#3f2bb0"
  course-wash: "rgba(63, 43, 176, 0.09)"
  series-ochre: "#9a6a00"
  night-slate: "#111416"
  night-raised: "#171b1e"
  night-hover: "#1e2327"
  night-hairline: "#2a3035"
  night-control-stroke: "#3b4249"
  night-group-rule: "#59616a"
  night-bar-grey: "#7b838c"
  night-ink-quiet: "#959ca4"
  night-ink-secondary: "#b6bcc2"
  night-ink: "#f1f3f4"
  night-stress-band: "#1a1e22"
  night-better: "#6ea8ee"
  night-worse: "#f27e7c"
  night-course-lavender: "#d6c9ff"
  night-course-wash: "rgba(214, 201, 255, 0.12)"
  night-series-ochre: "#d9a53f"
typography:
  headline-present:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "clamp(1.75rem, 3.4vw, 3.25rem)"
    fontWeight: 650
    lineHeight: 1.1
    letterSpacing: "-0.02em"
  headline:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "1.375rem"
    fontWeight: 650
    lineHeight: 1.25
    letterSpacing: "-0.01em"
  title:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "1.0625rem"
    fontWeight: 650
    letterSpacing: "-0.01em"
  section:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "1rem"
    fontWeight: 650
    letterSpacing: "-0.01em"
  body:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "15px"
    fontWeight: 400
    lineHeight: 1.5
    fontFeature: "tnum"
  body-small:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "0.875rem"
    fontWeight: 400
    fontFeature: "tnum"
  data:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "0.8125rem"
    fontWeight: 400
    fontFeature: "tnum"
  label:
    fontFamily: "ui-sans-serif, -apple-system, BlinkMacSystemFont, \"Segoe UI\", Roboto, \"Helvetica Neue\", Arial, sans-serif"
    fontSize: "0.75rem"
    fontWeight: 500
    fontFeature: "tnum"
  mono:
    fontFamily: "ui-monospace, \"SF Mono\", Menlo, Consolas, \"Liberation Mono\", monospace"
    fontSize: "0.8125em"
    fontWeight: 400
rounded:
  none: "0"
spacing:
  cell: "28px"
  gutter: "32px"
  gutter-narrow: "16px"
  column-gap: "40px"
  panel: "32px"
  page-end: "64px"
components:
  scorecard-cell:
    backgroundColor: "{colors.bulletin-paper}"
    width: "{spacing.cell}"
    height: "{spacing.cell}"
    rounded: "{rounded.none}"
  scorecard-cell-hover:
    backgroundColor: "{colors.hover-wash}"
  scorecard-cell-crosshair:
    backgroundColor: "{colors.course-wash}"
  scorecard-cell-selected:
    backgroundColor: "{colors.course-wash}"
  scorecard-pair-label:
    textColor: "{colors.ink}"
    typography: "{typography.data}"
    height: "{spacing.cell}"
  research-question-row:
    textColor: "{colors.ink}"
    typography: "{typography.title}"
    padding: "11px 0 12px"
  research-question-row-hover:
    backgroundColor: "{colors.raised-paper}"
  view-tab:
    textColor: "{colors.ink-quiet}"
    padding: "14px 18px 12px"
  view-tab-active:
    textColor: "{colors.ink}"
  button-quiet:
    backgroundColor: "transparent"
    textColor: "{colors.ink-secondary}"
    rounded: "{rounded.none}"
    padding: "7px 11px"
  button-quiet-hover:
    backgroundColor: "{colors.hover-wash}"
    textColor: "{colors.ink}"
  segment-pressed:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.bulletin-paper}"
    rounded: "{rounded.none}"
  select-quiet:
    backgroundColor: "{colors.bulletin-paper}"
    textColor: "{colors.ink}"
    rounded: "{rounded.none}"
    padding: "7px 30px 7px 10px"
  data-table-header:
    textColor: "{colors.ink-quiet}"
    typography: "{typography.label}"
    padding: "6px 10px 6px 0"
  data-table-row-selected:
    backgroundColor: "{colors.course-wash}"
  tooltip:
    backgroundColor: "{colors.ink}"
    textColor: "{colors.bulletin-paper}"
    padding: "7px 9px"
    width: "300px"
---

# Design System: FX forecasting results

## Overview

**Creative North Star: "The Verification Bulletin"**

The dashboard reads like a forecast centre's verification bulletin: a ruled sheet of tabular figures where every result is stated as a verdict against a reference forecast. The module is not a card or a chart tile but the scorecard cell, a 28px square holding one triangle. Pairs run down, models run across in task blocks, and the reader's eye finds the pattern (where the blue triangles cluster, where everything is hollow) before reading a single number.

Density is high and quiet. One grotesque system stack sets every label and figure, numbers are tabular and right-aligned against hard column rules, and structure is drawn only with 1px hairlines on a near-white sheet (a night-shift slate in dark mode). Colour is spent on meaning, never on decoration: blue for beating the reference, red for losing to it, a stepped warm-grey ramp for everything else, and a single reserved violet for whatever the reader has selected.

The system refuses the stat-tile row and the grid of chart cards. Depth, rounding and shadow are absent; emphasis comes from rule weight (a full-ink rule opens a block, grey hairlines divide within it), type weight (650 against 400) and the violet selection.

**Key Characteristics:**
- The scorecard cell (28px square, one 13px triangle) is the module everything else is built around.
- Structure from 1px rules only; an ink-black rule opens each block, grey hairlines subdivide.
- One system sans stack for everything; tabular figures throughout; mono only for file paths, code and key hints.
- Blue and red carry better and worse; one violet carries selection; nothing else is coloured.
- Square corners, no shadows, no containers.
- Light and dark are complete token sets, switchable by system preference or a three-way theme button.

## Colors

A stepped neutral ramp with exactly three meanings painted over it: better, worse, and selected.

### Primary
- **Course Violet** (light `course-violet`; dark `night-course-lavender`): the selection colour and nothing else. It marks the selected scorecard cell (2px inset ring plus wash), the selected model's column header, the model name in the evidence title, the selected table row's first cell, the current Explorer index link, the selected strategy's line and bar in charts, the focus ring, the caret and text selection. The crosshair wash on the selected cell's row and column is the same hue at 9% (`course-wash`; 12% lavender in dark).

### Secondary
- **Verdict Blue** (`better`; dark `night-better`): "beats the reference". Fills and strokes the upward triangles, the positive loss-difference area under evidence traces (`better-wash`), and the dashed ideal-calibration line. Running text and table figures use the darker `better-text` so they hold contrast on the paper.
- **Verdict Red** (`worse`; dark `night-worse`): "loses to the reference". Downward triangles, the negative loss-difference area (`worse-wash`), and negative figures in tables via `worse-text`.

### Tertiary
- **Series Ochre** (`series-ochre`; dark `night-series-ochre`): the second comparison series in multi-line charts and their inline legends, when a chart needs two non-selected lines that are not the ink line.

### Neutral
- **Bulletin Paper** (`bulletin-paper`; dark `night-slate`): the page, sticky scorecard row labels, the present view, and the select field.
- **Raised Paper** (`raised-paper`; dark `night-raised`): hover tint for research-question rows and data-table rows; the Streamlit secondary background.
- **Hover Wash** (`hover-wash`; dark `night-hover`): hover tint for scorecard cells and quiet buttons.
- **Hairline** (`hairline`; dark `night-hairline`): every subdividing rule, chart gridlines, the present-view progress track.
- **Control Stroke** (`control-stroke`; dark `night-control-stroke`): chart axes and the grey "other strategies" lines.
- **Group Rule** (`group-rule`; dark `night-group-rule`): the slightly stronger rule between model families inside a scorecard block, and disabled segment text.
- **Control Outline** (`bar-grey`; dark `night-bar-grey`): the 1px outline of quiet buttons, segments, the select and the theme button, and the grey "no test" dash and zero counts. It holds at least 3:1 against the page in both themes, so controls and empty marks stay visible.
- **Bar Grey** (`bar-grey`): unselected bars (feature importance, calibration).
- **Ink Quiet** (`ink-quiet`; dark `night-ink-quiet`): labels, captions, definition terms, axis text, research-question ids.
- **Ink Secondary** (`ink-secondary`; dark `night-ink-secondary`): ledes, model column headers, code and paths, quiet-button text.
- **Ink** (`ink`; dark `night-ink`): body text, headings, the opening rules of blocks, the active tab underline, the pressed segment fill, the tooltip block, the zero line in charts.
- **Stress Band** (`stress-band`; dark `night-stress-band`): shaded bands behind chart regions marking in-sample years and documented stress episodes.

### Named Rules
**The Verdict Pair Rule.** Blue means the forecast beats its reference and red means it loses; neither colour is used for anything that is not a better/worse statement.

**The One Violet Rule.** Violet is reserved for the reader's current selection, everywhere at once (cell, column, row, line, bar, focus). Nothing else on the page is violet.

**The Lightness Split Rule.** The selection colour stays apart from Verdict Blue by lightness, not hue alone: deep violet below the blue in light mode, pale lavender above it in dark mode, so the two survive colour-blind viewing.

## Typography

**Body Font:** the system grotesque stack (ui-sans-serif, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif)
**Label/Mono Font:** the system monospace stack (ui-monospace, SF Mono, Menlo, Consolas, Liberation Mono), only for file paths, code and keyboard hints

**Character:** one neutral grotesque, set tight and tabular, so numbers line up like a printed bulletin and nothing competes with the triangles.

### Hierarchy
- **Headline, present** (650, fluid 1.75 to 3.25rem, line-height 1.1, -0.02em, max 30ch): the one large statement per slide in the Present view, with a takeaway line under it (fluid 1 to 1.6rem, Ink Secondary, max 60ch).
- **Headline** (650, 1.375rem, 1.25): the page title in the bulletin strip and each Explorer panel heading. Drops to 1.125rem under 860px.
- **Title** (650, 1.0625 to 1.125rem, -0.01em): the one-word research-question verdict and the evidence title.
- **Section** (650, 1rem): headings of the Verdicts board and the research-question column; Explorer sub-heads at 0.9375rem.
- **Body** (400, 15px, 1.5, tabular figures): base text. Ledes and answers step down to 0.875rem in Ink Secondary, capped at 72 to 80ch.
- **Data** (0.8125rem): scorecard pair labels (600), data-table cells, evidence statistics (values at 550, right-aligned), quiet controls (500).
- **Label** (0.75rem, 500 to 600): column headers in Ink Quiet, model names set vertically above scorecard columns, bulletin metadata terms, sources, count rows, chart notes. Sentence case; no uppercase, no tracking.
- **Chart text** (12px, Ink Quiet, tabular): axis and direct labels inside SVG charts (labels at 600); 18px in the Present view.

### Named Rules
**The One Stack Rule.** Every label, heading and number uses the single system sans stack; mono appears only for paths, code and key hints, at 0.8125em.

**The Tabular Figures Rule.** Tabular numerals are on at the body level, and numeric columns align right against their rule.

## Layout

The page is a bulletin strip over a view switch, then a single main column with a 32px gutter (16px under 860px) and 64px of space at the foot.

- **Bulletin strip:** title and course line on one baseline-aligned row, then a metadata row of term/value pairs separated by vertical hairlines (20px right padding and margin), then the view tabs on their own hairline. A full-ink rule closes the strip.
- **Verdicts view:** a two-column grid, the scorecard board in a flexible column and the research questions in a 300 to 380px column, 40px apart, the right column separated by a vertical hairline and 32px inset. The glyph legend sits directly under the grid, and the evidence panel unfolds under the legend.
- **Explorer view:** a 168px sticky index (16px from the top) beside a flexible content column, 40px apart. Panels are separated by a hairline and 32px of space; side-by-side charts use an equal two-up grid with a 32px gap.
- **Present view:** fixed full-screen, a slide with 6vh/6vw padding and 2.2vh vertical rhythm, over a bar holding Previous, count, a 2px progress line, the key hint, Next and Leave.
- **Scorecard geometry:** the cell size is the unit; the corner column is four cells wide (three under 860px), model headers are 9.25em tall with names running vertically, and the table scrolls horizontally inside its wrapper with the pair column sticky.

Responsive behaviour: at 1100px and below the Verdicts view stacks (the research questions lose their side rule) and the evidence panel stacks its stats over its chart. At 860px and below the gutter halves, the Explorer index becomes a horizontally scrolling row above the content, two-up charts stack, bulletin metadata becomes a two-column grid without dividers, tab padding tightens and keyboard hints hide. In the Present view the cell grows fluidly (26 to 42px) and glyphs grow to 18px.

### Named Rules
**The Hairline Rule.** Structure is drawn with 1px rules only. A full-ink rule opens a block (the end of the bulletin strip, the first research question, the first scorecard row, the count rows, the evidence panel, table header underlines, the Explorer index); grey hairlines divide inside it; the Group Rule tone marks family boundaries within a scorecard block.

## Elevation & Depth

The system is flat. There are no drop shadows, no layered surfaces and no containers; hierarchy comes from rule weight, type weight and tint. Two things sit above the page: the tooltip, an inverted ink block (no shadow, no border) that follows the pointer, and the Present view, a fixed full-screen layer on the page colour. The only `box-shadow` in the build is the 2px inset ring that marks the selected scorecard cell, which is a stroke, not elevation.

### Named Rules
**The Flat Bulletin Rule.** Nothing lifts. Selection is a violet ring and wash, hover is a flat tint, and overlays are solid blocks.

## Shapes

Every corner is square (`rounded.none`): controls, segments, the select, the tooltip, the progress line, and the Streamlit theme (`baseRadius = "none"`). The only non-rectangular forms are the data marks: verdict triangles, the "no test" dash, chart lines with round joins and caps, and the research-question chevron drawn from two 1.5px borders rotated 45 degrees.

### Named Rules
**The Square Corner Rule.** Border radius is zero everywhere; roundness belongs only to line joins in charts.

## Components

### Scorecard Cell (signature)
The module of the whole system: one model on one pair, judged against the task's reference.
- **Shape:** a 28px square bounded by hairlines on its left and top; a Group Rule left edge where a model family begins, and a full-ink left edge where a new task block begins.
- **Glyph:** a 13px triangle centred in the cell (18px in Present). Filled upward: significantly better. Hollow upward: better, not significant. Hollow downward: worse, not significant. Filled downward: significantly worse. A short grey dash: no test or not run. Up glyphs take Verdict Blue, down glyphs Verdict Red, the dash Control Outline.
- **Drawing:** triangles come from one SVG sprite per page (12-unit viewBox; hollow forms stroked at 1.5 with round joins), referenced by each cell. Where a host page strips SVG (the Streamlit form), the same five shapes are drawn as `currentColor` blocks clipped by CSS paths.
- **States:** hover tints the cell Hover Wash. Selecting a cell (pointer, or arrow keys with Home/End and Enter/Space) crosshairs its row and column in Course Wash, rings the cell with a 2px inset violet stroke, turns the column's model name violet at 650, and unfolds the evidence panel beneath the legend. Keyboard focus is a 2px violet outline inset by 2px.
- **Count rows:** two 22px rows under a full-ink rule count pairs significantly better and worse per model, centred, 600 in Ink Secondary; zero shows as a grey middle dot.
- **Headers:** a block header (title at 650 over its loss-and-reference subtitle in Ink Quiet) over vertical model names at 0.75rem; the crosshaired column header takes the wash and full ink.

### Glyph Legend
A wrapping row under a hairline, 0.8125rem Ink Secondary, each entry its glyph (13px) followed by its meaning, 22px between entries.

### Evidence Panel
Opens under a full-ink rule with a clip-path unfold (220ms). A two-column grid: on the left the title (model name in violet), a one-sentence verdict, a definition grid of statistic, p-value, loss gap and days (terms in Ink Quiet, values right-aligned at 550), and the source table path; on the right a monthly cumulative loss-difference trace. Empty state is a single 0.875rem prompt in Ink Quiet.

### Research Question Row
- **Structure:** a disclosure row on a three-column grid (2.6em id column, text, 12px chevron). The id sits in Ink Quiet at 0.75rem, the one-word verdict at Title size, and a one-line result in Ink Secondary below.
- **Rules:** the first row opens with a full-ink rule; rows are divided by hairlines and closed by one.
- **States:** hover tints Raised Paper; open rotates the chevron and unfolds the body (200ms), which holds the question in Ink Quiet, the answer, the tests with their source paths in mono, and an "Open in the Explorer" quiet button.

### Buttons and Segments
- **Quiet button:** square, 1px Control Stroke outline, transparent fill, 0.8125rem at 500 in Ink Secondary; hover fills Hover Wash and turns the text to Ink. Smaller (0.75rem, 5px 9px) inside research-question rows.
- **Segmented control:** quiet buttons joined edge to edge (shared borders). Pressed fills solid Ink with paper text; disabled drops to Group Rule text with no hover.
- **Theme button:** a quiet button at the right end of the tab row cycling system, light and dark, remembered across visits.

### Inputs / Fields
- **Select:** square, 1px Control Stroke outline on the page colour, a 10px drawn chevron 10px from the right edge, 0.8125rem at 500.
- **Control labels:** 0.75rem Ink Quiet beside their control; control rows wrap with 12px by 20px gaps.

### Navigation
- **View tabs:** text tabs at 0.875rem/600 in Ink Quiet with a mono key hint (1, 2, 3); hover and active turn Ink, and the active tab carries a 2px Ink underline that sits on the strip's hairline.
- **Explorer index:** a list of 0.8125rem links divided by hairlines under a full-ink top rule; the current section turns violet at 600. Under 860px it becomes a horizontal scroller.

### Data Tables
Full-width, 0.8125rem. Headers are 0.75rem/600 Ink Quiet, bottom-aligned over a full-ink rule; rows are divided by hairlines with 5px vertical padding and no vertical rules. Numeric columns right-align. Negative and positive figures take the text-weight red and blue; better/worse markers add 600. Rows hover to Raised Paper; pickable rows (strategies) are focusable and, when selected, wash violet with the first cell in violet at 600.

### Charts
Hand-built SVG on the same tokens: hairline gridlines, Control Stroke axes, 12px Ink Quiet text, a full-ink zero line, Stress Band shading for in-sample years and stress episodes. Lines are 2px with round joins; the selected series is violet at 2.5px; unselected alternatives drop to 1px Control Stroke or 1.25px Group Rule. Loss-difference areas use the verdict washes. Direct labels at 600 replace legends where possible; inline legends use a 14px stroke swatch. A dashed Ink Quiet crosshair and dots ringed in the page colour track the pointer, with the tooltip showing values.

### Present View
One statement per slide at the present headline size, a takeaway, then a stage holding an enlarged scorecard, chart or a list of verdicts (5ch id column, verdict at 650, result in Ink Quiet, hairline-divided). Arrow keys, Page Up/Down, Space, Home, End and Escape drive it; each slide enters with a 12px slide and fade (260ms); the 2px progress line scales across in 200ms.

### Streamlit Form
The local app takes the same tokens: its theme mirrors paper, raised paper, hairline, ink and violet (lavender in dark), with square corners and the system sans. It embeds the scorecard, legend and research-question styles resolved for the active theme, draws glyphs with the CSS fallback, and styles its charts from the same token values (hairline grid, Control Stroke axes, violet selected series, Stress Band).

### Motion
Colour changes on hover and state run 120 to 150ms ease-out. Reveals (research-question body, evidence panel) unfold with a top-down clip at 200 to 220ms on `cubic-bezier(0.16, 1, 0.3, 1)`. All animation and transition is removed under reduced-motion preference.

## Do's and Don'ts

### Do:
- **Do** express every result as a triangle in a 28px scorecard cell before reaching for a chart; the cell is the module.
- **Do** keep blue and red for better and worse against the reference, using the darker text variants for figures in running text and tables.
- **Do** paint the reader's selection violet everywhere it appears at once: cell ring, crosshair wash, column header, table row, chart line and focus ring.
- **Do** draw structure with 1px rules, opening each block with a full-ink rule and dividing within it by hairlines.
- **Do** set every number tabular and right-aligned, and put its source table path beside it in mono.
- **Do** define any new colour in both the light block and the dark blocks of the stylesheet.

### Don't:
- **Don't** use cards, drop shadows or rounded containers; the only shadow is the inset selection ring on a cell.
- **Don't** build a row of stat tiles or a grid of chart cards.
- **Don't** use violet for anything other than the current selection.
- **Don't** introduce a second typeface; mono is for paths, code and key hints only.
- **Don't** set labels in uppercase or with tracking; labels are sentence case at 0.75rem.
- **Don't** round any corner.
