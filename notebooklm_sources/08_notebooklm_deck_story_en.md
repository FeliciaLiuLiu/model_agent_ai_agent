# 08 Deck Storyline (White, Simple, Model-Developer First)

## Design Intent
- Audience: model developers.
- Tone: technical and practical.
- Style: white background, simple layout, low decoration, high readability.

## Story Arc
1. Start from the problem: model training quality depends on input data quality and data understanding.
2. Show input flexibility: files, SQL, Python loader, notebook.
3. Immediately show constraints: one input mode per run, key dependency for multi-table merge.
4. Present EDA (Pandas) full system design: Input -> DataLoader -> Runner -> Sections -> Output.
5. Present EDA Spark full system design with distributed execution perspective.
6. Show sections and functions under each section block.
7. Show exact usage patterns: CLI + API, each in non-interactive and interactive modes.
8. Emphasize outputs (`eda_results.json`, `EDA_Report.pdf`) as the core artifacts.
9. Close with action plan: how report findings become Data Cleaning and Feature Engineering tasks.

## Message Priority (Top to Bottom)
1. Input flexibility with explicit boundaries.
2. System design clarity.
3. Usage clarity (CLI/API + interactive/non-interactive).
4. Output interpretation for next modeling decisions.

## What Not to Include
- Legacy process history.
- Unrelated platform comparisons.
- Long conceptual EDA theory without runnable commands.

## Visual Rules for NotebookLM Prompt
- White canvas.
- Black/gray text with one accent color (blue or green).
- Architecture and comparison tables over decorative graphics.
- Command blocks in monospaced style.
