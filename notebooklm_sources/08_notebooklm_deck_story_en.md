# 08 Deck Storyline (White, Simple, Model-Developer First)

## Design Intent
- Audience: model developers.
- Tone: technical and practical.
- Style: white background, simple layout, low decoration, high readability.

## Story Arc
1. Start from the problem: model training quality depends on input data quality and data understanding.
2. Show input flexibility: files, SQL, Python loader, notebook.
3. Immediately show constraints: one input mode per run, key dependency for multi-table merge.
4. Present EDA (Pandas) and EDA Spark (PySpark) system designs.
5. Show section blocks and functions.
6. Put **all usage patterns on one single slide** first:
- CLI non-interactive,
- CLI interactive,
- API non-interactive,
- API interactive.
- This slide must use generic placeholders (no specific dataset path).
7. Then present dataset-specific case slides (EDA SQL case and EDA Spark `Paypal_data.py` case).
8. Emphasize outputs (`eda_results.json`, `EDA_Report.pdf`) as the core artifacts.
9. Close with action plan: turn EDA findings into Data Cleaning and Feature Engineering tasks.

## Message Priority (Top to Bottom)
1. Input flexibility with explicit boundaries.
2. System design clarity.
3. Usage clarity (one consolidated usage slide + separate case slides).
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
