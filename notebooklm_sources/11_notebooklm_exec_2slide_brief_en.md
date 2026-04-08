# 11 Executive Slide Deck Brief (2 Slides, Senior Management)

## Recommended NotebookLM Setup
- Audience: senior management.
- Format: Presenter Slides.
- Length: short.
- Language: English.
- Tone: executive, concise, consulting-style.
- Visual style: white background, minimal text, simple architecture visual, no code blocks.

## Core Story
- Position EDA and EDA Spark as one product, not two separate tools.
- Main message: one analytical pipeline, dual execution engines.
- Management takeaway: better standardization, better scale, better governance.

## Slide 1
### Title
One EDA Pipeline, Dual Engines

### Headline Message
- One workflow for data profiling and diagnostic analysis.
- Two engines for different scale needs: Pandas and Spark.
- Same logic, same outputs, different execution envelope.

### Core Content
- Unified input layer: files, SQL, notebooks, Python loaders, in-memory DataFrames, and multi-table datasets.
- Common pipeline: data quality, target insight, feature profiling, feature relationships, and drift analysis.
- Dual engines:
  - Pandas for fast local and analyst-led workflows.
  - Spark for distributed processing on large datasets.
- Standard outputs: PDF report, JSON results, and chart assets.
- Access model: available through both CLI and API.

### Speaker Message
- This platform creates one reusable EDA standard across both local and enterprise-scale workloads.
- It reduces the need to maintain separate analysis logic for small and large datasets.

## Slide 2
### Title
Why It Matters to the Business

### Platform Advantages
- Flexible data loader: accepts files, SQL, notebooks, Python loaders, in-memory DataFrames, and multi-table inputs.
- Advantage: reduces data-preparation friction and allows one platform to work across varied project setups.
- Flexible access layer: supports both CLI and API.
- Advantage: enables both operational execution and embedded programmatic use.
- Customizable scope: users can select sections and columns.
- Advantage: supports focused diagnostics instead of forcing full-report runs every time.
- Dual-engine design: Pandas for local workflows, Spark for large-scale processing.
- Advantage: preserves one analytical standard across both lightweight and enterprise-scale workloads.
- Standardized outputs: PDF, JSON, and chart assets.
- Advantage: improves repeatability, review quality, auditability, and cross-team communication.
- Extendable analytical framework: the current six sections provide a core foundation, but functions and checks can be expanded for business-specific needs.
- Advantage: supports future customization for domain use cases without rebuilding the platform from scratch.

### Why Use This Instead of Custom Scripts
- Faster delivery: avoids rebuilding common EDA logic for each project.
- Better consistency: standardizes profiling methods and outputs across teams.
- Better scalability: supports both local and distributed execution without redesign.
- Better maintainability: centralizes reusable logic instead of creating fragmented scripts.
- Better governance: produces structured artifacts that are easier to review, share, and reuse.
- Better extensibility: allows new sections, rules, and diagnostics to be added as business requirements evolve.

### Development Path
- Near term: broaden reusable templates, improve section-level presets, and strengthen workflow adoption across teams.
- Near term: add auto-generated executive summaries and recommended next steps from standardized EDA outputs.
- Medium term: add agentic guidance for issue triage, section selection, and follow-up remediation suggestions.
- Medium term: enable cross-run comparison, monitoring, and change tracking for recurring datasets.
- Long term: evolve the platform into an intelligent diagnostic layer that continuously supports model developers, validators, and management reviews.

### Speaker Message
- The immediate value is flexibility, standardization, and scale.
- The longer-term value is to move from static reporting toward intelligent, AI-assisted diagnostics.

## What NotebookLM Should Emphasize
- Keep the deck to 2 slides only.
- Use short, consulting-style bullets.
- Avoid implementation detail and code examples.
- Treat EDA and EDA Spark as one platform with two engines.
- Prioritize business value over technical walkthrough.

## Custom Prompt for NotebookLM
```text
Create a 2-slide English executive presentation for senior management about EDA and EDA Spark in this repository.

Narrative requirements:
- Present them as one platform with dual engines, not as two separate projects.
- Core message: one EDA pipeline, dual engines.
- Focus on business value, scale, and governance.

Visual requirements:
- White, clean, executive style.
- Simple architecture graphic on Slide 1.
- No code snippets.
- Short bullets only.

Slide 1 requirements:
- Title it around "One EDA Pipeline, Dual Engines".
- Show a simple Input -> Common Pipeline -> Pandas / Spark -> Standard Outputs structure.
- Explicitly mention:
  - multiple input types,
  - common analytical workflow,
  - pandas for local analysis,
  - Spark for large-scale analysis,
  - outputs as PDF, JSON, and charts,
  - access through CLI and API.

Slide 2 requirements:
- Focus on why this matters to the business.
- Explicitly emphasize these advantages:
  - the data loader supports multiple input types and why that is valuable,
  - the access layer supports both CLI and API,
  - users can customize sections and columns,
  - the platform supports both local and Spark execution,
  - standardized outputs improve reuse, governance, and communication.
- Explicitly mention extensibility:
  - the current six sections are the core structure,
  - the platform can be extended with additional functions and business-specific checks over time.
- Include 5 concise bullets on why a model developer should use this platform instead of building custom scripts.
- Include a near-term to long-term development path, with selective use of agentic AI concepts.

Output requirements:
- Exactly 2 slides.
- Executive tone.
- Optimize for senior-management readability, not technical depth.
```

## Suggested Source Set for NotebookLM
- `00_overview.md`
- `02_eda_pipeline.md`
- `03_eda_spark_pipeline.md`
- `05_usage_modes.md`
- `11_notebooklm_exec_2slide_brief_en.md`
