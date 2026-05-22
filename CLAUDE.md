# Project overview

Spatial transcriptomics segmentation project. Primary platform: Xenium Prime FFPE skin. Active workstream: V0 boundary-prior pipeline (transcripts → per-pixel lineage → boundary map → CP-SAM cut).

## Three-tier directory structure

- **`sandbox/`** — exploratory iteration space. Standalone notebooks and scripts where new ideas are tested. Produces findings or documents dead ends. **No production code lives here.**
- **`analyses/`** — notebooks grounded in a specific dataset, each ending with a direct data product (a gene signature, a set of extracted ROIs, a QC table, etc.). Inputs and outputs explicitly documented at the top of every notebook.
- **`pipelines/`** — production-ready code. New pipelines follow the **`pipelines/V0/`** module pattern: a self-contained folder with `lib.py`, numbered notebooks, diagnostic `_*.py` scripts, and an `out/` folder. Never a single notebook at the top level.

Tier-specific rules live in `sandbox/CLAUDE.md`, `analyses/CLAUDE.md`, `pipelines/CLAUDE.md`.

## Existing vs. new conventions

Existing files may not yet follow these conventions. **New work follows them; existing work gets cleaned up opportunistically as it's touched.** Do not bulk-reorganize without being asked.

## Working style

Conversational. The user asks for something, reads the figures it produces, and asks again. Notebooks are documentation artifacts maintained silently in the background — they never interrupt the working conversation. When the user says "clean this up", a session has reached a landing point and a review pass is expected.

## Scientist standing instructions

These apply to every active coding session:

1. **Read `FIGURE_STANDARDS.md` at the start of every session.**
2. **Verify every data load with an output cell.** After any data load, emit a verification cell that prints shape, dtype, channel names or column names, and a quick sanity check (e.g. unique values for categorical fields). Never claim what was loaded without this output cell.
3. **Save figures to `scratch/figures/` by default.** Move a figure to a permanent location only when explicitly asked. Filenames must be self-describing: `{dataset}_{what-is-shown}_{key-params}.{ext}`.
4. **Maintain a parallel notebook silently** during sessions. The notebook is a documentation artifact — never interrupts the conversation.
5. **Figure feedback updates `FIGURE_STANDARDS.md` first.** When the user gives feedback on a figure, write the new rule to `FIGURE_STANDARDS.md` immediately — phrased as a general rule, not specific to this figure — *before* remaking the figure.

## Root files

- `FIGURE_STANDARDS.md` — figure conventions; updated continuously from feedback.
- `findings_registry.yaml` — durable findings that downstream code should know about. Entries are proposed by the Knowledge Cartographer, never written by hand.

## Agent roles

These are roles invoked conversationally, not implemented in code. When the user says "act as <role>" or invokes a role inline, follow that role's checklist.

Each role has a detailed checklist in `.claude/roles/`:

- **Reviewer** (`.claude/roles/reviewer.md`) — invoked with "clean this up": audits documentation, narrative, figure quality, data assertions vs. output cells, and that figure feedback was written to `FIGURE_STANDARDS.md`.
- **Knowledge Cartographer** (`.claude/roles/cartographer.md`) — invoked when a session yields a real finding: proposes a `findings_registry.yaml` entry for approval.
- **CI/CD Warden** (`.claude/roles/warden.md`) — invoked before tagging a pipeline version: checks notebooks are runnable on canonical test data.
- **Impact Analyst** (`.claude/roles/impact_analyst.md`) — invoked after a registry entry is approved: identifies affected downstream notebooks and proposes action.
- **Devil's Advocate** (`.claude/roles/devils_advocate.md`) — invoked before finalizing an algorithm decision: challenges parameters, asks for negative controls, checks comparison fairness.
