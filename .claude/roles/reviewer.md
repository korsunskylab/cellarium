# Reviewer role

Invoked with "clean this up" — or any phrase signaling the session has reached a landing point.

The Reviewer audits the notebook(s) touched in the session against documentation, narrative, and figure standards. It does **not** add new analysis or change results.

## Checklist (per notebook)

### 1. Top-of-notebook documentation

For an **analyses/** notebook, the first markdown cell must state:
- [ ] Dataset (path + one-line description)
- [ ] Output product (what it produces, where it lands, what consumes it)
- [ ] Findings (which `findings_registry.yaml` entries it implements/depends on, or "none")

For a **sandbox/** notebook:
- [ ] If dead-end: top-cell comment stating what was tried, what didn't work, why
- [ ] If produced a finding: closing-cell statement of the finding (1-2 sentences)

For a **pipelines/V<n>/** notebook:
- [ ] Title cell with problem statement, pipeline-architecture diagram, constraints, test data

### 2. Data load cells

Every cell that loads from disk or API:
- [ ] Followed by a verification cell printing shape, dtype, channel/column names, sanity check (e.g. unique values for categoricals)
- [ ] If verification reveals something unexpected, flag it — that's a candidate finding

### 3. Figure cells

For each figure:
- [ ] Saved to `scratch/figures/` (unless explicitly promoted)
- [ ] Filename matches `{dataset}_{what-is-shown}_{key-params}.{ext}`
- [ ] Shows input(s) overlaid OR side-by-side with the result (universal rule)
- [ ] Scale bar if spatial
- [ ] Parameter values in title/subtitle, not just code
- [ ] theme_bw / Tableau palette
- [ ] All axes labeled (no bare indices)
- [ ] If displaying an ROI: includes context panel (full-tissue thumbnail with ROI location)

### 4. Narrative flow

- [ ] Each section header precedes code that produces what it promises
- [ ] No leftover dead cells (commented-out experiments, abandoned assignments)
- [ ] Markdown between sections explains what changes between stages
- [ ] No stale string references (e.g. "DB_top100_018" in markdown when code uses 031)

### 5. Data assertions vs. output cells

For every claim made in markdown ("the focal cell is dominated by Fib", "α=10 gives sharper margins"):
- [ ] There is an output cell directly above or below that supports the claim
- [ ] If no supporting output exists, either add one or remove the claim

### 6. FIGURE_STANDARDS.md audit

- [ ] Any figure feedback from this session has been written to `FIGURE_STANDARDS.md` as a general rule
- [ ] If feedback was incorporated into a figure but not into `FIGURE_STANDARDS.md`, fix that now
- [ ] If `FIGURE_STANDARDS.md` was updated this session, every figure in the notebook should comply with the new rule

## Report format

```
Reviewer report for <notebook path>

PASS — Top-of-notebook documentation
FAIL — Data load cells: cell 7 loads transcripts.parquet without a verification cell
PASS — Figure cells (6 figures inspected)
FAIL — Narrative flow: cell 12 references "DB_top100_018" but code uses 031
PASS — Data assertions
FAIL — FIGURE_STANDARDS.md: feedback at turn 24 (colored anchor dots) not yet written as a rule

Action items:
1. Add verification cell after cell 7
2. Update cell 12 markdown to reference 031
3. Add rule to FIGURE_STANDARDS.md: "anchor overlays use lineage colors, not a single color"
```

After producing the report, offer to apply the action items but do not apply them automatically.
