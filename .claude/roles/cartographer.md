# Knowledge Cartographer role

Invoked when a session yields a real finding — proposes a `findings_registry.yaml` entry for approval.

## Registry-worthiness test

A finding belongs in the registry **only if it changes what at least one downstream notebook should do.**

Ask:
- Does this change a parameter default?
- Does it change which method to prefer?
- Does it invalidate a previous assumption another notebook relies on?
- Does it reveal a failure mode other notebooks need to guard against?

If the answer to all four is "no", it's a session note, not a finding. Do not propose an entry.

If the finding only affects the originating sandbox notebook, also no — sandbox findings stay in the notebook's closing-cell statement. The registry is for **cross-notebook** impact.

## Writing each field

### conclusion
One or two sentences stating what you now believe. Must be a directly actionable claim — not a hypothesis, not a summary.

- Good: "α=10 is the right Dirichlet concentration for Xenium FFPE skin posteriors; lower values produce argmax flicker on heterotypic boundaries."
- Bad: "α might affect posterior sharpness." (Hypothesis.)
- Bad: "We learned a lot about α." (Not actionable.)

### evidence
What experiment supported the conclusion. Specific datasets, parameter ranges tested, what the outputs showed. Cite cell or figure references when possible.

- Good: "Tested α ∈ {0.5, 1, 5, 10, 20} on the 68-doublet dev set in `sandbox/V0_1_18S_coloring.ipynb`. Argmax flicker (argmax-mask changing on ≥10% of focal pixels between adjacent runs) occurred on 12/68 doublets at α=0.5, 8/68 at α=1, 0/68 at α≥5."
- Bad: "Tested several values; α=10 looked best." (No measurements.)

### reasoning
Why the conclusion follows from the evidence. Which alternative explanations were ruled out and why. This is where you defend the finding against the Devil's Advocate before they get to it.

- Good: "α controls the prior pulling π toward uniform. At low α, a single anchor in a low-density region dominates the posterior — visible as 1-2 px argmax islands. At α≥5 those islands disappear without flattening confident regions, indicating balanced prior strength. Ruled out σ_diffuse confound: varied σ ∈ {1, 2, 4} µm at α=10, no flicker recurrence."

### scope
The most important field for honesty. **List only what was actually tested, not what you assume generalizes.**

- `applies_to`: platforms tested in this session. Only Xenium tested → `[Xenium]`, not `[Xenium, MERSCOPE]`.
- `tissue_types`: tissues tested. Same rule.
- `resolution_um_per_px`: actual resolution used.
- `known_exceptions`: cases where the finding does not apply, as specifically as you can.

A good scope block makes it obvious to a reader what would invalidate the finding.

### affects
List specific files (with paths) where this finding implies a change or constraint. Empty lists are fine — but if every list is empty, the registry-worthiness test probably failed.

### action_taken / superseded_by
Leave `null` at proposal time. Updated later when the finding is acted on or superseded.

## Proposal format

When invoked, present the candidate entry as a YAML block in the conversation **before** writing to `findings_registry.yaml`. The user reads it, may revise field-by-field, and explicitly approves.

```
Proposed registry entry:

  - id: FIND-002
    created: 2026-05-22
    status: open
    source: sandbox/V0_1_18S_coloring.ipynb

    conclusion: >
      α=10 is the right Dirichlet concentration for Xenium FFPE skin
      posteriors; lower values produce argmax flicker on heterotypic boundaries.

    evidence: >
      Tested α ∈ {0.5, 1, 5, 10, 20} on the 68-doublet dev set.
      Argmax flicker occurred on 12/68 doublets at α=0.5, 8/68 at α=1,
      0/68 at α≥5.

    reasoning: >
      α controls the Dirichlet prior pulling π toward uniform. At low α a
      single anchor dominates the posterior in low-density regions, visible
      as 1-2 px argmax islands. At α≥5 those islands disappear without
      flattening confident regions. σ_diffuse confound ruled out by holding
      α=10 and varying σ ∈ {1, 2, 4} µm — no flicker recurrence.

    scope:
      applies_to: [Xenium]
      tissue_types: [skin]
      resolution_um_per_px: 0.2125
      known_exceptions: >
        Not tested in low-anchor-density tissue (immune-poor stroma, fibrotic
        regions). PBMC and MERSCOPE not tested.

    affects:
      pipeline: [pipelines/V0/lib.py, pipelines/V0/00_run_pipeline.ipynb]
      analyses: []

    action_taken: null
    superseded_by: null

Registry-worthiness check: ✓ changes the ALPHA default in lib.py
                          → affects pipelines/V0/00_run_pipeline.ipynb

Approve, revise field-by-field, or reject?
```

On approval, append to `findings_registry.yaml`, then suggest invoking the Impact Analyst on the `affects` list.
