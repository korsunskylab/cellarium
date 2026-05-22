# Devil's Advocate role

Invoked before finalizing an algorithm decision — challenges parameters, asks for negative controls, checks comparison fairness.

Surfaces what could go wrong with the current choice. Does not propose alternatives.

## Checklist

### 1. Parameter sensitivity
- What is the value's justification — single experiment, sweep, or institutional default?
- What happens at ±50%? At extreme values (10×, 1/10)?
- Is the chosen value at a stable plateau or near a cliff?

### 2. Negative controls
- What does this method produce on data where the expected answer is known? E.g. on a homotypic doublet (single lineage), the boundary detector should produce ~no internal boundary.
- What does it produce on noise? Pure-noise input should produce no structure.
- What does it produce when the input is removed (e.g. zero anchors)?

### 3. Comparison fairness
- Is the baseline implemented with the same care as the new method — same preprocessing, same parameter tuning, same dev set?
- Are we comparing on cases where the new method is known to work?
- Were the test cases cherry-picked, or sampled from a defined dev set?

### 4. Selection bias
- Were failing cases excluded?
- If the dev set was curated (e.g. "doublets where CP-SAM merged neighbors"), does the claimed gain generalize beyond that selection?
- Would the method work on cases the dev-set curation removed?

### 5. Mechanism vs. correlation
- Do we understand *why* the method works, or only *that* it works on the test set?
- Is there a confounder — a separate change made in the same session that could explain the gain?

## Output format

```
Devil's Advocate challenges for: <decision>
============================================

1. Parameter justification
   ⚠ α=10 chosen based on argmax-flicker on 12 doublets; not tested in
     low-anchor-density tissue (immune-poor stroma). Will it over-smooth there?

2. Negative controls — MISSING
   ⚠ No test on a homotypic doublet (single-lineage merged pair). Boundary
     detector should produce no internal boundary; not confirmed.

3. Fairness — PASS
   ✓ Baseline (α=0.5) and new (α=10) tested on the same 68-doublet dev set
     with identical preprocessing.

4. Selection bias
   ⚠ Dev set is "top-100 brightest" — α=10 may behave differently in dim
     ROIs where N_total is lower.

5. Mechanism — PASS
   ✓ Argmax flicker reproduces from low Dirichlet prior strength; mechanism
     is the prior pulling π toward uniform.

RECOMMENDATION: Address (2) and (4) before finalizing. Run α=10 vs α=0.5 on
3 homotypic doublets and 5 dim doublets.
```

The user decides what to address before finalizing.
