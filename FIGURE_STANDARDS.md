# Figure standards

## Always required
- Save to scratch/figures/ by default. Promote explicitly when the figure is a keeper.
- Self-describing filename: {dataset}_{what-is-shown}_{key-params}.{ext}
  Example: xenium_skin_roi3_gradient-field_sigma1.2.png
- Every dev figure shows the input(s) that produced it: overlaid on, or side-by-side with, the result. Applies universally — if the user asks for a result figure, the inputs are part of the figure.
- If displaying an ROI, include a context panel: thumbnail of full tissue with the ROI location indicated
- Scale bar on all spatial figures
- Parameter values in figure title or subtitle, not just in the code
- theme_bw, Tableau color palette
- No bare index axes — label what the axis represents

## When you receive figure feedback
1. Write the new rule here immediately, before revising the figure
2. State it as a general rule, not specific to this figure
3. Note the original problem it corrects in a comment on the same line
