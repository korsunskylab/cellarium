"""Builder for sandbox/label_smoothing_methods.ipynb (R kernel).

Synthetic test bed for transcript-classification methods that respect a
soft prior with mixed strength (type-specific anchors + ambiguous/multi-class
followers). Tests three methods on the same data:

  1. Naive pooling     — mean of K-NN priors, no anchoring
  2. Label propagation — iterative averaging with anchors clamped
  3. Potts model       — Gibbs sampling on a Potts MRF with anchored states

The Potts implementation is pure R for clarity; small N (~1000-2000) so loops
are fine. No production-grade C++ required per user's instruction.
"""
import nbformat as nbf
from pathlib import Path

nb = nbf.v4.new_notebook()
cells = []
md   = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells.append(md("""# Notebook 03 — Label-smoothing method comparison on synthetic data

Test bed for three approaches to classifying transcripts when each transcript has a *soft prior* over cell-type labels and priors range from strongly informative (specific genes → near-one-hot) to nearly uninformative (housekeeping → near-uniform). The goal is to assign each transcript to a single class while:

- **Never reassigning a type-specific point** to another class (anchors are sacred).
- **Using more information from high-confidence neighbors** than from ambiguous neighbors when classifying uncertain transcripts.

We generate synthetic 2-D transcripts in known cell-type regions with controlled mixes of:
- type-specific points (probability mass concentrated on one type)
- type-ambiguous points (~uniform priors, with controllable noise)
- multi-class points (uniform over a subset, like PTPRC across all immune types)

And compare three methods:
- **(a) Naive pooling** — mean of K-NN priors. Baseline; no anchoring.
- **(b) Confidence-aware label propagation** — iterative averaging with anchors clamped.
- **(c) Potts model** — Gibbs-sampled Markov Random Field with anchor states fixed."""))

cells.append(md("## 0. Setup"))

cells.append(code("""SEED <- 42L
set.seed(SEED)

suppressPackageStartupMessages({
  library(ggplot2); library(dplyr); library(patchwork)
  library(RANN)   # K-NN
})"""))

cells.append(md("""## 1. Generate synthetic data

### 1.1 Layout: 4 cell types in a CHECKERBOARD on a 6×6 grid

Every cell has only **different-type direct neighbors** (horizontal/vertical/diagonal). There's no same-type cluster of cells to use as a coherent reservoir — LP has to propagate purely from within-cell anchors. This is the hardest possible spatial layout for label-propagation methods. Combined with touching cells (no gap) and weak anchors (`specific_anchor = 0.5`), this is the proper stress test."""))

cells.append(code("""# ─── Cell-type layout ─────────────────────────────────────────────────
K          <- 4
type_names <- c("A", "B", "C", "D")
type_palette <- c(A = "#D62728", B = "#1F77B4", C = "#2CA02C", D = "#FF7F0E")

# 6×6 grid of cells (36 cells total)
n_cells_x <- 6; n_cells_y <- 6
cell_size <- 10    # microns per cell
cell_centers <- expand.grid(i = 1:n_cells_x, j = 1:n_cells_y)

# CHECKERBOARD type layout — every cell has only different-type direct neighbors.
# Type by parity of (i, j) indices:
#   (i odd, j odd) → A     (i even, j odd) → B
#   (i odd, j even) → C    (i even, j even) → D
i_even <- (cell_centers$i %% 2L) == 0L
j_even <- (cell_centers$j %% 2L) == 0L
cell_centers$type <- type_names[1L + i_even + 2L * j_even]

cell_centers$cx <- cell_centers$i * cell_size - cell_size / 2
cell_centers$cy <- cell_centers$j * cell_size - cell_size / 2

ggplot(cell_centers, aes(cx, cy, color = type)) +
  geom_point(size = 6) +
  scale_color_manual(values = type_palette) +
  coord_fixed() + theme_minimal() +
  labs(title = "Cell layout — 6×6 grid, 4 types in 2×2 region pattern",
       x = "x (µm)", y = "y (µm)")"""))

cells.append(md("""### 1.2 Transcripts within cells with mixed priors

Each cell produces ~`n_tx_per_cell` transcripts at random positions within its radius. Each transcript gets one of three "kinds" with knobs `p_specific`, `p_ambiguous`, `p_multiclass` (must sum to 1):

- **`specific`** — prior is near-one-hot at the cell's true type
- **`ambiguous`** — prior is uniform plus Gaussian noise (controlled by `ambig_noise`)
- **`multiclass`** — prior is uniform over a subset of types that *includes* the true type (e.g., types {A, B} form an "epithelial" family, {C, D} a "stromal" family)

**Hard-case geometry:** transcripts are uniformly distributed inside *square* cells that fully tile the grid (no gaps between cells). A transcript sitting at the edge of its cell will have K-NN neighbors that span the boundary into the adjacent cell of a different type. This is the stress test for label propagation — at boundaries the local-mean signal is mixed."""))

cells.append(code("""# ─── Tunables ────────────────────────────────────────────────────────
n_tx_per_cell    <- 50      # transcripts per cell
inter_cell_gap   <- 0       # µm — 0 = touching squares; >0 adds a buffer between cells
half_extent      <- cell_size / 2 - inter_cell_gap / 2

p_specific       <- 0.40    # fraction of "specific gene" transcripts
p_ambiguous      <- 0.45    # fraction of "ambiguous / housekeeping" transcripts
p_multiclass     <- 0.15    # fraction of "multi-class gene" transcripts (e.g., PTPRC across immune)
stopifnot(abs(p_specific + p_ambiguous + p_multiclass - 1) < 1e-9)

specific_anchor  <- 0.50    # mass on true type for specific points; rest split uniformly
                            # (was 0.85; lowered to mimic real markers like CD68 which are
                            # tilted-but-not-razor-sharp toward their type)
ambig_noise      <- 0.05    # sd of Gaussian noise added to uniform for ambiguous priors

# Multi-class families: types that look similar (e.g. all immune, all epithelial)
families <- list(
  c("A", "B"),   # "family 1" — types A & B
  c("C", "D")    # "family 2" — types C & D
)

# ─── Generate transcripts ────────────────────────────────────────────
mk_specific_prior <- function(true_type) {
  p <- rep((1 - specific_anchor) / (K - 1), K); names(p) <- type_names
  p[true_type] <- specific_anchor
  p
}
mk_ambiguous_prior <- function(...) {
  p <- 1 / K + rnorm(K, 0, ambig_noise)
  p <- pmax(p, 1e-6); p / sum(p)
}
mk_multiclass_prior <- function(true_type) {
  family <- families[[which(sapply(families, function(f) true_type %in% f))]]
  p <- rep(0, K); names(p) <- type_names
  p[family] <- 1 / length(family)
  # Tiny noise so two multi-class transcripts aren't identical
  p <- p + abs(rnorm(K, 0, 0.005)); p / sum(p)
}

rows <- list()
for (i in seq_len(nrow(cell_centers))) {
  ct <- cell_centers$type[i]; cx <- cell_centers$cx[i]; cy <- cell_centers$cy[i]
  for (j in seq_len(n_tx_per_cell)) {
    # Uniform within a SQUARE cell (no gap to neighbors by default → boundary stress test)
    x <- cx + runif(1, -half_extent, half_extent)
    y <- cy + runif(1, -half_extent, half_extent)

    kind <- sample(c("specific", "ambiguous", "multiclass"),
                   1, prob = c(p_specific, p_ambiguous, p_multiclass))
    prior <- switch(kind,
                    specific   = mk_specific_prior(ct),
                    ambiguous  = mk_ambiguous_prior(),
                    multiclass = mk_multiclass_prior(ct))
    rows[[length(rows) + 1]] <- c(list(x = x, y = y, cell_id = i,
                                       true_type = ct, kind = kind),
                                  setNames(as.list(prior), type_names))
  }
}
df_tx <- do.call(rbind, lapply(rows, as.data.frame))
df_tx$true_type <- factor(df_tx$true_type, levels = type_names)
df_tx$kind      <- factor(df_tx$kind, levels = c("specific", "ambiguous", "multiclass"))
N <- nrow(df_tx)

cat(sprintf("N transcripts: %d  (cells: %d, K types: %d)\\n", N, nrow(cell_centers), K))
cat("By kind:\\n"); print(table(df_tx$kind))
cat("\\nBy true type:\\n"); print(table(df_tx$true_type))"""))

cells.append(md("### 1.3 Visualize ground truth + priors"))

cells.append(code("""prior_mat <- as.matrix(df_tx[, type_names])
df_tx$prior_argmax <- factor(type_names[max.col(prior_mat, ties.method = "first")],
                              levels = type_names)
df_tx$prior_entropy <- -rowSums(prior_mat * log(pmax(prior_mat, 1e-12))) / log(K)  # ∈ [0, 1]

options(repr.plot.width = 14, repr.plot.height = 4.5)
p_truth <- ggplot(df_tx, aes(x, y, color = true_type)) +
  geom_point(size = 0.7) +
  scale_color_manual(values = type_palette, name = "true type") +
  coord_fixed() + theme_void() +
  labs(title = "Ground truth — true_type per transcript")

p_kind <- ggplot(df_tx, aes(x, y, color = kind)) +
  geom_point(size = 0.7) +
  scale_color_manual(values = c(specific = "black", ambiguous = "grey70",
                                 multiclass = "purple")) +
  coord_fixed() + theme_void() +
  labs(title = "Transcript kind (specific/ambiguous/multi-class)")

p_argmax <- ggplot(df_tx, aes(x, y, color = prior_argmax)) +
  geom_point(size = 0.7) +
  scale_color_manual(values = type_palette, name = "prior argmax") +
  coord_fixed() + theme_void() +
  labs(title = "Argmax of prior (what naive single-tx classification gets you)")

p_truth | p_kind | p_argmax"""))

cells.append(code("""options(repr.plot.width = 7, repr.plot.height = 5)
ggplot(df_tx, aes(x, y, color = prior_entropy)) +
  geom_point(size = 0.7) +
  scale_color_viridis_c(option = "magma", limits = c(0, 1), name = "entropy\\n(0=specific\\n1=uniform)") +
  coord_fixed() + theme_void() +
  labs(title = "Prior entropy (the higher, the less informative the gene)")"""))

cells.append(md("""## 2. Spatial graph

K-NN graph on transcript positions. K-NN with k=10 is a stand-in for the Tessera-style Delaunay we'd use in real data; the methods themselves don't care which graph as long as it's local."""))

cells.append(code("""K_NN <- 10
coords <- as.matrix(df_tx[, c("x", "y")])
knn   <- RANN::nn2(coords, k = K_NN + 1)$nn.idx[, -1]   # drop self; n × K_NN

cat(sprintf("K-NN built: %d nodes × %d neighbors each\\n", nrow(knn), ncol(knn)))"""))

cells.append(md("""## 3. Methods

### 3.1 Naive pooling — mean of K-NN priors

Simplest possible smoother: each transcript's posterior = unweighted mean of its prior plus those of its K nearest neighbors. **No anchor protection.** Used as a baseline."""))

cells.append(code("""naive_pool <- function(prior, knn_idx) {
  n <- nrow(prior); K <- ncol(prior)
  post <- matrix(0, n, K, dimnames = dimnames(prior))
  for (i in seq_len(n)) {
    idx <- c(i, knn_idx[i, ])
    post[i, ] <- colMeans(prior[idx, , drop = FALSE])
  }
  post
}

t0 <- Sys.time()
post_naive <- naive_pool(prior_mat, knn)
cat(sprintf("naive pooling: %.2f s\\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))"""))

cells.append(md("""### 3.2 Soft, confidence-aware label propagation

Each transcript gets a per-transcript α ∈ [0, 1] controlling how much of its **prior** is preserved each iteration vs. how much new information **flows in** from K-NN neighbors:

$$p_i^{(t+1)} = \\alpha_i \\cdot p_i^{(0)} \\;+\\; (1 - \\alpha_i) \\cdot \\overline{p}_{j \\in N(i)}^{(t)}$$

- α = 1 → fully clamped, prior never updates (the old hard-anchor behavior).
- α = 0.85 → mostly prior, slow neighbor inflow (soft anchor — protects high-confidence transcripts while letting them be nudged by strong local consensus).
- α = 0.5 → balanced (multi-class genes: keep family bias but defer to neighborhood within the family).
- α = 0 → pure neighborhood averaging (ambiguous transcripts).

Tunables `alpha_specific / alpha_multiclass / alpha_ambiguous` set the inflow rates per kind."""))

cells.append(code("""label_prop_soft <- function(prior, knn_idx, alpha, n_iter = 15) {
  # alpha: length-n vector in [0, 1]. Higher α = more prior preserved.
  n <- nrow(prior); K <- ncol(prior)
  p <- prior
  movers <- which(alpha < 1)   # α=1 transcripts are fully clamped, skip update
  for (iter in seq_len(n_iter)) {
    new_p <- p
    for (i in movers) {
      neighbor_mean <- colMeans(p[knn_idx[i, ], , drop = FALSE])
      new_p[i, ] <- alpha[i] * prior[i, ] + (1 - alpha[i]) * neighbor_mean
    }
    p <- new_p
  }
  p
}

# Per-kind α values
alpha_specific   <- 0.85   # high-confidence: mostly keep prior, small neighbor inflow
alpha_multiclass <- 0.50   # medium: balance prior family-bias with local consensus
alpha_ambiguous  <- 0.00   # low: fully neighbor-determined

alpha_vec <- c(specific = alpha_specific,
               multiclass = alpha_multiclass,
               ambiguous = alpha_ambiguous)[as.character(df_tx$kind)]

# is_anchor still used by Potts in 3.3 (Potts does hard anchoring)
is_anchor <- df_tx$kind == "specific"

cat(sprintf("α distribution: specific=%.2f (n=%d), multiclass=%.2f (n=%d), ambiguous=%.2f (n=%d)\\n",
            alpha_specific,   sum(df_tx$kind == "specific"),
            alpha_multiclass, sum(df_tx$kind == "multiclass"),
            alpha_ambiguous,  sum(df_tx$kind == "ambiguous")))

t0 <- Sys.time()
post_lp <- label_prop_soft(prior_mat, knn, alpha_vec, n_iter = 15)
cat(sprintf("soft label propagation: %.2f s (15 iter)\\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))"""))

cells.append(md("""### 3.3 Potts model (Gibbs sampling, anchors clamped)

Each transcript holds a discrete state s_i ∈ {1, …, K}. Energy:
$$ E(s) = -\\sum_i \\log p_{\\text{prior}}(s_i) - \\beta \\sum_{(i,j) \\in E} \\mathbb{1}[s_i = s_j] $$

Gibbs sampling: at each transcript i, sample s_i ∝ p_prior(s_i = k) · exp(β · #neighbors with label k). Anchors are pinned to their argmax(prior) and never updated. Posterior = empirical distribution from MCMC samples after a burn-in.

β is the smoothness knob — too small ≈ no smoothing, too large ≈ one giant region. We'll start with β = 1.5 (well below the phase transition for K-NN with K=10 neighbors)."""))

cells.append(code("""potts_gibbs_anchored <- function(prior, knn_idx, is_anchor,
                                  beta = 1.5, n_iter = 200, burn = 100,
                                  seed = 42L) {
  set.seed(seed)
  n <- nrow(prior); K <- ncol(prior)
  s <- max.col(prior, ties.method = "first")       # init: argmax of prior
  # Anchors: pin to their argmax
  # (init already at argmax, just never update)
  post_counts <- matrix(0L, n, K)
  log_prior <- log(prior + 1e-12)
  not_anchor <- which(!is_anchor)

  for (iter in seq_len(n_iter)) {
    for (i in not_anchor) {
      counts_k <- tabulate(s[knn_idx[i, ]], nbins = K)
      log_p    <- log_prior[i, ] + beta * counts_k
      probs    <- exp(log_p - max(log_p)); probs <- probs / sum(probs)
      s[i]     <- sample.int(K, 1L, prob = probs)
    }
    if (iter > burn) {
      post_counts[cbind(seq_len(n), s)] <- post_counts[cbind(seq_len(n), s)] + 1L
    }
  }
  post <- post_counts / max(1L, n_iter - burn)
  dimnames(post) <- dimnames(prior)
  post
}

t0 <- Sys.time()
post_potts <- potts_gibbs_anchored(prior_mat, knn, is_anchor,
                                    beta = 1.5, n_iter = 200, burn = 100,
                                    seed = SEED)
cat(sprintf("Potts (β=1.5, 200 iter, 100 burn): %.2f s\\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))"""))

cells.append(md("""## 4. Comparison

### 4.1 Per-transcript classification accuracy

Argmax of each method's posterior vs the ground-truth `true_type`. Broken down by transcript kind so we can see how each method handles anchors vs ambiguous vs multi-class."""))

cells.append(code("""argmax_class <- function(post) factor(type_names[max.col(post, ties.method = "first")],
                                       levels = type_names)
df_tx$pred_prior_argmax <- df_tx$prior_argmax  # already computed
df_tx$pred_naive        <- argmax_class(post_naive)
df_tx$pred_lp           <- argmax_class(post_lp)
df_tx$pred_potts        <- argmax_class(post_potts)

methods <- c("prior_argmax", "naive", "lp", "potts")
accuracy <- function(pred, truth) mean(pred == truth)

acc_overall <- sapply(methods, function(m)
  accuracy(df_tx[[paste0("pred_", m)]], df_tx$true_type))

acc_by_kind <- sapply(methods, function(m)
  tapply(df_tx[[paste0("pred_", m)]] == df_tx$true_type, df_tx$kind, mean))

cat("Overall accuracy (predicted == true_type):\\n")
print(round(acc_overall, 3))

cat("\\nAccuracy by transcript kind:\\n")
print(round(acc_by_kind, 3))"""))

cells.append(md("### 4.2 Anchor preservation — did any specific-gene transcript get reassigned?"))

cells.append(code("""anchor_reassignment <- sapply(methods, function(m) {
  pred <- df_tx[[paste0("pred_", m)]]
  is_specific <- df_tx$kind == "specific"
  sum(pred[is_specific] != df_tx$true_type[is_specific])
})
cat("Number of specific-gene transcripts assigned to a non-true class (lower = better, ideal = 0):\\n")
print(anchor_reassignment)"""))

cells.append(md("### 4.3 Spatial maps — truth vs. each method"))

cells.append(code("""plot_method <- function(col_name, title) {
  ggplot(df_tx, aes(x, y, color = .data[[col_name]])) +
    geom_point(size = 0.7) +
    scale_color_manual(values = type_palette, name = NULL, drop = FALSE) +
    coord_fixed() + theme_void() +
    theme(plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(title = title)
}

options(repr.plot.width = 18, repr.plot.height = 9)
(plot_method("true_type",          "TRUTH") |
 plot_method("pred_prior_argmax",  "argmax of prior (no smoothing)") |
 plot_method("pred_naive",         "naive K-NN pooling")) /
(plot_method("pred_lp",            "label propagation (anchored)") |
 plot_method("pred_potts",         "Potts model (anchored, β=1.5)") |
 plot_method("true_type",          "TRUTH (repeated for comparison)"))"""))

cells.append(md("### 4.4 Posterior uncertainty — entropy maps"))

cells.append(code("""entropy_norm <- function(p) -rowSums(p * log(pmax(p, 1e-12))) / log(ncol(p))
df_tx$ent_prior <- df_tx$prior_entropy
df_tx$ent_naive <- entropy_norm(post_naive)
df_tx$ent_lp    <- entropy_norm(post_lp)
df_tx$ent_potts <- entropy_norm(post_potts)

plot_entropy <- function(col, title) {
  ggplot(df_tx, aes(x, y, color = .data[[col]])) +
    geom_point(size = 0.7) +
    scale_color_viridis_c(option = "magma", limits = c(0, 1), name = "entropy") +
    coord_fixed() + theme_void() +
    theme(plot.title = element_text(hjust = 0.5, size = 10)) +
    labs(title = title)
}

options(repr.plot.width = 16, repr.plot.height = 4.5)
(plot_entropy("ent_prior", "prior entropy") |
 plot_entropy("ent_naive", "naive posterior") |
 plot_entropy("ent_lp",    "label prop posterior") |
 plot_entropy("ent_potts", "Potts posterior"))"""))

cells.append(md("""## 5. Sensitivity sweep — vary the specific/ambiguous mix

How does each method degrade as the proportion of specific (anchor) points decreases? Re-run the pipeline at several `p_specific` levels with `p_multiclass` fixed and `p_ambiguous = 1 - p_specific - p_multiclass`. Plot overall accuracy vs `p_specific` for each method.

This is the sensitivity question your notes raise: **what's `p_specific` in real data?** Answer in the real Xenium panel ≈ 13.6% (from notebook 02). So we want to see how the methods perform near that level."""))

cells.append(code("""run_pipeline <- function(p_spec, p_multi = 0.1, ambig_noise_val = 0.05, seed = 1L) {
  set.seed(seed)
  p_amb <- 1 - p_spec - p_multi
  if (p_amb < 0) stop("p_spec + p_multi must be ≤ 1")

  rows <- list()
  for (i in seq_len(nrow(cell_centers))) {
    ct <- cell_centers$type[i]; cx <- cell_centers$cx[i]; cy <- cell_centers$cy[i]
    for (j in seq_len(n_tx_per_cell)) {
      x <- cx + runif(1, -half_extent, half_extent)
      y <- cy + runif(1, -half_extent, half_extent)
      kind <- sample(c("specific", "ambiguous", "multiclass"),
                     1, prob = c(p_spec, p_amb, p_multi))
      prior <- switch(kind,
                      specific   = mk_specific_prior(ct),
                      ambiguous  = { local_noise <- ambig_noise_val
                                     p <- 1 / K + rnorm(K, 0, local_noise)
                                     p <- pmax(p, 1e-6); p / sum(p) },
                      multiclass = mk_multiclass_prior(ct))
      rows[[length(rows) + 1]] <- c(list(x = x, y = y,
                                         true_type = ct, kind = kind),
                                    setNames(as.list(prior), type_names))
    }
  }
  d <- do.call(rbind, lapply(rows, as.data.frame))
  prior_m <- as.matrix(d[, type_names])
  knn_idx <- RANN::nn2(as.matrix(d[, c("x", "y")]), k = K_NN + 1)$nn.idx[, -1]
  is_anc  <- d$kind == "specific"
  alpha   <- c(specific = alpha_specific,
                multiclass = alpha_multiclass,
                ambiguous = alpha_ambiguous)[as.character(d$kind)]

  p_argmax <- factor(type_names[max.col(prior_m, ties.method = "first")], levels = type_names)
  p_naive  <- argmax_class(naive_pool(prior_m, knn_idx))
  p_lp     <- argmax_class(label_prop_soft(prior_m, knn_idx, alpha, n_iter = 15))
  p_potts  <- argmax_class(potts_gibbs_anchored(prior_m, knn_idx, is_anc,
                                                 beta = 1.5, n_iter = 200, burn = 100,
                                                 seed = seed))
  truth <- factor(d$true_type, levels = type_names)
  c(prior_argmax = mean(p_argmax == truth),
    naive        = mean(p_naive  == truth),
    lp           = mean(p_lp     == truth),
    potts        = mean(p_potts  == truth))
}

p_spec_grid <- c(0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 0.75)
t0 <- Sys.time()
sweep_acc <- sapply(p_spec_grid, function(ps) run_pipeline(ps, seed = 1L))
cat(sprintf("sweep: %.1f s\\n",
            as.numeric(difftime(Sys.time(), t0, units = "secs"))))

sweep_df <- as.data.frame(t(sweep_acc))
sweep_df$p_specific <- p_spec_grid
sweep_long <- tidyr::pivot_longer(sweep_df, -p_specific,
                                   names_to = "method", values_to = "accuracy")

options(repr.plot.width = 9, repr.plot.height = 5)
ggplot(sweep_long, aes(p_specific, accuracy, color = method)) +
  geom_line(linewidth = 1) + geom_point(size = 3) +
  geom_vline(xintercept = 0.136, linetype = "dashed", color = "grey50") +
  annotate("text", x = 0.136, y = 0.45, label = "real data\\n(13.6% specific)",
            hjust = -0.1, size = 3, color = "grey40") +
  scale_y_continuous(limits = c(0, 1), labels = scales::percent) +
  scale_x_continuous(labels = scales::percent) +
  theme_bw() +
  labs(x = "fraction of specific (anchor) transcripts",
       y = "overall classification accuracy",
       title = "Method accuracy vs. anchor density",
       subtitle = "dashed line = real data anchor density (notebook 02)")"""))

cells.append(md("""## 6. Notes for interpreting results

A good method, per the design criteria:

1. **Never reassign type-specific points** — `anchor_reassignment` should be 0 for the anchored methods (label prop + Potts) but >0 for naive pooling (no protection).
2. **Use high-confidence neighbors more** — this shows up as: as `p_specific` shrinks toward zero, anchored methods degrade more gracefully than naive pooling because they propagate from the few strong anchors rather than averaging into uniformity.
3. **Be uncertain where the data is uncertain** — the entropy maps in §4.4 should show posterior entropy *highest at boundaries* (which is correct) and *low in the bulk of each region* (also correct). Naive pooling tends to keep entropy uniformly high; Potts collapses entropy to 0 (hard labels) and label propagation lands somewhere in the middle.

If the sweep shows Potts and label propagation tracking each other closely above some `p_specific` threshold and both beating naive pooling, then **either is fine as the actual implementation choice**. Differentiators below that point:

- **Potts** has sharper boundaries (hard categorical), better for the downstream cpsam cut application.
- **Label propagation** has interpretable posterior probabilities, better for downstream uncertainty-weighted use.

Final knob to remember: `β` for Potts. If the real-data Potts run looks too smoothed or too noisy, increase / decrease β by 2× and re-run."""))

# ── R kernel metadata ────────────────────────────────────────────────────
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "R", "language": "R", "name": "ir"},
    "language_info": {
        "codemirror_mode": "r",
        "file_extension": ".r",
        "mimetype": "text/x-r-source",
        "name": "R",
        "pygments_lexer": "r",
        "version": "4.x",
    },
}

out = Path("/Users/ik936/Partners HealthCare Dropbox/Ilya Korsunsky/cellarium/sandbox/label_smoothing_methods.ipynb")
out.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, str(out))
print("wrote:", out)
