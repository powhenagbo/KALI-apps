# ─────────────────────────────────────────────────────────────────────────────
# KALI Phylogenetic Tree Comparison
#  (left, roundrect) vs _interval (right, ellipse)
# With tip matching statistics shown as legend on the right side
# ─────────────────────────────────────────────────────────────────────────────

library(ggplot2)
library(dplyr)
library(ggtree)
library(ape)
library(ggnewscale)
library(scales)

# === Paths ===
input_dir  <- "/Users/pauloa/Desktop/Virus/kali_app/projects/outputs"
output_dir <- "/Users/pauloa/Desktop/Virus/kali_app/projects/new"
dir.create(output_dir, showWarnings = FALSE)

# ── 1. Load trees ─────────────────────────────────────────────────────────────
options(ignore.negative.edge = TRUE)

x <- read.tree(file.path(input_dir, "PR_k5_b200.nwk"))
y <- read.tree(file.path(input_dir, "PI_k5_bins200.nwk"))

x$edge.length[x$edge.length < 0] <- 0
y$edge.length[y$edge.length < 0] <- 0

x$tip.label <- gsub("_", " ", x$tip.label)
y$tip.label <- gsub("_", " ", y$tip.label)

fix_tips <- function(s) {
  s <- gsub("O139-H28", "O139:H28", s)
  s <- gsub("O157-H7",  "O157:H7",  s)
  s <- gsub("O127-H6",  "O127:H6",  s)
  s <- gsub("sonnei 1", "sonnei_1",  s)
  s <- gsub("sonnei 2", "sonnei_2",  s)
  s <- gsub("sonnei 3", "sonnei_3",  s)
  s
}

x$tip.label <- fix_tips(x$tip.label)
y$tip.label <- fix_tips(y$tip.label)

cat(" tips:          ", length(x$tip.label), "\n")
cat("_interval tips: ", length(y$tip.label), "\n")

# ── 2. Tip matching statistics ────────────────────────────────────────────────
common_tips  <- intersect(x$tip.label, y$tip.label)
only_in_x    <- setdiff(x$tip.label, y$tip.label)
only_in_y    <- setdiff(y$tip.label, x$tip.label)
total_max    <- max(length(x$tip.label), length(y$tip.label))
match_pct    <- round(100 * length(common_tips) / total_max, 1)

cat("=== Tip Matching ===\n")
cat("Left tree tips:            ", length(x$tip.label), "\n")
cat("Right tree tips:           ", length(y$tip.label), "\n")
cat("Shared between both trees: ", length(common_tips), "/", total_max,
    " =", match_pct, "%\n")
cat("Only in left tree:         ", length(only_in_x), "\n")
cat("Only in right tree:        ", length(only_in_y), "\n")

# ── 3. Load metadata ──────────────────────────────────────────────────────────
meta_raw <- read.csv(file.path(input_dir, "metadata_labels.csv"),
                     stringsAsFactors = FALSE)

names(meta_raw)      <- trimws(names(meta_raw))
meta_raw$Strain_Name <- trimws(meta_raw$Strain_Name)
meta_raw$Color       <- trimws(meta_raw$Color)

meta_join <- data.frame(
  label     = meta_raw$Strain_Name,
  Species   = ifelse(meta_raw$Color == "blue", "E. coli", "Shigella"),
  ShortName = gsub("_", " ", meta_raw$Strain_Name),
  stringsAsFactors = FALSE
)

cat("Tips matched: ", sum(x$tip.label %in% meta_join$label),
    "/", length(x$tip.label), "\n")

# ── 4. Find MRCA nodes ────────────────────────────────────────────────────────
ecoli_tips_x    <- which(x$tip.label %in%
                          meta_join$label[meta_join$Species == "E. coli"])
shigella_tips_x <- which(x$tip.label %in%
                          meta_join$label[meta_join$Species == "Shigella"])
ecoli_tips_y    <- which(y$tip.label %in%
                          meta_join$label[meta_join$Species == "E. coli"])
shigella_tips_y <- which(y$tip.label %in%
                          meta_join$label[meta_join$Species == "Shigella"])

ecoli_node_x    <- getMRCA(x, ecoli_tips_x)
shigella_node_x <- getMRCA(x, shigella_tips_x)
ecoli_node_y    <- getMRCA(y, ecoli_tips_y)
shigella_node_y <- getMRCA(y, shigella_tips_y)

# ── 5. Extract tree data ──────────────────────────────────────────────────────
base1 <- ggtree(x, layout = 'roundrect', branch.length = 'none')
base2 <- ggtree(y, layout = 'ellipse',   branch.length = 'none')

d1 <- base1$data
d2 <- base2$data

d1$Species   <- meta_join$Species[match(d1$label,   meta_join$label)]
d1$ShortName <- meta_join$ShortName[match(d1$label, meta_join$label)]
d2$Species   <- meta_join$Species[match(d2$label,   meta_join$label)]
d2$ShortName <- meta_join$ShortName[match(d2$label, meta_join$label)]

# ── 6. Flip right tree ────────────────────────────────────────────────────────
label_space <- 6
gap         <- 14
d2$x <- max(d2$x) - d2$x + max(d1$x) + label_space + gap

cat("Left tree max x:  ", max(d1$x), "\n")
cat("Right tree min x: ", min(d2$x[d2$isTip]), "\n")
cat("Gap between tips: ", min(d2$x[d2$isTip]) - max(d1$x), "\n")

# ── 6b. Detect position changes between trees ─────────────────────────────────
# Compare the vertical rank (y position) of each shared tip in both trees
tips_x <- d1 %>% filter(isTip == TRUE, label %in% common_tips) %>%
  select(label, y) %>% rename(y_left = y)
tips_y <- d2 %>% filter(isTip == TRUE, label %in% common_tips) %>%
  select(label, y) %>% rename(y_right = y)

tip_comparison <- inner_join(tips_x, tips_y, by = "label") %>%
  mutate(
    rank_left  = rank(y_left),
    rank_right = rank(y_right),
    rank_diff  = abs(rank_left - rank_right),
    # A tip "differs" if its rank changes by more than 2 positions
    position_changed = rank_diff > 2,
    line_color = ifelse(position_changed, "changed", "stable")
  )

n_changed <- sum(tip_comparison$position_changed)
n_stable  <- sum(!tip_comparison$position_changed)

cat("\n=== Position Changes ===\n")
cat("Tips with position change (rank diff > 2):", n_changed, "\n")
cat("Tips with stable position:                 ", n_stable, "\n")
if (n_changed > 0) {
  changed_tips <- tip_comparison %>% filter(position_changed) %>%
    arrange(desc(rank_diff))
  cat("Changed tips (sorted by magnitude):\n")
  for (i in seq_len(nrow(changed_tips))) {
    cat("  ", changed_tips$label[i],
        " — rank shift:", changed_tips$rank_diff[i], "\n")
  }
}

# Tip-matching legend text (bigger box, see size= in annotate("label") below)
match_label <- paste0(
  "Tip Matching\n",
  "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
  "Left:    ", length(x$tip.label), " tips\n",
  "Right:   ", length(y$tip.label), " tips\n",
  "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
  "Shared:  ", length(common_tips), " / ", total_max, "\n",
  "Match:   ", match_pct, "%\n",
  "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
  "Position changed: ", n_changed, "\n",
  "Position stable:  ", n_stable, "\n",
  "\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\n",
  "\u2014\u2014 stable   \u2015\u2015 changed"
)

# ── Connecting lines: tight, visible green (stable) + red (changed) ───────────
connecting_data <- bind_rows(
  d1 %>% filter(isTip == TRUE, label %in% common_tips),
  d2 %>% filter(isTip == TRUE, label %in% common_tips)
) %>%
  left_join(tip_comparison %>% select(label, line_color), by = "label")

# Legend position
legend_x <- max(d2$x, na.rm = TRUE) + 1.5
legend_y <- max(d1$y, na.rm = TRUE) / 2

# ── 7. Build combined plot ────────────────────────────────────────────────────
final_plot <- ggplot() +

  # ── Left tree highlights ──
  geom_hilight(data = d1,
    mapping = aes(subset = node == ecoli_node_x, node = node),
    fill = '#2980b9', alpha = 0.12, extend = 0.3) +
  geom_hilight(data = d1,
    mapping = aes(subset = node == shigella_node_x, node = node),
    fill = '#27ae60', alpha = 0.12, extend = 0.3) +

  # ── Left tree branches ──
  geom_tree(data = d1, layout = 'roundrect',
            color = '#2c3e50', size = 1.1) +

  # ── Left tree tip points ──
  geom_point(data  = d1 %>% filter(isTip == TRUE),
    aes(x = x, y = y, color = Species),
    size = 4, alpha = 0.95) +

  # ── Left tree tip labels ──
  geom_text(data = d1 %>% filter(isTip == TRUE, !is.na(ShortName)),
    aes(x = x + 0.2, y = y, label = ShortName),
    hjust = 0, size = 6.5, color = "black", fontface = "bold") +

  # ── Right tree highlights ──
  geom_hilight(data = d2,
    mapping = aes(subset = node == ecoli_node_y, node = node),
    fill = '#2980b9', alpha = 0.12, extend = 0.3) +
  geom_hilight(data = d2,
    mapping = aes(subset = node == shigella_node_y, node = node),
    fill = '#27ae60', alpha = 0.12, extend = 0.3) +

  # ── Right tree branches ──
  geom_tree(data = d2, layout = 'ellipse',
            color = '#2c3e50', size = 1.1) +

  # ── Right tree tip points ──
  geom_point(data  = d2 %>% filter(isTip == TRUE),
    aes(x = x, y = y, color = Species),
    size = 4, alpha = 0.95) +

  # ── Right tree tip labels ──
  geom_text(data = d2 %>% filter(isTip == TRUE, !is.na(ShortName)),
    aes(x = x - 0.2, y = y, label = ShortName),
    hjust = 1, size = 6.5, color = "black", fontface = "bold") +

  # ── Colour scale ──
  scale_color_manual(
    values   = c('E. coli' = '#2980b9', 'Shigella' = '#27ae60'),
    na.value = '#2c3e50',
    name     = 'Species') +

  # ── Connecting lines: stable = tight dashed green | changed = tight dashed red ──
  geom_line(
    data = connecting_data %>% filter(line_color == "stable"),
    aes(x = x, y = y, group = label),
    color = '#1e8449', alpha = 0.85, linewidth = 1.0, linetype = '42') +
  geom_line(
    data = connecting_data %>% filter(line_color == "changed"),
    aes(x = x, y = y, group = label),
    color = '#e74c3c', alpha = 0.9, linewidth = 1.2, linetype = '42') +

  # ── Rank shift labels on changed tips ──────────────────────────────────────
  {
    changed_mid <- connecting_data %>%
      filter(line_color == "changed") %>%
      group_by(label) %>%
      summarise(mid_x = mean(x), mid_y = mean(y), .groups = "drop") %>%
      left_join(tip_comparison %>% select(label, rank_diff), by = "label")
    geom_label(data = changed_mid,
      aes(x = mid_x, y = mid_y, label = paste0("\u0394", rank_diff)),
      size = 3.5, color = "#e74c3c", fill = "white",
      label.size = 0.3, label.padding = unit(0.2, "lines"), fontface = "bold")
  } +

  # ── Tree title annotations (bigger) ──
  annotate("text",
    x = max(d1$x) / 2, y = max(d1$y) + 0.8,
    label = "KALI_Non-Hash (k=5, bins=200)",
    hjust = 0.5, size = 8, fontface = "bold", color = "#2980b9") +
  annotate("text",
    x = mean(range(d2$x[d2$isTip], na.rm = TRUE)),
    y = max(d2$y) + 0.8,
    label = "KALI_Hashl (k=5, bins=200)",
    hjust = 0.5, size = 8, fontface = "bold", color = "#27ae60") +

  # ── Tip matching statistics box (bigger text) ────────────────────────────────
  annotate("label",
    x            = legend_x,
    y            = legend_y,
    label        = match_label,
    hjust        = 0,
    vjust        = 0.5,
    size         = 6,
    color        = "grey20",
    family       = "mono",
    lineheight   = 1.5,
    label.size   = 0.5,
    label.padding = unit(0.6, "lines"),
    fill         = "#f0f4f8",
    alpha        = 0.95) +

  # ── Layout ──
  xlim(-1, max(d2$x, na.rm = TRUE) + 12) +
  theme_tree2() +
  theme(
    legend.position  = 'bottom',
    legend.box       = 'horizontal',
    legend.text      = element_text(size = 13),
    legend.title     = element_text(size = 13, face = 'bold'),
    plot.title       = element_text(hjust = 0.5, face = 'bold', size = 24),
    plot.subtitle    = element_text(hjust = 0.5, size = 14, color = 'grey40'),
    axis.text.x      = element_blank(),
    axis.ticks.x     = element_blank(),
    panel.background = element_rect(fill = 'white'),
    plot.background  = element_rect(fill = 'white')) +
  labs(
    title    = "Neighbour-Joining Tree Comparison: KALI_Non-Hash vs KALI_Hash",
    subtitle = paste0(
      "30 genomes — 18 E. coli (blue) + 12 Shigella (green) | k=3 | ",
      "Green = stable position | Red = position changed (\u0394 = rank shift) | ",
      n_changed, " changed, ", n_stable, " stable"
    ))

# ── 8. Save ───────────────────────────────────────────────────────────────────
ggsave(file.path(output_dir, "f5_b200_nj_comparison.pdf"),
  plot = final_plot, width = 36, height = 20, limitsize = FALSE)

ggsave(file.path(output_dir, "f5_b200_nj_comparison.png"),
  plot = final_plot, width = 36, height = 20, dpi = 150, limitsize = FALSE)

cat("\nSaved to:", output_dir, "\n")
cat("  f5_b200_nj_comparison.pdf\n")
cat("  f5_b200_nj_comparison.png\n")
