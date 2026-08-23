#!/usr/bin/env python3
"""
generate_diagrams.py
====================
Generates dissertation diagrams for:
"Scalable Alignment-Free Bacterial Genome Comparison"

USAGE
-----
    python generate_diagrams.py


"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse

os.makedirs("diagrams", exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1: Pipeline Overview
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig1():
    fig, ax = plt.subplots(figsize=(18, 13))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    colors = {
        "input":   "#2c3e50",
        "KALI_non-hash":  "#2980b9",
        "KALI_hash": "#27ae60",
        "fusion":  "#e67e22",
        "output":  "#8e44ad",
    }

    def draw_box(ax, x, y, w, h, label, sublabel="", color="#2c3e50", fontsize=10):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.1", linewidth=1.5,
            edgecolor=color, facecolor=color + "22"
        )
        ax.add_patch(box)
        ax.text(x, y + (0.15 if sublabel else 0), label,
                ha="center", va="center", fontsize=fontsize,
                fontweight="bold", color=color)
        if sublabel:
            ax.text(x, y - 0.25, sublabel, ha="center", va="center",
                    fontsize=12, color="#555555")

    def arrow(ax, x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.8))

    # Input box
    draw_box(ax, 7, 9.0, 4, 0.9,
             "FASTA Genome Files (.fna)",
             "Input: E. coli whole-genome assemblies (NCBI)",
             colors["input"], 16)

    # Three method boxes
    draw_box(ax, 2.2, 6.8, 3.6, 1.1, "KALI_non-hash",
             "Regex motif scan\nFragment length histogram\n(k=3,4,5,6,7 — multi-k)", colors["KALI_non-hash"], 16)
    draw_box(ax, 7,   6.8, 3.6, 1.1, "KALI_hash",
             "Rolling hash encoding\nSpacing distribution histogram\n(k=3,4,5,6,7 — multi-k)", colors["KALI_hash"], 16)
    draw_box(ax, 11.8, 6.8, 3.6, 1.1, "KALI_tensor",
             "Variance-weighted fusion\nPI + PR layers across k values\nPCA across layer dimension", colors["fusion"], 16)

    # Arrows from input to methods
    arrow(ax, 5.0,  8.55, 2.2,  7.35)
    arrow(ax, 7.0,  8.55, 7.0,  7.35)
    arrow(ax, 9.0,  8.55, 11.8, 7.35)

    # Shared bins callout
    ax.text(4.6, 6.1,
            "Shared bin ranges\n(key design decision)",
            ha="center", va="center", fontsize=11, color="#c0392b", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fdecea",
                      edgecolor="#c0392b", lw=1))
    ax.annotate("", xy=(3.2, 6.3),  xytext=(4.1, 6.2),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))
    ax.annotate("", xy=(6.3, 6.3),  xytext=(5.1, 6.2),
                arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2))

    # Distance matrix boxes
    draw_box(ax, 2.2,  4.5, 3.6, 0.9, "Distance Matrix",
             "KALI_non-hash_k3_b50.csv",        colors["KALI_non-hash"],  14)
    draw_box(ax, 7.0,  4.5, 3.6, 0.9, "Distance Matrix",
             "KALI_hash_k3_bins50.csv",    colors["KALI_hash"], 14)
    draw_box(ax, 11.8, 4.5, 3.6, 0.9, "Fused Distance Matrix",
             "KALI_tensor_fused.csv",       colors["fusion"],  14)

    arrow(ax, 2.2,  6.25, 2.2,  4.95)
    arrow(ax, 7.0,  6.25, 7.0,  4.95)
    arrow(ax, 11.8, 6.25, 11.8, 4.95)

    # Validation double-arrow
    ax.annotate("", xy=(5.2, 4.5), xytext=(3.4, 4.5),
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2))
    ax.text(4.3, 4.8, "Validation\n(Pearson r, Spearman ρ)",
            ha="center", va="center", fontsize=10, color="#c0392b", style="italic")

    # Output box
    draw_box(ax, 7, 2.9, 9, 0.9,
             "Comparative Analysis Output",
             "Phylogenetic distances  |  Strain typing  |  Outbreak surveillance",
             colors["output"], 16)
    arrow(ax, 2.2,  4.05, 4.0,  3.35)
    arrow(ax, 7.0,  4.05, 7.0,  3.35)
    arrow(ax, 11.8, 4.05, 10.0, 3.35)

    # AI Interpreter box
    ai_color = "#6c3483"
    draw_box(ax, 7, 1.5, 9, 0.85,
             "KALI-AI Interpretation Module",
             "LLM-powered natural language querying  |  Context-augmented generation  |  Groq / Ollama / OpenRouter",
             ai_color, 16)
    arrow(ax, 7.0, 2.45, 7.0, 1.93)

    # AI callout annotation
    ax.text(11.9, 1.5,
            "Ask plain-language questions\nabout your results in real time",
            ha="center", va="center", fontsize=11, color=ai_color, style="italic",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5eef8",
                      edgecolor=ai_color, lw=1))
    ax.annotate("", xy=(10.3, 1.5), xytext=(11.2, 1.5),
                arrowprops=dict(arrowstyle="->", color=ai_color, lw=1.2))

    ax.set_title("Figure 1: KALI Alignment-Free Genome Comparison Pipeline",
                 fontsize=18, fontweight="bold", pad=12, color="#2c3e50")
    plt.tight_layout()
    plt.savefig("diagrams/fig1_pipeline.png", dpi=150,
                bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print("✓  Figure 1 saved → diagrams/fig1_pipeline.png")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6: Method Selection Decision Tree
# ─────────────────────────────────────────────────────────────────────────────

def draw_fig6():
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 9)
    ax.axis("off")
    fig.patch.set_facecolor("#f8f9fa")

    def dbox(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
        box = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.15", linewidth=1.5,
            edgecolor=color, facecolor=color
        )
        ax.add_patch(box)
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, fontweight="bold",
                color=text_color, multialignment="center")

    def qarrow(ax, x1, y1, x2, y2, label="", lc="#7f8c8d"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=lc, lw=1.8))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=8, color=lc, style="italic")

    # Root question
    dbox(ax, 9.0, 8.2, 6, 0.9,
         "What is your primary research task?", "#2c3e50", 14)

    # Branch questions — evenly spread across wider canvas
    dbox(ax, 2.8, 6.2, 4.2, 1.1,
         "Strain-level\nphylogenetics\nor outbreak typing?", "#34495e", 12)
    dbox(ax, 9.0, 6.2, 4.2, 1.1,
         "Large-scale\nscreening across\ndistant organisms?", "#34495e", 12)
    dbox(ax, 15.2, 6.2, 4.2, 1.1,
         "Multi-resolution\ngenomic\nfingerprinting?", "#34495e", 12)

    qarrow(ax, 7.0,  7.75, 3.5,  6.75, "Strain typing")
    qarrow(ax, 9.0,  7.75, 9.0,  6.75, "Screening")
    qarrow(ax, 11.0, 7.75, 14.5, 6.75, "Multi-scale")

    # Sub-branches for strain typing
    dbox(ax, 1.4, 4.4, 2.8, 0.95, "Need exact\nKALI_non-hash match?",  "#7f8c8d", 11, "white")
    dbox(ax, 4.6, 4.4, 2.8, 0.95, "Speed is\npriority?",        "#7f8c8d", 11, "white")

    qarrow(ax, 2.2, 5.65, 1.4, 4.88, "Yes")
    qarrow(ax, 3.4, 5.65, 4.6, 4.88, "No")

    # Recommendation boxes — well spaced, no overlap
    dbox(ax, 1.4,  2.5, 2.8, 1.2,
         "KALI_non-hash\nk=3–4, bins=50\ncosine distance",      "#2980b9", 11)
    dbox(ax, 4.6,  2.5, 2.8, 1.2,
         "KALI_hash\n--reduce mean\nk=3, bins=50",              "#27ae60", 11)
    dbox(ax, 9.0,  2.5, 3.2, 1.2,
         "KALI_hash / Mash\nk-mer frequency\nfor speed at scale",       "#8e44ad", 11)
    dbox(ax, 15.2, 2.5, 3.2, 1.2,
         "kali_tensor\nvariance-weighted fusion\nk=3 4 5, bins=200",  "#e67e22", 11)

    qarrow(ax, 1.4,  3.93, 1.4,  3.1)
    qarrow(ax, 4.6,  3.93, 4.6,  3.1)
    qarrow(ax, 9.0,  5.65, 9.0,  3.1)
    qarrow(ax, 15.2, 5.65, 15.2, 3.1)

    # Warning banner
    ax.text(
        3.0, 1.5,
        "⚠  High aggregate ρ can mask within-cluster disagreement.\n"
        "    Always validate with within-cluster Spearman ρ.",
        ha="center", fontsize=10, color="#c0392b", style="italic",
        bbox=dict(boxstyle="round", facecolor="#fdecea", edgecolor="#c0392b")
    )

    ax.set_title(
        "Figure 3: Method Selection Framework for "
        "Alignment-Free Bacterial Genome Comparison",
        fontsize=15, fontweight="bold", color="#2c3e50", pad=10)

    plt.tight_layout()
    plt.savefig("diagrams/fig6_decision.png", dpi=150,
                bbox_inches="tight", facecolor="#f8f9fa")
    plt.close()
    print("✓  Figure 6 saved → diagrams/fig6_decision.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Generating dissertation diagrams...\n")
    draw_fig1()
    draw_fig6()
    print("\nAll done — diagrams saved to diagrams/")
