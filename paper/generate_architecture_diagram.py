"""
Generate a publication-quality architecture diagram for the GAR training pipeline.

Produces a horizontal-flow diagram showing:
  Environments -> Shared Policy Network -> Losses -> Gradients -> Agreement -> Update

Output: architecture_diagram.png at 300 DPI
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
BLUE        = "#3B7DD8"
BLUE_LIGHT  = "#C5D9F0"
GREEN       = "#3DA35D"
GREEN_LIGHT = "#C8E6C9"
ORANGE      = "#E8820C"
ORANGE_LIGHT = "#FDDCB5"
RED         = "#C62828"
RED_LIGHT   = "#FFCDD2"
GREY_BG     = "#FAFAFA"
TEXT_DARK    = "#212121"
TEXT_MID     = "#555555"
ARROW_CLR   = "#666666"

# ---------------------------------------------------------------------------
# Helper: draw a rounded box with optional header
# ---------------------------------------------------------------------------
def draw_box(ax, xy, w, h, facecolor, edgecolor, label=None, label_size=9,
             fontweight="bold", alpha=1.0, linewidth=1.2, zorder=2,
             text_color=TEXT_DARK):
    """Draw a rounded rectangle with a centred label."""
    box = FancyBboxPatch(
        xy, w, h,
        boxstyle="round,pad=0.02",
        facecolor=facecolor, edgecolor=edgecolor,
        linewidth=linewidth, alpha=alpha, zorder=zorder,
    )
    ax.add_patch(box)
    if label is not None:
        cx = xy[0] + w / 2
        cy = xy[1] + h / 2
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=label_size, fontweight=fontweight,
                color=text_color, zorder=zorder + 1)
    return box


def draw_arrow(ax, start, end, color=ARROW_CLR, lw=1.3, style="-|>",
               connectionstyle="arc3,rad=0", shrinkA=2, shrinkB=2, zorder=3):
    """Draw a curved or straight arrow between two points."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        connectionstyle=connectionstyle,
        color=color, linewidth=lw,
        shrinkA=shrinkA, shrinkB=shrinkB,
        zorder=zorder,
        mutation_scale=12,
    )
    ax.add_patch(arrow)
    return arrow


# ---------------------------------------------------------------------------
# Main figure
# ---------------------------------------------------------------------------
def generate_diagram(save_path: str):
    fig, ax = plt.subplots(figsize=(14, 5.2))
    ax.set_xlim(-0.5, 14.5)
    ax.set_ylim(-0.8, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # -----------------------------------------------------------------------
    # Column positions (left edges)
    # -----------------------------------------------------------------------
    col_env   = 0.0
    col_policy = 3.2
    col_loss  = 6.6
    col_grad  = 8.8
    col_agree = 11.0
    col_update = 13.0

    # Row positions (bottom edges) for three variations
    rows = [3.0, 1.6, 0.2]          # top, mid, bottom
    deltas = [0.0, 0.25, 0.5]
    box_w_env = 2.2
    box_h = 0.9

    # ===================================================================
    # 1. Environment boxes (blue)
    # ===================================================================
    env_centres = []
    for i, (y, delta) in enumerate(zip(rows, deltas)):
        draw_box(ax, (col_env, y), box_w_env, box_h,
                 facecolor=BLUE_LIGHT, edgecolor=BLUE,
                 label=f"Env  ($\\delta={delta}$)", label_size=9.5)
        env_centres.append((col_env + box_w_env, y + box_h / 2))

    # Section label
    ax.text(col_env + box_w_env / 2, 4.25,
            "Environment Variations", ha="center", fontsize=10,
            fontweight="bold", color=BLUE)

    # ===================================================================
    # 2. Shared Policy Network (green) -- single tall box
    # ===================================================================
    pn_x = col_policy
    pn_w = 2.6
    pn_h = rows[0] + box_h - rows[2]  # spans all three rows
    pn_y = rows[2]

    # Outer container
    draw_box(ax, (pn_x, pn_y), pn_w, pn_h,
             facecolor=GREEN_LIGHT, edgecolor=GREEN,
             label=None, linewidth=1.6)

    # Section label
    ax.text(pn_x + pn_w / 2, 4.25,
            "Shared Policy Network", ha="center", fontsize=10,
            fontweight="bold", color=GREEN)

    # Internal layers
    layer_w = 2.0
    layer_h = 0.42
    layer_x = pn_x + (pn_w - layer_w) / 2
    layer_labels = ["Input Layer  (obs)", "Hidden 1  (128, ReLU)",
                    "Hidden 2  (128, ReLU)", "Action Output  ($\\pi(a|s)$)"]
    layer_gap = (pn_h - 4 * layer_h) / 5
    layer_centres = []
    for j, lbl in enumerate(layer_labels):
        ly = pn_y + pn_h - layer_h - (j + 1) * layer_gap - j * layer_h
        draw_box(ax, (layer_x, ly), layer_w, layer_h,
                 facecolor="white", edgecolor=GREEN,
                 label=lbl, label_size=7.5, fontweight="normal",
                 linewidth=0.8, text_color=TEXT_MID)
        layer_centres.append((layer_x + layer_w / 2, ly + layer_h / 2))

    # Internal arrows between layers
    for j in range(len(layer_centres) - 1):
        draw_arrow(ax, (layer_centres[j][0], layer_centres[j][1] - layer_h / 2 - 0.01),
                   (layer_centres[j + 1][0], layer_centres[j + 1][1] + layer_h / 2 + 0.01),
                   color=GREEN, lw=0.9, shrinkA=0, shrinkB=0)

    # Arrows: envs -> policy network
    pn_left = pn_x
    for ec in env_centres:
        # Clamp target y to the network box range
        target_y = np.clip(ec[1], pn_y + 0.1, pn_y + pn_h - 0.1)
        draw_arrow(ax, ec, (pn_left, target_y), color=BLUE, lw=1.3)

    # Small "obs" labels on connecting arrows
    for i, ec in enumerate(env_centres):
        mid_x = (ec[0] + pn_left) / 2
        mid_y = (ec[1] + np.clip(ec[1], pn_y + 0.1, pn_y + pn_h - 0.1)) / 2
        ax.text(mid_x, mid_y + 0.18, "$o_{}$".format(i),
                ha="center", fontsize=7.5, color=TEXT_MID, style="italic")

    # ===================================================================
    # 3. Loss boxes (orange outlines)
    # ===================================================================
    pn_right = pn_x + pn_w
    loss_w = 1.5
    loss_centres_right = []
    for i, (y, delta) in enumerate(zip(rows, deltas)):
        draw_box(ax, (col_loss, y), loss_w, box_h,
                 facecolor=ORANGE_LIGHT, edgecolor=ORANGE,
                 label=f"$\\mathcal{{L}}_{{\\delta={delta}}}$",
                 label_size=10)
        loss_centres_right.append((col_loss + loss_w, y + box_h / 2))

        # Arrow from policy to loss
        src_y = np.clip(y + box_h / 2, pn_y + 0.1, pn_y + pn_h - 0.1)
        draw_arrow(ax, (pn_right, src_y), (col_loss, y + box_h / 2),
                   color=ORANGE, lw=1.3)

    ax.text(col_loss + loss_w / 2, 4.25,
            "Per-Variation Loss", ha="center", fontsize=10,
            fontweight="bold", color=ORANGE)

    # ===================================================================
    # 4. Gradient boxes
    # ===================================================================
    grad_w = 1.5
    grad_centres_right = []
    grad_labels = ["$g_0$", "$g_1$", "$g_2$"]
    for i, (y, gl) in enumerate(zip(rows, grad_labels)):
        draw_box(ax, (col_grad, y), grad_w, box_h,
                 facecolor=ORANGE_LIGHT, edgecolor=ORANGE,
                 label=gl, label_size=11)
        grad_centres_right.append((col_grad + grad_w, y + box_h / 2))

        # Arrow from loss to gradient
        draw_arrow(ax, loss_centres_right[i],
                   (col_grad, y + box_h / 2),
                   color=ORANGE, lw=1.3)

    ax.text(col_grad + grad_w / 2, 4.25,
            "Gradients", ha="center", fontsize=10,
            fontweight="bold", color=ORANGE)

    # ===================================================================
    # 5. Agreement module (red)
    # ===================================================================
    agree_w = 1.7
    agree_h_total = rows[0] + box_h - rows[2]
    agree_y = rows[2]

    draw_box(ax, (col_agree, agree_y), agree_w, agree_h_total,
             facecolor=RED_LIGHT, edgecolor=RED,
             label=None, linewidth=1.6)

    # Internal labels
    ax.text(col_agree + agree_w / 2, agree_y + agree_h_total * 0.75,
            "Cosine\nSimilarity\nWeighting", ha="center", va="center",
            fontsize=8.5, fontweight="bold", color=RED)

    ax.text(col_agree + agree_w / 2, agree_y + agree_h_total * 0.35,
            r"$w_i = \frac{\max(0,\;\cos(g_i, \bar{g}))}"
            r"{\sum_j \max(0,\;\cos(g_j, \bar{g}))}$",
            ha="center", va="center", fontsize=7, color=TEXT_DARK)

    ax.text(col_agree + agree_w / 2, agree_y + agree_h_total * 0.12,
            r"$g^* = \sum_i w_i \, g_i$",
            ha="center", va="center", fontsize=8.5,
            fontweight="bold", color=RED)

    ax.text(col_agree + agree_w / 2, 4.25,
            "Gradient Agreement", ha="center", fontsize=10,
            fontweight="bold", color=RED)

    # Arrows: gradients -> agreement
    agree_left = col_agree
    for gc in grad_centres_right:
        target_y = np.clip(gc[1], agree_y + 0.1, agree_y + agree_h_total - 0.1)
        draw_arrow(ax, gc, (agree_left, target_y), color=ORANGE, lw=1.3)

    # ===================================================================
    # 6. Update arrow back to policy
    # ===================================================================
    # Agreed gradient output point (right side of agreement box at mid-height)
    agree_right_x = col_agree + agree_w
    agree_mid_y = agree_y + agree_h_total / 2

    # Small "agreed gradient" label box
    upd_w = 1.1
    upd_h = 0.75
    upd_y = agree_mid_y - upd_h / 2
    draw_box(ax, (col_update, upd_y), upd_w, upd_h,
             facecolor=RED_LIGHT, edgecolor=RED,
             label="$g^*$\nUpdate", label_size=9)

    draw_arrow(ax, (agree_right_x, agree_mid_y),
               (col_update, agree_mid_y), color=RED, lw=1.5)

    # Curved arrow from "Update" back down and left to the policy network
    # We draw this as a path that goes: right of update -> below -> left to policy bottom
    feedback_y = -0.45
    feedback_points = [
        (col_update + upd_w / 2, upd_y),             # bottom of update box
        (col_update + upd_w / 2, feedback_y),         # go down
        (pn_x + pn_w / 2, feedback_y),                # go left
        (pn_x + pn_w / 2, pn_y),                      # go up to policy
    ]

    # Draw as connected line segments with an arrowhead at the end
    for k in range(len(feedback_points) - 1):
        is_last = (k == len(feedback_points) - 1)
        style = "-|>" if is_last else "-"
        draw_arrow(ax, feedback_points[k], feedback_points[k + 1],
                   color=RED, lw=1.5, style=style,
                   shrinkA=0 if k > 0 else 2,
                   shrinkB=2 if is_last else 0)

    ax.text((col_update + upd_w / 2 + pn_x + pn_w / 2) / 2, feedback_y - 0.25,
            "Update shared policy parameters $\\theta$",
            ha="center", va="center", fontsize=8.5, color=RED, fontweight="bold")

    # ===================================================================
    # Final touches
    # ===================================================================
    plt.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved diagram to {save_path}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "architecture_diagram.png")
    generate_diagram(save_path)
