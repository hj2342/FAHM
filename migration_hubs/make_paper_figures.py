"""
make_paper_figures.py
---------------------
Create light-background copies of the main project figures for use in the
paper draft. The plotting pipeline currently emits dark-theme figures, so this
script remaps the known dark theme colors to a white-paper equivalent while
preserving the main data colors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "outputs" / "figures"
DST_DIR = ROOT / "outputs" / "paper_figures"

FIG_NAMES = [
    "fig1_eda_overview.png",
    "fig2_pressure.png",
    "fig3_flow_heatmap.png",
    "fig4_network.png",
    "fig5_centrality.png",
    "fig6_correlation_matrix.png",
    "fig7_ols_coefficients.png",
    "fig8_psm.png",
    "fig9_position_heterogeneity.png",
    "fig10_feature_importance.png",
    "fig11_common_support.png",
    "fig12_group_gap.png",
    "fig13_target_league_gap.png",
    "fig14_serie_a_selection_threshold.png",
]


def _dist(rgb: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    base = np.asarray(color, dtype=np.int32)
    diff = rgb.astype(np.int32) - base
    return np.sqrt((diff * diff).sum(axis=2))


def _convert_one(src: Path, dst: Path) -> None:
    img = Image.open(src).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].copy()
    alpha = arr[:, :, 3]

    bg_mask = (_dist(rgb, (13, 17, 23)) < 28) | (_dist(rgb, (22, 27, 34)) < 28)
    grid_mask = _dist(rgb, (45, 51, 59)) < 24
    muted_mask = _dist(rgb, (139, 148, 158)) < 36

    chroma = rgb.max(axis=2).astype(np.int16) - rgb.min(axis=2).astype(np.int16)
    mean_luma = rgb.mean(axis=2)
    white_text_mask = (mean_luma > 205) & (chroma < 38)

    out = rgb.copy()
    out[bg_mask] = np.array([255, 255, 255], dtype=np.uint8)
    out[grid_mask] = np.array([222, 226, 230], dtype=np.uint8)
    out[muted_mask] = np.array([95, 99, 104], dtype=np.uint8)
    out[white_text_mask] = np.array([0, 0, 0], dtype=np.uint8)
    out[alpha == 0] = np.array([255, 255, 255], dtype=np.uint8)

    Image.fromarray(out, mode="RGB").save(dst)


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    made = 0
    for name in FIG_NAMES:
        src = SRC_DIR / name
        dst = DST_DIR / name
        if not src.is_file():
            print(f"[PaperFigures] Missing source figure, skipped: {src.name}")
            continue
        _convert_one(src, dst)
        made += 1
        print(f"[PaperFigures] Wrote {dst}")

    print(f"[PaperFigures] Finished. Created {made} paper-ready figure copies.")


if __name__ == "__main__":
    main()
