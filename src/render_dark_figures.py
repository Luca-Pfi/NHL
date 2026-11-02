"""Render every chart again with the dark palette, into figures/dark/ (used by the web page in dark mode).

Same analysis code and numbers as src/analysis.py; only the style tokens change.
Usage: python src/render_dark_figures.py
"""
import shutil
import tempfile
from pathlib import Path

import style as S

# dark steps of the same validated palette slots + dark chart chrome
S.C_T6, S.C_B6, S.C_D = "#3987e5", "#d95926", "#199e70"
S.INK, S.INK2, S.MUTED = "#e6edf3", "#b4bdc8", "#8b949e"
S.GRID, S.AXIS, S.SURFACE = "#262d36", "#39424d", "#161b22"
S.NEUTRAL = "#5c6570"
# dark sequential ramp: low values recede toward the surface, high values brighten
S.BLUE_RAMP = ["#172638", "#1a3150", "#1d3c67", "#20477e", "#245396", "#285fae", "#2c6bc6", "#3277d8", "#3987e5"]
S.COLOR.update({"T6": S.C_T6, "B6": S.C_B6, "D": S.C_D})

import analysis as A  # noqa: E402  (must import after the style overrides)


def main():
    A.FIG = A.ROOT / "figures" / "dark"
    with tempfile.TemporaryDirectory() as tmp:  # keep results/ untouched; the numbers are identical
        A.RES = Path(tmp)
        shutil.copy(A.ROOT / "results" / "model_info.json", A.RES / "model_info.json")
        A.main()
    print(f"dark figures written to {A.FIG}")


if __name__ == "__main__":
    main()
