"""Palatino for the paper figures (shared by menagerie_bench/paper_plots.py and panda_safe_mjpc/plot_speedup_paper.py).

TeX Gyre Pagella, the LaTeX Palatino (tgpagella / newpxtext). The files are looked up in ~/.local/share/fonts/tex-gyre
(TrueType conversions of the TeX Gyre OTFs, so matplotlib embeds them as Type 42: the OTFs have CFF outlines, which the
PDF backend embeds as CID Type 0C under a Type 42 label and poppler flags as a mismatch) and, failing that, the OTFs of
the flatpak runtimes; URW's P052 (Palladio, the `palatino`/`mathpazo` font) is the fallback family.

    from paper_fonts import use_palatino
    family = use_palatino()          # sets rcParams (font.family, font.serif, mathtext, pdf.fonttype 42) and returns the family name
"""
import glob, os

FAMILY = "TeX Gyre Pagella"
DIRS = [os.path.expanduser("~/.local/share/fonts/tex-gyre")] + sorted(glob.glob("/var/lib/flatpak/runtime/*/x86_64/*/*/files/share/fonts/tex-gyre"))


def use_palatino():
    import matplotlib
    from matplotlib import font_manager
    files = [f for d in DIRS for f in glob.glob(os.path.join(d, "texgyrepagella-*.ttf"))]
    if not files:
        files = [f for d in DIRS for f in glob.glob(os.path.join(d, "texgyrepagella-*.otf"))]
    for f in files:
        font_manager.fontManager.addfont(f)
    names = {f.name for f in font_manager.fontManager.ttflist}
    fam = FAMILY if FAMILY in names else ("P052" if "P052" in names else "DejaVu Serif")
    if fam != FAMILY:
        print(f"paper_fonts: {FAMILY} not found, using {fam}")
    matplotlib.rcParams.update({"font.family": "serif", "font.serif": [fam, "P052", "Palatino Linotype", "Palatino", "DejaVu Serif"],
                                "pdf.fonttype": 42, "ps.fonttype": 42,
                                "mathtext.fontset": "custom", "mathtext.rm": fam, "mathtext.it": f"{fam}:italic", "mathtext.bf": f"{fam}:bold"})
    return fam
