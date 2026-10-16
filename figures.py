# Author: Mehdi Elasmi
# Figures for the paper and quick-look images.
# Paper figures: standalone LaTeX files (pgfplots) with their data, compiled to PDF. Fields are
# embedded as exact raster images in pgfplots axes (or as vector cells with vector_fields=True);
# axes, labels, colorbars and line plots are typeset by LaTeX.
# Quick looks: PNG images (matplotlib) of the same panel descriptions.
import os
import shutil
import subprocess
import numpy as np
import matplotlib
import matplotlib.image
import matplotlib.colors
from matplotlib.colors import LinearSegmentedColormap

# sequential (magnitudes) and diverging (signed quantities) colour scales
SEQ = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
       "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
DIV = ["#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb", "#f0efec",
       "#f6cfce", "#ee9493", "#e34948", "#b42f2e", "#7d1a1a"]
COLORMAPS = {"seq": SEQ, "div": DIV}

# fixed colour and line style per method (identity never depends on the order of plotting)
METHODS = {
    "fine FEM": dict(color="#0b0b0b", dash="solid", mark="none"),
    "LOD-ANN": dict(color="#2a78d6", dash="solid", mark="none"),
    "LOD": dict(color="#eb6834", dash="dashed", mark="none"),
    "coarse FEM": dict(color="#1baf7a", dash="dotted", mark="none"),
    "frozen LOD": dict(color="#eda100", dash="dashdotted", mark="none"),
    "Deep Ritz": dict(color="#e87ba4", dash="densely dashed", mark="none"),
    "nodal basis": dict(color="#52514e", dash="dotted", mark="none"),
}
INK, INK2 = "#0b0b0b", "#52514e"


def mpl_cmap(name):
    return LinearSegmentedColormap.from_list(name, COLORMAPS[name])


def _rgb255(hexcolor):
    h = hexcolor.lstrip("#")
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))


def _pgf_colormaps():
    lines = []
    for name, stops in COLORMAPS.items():
        lines.append(r"\pgfplotsset{colormap={%s}{%s}}" % (
            name, " ".join("rgb255=(%d,%d,%d)" % _rgb255(c) for c in stops)))
    for name, style in METHODS.items():
        lines.append(r"\definecolor{%s}{HTML}{%s}" % (_color_name(name), style["color"].lstrip("#").upper()))
    return "\n".join(lines)


def _color_name(method):
    return "c" + "".join(ch for ch in method if ch.isalnum())


def _tex(s):
    # plain text for LaTeX: escape % and _ unless already escaped (math between $...$ is kept)
    import re
    if "$" in s:
        return s
    return re.sub(r"(?<!\\)_", r"\\_", re.sub(r"(?<!\\)%", r"\\%", s))


# ---------------------------------------------------------------------------------------------
# field data
def grid_field(values, Mesh, where="vertices"):
    """2D array (ny, nx) of a function on the fine mesh: values at the fine vertices (Q1
    function) or at the fine elements (piecewise constant)."""
    n = Mesh.N_fine + 1 if where == "vertices" else Mesh.N_fine
    return np.reshape(np.asarray(values).ravel(), (n[1], n[0]))


def _upsample(Z, kind, size=720):
    # raster of the field: bilinear (Q1 functions at vertices) or nearest (element values)
    ny, nx = Z.shape
    if kind == "elements":
        r = max(1, size//max(nx, ny))
        return np.kron(Z, np.ones((r, r)))
    xs = np.linspace(0, nx-1, max(size, nx))
    ys = np.linspace(0, ny-1, max(size, ny))
    i0 = np.clip(np.floor(xs).astype(int), 0, nx-2)
    j0 = np.clip(np.floor(ys).astype(int), 0, ny-2)
    tx = (xs - i0)[None, :]
    ty = (ys - j0)[:, None]
    Z00 = Z[np.ix_(j0, i0)]
    Z01 = Z[np.ix_(j0, i0+1)]
    Z10 = Z[np.ix_(j0+1, i0)]
    Z11 = Z[np.ix_(j0+1, i0+1)]
    return (1-ty)*((1-tx)*Z00 + tx*Z01) + ty*((1-tx)*Z10 + tx*Z11)


def field_panel(Z, title="", cmap="seq", vmin=None, vmax=None, log=False, kind="vertices",
                extent=(0., 1., 0., 1.), cbar_label="", colorbar=True, xlabel="$x$", ylabel="$y$",
                marker=None):
    """Description of a field panel. Z: (ny, nx) array (grid_field). kind: "vertices" (Q1,
    bilinear) or "elements" (piecewise constant). log: logarithmic colour scale. cmap "div" is
    symmetric around 0 if vmin/vmax are not given. marker: optional (x, y) point to mark."""
    Z = np.asarray(Z, dtype=float)
    if cmap == "div" and vmin is None and vmax is None:
        m = np.max(np.abs(Z)) or 1.
        vmin, vmax = -m, m
    if vmin is None:
        vmin = np.min(Z)
    if vmax is None:
        vmax = np.max(Z)
    if vmax <= vmin:
        vmax = vmin + 1E-12
    return dict(type="field", Z=Z, title=title, cmap=cmap, vmin=float(vmin), vmax=float(vmax),
                log=log, kind=kind, extent=tuple(extent), cbar_label=cbar_label,
                colorbar=colorbar, xlabel=xlabel, ylabel=ylabel, marker=marker)


def series(x, y, method=None, label=None, color=None, dash="solid", mark="none", width=1.0,
           point_label=True, point_labels=None):
    """Line (or points with mark) of a line panel; method selects the fixed style of a method."""
    style = dict(METHODS.get(method, {}))
    if color is not None:
        style["color"] = color
    style.setdefault("color", INK2)
    style.setdefault("dash", dash)
    style.setdefault("mark", mark)
    if mark != "none":
        style["mark"] = mark
    return dict(x=np.asarray(x, dtype=float), y=np.asarray(y, dtype=float),
                label=label if label is not None else (method or ""), method=method,
                width=width, point_label=point_label, point_labels=point_labels, **style)


def line_panel(series_list, title="", xlabel="", ylabel="", ymode="normal", xmode="normal",
               legend=True, legend_pos="north east", ymin=None, ymax=None, labels_at_points=False):
    return dict(type="lines", series=series_list, title=title, xlabel=xlabel, ylabel=ylabel,
                ymode=ymode, xmode=xmode, legend=legend, legend_pos=legend_pos, ymin=ymin,
                ymax=ymax, labels_at_points=labels_at_points)


# ---------------------------------------------------------------------------------------------
# LaTeX / pgfplots output
def _num(v):
    return "%.6g" % v


def _log_ticks(vmin, vmax):
    ticks = []
    for e in range(int(np.floor(np.log10(vmin)))-1, int(np.ceil(np.log10(vmax)))+1):
        for m in (1, 2, 3, 5):
            t = m*10.**e
            if vmin*(1-0.03) <= t <= vmax*(1+0.03):
                ticks.append(t)
    if len(ticks) > 7:
        ticks = [t for t in ticks if not np.isclose(t/10**np.floor(np.log10(t)), 3)]
    if vmax > 30*vmin and len(ticks) > 6:
        # wide ranges: powers of ten only (1-2-5 ticks would be too dense)
        ticks = [t for t in ticks if np.isclose(t/10**np.floor(np.log10(t)), 1)] or ticks
    if len(ticks) > 7:
        ticks = [t for t in ticks if np.isclose(t/10**np.floor(np.log10(t)), 1)] or ticks
    while len(ticks) > 8:
        ticks = ticks[::2]
    return ticks


def _fmt_tick(t):
    # large and small powers of ten as 10^k
    if t >= 1E4 or t < 1E-2:
        e = int(np.floor(np.log10(t)+1E-9))
        m = t/10.**e
        return "$10^{%d}$" % e if np.isclose(m, 1) else "$%g\\cdot 10^{%d}$" % (round(m, 2), e)
    return ("%g" % t)


def _field_axis(p, name, i, outdir, vector_fields, first_col=True):
    x0, x1, y0, y1 = p["extent"]
    Z, vmin, vmax = p["Z"], p["vmin"], p["vmax"]
    if p["log"]:
        Z = np.log10(np.maximum(Z, 1E-300))
        vmin, vmax = np.log10(vmin), np.log10(vmax)
    opts = ["xmin=%s, xmax=%s, ymin=%s, ymax=%s" % tuple(map(_num, (x0, x1, y0, y1))),
            "title={%s}" % _tex(p["title"]), "colormap name=%s" % p["cmap"],
            "point meta min=%s, point meta max=%s" % (_num(vmin), _num(vmax)),
            "xlabel={%s}" % p["xlabel"], "ylabel={%s}" % (p["ylabel"] if first_col else "")]
    if p["colorbar"]:
        fixed = p["log"] or (1E-2 <= max(abs(vmin), abs(vmax)) < 1E4)
        cb = ["width=0.18cm", "tick label style={font=\\scriptsize}"]
        if fixed:
            cb += ["title={%s}" % _tex(p["cbar_label"]) if p["cbar_label"] else "",
                   "title style={font=\\small}", "scaled y ticks=false",
                   "yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=3}"]
        if p["log"]:
            ticks = _log_ticks(p["vmin"], p["vmax"])
            cb.append("ytick={%s}, yticklabels={%s}" % (
                ",".join(_num(np.log10(t)) for t in ticks), ",".join(_fmt_tick(t) for t in ticks)))
        opts.append("colorbar, colorbar style={%s}" % ", ".join(c for c in cb if c))
    body = []
    if vector_fields:
        ny, nx = Z.shape
        fname = "%s_field%d.dat" % (name, i)
        if p["kind"] == "elements":
            xs = np.linspace(x0, x1, nx+1)
            ys = np.linspace(y0, y1, ny+1)
            Zp = np.pad(Z, ((0, 1), (0, 1)), mode="edge")
            X, Y = np.meshgrid(xs, ys)
            body.append(r"\addplot[matrix plot*, mesh/cols=%d, point meta=explicit, shader=flat corner] table[meta=z]{%s};" % (nx+1, fname))
        else:
            X, Y = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))
            Zp = Z
            body.append(r"\addplot[matrix plot*, mesh/cols=%d, point meta=explicit, shader=interp] table[meta=z]{%s};" % (nx, fname))
        np.savetxt(os.path.join(outdir, fname), np.column_stack([X.ravel(), Y.ravel(), Zp.ravel()]),
                   header="x y z", comments="", fmt="%.6g")
    else:
        img = "%s_field%d.png" % (name, i)
        R = _upsample(Z, p["kind"])
        matplotlib.image.imsave(os.path.join(outdir, img), np.flipud(R), cmap=mpl_cmap(p["cmap"]),
                                vmin=vmin, vmax=vmax)
        body.append(r"\addplot graphics[xmin=%s, xmax=%s, ymin=%s, ymax=%s] {%s};" % (
            _num(x0), _num(x1), _num(y0), _num(y1), img))
    if p["marker"] is not None:
        body.append(r"\addplot[only marks, mark=*, mark size=1.6pt, draw=white, fill=%s] coordinates {(%s,%s)};" % (
            "black", _num(p["marker"][0]), _num(p["marker"][1])))
    return opts, body


def _axis_ticks(values, mode, axis):
    # explicit ticks 1, 2, 5 x 10^k on logarithmic axes (also for ranges below one decade)
    v = np.concatenate([np.ravel(x) for x in values])
    v = v[np.isfinite(v) & (v > 0)]
    if mode != "log" or np.size(v) == 0:
        return []
    ticks = _log_ticks(np.min(v), np.max(v))
    if len(ticks) < 2:
        ticks = _log_ticks(np.min(v)/2.5, np.max(v)*2.5)
    return ["%stick={%s}, %sticklabels={%s}" % (axis, ",".join(_num(t) for t in ticks), axis,
                                              ",".join(_fmt_tick(t) for t in ticks))]


def _line_axis(p, name, i, outdir):
    scatter = p["labels_at_points"]
    opts = ["title={%s}" % _tex(p["title"]), "xlabel={%s}" % _tex(p["xlabel"]),
            "ylabel={%s}" % _tex(p["ylabel"]), "ymode=%s" % ("log" if p["ymode"] == "log" else "normal"),
            "xmode=%s" % ("log" if p["xmode"] == "log" else "normal"),
            "grid=major, grid style={line width=0.2pt, draw=black!10}",
            "every axis plot/.append style={line join=round}",
            "enlarge x limits=%s, enlarge y limits=%s" % (("0.25", "0.25") if scatter else ("false", "0.08"))]
    if scatter:
        opts.append("clip=false")
    opts += _axis_ticks([s["y"] for s in p["series"]], p["ymode"], "y")
    opts += _axis_ticks([s["x"] for s in p["series"]], p["xmode"], "x")
    if p["ymin"] is not None:
        opts.append("ymin=%s" % _num(p["ymin"]))
    if p["ymax"] is not None:
        opts.append("ymax=%s" % _num(p["ymax"]))
    if p["legend"] and p["legend_pos"] == "below":
        opts.append("legend style={at={(0.5,-0.32)}, anchor=north, legend columns=-1, font=\\scriptsize, draw=none, /tikz/every even column/.append style={column sep=0.3cm}}")
    elif p["legend"]:
        opts.append("legend pos=%s, legend style={font=\\scriptsize, draw=none, fill=white, fill opacity=0.85, text opacity=1}, legend cell align=left" % p["legend_pos"])
    body = []
    for k, s in enumerate(p["series"]):
        fname = "%s_line%d_%d.dat" % (name, i, k)
        labels = s.get("point_labels")
        with open(os.path.join(outdir, fname), "w") as f:
            f.write("x y" + (" label" if labels else "") + "\n")
            for n, (xv, yv) in enumerate(zip(s["x"], s["y"])):
                f.write("%.8g %.8g" % (xv, yv) + (" {%s}" % labels[n] if labels else "") + "\n")
        rgb = "{rgb,255:red,%d;green,%d;blue,%d}" % _rgb255(s["color"])
        style = ["color=%s" % (_color_name(s["method"]) if s["method"] in METHODS and s["color"] == METHODS[s["method"]]["color"] else rgb),
                 s["dash"], "line width=%spt" % _num(s["width"])]
        style.append("mark=%s" % s["mark"] if s["mark"] != "none" else "no markers")
        meta = ""
        if labels:
            # one label per point (third column of the data file)
            style.append("nodes near coords, point meta=explicit symbolic, nodes near coords style={font=\\tiny, anchor=south east, text=black!70}")
            meta = ", meta=label"
        elif p["labels_at_points"] and s["point_label"]:
            style.append("nodes near coords={%s}, nodes near coords style={font=\\scriptsize, anchor=west}" % _tex(s["label"]))
        body.append(r"\addplot[%s] table[x=x, y=y%s]{%s};" % (", ".join(style), meta, fname))
        if p["legend"]:
            body.append(r"\addlegendentry{%s}" % _tex(s["label"]))
    return opts, body


def save_tex(path, panels, ncols=None, width_cm=4.2, height_cm=None, vector_fields=False,
             compile_pdf=True, png=True, vsep="1.6cm"):
    """Writes path.tex (standalone, pgfplots) with its data files into the folder of path and
    compiles it to path.pdf (if pdflatex is available). Also writes a quick-look path.png."""
    outdir = os.path.dirname(os.path.abspath(path))
    name = os.path.basename(path)
    os.makedirs(outdir, exist_ok=True)
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n/ncols))
    has_cbar = any(p["type"] == "field" and p["colorbar"] for p in panels)
    height_cm = height_cm or (width_cm if all(p["type"] == "field" for p in panels) else 0.75*width_cm)
    hsep = "2.2cm" if has_cbar else "1.6cm"
    lines = [r"\documentclass[border=3pt]{standalone}", r"\usepackage{amsmath}", r"\usepackage{pgfplots}",
             r"\pgfplotsset{compat=1.18}", r"\usepgfplotslibrary{groupplots}", _pgf_colormaps(),
             r"\begin{document}", r"\begin{tikzpicture}",
             r"\begin{groupplot}[group style={group size=%d by %d, horizontal sep=%s, vertical sep=%s}, "
             r"width=%scm, height=%scm, scale only axis, enlargelimits=false, axis on top, "
             r"tick label style={font=\scriptsize}, label style={font=\small}, title style={font=\small}]" % (
                 ncols, nrows, hsep, vsep, _num(width_cm), _num(height_cm))]
    for i, p in enumerate(panels):
        if p["type"] == "field":
            opts, body = _field_axis(p, name, i, outdir, vector_fields, first_col=(i % ncols == 0))
        else:
            opts, body = _line_axis(p, name, i, outdir)
        lines.append(r"\nextgroupplot[%s]" % ", ".join(o for o in opts if o))
        lines += body
    lines += [r"\end{groupplot}", r"\end{tikzpicture}", r"\end{document}", ""]
    tex = os.path.join(outdir, name + ".tex")
    with open(tex, "w") as f:
        f.write("\n".join(lines))
    if png:
        save_png(os.path.join(outdir, name + ".png"), panels, ncols)
    if compile_pdf:
        return compile_tex(tex)
    return tex


def compile_tex(tex):
    if shutil.which("pdflatex") is None:
        print("Note: pdflatex not found, " + tex + " not compiled")
        return tex
    outdir, fname = os.path.split(tex)
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", fname],
                       cwd=outdir, capture_output=True, text=True)
    base = fname[:-4]
    for ext in (".aux", ".log"):
        if r.returncode == 0 and os.path.exists(os.path.join(outdir, base+ext)):
            os.remove(os.path.join(outdir, base+ext))
    if r.returncode != 0:
        print("Warning: compiling " + tex + " failed, see " + base + ".log")
        return tex
    return os.path.join(outdir, base + ".pdf")


# ---------------------------------------------------------------------------------------------
# quick looks (matplotlib)
def save_png(path, panels, ncols=None, dpi=130):
    import matplotlib.pyplot as plt
    n = len(panels)
    ncols = ncols or n
    nrows = int(np.ceil(n/ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.0*ncols, 3.4*nrows), squeeze=False)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for p, ax in zip(panels, axes.ravel()):
        if p["type"] == "field":
            x0, x1, y0, y1 = p["extent"]
            norm = matplotlib.colors.LogNorm(p["vmin"], p["vmax"]) if p["log"] else \
                matplotlib.colors.Normalize(p["vmin"], p["vmax"])
            im = ax.imshow(_upsample(p["Z"], p["kind"], 360), origin="lower", extent=(x0, x1, y0, y1),
                           cmap=mpl_cmap(p["cmap"]), norm=norm, interpolation="nearest")
            if p["marker"] is not None:
                ax.plot(*p["marker"], "o", ms=5, mfc=INK, mec="white", mew=1.2)
            if p["colorbar"]:
                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.outline.set_visible(False)
                if p["cbar_label"]:
                    cb.set_label(p["cbar_label"], color=INK2)  # matplotlib mathtext
            ax.set_aspect("equal")
            ax.set_xlabel(p["xlabel"].replace("$", ""), color=INK2)
            ax.set_ylabel(p["ylabel"].replace("$", ""), color=INK2)
        else:
            for s in p["series"]:
                ls = {"solid": "-", "dashed": "--", "dotted": ":", "dashdotted": "-.",
                      "densely dashed": (0, (3, 1)), "densely dashdotted": (0, (3, 1, 1, 1))}.get(s["dash"], "-")
                ax.plot(s["x"], s["y"], linestyle=ls, color=s["color"], lw=1.6, label=s["label"],
                        marker="o" if s["mark"] != "none" else None, ms=5)
                if s.get("point_labels"):
                    for xv, yv, lab in zip(s["x"], s["y"], s["point_labels"]):
                        ax.annotate(lab, (xv, yv), xytext=(-3, 3), textcoords="offset points",
                                    fontsize=7, ha="right", color=INK2)
                elif p["labels_at_points"] and s["point_label"]:
                    for xv, yv in zip(s["x"], s["y"]):
                        ax.annotate(s["label"], (xv, yv), xytext=(4, 0), textcoords="offset points",
                                    fontsize=8, va="center", color=INK)
            if p["ymode"] == "log":
                ax.set_yscale("log")
            if p["xmode"] == "log":
                ax.set_xscale("log")
            if p["ymin"] is not None or p["ymax"] is not None:
                ax.set_ylim(p["ymin"], p["ymax"])
            ax.grid(color="#e6e5e1", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            ax.set_xlabel(p["xlabel"].replace("$", ""), color=INK2)
            ax.set_ylabel(p["ylabel"].replace("$", ""), color=INK2)
            if p["legend"]:
                ax.legend(frameon=False, fontsize=8)
        ax.set_title(p["title"].replace("$", ""), color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def write_overview(folder, entries, title="Figures", compile_pdf=True):
    """LaTeX document folder/figures.tex that includes the compiled figures with captions;
    entries: list of (name, caption)."""
    lines = [r"\documentclass[a4paper,11pt]{article}", r"\usepackage[margin=2cm]{geometry}",
             r"\usepackage{graphicx}", r"\usepackage{amsmath}", r"\begin{document}", r"\section*{%s}" % _tex(title)]
    for name, caption in entries:
        if not os.path.exists(os.path.join(folder, name + ".pdf")) and compile_pdf:
            continue
        lines += [r"\begin{figure}[htbp]\centering",
                  r"\includegraphics[width=\linewidth,height=0.8\textheight,keepaspectratio]{%s.pdf}" % name,
                  r"\caption{%s (\texttt{%s})}" % (caption, name.replace("_", r"\_")), r"\end{figure}",
                  r"\clearpage"]
    lines += [r"\end{document}", ""]
    tex = os.path.join(folder, "figures.tex")
    with open(tex, "w") as f:
        f.write("\n".join(lines))
    return compile_tex(tex) if compile_pdf else tex
