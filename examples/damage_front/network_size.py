# Author: Mehdi Elasmi
# Accuracy of DR-LOD against the network size: networks of several widths and depths are trained for
# a few representative elements (same training as train_model.py), and the relative errors of their
# corrections against the exact LOD correctors of eq. (7) are measured in the energy norm on the test
# parameters.
#   python network_size.py train [widths=[16,32,64,128]] [depths=[2,4,8]] [key=value ...]
#   python network_size.py         (evaluation only, uses the trained networks)
#   python network_size.py plot    (only redraw the figure from network_size.json)
#   root=<folder>: other output folder (default results_size/)
# Results: results_size/network_size.json and the figure results_size/network_size.tex/.pdf. The
# networks are stored in results_size/<case>/; existing networks of the same size in results/ (e.g.
# the full w128 d8 training of train_model.py) are reused.
from pathlib import Path
import sys
import os
import ast
import json
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))
import numpy as np
import training
import util
import figures
from solve_parabolic import problem
import train_model

# corner, boundary edge, and two interior elements of the coarse mesh of train_model.py
_N = train_model.config["N_coarse"][0]
ELEMENTS = [0, _N//3, (_N//3)*_N + _N//3, (_N//2)*_N + _N//2]
ROOT = str(Path(__file__).parent / "results_size") + "/"


def size_config(base, width, depth, overrides):
    cfg = training.config_from_argv(base, overrides)
    cfg.update(width_net=width, depth_net=depth)
    # reuse a complete training of the same size in results/
    full = training.results_path(cfg)
    if all(os.path.exists(full+"correctedHatModel%d.h5" % k) for k in ELEMENTS):
        return cfg, full
    cfg.update(indices=ELEMENTS, result_root=ROOT, plots=False, assemble_GModel=False)
    return cfg, training.results_path(cfg)


def corrector_errors(cfg, path, params):
    """Relative errors of the network corrections of ELEMENTS for the parameters params (only basis
    functions of free nodes), in the energy norm |v_Tj - C_T Lambda_j|_a / |C_T Lambda_j|_a and in
    the L2 norm |v_Tj - C_T Lambda_j|_L2 / |C_T Lambda_j|_L2."""
    Mesh, PDEcoeff, FE = training.build_setting(cfg, params, params)
    free = np.setdiff1d(np.arange(np.prod(FE.N_coarse+1)), util.boundaryNodes(FE.N_coarse))
    M = util.assembleGlobalMatrices(util.localMass(FE.N_fine, a=FE.a), FE.N_fine, 'cg').tocsr()
    errs, errs_l2, abs_a, abs_l2 = [], [], [], []
    for k in ELEMENTS:
        model = util.loadModel(path, k)[0]
        X = Mesh.vertices_fine[Mesh.dofF[k], :]
        N = np.size(X, 0)
        P = FE.parameterMatrix(params)[:, FE.localParameters(k)]
        inp = np.concatenate((np.tile(X, (len(params), 1)), np.repeat(P, N, 0)), 1).astype(np.float32)
        out = np.reshape(model(inp).numpy(), (len(params), N, -1))
        corr = util.localization(X, X[0, :], X[-1, :])[None] * out
        if cfg["exact_constraint"]:
            W = FE.constraintBasis(k)
            corr = corr - np.einsum('nr,crk->cnk', W, np.einsum('nr,cnk->crk', W, corr))
        M_T = M[np.ix_(FE.dofF[k], FE.dofF[k])]
        for i, p in enumerate(params):
            A = PDEcoeff(FE.Mids, [p]).ravel()
            S = FE.stiffness(A)
            S_T = S[np.ix_(FE.dofF[k], FE.dofF[k])]
            C = FE.patchCorrector(A, k, S)
            for o, j in enumerate(FE.nodesID2elem_coarse[k, :]):
                if j in free:
                    e = corr[i, :, o] - C[:, o]
                    abs_a.append(np.sqrt(e @ (S_T @ e)))
                    abs_l2.append(np.sqrt(e @ (M_T @ e)))
                    errs.append(abs_a[-1] / np.sqrt(C[:, o] @ (S_T @ C[:, o])))
                    errs_l2.append(abs_l2[-1] / np.sqrt(C[:, o] @ (M_T @ C[:, o])))
    return np.array(errs), np.array(errs_l2), np.array(abs_a), np.array(abs_l2), int(model.count_params())


def plot(rows):
    """Figure (1 x 3): corrector errors in the energy norm and in the L2 norm against the number of
    weights (mean: solid, maximum: dashed), and the mean energy error against the training time per
    network (the time does not depend on the norm, and the L2 errors behave alike); one colour per
    depth, the points are labelled with the width."""
    colors = {2: "#eb6834", 4: "#1baf7a", 8: "#2a78d6"}
    panels, cost = [], []
    for key, norm in [("err", "energy"), ("err_l2", "$L^2$")]:
        size = []
        for depth in sorted(set(r["depth"] for r in rows)):
            r = sorted([x for x in rows if x["depth"] == depth and key+"_mean" in x], key=lambda x: x["weights"])
            if not r:
                continue
            c = colors.get(depth, "#52514e")
            w = [x["weights"] for x in r]
            labels = ["w%d" % x["width"] for x in r]
            size.append(figures.series(w, [x[key+"_mean"] for x in r], label="depth %d, mean" % depth,
                                       color=c, mark="*", point_labels=labels))
            size.append(figures.series(w, [x[key+"_max"] for x in r], label="depth %d, max" % depth,
                                       color=c, dash="dashed", point_label=False))
            if key == "err":
                cost.append(figures.series([x["time_per_network_s"]/60 for x in r], [x[key+"_mean"] for x in r],
                                           label="depth %d" % depth, color=c, mark="*", point_labels=labels))
        panels.append(figures.line_panel(size, "%s norm: mean (solid), max (dashed)" % norm,
                                         "weights per network", "rel.\\ corrector error [\\%]",
                                         xmode="log", ymode="log", legend=False, labels_at_points=True))
    panels.append(figures.line_panel(cost, "mean energy error vs.\\ training time",
                                     "training time per network [min]", "mean rel.\\ corrector error [\\%]",
                                     xmode="log", ymode="log", legend_pos="north east", labels_at_points=True))
    figures.save_tex(ROOT+"network_size", panels, ncols=3, width_cm=5.0)


def plot_abs(rows):
    """Figure: absolute corrector errors |v_Tj - C_T Lambda_j| in the energy norm and in the L2 norm
    (not divided by the size of the correctors) against the number of weights; mean (solid) and
    maximum (dashed) over all correctors and test parameters."""
    colors = {2: "#eb6834", 4: "#1baf7a", 8: "#2a78d6"}
    panels = []
    for key, title, ylab in [("abs", "energy norm", "$\\|v_{T,j}-\\mathcal C_T\\Lambda_j\\|_a$"),
                             ("abs_l2", "$L^2$ norm", "$\\|v_{T,j}-\\mathcal C_T\\Lambda_j\\|_{L^2}$")]:
        lines = []
        for depth in sorted(set(r["depth"] for r in rows)):
            r = sorted([x for x in rows if x["depth"] == depth and key+"_mean" in x], key=lambda x: x["weights"])
            if not r:
                continue
            c = colors.get(depth, "#52514e")
            w = [x["weights"] for x in r]
            lines.append(figures.series(w, [x[key+"_mean"] for x in r], label="depth %d" % depth, color=c,
                                        mark="*", point_labels=["w%d" % x["width"] for x in r]))
            lines.append(figures.series(w, [x[key+"_max"] for x in r], label="depth %d, max" % depth,
                                        color=c, dash="dashed", point_label=False))
        panels.append(figures.line_panel(lines, title + ": mean (solid), max (dashed)", "weights per network",
                                         ylab, xmode="log", ymode="log", legend=False, labels_at_points=True))
    figures.save_tex(ROOT+"network_size_abs", panels, ncols=2, width_cm=5.0)


def main(argv):
    global ROOT
    if "plot" in argv:
        # only redraw the figure from network_size.json
        opts = dict(a.split("=", 1) for a in argv if a.startswith("root="))
        ROOT = opts.get("root", ROOT).rstrip("/") + "/"
        with open(ROOT+"network_size.json") as f:
            rows = json.load(f)
        plot(rows)
        if any("abs_mean" in r for r in rows):
            plot_abs(rows)
        print("written:", ROOT+"network_size.pdf")
        return
    train = "train" in argv
    opts = dict(a.split("=", 1) for a in argv if a.split("=", 1)[0] in ["widths", "depths", "root"])
    ROOT = opts.get("root", ROOT).rstrip("/") + "/"
    widths = ast.literal_eval(opts.get("widths", "[16, 32, 64, 128]"))
    depths = ast.literal_eval(opts.get("depths", "[2, 4, 8]"))
    overrides = [a for a in argv if "=" in a and a.split("=", 1)[0] not in ["widths", "depths", "root"]]
    P = problem(train_model.config)
    test = [c for c in P["cases"] if c["label"] == "test"][0]["coeff"]
    rows = []
    for depth in depths:
        for width in widths:
            cfg, path = size_config(train_model.config, width, depth, overrides)
            trained = all(os.path.exists(path+"correctedHatModel%d.h5" % k) for k in ELEMENTS)
            if not trained:
                if not train:
                    print("w%d d%d: not trained (run with 'train')" % (width, depth))
                    continue
                print("=== training w%d d%d, elements %s" % (width, depth, ELEMENTS), flush=True)
                training.run_training(cfg)
            e, e_l2, a_abs, l2_abs, n_weights = corrector_errors(cfg, path, test)
            t = np.nan
            if os.path.exists(path+"training_times.json"):
                with open(path+"training_times.json") as f:
                    per = json.load(f).get("per_network", {})
                t = float(np.mean([per[str(k)] for k in ELEMENTS if str(k) in per])) if per else np.nan
            rows.append(dict(width=width, depth=depth, weights=n_weights, err_mean=100*float(e.mean()),
                             err_max=100*float(e.max()), err_l2_mean=100*float(e_l2.mean()),
                             err_l2_max=100*float(e_l2.max()),
                             abs_mean=float(a_abs.mean()), abs_max=float(a_abs.max()),
                             abs_l2_mean=float(l2_abs.mean()), abs_l2_max=float(l2_abs.max()),
                             n_correctors=int(np.size(e)), time_per_network_s=t))
            print("w%-4d d%-2d weights %7d | corrector error energy: mean %6.3f %% max %6.3f %%, L2: mean %6.3f %% max %6.3f %% | %.0f s per network"
                  % (width, depth, n_weights, rows[-1]["err_mean"], rows[-1]["err_max"],
                     rows[-1]["err_l2_mean"], rows[-1]["err_l2_max"], t), flush=True)
    if not rows:
        return
    os.makedirs(ROOT, exist_ok=True)
    # merge with the results of earlier runs (other sizes), the new ones replace the old ones
    if os.path.exists(ROOT+"network_size.json"):
        with open(ROOT+"network_size.json") as f:
            old = json.load(f)
        new = {(r["width"], r["depth"]) for r in rows}
        rows = [r for r in old if (r["width"], r["depth"]) not in new] + rows
    rows = sorted(rows, key=lambda r: (r["depth"], r["width"]))
    with open(ROOT+"network_size.json", "w") as f:
        json.dump(rows, f, indent=1)
    plot(rows)
    plot_abs(rows)
    print("written:", ROOT+"network_size.json", ROOT+"network_size.pdf")


if __name__ == '__main__':
    main(sys.argv[1:])
