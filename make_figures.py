# Author: Mehdi Elasmi
"""Figures of the results of a trained example, for the paper and for the comparison of methods.

    python make_figures.py examples/<example> [key=value ...] [--baseline] [--no-compile]
                           [--vector-fields]

key=value: the same overrides as for train_model.py (they determine the results folder).
--baseline: also run the classical Deep Ritz baseline (one network on the whole domain, trained in
every time step, baselines.DeepRitzStepper) for the test case.
--no-compile: only write the LaTeX files (compile them later with pdflatex).
--vector-fields: fields as vector graphics instead of raster images inside the pgfplots axes.

The example folder must contain train_model.py (config) and solve_parabolic.py with a function
problem(config) (right-hand side, initial value, heat capacity, evaluation cases). The figures,
their data, summary.tex/.json and an overview figures.tex/.pdf are written to
<results folder>/figures/.
"""
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import glob
import json
import timeit
import importlib.util
import numpy as np
import training
import parabolic
import solvers
import util
import figures as F


def load_example(folder, overrides):
    folder = os.path.abspath(folder)
    sys.path.insert(0, folder)
    names = ["train_model", "solve_parabolic"]
    for mod in names:
        sys.modules.pop(mod, None)
    modules = {}
    for mod in names:
        spec = importlib.util.spec_from_file_location(mod, os.path.join(folder, mod+".py"))
        modules[mod] = importlib.util.module_from_spec(spec)
        sys.modules[mod] = modules[mod]
        spec.loader.exec_module(modules[mod])
    assert hasattr(modules["solve_parabolic"], "problem"), \
        "solve_parabolic.py of the example needs a function problem(config)"
    config = training.config_from_argv(modules["train_model"].config, overrides)
    return config, modules["solve_parabolic"].problem


def rel_error(u, ref):
    # relative error per time step (columns), from the first step on
    return np.linalg.norm(u[:, 1:]-ref[:, 1:], axis=0)/np.linalg.norm(ref[:, 1:], axis=0)


def minimal_energy(FE, PDEcoeff, params, idx):
    """Exact minimal energy of the training problem of the element (or node) idx, summed over the
    outputs and the coefficients with the parameters params: sum of J(alpha*) = -1/2 r'alpha*, where
    alpha* are the exact discrete corrections (Corollary 3.10 of the paper)."""
    total = 0.
    for c in params:
        A = PDEcoeff(FE.Mids, [c]).flatten()
        total += -0.5*np.sum(FE.patchRHS(A, idx)*FE.patchCorrector(A, idx))
    return total


def training_time(loadPath):
    # offline cost: recorded by training.py, otherwise estimated from the model files
    path = loadPath+"training_times.json"
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)["total"], "recorded"
    files = sorted(glob.glob(glob.escape(loadPath)+"correctedHatModel*.h5"), key=os.path.getmtime)
    if len(files) < 2:
        return np.nan, "unknown"
    t = [os.path.getmtime(f) for f in files]
    return (t[-1]-t[0])*len(files)/(len(files)-1), "estimated from file times"


def read_logs(loadPath, index):
    # scalars of the TensorBoard logs of the network index (latest run)
    import tensorflow as tf
    runs = sorted(glob.glob(glob.escape(loadPath)+"logs/train/*/BF_"+str(index)))
    if not runs:
        return {}
    out = {}
    for f in glob.glob(os.path.join(glob.escape(runs[-1]), "events.out.tfevents.*")):
        for e in tf.compat.v1.train.summary_iterator(f):
            for v in e.summary.value:
                val = tf.make_ndarray(v.tensor) if v.HasField("tensor") else v.simple_value
                out.setdefault(v.tag, []).append((e.step, float(np.ravel(val)[0])))
    return {k: np.array(sorted(v)) for k, v in out.items()}


def crop(Z, G, FE, pad=1):
    # crop a vertex field to the support of the basis function G (plus pad fine nodes)
    n = FE.N_fine+1
    Gz = np.reshape(np.abs(G), (n[1], n[0]))
    iy, ix = np.where(Gz > 1E-10*Gz.max())
    i0, i1 = max(ix.min()-pad, 0), min(ix.max()+pad, n[0]-1)
    j0, j1 = max(iy.min()-pad, 0), min(iy.max()+pad, n[1]-1)
    xs = np.unique(FE.vertices_fine[:, 0])
    ys = np.unique(FE.vertices_fine[:, 1])
    return Z[j0:j1+1, i0:i1+1], (xs[i0], xs[i1], ys[j0], ys[j1]), (i0, i1)


def tex_num(v):
    # number for LaTeX: 3 significant digits, powers of ten as 5\cdot 10^{3}
    s = "%.3g" % v
    if "e" in s:
        m, e = s.split("e")
        e = int(e)
        return ("10^{%d}" % e) if m == "1" else "%s\\cdot 10^{%d}" % (m, e)
    return s


def representative_nodes(NC):
    # interior node at the center, next to a corner, next to the middle of an edge
    nodes = []
    for ix, iy in [(NC[0]//2, NC[1]//2), (1, 1), (NC[0]//2, 1)]:
        j = iy*(NC[0]+1)+ix
        if 0 < ix < NC[0] and 0 < iy < NC[1] and j not in nodes:
            nodes.append(j)
    return nodes


def main(argv):
    flags = [a for a in argv if a.startswith("--")]
    args = [a for a in argv if not a.startswith("--")]
    assert len(args) >= 1, __doc__
    compile_pdf = "--no-compile" not in flags
    vector = "--vector-fields" in flags
    config, problem = load_example(args[0], args[1:])
    loadPath = training.results_path(config)
    assert os.path.exists(loadPath), "not trained yet: " + loadPath
    out = loadPath+"figures/"
    os.makedirs(out, exist_ok=True)
    P = problem(config)
    save = dict(compile_pdf=compile_pdf, vector_fields=vector)
    entries, summary = [], {"example": os.path.basename(os.path.abspath(args[0])), "cases": {}}

    # ---- all methods for all cases
    R = {}
    for case in P["cases"]:
        lab = case["label"]
        print("=== case", lab)
        res = parabolic.run_case(loadPath, case["coeff"], case["T"], P["rhs"], P["u0"],
                                 mass_fun=P["mass_fun"], label=lab, plotCoeff=False)
        FE, T = res["FE"], res["T"]
        N = np.size(T)
        args_t = (res["A_stiff"], res["A_mass"], res["rhs"], res["u0"], T)
        t0 = timeit.default_timer()
        res["u_FEM"], _ = solvers.solveFineFEMParabolic(FE, *args_t)
        t_fem = (timeit.default_timer()-t0)/(N-1)
        t0 = timeit.default_timer()
        res["u_coarse"], _ = solvers.solveCoarseFEMParabolic(FE, *args_t)
        t_coarse = (timeit.default_timer()-t0)/(N-1)
        G_frozen = util.Bases([res["G_LOD"].sparse(0)]*N)
        res["u_frozen"], _ = solvers.solveHomParabolic(FE, G_frozen, *args_t)
        tm = res["timings"]
        cost = {"fine FEM": t_fem, "coarse FEM": t_coarse,
                "LOD": tm["lod_basis"] + tm["coarse_step"],
                "frozen LOD": tm["lod_basis"]/(N-1) + tm["coarse_step"],
                "LODANN": tm["ann_basis"] + tm["coarse_step"]}
        sol = {"coarse FEM": res["u_coarse"], "frozen LOD": res["u_frozen"],
               "LODANN": res["u_PINN"], "LOD": res["u_LOD"]}
        if "--baseline" in flags and lab == "test":
            # the baseline is expensive: computed once and stored in figures/deep_ritz_test.npz
            # (delete the file to recompute it)
            cache = out+"deep_ritz_%s.npz" % lab
            c = np.load(cache) if os.path.exists(cache) else None
            if c is not None and np.shape(c["T"]) == np.shape(T) and np.allclose(c["T"], T):
                res["u_DR"], cost["Deep Ritz"] = c["u"], float(c["cost"])
                print("=== Deep Ritz baseline loaded from", cache)
            else:
                # no cache, or computed for other time points
                import baselines
                print("=== Deep Ritz baseline (test case)")
                res["u_DR"], cost["Deep Ritz"] = baselines.DeepRitzStepper(FE).solve(*args_t)
                np.savez(cache, u=res["u_DR"], cost=cost["Deep Ritz"], T=T)
            sol["Deep Ritz"] = res["u_DR"]
        free = res["free"]
        res["basis_error"] = {"LODANN": float(np.mean(parabolic.rel_basis_error(res["G_PINN"], res["G_LOD"], free)))}
        res["errors"] = {m: rel_error(u, res["u_FEM"]) for m, u in sol.items()}
        res["errors"]["LODANN vs LOD"] = rel_error(res["u_PINN"], res["u_LOD"])
        res["cost"] = cost
        R[lab] = res
        summary["cases"][lab] = {
            "error_final_vs_fineFEM": {m: float(e[-1]) for m, e in res["errors"].items()},
            "error_mean_vs_fineFEM": {m: float(np.mean(e)) for m, e in res["errors"].items()},
            "online_cost_per_step_s": cost,
            "basis_error_vs_LOD_percent": res["basis_error"],
            "timings": {k: float(v) for k, v in tm.items()}}
    t_train, how = training_time(loadPath)
    summary["offline_training_s"] = t_train
    summary["offline_training_note"] = how

    cases = list(R.keys())
    test = R[cases[-1]]
    FE = test["FE"]
    T = test["T"]
    N = np.size(T)
    times = [1 if N > 1 else 0, N//2, N-1]
    tl = lambda i, r=test: "$t=%s$" % tex_num(r["T"][i])
    NC = FE.N_coarse
    xs = np.unique(FE.vertices_fine[:, 0])
    ys = np.unique(FE.vertices_fine[:, 1])
    coef = lambda r, i: np.reshape(r["A_stiff"][:, i], (FE.N_fine[1], FE.N_fine[0]))

    def coef_panel(Z, title, colorbar=True):
        return F.field_panel(Z, title, kind="elements", log=Z.max() > 10*Z.min() > 0,
                             cbar_label="$a$", colorbar=colorbar)

    # ---- training losses (cf. Fig. 3)
    idx = (NC[1]//2)*NC[0]+NC[0]//2 if config["elementwise"] else (NC[1]//2)*(NC[0]+1)+NC[0]//2
    logs = read_logs(loadPath, idx)
    if "energy's loss" in logs:
        e = logs["energy's loss"]
        lines = [F.series(e[:, 0], e[:, 1], label="Adam", color=F.METHODS["LODANN"]["color"])]
        if "L-BFGS energy's loss" in logs:
            l = logs["L-BFGS energy's loss"]
            lines.append(F.series(l[:, 0], l[:, 1], label="L-BFGS", color=F.METHODS["LOD"]["color"], dash="dashed"))
        # relative error of the network corrections in the energy norm over the iterations, from the
        # loss and the exact minimal energy L* (Corollary 3.10 of the paper):
        # ((L - L*)/|L*|)^(1/2) = (sum ||P beta - alpha*||_S^2 / sum ||alpha*||_S^2)^(1/2).
        # Adam: all training coefficients (mini-batch losses are scaled to the full sum),
        # L-BFGS: the fixed subset of the coefficients used by the fine tuning
        PDEcoeff = test["PDEcoeff"]
        n_par = getattr(PDEcoeff, "n_par", 1)
        train = np.loadtxt(loadPath+"trainParameter_Iter0.txt")
        train = np.reshape(train, (-1, n_par)) if n_par > 1 else np.ravel(train)
        N = np.shape(train)[0]
        sub = np.arange(N)
        if 0 < config.get("lbfgs_coeffs", 0) < N:
            sub = np.sort(np.random.default_rng(idx).choice(N, config["lbfgs_coeffs"], replace=False))
        L_star = {"Adam": minimal_energy(FE, PDEcoeff, train, idx)}
        L_star["L-BFGS"] = L_star["Adam"] if np.size(sub) == N else minimal_energy(FE, PDEcoeff, train[sub], idx)
        err = []
        for s in lines:
            g = (s["y"]-L_star[s["label"]])/abs(L_star[s["label"]])
            keep = g > 0
            err.append(F.series(s["x"][keep]+1, np.sqrt(g[keep]), label=s["label"], color=s["color"], dash=s["dash"]))
        summary["training_error_network_%d" % idx] = {s["label"]: float(s["y"][-1]) for s in err}
        energy = [F.series(s["x"]+1, s["y"], label=s["label"], color=s["color"], dash=s["dash"]) for s in lines]
        panels = [F.line_panel(energy, "energy (network %d)" % idx, "iteration", "$\\mathcal{L}_{\\mathrm{energy}}$",
                               xmode="log"),
                  F.line_panel(err, "relative error of the corrections", "iteration",
                               "$((\\mathcal{L}-\\mathcal{L}^\\star)/|\\mathcal{L}^\\star|)^{1/2}$", ymode="log", xmode="log",
                               legend=False)]
        if "interpolation condition" in logs and not config["exact_constraint"]:
            ic = logs["interpolation condition"]
            panels.append(F.line_panel([F.series(ic[:, 0]+1, ic[:, 1], label="Adam")], "interpolation condition",
                                       "iteration", "$\\mathcal{L}_{\\mathrm{interp}}$", ymode="log", xmode="log", legend=False))
        F.save_tex(out+"loss", panels, **save)
        entries.append(("loss", "Training of the network %d: energy loss and relative error of the network corrections in the energy norm (Corollary 3.10)" % idx))

    # ---- coefficients (cf. Fig. 9)
    first = R[cases[0]]
    n0 = np.shape(first["A_stiff"])[1]
    panels = [coef_panel(coef(first, i), "%s coefficient %d" % (cases[0], i)) for i in [0, n0//2, n0-1]]
    panels.append(coef_panel(coef(test, N//2), "%s coefficient %d" % (cases[-1], N//2)))
    F.save_tex(out+"coefficients", panels, ncols=4, **save)
    entries.append(("coefficients", "Coefficients of the %s set and of the %s set" % (cases[0], cases[-1])))

    # ---- basis functions (cf. Figs. 4, 10) and cross sections (Fig. 5), test coefficient N//2
    i = N//2
    nodes = representative_nodes(NC)
    rows = [[], [], []]
    cross = []
    for j in nodes:
        L, A = test["G_LOD"][:, j, i], test["G_PINN"][:, j, i]
        ZL, ext, (i0, i1) = crop(F.grid_field(L, FE), L, FE)
        ZA, _, _ = crop(F.grid_field(A, FE), L, FE)
        vmin, vmax = min(ZL.min(), ZA.min()), max(ZL.max(), ZA.max())
        node = FE.vertices_coarse[j]
        rows[0].append(F.field_panel(ZL, "LOD, node %d" % j, vmin=vmin, vmax=vmax, extent=ext, marker=tuple(node)))
        rows[1].append(F.field_panel(ZA, "LODANN, node %d" % j, vmin=vmin, vmax=vmax, extent=ext, marker=tuple(node)))
        rows[2].append(F.field_panel(ZA-ZL, "LODANN $-$ LOD", cmap="div", extent=ext))
        # cross section through the node
        r = np.argmin(np.abs(ys-node[1]))
        x = xs[i0:i1+1]
        hat = np.hstack(FE.basisFunctions())[:, j]
        sec = lambda v: F.grid_field(v, FE)[r, i0:i1+1]
        cross.append(F.line_panel([F.series(x, sec(hat), "nodal basis", label="$\\Lambda_j$"),
                                   F.series(x, sec(A), "LODANN"), F.series(x, sec(L), "LOD")],
                                  "node %d, $y=%.3g$" % (j, node[1]), "$x$", "", legend=(len(cross) == 0),
                                  legend_pos="below"))
    F.save_tex(out+"basis_functions", rows[0]+rows[1]+rows[2], ncols=len(nodes), **save)
    entries.append(("basis_functions", "Corrected basis functions of LOD and LODANN and their difference (%s coefficient %d)" % (cases[-1], i)))
    F.save_tex(out+"basis_cross_sections", cross, ncols=len(nodes), **save)
    entries.append(("basis_cross_sections", "Cross sections of the basis functions through the node"))

    # ---- solutions (cf. Figs. 6, 11)
    rows = [[], [], [], []]
    for k in times:
        Uh = F.grid_field(test["u_FEM"][:, k], FE)
        UL, UA = F.grid_field(test["u_LOD"][:, k], FE), F.grid_field(test["u_PINN"][:, k], FE)
        # common colour scale of the three solutions at this time
        vmin, vmax = min(Uh.min(), UL.min(), UA.min()), max(Uh.max(), UL.max(), UA.max())
        rows[0].append(coef_panel(coef(test, k), "coefficient, "+tl(k)))
        rows[1].append(F.field_panel(Uh, "fine FEM, "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"))
        rows[2].append(F.field_panel(UL, "LOD, "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"))
        rows[3].append(F.field_panel(UA, "LODANN, "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"))
    F.save_tex(out+"solutions", sum(rows, []), ncols=3, **save)
    entries.append(("solutions", "Coefficient, fine FEM (reference), LOD and LODANN solution (%s set)"
                    % cases[-1]))

    # ---- cross sections of the solutions (cf. Fig. 7), all methods
    r = np.argmin(np.abs(ys-0.5*(ys[0]+ys[-1])))
    panels = []
    for n, k in enumerate(times):
        sec = lambda u: F.grid_field(u[:, k], FE)[r, :]
        ser = [F.series(xs, sec(test["u_FEM"]), "fine FEM")]
        for m, key in [("coarse FEM", "u_coarse"), ("frozen LOD", "u_frozen"), ("Deep Ritz", "u_DR"),
                       ("LODANN", "u_PINN"), ("LOD", "u_LOD")]:
            if key in test:
                ser.append(F.series(xs, sec(test[key]), m))
        panels.append(F.line_panel(ser, tl(k)+", $y=%.3g$" % ys[r], "$x$", "$u$", legend=(n == 0),
                                   legend_pos="below"))
    F.save_tex(out+"solution_cross_sections", panels, ncols=3, **save)
    entries.append(("solution_cross_sections", "Cross sections of the solutions of all methods (%s set)" % cases[-1]))

    # ---- pointwise errors (cf. Fig. 8)
    rows = [[], [], []]
    E = {k: (np.abs(F.grid_field(test["u_LOD"][:, k]-test["u_FEM"][:, k], FE)),
             np.abs(F.grid_field(test["u_PINN"][:, k]-test["u_FEM"][:, k], FE)),
             np.abs(F.grid_field(test["u_PINN"][:, k]-test["u_LOD"][:, k], FE))) for k in times}
    for k in times:
        m = max(E[k][0].max(), E[k][1].max())
        rows[0].append(F.field_panel(E[k][0], "$|u_{\\mathrm{LOD}}-u_h|$, "+tl(k), vmin=0, vmax=m))
        rows[1].append(F.field_panel(E[k][1], "$|u_{\\mathrm{LODANN}}-u_h|$, "+tl(k), vmin=0, vmax=m))
        rows[2].append(F.field_panel(E[k][2], "$|u_{\\mathrm{LODANN}}-u_{\\mathrm{LOD}}|$, "+tl(k), vmin=0))
    F.save_tex(out+"pointwise_errors", rows[0]+rows[1]+rows[2], ncols=3, **save)
    entries.append(("pointwise_errors", "Pointwise errors against the fine FEM solution $u_h$ and between LODANN and LOD (%s set)" % cases[-1]))

    # ---- comparison: errors over time
    panels = []
    sers = [[F.series(R[lab]["T"][1:], 100*e, m) for m, e in R[lab]["errors"].items() if m in F.METHODS]
            for lab in cases]
    full = int(np.argmax([len(s) for s in sers]))  # legend in the panel with all methods
    for n, lab in enumerate(cases):
        panels.append(F.line_panel(sers[n], "%s set" % lab, "$t$", "relative error vs.\\ fine FEM [\\%]",
                                   ymode="log", legend=(n == full)))
    F.save_tex(out+"errors_over_time", panels, ncols=len(panels), width_cm=5.0, **save)
    entries.append(("errors_over_time", "Relative error against the fine FEM solution over time"))

    # ---- comparison: accuracy vs online cost (test set)
    pts = [F.series([test["cost"][m]], [100*test["errors"][m][-1]], m, mark="*")
           for m in test["errors"] if m in F.METHODS and m in test["cost"]]
    ev = [100*test["errors"][m][-1] for m in test["errors"] if m in F.METHODS]
    pts.append(F.series([test["cost"]["fine FEM"]]*2, [min(ev)/1.5, max(ev)*1.5], "fine FEM",
                        point_label=False))
    pts.append(F.series([test["cost"]["fine FEM"]], [max(ev)*1.5], "fine FEM",
                        label="fine FEM (reference)"))
    F.save_tex(out+"accuracy_cost", [F.line_panel(
        pts, "%s set, final time" % cases[-1], "online cost per time step [s]",
        "relative error vs.\\ fine FEM [\\%]", xmode="log", ymode="log", legend=False,
        labels_at_points=True)], width_cm=7.0, **save)
    entries.append(("accuracy_cost", "Accuracy against online cost per time step (%s set). Fine FEM (reference): %.3g s per step; LODANN offline training: %.3g h (%s). The classical LOD timings refer to the unoptimized implementation in LOD.py." % (
        cases[-1], test["cost"]["fine FEM"], t_train/3600, how)))

    # ---- classical Deep Ritz baseline
    if "u_DR" in test:
        k = N-1
        Uh, UD, UA = (F.grid_field(test[key][:, k], FE) for key in ["u_FEM", "u_DR", "u_PINN"])
        vmin, vmax = min(Uh.min(), UD.min(), UA.min()), max(Uh.max(), UD.max(), UA.max())
        e1, e2 = np.abs(UD-Uh), np.abs(UA-Uh)
        m = max(e1.max(), e2.max())
        F.save_tex(out+"deep_ritz", [
            F.field_panel(Uh, "fine FEM, "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"),
            F.field_panel(UD, "Deep Ritz (no LOD), "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"),
            F.field_panel(UA, "LODANN, "+tl(k), vmin=vmin, vmax=vmax, cbar_label="$u$"),
            F.field_panel(e1, "$|u_{\\mathrm{DR}}-u_h|$", vmin=0, vmax=m),
            F.field_panel(e2, "$|u_{\\mathrm{LODANN}}-u_h|$", vmin=0, vmax=m)], ncols=3, **save)
        entries.append(("deep_ritz", "Classical Deep Ritz network on the whole domain (trained in every time step) compared with LODANN and the fine FEM solution"))

    # ---- summary
    methods = [m for m in ["coarse FEM", "frozen LOD", "LOD", "LODANN", "Deep Ritz"] if m in test["errors"]]
    with open(out+"summary.json", "w") as f:
        json.dump(summary, f, indent=1, default=float)
    lines = [r"\begin{tabular}{l" + "r"*(2*len(cases)+1) + "}", r"\hline",
             "method & " + " & ".join("%s: final & %s: mean" % (c, c) for c in cases) + r" & cost/step [s]\\", r"\hline"]
    for m in methods:
        vals = []
        for c in cases:
            e = R[c]["errors"].get(m)
            vals += ["%.2f\\,\\%%" % (100*e[-1]), "%.2f\\,\\%%" % (100*np.mean(e))] if e is not None else ["--", "--"]
        lines.append(m + " & " + " & ".join(vals) + " & %.3g\\\\" % test["cost"].get(m, np.nan))
    lines += [r"\hline", "fine FEM (reference) & " + " & ".join(["--"]*2*len(cases)) + " & %.3g\\\\" % test["cost"]["fine FEM"],
              r"\hline"]
    for m in ["LODANN"]:
        if m in test["basis_error"]:
            # mean relative error of the basis functions against LOD (in the "final" column)
            vals = []
            for c in cases:
                vals += ["%.2f\\,\\%%" % R[c]["basis_error"][m], "--"]
            lines.append(m + " basis vs.\\ LOD & " + " & ".join(vals) + r" & \\")
    lines += [r"\hline", r"\end{tabular}", ""]
    with open(out+"summary.tex", "w") as f:
        f.write("\n".join(lines))
    with open(out+"summary_table.tex", "w") as f:
        f.write("\n".join([r"\documentclass[border=4pt]{standalone}", r"\begin{document}",
                           r"\input{summary.tex}", r"\end{document}", ""]))
    if compile_pdf:
        F.compile_tex(out+"summary_table.tex")
    entries.append(("summary_table", "Relative errors against the fine FEM solution (final time and mean over time) and online cost per time step"))
    F.write_overview(out, entries, title="Results: " + summary["example"], compile_pdf=compile_pdf)
    print("figures written to", out)
    print(json.dumps(summary["cases"][cases[-1]]["error_final_vs_fineFEM"], indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
