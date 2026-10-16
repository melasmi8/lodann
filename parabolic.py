# Author: Mehdi Elasmi
# Shared evaluation of the parabolic examples: LOD vs. LODANN (see examples/*/solve_parabolic.py)
import timeit
import numpy as np
import matplotlib.pyplot as plt
import evaluate_model
import coefficients
import LOD
import solvers
import util
import figures


def rel_basis_error(G, G_ref, nodes):
    # relative errors (in %) of the basis functions of the given coarse nodes, shape (nodes, times)
    G, G_ref = util.as_bases(G), util.as_bases(G_ref)
    out = np.zeros((np.size(nodes), len(G)))
    for i in range(len(G)):
        D = (G.sparse(i) - G_ref.sparse(i))[:, nodes]
        R = G_ref.sparse(i)[:, nodes]
        out[:, i] = np.sqrt(np.asarray(D.multiply(D).sum(0)).ravel()) \
            / np.sqrt(np.asarray(R.multiply(R).sum(0)).ravel())*100
    return out


def rel_error(u, u_ref):
    return np.linalg.norm(u-u_ref)/np.linalg.norm(u_ref)*100


def method_name(loadPath):
    # name of the trained method in the outputs
    return "LODANN"


def run_case(loadPath, evalCoeff, T, rhs, u0, mass_fun=None, label="train", target=True, plotCoeff=True):
    """T: time points (None: equidistant in [0, 1]),
    u0: initial condition, either a vector on the fine mesh or a function of the fine vertices.
    Solves the parabolic problem with the classical LOD and the LODANN basis functions, where
    the coefficient at time T[i] is the one of the parameter evalCoeff[i] (or the training
    parameters for evalCoeff="train_coeff"). With target=True, the basis functions that solve the
    discrete correction problems used in training are computed as well, which separates the
    error of the networks from the difference between the nodal-patch and the element-wise (LOD)
    formulation (for element-wise training both coincide)."""
    start = timeit.default_timer()
    M = method_name(loadPath)
    t_ann = {}
    FE, PDEcoeff, eval_coeff, G_PINN = evaluate_model.evaluateModel(
        loadPath, evalCoeff=evalCoeff, timings=t_ann)
    if callable(u0):
        # initial condition as a function of the fine vertices
        u0 = u0(FE.vertices_fine)
    N_eval = np.shape(eval_coeff)[0]
    if T is None:
        # equidistant time points in [0, 1], one per coefficient
        T = np.linspace(0, 1, N_eval, dtype=np.float32)
    assert np.size(T) == N_eval, "one coefficient per time point required"
    A_stiff = np.zeros((np.size(FE.Mids, 0), N_eval))
    A_mass = np.ones_like(A_stiff)
    G_LOD = []
    G_target = [] if target else None
    free = np.setdiff1d(np.arange(np.prod(FE.N_coarse+1)),
                        util.boundaryNodes(FE.N_coarse))
    print("- Solve problem for all "+label+" time steps: i.e., ", T)
    t_lod_basis = 0.
    for i, coeff in enumerate(eval_coeff):
        if plotCoeff:
            # plot the coefficient at the element midpoints where it is used
            fig, ax = plt.subplots()
            coefficients.plotandSaveCoeff(
                loadPath, PDEcoeff, FE.Mids, [coeff], i, ax, fig, label)
        A = PDEcoeff(FE.Mids, [coeff]).flatten()
        A_stiff[:, i] = A
        if mass_fun is not None:
            A_mass[:, i] = mass_fun(FE.Mids).flatten()
        t0 = timeit.default_timer()
        G_LOD.append(LOD.lodCorrector(FE, A)[0])
        t_lod_basis += timeit.default_timer() - t0
        if target:
            if FE.elementwise:
                # the element-wise problems are exactly the LOD correctors of eq. (7)
                G_target.append(G_LOD[-1])
            else:
                G_target.append(FE.discreteTarget(A, free))
    G_LOD = util.Bases(G_LOD)
    if target:
        G_target = util.Bases(G_target)
    u_LOD, _ = solvers.solveHomParabolic(
        FE, G_LOD, A_stiff, A_mass, rhs, u0, T)
    t0 = timeit.default_timer()
    u_PINN, _ = solvers.solveHomParabolic(
        FE, G_PINN, A_stiff, A_mass, rhs, u0, T)
    t_solve = timeit.default_timer() - t0

    # basis functions of global boundary nodes are not used (Dirichlet condition) -> free nodes only
    err_BF = rel_basis_error(G_PINN, G_LOD, free)
    print("relative error basis functions (free nodes x times): ", err_BF, "%")
    print("mean relative error basis functions (%s vs LOD): %.3f" % (M,
          np.mean(err_BF)), "%")
    err = rel_error(u_PINN, u_LOD)
    print(" - "+label+" error (all times): err = %.3f" % err, "%")
    print(" - "+label+" error at final time: err = %.3f" %
          rel_error(u_PINN[:, -1], u_LOD[:, -1]), "%")
    # times [s]: LOD basis per coefficient (classical), LODANN basis per coefficient (networks
    # and assembly, without loading the models), coarse time step (with a given basis)
    timings = dict(lod_basis=t_lod_basis/N_eval,
                   ann_basis=(t_ann["networks"]+t_ann["assembly"])/N_eval,
                   ann_load=t_ann["load"], coarse_step=t_solve/max(N_eval-1, 1))
    print(" - times per coefficient: LOD basis %.4f s, %s basis %.4f s (loading the networks once %.2f s), coarse time step %.4f s"
          % (timings["lod_basis"], M, timings["ann_basis"], timings["ann_load"], timings["coarse_step"]))
    results = dict(FE=FE, PDEcoeff=PDEcoeff, eval_coeff=eval_coeff, T=T, G_PINN=G_PINN,
                   G_LOD=G_LOD, u_PINN=u_PINN, u_LOD=u_LOD, err=err, free=free,
                   A_stiff=A_stiff, A_mass=A_mass, u0=u0, rhs=rhs, timings=timings, method=M)
    if target:
        u_target, _ = solvers.solveHomParabolic(
            FE, G_target, A_stiff, A_mass, rhs, u0, T)
        print(" - decomposition into network error and formulation gap:")
        print("   mean relative error basis functions, %s vs discrete target: %.3f" % (M,
              np.mean(rel_basis_error(G_PINN, G_target, free))), "%")
        print("   mean relative error basis functions, discrete target vs LOD: %.3f" %
              np.mean(rel_basis_error(G_target, G_LOD, free)), "%")
        print("   error at final time, %s vs discrete target: %.3f" % (M,
              rel_error(u_PINN[:, -1], u_target[:, -1])), "%")
        print("   error at final time, discrete target vs LOD: %.3f" %
              rel_error(u_target[:, -1], u_LOD[:, -1]), "%")
        results.update(G_target=G_target, u_target=u_target)
    stop = timeit.default_timer()
    print("total time: %.2f" % (stop - start), "s")
    return results


def plot_solutions(loadPath, res, label, first=0):
    # quick looks: LOD, LODANN and their difference at the first, half, and final time
    FE, u_PINN, u_LOD, T = res["FE"], res["u_PINN"], res["u_LOD"], res["T"]
    N = np.shape(res["eval_coeff"])[0]
    for idx, name in [(first, "t=0"), (N//2, "t=halfTime"), (N-1, "t=last")]:
        L = figures.grid_field(u_LOD[:, idx], FE)
        A = figures.grid_field(u_PINN[:, idx], FE)
        vmin, vmax = min(L.min(), A.min()), max(L.max(), A.max())
        t = "$t=%.3g$" % T[idx]
        figures.save_png(loadPath+"solution_"+label+"_"+name+".png", [
            figures.field_panel(L, "LOD, "+t, vmin=vmin, vmax=vmax, cbar_label="u"),
            figures.field_panel(A, res["method"]+", "+t, vmin=vmin, vmax=vmax, cbar_label="u"),
            figures.field_panel(A-L, res["method"]+" $-$ LOD", cmap="div")])
