# Author: Mehdi Elasmi
import numpy as np
import util


def solveHomParabolic(Mesh, G_time, A_stiff, A_mass, f, u0, time):
    """Backward Euler with time-dependent (multiscale) basis functions, cf. (10a).
    Step m (from time[m-1] to time[m]) uses the coefficients A_stiff[:, m], A_mass[:, m] and the
    basis G_time[:, :, m]; column 0 belongs to the initial time and is not used for stepping.
    u0 is a function on the fine mesh. The previous solution enters as a fine-scale function
    since the coarse coefficients of different time steps refer to different bases."""
    NC, NF = Mesh.N_coarse, Mesh.N_fine
    boundary_dof = util.boundaryNodes(NC)
    dof = np.setdiff1d(np.arange(np.prod(NC+1)), boundary_dof)
    u_t = np.zeros((np.prod(NF+1), np.size(time)))
    u_t[:, 0:1] = np.reshape(u0, (-1, 1))
    f_fine = f(Mesh.vertices_fine)
    S_loc = util.localStiffness(NF, a=Mesh.a)
    M_loc = util.localMass(NF, a=Mesh.a)
    G_time = util.as_bases(G_time)
    for m in range(1, np.size(time)):
        S_fine = util.assembleGlobalMatrices(
            S_loc, NF, 'cg', weight=A_stiff[:, m].flatten())
        M_fine = util.assembleGlobalMatrices(
            M_loc, NF, 'cg', weight=A_mass[:, m].flatten())
        # sparse basis (each basis function lives on its patch)
        G = G_time.sparse(m)[:, dof]
        dt = time[m] - time[m-1]
        A = (G.T @ ((M_fine/dt + S_fine) @ G)).toarray()
        rhs = G.T @ (M_fine @ (u_t[:, m-1:m]/dt + f_fine))
        u_t[:, m:m+1] = G @ np.linalg.solve(A, rhs)
    return u_t, 0


def solveFineFEMParabolic(Mesh, A_stiff, A_mass, f, u0, time):
    """Backward Euler with Q1 finite elements on the fine mesh, eq. (10b) (reference solution).
    Column m of A_stiff, A_mass (values at the fine elements) is used for the step to time[m]."""
    import scipy.sparse.linalg as sppla
    NF = Mesh.N_fine
    dof = np.setdiff1d(np.arange(np.prod(NF+1)), util.boundaryNodes(NF))
    u_t = np.zeros((np.prod(NF+1), np.size(time)))
    u_t[:, 0:1] = np.reshape(u0, (-1, 1))
    f_fine = f(Mesh.vertices_fine)
    S_loc = util.localStiffness(NF, a=Mesh.a)
    M_loc = util.localMass(NF, a=Mesh.a)
    for m in range(1, np.size(time)):
        dt = time[m] - time[m-1]
        S = util.assembleGlobalMatrices(S_loc, NF, 'cg', weight=A_stiff[:, m].flatten())
        M = util.assembleGlobalMatrices(M_loc, NF, 'cg', weight=A_mass[:, m].flatten())
        K = (M/dt + S).tocsc()[dof, :][:, dof]
        u_t[dof, m] = sppla.spsolve(K, (M @ (u_t[:, m-1:m]/dt + f_fine))[dof, 0])
    return u_t, 0


def solveCoarseFEMParabolic(Mesh, A_stiff, A_mass, f, u0, time):
    """Backward Euler with classical Q1 finite elements on the coarse mesh (same number of unknowns
    as the LOD): Galerkin method in V_H with exact (fine-mesh) integration of the coefficients and
    the right-hand side, i.e. the LOD scheme with the uncorrected nodal basis. Returns the
    solution on the fine mesh (for comparison)."""
    P = util.cgProlongation(Mesh.N_coarse, Mesh.N_fine_per_element)
    return solveHomParabolic(Mesh, util.Bases([P]*np.size(time)), A_stiff, A_mass, f, u0, time)
