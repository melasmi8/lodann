# Authors: Mehdi Elasmi, Felix Krumbiegel
import numpy as np
import util
import scipy.sparse as spp


def lodInterpolation(World, *args):
    # standard interpolation
    NC = World.N_coarse
    NR = World.N_fine_per_element
    NF = World.N_fine

    C = util.cg2dg(NF)
    M = util.assembleGlobalMatrices(util.localMass(NF, a=World.a), NF, 'dg')
    P = util.dgProlongation(NC, NR)
    MInv = util.assembleGlobalMatrices(
        util.localMassInv(NC, a=World.a), NC, 'dg')
    E = util.averaging(NC)
    return E @ MInv @ P.T @ M @ C


def lodCorrector(Mesh, A):
    d = Mesh.dim
    NC = Mesh.N_coarse
    NR = Mesh.N_fine_per_element
    NF = Mesh.N_fine
    ell = Mesh.ell

    # interpolation operator
    Pi = lodInterpolation(Mesh)

    S = util.assembleGlobalMatrices(
        util.localStiffness(NF, a=Mesh.a), NF, 'dg', weight=A)

    # maps a cg function onto a dg function
    cg2dg = util.cg2dg(NF)

    # Patches[j, k] != 0 iff element j is in ell-patch around element k
    Patches = util.getPatch(NC, ell)
    # dg Prolongation from coarse nodes to fine nodes
    Prol = util.dgProlongation(NC, NR)
    # dg Projection from coarse elements to fine elements
    Proj = util.dgProjection(NC, NR)

    C = [None] * np.prod(NC)

    def correctorPatch(j):
        # PatchC[k] = 0, if element k is not in Patch of element j
        # PatchC[k] = 1 or 2, if element k is in Patch of element j
        # PatchC[j] = 2
        PatchC = Patches[:, j]
        # dofC are the indices of coarse free nodes within th patch
        # i.e. global boundary is exculded
        dofC = util.getDofs(NC, np.squeeze(PatchC.toarray()) > 0,
                            rm_glob_bound=True)

        # PatchF[k] = 0, if fine element k is not in Patch of coarse element j
        # PatchF[k] = 1 or 2, if fine element k is in Patch of coarse element j
        # PatchF[k] = 2, if fine element k coarse element j
        PatchF = Proj @ PatchC
        # dofF are the indices of fine inner nodes within the patch,
        # i.e. boundary of Patch and global boundary are excluded
        dofF = util.getDofs(NF, np.squeeze(PatchF.toarray()) > 0,
                            rm_loc_bound=True)
        # dofFdg are the indices of dg nodes within the patch
        dofFdg = util.getDofsdg(NF, np.squeeze(PatchF.toarray() > 0))

        # cg stiffness matrix restricted to the dofs within the patch
        Scg = cg2dg[np.ix_(dofFdg, dofF)].T @ S[np.ix_(dofFdg, dofFdg)]\
            @ cg2dg[np.ix_(dofFdg, dofF)]
        # LHS of the corrector problem
        # / S   Pi.T \ / x \  =  / rhs \
        # \ Pi  0    / \ l /  =  \  0  /
        LHS = spp.bmat([[Scg, Pi[np.ix_(dofC, dofF)].T],
                        [Pi[np.ix_(dofC, dofF)], None]], format='csc')

        # fine dg nodes within the coarse element j
        dofEdg = util.getDofsdg(NF, np.squeeze(PatchF.toarray()) > 1)
        rhs = cg2dg[np.ix_(dofEdg, dofF)].T @ S[np.ix_(dofEdg, dofEdg)]\
            @ Prol[np.ix_(dofEdg, (j*2**d)+np.arange(2**d, dtype='int64'))]
        # RHS of the corrector problem
        RHS = spp.bmat([[rhs], [spp.lil_matrix((np.size(dofC), 2**d))]],
                       format='csc')
        x = spp.linalg.spsolve(LHS, RHS)
        corr = spp.lil_matrix((np.prod(NF+1), 2**d))
        corr[dofF, :] = x[:np.size(dofF), :]
        return corr
    for j in range(np.prod(NC)):
        C[j] = correctorPatch(j)
    cg2dg = util.cg2dg(NC)
    P = util.cgProlongation(NC, NR)
    # G = 1 - C
    G = P - spp.hstack(C, format='csr') @ cg2dg
    return G, spp.hstack(C, format='csr') @ cg2dg
