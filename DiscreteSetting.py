# Author: Mehdi Elasmi
import numpy as np
import util
import Geo
from scipy.sparse import block_diag
import scipy.sparse as spp
import scipy.sparse.linalg as sppla
import LOD


class FE_Nodal(Geo.GEO):
    def __init__(self, geo, PDEcoeff, coeff_parameters, test_parameters, withData=False, np_dtype=np.float32):
        super().__init__(geo.domain, geo.N_coarse, geo.N_fine_per_element,
                         geo.ell, geo.np_dtype, geo.elementwise)
        self.geo = geo
        self.dtype = np_dtype
        self.withData = withData
        self.PDEcoeff = PDEcoeff
        self.coeff_parameters = coeff_parameters
        self.test_parameters = test_parameters
        self.N_coeff = np.size(coeff_parameters, 0)
        self.N_test = np.size(test_parameters, 0)

        self.N_v_ppc_coarse = self.N_v_pp_coarse * self.N_coeff
        self.N_v_ppc_fine = self.N_v_pp_fine * self.N_coeff

        self.I = self.interpolator(
            self.N_coarse, self.N_fine_per_element, self.a)

        if self.withData:
            self.TrainData = self.generateData(coeff_parameters)
            self.TestData = self.generateData(test_parameters)

    def generateData(self, coeff_parameters, index="all"):
        if str(index) == "all":
            if self.elementwise:
                C = []
                for _, coeff in enumerate(coeff_parameters):
                    A = self.PDEcoeff(self.Mids, [coeff]).flatten()
                    C_tmp, _ = LOD.lodCorrector(self, A)
                    C.append(C_tmp.toarray())
            else:
                C = []
                for _, coeff in enumerate(coeff_parameters):
                    A = self.PDEcoeff(self.Mids, [coeff]).flatten()
                    _, C_tmp = LOD.lodCorrector(self, A)
                    C.append(C_tmp.toarray())
            return C

    def basisFunctions(self, index_BF="all"):
        if str(index_BF) == "all":
            BF = []
            if self.dim == 1:
                for i in range(self.N_vertices_coarse[0]):
                    BF.append(np.prod(np.maximum(
                        1. - (1/self.H)*np.abs(self.vertices_fine-self.H*i), 0.), axis=1)[:, None])
            elif self.dim == 2:
                for j in range(self.N_vertices_coarse[1]):
                    for i in range(self.N_vertices_coarse[0]):
                        index_BF = np.array([i, j])
                        BF.append(np.prod(np.maximum(
                            1. - np.abs(self.vertices_fine/self.H-index_BF), 0.), axis=1)[:, None])
        else:
            BF = np.prod(np.maximum(
                1. - (1/self.H)*np.abs(self.vertices_fine-self.H*index_BF), 0.), axis=1)[:, None]
        return BF

    def shapeFunctions1D(self, x, H, H_index_ShF):
        # local coordinate in the element [H_index_ShF, H_index_ShF+H], with a tolerance since
        # the end point is not exactly representable (e.g. 0.5-1/3 > 1/6 in floating point)
        xi = (x-H_index_ShF)/H
        supp = (xi <= 1+1E-10) * (xi >= -1E-10)
        xi = np.clip(xi, 0., 1.)
        return np.concatenate(((1-xi)*supp, xi*supp), 1)

    def shapeFunctions(self, index_element="all"):
        if index_element == "all":
            shF = []
            if self.dim == 1:
                for i in range(self.N_coarse[0]):
                    shF.append(self.shapeFunctions1D(
                        self.vertices_fine, self.H, self.H*i))
            elif self.dim == 2:
                for j in range(self.N_coarse[1]):
                    for i in range(self.N_coarse[0]):
                        shFx = np.tile(self.shapeFunctions1D(
                            self.vertices_fine[:, [0]], self.H[0], self.H[0]*i), (1, 2))
                        shFy = np.repeat(self.shapeFunctions1D(
                            self.vertices_fine[:, [1]], self.H[1], self.H[1]*j), 2, 1)
                        shF.append(shFx[:, [0, 1, 3, 2]] * shFy)
        else:
            if self.dim == 1:
                shF = self.shapeFunctions1D(
                    self.vertices_fine, self.H, self.H*index_element)
            elif self.dim == 2:
                if np.size(index_element) == 2:
                    shFx = np.tile(self.shapeFunctions1D(
                        self.vertices_fine[:, [0]], self.H[0], self.H[0]*index_element[0]), (1, 2))
                    shFy = np.repeat(self.shapeFunctions1D(
                        self.vertices_fine[:, [1]], self.H[1], self.H[1]*index_element[1]), 2, 1)
                else:
                    nodesId = self.nodesID2elem_coarse[index_element, :]
                    shFx = np.tile(self.shapeFunctions1D(self.vertices_fine[:, [
                                   0]], self.H[0], self.vertices_coarse[nodesId[0], 0]), (1, 2))
                    shFy = np.repeat(self.shapeFunctions1D(self.vertices_fine[:, [
                                     1]], self.H[1], self.vertices_coarse[nodesId[0], 1]), 2, 1)
                shF = shFx[:, [0, 1, 3, 2]] * shFy
        return shF

    def systemMatrixBlocks(self, coeff_parameters, index_element):
        # patch stiffness matrices, one per coefficient (rows of coeff_parameters)
        S_fine = []
        for _, coeff in enumerate(coeff_parameters):
            A = self.PDEcoeff(self.Mids, coeff).flatten()
            S_fine_tmp = util.assembleGlobalMatrices(util.localStiffness(
                self.N_fine, a=self.a), self.N_fine, 'cg', weight=A)
            S_fine.append(
                S_fine_tmp[np.ix_(self.dofF[index_element], self.dofF[index_element])].tocsr())
        return S_fine

    def systemMatrixDOF(self, coeff_parameters, index_element):
        # block-diagonal over the coefficients, kept sparse
        return block_diag(self.systemMatrixBlocks(coeff_parameters, index_element), format="csr")

    @property
    def n_par(self):
        # number of parameters of the coefficient (vector valued parameters: n_par > 1)
        return getattr(self.PDEcoeff, "n_par", 1)

    def parameterMatrix(self, params):
        # coefficient parameters as a matrix with one row per coefficient
        params = np.asarray(params)
        return np.reshape(params, (np.shape(params)[0], -1))

    def localParameters(self, index):
        """Indices of the parameters on which the correction problem index depends: all for a
        scalar parameter; for coefficients with relevant_parameters (e.g. one parameter per
        region), only the parameters of the regions intersecting the patch."""
        if hasattr(self.PDEcoeff, "relevant_parameters"):
            return self.PDEcoeff.relevant_parameters(self.Mids[self.PatchF[index], :])
        return np.arange(self.n_par)

    def stiffness(self, A):
        # fine stiffness matrix for the coefficient A (values at the fine elements)
        return util.assembleGlobalMatrices(util.localStiffness(
            self.N_fine, a=self.a), self.N_fine, 'cg', weight=A).tocsr()

    def patchInterpolation(self, index):
        """Interpolation constraint of the correction problem index, restricted to the patch
        (columns: fine patch nodes dofF), as used in training.
        element-wise (eq. (7)): quasi-interpolation I_H at the coarse nodes of the patch N^l(K)
            except the global boundary nodes (as in LOD.lodCorrector)
        nodal: modified quasi-interpolation (util.enforce_BC_interp) at the coarse patch nodes"""
        if self.elementwise:
            dofC = np.setdiff1d(
                self.dofC[index], util.boundaryNodes(self.N_coarse))
            return self.I[np.ix_(dofC, self.dofF[index])]
        I = util.enforce_BC_interp(self.I.copy(), self.N_coarse, self.N_fine_per_element,
                                   self.boundaryID_coarse[index])
        return I[np.ix_(self.dofC[index], self.dofF[index])]

    def innerPatchNodes(self, index):
        # mask of the patch nodes that are neither on the patch boundary nor on the global boundary
        Xf = self.vertices_fine[self.dofF[index], :]
        return ~util.on_patch_boundary(Xf, Xf[0, :], Xf[-1, :])

    def patchBasisFunctions(self, index):
        """Coarse functions whose corrections are computed by the problem index (fine nodes x outputs).
        element-wise: the 2^d shape functions of the element K = index, i.e., the nodal basis
            functions restricted to K, ordered as the nodes in nodesID2elem_coarse[index]
        nodal: the nodal basis function Lambda_index"""
        if self.elementwise:
            return self.shapeFunctions(index)
        return self.basisFunctions(np.array([index % (self.N_coarse[0]+1), index//(self.N_coarse[0]+1)]))

    def patchRHS(self, A, index, S=None):
        """Right-hand sides of the correction problem index at the patch nodes (patch nodes x outputs).
        element-wise (eq. (7)): int_K a grad(Lambda) . grad(v) for the shape functions of K
        nodal: int a grad(Lambda_j) . grad(v) over supp(Lambda_j)"""
        if self.elementwise:
            inK = util.dgProjection(self.N_coarse, self.N_fine_per_element) @ \
                (np.arange(np.prod(self.N_coarse)) == index)
            S = self.stiffness(A*inK)
        elif S is None:
            S = self.stiffness(A)
        return (S @ self.patchBasisFunctions(index))[self.dofF[index], :]

    def patchCorrector(self, A, index, S=None):
        """Exact discrete solution of the correction problem index (patch nodes x outputs): the
        minimizer of 0.5 v'Sv - rhs'v over the patch functions that vanish on the patch boundary and
        satisfy the interpolation constraint (patchInterpolation). For element-wise problems this is
        the corrector of eq. (7), i.e., it coincides with LOD.lodCorrector."""
        if S is None:
            S = self.stiffness(A)
        dofF = self.dofF[index]
        inner = np.where(self.innerPatchNodes(index))[0]
        S_loc = S[np.ix_(dofF[inner], dofF[inner])]
        R = self.patchRHS(A, index, S)[inner, :]
        # orthonormal basis of the constraints on the inner dofs (full rank, so the saddle point
        # system is regular and can be solved with a sparse direct solver)
        W = self.constraintBasis(index)[inner, :]
        r = W.shape[1]
        K = spp.bmat([[S_loc, spp.csr_matrix(W)],
                      [spp.csr_matrix(W.T), None]], format="csc")
        rhs = np.vstack([R, np.zeros((r, R.shape[1]))])
        x = sppla.splu(K).solve(rhs)
        V = np.zeros((np.size(dofF), R.shape[1]))
        V[inner, :] = x[:np.size(inner), :]
        return V

    def constraintBasis(self, index):
        """Orthonormal basis W (patch nodes x r) of the constraints of the correction problem index:
        the row space of the patch interpolation restricted to the inner patch nodes (zero rows for
        the other nodes). For v vanishing outside the inner nodes, v - W W'v is the orthogonal
        projection onto the admissible corrections, i.e., it satisfies the interpolation constraint
        exactly and vanishes on the patch boundary."""
        inner = self.innerPatchNodes(index)
        B = self.patchInterpolation(index)[:, inner]
        _, sv, Vt = np.linalg.svd(B, full_matrices=False)
        r = int(np.sum(sv > 1E-10*np.max(sv))) if np.size(sv) > 0 else 0
        W = np.zeros((np.size(inner), r))
        W[inner, :] = Vt[:r, :].T
        return W

    def assembleBasis(self, corrections):
        """Corrected basis functions (fine nodes x coarse nodes) from the corrections of the patch
        problems, given as {index: patch nodes x outputs}: Lambda_j minus the corrections of
        Lambda_j, summed over all elements for element-wise problems (eq. (8)). Coarse nodes
        without correction keep the nodal basis function."""
        G = np.hstack(self.basisFunctions())
        for index, V in corrections.items():
            dofF = self.dofF[index]
            if self.elementwise:
                nodes = self.nodesID2elem_coarse[index, :]
            else:
                nodes = [index]
            for k, j in enumerate(nodes):
                G[dofF, j] -= V[:, k]
        return G

    def assembleBases(self, corrections, n_coeff):
        """As assembleBasis, for n_coeff coefficients at once and sparse: corrections {index: array
        (n_coeff, patch nodes, outputs)}, returns util.Bases (one sparse fine x coarse matrix per
        coefficient). The entries of all corrections are collected once (vectorized over the
        coefficients) and summed by the sparse format."""
        Lam = spp.csc_matrix(np.hstack(self.basisFunctions()))
        rows, cols, vals = [], [], []
        for index, V in corrections.items():
            dofF = self.dofF[index]
            nodes = self.nodesID2elem_coarse[index, :] if self.elementwise else [index]
            for k, j in enumerate(nodes):
                rows.append(dofF)
                cols.append(np.full(np.size(dofF), j))
                vals.append(V[:, :, k])
        rows, cols = np.concatenate(rows), np.concatenate(cols)
        vals = np.concatenate(vals, axis=1)
        return util.Bases([Lam - spp.csc_matrix((vals[i], (rows, cols)), shape=Lam.shape)
                           for i in range(n_coeff)])

    def discreteTarget(self, A, indices="all"):
        """Corrected basis functions from the exact solutions of the correction problems that the
        networks are trained on (indices: problems to solve, all elements resp. nodes by default).
        element-wise: classical LOD of eq. (7)-(8)
        nodal: nodal patch problems (columns of nodes not in indices are nodal basis functions)"""
        if str(indices) == "all":
            indices = range(np.prod(self.N_coarse)) if self.elementwise else range(
                np.prod(self.N_coarse+1))
        S = self.stiffness(A)
        return self.assembleBasis({index: self.patchCorrector(A, index, S) for index in indices})


    def interpolator(self, NC, NR, a):
        NF = NC*NR
        C = util.cg2dg(NF)
        M = util.assembleGlobalMatrices(util.localMass(NF, a=a), NF, 'dg')
        P = util.dgProlongation(NC, NR)
        MInv = util.assembleGlobalMatrices(
            util.localMassInv(NC, a=a), NC, 'dg')
        E = util.averaging(NC)
        I = E @ MInv @ P.T @ M @ C
        return I.toarray()


