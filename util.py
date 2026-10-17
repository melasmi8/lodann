# Authors: Mehdi Elasmi, Felix Krumbiegel
import os
import numpy as np
import scipy.sparse as spp
import tensorflow as tf

# %% assemble system matrices


def localMass(N, a=None):
    d = np.size(N)
    assert d == 1 or d == 2
    if a is None:
        a = np.ones((d,))
    assert d == np.size(a)
    if d == 1:
        M = np.array([[2, 1],
                      [1, 2]])/6
    elif d == 2:
        M = np.array([[4, 2, 1, 2],
                      [2, 4, 2, 1],
                      [1, 2, 4, 2],
                      [2, 1, 2, 4]])/36
    return np.prod(a)/np.prod(N) * M


def localMassInv(N, a=None):
    d = np.size(N)
    assert d == 1 or d == 2
    if a is None:
        a = np.ones((d,))
    assert d == np.size(a)
    if d == 1:
        M = np.array([[2, -1],
                      [-1, 2]])*2
    elif d == 2:
        M = np.array([[4, -2, 1, -2],
                      [-2, 4, -2, 1],
                      [1, -2, 4, -2],
                      [-2, 1, -2, 4]])*4
    return np.prod(N)/np.prod(a) * M


def localStiffness(N, a=None):
    d = np.size(N)
    assert d == 1 or d == 2
    if a is None:
        a = np.ones((d,))
    assert d == np.size(a)
    B = a/N  # linear transform is np.diag(a/N)
    if d == 1:
        S = np.array([[1, -1],
                      [-1, 1]])/B
    elif d == 2:
        # integral of first component of the gradient:
        S = np.array([[2, -2, -1, 1],
                      [-2, 2, 1, -1],
                      [-1, 1, 2, -2],
                      [1, -1, -2, 2]])/6/(B[0])**2
        # integral of second component of the gradient:
        S += np.array([[2, 1, -1, -2],
                       [1, 2, -2, -1],
                       [-1, -2, 2, 1],
                       [-2, -1, 1, 2]])/6/(B[1])**2
        S *= np.prod(B)  # det B
    return S


# assembly maps (element weights -> matrix entries), one per mesh and element matrix
_ASSEMBLY_MAPS = {}


def _assembly_map(loc, N):
    """Map of the cg assembly of the element matrix loc on the mesh with N elements: the matrix
    sum_e w_e loc (at the nodes of element e) has the fixed CSR pattern (indices, indptr) and the
    entries B @ w. Computed once, then each assembly is one sparse matrix-vector product."""
    key = (tuple(np.ravel(N)), np.shape(loc), np.asarray(loc, dtype=float).tobytes())
    if key not in _ASSEMBLY_MAPS:
        T = elems(N)
        n_el, k = T.shape
        n = int(np.prod(np.asarray(N)+1))
        rows = np.repeat(T, k, axis=1).ravel().astype(np.int64)   # (e, i, j) -> T[e, i]
        cols = np.tile(T, (1, k)).ravel().astype(np.int64)        # (e, i, j) -> T[e, j]
        vals = np.tile(np.asarray(loc, dtype=float).ravel(), n_el)
        element = np.repeat(np.arange(n_el), k*k)
        entries, position = np.unique(rows*n + cols, return_inverse=True)  # sorted: CSR order
        indptr = np.searchsorted(entries//n, np.arange(n+1))
        B = spp.csr_matrix((vals, (position, element)), shape=(np.size(entries), n_el))
        _ASSEMBLY_MAPS[key] = (B, (entries % n).astype(np.int32), indptr.astype(np.int32), n)
    return _ASSEMBLY_MAPS[key]


def assembleGlobalMatrices(loc, N, mtype, weight=None):
    assert mtype in ['cg', 'dg']
    if weight is not None:
        assert np.size(weight) == np.prod(N)
    if weight is None:
        weight = np.ones((np.prod(N),))
    if mtype == 'dg':
        return spp.kron(spp.diags(weight, format='csr'), loc, format='csr')
    elif mtype == 'cg':
        # = cg2dg(N).T @ kron(diag(weight), loc) @ cg2dg(N), via the precomputed assembly map
        B, indices, indptr, n = _assembly_map(loc, N)
        return spp.csr_matrix((B @ np.ravel(weight), indices, indptr), shape=(n, n))


# %% boundary Nodes
def boundaryNodes(N):
    d = np.size(N)
    assert d == 1 or d == 2
    if d == 1:
        return np.array([0, N[0]], dtype='int64')
    elif d == 2:
        x = np.arange(N[0]+1, dtype='int64')
        y = (N[0]+1) * np.arange(N[1]+1, dtype='int64')
        return np.unique(np.concatenate((x, y, y+N[0], x+(N[0]+1)*N[1]),
                                        axis=0))


# %% continuous to discontinuous map
def cg2dg(N):
    T = elems(N)
    return spp.csr_matrix((np.ones((np.size(T),)), (np.arange(np.size(T)),
                                                    T.flatten())))


def elems(N, NC=None):
    d = np.size(N)
    assert d == 1 or d == 2
    if NC is not None:
        assert np.size(NC) == np.size(N)
    if NC is None:
        NC = np.ones_like(N, dtype='int64')
    NF = N * NC
    L = np.arange(N[0], dtype='int64')
    if d == 1:
        return np.array([L, L+1]).T
    elif d == 2:
        L2 = (NF[0]+1) * np.arange(N[1], dtype='int64').reshape((-1, 1))
        L = (L + L2).flatten()  # bottom left corners
        return np.array([L, L+1, L+NF[0]+2, L+NF[0]+1]).T


def vertices(N, mids=False, a=None):
    d = np.size(N)
    if a is None:
        a = np.ones((d,))
    assert d == np.size(a)
    x = np.linspace(0, 1, N[0]+1)
    if d == 1:
        if mids:
            return a[0] * np.array([x[:-1]+1/(2*N[0])]).T
        elif not mids:
            return a[0] * np.array([x]).T
    elif d == 2:
        y = np.linspace(0, 1, N[1]+1)
        if mids:
            x = a[0] * np.tile(x[:-1]+1/(2*N[0]), (N[1],))
            y = a[1] * np.tile(y[:-1]+1/(2*N[1]), (N[0], 1)).flatten('F')
            return np.array([x, y]).T
        elif not mids:
            x = a[0] * np.tile(x, (N[1]+1,))
            y = a[1] * np.tile(y, (N[0]+1, 1)).flatten('F')
            return np.array([x, y]).T




# %% discontinuous prolongation
def localProlongation(NR):
    d = np.size(NR)
    D = np.zeros((2**d*np.prod(NR), 2**d))
    L = np.array([np.arange(NR[0]),
                  np.arange(1, NR[0]+1)])/np.prod(NR)
    J = np.kron(np.ones(2**d, dtype='int64'),
                np.arange(np.prod(2*NR), dtype='int64'))
    K = np.kron(np.arange(2**d, dtype='int64'),
                np.ones(np.prod(2*NR), dtype='int64'))
    if d == 1:
        L = np.concatenate((L[::-1, ::-1].flatten('F'), L.flatten('F')),
                           axis=0)
        D[J, K] = L
        return D
    elif d == 2:
        L2 = np.array([np.arange(NR[1]),
                       np.arange(1, NR[1]+1)])
        L = np.concatenate((np.concatenate((np.kron(L2[1, ::-1],
                                                    L[::-1, ::-1]),
                                            np.kron(L2[0, ::-1], L[:, ::-1])),
                                           axis=0).flatten('F'),
                            np.concatenate((np.kron(L2[1, ::-1], L[:, :]),
                                            np.kron(L2[0, ::-1], L[::-1, :])),
                                               axis=0).flatten('F'),
                            np.concatenate((np.kron(L2[0, :], L[:, :]),
                                            np.kron(L2[1, :], L[::-1, :])),
                                           axis=0).flatten('F'),
                            np.concatenate((np.kron(L2[0, :], L[::-1, ::-1]),
                                            np.kron(L2[1, :], L[:, ::-1])),
                                           axis=0).flatten('F')), axis=0)
        D[J, K] = L
        return D


def dgProlongation(NC, NR):
    d = np.size(NC)
    assert d == 1 or d == 2
    NF = NC * NR
    D = np.tile(localProlongation(NR), (1, np.prod(NC)))
    K = np.tile(np.arange(np.prod(2*NC), dtype='int64'), (D.shape[0], 1))
    J = np.tile(np.arange(NR[0]*2**d, dtype='int64'), (2**d, 1)).T
    J = np.tile(J, (1, NC[0])) + np.kron(NR[0]*2**d*np.arange(NC[0]),
                                         np.ones_like(J, dtype='int64'))
    if d == 1:
        return spp.csr_matrix((D.flatten(), (J.flatten(), K.flatten())))
    elif d == 2:
        J = np.tile(J, (NR[1], 1))\
            + np.kron(NF[0]*2**d*np.arange(NR[1]).reshape((-1, 1)),
                      np.ones_like(J))
        J = np.tile(J, (1, NC[1])) + np.kron(NF[0]*NR[1]*2**d*np.arange(NC[1]),
                                             np.ones_like(J))
        return spp.csr_matrix((D.flatten(), (J.flatten(), K.flatten())))


def cgProlongation(NC, NR):
    d = np.size(NC)
    assert d == 1 or d == 2
    L = np.arange(NR[0]+1)/NR[0]
    L2 = np.tile(np.concatenate((L, L[-2::-1]), axis=0), (NC[0]-1))
    if d == 1:
        L = np.concatenate((L, L[-2::-1]))
        D = np.tile(L, (NC[0]+1,))
        K = np.kron(np.arange(NC[0]+1), np.ones_like(L))
        J = np.arange(-NR[0], NR[0]+1)
        J2 = NR[0] * np.arange(NC[0]+1).reshape(-1, 1)
        J = (J + J2).flatten()
        return spp.csr_matrix((D[NR[0]:-NR[0]],
                               (J[NR[0]:-NR[0]].astype('int'),
                                K[NR[0]:-NR[0]].astype('int'))))
    if d == 2:
        L2 = np.arange(NR[1]+1)/NR[1]
        L = np.concatenate((L, L[-2::-1]), axis=0).reshape(-1, 1)
        L2 = np.concatenate((L2, L2[-2::-1]), axis=0)
        L = np.kron(L, L2)
        D = np.tile(L, (NC[0]+1, NC[1]+1))
        K = np.kron(np.arange(np.prod(NC+1)).reshape(NC+1, order='F'),
                    np.ones_like(L)).astype('int')
        J = np.arange(-NR[0], NR[0]+1).reshape(-1, 1)
        J2 = (NC[0]*NR[0]+1) * np.arange(-NR[1], NR[1]+1)
        J = J + J2
        J2 = NR[0] * np.arange(NC[0]+1).reshape(-1, 1)
        J3 = (NC[0]*NR[0]+1)*NR[1] * np.arange(NC[1]+1)
        J2 = np.kron(J2 + J3, np.ones_like(J))
        J = (np.tile(J, NC+1) + J2).astype('int')
        return spp.csr_matrix((D[NR[0]:-NR[0], NR[1]:-NR[1]].flatten(),
                               (J[NR[0]:-NR[0], NR[1]:-NR[1]].flatten(),
                                K[NR[0]:-NR[0], NR[1]:-NR[1]].flatten())))


# %% projection
# projects coarse elements to fine elements
def dgProjection(NC, NR):
    d = np.size(NC)
    assert d == 1 or d == 2
    NF = NC * NR
    J = np.arange(np.prod(NF), dtype='int64')
    K = np.kron(np.arange(NC[0], dtype='int64'),
                np.ones((NR[0],), dtype='int64'))
    if d == 1:
        return spp.csr_matrix((np.ones_like(J), (J, K)))
    elif d == 2:
        K2 = np.kron(np.arange(NC[1], dtype='int64'),
                     np.ones((NR[1],), dtype='int64')).reshape((-1, 1))
        K = (K + NC[0]*K2).flatten()
        return spp.csr_matrix((np.ones_like(J), (J, K)))


# %% averaging operator continuous to discontinuous
def averaging(N):
    C = cg2dg(N)
    return spp.diags(1/np.asarray(C.sum(axis=0)).ravel(), 0, format='csr') @ C.T


# %% patch matrix
def getPatch(NC, ell):
    d = np.size(NC)
    J = elems(NC).flatten('F')
    K = np.tile(np.arange(np.prod(NC)), (2**d,))
    NpE = spp.csr_matrix((np.ones_like(K), (J.flatten(), K)))
    N1Patch = NpE.T @ NpE
    Patch = N1Patch.copy()
    for _ in range(1, ell):
        Patch = N1Patch @ Patch
    Patch.data[:] = 1
    Patch += spp.eye(np.prod(NC), format='csr')
    return Patch


# %% get nodes from elements
def getNodes(N, patch=None, getNeighbours=False):
    T = elems(N)
    if patch is None:
        return np.unique(T, return_counts=getNeighbours)
    if patch is not None:
        return np.unique(T[patch, :], return_counts=getNeighbours)


def getDofs(N, Patch, rm_glob_bound=False, rm_loc_bound=False):
    if rm_loc_bound:
        boundary = boundaryNodes(N)
        _, globalNgh = getNodes(N, getNeighbours=True)
        globalNgh[boundary] = 0
        nodes, localNgh = getNodes(N, patch=Patch, getNeighbours=True)
        return np.setdiff1d(nodes[globalNgh[nodes] == localNgh], boundary)
    nodes = getNodes(N, patch=Patch)
    if rm_glob_bound:
        boundary = boundaryNodes(N)
        return np.setdiff1d(nodes, boundary)
    else:
        return nodes


def getDofsdg(N, Patch):
    d = np.size(N)
    nodes = np.arange(2**d*np.prod(N), dtype='int64').reshape((-1, 2**d))
    return nodes[Patch, :].flatten()




def getNeighboors_nodes(NC, suppInd):
    nodesID2elem_coarse = elems(NC)
    dim = np.size(NC)
    neighboors = np.zeros([2**dim*np.size(suppInd), np.prod(NC)])
    for i in range(2**dim):
        for j in range(np.size(suppInd)):
            neighboors[i+2**dim*j, :] = np.any(
                nodesID2elem_coarse == nodesID2elem_coarse[suppInd[j], :][i], axis=1)
    return neighboors


def getPatches_nodes(NC, NR, ell, index_BF="all"):
    dim = np.size(NC)
    NF = NC*NR
    nodesID2elem_coarse = elems(NC)
    proj = dgProjection(NC, NR)
    if index_BF == "all":
        PatchC, PatchF, dofC, dofF = [], [], [], []
        for i in range(np.prod(NC+1)):
            suppNodalBF = np.any(nodesID2elem_coarse == i, axis=1)
            Patches = np.where(suppNodalBF)[0]
            for j in range(ell):
                neighboors = getNeighboors_nodes(NC, Patches)
                Patches = np.where(np.sum(neighboors, 0))[0]
            PatchC_tmp = suppNodalBF.copy()
            PatchC_tmp[Patches] = [True]*np.size(Patches)
            dofC_tmp = getDofs(NC, PatchC_tmp > 0, rm_glob_bound=False)
            PatchF_tmp = proj @ PatchC_tmp
            dofF_tmp = getDofs(NF, PatchF_tmp > 0, rm_loc_bound=False)
            PatchC.append(np.where(PatchC_tmp)[0])
            PatchF.append(np.where(PatchF_tmp)[0])
            dofC.append(dofC_tmp)
            dofF.append(dofF_tmp)
    else:
        if dim == 2:
            if np.size(index_BF) == 2:
                index_BF = index_BF[1] * (NC[0]+1) + index_BF[0]
        suppNodalBF = np.any(nodesID2elem_coarse == index_BF, axis=1)
        Patches = np.where(suppNodalBF)[0]
        for i in range(ell):
            neighboors = getNeighboors_nodes(NC, Patches)
            Patches = np.where(np.sum(neighboors, 0))[0]
        PatchC = suppNodalBF.copy()
        PatchC[Patches] = [True]*np.size(Patches)
        dofC = getDofs(NC, PatchC > 0, rm_glob_bound=False)
        proj = dgProjection(NC, NR)
        PatchF = proj @ PatchC
        dofF = getDofs(NF, PatchF > 0, rm_loc_bound=False)
        PatchC = np.where(PatchC)[0]
        PatchF = np.where(PatchF)[0]
    return dofC, dofF, PatchC, PatchF


def getPatches_elements(NC, NR, ell, index_element="all"):
    NF = NC*NR
    dim = np.size(NC)
    Patches = getPatch(NC, ell)
    Proj = dgProjection(NC, NR)
    if index_element == "all":
        PatchC, PatchF, dofC, dofF = [], [], [], []
        for i in range(np.prod(NC)):
            PatchC_tmp = Patches[:, i]
            dofC_tmp = getDofs(NC, np.squeeze(PatchC_tmp.toarray()) > 0,
                               rm_glob_bound=False)
            PatchF_tmp = Proj @ PatchC_tmp
            dofF_tmp = getDofs(NF, np.squeeze(PatchF_tmp.toarray()) > 0,
                               rm_loc_bound=False)
            # print(PatchC_tmp.toarray())
            PatchC.append(np.where(PatchC_tmp.toarray())[0])
            PatchF.append(np.where(PatchF_tmp.toarray())[0])
            dofC.append(dofC_tmp)
            dofF.append(dofF_tmp)
    else:
        if dim == 2:
            if np.size(index_element) == 2:
                index_element = index_element[1] * NC[0] + index_element[0]
        PatchC = Patches[:, index_element]
        dofC = getDofs(NC, np.squeeze(PatchC.toarray()) > 0,
                       rm_glob_bound=False)
        PatchF = Proj @ PatchC
        dofF = getDofs(NF, np.squeeze(PatchF.toarray()) > 0,
                       rm_loc_bound=False)
        PatchC = np.where(PatchC.toarray())[0]
        PatchF = np.where(PatchF.toarray())[0]
    return dofC, dofF, PatchC, PatchF


def getBoundaryPoints(nodesID2elem_coarse, nodesID2elem_fine, PatchC, PatchF):
    dim_inter = int(np.size(nodesID2elem_coarse, 1)/2)
    boundaryPoints_coarse, boundaryPoints_fine = [], []
    for i in range(len(PatchC)):
        bp, cp = np.unique(
            nodesID2elem_coarse[PatchC[i], :].flatten(), return_counts=True)
        boundaryPoints_coarse.append(bp[np.where(cp <= dim_inter)[0]])
        bpf, cpf = np.unique(
            nodesID2elem_fine[PatchF[i], :].flatten(), return_counts=True)
        boundaryPoints_fine.append(bpf[np.where(cpf <= dim_inter)[0]])
    return boundaryPoints_coarse, boundaryPoints_fine


def enforce_BC_interp(I, NC, NR, boundaryID_coarse):
    # note: modifies I in place, pass a copy to keep the original operator
    dim = np.size(NC)
    I[boundaryID_coarse, :] = 0.0
    # fine nodes that coincide with coarse nodes
    k1 = np.kron(np.arange(0, NC[0]+1), (NR[0]))
    if dim == 2:
        interFineCoarse = (np.tile(
            k1, (NC[1]+1, 1))+(NC[0]*NR[0]+1)*np.arange(0, NC[1]+1)[:, None]).flatten()
    elif dim == 1:
        interFineCoarse = k1
    I[:, interFineCoarse] = 0.0
    I[np.ix_(boundaryID_coarse, interFineCoarse)] = np.ones(
        (np.size(boundaryID_coarse), np.size(interFineCoarse)))
    return I


def on_patch_boundary(X, Xa, Xl):
    # True for the nodes X lying on the boundary of the rectangular patch [Xa, Xl]
    tol = 1E-9*np.max(Xl-Xa)
    return np.any((X <= Xa+tol) | (X >= Xl-tol), axis=1)


def localization(X, Xa, Xl, sigma=0.001):
    # localization function of the network output on the patch [Xa, Xl] evaluated at the nodes X:
    # smooth step that is ~1 inside the patch and exactly 0 on the patch boundary
    inp = (X-Xa)/(Xl-Xa)
    locFun = np.prod(2*np.arctan(np.sin(np.pi * inp)/sigma)/np.pi, axis=1)[:, None]
    locFun[on_patch_boundary(X, Xa, Xl)] = 0.0
    return locFun


def to_sparse_tensor(S, dtype):
    # converts a scipy sparse matrix into a (reordered) tf.SparseTensor
    S = spp.coo_matrix(S)
    S.eliminate_zeros()
    indices = np.stack([S.row, S.col], axis=1).astype(np.int64)
    return tf.sparse.reorder(tf.SparseTensor(indices, tf.cast(S.data, dtype), S.shape))


class Bases:
    """Bases for several coefficients (time points): one sparse matrix (fine nodes x coarse nodes) per
    coefficient. G[:, j, i] is column j of basis i (dense vector), G[:, :, i] basis i as a dense
    matrix (only for small meshes / plots), G.sparse(i) basis i as a sparse matrix."""

    def __init__(self, matrices):
        self.mats = [spp.csc_matrix(m) for m in matrices]

    def __len__(self):
        return len(self.mats)

    @property
    def shape(self):
        return self.mats[0].shape + (len(self.mats),)

    def sparse(self, i):
        return self.mats[i]

    def __getitem__(self, key):
        rows, cols, i = key
        assert rows == slice(None), "Bases: only [:, columns, coefficient]"
        M = self.mats[i][:, cols]
        return M.toarray().ravel() if np.ndim(cols) == 0 and not isinstance(cols, slice) else M.toarray()


def as_bases(G):
    # Bases from a dense array (fine nodes x coarse nodes x coefficients) or Bases
    return G if isinstance(G, Bases) else Bases([G[:, :, i] for i in range(np.shape(G)[2])])








def save_h5(model, path):
    """model.save(path) in the HDF5 format of the existing results, without the warnings of Keras
    2.14 (HDF5 is "legacy"; "Compiled the loaded model ..." is printed although nothing is loaded)."""
    import warnings
    logger = tf.get_logger()
    level = logger.level
    logger.setLevel("ERROR")
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*HDF5 file.*")
            model.save(path, include_optimizer=False)
    finally:
        logger.setLevel(level)


def load_h5(path):
    # models are only evaluated or trained with own optimizers: no compilation (and no warning)
    return tf.keras.models.load_model(path, compile=False)


def loadModel(loadPath, index):
    model = load_h5(loadPath+"correctedHatModel"+str(index)+".h5")
    # Define a dictionary mapping the names of custom objects to their classes
    # custom_objects = {'myDense': myDenseLayer.myDense}

    # # # Load the model with custom_objects parameter
    # model = tf.keras.models.load_model(loadPath+"correctedHatModel"+str(index)+".h5", custom_objects=custom_objects)
    # self-adaptive weights of the penalty formulation (not needed with the exact constraint)
    files = [loadPath+name+str(index)+".txt" for name in ["sa_weights_", "sa_weightsData_"]]
    sa_weights, sa_weightsData = [np.loadtxt(f, dtype=np.float32) if os.path.exists(f) else None
                                  for f in files]
    return model, sa_weights, sa_weightsData
