# Author: Mehdi Elasmi
import numpy as np
import util


class GEO:
    def __init__(self, domain, N_coarse, N_fine_per_element, ell=100, np_dtype=np.float32, elementwise=False):
        self.np_dtype = np_dtype
        self.ell = ell
        self.dim = np.size(N_coarse)
        assert (np.size(domain) == 2*self.dim)
        assert (np.size(N_fine_per_element) == self.dim)
        self.domain = domain
        self.N_coarse = N_coarse
        self.N_fine_per_element = N_fine_per_element
        self.H = (self.domain[:, 1]-self.domain[:, 0])/N_coarse
        self.h = self.H/N_fine_per_element
        self.a = domain[:, 1]
        self.N_fine = N_coarse * N_fine_per_element
        self.N_vertices_fine = self.N_fine + self.dim * [1]
        self.N_vertices_coarse = self.N_coarse + self.dim * [1]
        self.vertices_coarse = util.vertices(self.N_coarse, a=self.a)
        self.vertices_fine = util.vertices(self.N_fine, a=self.a)
        self.Mids = util.vertices(self.N_fine, mids=True, a=self.a)
        self.Mids_coarse = util.vertices(self.N_coarse, mids=True, a=self.a)
        self.nodesID2elem_coarse = util.elems(self.N_coarse)
        self.nodesID2elem_fine = util.elems(self.N_fine)
        self.elementwise = elementwise
        if elementwise:
            self.dofC, self.dofF, self.PatchC, self.PatchF = util.getPatches_elements(
                self.N_coarse, self.N_fine_per_element, ell)
            self.N_v_pp_coarse = [np.size(self.dofC[i])
                                  for i in range(np.prod(N_coarse))]
            self.N_v_pp_fine = [np.size(self.dofF[i])
                                for i in range(np.prod(N_coarse))]
            self.boundaryID_coarse, self.boundaryID_fine = util.getBoundaryPoints(
                self.nodesID2elem_coarse, self.nodesID2elem_fine, self.PatchC, self.PatchF)
            self.Xfine_all = [self.vertices_fine[self.dofF[i], :]
                              for i in range(np.prod(N_coarse))]
            self.patchDomains = [np.vstack([[self.Xfine_all[i][0, :]], [
                                           self.Xfine_all[i][-1, :]]]) for i in range(np.prod(N_coarse))]
        else:
            self.dofC, self.dofF, self.PatchC, self.PatchF = util.getPatches_nodes(
                self.N_coarse, self.N_fine_per_element, ell)
            self.N_v_pp_coarse = [np.size(self.dofC[i])
                                  for i in range(np.prod(N_coarse+1))]
            self.N_v_pp_fine = [np.size(self.dofF[i])
                                for i in range(np.prod(N_coarse+1))]
            self.boundaryID_coarse, self.boundaryID_fine = util.getBoundaryPoints(
                self.nodesID2elem_coarse, self.nodesID2elem_fine, self.PatchC, self.PatchF)
            self.Xfine_all = [self.vertices_fine[self.dofF[i], :]
                              for i in range(np.prod(N_coarse+1))]
            self.patchDomains = [np.vstack([[self.Xfine_all[i][0, :]], [
                                           self.Xfine_all[i][-1, :]]]) for i in range(np.prod(N_coarse+1))]


    def input_transform(self, X, p):
        print("Warning: input_transform is not yet implemented!")
        return X, p



