from .CorrelationFunction import *
from .FitFunc import *
import numpy as np
import scipy.optimize as opt
from scipy.linalg import eigh, eig
from .JackknifeDistribution import *
from .RawDataDistribution import RawDataDistribution
from numpy import log
from itertools import product
from numpy.linalg import norm
from .DoubleJackknifeDistribution import DoubleJackknifeDistribution
from .BlockDoubleJackknifeDistribution import BlockDoubleJackknifeDistribution
import matplotlib.pyplot as plt
import math
from numba import njit
from numba.experimental import jitclass


class GEVP:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        #self.previous_eigen = np.array([])
        self.init_idx = np.array([])
        self.custom_sort = False


    #static method 

    def CorrelationMatrix(self,data,t0,t):
        """
        Organizes preprocessed (resampled) data
        into correlation matrix of size (num_ops by num_ops)

        returns: Correlation matrix for each jackknife sample at t and t0
        """
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        #Cmat = np.zeros((self.num_ops,self.num_ops))
        #Ct = np.zeros((self.num_ops,self.num_ops))
        C_mats = np.zeros((self.size,self.current_n,self.current_n))
        C_t = np.zeros((self.size,self.current_n,self.current_n))
        
        for k in range(self.size):
            for i in range(np.shape(data)[0]):
                for j in range(np.shape(data)[0]):
                    if i<=j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]
            #C_mats[k] = Cmat
            #C_t[k] = Ct

        return C_t, C_mats
    
    def sortVector(self, V_init, V_current):

        sorted_vecs = np.empty(len(V_init))

        for i, v_i in enumerate(V_init):
            overlaps = [np.dot(v_i/norm(v_i), v/norm(v)) for v in V_current]
            max_index = np.argmax(overlaps)
            #print(max_index)
            sorted_vecs[i] = max_index

        return np.argsort(sorted_vecs)
    
    def sortVector1(self, V_init, V_current, idx):

        sorted_vecs = np.empty(len(V_current))
        if len(V_init)  == 0:
            for i, v_i in enumerate(V_init):
                overlaps = [np.dot(v_i/norm(v_i), v/norm(v)) for v in V_current]
                #print out overlaps
                max_index = np.argmax(overlaps)
                sorted_vecs[i] = max_index
        else:
            sorted_vecs = idx

        return np.argsort(sorted_vecs)


    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value
            #print(idx)

            #sort both eigvals and eigvec
            if self.custom_sort == True:
                idx2 = self.sortVector(self.init_idx, eigvecs)
            else:
                idx2 = idx

            eigen_values[i] = eigvals[idx2]
            eigen_vectors[i] = eigvecs[:,idx2]

        #print(type(eigen_vectors))
        
        return np.real(eigen_values), np.real(eigen_vectors).mean(axis=0)
    
    def GEVP_init(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value

            if np.any(np.imag(eigvals[idx])) == 0:
                eigen_values[i] = np.real(eigvals[idx])
                eigen_vectors[i] = np.real(eigvecs[:,idx])
            else:
                pass
        
        return np.real(eigen_vectors[2])
    
    def Rebasing(self, C_t0, C_t, eigen_vec):
        
        self.current_n -= 1
        #self.num_ops = self.current_n
        #V = eigen_vec
        Cor_t0_dist_rebased = np.zeros((self.size,self.current_n,self.current_n))
        Cor_t_dist_rebased = np.zeros((self.size,self.current_n,self.current_n))
        for i, (Cor_t0, Cor_t) in enumerate(zip(C_t0, C_t)):
            Cor_t_dist_rebased[i] = eigen_vec[:, :self.current_n].conj().T @ Cor_t @ eigen_vec[:, :self.current_n]
            Cor_t0_dist_rebased[i]= eigen_vec[:, :self.current_n].conj().T @ Cor_t0 @ eigen_vec[:, :self.current_n]

            #print(eigvec[:, :self.current_n].T) #scale such that at t=0 Cor diag on the order of 1

       

        return Cor_t0_dist_rebased, Cor_t_dist_rebased
    
    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]  #Unsure of class for this

        for i in range(self.current_n):
            for j in range(self.size):
                #Replace with nans and add catch logic
                if np.real(eig_val_t_t0[j][i]) != 0 and not np.isnan(np.real(eig_val_t_t0[j][i])) and not np.isinf(np.real(eig_val_t_t0[j][i])):
                        val = (np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))
                        if np.imag(val) == 0:
                            if val > 0:
                                eff_result[i][j] = -(np.real(log(val))) #-log(A/B)
                else:
                    pass
                #eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i])))
 

        return eff_result 
    
    def pre_run(self, t0, t, t0_1, t_1):
        Dt = t - t0
        self.init_idx = self.GEVP_init(*self.CorrelationMatrix(self.data, 0,Dt))
        #print(self.init_idx)
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)
        C3, C4 = self.CorrelationMatrix(self.data, t0_1, t_1)

        eigvals, eigvecs = self.GEVP(C3, C4)
            
        C1, C2 = self.Rebasing(C1, C2, eigvecs)

            
            
        
        # Return final eigenvalues from last GEVP
        #self.current_n = self.num_ops
        return self.GEVP(C1, C2)[0]
    
    def run(self, t0, t, t0_1, t_1):
        self.custom_sort = sorted
        return self.eff_energy(self.pre_run(t0, t, t0_1, t_1),self.pre_run(t0, t+1, t0_1, t_1))
        #return self.pre_run(N_times, t0, t)

# class GEVP_OG:


#     def __init__(self, data):
#         self.data = data
#         self.num_ops = int(np.shape(data)[0])
#         self.current_n = self.num_ops
#         self.size = int(data[0][0].value(0).size())
#         self.init_idx = np.array([])
#         self.custom_sort = False

#     def CorrelationMatrix(self, data, t0, t):
#         """
#         Build jackknife-resolved correlation matrices C(t0) and C(t).

#         Returns
#         -------
#         C_t0 : ndarray
#             Shape (n_samples, n_ops, n_ops).
#         C_t : ndarray
#             Shape (n_samples, n_ops, n_ops).
#         """
#         n_ops = int(np.shape(data)[0])

#         self.num_ops = n_ops
#         self.current_n = n_ops

#         C_t0 = np.zeros((self.size, n_ops, n_ops), dtype=float)
#         C_t = np.zeros((self.size, n_ops, n_ops), dtype=float)

#         for k in range(self.size):
#             for i in range(n_ops):
#                 for j in range(n_ops):
#                     # Use upper triangle if only i <= j is explicitly stored.
#                     a, b = (i, j) if i <= j else (j, i)

#                     C_t0[k, i, j] = data[a][b].value(t0)[k]
#                     C_t[k, i, j] = data[a][b].value(t)[k]

#             # Defensive symmetrization against tiny numerical asymmetries.
#             #C_t0[k] = 0.5 * (C_t0[k] + C_t0[k].T)
#             #C_t[k] = 0.5 * (C_t[k] + C_t[k].T)

#         return C_t0, C_t

#     def sortVector(self, V_init, V_current):

#         V_init = np.asarray(V_init)
#         V_current = np.asarray(V_current)

#         n_vecs = V_current.shape[1]
#         idx = []
#         used = set()

#         for i in range(n_vecs):
#             v_ref = V_init[:, i]
#             norm_ref = np.linalg.norm(v_ref)

#             overlaps = np.full(n_vecs, -np.inf, dtype=float)

#             for j in range(n_vecs):
#                 if j in used:
#                     continue

#                 v_cur = V_current[:, j]
#                 norm_cur = np.linalg.norm(v_cur)

#                 if norm_ref == 0.0 or norm_cur == 0.0:
#                     continue

#                 # Absolute value handles arbitrary eigenvector sign flips.
#                 overlaps[j] = abs(np.vdot(v_ref, v_cur)) / (norm_ref * norm_cur)

#             best_j = int(np.argmax(overlaps))
#             idx.append(best_j)
#             used.add(best_j)

#         return np.array(idx, dtype=int)

#     def sortVector1(self, V_init, V_current, idx):
#         """
#         Kept for compatibility with your original class.

#         If idx is supplied, return it. Otherwise sort by overlap.
#         """
#         if idx is not None and len(idx) != 0:
#             return np.asarray(idx, dtype=int)

#         if V_init is None or len(V_init) == 0:
#             return np.arange(V_current.shape[1], dtype=int)

#         return self.sortVector(V_init, V_current)

#     def GEVP(self, C_t0, C_t):
#         """
#         Solve the generalized eigenvalue problem for each jackknife sample:

#             C(t) v = lambda C(t0) v.

#         Returns
#         -------
#         eigen_values : ndarray
#             Shape (n_samples, current_n).
#         eigen_vectors : ndarray
#             Shape (n_samples, current_n, current_n).
#             Eigenvectors are stored as columns in eigen_vectors[k].
#         """
#         n_samples = C_t0.shape[0]
#         n = C_t0.shape[1]
#         self.current_n = n

#         eigen_values = np.zeros((n_samples, n), dtype=float)
#         eigen_vectors = np.zeros((n_samples, n, n), dtype=float)

#         for k, (Cor_t0, Cor_t) in enumerate(zip(C_t0, C_t)):
#             # Defensive symmetrization.
#             Cor_t0 = 0.5 * (Cor_t0 + Cor_t0.T)
#             Cor_t = 0.5 * (Cor_t + Cor_t.T)

#             try:
#                 eigvals, eigvecs = eigh(Cor_t, Cor_t0)
#             except np.linalg.LinAlgError:
#                 # If C(t0) is not positive definite, return NaNs for this sample.
#                 eigen_values[k, :] = np.nan
#                 eigen_vectors[k, :, :] = np.nan
#                 continue

#             # Default ordering:
#             # lambda_n ~ exp[-E_n (t - t0)], so largest lambda is lowest energy.
#             idx = np.argsort(eigvals)[::-1]

#             # Optional overlap-based sorting using reference eigenvectors.
#             if self.custom_sort and self.init_idx is not None and len(self.init_idx) != 0:
#                 idx = self.sortVector(self.init_idx, eigvecs)

#             eigen_values[k, :] = np.real(eigvals[idx])
#             eigen_vectors[k, :, :] = np.real(eigvecs[:, idx])

#         return eigen_values, eigen_vectors

#     def GEVP_init(self, C_t0, C_t):
#         """
#         Compute reference eigenvectors for overlap sorting.

#         This replaces your original hard-coded eigen_vectors[2] choice.
#         It uses sample 0 by default, which keeps the same general behavior but
#         avoids choosing an arbitrary jackknife sample 2.

#         If your jackknife object has a true central sample, you can modify this
#         method to use that instead.
#         """
#         eigen_values, eigen_vectors = self.GEVP(C_t0, C_t)

#         # Use first sample as reference.
#         return eigen_vectors[0]

#     # def Rebasing(self, C_t0, C_t, eigen_vec):
       
#     #     n_samples = C_t0.shape[0]
#     #     n = C_t0.shape[1]
#     #     n_keep = n - 1

#     #     self.current_n = n_keep

#     #     Cor_t0_dist_rebased = np.zeros((n_samples, n_keep, n_keep), dtype=float)
#     #     Cor_t_dist_rebased = np.zeros((n_samples, n_keep, n_keep), dtype=float)

#     #     for k, (Cor_t0, Cor_t, eigvec) in enumerate(zip(C_t0, C_t, eigen_vec)):
#     #         V = eigvec[:, :n_keep]

#     #         Cor_t0_dist_rebased[k] = V.T @ Cor_t0 @ V
#     #         Cor_t_dist_rebased[k] = V.T @ Cor_t @ V

#     #         # Defensive symmetrization.
#     #         Cor_t0_dist_rebased[k] = 0.5 * (
#     #             Cor_t0_dist_rebased[k] + Cor_t0_dist_rebased[k].T
#     #         )
#     #         Cor_t_dist_rebased[k] = 0.5 * (
#     #             Cor_t_dist_rebased[k] + Cor_t_dist_rebased[k].T
#     #         )

#     #     return Cor_t0_dist_rebased, Cor_t_dist_rebased

#     def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
#         """
#         Compute jackknife effective energies from generalized eigenvalues.

#         Keeps your original output style:
#             a list of JackknifeDistribution objects, one per state.

#         Requires JackknifeDistribution to be defined in your environment.
#         """
#         n_states = eig_val_t_t0.shape[1]
#         self.current_n = n_states

#         eff_result = [JackknifeDistribution(self.size) for _ in range(n_states)]

#         for state in range(n_states):
#             for sample in range(self.size):
#                 lam_t = np.real(eig_val_t_t0[sample, state])
#                 lam_t1 = np.real(eig_val_t1_t0[sample, state])

#                 if np.isnan(lam_t) or np.isnan(lam_t1):
#                     eff_result[state][sample] = np.nan
#                     continue

#                 if lam_t <= 0.0 or lam_t1 <= 0.0:
#                     eff_result[state][sample] = np.nan
#                     continue

#                 eff_result[state][sample] = -log(lam_t1 / lam_t)

#         return eff_result

#     def pre_run(self, t0, t):
#         """
#         Same usage as your original method.

#         Returns
#         -------
#         eigenvalues : ndarray
#             Shape (n_samples, n_ops), containing lambda_n(t,t0).
#         """
#         Dt = t - t0

#         # Build reference eigenvectors from C(Dt) relative to C(0).
#         # This matches the spirit of your original:
#         #
#         #     self.GEVP_init(*self.CorrelationMatrix(self.data, 0, Dt))
#         #
#         # but the internals are now safer.
#         C_ref_t0, C_ref_t = self.CorrelationMatrix(self.data, 0, Dt)
#         self.init_idx = self.GEVP_init(C_ref_t0, C_ref_t)

#         C_t0, C_t = self.CorrelationMatrix(self.data, t0, t)

#         eigen_values, eigen_vectors = self.GEVP(C_t0, C_t)

#         return eigen_values

#     def run(self, t0, t):
#         """
#         Same usage as your original method.

#         Returns
#         -------
#         eff_result : list[JackknifeDistribution]
#             Effective energy jackknife distributions for each state.
#         """
#         # Your original had:
#         #
#         #     self.custom_sort = sorted
#         #
#         # which accidentally stores Python's built-in sorted function.
#         # This is the corrected version.
#         self.custom_sort = False

#         eig_t = self.pre_run(t0, t)
#         eig_t1 = self.pre_run(t0, t + 1)

#         return self.eff_energy(eig_t, eig_t1)


#########################################################################



class GEVP_OG_test:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        #self.previous_eigen = np.array([])
        self.init_idx = np.array([])
        self.custom_sort = False


    #static method 

    def CorrelationMatrix(self,data,t0,t):
        """
        Organizes preprocessed (resampled) data
        into correlation matrix of size (num_ops by num_ops)

        returns: Correlation matrix for each jackknife sample at t and t0
        returns k-samples of C_ij

        """
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        C_mats = np.zeros((self.size,self.current_n,self.current_n))
        C_t = np.zeros((self.size,self.current_n,self.current_n))
        
        for k in range(self.size):
            for i in range(np.shape(data)[0]):
                for j in range(np.shape(data)[0]):
                    if i<=j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]


        return C_t, C_mats

    def GEVP(self, C_t0, C_t):
        """
        C_t0 indexed by jackknife index and row and column
        """
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            print(C_t0)
            try:
                eigvals, eigvecs = eigh(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
                idx = np.argsort(-eigvals) #Max eig value
                

                eigen_values[i] = eigvals[idx]
                eigen_vectors[i] = eigvecs[:,idx]
            except:
                eigen_values.fill(np.nan)
                eigen_vectors.fill(np.nan)

                return eigen_values, eigen_vectors
        
        return np.real(eigen_values), np.real(eigen_vectors)
    

    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]  #Unsure of class for this

        for i in range(self.current_n):
            for j in range(self.size):
                if np.real(eig_val_t_t0[j][i]) != 0.0:
                    val = (np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))
                    if val > 0:
                        eff_result[i][j] = -(log(val)) #-log(A/B)
                    else:
                        eff_result[i][j] = np.nan
                else:
                    eff_result[i] = np.full(self.size, np.nan)

        return eff_result 
    
    def output_Correlation_Mat(self):
        pass
    
    def pre_run(self, t0, t):
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)

        return self.GEVP(C1, C2)[0]
    
    def run(self, t0, t):
        #self.custom_sort = sorted
        return self.eff_energy(self.pre_run(t0, t),self.pre_run(t0, t+1))  

  


class GEVP_OG:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        # self.previous_eigen = np.array([])
        self.init_idx = np.array([])
        self.custom_sort = False

        # Added: numerical cutoff for whitening C(t0)
        self.rcond = 1e-8


    def CorrelationMatrix(self, data, t0, t):
        """
        Organizes preprocessed resampled data into correlation matrices.

        Returns
        -------
        C_t:
            C(t0), shape (size, current_n, current_n)

        C_mats:
            C(t), shape (size, current_n, current_n)
        """

        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops

        C_mats = np.zeros((self.size, self.current_n, self.current_n))
        C_t = np.zeros((self.size, self.current_n, self.current_n))

        for k in range(self.size):
            for i in range(np.shape(data)[0]):
                for j in range(np.shape(data)[0]):
                    if i <= j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]

            # Added: force symmetry sample-by-sample
            C_mats[k] = 0.5 * (C_mats[k] + C_mats[k].T)
            C_t[k] = 0.5 * (C_t[k] + C_t[k].T)

        return C_t, C_mats


    def WhiteningMatrix(self, Cor_t0):
        """
        Compute W = C(t0)^(-1/2).

        Used to convert

            C(t) v = lambda C(t0) v

        into the ordinary symmetric eigenvalue problem

            W C(t) W u = lambda u.
        """

        Cor_t0 = 0.5 * (Cor_t0 + Cor_t0.T)

        eigvals, eigvecs = eigh(Cor_t0)

        max_eval = np.max(np.abs(eigvals))
        cutoff = self.rcond * max_eval

        if np.any(eigvals <= cutoff):
            raise ValueError(
                f"C(t0) is not positive definite. "
                f"min eigenvalue = {np.min(eigvals)}, cutoff = {cutoff}"
            )

        inv_sqrt = 1.0 / np.sqrt(eigvals)

        W = eigvecs @ np.diag(inv_sqrt) @ eigvecs.T

        return W


    def GEVP(self, C_t0, C_t):
        """
        Solves

            C(t) v = lambda C(t0) v

        for each jackknife sample.

        Returns
        -------
        eigen_values:
            shape (size, current_n)

        eigen_vectors:
            shape (size, current_n, current_n)
        """

        eigen_values = np.full((self.size, self.current_n), np.nan)
        eigen_vectors = np.full((self.size, self.current_n, self.current_n), np.nan)

        for i, (Cor_t0, Cor_t) in enumerate(zip(C_t0, C_t)):

            Cor_t0 = 0.5 * (Cor_t0 + Cor_t0.T)
            Cor_t = 0.5 * (Cor_t + Cor_t.T)

            try:
                W = self.WhiteningMatrix(Cor_t0)

                M = W @ Cor_t @ W
                M = 0.5 * (M + M.T)

                eigvals, eigvecs_white = eigh(M)

                # Sort largest eigenvalue first.
                idx = np.argsort(eigvals)[::-1]

                eigvals = eigvals[idx]
                eigvecs_white = eigvecs_white[:, idx]

                # Convert whitened eigenvectors back to generalized eigenvectors.
                eigvecs = W @ eigvecs_white

                eigen_values[i] = eigvals
                eigen_vectors[i] = eigvecs

            except ValueError:
                # If C(t0) is bad for this sample, leave nan entries.
                continue

        return np.real(eigen_values), np.real(eigen_vectors)


    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        """
        Computes

            E_eff(t) = -log(lambda(t+1,t0) / lambda(t,t0))

        Returns the original structure:

            eff_result[state][jackknife_sample]
        """

        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]

        for i in range(self.current_n):
            for j in range(self.size):

                lam_t = np.real(eig_val_t_t0[j][i])
                lam_t1 = np.real(eig_val_t1_t0[j][i])

                if np.isfinite(lam_t) and np.isfinite(lam_t1) and lam_t != 0.0:
                    val = lam_t1 / lam_t

                    if np.isfinite(val) and val > 0:
                        eff_result[i][j] = -log(val)
                    else:
                        pass
                        print(
                            f"[Invalid effective energy] "
                            f"state={i}, jackknife_sample={j}, "
                            f"lambda(t)={lam_t}, lambda(t+1)={lam_t1}, "
                            f"ratio={val}"
                        )
                        #eff_result[i][j] = np.nan

                else:
                    pass
                    print(
                        f"[Invalid eigenvalue] "
                        f"state={i}, jackknife_sample={j}, "
                        f"lambda(t)={lam_t}, lambda(t+1)={lam_t1}"
                    )
                    #eff_result[i][j] = np.nan

        return eff_result


    def pre_run(self, t0, t):
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)

        return self.GEVP(C1, C2)[0]


    def run(self, t0, t):
        return self.eff_energy(self.pre_run(t0, t), self.pre_run(t0, t + 1))

class GEVP2:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).shape()
        self.current_n = np.shape(data)[0]
        #self.previous_eigen = np.array([])
        self.init_idx = np.array([])
        self.custom_sort = False


    #static method 

    def CorrelationMatrix(self,data,t0,t):
        #print(self.size)
        """
        Organizes preprocessed (resampled) data
        into correlation matrix of size (num_ops by num_ops)

        returns: Correlation matrix for each jackknife sample at t and t0
        """
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        #Cmat = np.zeros((self.num_ops,self.num_ops))
        #Ct = np.zeros((self.num_ops,self.num_ops))
        C_mats = np.zeros((*self.size,self.current_n,self.current_n))
        C_t = np.zeros((*self.size,self.current_n,self.current_n))
        
        #try some multithreading
        for k in product(*[range(N) for N in self.size]):
            for i in range(np.shape(data)[0]):
                for j in range(np.shape(data)[0]):
                    if i<=j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]
                #C_mats[k] = Cmat
                #C_t[k] = Ct

        return C_t, C_mats
    


    def GEVP_old(self, C_t0, C_t):
        eigen_values = np.zeros((*self.size, self.current_n))
        eigen_vectors = np.zeros((*self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value
            #print(idx)

            #sort both eigvals and eigvec
            if self.custom_sort == True:
                idx2 = self.sortVector(self.init_idx, eigvecs)
            else:
                idx2 = idx

            eigen_values[i] = eigvals[idx2]
            eigen_vectors[i] = eigvecs[:,idx2]
        
        return np.real(eigen_values), np.real(eigen_vectors)
    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((*self.size, self.current_n))
        eigen_vectors = np.zeros((*self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for k in product(*[range(N) for N in self.size]):
            eigvals, eigvecs = eig(C_t[k], C_t0[k])  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value
            #print(idx)

            #sort both eigvals and eigvec
            if self.custom_sort == True:
                idx2 = self.sortVector(self.init_idx, eigvecs)
            else:
                idx2 = idx

            eigen_values[k] = eigvals[idx2]
            eigen_vectors[k] = eigvecs[:,idx2]
        
        return np.real(eigen_values), np.real(eigen_vectors)
    
    
    def Rebasing(self, C_t0, C_t, eigen_vec):
        
        self.current_n -= 1
        #self.num_ops = self.current_n
        V = eigen_vec
        Cor_t0_dist_rebased = np.zeros((self.size,self.current_n,self.current_n))
        Cor_t_dist_rebased = np.zeros((self.size,self.current_n,self.current_n))
        for i, (Cor_t0, Cor_t, eigvec) in enumerate(zip(C_t0, C_t, V)):
            Cor_t_dist_rebased[i] = eigvec[:, :self.current_n].conj().T @ Cor_t @ eigvec[:, :self.current_n]
            Cor_t0_dist_rebased[i]= eigvec[:, :self.current_n].conj().T @ Cor_t0 @ eigvec[:, :self.current_n]


        return Cor_t0_dist_rebased, Cor_t_dist_rebased
    
    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        #Modify for all distrubution types 
        eff_result = [BlockDoubleJackknifeDistribution(741, 8) for j in range(self.current_n)]  #Unsure of class for this
        #use flattened indices
        for i in range(self.current_n):
            for k in product(*[range(N) for N in self.size]):
                # if log gives error return nan
                eff_result[i][k] = -(log(np.real(eig_val_t1_t0[k][i])/np.real(eig_val_t_t0[k][i]))) #-log(A/B)

        return eff_result 
    
    def pre_run(self, N_times, t0, t):
        Dt = t - t0
        #self.init_idx = self.GEVP_init(*self.CorrelationMatrix(self.data, 0,Dt))
        #print(self.init_idx)
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)
        #C3, C4 = self.CorrelationMatrix(self.data, t0, t+1)
        for _ in range(N_times):
            eigvals, eigvecs = self.GEVP(C1, C2)
            
            C1, C2 = self.Rebasing(C1, C2, eigvecs)
            #print(np.shape(C1))

            #size = C1.shape[1]
            if self.num_ops <= 1:
                break  # Cannot reduce further
            
            
        
        # Return final eigenvalues from last GEVP
        #self.current_n = self.num_ops
        return self.GEVP(C1, C2)[0]
    
    def run(self, N_times, t0, t, sorted):
        self.custom_sort = sorted
        return self.eff_energy(self.pre_run(N_times, t0, t), self.pre_run(N_times, t0, t+1))
        #return self.pre_run(N_times, t0, t)



class GEVP4:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        self.init_idx = np.array([])
        self.custom_sort = False


    def CorrelationMatrix(self, data, t0, t):
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        C_mats = np.zeros((self.size, self.current_n, self.current_n))
        C_t = np.zeros((self.size, self.current_n, self.current_n))
        
        for k in range(self.size):
            for i in range(self.num_ops):
                for j in range(self.num_ops):
                    if i <= j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]
        return C_t, C_mats


    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        for i, (Cor_t0, Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)
            idx = np.argsort(-eigvals)  # sort descending
            eigen_values[i] = eigvals[idx]
            eigen_vectors[i] = eigvecs[:, idx]
        return np.real(eigen_values), np.real(eigen_vectors)
    
    
    def Rebasing(self, C_t0, C_t, eigen_vec):
        self.current_n -= 1
        V = eigen_vec
        Cor_t0_dist_rebased = np.zeros((self.size, self.current_n, self.current_n))
        Cor_t_dist_rebased = np.zeros((self.size, self.current_n, self.current_n))
        for i, (Cor_t0, Cor_t, eigvec) in enumerate(zip(C_t0, C_t, V)):
            Cor_t_dist_rebased[i] = eigvec[:, :self.current_n].conj().T @ Cor_t @ eigvec[:, :self.current_n]
            Cor_t0_dist_rebased[i] = eigvec[:, :self.current_n].conj().T @ Cor_t0 @ eigvec[:, :self.current_n]
        return Cor_t0_dist_rebased, Cor_t_dist_rebased
    
    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        eff_result = np.zeros((self.current_n, self.size))
        for i in range(self.current_n):
            for j in range(self.size):
                eff_result[i, j] = -(log(np.real(eig_val_t1_t0[j][i]) / np.real(eig_val_t_t0[j][i])))
        return eff_result 
    
    def pre_run(self, N_times, t0, t):
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)
        for _ in range(N_times):
            eigvals, eigvecs = self.GEVP(C1, C2)
            C1, C2 = self.Rebasing(C1, C2, eigvecs)
            if self.num_ops <= 1:
                break
        return self.GEVP(C1, C2)[0]
    
    def run(self, N_times, t0, t, sorted):
        self.custom_sort = sorted
        return self.eff_energy(self.pre_run(N_times, t0, t),
                               self.pre_run(N_times, t0, t+1))

    # NEW: scan across all time slices
    def rebasing_scan_all(self, N_times, t0, t_min, t_max):
        results = {}
        for t in range(t_min, t_max):
            eigvals_t = self.pre_run(N_times, t0, t)
            eigvals_t1 = self.pre_run(N_times, t0, t+1)
            effE = self.eff_energy(eigvals_t, eigvals_t1)
            results[t] = effE
        return results

    # NEW: plot effective energies vs t
    def plot_eff_energies(self, scan_results):
        plt.figure(figsize=(10,6))
        t_values = sorted(scan_results.keys())
        num_states = scan_results[t_values[0]].shape[0]

        for n in range(num_states):
            means = [np.mean(scan_results[t][n]) for t in t_values]
            errs = [np.std(scan_results[t][n]) for t in t_values]
            plt.errorbar(t_values, means, yerr=errs, fmt='o-', label=f"State {n}")

        plt.xlabel("t")
        plt.ylabel("Effective energy")
        plt.title("Rebased GEVP Effective Energies")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()




class GEVP5:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        self.init_idx = np.array([])
        self.custom_sort = False

    def CorrelationMatrix(self, data, t0, t):
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        C_mats = np.zeros((self.size, self.current_n, self.current_n))
        C_t = np.zeros((self.size, self.current_n, self.current_n))

        for k in range(self.size):
            for i in range(self.num_ops):
                for j in range(self.num_ops):
                    if i <= j:
                        C_mats[k][i][j] = data[i][j].value(t)[k]
                        C_t[k][i][j] = data[i][j].value(t0)[k]
                    else:
                        C_mats[k][i][j] = data[j][i].value(t)[k]
                        C_t[k][i][j] = data[j][i].value(t0)[k]
        return C_t, C_mats

    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        for i, (Cor_t0, Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)
            idx = np.argsort(-eigvals)
            eigen_values[i] = eigvals[idx]
            eigen_vectors[i] = eigvecs[:, idx]
        return np.real(eigen_values), np.real(eigen_vectors)

    def Rebasing(self, C_t0, C_t, eigen_vec):
        self.current_n -= 1
        V = eigen_vec
        Cor_t0_dist_rebased = np.zeros((self.size, self.current_n, self.current_n))
        Cor_t_dist_rebased = np.zeros((self.size, self.current_n, self.current_n))
        for i, (Cor_t0, Cor_t, eigvec) in enumerate(zip(C_t0, C_t, V)):
            Cor_t_dist_rebased[i] = (
                eigvec[:, : self.current_n].conj().T @ Cor_t @ eigvec[:, : self.current_n]
            )
            Cor_t0_dist_rebased[i] = (
                eigvec[:, : self.current_n].conj().T @ Cor_t0 @ eigvec[:, : self.current_n]
            )
        return Cor_t0_dist_rebased, Cor_t_dist_rebased

    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        eff_result = np.zeros((self.current_n, self.size))
        for i in range(self.current_n):
            for j in range(self.size):
                val = np.real(eig_val_t1_t0[j][i]) / np.real(eig_val_t_t0[j][i])
                if val > 0:
                    eff_result[i, j] = -log(val)
                else:
                    eff_result[i, j] = np.nan
        return eff_result

    def rebasing_at_different_t(self, t0, t_min, t_max):
        """
        Try rebasing once at each candidate timeslice t_rebase,
        then compute effective energies for later times.
        Returns dictionary: {t_rebase: {t: effE}}
        """
        results = {}

        for t_rebase in range(t_min, t_max):
            # Step 1: correlation matrices at (t0, t_rebase)
            C_t0, C_tr = self.CorrelationMatrix(self.data, t0, t_rebase)

            # Step 2: solve GEVP at rebase time
            eigvals, eigvecs = self.GEVP(C_t0, C_tr)

            # Step 3: rebase
            C_t0_rebased, C_tr_rebased = self.Rebasing(C_t0, C_tr, eigvecs)

            # Step 4: scan later times
            eff_energies = {}
            for t in range(t_rebase + 1, t_max):
                _, C_t = self.CorrelationMatrix(self.data, t0, t)
                _, C_t1 = self.CorrelationMatrix(self.data, t0, t + 1)

                eigvals_t, _ = self.GEVP(C_t0_rebased, C_t)
                eigvals_t1, _ = self.GEVP(C_t0_rebased, C_t1)
                effE = self.eff_energy(eigvals_t, eigvals_t1)

                eff_energies[t] = effE

            results[t_rebase] = eff_energies

        return results

    def plot_rebasing_scan(self, scan_results):
        plt.figure(figsize=(10, 6))

        for t_rebase, effdict in scan_results.items():
            if not effdict:  # skip empty
                continue
            t_values = sorted(effdict.keys())
            num_states = list(effdict.values())[0].shape[0]

            for n in range(num_states):
                means = [np.nanmean(effdict[t][n]) for t in t_values]
                errs = [np.nanstd(effdict[t][n]) for t in t_values]
                plt.errorbar(
                    t_values,
                    means,
                    yerr=errs,
                    fmt="o-",
                    label=f"State {n}, rebase@{t_rebase}",
                )

        plt.xlabel("t")
        plt.ylabel("Effective energy")
        plt.title("Effective energies for different rebasing times")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


class GEVP_bd:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        if isinstance(data[0][0].value(0), BlockDoubleJackknifeDistribution):
            self.size_inner = data[0][0].value(0)[0].size()
        self.current_n = np.shape(data)[0]
        #self.previous_eigen = np.array([])
        self.init_idx = np.array([])
        self.custom_sort = False


    #static method 
    
    def CorrelationMatrix(self,data,t0,t):
        """
        Organizes preprocessed (resampled) data
        into correlation matrix of size (num_ops by num_ops)

        returns: Correlation matrix for each jackknife sample at t and t0
        """
        self.num_ops = np.shape(data)[0]
        self.current_n = self.num_ops
        #Cmat = np.zeros((self.num_ops,self.num_ops))
        #Ct = np.zeros((self.num_ops,self.num_ops))
        C_mats = np.zeros((self.size,self.size_inner,self.current_n,self.current_n))
        C_t = np.zeros((self.size,self.size_inner,self.current_n,self.current_n))
        
        for k in range(self.size):
            for l in range(self.size_inner):
                for i in range(np.shape(data)[0]):
                    for j in range(np.shape(data)[0]):
                        if i<=j:
                            C_mats[k][l][i][j] = data[i][j].value(t)[k][l]
                            C_t[k][l][i][j] = data[i][j].value(t0)[k][l]
                        else:
                            C_mats[k][l][i][j] = data[j][i].value(t)[k][l]
                            C_t[k][l][i][j] = data[j][i].value(t0)[k][l]
            #C_mats[k] = Cmat
            #C_t[k] = Ct

        return C_t, C_mats
    
    def sortVector(self, V_init, V_current):

        sorted_vecs = np.empty(len(V_init))

        for i, v_i in enumerate(V_init):
            overlaps = [np.dot(v_i/norm(v_i), v/norm(v)) for v in V_current]
            max_index = np.argmax(overlaps)
            #print(max_index)
            sorted_vecs[i] = max_index

        return np.argsort(sorted_vecs)
    
    def sortVector1(self, V_init, V_current, idx):

        sorted_vecs = np.empty(len(V_current))
        if len(V_init)  == 0:
            for i, v_i in enumerate(V_init):
                overlaps = [np.dot(v_i/norm(v_i), v/norm(v)) for v in V_current]
                #print out overlaps
                max_index = np.argmax(overlaps)
                sorted_vecs[i] = max_index
        else:
            sorted_vecs = idx

        return np.argsort(sorted_vecs)

    
    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.size_inner, self.current_n))
        eigen_vectors = np.zeros((self.size, self.size_inner, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for k in range(self.size):
            for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0[k], C_t[k])):
                # eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
                # idx = np.argsort(-eigvals) #Max eig value
                # #print(idx)

                # #sort both eigvals and eigvec
                # if self.custom_sort == True:
                #     idx2 = self.sortVector(self.init_idx, eigvecs)
                # else:
                #     idx2 = idx

                # eigen_values[k][i] = eigvals[idx2]
                # eigen_vectors[k][i] = eigvecs[:,idx2]

                try:
                    eigvals, eigvecs = eigh(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
                    idx = np.argsort(-eigvals) #Max eig value
                        

                    eigen_values[k][i] = eigvals[idx]
                    eigen_vectors[k][i] = eigvecs[:,idx]
                except:
                    eigen_values.fill(np.nan)
                    eigen_vectors.fill(np.nan)

                    return eigen_values, eigen_vectors
            

        #print(type(eigen_vectors))
        
        return np.real(eigen_values), (np.real(eigen_vectors).mean(axis=0)).mean(axis=0)
    
    def GEVP_init(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.size_inner, self.current_n))
        eigen_vectors = np.zeros((self.size, self.size_inner, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for k in range(self.size):
            for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0[k], C_t[k])):
                eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
                idx = np.argsort(-eigvals) #Max eig value


                eigen_values[k][i] = eigvals[idx]
                eigen_vectors[k][i] = eigvecs[:,idx]
        
        return np.real(eigen_vectors[2][0])
    
    def Rebasing(self, C_t0, C_t, eigen_vec):
        
        self.current_n -= 1
        #self.num_ops = self.current_n
        #V = eigen_vec
        Cor_t0_dist_rebased = np.zeros((self.size, self.size_inner, self.current_n,self.current_n))
        Cor_t_dist_rebased = np.zeros((self.size, self.size_inner,self.current_n,self.current_n))
        for k in range(self.size):
            for i, (Cor_t0, Cor_t) in enumerate(zip(C_t0[k], C_t[k])):
                Cor_t_dist_rebased[k][i] = eigen_vec[:, :self.current_n].conj().T @ Cor_t @ eigen_vec[:, :self.current_n]
                Cor_t0_dist_rebased[k][i]= eigen_vec[:, :self.current_n].conj().T @ Cor_t0 @ eigen_vec[:, :self.current_n]

            #print(eigvec[:, :self.current_n].T) #scale such that at t=0 Cor diag on the order of 1

       

        return Cor_t0_dist_rebased, Cor_t_dist_rebased
    
    def eff_energy(self, eig_val_t_t0, eig_val_t1_t0):
        block = math.ceil(self.size_inner/self.size)
        eff_result = [BlockDoubleJackknifeDistribution(self.size*block, block) for j in range(self.current_n)]  #Unsure of class for this
        print(eff_result[0].shape())
        for i in range(self.current_n):
            for j in range(self.size):
                for k in range(self.size_inner):
                    if np.abs(np.imag(eig_val_t_t0[j][k][i])) > 1e-14:
                        pass
                    if np.real(eig_val_t_t0[j][k][i]) != 0 and not np.isnan(np.real(eig_val_t_t0[j][k][i])) and not np.isinf(np.real(eig_val_t_t0[j][k][i])):
                        val = (np.real(eig_val_t1_t0[j][k][i])/np.real(eig_val_t_t0[j][k][i]))
                        if val > 0:
                            eff_result[i][j][k] = -(log(val)) #-log(A/B)
                    else:
                        pass
                    #eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i])))
    

        return eff_result 
    
    def pre_run(self, t0, t, t0_1, t_1, rebase):
        Dt = t - t0
        self.init_idx = self.GEVP_init(*self.CorrelationMatrix(self.data, 0,Dt))
        #print(self.init_idx)
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)
        C3, C4 = self.CorrelationMatrix(self.data, t0_1, t_1)

        eigvals, eigvecs = self.GEVP(C3, C4)
        if rebase == True:    
            C1, C2 = self.Rebasing(C1, C2, eigvecs)

            
            
        
        # Return final eigenvalues from last GEVP
        #self.current_n = self.num_ops
        return self.GEVP(C1, C2)[0]
    
    def run(self, t0, t, t0_1=0, t_1=0, rebase = True):
        self.custom_sort = sorted
        return self.eff_energy(self.pre_run(t0, t, t0_1, t_1, rebase),self.pre_run(t0, t+1, t0_1, t_1, rebase))


