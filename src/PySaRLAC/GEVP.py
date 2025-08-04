from .CorrelationFunction import *
from .FitFunc import *
import numpy as np
import scipy.optimize as opt
from scipy.linalg import eigh, eig
from .JackknifeDistribution import *
from numpy import log
from numpy.linalg import norm


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
        
        return np.real(eigen_values), np.real(eigen_vectors)
    
    def GEVP_init(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value


            eigen_values[i] = eigvals[idx]
            eigen_vectors[i] = eigvecs[:,idx]
        
        return np.real(eigen_vectors[2])
    
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
        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]

        for i in range(self.current_n):
            for j in range(self.size):
                # if log gives error return nan
                eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))) #-log(A/B)

        return eff_result 
    
    def pre_run(self, N_times, t0, t):
        Dt = t - t0
        self.init_idx = self.GEVP_init(*self.CorrelationMatrix(self.data, 0,Dt))
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
        
            
