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


            eigen_values[i] = eigvals[idx]
            eigen_vectors[i] = eigvecs[:,idx]
        
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
                # if log gives error return nan
                eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))) #-log(A/B)

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

class GEVP_OG:
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
        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]  #Unsure of class for this

        for i in range(self.current_n):
            for j in range(self.size):
                # if log gives error return nan
                eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))) #-log(A/B)

        return eff_result 
    
    def pre_run(self, t0, t):
        Dt = t - t0
        self.init_idx = self.GEVP_init(*self.CorrelationMatrix(self.data, 0,Dt))
        #print(self.init_idx)
        C1, C2 = self.CorrelationMatrix(self.data, t0, t)
        #C3, C4 = self.CorrelationMatrix(self.data, t0_1, t_1)

        #eigvals, eigvecs = self.GEVP(C3, C4)
            
        #C1, C2 = self.Rebasing(C1, C2, eigvecs)

            
            
        
        # Return final eigenvalues from last GEVP
        #self.current_n = self.num_ops
        return self.GEVP(C1, C2)[0]
    
    def run(self, t0, t):
        self.custom_sort = sorted
        return self.eff_energy(self.pre_run(t0, t),self.pre_run(t0, t+1))
        #return self.pre_run(N_times, t0, t)       
            
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


class GEVP3:
    def __init__(self, data):
        self.data = data
        self.num_ops = np.shape(data)[0]
        self.size = data[0][0].value(0).size()
        self.current_n = np.shape(data)[0]
        #self.previous_eigen = np.array([])
        self.init_idx = np.array([])
  


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


    def GEVP(self, C_t0, C_t):
        eigen_values = np.zeros((self.size, self.current_n))
        eigen_vectors = np.zeros((self.size, self.current_n, self.current_n))
        #print(np.shape(eigen_values))
        for i, (Cor_t0,Cor_t) in enumerate(zip(C_t0, C_t)):
            eigvals, eigvecs = eig(Cor_t, Cor_t0)  #replace with eig package, see if all positive values
            idx = np.argsort(-eigvals) #Max eig value
            #print(idx)


            eigen_values[i] = eigvals[idx]
            eigen_vectors[i] = eigvecs[:,idx]
        
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
        eff_result = [JackknifeDistribution(self.size) for j in range(self.current_n)]  #Unsure of class for this

        for i in range(self.current_n):
            for j in range(self.size):
                # if log gives error return nan
                eff_result[i][j] = -(log(np.real(eig_val_t1_t0[j][i])/np.real(eig_val_t_t0[j][i]))) #-log(A/B)

        return eff_result 
    
    def pre_run(self, N_times, t0, t):
    
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

