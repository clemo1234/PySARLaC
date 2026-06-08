import sys
import os



# Get the absolute path of the directory containing the module
module_dir = os.path.abspath('../../src')

# Add the directory to sys.path
sys.path.insert(0, module_dir)
#for path in sys.path:
#    print(path)

import PySaRLAC as sl
import random
import math
import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as pyplot
random.seed(1234)

idx_op_map = ["PiPiGnd","PiPiExc","Sigma"]
op_idx_map = dict()
for i in range(len(op_idx_map)):
    op_idx_map[idx_op_map[i]] = i

f = h5py.File('../../../combined_data.hdf5','r')

f.keys()

jdata = f['j_data']
con = jdata['contains']['entries']
for e in con.keys():
    op1 = con[e].attrs["first"][0]
    op2 = con[e].attrs["second"][0]
    print(op1,op2,idx_op_map[op1],idx_op_map[op2])


Lt=64
block_size = 8
nsample_unblocked = 741
nblock = nsample_unblocked // block_size
jdata = f['j_data']
jdata_cors = [ [None for j in range(3)] for i in range(3)] #3 ops
for i in range(3):
    for j in range(i,3):
        jdata_cors[i][j] = sl.CorrelationFunction(Lt)
        cdata = jdata['correlators']['m']["elem_%d_%d" % (i,j)]['series']
        for t in range(Lt):
            jvals = cdata["elem_%d" % t]['second'].attrs['data']
            assert len(jvals) == nblock
            jdist = sl.JackknifeDistribution(nblock)
            for s in range(nblock):
                jdist[s] = jvals[s]                        
            print(i,j,t,jdist)
            jdata_cors[i][j].setValue(t, jdist)
            jdata_cors[i][j].setCoord(t, float(t))


#For block double jackknife we expect a flattened array of size nblock * ( block_size*nblock - block_size ) = 66976
nouter_samp = nblock
ninner_samp = block_size*nblock - block_size
nunrolled= nouter_samp * ninner_samp
bdjdata = f['bdj_data']
bdjdata_cors = [ [None for j in range(3)] for i in range(3)]  #3 ops
for i in range(3):    
    for j in range(i,3):
        cdata = bdjdata['correlators']['m']["elem_%d_%d" % (i,j)]['series']   
        bdjdata_cors[i][j] = sl.CorrelationFunction(Lt)
        for t in range(Lt):
            jvals = np.array(cdata["elem_%d" % t]['second']['data']['unrolled_data'])
            assert len(jvals) == nunrolled
            jdist = sl.BlockDoubleJackknifeDistribution(nsample_unblocked, block_size)
            assert jdist.size() == nouter_samp and jdist[0].size() == ninner_samp
            u = 0
            for so in range(nouter_samp):
                jdist[so].sampleVector()[:] = jvals[u:u+ninner_samp]
                u+=ninner_samp
                #for si in range(ninner_samp):
                #    jdist[so][si] = jvals[u]
                #    u+=1
            print(i,j,t,jdist[0])
            bdjdata_cors[i][j].setValue(t, jdist)
            bdjdata_cors[i][j].setCoord(t, float(t))


jdata_cors_no_sigma = [[jdata_cors[0][0],jdata_cors[0][1]],[jdata_cors[0][1],jdata_cors[1][1]]]
jdata_cors_no_pipiEx = [[jdata_cors[0][0],jdata_cors[0][2]],[jdata_cors[0][2],jdata_cors[2][2]]]
jdata_cors_pipi_111 = [[jdata_cors[0][0]]]

bdjdata_cors_no_sigma = [[bdjdata_cors[0][0],bdjdata_cors[0][1]],[bdjdata_cors[0][1],bdjdata_cors[1][1]]]
bdjdata_cors_no_pipiEx = [[bdjdata_cors[0][0],bdjdata_cors[0][2]],[bdjdata_cors[0][2],bdjdata_cors[2][2]]]
bdjdata_cors_pipi_111 = [[bdjdata_cors[0][0]]]

alpha = 1
beta = 1
gamma = 1

coeffs = [alpha, beta, gamma]
#print(type(jdata_cors[1][0]))

#jdata_cors

for i in range(3):
    for j in range(i,3):
        #print(i,j)
        jdata_cors[i][j] = jdata_cors[i][j]*coeffs[i]*coeffs[j]

def gevp_output(input_data, E_n, t_max, Dt, rebasing = False, rebased_time_slice = 1, rebased_Dt = 2):
    #time_array = np.arange(0, t_max)
    
    data = sl.CorrelationFunction(t_max)
    if rebasing == False:
        gevp_obj = sl.GEVP_OG_test(input_data)
        if type(E_n) is not int:
            cor_list = np.empty(len(E_n), dtype=sl.CorrelationFunction)
            for idx in range(len(cor_list)):  
                data = sl.CorrelationFunction(t_max)
                for t0 in range(0,t_max):
                        #print(E_n[idx])
                        t = t0 + Dt
                        #print(t0)
                        data.setValue(t0, gevp_obj.run(t0,t)[E_n[idx]])
                        data.setCoord(t0, float(t0))
                cor_list[idx] = data
            return  cor_list
        else:
                print("bad")
                for t0 in range(0,t_max):
                        t = t0 + Dt
                        data.setValue(t0, gevp_obj.run(t0,t)[E_n])
                        data.setCoord(t0, float(t0))
                return data
             
    if rebasing == True:
        gevp_obj = sl.GEVP(input_data)
        if type(E_n) is not int:
            cor_list = np.empty(len(E_n), dtype=sl.CorrelationFunction)
            for idx in range(len(cor_list)):  
                data = sl.CorrelationFunction(t_max)
                for t0 in range(0,t_max):
                        #print(E_n[idx])
                        t = t0 + Dt
                        #print(t0)
                        data.setValue(t0, gevp_obj.run(t0,t,rebased_time_slice, rebased_time_slice + 
                                                  rebased_Dt)[E_n[idx]])
                        data.setCoord(t0, float(t0))
                cor_list[idx] = data
            return  cor_list
    
def gevp_output2(input_data, E_n, t_max, Dt, rebasing = False, rebased_time_slice = 1, rebased_Dt = 2):
    #time_array = np.arange(0, t_max)
    data = sl.CorrelationFunction(t_max)
    if rebasing == False:
        gevp_obj = sl.GEVP_bd(input_data)
        # for t0 in range(0,t_max-1):
        #         t = t0 + Dt
        #         data.setValue(t0, gevp_obj.run(t0,t, rebase = False)[E_n])
        #         data.setCoord(t0, float(t0))
        # return  data
        if type(E_n) is not int:
            cor_list = np.empty(len(E_n), dtype=sl.CorrelationFunction)
            for idx in range(len(cor_list)):  
                data = sl.CorrelationFunction(t_max)
                for t0 in range(0,t_max):
                        #print(E_n[idx])
                        t = t0 + Dt
                        #print(t0)
                        data.setValue(t0, gevp_obj.run(t0,t)[E_n[idx]])
                        data.setCoord(t0, float(t0))
                cor_list[idx] = data
            return  cor_list
        else:
                print("bad")
                for t0 in range(0,t_max):
                        t = t0 + Dt
                        data.setValue(t0, gevp_obj.run(t0,t)[E_n])
                        data.setCoord(t0, float(t0))
                return data
    if rebasing == True:
        gevp_obj = sl.GEVP_bd(input_data)
        if type(E_n) is not int:
            cor_list = np.empty(len(E_n), dtype=sl.CorrelationFunction)
            for idx in range(len(cor_list)):  
                data = sl.CorrelationFunction(t_max)
                for t0 in range(0,t_max):
                        #print(E_n[idx])
                        t = t0 + Dt
                        #print(t0)
                        data.setValue(t0, gevp_obj.run(t0,t,rebased_time_slice, rebased_time_slice + 
                                                  rebased_Dt)[E_n[idx]])
                        data.setCoord(t0, float(t0))
                cor_list[idx] = data
            return  cor_list