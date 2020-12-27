import numpy as np
import pandas as pd
import math
import torch

import matplotlib.pyplot as plt

import ot
import ot.plot

import os.path
from os import path

from scipy.io import loadmat
from scipy.special import comb

import itertools

from multiprocessing import Pool, RawArray

from functools import partial

from solver.tlb_kl_sinkhorn_solver import TLBSinkhornSolver


###############################################################################
# Function Declarations
###############################################################################

def readProteinCodes(file):
    next(file) # skip header line
    codes = []
    for line in file:
        res = line.split(',')
        domain = res[0]
        cathcode = res[1].strip('\n')
        codes.append([domain, cathcode])
    return codes

###############################################################################
# Run Unbalanced Gromov-Wasserstein Process
###############################################################################

# A global dictionary storing the variables passed from the initializer.
var_dict = {}

def init_worker(rhos, ents, solver, geodists):
    # Using a dictionary is not strictly necessary. You can also
    # use global variables.
    var_dict['rhos'] = rhos
    var_dict['ents'] = ents
    var_dict['solver'] = solver
    
    numproteins = len(geodists)
    for i in range(numproteins):
        var_dict['D' + str(i)] = geodists[i]

def runGW(idx):
    ind1, ind2, i, j = idx
    
    # load in the global variables that we read from
    rhos = var_dict['rhos']
    ents = var_dict['ents']
    solver = var_dict['solver']
    
    rho = rhos[i]
    ent = ents[j]
    
    D1 = np.frombuffer(var_dict['D' + str(ind1)], dtype=np.float64)
    D2 = np.frombuffer(var_dict['D' + str(ind2)], dtype=np.float64)

    # number of atoms in each protein
    num_atoms1 = int(math.sqrt(D1.size))
    num_atoms2 = int(math.sqrt(D2.size))
    
    # reshape the distance matrices
    D1 = np.reshape(D1, [num_atoms1, num_atoms1])
    D2 = np.reshape(D2, [num_atoms2, num_atoms2])
    
    # empirical distributions over atoms
    mu1 = np.ones(num_atoms1) / num_atoms1
    mu2 = np.ones(num_atoms2) / num_atoms2
    
    # Peyre et al. unbalanced GW
    try:
        #coupling = ot.gromov.gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss')
        #coupling = ot.gromov.entropic_gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss', ent)
        
        coupling, _ = solver.tlb_sinkhorn(torch.from_numpy(mu1), torch.from_numpy(D1),
                                          torch.from_numpy(mu2), torch.from_numpy(D2),
                                          rho, ent)
        coupling = coupling.numpy()
    except:
        result = (math.inf, math.inf)
        return idx, result
    
    mass = np.sum(coupling)
    
    # if mass of coupling is zero, GW analysis failed
    if mass <= 1e-15:
        result = (math.inf, math.inf)
        return idx, result
    
    # compute GW cost for inferred coupling
    constD, hD1, hD2 = ot.gromov.init_matrix(D1, D2, mu1, mu2, loss_fun='square_loss')
    GWdist = ot.gromov.gwloss(constD, hD1, hD2, coupling)
    result = (GWdist, mass)
    
    return idx, result



###############################################################################
# Create Many UGW Programs for Different Protein Pairs/Parameter Combinations
###############################################################################

if __name__ == '__main__':
    # Get protein geodseic distances
    mat0 = loadmat('ProteinData0.mat')
    mat1 = loadmat('ProteinData1.mat')
    mat2 = loadmat('ProteinData2.mat')
    mat3 = loadmat('ProteinData3.mat')
    mat4 = loadmat('ProteinData4.mat')
    
    protein_data = np.concatenate([mat0['X_0'], mat1['X_1'], mat2['X_2'], mat3['X_3'], mat4['X_4']])
    #protein_data = loadmat('ProteinData_CA.mat')['X1']
    
    geodists = protein_data[:, 0]
    domains = protein_data[:, 1]
    cath_codes = protein_data[:, 2]
    
    numproteins = domains.size
    
    # compute max value in each distance matrix
    diams = np.zeros(numproteins)
    norms = np.zeros(numproteins)
    for i in range(numproteins):
        diams[i] = np.max(geodists[i])
        norms[i] = np.linalg.norm(geodists[i], 'fro')
    
    # Initialize the UGW solver
    nits = 50
    nits_sinkhorn = 1000
    tol = 1e-10
    tol_sinkhorn = 1e-7
    maxtime = 1200
    solver = TLBSinkhornSolver(nits, nits_sinkhorn, tol, tol_sinkhorn, maxtime)
    
    rhos = np.array([5e-4]) #np.array([1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3])
    numrhos = rhos.size
    
    ents = np.array([4e-5]) #np.array([1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4])
    numents = ents.size
    
    distMats = np.empty((numproteins, numproteins, numrhos, numents))
    distMats[:] = np.nan
    masses = np.empty((numproteins, numproteins, numrhos, numents))
    masses[:] = np.nan
    
    expname = 'unbalanced_GW'
    if not path.exists(expname):
        os.makedirs(expname)
    
    np.savez(expname + '/params.npz', expname, nits=nits, nits_sinkhorn=nits_sinkhorn,
             tol=tol, tol_sinkhorn=tol_sinkhorn, maxtime=maxtime, rhos=rhos, ents=ents)
    
    outname = expname + '/results.npz'
    #outname = expname + '/results_ac.npz'
    if path.exists(outname):
        # put old file results into new array
        results = np.load(outname)
        distMats = results['distMats']
        masses = results['masses']
    
    # remove proteins that take too long from the analysis
    protein_inds = np.arange(numproteins)
    bad_inds = np.where(np.isinf(diams))[0]
    protein_inds = np.setdiff1d(protein_inds, bad_inds, True)

    # get all parameter combinations
    all_idxs = [(i, j, k, l) for i, j, k, l in
           itertools.product(protein_inds, protein_inds, range(numrhos), range(numents))]
    
    # keep only those parameter combinations which have not been tested
    idxs = []
    for idx in all_idxs:
        if np.isnan(distMats[idx]):
            idxs.append(idx)
    
    # create shared memory for all processes
    shared_rhos = RawArray('d', rhos)
    shared_ents = RawArray('d', ents)
    shared_geodists = list()
    for i in range(numproteins):
        shared_geodists.append(RawArray('d', geodists[i].flatten()))

    # initialize pool of processes
    p = Pool(processes=20, maxtasksperchild=1, initializer=init_worker, initargs=(shared_rhos, shared_ents, solver, shared_geodists))
    print('Num CPUs: ' + str(os.cpu_count()))
    
    # run the GW function for each process
    freq = 1
    count = 0
    for (idx, res) in p.imap_unordered(partial(runGW), idxs):
        ind1, ind2, i, j = idx
        print(str(count) + ': ' + str(idx))
        
        # save GW distances between protein pairs
        GWdist, mass = res
        distMats[ind1, ind2, i, j] = GWdist
        masses[ind1, ind2, i, j] = mass
        
        # save results with a certain frequency
        if count % freq == 0:
            print('saved')
            np.savez(outname, distMats=distMats, masses=masses)
        count += 1
    
    p.close()
    p.join()
    
    # save final results
    print('saved')
    np.savez(outname, distMats=distMats, masses=masses)
