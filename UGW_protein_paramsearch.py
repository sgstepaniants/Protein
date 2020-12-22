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
from scipy.spatial import distance

from skimage.filters import threshold_minimum, threshold_otsu

import itertools

from multiprocessing import Pool

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

def makeCodeDicts(domains, cath_codes):
    n = len(domains)
    
    domainToCath = {}
    cToDomain = {}
    caToDomain = {}
    catToDomain = {}
    cathToDomain = {}
    for i in range(n):
        domain = domains[i][0]
        cath = cath_codes[i][0]
        
        groups = cath.split('.')
        c = groups[0]
        ca = '.'.join(groups[:2])
        cat = '.'.join(groups[:3])
        
        if c in cToDomain:
            cToDomain[c].append((i, domain))
        else:
            cToDomain[c] = [(i, domain)]
        
        if ca in caToDomain:
            caToDomain[ca].append((i, domain))
        else:
            caToDomain[ca] = [(i, domain)]
            
        if cat in catToDomain:
            catToDomain[cat].append((i, domain))
        else:
            catToDomain[cat] = [(i, domain)]
            
        if cath in cathToDomain:
            cathToDomain[cath].append((i, domain))
        else:
            cathToDomain[cath] = [(i, domain)]
        
        if domain in domainToCath:
            domainToCath[domain].append((i, cath))
        else:
            domainToCath[domain] = [(i, cath)]
    
    return domainToCath, cToDomain, caToDomain, catToDomain, cathToDomain

# remove NaNs rows and columns from a symmetric matrix
def removeNans(mat):
    n = mat.shape[0]
    inds = np.arange(n)
    
    nanCounts = np.sum(np.isnan(mat), axis=0)
    if np.all(nanCounts == 0):
        return mat, inds
    
    uniqueCounts = np.flip(np.unique(nanCounts))
    for count in uniqueCounts:
        filtered_inds = inds[nanCounts < count]
        filtered_mat = mat[filtered_inds][:, filtered_inds]
        filtered_numNans = np.sum(np.isnan(filtered_mat))
        if filtered_numNans == 0:
            return filtered_mat, filtered_inds

# from https://github.com/letiantian/kmedoids
def kMedoids(D, k, tmax=100):
    # determine dimensions of distance matrix D
    m, n = D.shape

    if k > n:
        raise Exception('too many medoids')

    # find a set of valid initial cluster medoid indices since we
    # can't seed different clusters with two points at the same location
    valid_medoid_inds = set(range(n))
    invalid_medoid_inds = set([])
    rs,cs = np.where(D==0)
    # the rows, cols must be shuffled because we will keep the first duplicate below
    index_shuf = list(range(len(rs)))
    np.random.shuffle(index_shuf)
    rs = rs[index_shuf]
    cs = cs[index_shuf]
    for r,c in zip(rs,cs):
        # if there are two points with a distance of 0...
        # keep the first one for cluster init
        if r < c and r not in invalid_medoid_inds:
            invalid_medoid_inds.add(c)
    valid_medoid_inds = list(valid_medoid_inds - invalid_medoid_inds)

    if k > len(valid_medoid_inds):
        raise Exception('too many medoids (after removing {} duplicate points)'.format(
            len(invalid_medoid_inds)))

    # randomly initialize an array of k medoid indices
    M = np.array(valid_medoid_inds)
    np.random.shuffle(M)
    M = np.sort(M[:k])

    # create a copy of the array of medoid indices
    Mnew = np.copy(M)

    # initialize a dictionary to represent clusters
    C = {}
    for t in range(tmax):
        # determine clusters, i.e. arrays of data indices
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]
        # update cluster medoids
        for kappa in range(k):
            J = np.mean(D[np.ix_(C[kappa],C[kappa])],axis=1)
            j = np.argmin(J)
            Mnew[kappa] = C[kappa][j]
        np.sort(Mnew)
        # check for convergence
        if np.array_equal(M, Mnew):
            break
        M = np.copy(Mnew)
    else:
        # final update of cluster memberships
        J = np.argmin(D[:,M], axis=1)
        for kappa in range(k):
            C[kappa] = np.where(J==kappa)[0]

    # return results
    return M, C

def rand_index(clusters, classes):
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)

###############################################################################
# Run Unbalanced Gromov-Wasserstein Process
###############################################################################

def runGW(idx, rhos, ents, solver_params, geodists):
    ind1, ind2, i, j = idx
    result = (math.nan, math.nan)
    
    rho = rhos[i]
    ent = ents[j]
    
    D1 = geodists[ind1]
    D2 = geodists[ind2]
    
    # number of atoms in each protein
    num_atoms1 = D1.shape[0]
    num_atoms2 = D2.shape[0]
    
    # empirical distributions over atoms
    mu1 = np.ones(num_atoms1) / num_atoms1
    mu2 = np.ones(num_atoms2) / num_atoms2
    
    # Peyre et al. unbalanced GW
    nits, nits_sinkhorn, tol, tol_sinkhorn, maxtime = solver_params
    solver = TLBSinkhornSolver(nits, nits_sinkhorn, tol, tol_sinkhorn, maxtime)
    try:
        #coupling = ot.gromov.gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss')
        #coupling = ot.gromov.entropic_gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss', ent)
        
        coupling, _ = solver.tlb_sinkhorn(torch.from_numpy(mu1), torch.from_numpy(D1),
                                          torch.from_numpy(mu2), torch.from_numpy(D2),
                                          rho, ent)
        coupling = coupling.numpy()
    except:
        return idx, result
    
    mass = np.sum(coupling)
    
    # if mass of coupling is zero, GW analysis failed
    if mass <= 1e-15:
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
    diams = np.zeros(numproteins)i
    norms = np.zeros(numproteins)
    for i in range(numproteins):
        diams[i] = np.max(geodists[i])
        norms[i] = np.linalg.norm(geodists[i], 'fro')
    
    # make dictionary of CATH code to domain
    domainToCath, cToDomain, caToDomain, catToDomain, cathToDomain = makeCodeDicts(protein_data[:, 1], protein_data[:, 2])
        
    # Initialize the UGW solver
    nits = 50
    nits_sinkhorn = 1000
    tol = 1e-10
    tol_sinkhorn = 1e-7
    maxtime = 1200
    solver_params = (nits, nits_sinkhorn, tol, tol_sinkhorn, maxtime)
    
    rhos = np.array([1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3])
    numrhos = rhos.size
    
    ents = np.array([1e-5, 2.5e-5, 5e-5, 7.5e-5, 1e-4, 2.5e-4, 5e-4])
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
    proteins_inds = range(numproteins)
    bad_inds = np.where(diams == math.inf)
    protein_inds = np.setdiff1d(proteins_inds, bad_inds, True)
    
    # get all parameter combinations
    all_idxs = [(i, j, k, l) for i, j, k, l in
           itertools.product(proteins_inds, proteins_inds, range(numrhos), range(numents))]
    
    # keep only those parameter combinations which have not been tested
    idxs = []
    for idx in all_idxs:
        if np.isnan(distMats[idx]):
            idxs.append(idx)
    
    # initialize pool of processes
    p = Pool(processes=20)
    print('Num CPUs: ' + str(os.cpu_count()))
    
    # run the GW function for each process
    freq = 1
    count = 0
    for (idx, res) in p.imap_unordered(partial(runGW, rhos=rhos, ents=ents, solver_params=solver_params, geodists=geodists), idxs):
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
