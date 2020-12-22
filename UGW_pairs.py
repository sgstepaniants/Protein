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

from sklearn import metrics

import timeout_decorator

from solver.tlb_kl_sinkhorn_solver import TLBSinkhornSolver

###############################################################################
# Function Declarations
###############################################################################

# Balanced Gromov-Wasserstein (Solomon et al.)
def GromovWasserstein(mu0, mu1, D0, D1, ent, eta, thresh=1e-7):
    n0 = mu0.size
    n1 = mu1.size
    
    gamma = np.ones((n0, n1))
    while True:
        K = np.exp((D0*mu0).dot(gamma*mu1).dot(D1)/ent)
        gamma_new = SinkhornProjection(np.power(K, eta)*np.power(gamma, 1-eta), mu0, mu1, thresh)
        
        diff = np.linalg.norm(gamma - gamma_new)
        if diff < thresh:
            return gamma
        gamma = gamma_new

# Sinkhorn subroutine used in balanced Gromov-Wasserstein
def SinkhornProjection(K, mu0, mu1, thresh=1e-7):
    n0 = mu0.size
    n1 = mu1.size
    
    v0 = np.ones(n0)
    v1 = np.ones(n1)
    while True:
        v0_new = np.reciprocal(K.dot(v1*mu1))
        v1_new = np.reciprocal(np.transpose(K).dot(v0_new*mu0))
        
        diff = max(np.linalg.norm(v0 - v0_new), np.linalg.norm(v1 - v1_new))
        if diff < thresh:
            return ((K*v1).T * v0).T
        v0 = v0_new
        v1 = v1_new

# Unbalanced Gromov-Wasserstein (Solomon et al. with added KL regularization)
def UnbalancedGromovWasserstein(mu0, mu1, D0, D1, ent, lambda0, lambda1, eta, thresh=1e-7):
    n0 = mu0.size
    n1 = mu1.size
    
    gamma = np.ones((n0, n1))
    while True:
        K = np.exp((D0*mu0).dot(gamma*mu1).dot(D1)/ent)
        gamma_new = UnbalancedSinkhornProjection(np.power(K, eta)*np.power(gamma, 1-eta),
                                                 mu0, mu1, ent, lambda0, lambda1, thresh)
        
        diff = np.linalg.norm(gamma - gamma_new)
        print(diff)
        if diff < thresh:
            return gamma
        gamma = gamma_new

# Sinkhorn subroutine used in unbalanced Gromov-Wasserstein
def UnbalancedSinkhornProjection(K, mu0, mu1, ent, lambda0, lambda1, thresh=1e-7):
    n0 = mu0.size
    n1 = mu1.size
    
    v0 = np.ones(n0)
    v1 = np.ones(n1)
    while True:
        v0_new = np.power(np.reciprocal(K.dot(v1*mu1)), lambda0*np.reciprocal(lambda0+ent*mu0))
        v1_new = np.power(np.reciprocal(np.transpose(K).dot(v0_new*mu0)), lambda1*np.reciprocal(lambda1+ent*mu1))
        
        diff = max(np.linalg.norm(v0 - v0_new), np.linalg.norm(v1 - v1_new))
        if diff < thresh:
            return ((K*v1).T * v0).T
        v0 = v0_new
        v1 = v1_new

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
# Load in protein atom positions/distances
###############################################################################

#file = open('cathcodes.txt', 'r')
#codes = readProteinCodes(file)

# Get protein geodseic distances
#mat0 = loadmat('ProteinData0.mat')
#mat1 = loadmat('ProteinData1.mat')
#mat2 = loadmat('ProteinData2.mat')
#mat3 = loadmat('ProteinData3.mat')
#mat4 = loadmat('ProteinData4.mat')

#protein_data = np.concatenate([mat0['X_0'], mat1['X_1'], mat2['X_2'], mat3['X_3'], mat4['X_4']])

protein_data = loadmat('ProteinData_CA.mat')['X1']

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

# make dictionary of CATH code to domain
domainToCath, cToDomain, caToDomain, catToDomain, cathToDomain = makeCodeDicts(protein_data[:, 1], protein_data[:, 2])


###############################################################################
# Run Unbalanced Gromov-Wasserstein
###############################################################################

# Parameters
#rho = 1000     # Peyre et al. mass constraint parameter
#ent = 10      # entropic regularization, controls width of coupling band


# choose subset of proteins to compare
inds = np.arange(255)

#solver = TLBSinkhornSolver(nits=500, nits_sinkhorn=1000, gradient=False, tol=1e-3, tol_sinkhorn=1e-5)

costs = np.zeros((numproteins, numproteins))
costs[:] = np.nan
masses = np.zeros((numproteins, numproteins))
masses[:] = np.nan

#outname = 'results/rho=' + str(rho) + '_ent=' + str(ent) + '.npz'
#outname = 'results/rho=' + str(rho) + '_ent=' + str(ent) + '_alphacarbons.npz'
#outname = 'results/' + 'ent=' + str(ent) + '_balanced_alphacarbons.npz'
outname = 'results/balanced_alphacarbons.npz'
if path.exists(outname):
    # put old file results into new array
    results = np.load(outname)
    costs = results['costs']
    masses = results['masses']

#bad_inds = []
bad_inds = np.where(diams == math.inf)[0]
#bad_inds = [25, 76, 89]
#bad_inds = [15, 25, 65, 76, 89]
#bad_inds = [15, 25, 38, 65, 76, 88, 89, 94, 99]

# remove proteins that take too long from the analysis
inds = np.setdiff1d(inds, bad_inds, True)
numinds = inds.size

freq = 10 # how often to save
count = 0
for i in range(numinds):
    ind1 = inds[i]
    for j in range(i, numinds):
        ind2 = inds[j]
        
        #if not costs[ind1, ind2] == 0:
        #    continue
        if not np.isnan(costs[ind1, ind2]):
            continue
        
        print('Comparing proteins (' + str(ind1) + ', ' + str(ind2) + ')')
        
        dom1 = domains[ind1][0]
        dom2 = domains[ind2][0]
        cath1 = cath_codes[ind1][0]
        cath2 = cath_codes[ind2][0]
        D1 = geodists[ind1]
        D2 = geodists[ind2]
        
        # number of atoms in each protein
        num_atoms1 = D1.shape[0]
        num_atoms2 = D2.shape[0]
        
        # empirical distributions over atoms
        mu1 = np.ones(num_atoms1) / num_atoms1
        mu2 = np.ones(num_atoms2) / num_atoms2
        
        # Peyre et al. unbalanced GW
        try:
            coupling = ot.gromov.gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss')
            #coupling = ot.gromov.entropic_gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss', ent)
            
            #coupling, _ = solver.tlb_sinkhorn(torch.from_numpy(mu1), torch.from_numpy(D1),
            #                                  torch.from_numpy(mu2), torch.from_numpy(D2),
            #                                  rho, ent)
            #coupling = coupling.numpy()
        except timeout_decorator.TimeoutError as e:
            print(e)
            continue
        
        mass = np.sum(coupling)
        
        # if mass of coupling is zero, GW analysis failed
        if mass <= 1e-15:
            continue
        
        # compute GW cost for inferred coupling
        constD, hD1, hD2 = ot.gromov.init_matrix(D1, D2, mu1, mu2, loss_fun='square_loss')
        cost = ot.gromov.gwloss(constD, hD1, hD2, coupling)
        
        #couplings[ind1, ind2] = coupling
        costs[ind1, ind2] = cost
        masses[ind1, ind2] = mass
        count += 1
        
        if count % freq == 0:
            print('saving')
            np.savez(outname, costs=costs, masses=masses)

# make cost and mass matrices symmetric
i_lower = np.tril_indices(numproteins, -1)
costs[i_lower] = costs.T[i_lower]
masses[i_lower] = masses.T[i_lower]
np.savez(outname, costs=costs, masses=masses)

# plot cost matrix
plt.figure(1)
plt.pcolormesh(costs)
plt.colorbar()
plt.show()

# plot mass matrix
plt.figure(2)
plt.pcolormesh(masses)
plt.colorbar()
plt.show()


###############################################################################
# Cluster GW Distances and Compute Rand Index of Classification
###############################################################################

# keep only indices with non-nan distances
filtered_GWdists, filtered_inds = removeNans(costs)
numfiltered = len(filtered_inds)

filtered_masses = masses[filtered_inds][:, filtered_inds]
#filtered_GWdists = filtered_GWdists / (np.outer(diams[filtered_inds], diams[filtered_inds]) * filtered_masses**2)
filtered_GWdists = filtered_GWdists / (np.outer(diams[filtered_inds], diams[filtered_inds]))
#filtered_GWdists = filtered_GWdists / np.outer(norms[filtered_inds], norms[filtered_inds])
np.fill_diagonal(filtered_GWdists, 0)

# plot nan-filtered GW cost matrix
plt.figure(3)
plt.title('GW filtered distance matrix', fontweight='bold')
plt.pcolormesh(filtered_GWdists)
plt.colorbar()
plt.show()

plt.figure(4)
plt.title('GW coupling masses', fontweight='bold')
plt.pcolormesh(filtered_masses)
plt.colorbar()
plt.show()


# get true labels of proteins based on their C, CA, CAT, or CATH codes
true_labels = np.zeros(numproteins, dtype=int)
true_labels[:] = np.nan
codeDict = caToDomain
codecount = 0
for code in codeDict:
    indWithCodeExists = False
    for tple in codeDict[code]:
        ind = tple[0]
        true_labels[ind] = codecount
    codecount += 1
true_k = len(np.unique(true_labels[filtered_inds]))

# get classification of proteins from k-medoids clustering
k = 25
medoid_inds, labelDict = kMedoids(filtered_GWdists, k)
GW_labels = np.zeros(numfiltered, dtype=int)
for i in range(k):
    GW_labels[labelDict[i]] = i

print('True Labels: ' + str(true_labels[filtered_inds]))
print('True Number of Classes: ' + str(true_k))
print('GW Labels:      ' + str(GW_labels))

randInd = rand_index(true_labels[filtered_inds], GW_labels)
adjRandInd = metrics.adjusted_rand_score(true_labels[filtered_inds], GW_labels)
print('Rand Index: ' + str(randInd))
print('Adjusted Rand Index: ' + str(adjRandInd))


###############################################################################
# Cluster FATCAT Distances and Compute Rand Index of Classification
###############################################################################

# read in FATCAT p-values
fatcat_pvals = pd.read_csv('FATCAT-P-value.csv')
fatcat_dists = np.zeros((numproteins, numproteins))
fatcat_dists[:] = np.nan

rowdomains = fatcat_pvals['Unnamed: 0']
for coldomain in fatcat_pvals.keys():
    if coldomain in domainToCath:
        colind = domainToCath[coldomain][0][0]
        arr = fatcat_pvals[coldomain]
        arrlen = len(arr)
        for i in range(arrlen):
            rowdomain = rowdomains[i]
            rowind = domainToCath[rowdomain][0][0]
            fatcat_dists[rowind, colind] = arr[i]

# symmetrize the FATCAT p-value matrix
i_lower = np.tril_indices(numproteins, -1)
fatcat_dists[i_lower] = fatcat_dists.T[i_lower]

# plot FATCAT distance matrix
plt.figure(3)
plt.title('FATCAT distance matrix', fontweight='bold')
plt.pcolormesh(fatcat_dists)
plt.colorbar()
plt.show()

# filter our nan value from FATCAT p-values
filtered_fatcat_dists, filtered_inds = removeNans(fatcat_dists)
numfiltered = len(filtered_inds)

true_k = len(np.unique(true_labels[filtered_inds]))

# get classification of proteins from k-medoids clustering
k = 24
medoid_inds, labelDict = kMedoids(filtered_fatcat_dists, k)
fatcat_labels = np.zeros(numfiltered, dtype=int)
for i in range(k):
    fatcat_labels[labelDict[i]] = i

print('True Labels: ' + str(true_labels[filtered_inds]))
print('True Number of Classes: ' + str(true_k))
print('FATCAT Labels:      ' + str(fatcat_labels))

randInd = rand_index(true_labels[filtered_inds], fatcat_labels)
adjRandInd = metrics.adjusted_rand_score(true_labels[filtered_inds], fatcat_labels)
print('Rand Index: ' + str(randInd))
print('Adjusted Rand Index: ' + str(adjRandInd))
