import numpy as np
import math
import torch

import matplotlib.pyplot as plt

import ot
import ot.plot

from scipy.io import loadmat
from scipy.spatial import distance
from scipy.stats import wishart

from solver.unbalanced_sinkhorn import sinkhorn_log
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
    cathToDomain = {}
    for i in range(n):
        domain = domains[i][0]
        cath = cath_codes[i][0]
        if domain in domainToCath:
            domainToCath[domain].append((i, cath))
        else:
            domainToCath[domain] = [(i, cath)]
            
        if cath in cathToDomain:
            cathToDomain[cath].append((i, domain))
        else:
            cathToDomain[cath] = [(i, domain)]
    
    return domainToCath, cathToDomain


###############################################################################
# Load in protein atom positions/distances
###############################################################################

#file = open('cathcodes.txt', 'r')
#codes = readProteinCodes(file)

# Get protein geodseic distances
mat0 = loadmat('ProteinData0.mat')
mat1 = loadmat('ProteinData1.mat')
mat2 = loadmat('ProteinData2.mat')
mat3 = loadmat('ProteinData3.mat')
mat4 = loadmat('ProteinData4.mat')

protein_data = np.concatenate([mat0['X_0'], mat1['X_1'], mat2['X_2'], mat3['X_3'], mat4['X_4']])
geodists = protein_data[:, 0]
domains = protein_data[:, 1]
cath_codes = protein_data[:, 2]

# make dictionary of CATH code to domain
domainToCath, cathToDomain = makeCodeDicts(protein_data[:, 1], protein_data[:, 2])

# choose two proteins to compare
# very similar
ind1 = 16
ind2 = 17

# very different
#ind1 = 2
#ind2 = 166

dom1 = domains[ind1][0]
dom2 = domains[ind2][0]
cath1 = cath_codes[ind1][0]
cath2 = cath_codes[ind2][0]
D1 = geodists[ind1]
D2 = geodists[ind2]

plt.figure(1)
plt.title('Protein 1 Geodesic Distances \n (Domain: ' + dom1 + ', CATH: ' + cath1 + ')', fontweight='bold')
plt.pcolormesh(geodists[ind1])
plt.colorbar()

plt.figure(2)
plt.title('Protein 2 Geodesic Distances \n (Domain: ' + dom2 + ', CATH: ' + cath2 + ')', fontweight='bold')
plt.pcolormesh(geodists[ind2])
plt.colorbar()

# number of atoms in each protein
num_atoms1 = D1.shape[0]
num_atoms2 = D2.shape[0]


###############################################################################
# Run Unbalanced Gromov-Wasserstein
###############################################################################

# empirical distributions over atoms
mu1 = np.ones(num_atoms1) / num_atoms1
mu2 = np.ones(num_atoms2) / num_atoms2

# put higher weight on pairings which you want to discourage (not a necessary step)
#mask = np.zeros((num_atoms1, num_atoms2))
#plt.figure(2)
#plt.title('Mask', fontweight='bold')
#plt.pcolormesh(np.log(mask))
#plt.colorbar()


# Parameters
rho = 1e2      # Peyre et al. mass constraint parameter
ent = 1e-1    # entropic regularization, controls width of coupling band
#eta = 1e-1  # learning rate of Solomon et al. GW implementation above


# Apply one of the GW methods below

# POT GW
#coupling = ot.gromov.gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss')

# POT entropic GW
coupling = ot.gromov.entropic_gromov_wasserstein(D1, D2, mu1, mu2, 'square_loss', ent)

# POT partial GW (only transfer a fixed m amount of mass)
#coupling = ot.partial.partial_gromov_wasserstein(D1, D2, mu1, mu2, m=0.5)

# Solomon et al. GW
#gamma = GromovWasserstein(mu1, mu2, D1, D2, ent, eta, thresh=1e-4)
#coupling = gamma * np.outer(mu1, mu2)

# Solomon et al. unbalanced GW
#lambda1 = 0.1
#lambda2 = 0.1
#gamma = UnbalancedGromovWasserstein(mu1, mu2, D1, D2, ent, lambda1, lambda2, eta, thresh=1e-6)
#coupling = gamma * np.outer(mu1, mu2)

# Peyre et al. unbalanced GW
#solver = TLBSinkhornSolver(nits=500, nits_sinkhorn=1000, tol=1e-3, tol_sinkhorn=1e-5)
#coupling, gamma = solver.tlb_sinkhorn(torch.from_numpy(mu1), torch.from_numpy(D1),
#                                      torch.from_numpy(mu2), torch.from_numpy(D2), rho, ent)
#coupling = coupling.numpy()

# post-process the coupling
#thresh = np.max(coupling)
#coupling = coupling / np.sum(coupling) # normalize

# plot inferred coupling
plt.figure(3)
plt.title('Inferred Coupling Between \n Protein 1 (Domain: ' + dom1 + ', CATH: ' + cath1 + ') & \n Protein 2 (Domain: ' + dom2 + ', CATH: ' + cath2 + ')\n', fontweight='bold')
plt.pcolormesh(coupling)#, vmin = 0, vmax = thresh)
plt.xlabel('atoms in protein 2')
plt.ylabel('atoms in protein 1')
plt.colorbar()
#plt.savefig('inferred_coupling.png', dpi=300)
plt.show()

# mass contained in coupling
mass = np.sum(coupling)
print('Coupling Mass: ' + str(mass))

# compute GW cost for inferred coupling
constD, hD1, hD2 = ot.gromov.init_matrix(D1, D2, mu1, mu2, loss_fun='square_loss')
cost = ot.gromov.gwloss(constD, hD1, hD2, coupling)
cost = cost / (np.max(D1) * np.max(D2) * mass**2)
print('Adjusted GW Cost: ' + str(cost))
