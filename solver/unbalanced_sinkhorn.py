import numpy as np
import math

def sinkhorn_log(mu, nu, c, eps, niter=1000, tau=-0.5, rho=math.inf):
    # sinkhorn_log - stabilized sinkhorn over log domain with acceleration
    #
    #   gamma, u, v, Wprimal, Wdual, err = sinkhorn_log(mu, nu, c, eps, niter, tau, rho)
    #
    #   mu and nu are marginals.
    #   c is cost
    #   eps is regularization
    #   coupling is 
    #       gamma = exp( (-c+u*ones(1,N(2))+ones(N(1),1)*v')/epsilon );
    #
    #   niter is the number of iterations.
    #   tau is an avering step. 
    #       - tau=0 is usual sinkhorn
    #       - tau<0 produces extrapolation and can usually accelerate.
    #
    #   rho controls the amount of mass variation. Large value of rho
    #   impose strong constraint on mass conservation. rho=Inf (default)
    #   corresponds to the usual OT (balanced setting). 
    #
    #   Copyright (c) 2016 Gabriel Peyre
    
    lmbda = rho / (rho + eps)
    if rho == math.inf:
        lmbda = 1
    
    m = mu.size
    n = nu.size
    H1 = np.ones(m)
    H2 = np.ones(n)
    
    ave = lambda tau, u, u1: tau * u + (1 - tau) * u1
    
    
    lse = lambda A: np.log(np.sum(np.exp(A), axis=1))
    M = lambda u, v: (-c + np.outer(u, H2) + np.outer(H1, v)) / eps
    
    # kullback divergence
    H = lambda p: -np.sum(p * (np.log(p + 1e-20) - 1))
    KL  = lambda h, p: np.sum(h * np.log(h / p) - h + p)
    KLd = lambda u, p: np.sum(p * (np.exp(-u)-1))
    dotp = lambda x, y: np.sum(x * y)
    
    Wprimal = np.zeros(niter)
    Wdual = np.zeros(niter)
    err = np.zeros(niter)
    
    u = np.zeros(m); 
    v = np.zeros(n);
    for i in range(niter):
        u1 = u;
        u = ave(tau, u, lmbda*eps*np.log(mu) - lmbda*eps*lse(M(u,v)) + lmbda*u);
        v = ave(tau, v, lmbda*eps*np.log(nu) - lmbda*eps*lse(M(u,v).transpose()) + lmbda*v);
        
        # coupling
        gamma = np.exp(M(u,v));
        if rho == math.inf: # marginal violation
            Wprimal[i] = dotp(c, gamma) - eps*H(gamma)
            Wdual[i] = dotp(u, mu) + dotp(v, nu) - eps*np.sum(gamma)
            err[i] = np.linalg.norm(np.sum(gamma, axis=1)-mu)
        else: # difference with previous iterate
            Wprimal[i] = dotp(c, gamma) - eps*H(gamma) + rho*KL(np.sum(gamma, axis=1), mu) + rho*KL(np.sum(gamma, axis=0), nu)
            Wdual[i] = -rho*KLd(u/rho, mu) - rho*KLd(v/rho, nu) - eps*np.sum(gamma)
            err[i] = np.linalg.norm(u - u1, ord=1)

    return gamma, u, v, Wprimal, Wdual, err
