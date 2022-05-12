#!/usr/bin/env python
# coding: utf-8
"""
Created on 5/1/2021
@Author: Chao Yang (YCdreaming)
@Author Affiliation: State Key Laboratory of Synthetical Automation for Process Industries,
Northeastern University, Shenyang 110819, P.R. China.
@ dipls_version1.0: Domain Invariant Partial Least Square written by Numpy

The Algorithm proposed by @Author: Dr. Ramin Nikzad-Langerodi
Bottleneck Analytics GmbH
info@bottleneck-analytics.com
"""

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.linalg import eigh
import scipy.spatial.distance as scd
from scipy.spatial import distance_matrix
import warnings
warnings.filterwarnings("ignore")


def dipals(x, y, xs, xt, A, l, heuristic=False):
    """
    Domain-invariant partial least square regression(di-pls) perform PLS regression using labeled Source domain data,
    x(=xs) and y and unlabeled Target domain data(xt) with the goal to fit an (invariant) model that generalizes over
    both domains.

    References:
    (1) Ramin Nikzad-Langerodi, Werner Zellinger, Edwin Lughofer, and Susanner Saminger-Platz "Domain-Invariant Partial
    least Squares Regression" Analytical Chemistry 2018 90（11）， 6639-6701 DOI：10.1021/acs.analchem.8b00498
    (2) Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz and Bernhard Moser "Domain-Invariant Regression
    under Beer-Lambert's Law" In Proc. International Conference on Machine Learning and Applications, Boca Raton FL 2019.
    (3) Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz, Bernhard A. Moser, Domain adaptation for
    regression under Beer–Lambert’s law, Knowledge-Based Systems, 2020 (210) DOI: 10.1016/j.knosys.2020.106447.

    Parameters
    __________
    :param x: numpy array (N,K), labeled X data
    :param y: numpy array (N,1), Response variable
    :param xs: numpy array (N_S,K), Source domain data
    :param xt: numpy array (N_T,K), Target domain data
    :param A:  int, Number of latent variables
    :param l:  int or numpy array (Ax1), Regularization parameter: Typically 0<l<1e10 if Array is passed, a different l
    is used for each LV
    :param heuristic: if 'True' the regularization parameter is determined using a heuristic that gives equal weight to:
    i) Fitting the output variable y and ii) Minimizing domain discrepancy.
    :return:
    b:numpy array (K,1), Regression vector
    b0: int, Offset (Note: yhat=b0+x*b)
    T: numpy array (N, A), Training data Projections (scores)
    Ts: Source domain Projections (scores)
    Tt: Target domain Projections (scores)
    W: numpy array (K, A), Weight vector
    P: numpy array (K, A), Loadings vector
    E: numpy array (N_S, K), Residual of labeled X data
    Es: numpy array (N_S, K), Source domain residual matrix
    Et: numpy array (N_t, K), Target domain residual matrix
    Ey: numpy array (N_S, 1), Response variable residuals
    C: numpy array (A, 1), Regression vector, such that y = Ts*C
    opt_l: numpy array (A, 1), The heuristically determined regularization parameter for each LV (if heuristic='True')
    discrepancy: numpy array (A, ), Absolute difference between variance of source and target domain projections
    """

    # Get array dimensions
    (n, k) = np.shape(x)
    (ns, k) = np.shape(xs)
    (nt, k) = np.shape(xt)

    #Initialize matrices
    T = np.zeros([n, A])
    P = np.zeros([k, A])

    Tt = np.zeros([nt, A])
    Pt = np.zeros([k, A])

    Ts = np.zeros([ns, A])
    Ps = np.zeros([k, A])

    W = np.zeros([k, A])
    C = np.zeros([A, 1])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)

    I = np.eye(k)

    # Compute LVs
    for i in range(A):
        if type(I) is np.array: #Separate regularization params for each LV
            IA = I[i]
        elif (type(I) is np.int or type(I) is np.float64):
            IA = I
        else:
            IA = I[0]
        # Compute Domain-Invariant Weight Vector
        w_pls = ((y.T@x)/(y.T@y)) # Ordinary PLS solution

        # Convex relaxation of covariance difference matrix
        D = convex_relaxation(xs, xt)

        if (heuristic is True): #Regularization parameter heuristic

            w_pls = w_pls/np.linalg.norm(w_pls)
            gamma = (np.linalg.norm((x-y@w_pls))**2)/(w_pls@D@w_pls.T)
            opt_l[i] = gamma
            IA = gamma

        # Compute di-PLS weight vector
        # reg = (np.linalg.inv(I+IA/((y.T@y))*D))
        # w = w_pls@reg
        reg = I+IA/((y.T@y))*D
        w = scipy.linalg.solve(reg.T, w_pls.T, assume_a='sym').T #~10 times faster

        # Normalize w
        w = w/np.linalg.norm(w)

        # Absolute difference between variance of source and target domain projections
        discrepancy[i] = w@D@w.T

        # Compute scores
        t = x@w.T
        ts = x@w.T
        tt = xt@w.T

        # Regress y on t
        c = (y.T@t)/(t.T@t)

        # Compute loadings
        p = (t.T@x)/(t.T@t)
        ps = (ts.T@xs)/(ts.T@ts)
        pt = (tt.T@xt)/(tt.T@tt)

        # Deflate X and y (Gram-Schmit orthogonalization)
        x = x-t@p
        xs = xs-ts@ps
        xt = xt-tt@pt
        y = y-t*c

        # Store w, t, p, c
        W[:,i] = w
        T[:,i] = t.reshape(n)
        Ts[:,i] = ts.reshape(ns)
        Tt[:,i] = tt.reshape(nt)
        P[:,i] = p.reshape(k)
        Ps[:,i] = ps.reshape(k)
        Pt[:,i] = pt.reshape(k)
        C[i] = c

    # Calculate regression vector
    b = W@(np.linalg.inv(P.T@W))@C

    # Store residuals
    E = x
    Es = xs
    Et = xt
    Ey = y

    return b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy


def convex_relaxation(xs, xt):
    """
    Convex relaxation of covariance difference.
    The convex relaxation computes the eigenvalue decomposition of the (symetric) covariance
    difference matrix, inverts the signs of the negative eigenvalues and reconstructs it again.
    It can be shown that this relaxation corresponds to an upper bound on the covariance difference
    btw. source and target domain (see ref.)

    Reference:
    *Ramin Nikzad-Langerodi, Werner Zellinger, Susanne Saminger-Platz and Bernhard Moser
    "Domain-Invariant Regression under Beer-Lambert's Law" In Proc. International Conference
    on Machine Learning and Applications, Boca Raton FL 2019.

    Parameters
    __________

    :param xs: numpy array (Ns x k), Source domain matrix
    :param xt: numpy array (Nt x k), Target domain matrix
    :return:
    D: numpy array (k x k), Covariance difference matrix
    """

    # Preliminaries
    ns = np.shape(xs)[0]
    nt = np.shape(xt)[0]
    x = np.vstack([xs, xt])
    x = x[..., :] - np.mean(x, 0)

    # Compute difference between source and target covariance matrices
    rot = (1/ns*xs.T@xs - 1/nt*xt.T@xt)

    # Convex Relaxation
    w, v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v@eigs@v.T
    return D

def gengaus(length, mu, sigma, mag, noise=0):
    """
    Generate a spectrum-like Gaussian signal with random noise
    Params
    ______

    :param length: int, length of the signal (i.e., number of variables)
    :param mu: float, Mean of the Gaussian signal
    :param sigma: float, Standard deviation of the Gaussian
    :param mag: float, Magnitude of the signal
    :param noise: float, Amount of i.i.d noise
    :return:
    signal: numpy array(length x 1), Generated Gaussian signal
    """

    s = mag * scipy.stats.norm.pdf(np.arange(length), mu, sigma)
    n = noise * np.random.rand(length)
    signal = s + n
    return signal

def hellipse(X, alpha=0.05):
    """
    95% Confidence intervals for 2D Scatter Plots
    Parameters
    _________
    :param X: numpy array (N x 2), Scores Matrix
    :param alpha: float, Confidence level (default=0.05)
    :return:
    el: numpy array (2 x 100), x,y coordinates of ellipse points arranged in rows. To plot use plt.plot(el[0,:],el[1,:])
    """
    # Means
    mean_all = np.zeros((2, 1))
    mean_all[0] = np.mean(X[:, 0])
    mean_all[1] = np.mean(X[:, 1])

    # Covariance matrix
    X = X[:, :2]
    comat_all = np.cov(np.transpose(X))

    # SVD
    U, S, V = np.linalg.svd(comat_all)

    # Confidence limit computed as the 95% quantile of the F-Distribution
    N = np.shape(X)[0]
    quant = 1 - alpha
    Conf = (2*(N-2))/(N-2)*scipy.stats.f.ppf(quant, 2, (N-2))

    # Evalute Cl on (0, 2pi)
    el = np.zeros((2, 100))
    t = np.linspace(0, 2*np.pi, 100)
    for j in np.arange(100):
        sT = np.matmul(U, np.diag(np.sqrt(S*Conf)))
        el[:, j] = np.transpose(mean_all) + np.matmul(sT, np.array([np.cos(t[j]), np.sin(t[j])]))
    return el

def rmse(y, yhat):
    """
    Root mean squared error
    Parameters
    _________
    :param y: numpy array (N, 1), Measured Y
    :param yhat: numpy array (N, 1), Predicted Y
    :return:
    int, The Root Means Squared Error
    """
    return np.sqrt(((y-yhat)**2).mean())









