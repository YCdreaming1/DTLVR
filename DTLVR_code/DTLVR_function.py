#!/usr/bin/env python
# coding: utf-8
"""
Created on 6/1/2021
@Author: Chao Yang (YCdreaming)
@Author Affiliation: State Key Laboratory of Synthetical Automation for Process Industries,
Northeastern University, Shenyang 110819, P.R. China.
@ DTLVR_version1.0: Dynamic Transfer Latent Variable Regression written by Numpy
@ If you find any problems, please contact me by Email: yangc1109@163.com
"""

import numpy as np
np.random.seed(5)
import scipy.linalg
import scipy.stats
from scipy.linalg import eigh
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

def DTLVR(x, y, xs, xt, A, l, s,lambda1, heuristic=False):
    """
    Dynamic Transfer Latent Variable Regression (DTLVR) perform quality prediction using labeled Source domain data, x(=xs) and
    y and unlabeled Target domain data (xt) with the goal to fit an (invariant) model that generalizes over both domains.

    Parameters
    __________
    :param x: numpy array (N, K), labeled X dataset
    :param y: numpy array (N, 1), Response variable
    :param xs: numpy array (N_S, K), Source domain data
    :param xt: numpy array (N_T, K), Target domain data
    :param A: int, Number of latent variables
    :param l: int or numpy array (A x 1), regularization parameter: Typically 0<l<1e10 if Array is passed, a different l
    is used for each LV
    :param s: int, window range of lagged time for construct augmented matrix
    :param heuristic: if 'True' the regularization parameter is determined using a heuristic that gives equal weight to:
    i) Fitting the output variable y and ii) Minimizing domain discrepancy.
    :return:

    """

    # Get array dimensions
    (n, k) = np.shape(x)
    (ns, k) = np.shape(xs)
    (nt, k) = np.shape(xt)


    # Number of augmented data
    Ns = ns - s
    Nt = nt - s

    ys =y[s-1:n, :]  # the corresponding label with Ns+1 samples in the source domain
    Zs_s_pls = augmented_matrix(xs, s, Ns)  # the augmented matrix with Ns+1 samples (according to the DiPLS)
    # Construct augmented matrix for the source domain
    Zs_s_pca = augmented_matrix(xs, s, Ns-1)  # the augmented matrix with Ns samples (according to DiPCA)
    #print(Zs_s_pca.shape)
    xs_s_pca = xs[s:ns, :]  # the corresponding augmented vector xs_s+1 with Ns samples
    # Construct augmented matrix for the target domain
    Zt_s_pca = augmented_matrix(xt, s, Nt-1)  # the augmented matrix with Nt samples (according to DiPCA)
    xt_s_pca = xt[s:nt, :] # the corresponding augmented vector xt_s+1 with Nt samples
    
    # Initialize matrices
    "输入变量的权重矩阵"
    W = np.zeros([k, A])

    "输出变量的得分矩阵， 存放输出的隐含变量值"
    U = np.zeros([ns, A])

    "输入变量的得分矩阵，存放输入的隐含变量值"
    T = np.zeros([n, A])
    Ts = np.zeros([ns, A])
    Tt = np.zeros([nt, A])

    "输入负荷矩阵"
    P = np.zeros([k, A])
    Ps = np.zeros([k, A])
    Pt = np.zeros([k, A])

    "输出负荷矩阵"
    Q = np.zeros([A, 1])

    Alpha = np.zeros([s, A])
    opt_l = np.zeros(A)
    discrepancy = np.zeros(A)

    # Initialize Beta with [1,0,...,0].T, and u_s as some column of ys
    beta1 = np.ones([s, 1])/np.sqrt(s)
    # beta1[-1, 0] = 1
    beta2 = np.ones([s, 1])/np.sqrt(s)
    # beta2[-1, 0] = 1
    u_s = ys[:, 0].reshape(ys.shape[0], 1)

    I = np.eye(k)
    # lambda1=-10000

    # Iteration
    for i in range(A):
        if type(I) is np.array:  # Separate regularization params for each LV
            IA = I[i]
        elif (type(I) is np.int or type(I) is np.float64):
            IA = I
        else:
            IA = I[0]
        iterr = 1000
        w = np.random.rand(1, k)
        w = w / np.linalg.norm(w)
        temp = x@w.T
        V_old = 0
        count = 0
        while iterr > 0.000001 and count < 2000:
            # Compute Domain-Invariant Weight Vector 
            #w = (u_s.T @ Zs_s_pls @ np.kron(beta1, np.identity(k)))/(u_s.T @ u_s) # Ordinary DiPLS solution

            # Convex relaxation of covariance difference matrix
            D1 = convex_relaxation(Zs_s_pca, Zs_s_pca, Zt_s_pca, Zt_s_pca, beta2, k, type=1)
            D11 = (1 / np.shape(Zs_s_pca)[0] * (Zs_s_pca @ np.kron(beta1, np.identity(k))).T @ Zs_s_pca @ np.kron(beta1, np.identity(k)) - 1 / np.shape(Zt_s_pca)[0] * (Zt_s_pca @ np.kron(beta1, np.identity(k))).T @ Zt_s_pca @ np.kron(beta1, np.identity(k)))
            D2 = convex_relaxation(Zs_s_pca, xs_s_pca, Zt_s_pca, xt_s_pca, beta2, k, type=2)
            D21 = (1 / np.shape(Zs_s_pca)[0] * xs_s_pca.T @ Zs_s_pca @ np.kron(beta2, np.identity(k)) + 1 / np.shape(Zt_s_pca)[0] * xt_s_pca.T @ Zt_s_pca @ np.kron(beta2, np.identity(k))) #领域间共享动态信息
            # D3 = convex_relaxation(xs_s_pca, xs_s_pca, xt_s_pca, xt_s_pca, beta1, k, type=3)     #此处beta不起作用
            D3 = (1 / np.shape(xs_s_pca)[0] * xs_s_pca.T @ xs_s_pca - 1 / np.shape(xt_s_pca)[0] * xt_s_pca.T @ xt_s_pca)
            D31 = (1 / np.shape(xs_s_pca)[0] * xs_s_pca.T @ xs_s_pca + 1 / np.shape(xt_s_pca)[0] * xt_s_pca.T @ xt_s_pca)
            D4 = (1 / np.shape(Zs_s_pca)[0] * (Zs_s_pca @ np.kron(beta2, np.identity(k))).T @ Zs_s_pca @ np.kron(beta2, np.identity(k)) + 1 / np.shape(Zt_s_pca)[0] * (Zt_s_pca @ np.kron(beta2, np.identity(k))).T @ Zt_s_pca @ np.kron(beta2, np.identity(k)))
            # D4 = convex_relaxation(Zs_s_pca, Zs_s_pca, Zt_s_pca, Zt_s_pca, beta2, k, type=1)
            # D5 = (1 / np.shape(Zs_s_pca)[0] * (Zs_s_pca @ np.kron(beta1, np.identity(k))).T @ Zs_s_pca @ np.kron(beta1,np.identity(k)))
            # D = (D31-D21+D4+D11)
            #+D11 @ w.T @ w @ D11
            # D = (D31 + D4) - D21 + D11 @ W @ W.T @ D11
            D = -3*lambda1*(D31+D4)-D21+D11@ W @ W.T@ D11


            w = (u_s.T @ Zs_s_pls @ np.kron(beta1, np.identity(k))) / (u_s.T @ u_s)


            if (heuristic is True):  #Regularization parameter heuristic

                w = w/np.linalg.norm(w)
                gamma = (np.linalg.norm(Zs_s_pls @ np.kron(beta1, np.identity(k))-u_s @ w)**2)/(w @ D @ w.T)
                opt_l[i] = gamma
                IA = gamma


            # Compute DTLVR weight vector
            # reg = (np.linalg.inv(I+IA/((u_s.T @ u_s))*D))
            # w = w_dipls@reg
            reg = I+IA/((u_s.T @ u_s))*D
            w = scipy.linalg.solve(reg.T, w.T, assume_a='sym').T
            #w = w_dipls@np.linalg.inv(reg)

            # Normalize w
            w = w/np.linalg.norm(w)


            # Absolute difference between variance of source and target domain projections
            discrepancy[i] = w@D@w.T
            # print(discrepancy)


            # Compute scores
            t = x@w.T
            ts = xs@w.T
            tt = xt@w.T
            T_s = t[:Ns+1, :]
            Ts_s = ts[:Ns+1, :]
            Ts_s_pca = ts[:Ns, :]
            Tt_s = tt[:Nt+1, :]
            Tt_s_pca = tt[:Nt, :]
            for j in range(1, s):
                T_s = np.hstack((t[j:Ns+j+1, :], T_s))
                Ts_s = np.hstack((ts[j:Ns+j+1, :], Ts_s))
                Ts_s_pca = np.hstack((ts[j:Ns+j, :], Ts_s_pca))
                Tt_s = np.hstack((tt[j:Nt+j+1, :], Tt_s))
                Tt_s_pca = np.hstack((tt[j:Nt+j, :], Tt_s_pca))
            q = ys.T@Zs_s_pls@np.kron(beta1, w.T)
            q = q/np.linalg.norm(q)
            u_s = ys@q
            beta1 = np.linalg.inv(np.square(T_s.T @ T_s-Tt_s.T@Tt_s)+lambda1 * np.identity(s))@T_s.T@u_s
            # beta1 = np.linalg.inv(T_s.T @ T_s)@T_s.T@u_s
            beta1 = beta1/np.linalg.norm(beta1)
            # print('beta1:', beta1)
            # beta2 = Ts_s_pca.T @ ts[s:Ns + s, :] + Tt_s_pca.T @ tt[s:Nt + s, :]
            n1 = Ts_s_pca.shape[0]
            n2 = Tt_s_pca.shape[0]
            if n1<n2:
                beta2 = 1/lambda1*np.linalg.inv(np.identity(s)-3*(Ts_s_pca.T@Ts_s_pca+Tt_s_pca[:n1,:].T@Tt_s_pca[:n1,:]))@(Ts_s_pca.T@ts[s:Ns+s, :] + Tt_s_pca[:n1,:].T@tt[s:Ns+s, :])
            elif n1>n2:
                beta2 = 1/lambda1*np.linalg.inv(np.identity(s)-3*(Ts_s_pca[:n2,:].T@Ts_s_pca[:n2,:]+Tt_s_pca.T@Tt_s_pca))@(Ts_s_pca[:n2,:].T@ts[s:Nt + s, :]+Tt_s_pca.T@tt[s:Nt + s, :])
            else:
                beta2 = 1/lambda1*np.linalg.inv(np.identity(s)-3*(Ts_s_pca.T@Ts_s_pca+Tt_s_pca.T@Tt_s_pca))@(Ts_s_pca.T@ts[s:Ns+s, :] + Tt_s_pca.T@tt[s:Ns+s, :])
            beta2 = beta2/np.linalg.norm(beta2)
            # print('beta2:', beta2)
            iterr = np.linalg.norm(t-temp, ord=2)
            V = q.T @ ys.T @ Zs_s_pls @ np.kron(beta1, w.T)
            iterr = np.linalg.norm(V - V_old)
            temp = t
            count +=1

        # Inner model building, Build a linear model
        alpha = np.linalg.inv(T_s.T@T_s)@T_s.T@u_s
        u_s_hat = T_s@alpha
        # print("alpha:",alpha)
        print('beta1:', beta1)
        print('beta2:', beta2)


        # Compute loadings
        p = x.T@t/(t.T@t)
        ps = xs.T@ts/(ts.T@ts)
        pt = xt.T@tt/(tt.T@tt)
        alpha = alpha / np.linalg.norm(ps, ord=2)

        # Deflate x and y (Gram-Schmit orthogonalization)
        x = x-t@p.T
        xs = xs-ts@ps.T
        xt = xt-tt@pt.T
        q = ys.T @ u_s_hat @ np.linalg.inv(u_s_hat.T @ u_s_hat)
        q = q / np.linalg.norm(q, ord=2)
        ys = ys-u_s_hat@q.T

        # Store parameters
        W[:,i] = w
        T[:,i] = t.reshape(n)
        Ts[:,i] = ts.reshape(ns)
        Tt[:,i] = tt.reshape(nt)
        P[:,i] = p.reshape(k)
        Ps[:,i] = ps.reshape(k)
        Pt[:,i] = pt.reshape(k)
        #print(q.shape)
        Q[i] = q
        Alpha[:,i] = alpha.reshape(s)

    # Calculate regression vector
    b_R = W@(np.linalg.inv(P.T@W))

    # Store residuals
    E = x
    Es = xs
    Et = xt
    Ey = ys

    return b_R, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, Q, opt_l, Alpha, discrepancy, beta1, beta2


def convex_relaxation(xs, xs_lag, xt, xt_lag, beta, k, type=1):
    """
    Convex relaxation of covariance difference.
    The convex relaxation computes the eigenvalue decomposition of the (symetric) covariance
    difference matrix, inverts the signs of the negative eigenvalues and reconstructs it again.
    It can be shown that this relaxation corresponds to an upper bound on the covariance difference
    btw. source and target domain (see ref.)
    Parameters
    __________
    :param xs: numpy array ()
    :param xs_1:
    :param xt:
    :param xt_1:
    :return:
    """

    # Preliminaries
    ns = np.shape(xs)[0]
    nt = np.shape(xt)[0]
    x = np.vstack([xs, xt])
    # x_lag = np.vstack([xs_lag, xt_lag])
    # xs = xs[..., :] - np.mean(x, 0)
    # xt = xt[..., :] - np.mean(x, 0)
    # xs_lag = xs_lag[..., :] - np.mean(x_lag, 0)
    # xt_lag = xt_lag[..., :] - np.mean(x_lag, 0)

    # Compute difference between source and target covariance  matrices
    #rot = (1/ns*xs.T@xs_lag - 1/nt*xt.T@xt_lag)
    #rot = (1/ns*xs_lag.T@xs@np.kron(beta, np.identity(k))-1/nt*xt_lag.T@xt@np.kron(beta, np.identity(k)))
    if type ==1:
        rot = (1 / ns * (xs_lag@np.kron(beta, np.identity(k))).T @ xs @ np.kron(beta, np.identity(k)) - 1 / nt * (xt_lag@np.kron(beta, np.identity(k))).T @ xt @ np.kron(beta, np.identity(k)))
    elif type ==2:
        # a = (1 / ns * xs_lag.T @ xs @ np.kron(beta, np.identity(k))).T
        # b = (1 / nt * xt_lag.T @ xt @ np.kron(beta, np.identity(k))).T
        rot = (1 / ns * xs_lag.T @ xs @ np.kron(beta, np.identity(k)) - 1 / nt * xt_lag.T @ xt @ np.kron(beta, np.identity(k)))
    else:
        rot = (1/ns*xs.T@xs_lag - 1/nt*xt.T@xt_lag)


    # Convex Relaxation
    w, v = eigh(rot)
    eigs = np.abs(w)
    eigs = np.diag(eigs)
    D = v@eigs@v.T
    return D


def augmented_matrix(data, s, N):
    """
    Construct Augmented Matrix for feature data Z_s
    :param data: Original data
    :param s: int, lagged time 
    :param N: int, Number of augmented samples
    :return:
    Z_s: numpy array, Augmented data matrices.
    """
    X = []
    for i in range(s):
        X.append(data[i:i+N+1, :])
    Z_s = X[s-1]
    for i in range(s-2, -1, -1):
        Z_s = np.hstack((Z_s, X[i]))
    return Z_s

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
    return np.sqrt(((y - yhat) ** 2).mean())















                





























