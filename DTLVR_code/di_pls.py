#!/usr/bin/env python
# coding: utf-8
"""
Created on 6/1/2021
@Author: Chao Yang (YCdreaming)
@Author Affiliation: State Key Laboratory of Synthetical Automation for Process Industries,
Northeastern University, Shenyang 110819, P.R. China.
@ DIPLS_version1.0: Domain Invariant Partial Least Square written by Numpy
"""
import numpy as np

class di_pls():
    def __init__(self, gamma, lv):
        "lambda: regularization parameters for domain invariant"
        "lv: projected latent variables"
        self.gamma = gamma
        self.lv = lv
        
    "DIPLS Training"
    def train(self, X, Y, Xs, Xt):
        gamma = self.gamma
        lv = self.lv
        
        # Num_X: the number of X samples, Number_X_s: the number of X_s samples, Num_X_t: the number of X_t samples, 
        # Num_X is equal to Num_X_s.
        # X_dim: the dimensions of X, X_s_dim: the dimensions of X_s, X_t_dim: the dimensions of X_t. In fact, X_dim, 
        # X_s_dim and X_t_dim are all equal.
        
        Num_X = X.shape[0]
        Num_Xs = Xs.shape[0]
        Num_Xt = Xt.shape[0]
        Num_Y = Y.shape[0]
        X_dim = X.shape[1]
        Xs_dim = Xs.shape[1]
        Xt_dim = Xt.shape[1]

        "输出变量的得分矩阵，存放输出的隐含变量值"
        U = np.zeros((Num_X, lv))

        "输入变量的得分矩阵，存放输入的隐含变量值"
        T = np.zeros((Num_X, lv))
        Ts = np.zeros((Num_Xs, lv))
        Tt = np.zeros((Num_Xt, lv))
        
        "输入变量的权重矩阵"
        W = np.zeros((X_dim, lv))
        
        "输入负荷矩阵"
        P = np.zeros((X_dim, lv))
        Ps = np.zeros((Xs_dim, lv))
        Pt = np.zeros((Xt_dim, lv))

        "标准化数据的对应模型的回归系数向量"
        B = np.zeros((1, lv))
        
        "输出负荷矩阵"
        Q = np.zeros((1, lv))
        
        if Num_X != Num_Y:
            print("error: The number of input samples is not equal to that of output samples")
        if Num_Xs != Num_X:
            print("error: Num_X should be equal to Num_Xs")
        
        u0 = Y
        e0 = 100
        # Iteration
        k = 0
        while k < lv:
            u = u0
            e0 = 100
            number = 0
            w = np.zeros((1, X_dim))
            while e0 > 0.0000001 and number < 2000:
                w = (u.T @ X / (u.T @ u)) @ np.linalg.inv(np.ones((X_dim, X_dim)) + gamma / (2 * u.T @ u) * (Xs.T @ Xs / (Num_Xs - 1) - Xt.T @ Xt / (Num_Xt - 1)))
                w = w.T
                w = w / np.linalg.norm(w, ord=2)
                t = X @ w
                t_s = Xs @ w
                t_t = Xt @ w
                q = Y.T @ t / (t.T @ t)
                q = q / np.linalg.norm(q, ord=2)
                u = Y @ q / (q.T @ q)
                e0 = np.linalg.norm(u-u0)
                number += 1

            # Deflation
            p = X.T @ t / (t.T @ t)
            p = p / np.linalg.norm(p, ord=2)
            ps = Xs.T @ t_s / (t_s.T @ t_s)
            ps = ps / np.linalg.norm(ps, ord=2)
            pt = Xt.T @ t_t / (t_t.T @ t_t)
            pt = pt / np.linalg.norm(pt, ord=2)
            b = u.T @ t / (t.T @ t)
            X = X - t @ p.T
            Xs = Xs - t_s @ ps.T
            Xt = Xt - t_t @ pt.T
            Y = Y - b * t @ q.T
            u0 = Y

            # Parameters
            W[:,k] = w[:, 0]
            T[:,k] = t[:, 0]
            Ts[:,k] = t_s[:, 0]
            Tt[:,k] = t_t[:, 0]
            P[:,k] = p[:, 0]
            Ps[:,k] = ps[:, 0]
            Pt[:,k] = pt[:, 0]
            Q[:,k] = q[:, 0]
            B[:,k] = b[:, 0]
            U[:,k] = u[:, 0]

            k = k + 1
        params = {'W': W, 'T': T, 'Ts': Ts, 'Tt': Tt, 'P': P, 'Ps': Ps, 'Pt': Pt, 'Q': Q, 'B': B, 'U': U}
        return params

    "DIPLS Predicting"
    def predict(self, params, X):
        W = params['W']
        T = params['T']
        Ts = params['Ts']
        Tt = params['Tt']
        P = params['P']
        Ps = params['Ps']
        Pt = params['Pt']
        Q = params['Q']
        B = params['B']
        U = params['U']

        R = W @ np.linalg.pinv(P.T @ W)
        # print(W)
        #print(R)
        T = X @ R
        Y = T * np.diag(B) @ Q.T
        return Y



            
        
        
        
            
        
        
    
