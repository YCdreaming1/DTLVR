#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import scipy.io as scio
import matplotlib.pyplot as plt

# s: denote time lag
# p: denote the dimension of feature after projected


class DiPLS():
    def __init__(self, s, p):
        self.s = s
        self.p = p
        self.model = None
        
    def train(self, X, Y):
        s = self.s
        p = self.p

        m = X.shape[0]
        n_x = X.shape[1]
        n_y = Y.shape[1]
        N = m - s - 1
        #constract augmented matrix
        Y_s = Y[s:m, :]
        X_i = []
        for i in range(s+1):
            X_i.append(X[i:i+N+1, :])
        Z_s = X_i[s]
        for i in range(s-1, -1, -1):
            Z_s = np.hstack((Z_s, X_i[i]))
        # initialization
        W = np.zeros((n_x, p))
        T = np.zeros((m, p))
        P = np.zeros((n_x, p))
        Q = np.zeros((n_y, p))
        Alpha = np.zeros((s+1, p))
        # Initialize Beta with [1, 0, ..., 0].T, and u_s as some column of Y_s
        beta = np.zeros((s+1, 1))
        beta[0,0] = 1
        u_s = Y_s[:, 0].reshape(Y_s.shape[0], 1)
        
        # Iteration
        l = 0
        while l < p:
            iterr = 1000
            w = np.ones([n_x, 1])
            w = w / np.linalg.norm(w,ord=2)
            temp = X @ w
            V_old = 0
            count = 0
            while iterr > 0.000001 and count < 2000:
                w = np.kron(beta, np.identity(n_x)).T @ Z_s.T @ u_s
                w = w / np.linalg.norm(w, ord=2)
                t = X @ w
                T_s = t[0:N+1, :]
                for i in range(1, s + 1):
                    T_s = np.hstack((t[i:N+i+1, :], T_s))
                q = Y_s.T @ Z_s @ np.kron(beta, w)
                q = q / np.linalg.norm(q, ord = 2)
                u_s = Y_s @ q
                beta = T_s.T @ u_s
                beta = beta / np.linalg.norm(beta, ord = 2)
                iterr = np.linalg.norm(t - temp, ord = 2)
                V = q.T@Y_s.T@Z_s@np.kron(beta,w)
                iterr = np.linalg.norm(V - V_old, ord=2)
                V_old = V
                count += 1
            # Inner model building.Build a linear model between
            alpha = np.linalg.inv(T_s.T @ T_s) @ T_s.T @ u_s
            u_s_hat = T_s @ alpha
                
            # Deflation
            p1 = X.T @ t / (t.T @ t)
            X = X - t @ p1.T
            Y_s = Y_s - u_s_hat @ q.T
            
            # Parameters
            W[:, l] = w[:, 0]
            T[:, l] = t[:, 0]
            P[:, l] = p1[:, 0]
            Q[:, l] = q[:, 0]
            Alpha[:, l] = alpha[:, 0]
            # print(W)
            
            l = l + 1
        params = {'W': W, 'T': T, 'P': P, 'Q': Q, 'Alpha': Alpha, 'n_y': n_y,'beta':beta}
        return params
    def predict(self, params, X):
        W = params['W']
        T = params['T']
        P = params['P']
        Q = params['Q']
        Alpha = params['Alpha']
        n_y = params['n_y'] 
        s = self.s
        m = X.shape[0]
        n_x = X.shape[1]
        N = m - s - 1
        U_s = np.zeros((N + 1, T.shape[1]))
        R = W @ np.linalg.pinv(P.T @ W)
        T = X @ R
        for i in range(T.shape[1]):
            t = T[:, i].reshape(T.shape[0], 1)
            T_s = t[:N+1, :]
            for j in range(1, s + 1):
                T_s = np.hstack((t[j:N+j+1, :], T_s))
            u_s = T_s @ Alpha[:, i]
            U_s[:, i] = u_s
        Y = U_s @ Q.T
        
        return Y























