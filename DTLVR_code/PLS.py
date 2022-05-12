#!/usr/bin/env python
# coding: utf-8
"""
Created on 5/1/2021
@Author: Chao Yang (YCdreaming)
@Author Affiliation: State Key Laboratory of Synthetical Automation for Process Industries,
Northeastern University, Shenyang 110819, P.R. China.
@PLS_version1.0: Partial Least Square written by Numpy
"""

import numpy as np

class PLS():
    def __init__(self, l):
        "p: Latent Variables"
        self.l = l

    "PLS Training"
    def train(self, X, Y):
        l = self.l

        Num_DataX = X.shape[0]
        Num_DataY = Y.shape[0]
        Xdim = X.shape[1]
        Ydim = Y.shape[1]

        "输出变量的得分矩阵，存放输出的隐含变量值"
        U = np.zeros((Num_DataX, l))

        "输入变量的得分矩阵，存放输入的隐含变量值"
        T = np.zeros((Num_DataX, l))

        "输入变量的权重矩阵"
        W = np.zeros((Xdim, l))

        "输入负荷矩阵"
        P = np.zeros((Xdim, l))

        "标准化数据的对应模型的回归系数向量"
        B = np.zeros((1, l))

        "输出负荷矩阵"
        Q = np.zeros((1, l))

        if Num_DataX != Num_DataY:
            print('error: The number of input samples is not equal to that of output samples')

        u0 = Y
        e0 = 100
        # Iteration
        k = 0
        while k < l:
            u = u0
            e0 = 100
            number = 0
            w = np.zeros((Xdim, 1))
            while e0 > 0.000001 and number < 2000:
                w = X.T @ u / (u.T @ u)
                w = w / np.linalg.norm(w, ord=2)
                t = X @ w
                q = Y.T @ t / (t.T @ t)
                q = q / np.linalg.norm(q, ord=2)
                u = Y @ q / (q.T @ q)
                e0 = np.linalg.norm(u - u0)
                number += 1

            # Deflation
            p = X.T @ t / (t.T @ t)
            p = p / np.linalg.norm(p, ord=2)
            b = u.T @ t / (t.T @ t)
            X = X - t @ p.T
            Y = Y - b * t @ q.T
            u0 = Y

            # Parameters
            W[:,k] = w[:,0]
            T[:,k] = t[:,0]
            P[:,k] = p[:,0]
            Q[:,k] = q[:,0]
            B[:,k] = b[:,0]
            U[:,k] = u[:,0]

            k = k + 1
        params = {'W': W, 'T': T, 'P': P, 'Q': Q, 'B': B, 'U': U}
        return params

    "PLS Predicting"
    def predict(self, params, X):
        W = params['W']
        T = params['T']
        P = params['P']
        Q = params['Q']
        B = params['B']
        U = params['U']

        R = W @ np.linalg.inv(P.T @ W)
        T = X @ R
        Y = T * np.diag(B) @ Q.T
        return Y















