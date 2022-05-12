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
import matplotlib.pyplot as plt
import scipy.stats
import DTLVR_function as DF
np.random.seed(5)

class model():
    def __init__(self, x, y, xs, xt, A, lambda1, s):
        self.s = s
        self.x = x
        self.lambda1=lambda1
        self.n = np.shape(x)[0]
        self.ns = np.shape(xs)[0]
        self.nt = np.shape(xt)[0]
        self.k = np.shape(x)[1]
        self.y = y
        self.xs = xs
        self.xt = xt
        self.mu = np.mean(x, 0)
        self.mu_s = np.mean(xs, 0)
        self.mu_t = np.mean(xt, 0)
        self.T = []
        self.Ts = []
        self.Tt = []
        self.P =[]
        self.Ps = []
        self.Pt = []
        self.W = []
        self.A = A
        self.opt_l = []
        self.b0 = np.mean(y, 0)
        self.b_R = []
        self.yhat = []
        self.rmse = []
        self.Q = []
        self.E = []
        self.Es = []
        self.Et = []
        self.Ey = []
        self.Alpha = []
        self.discrepancy = []
        self.beta1=[]
        self.beta2=[]

    def train(self, l=0, centering=True, heuristic=True):
        """
        Train DTLVR model
        Parameters
        ----------
        :param l: float or numpy array (1 x A), Regularization parameter. Either a single or different l's for each can
        be passed
        :param centering: bool, If True Source and Target Domain Data are Mean Centered (default)
        :param heuristic: bool, If True the regularization parameter is set to a heuristic value
        :return:
        """

        # Mean Centering
        b0 = np.mean(self.y)
        y = self.y - b0

        if centering is True:
            x = self.x[..., :]- self.mu
            xs = self.xs[..., :] - self.mu_s
            xt = self.xt[..., :] - self.mu_t
        else:
            x = self.x
            xs = self.xs
            xt = self.xt

        # Train model and store matrices
        A = self.A
        s = self.s
        lambda1=self.lambda1
        (b_R, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, Q, opt_l, Alpha, discrepancy,beta1,beta2) = DF.DTLVR(x, y, xs, xt, A, l, s, lambda1, heuristic=heuristic)
        self.b_R = b_R
        self.b0 = b0
        self.T = T
        self.Ts = Ts
        self.Tt = Tt
        self.W = W
        self.P = P
        self.Ps = Ps
        self.Pt = Pt
        self.E = E
        self.Es = Es
        self.Et = Et
        self.Ey = Ey
        self.Q = Q
        self.Alpha = Alpha
        self.discrepancy = discrepancy
        self.beta1 = beta1
        self.beta2 = beta2

        if heuristic is True:
            self.opt_l = opt_l
        return W, beta1, beta2

    def predict(self, x_test, y_test, rescale='Target'):
        """
        Predict function for DTLVR models
        :param x_test: numpy array (N x k), X data
        :param y_test:  numpy array (N x 1), Y data(optional)
        :param rescale: str or numpy.ndarray, Determines Rescaling of the Test Data (Default is Rescaling to Target
        Domain Training Set), If Array is passed, than Test Data will be Rescaled to mean of the provided Array
        :return:
        """

        # Rescale Test Data
        if (type(rescale) is str):
            if (rescale == 'Target'):
                Xtest = x_test[..., :] - self.mu_t
            elif (rescale == 'Source'):
                Xtest = x_test[..., :] - self.mu_s
            elif (rescale == 'none'):
                Xtest = x_test
        elif (type(rescale) is np.ndarray):
            Xtest = x_test[..., :] - np.mean(rescale, 0)
        else:
            raise Exception('rescale must either be Source, Target or a Dataset')

        s = self.s
        T = self.T
        Q = self.Q
        Alpha = self.Alpha
        # print(Alpha)
        b_R = self.b_R
        (m_x, n_x) = np.shape(x_test)
        N = m_x - s
        U_s = np.zeros((N+1, T.shape[1]))
        T = x_test@b_R
        for i in range(T.shape[1]):
            t = T[:, i].reshape(T.shape[0], 1)
            T_s = t[:N+1, :]
            for j in range(1, s):
                T_s = np.hstack((t[j:N+j+1, :], T_s))
            u_s = T_s @ Alpha[:, i]
            U_s[:, i] = u_s
        yhat = U_s @ Q

        if y_test is np.ndarray:
            error = DF.rmse(yhat, y_test)
        else:
            error = np.nan
        return yhat, error



        
        
        
        




