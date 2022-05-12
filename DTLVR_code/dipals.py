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
import matplotlib.pyplot as plt
import dipls_functions as algo
import scipy.stats

class model():
    def __init__(self, x, y, xs, xt, A):
        self.x = x
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
        self.P = []
        self.Ps = []
        self.Pt = []
        self.W = []
        self.A = A
        self.opt_l = []
        self.b0 = np.mean(y, 0)
        self.b = []
        self.yhat = []
        self.rmse = []
        self.C = []

    def train(self, l=0, centering=True, heuristic=False):
        """
        Train di-PLS model
        Parameters
        __________
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
            x = self.x[..., :] - self.mu
            xs = self.xs[..., :] - self.mu_s
            xt = self.xt[..., :] - self.mu_t
        else:
            x = self.x
            xs = self.xs
            xt = self.xt

        # Train model and store matrices
        A = self.A
        (b, T, Ts, Tt, W, P, Ps, Pt, E, Es, Et, Ey, C, opt_l, discrepancy) = algo.dipals(x, y, xs, xt, A, l, heuristic = heuristic)
        self.b = b
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
        self.C = C
        self.discrepancy = discrepancy

        if heuristic is True:
            self.opt_l = opt_l
        return W

    def predict(self, x_test, y_test, rescale='Target'):
        """
        Predict function for di-PLS models
        Parameters
        __________
        :param x_test: numpy array (N x k), X data
        :param y_test: numpy array (N x 1), Y data(optional)
        :param rescale: str or numpy.ndarray, Determines Rescaling of the Test Data (Default is Rescaling to Target
        Domain Training Set), If Array is passed, than Test Data will be Rescaled to mean of the provided Array
        :return:
        yhat: numpy array (N x 1), Predicted Y
        RMSE: int, Root mean square error
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

        yhat = Xtest@self.b + self.b0

        if y_test is np.ndarray:
            error = algo.rmse(yhat, y_test)
        else:
            error = np.nan
        return yhat, error










