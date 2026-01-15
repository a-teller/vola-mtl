#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model confidence set implementation by Michael Gong.
(https://michael-gong.com/blogs/model-confidence-set/)
"""

import numpy as np
import pandas as pd

from numpy.random import rand
from numpy import ix_


def bootstrap_sample(data, B, w):
    t = len(data)
    p = 1 / w
    indices = np.zeros((t, B), dtype=int)
    indices[0, :] = np.ceil(t * rand(1, B))
    select = np.asfortranarray(rand(B, t).T < p)
    vals = np.ceil(rand(1, np.sum(np.sum(select))) * t).astype(int)
    indices_flat = indices.ravel(order="F")
    indices_flat[select.ravel(order="F")] = vals.ravel()
    indices = indices_flat.reshape([B, t]).T
    for i in range(1, t):
        indices[i, ~select[i, :]] = indices[i - 1, ~select[i, :]] + 1
    indices[indices > t] = indices[indices > t] - t
    indices -= 1
    return data[indices]


def compute_dij(losses, bsdata):
    t, M0 = losses.shape
    B = bsdata.shape[1]
    dijbar = np.zeros((M0, M0))
    for j in range(M0):
        dijbar[j, :] = np.mean(losses - losses[:, [j]], axis=0)

    dijbarstar = np.zeros((B, M0, M0))
    for b in range(B):
        meanworkdata = np.mean(losses[bsdata[:, b], :], axis=0)
        for j in range(M0):
            dijbarstar[b, j, :] = meanworkdata - meanworkdata[j]

    vardijbar = np.mean((dijbarstar - np.expand_dims(dijbar, 0)) ** 2, axis=0)
    vardijbar += np.eye(M0)
    return dijbar, dijbarstar, vardijbar


def calculate_PvalR(z, included, zdata0):
    empdistTR = np.max(np.max(np.abs(z), 2), 1)
    zdata = zdata0[ix_(included - 1, included - 1)]
    TR = np.max(zdata)
    pval = np.mean(empdistTR > TR)
    return pval


def calculate_PvalSQ(z, included, zdata0):
    empdistTSQ = np.sum(z**2, axis=1).sum(axis=1) / 2
    zdata = zdata0[ix_(included - 1, included - 1)]
    TSQ = np.sum(zdata**2) / 2
    pval = np.mean(empdistTSQ > TSQ)
    return pval


def iterate(dijbar, dijbarstar, vardijbar, alpha, algorithm="R"):
    B, M0, _ = dijbarstar.shape
    z0 = (dijbarstar - np.expand_dims(dijbar, 0)) / np.sqrt(
        np.expand_dims(vardijbar, 0)
    )
    zdata0 = dijbar / np.sqrt(vardijbar)

    excludedR = np.zeros([M0, 1], dtype=int)
    pvalsR = np.ones([M0, 1])

    for i in range(M0 - 1):
        included = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
        m = len(included)
        z = z0[ix_(range(B), included - 1, included - 1)]

        if algorithm == "R":
            pvalsR[i] = calculate_PvalR(z, included, zdata0)
        elif algorithm == "SQ":
            pvalsR[i] = calculate_PvalSQ(z, included, zdata0)

        scale = m / (m - 1)
        dibar = np.mean(dijbar[ix_(included - 1, included - 1)], 0) * scale
        dibstar = np.mean(dijbarstar[ix_(range(B), included - 1, included - 1)], 1) * (
            m / (m - 1)
        )
        vardi = np.mean((dibstar - dibar) ** 2, axis=0)
        tstat = dibar / np.sqrt(vardi)
        modeltoremove = np.argmax(tstat)
        excludedR[i] = included[modeltoremove]

    maxpval = pvalsR[0]
    for i in range(1, M0):
        if pvalsR[i] < maxpval:
            pvalsR[i] = maxpval
        else:
            maxpval = pvalsR[i]

    excludedR[-1] = np.setdiff1d(np.arange(1, M0 + 1), excludedR)
    pl = np.argmax(pvalsR > alpha)
    includedR = excludedR[pl:]
    excludedR = excludedR[:pl]
    return includedR - 1, excludedR - 1, pvalsR


def MCS(losses, alpha, B, w, algorithm):
    t, M0 = losses.shape
    bsdata = bootstrap_sample(np.arange(t), B, w)
    dijbar, dijbarstar, vardijbar = compute_dij(losses, bsdata)
    includedR, excludedR, pvalsR = iterate(
        dijbar, dijbarstar, vardijbar, alpha, algorithm=algorithm
    )
    return includedR, excludedR, pvalsR


class ModelConfidenceSet(object):
    def __init__(self, data, alpha, B, w, algorithm="SQ", names=None):
        if isinstance(data, pd.DataFrame):
            self.data = data.values
            self.names = data.columns.values if names is None else names
        elif isinstance(data, np.ndarray):
            self.data = data
            self.names = np.arange(data.shape[1]) if names is None else names

        if alpha < 0 or alpha > 1:
            raise ValueError(f"alpha must be in (0,1), got {alpha}")
        self.alpha = float(alpha)
        self.B = int(B)
        self.w = int(w)
        self.algorithm = algorithm

    def run(self):
        included, excluded, pvals = MCS(
            self.data, self.alpha, self.B, self.w, self.algorithm
        )
        self.included = self.names[included].ravel().tolist()
        self.excluded = self.names[excluded].ravel().tolist()
        self.pvalues = pd.Series(pvals.ravel(), index=self.excluded + self.included)
        return self
