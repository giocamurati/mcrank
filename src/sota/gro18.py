#!/usr/bin/env python

"""
Python wrapper for python_gro18
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

from python_gro18 import gro18
from scipy.special import logsumexp
import numpy as np

def rank(log_proba, knownkey, bins):
    print("Running Grosso's rank estimation")
    lp : np.ndarray = log_proba
    lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they     sum up to 1
    rank_min, rank_max, time_rank = gro18.rank(
            log_proba,
            knownkey,
            bins
    )
    rank_min = rank_min[0]
    rank_max = rank_max[0]
    time_rank = time_rank[0]
    return rank_min, rank_max, time_rank
