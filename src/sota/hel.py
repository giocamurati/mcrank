#!/usr/bin/env python

"""
Python wrapper for python_hel
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

from python_hel import hel
from scipy.special import logsumexp
import numpy as np

def enumeration(log_proba, a_plaintexts, knownkey, merge, bins, bit_bound_start,
        bit_bound_end):
    print("")
    print("Starting key enumeration using HEL")
    import ctypes
    from Crypto.Cipher import AES
    
    lp : np.ndarray = log_proba
    lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1

    pt1 = np.array(a_plaintexts[0], dtype=ctypes.c_ubyte)
    pt2 = np.array(a_plaintexts[1], dtype=ctypes.c_ubyte)
 
    print("Assuming that we know two plaintext/ciphertext pairs")
    _key = bytes(list(knownkey))
    _pt1 = bytes(list(pt1))
    _pt2 = bytes(list(pt2))

    cipher = AES.new(_key, AES.MODE_ECB)
 
    _ct1 = cipher.encrypt(_pt1)
    _ct2 = cipher.encrypt(_pt2)
    
    ct1 = list(_ct1)
    ct1 = np.array(ct1, dtype=ctypes.c_ubyte)
    ct2 = list(_ct2)
    ct2 = np.array(ct2, dtype=ctypes.c_ubyte)

    found = hel.bruteforce(lp, pt1, pt2, ct1, ct2, merge,
        bins, bit_bound_start, bit_bound_end)
    return bool(found[0])

def rank(log_proba, knownkey, merge, bins):
    print("Running Histogram-based rank estimation")
    lp : np.ndarray = log_proba
    lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they     sum up to 1
    rank_min, rank_rounded, rank_max, time_rank = hel.rank(
            log_proba,
            knownkey,
            merge,
            bins)
    rank_min = rank_min[0]
    rank_rounded = rank_rounded[0]
    rank_max = rank_max[0]
    time_rank = time_rank[0]
    return rank_min, rank_rounded, rank_max, time_rank
