#!/usr/bin/env python

"""
MCRank: Monte Carlo Key Ranking for Side Channel Evaluations
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import numpy as np
from scipy.special import logsumexp
import time
from mpmath import mp

def compute_key_lp(key: np.ndarray, lp: np.ndarray):
    """
    Return log probability of a given key.
    """
    return lp[np.arange(key.size), key].sum()

def generate_samples(key: np.ndarray, lp: np.ndarray, sample_size, rng, scale=1):
    """
    Generate samples based on log probabilities and rescaling factor.
    """
    lp *= scale
    lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
    # the above is equivalent to prob /= prob.sum(axis=1)[:, np.newaxis] but numerically stable
    prob_sum = np.exp(lp).cumsum(axis=1)  # cumulative sum of probabilities
    assert np.isclose(prob_sum[:, -1], 1).all()  # probabilities sum up to 1
    # each element of sample_lp is the log-probability of a sample item
    samples = []
    sample_lp = np.zeros(sample_size)
    for i, (lp_row, sum_row) in enumerate(zip(lp, prob_sum)):  # iterate over each key piece
        sample = sum_row.searchsorted(rng.random(sample_lp.size))
        sample_lp += lp_row[sample]
        samples.append(sample)
    samples = np.asarray(samples)
    assert np.max(sample_lp) <= 0  # no probability is higher than 1
    key_lp = lp[np.arange(key.size), key].sum()  # log-prob of the real key
    assert key_lp <= 0
    return samples, sample_lp, key_lp

def compute_rank(sample_lp: np.ndarray, key_lp: np.ndarray, bootstrap=False,
        bootstrap_alpha=0.003):
    """
    Estimate the rank (no rescaling).
    """
    n = sample_lp.size
    weaker_sample = sample_lp > key_lp
    weaker_n = weaker_sample.sum()
    weaker_percentage = 100 * weaker_n / sample_lp.size
    nonzero_log_estimations = -sample_lp[weaker_sample]
    if len(nonzero_log_estimations) == 0:
        return 0, 0, 0, 0, None, None
    predicted_rank = mp.exp(logsumexp(nonzero_log_estimations) - np.log(n))
    # corrected sample standard deviation
    # https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
    a = np.sum([mp.power(mp.exp(nle)-predicted_rank, 2) for nle in
            nonzero_log_estimations])
    b = mp.power(predicted_rank, 2) * (n - weaker_n)
    stdev = mp.sqrt((a + b) / (n - 1))
    # print(f"Standard deviation: {stdev:,} (2^{np.log2(stdev):.2f})")
    # standard error of the mean (NOT the standard deviation): https://en.wikipedia.org/wiki/Standard_error
    sem = stdev / np.sqrt(n)
    bs_lower = None
    bs_upper = None
    if bootstrap:
        estimations = np.zeros_like(sample_lp)
        estimations[:weaker_n] = np.exp(nonzero_log_estimations)
        bootstrap_results = bs.bootstrap(estimations, stat_func=bs_stats.mean, alpha=bootstrap_alpha)
        bs_lower = bootstrap_results.lower_bound
        bs_upper = bootstrap_results.upper_bound
    result = predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper
    return result

def print_result(weaker_n: int, weaker_percentage: float, key_lp: np.ndarray,
        predicted_rank: float, sem: float, start: float, end: float,
        bs_lower=None, bs_upper=None, bs_alpha=None):
    """
    Print the estimated rank and the confidence interval.
    """
    def ci2s(l, r):
        return f"2^{mp.log(l,2)}, 2^{mp.log(r,2)}"
    def acc(l, r):
        return f"{mp.log(r/l,2)}"

    if predicted_rank != 0:
        print(f"Weaker keys in the sample: {weaker_n:,} ({weaker_percentage: .2f}%)")
        print(f"Key log-probability: {key_lp:.2f}")
        print(f"Predicted rank: {predicted_rank} (2^{mp.log(predicted_rank,2)})")
        print(f"Standard estimation error: {100 * sem / predicted_rank}%")
        print(f"99.7% confidence interval: {ci2s(max(1,predicted_rank - 3 * sem), predicted_rank + 3 * sem)}")
        print(f"Accuracy: {acc(max(1,predicted_rank - 3 * sem), predicted_rank + 3 * sem)}")
        if bs_lower and bs_upper and bs_alpha:
            print(f"Bootstrapped {100-100*bs_alpha}% confidence interval: {ci2s(bs_lower, bs_upper)}")
    else:
        print("Predicted rank: 0")
    print(f"Time: {end - start:.6f} (s)")
    print(f"")

def estimate_rank(key: np.ndarray, lp2: np.ndarray, seed, sample_size,
        bootstrap=False, bootstrap_alpha=0.003, rescale_manual=False,
        rescale_manual_factor=1, rescale_automatic=False,
        rescale_automatic_threshold=5,rescale_automatic_increment=0.1,
        rescale_automatic_max_iter=200, rescale_automatic_partial=False,
        rescale_automatic_partial_percentage=0.1):
    """
    Estimate the rank (with rescaling if chosen).
    """
    lp = np.array(lp2)
    print("Running Monte Carlo-based rank estimation")
    start = time.perf_counter() 
    rng = np.random.default_rng(seed)
    
    # simple experiment with scale
    if rescale_manual:
        rescale = rescale_manual_factor
        samples, sample_lp, key_lp = generate_samples(key, lp,
                sample_size, rng,
                rescale)
        predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = compute_rank(sample_lp,
                key_lp, bootstrap, bootstrap_alpha)
    elif rescale_automatic:
        rescale = 1
        for i in range(rescale_automatic_max_iter):
            samples, sample_lp, key_lp = generate_samples(key, lp,
                    sample_size, rng,
                    rescale)
            predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = compute_rank(sample_lp,
                    key_lp, bootstrap, bootstrap_alpha)
            
            if abs(50 - weaker_percentage) < rescale_automatic_threshold:
                break
 
            if weaker_percentage > 50:
                rescale -= rescale_automatic_increment
            else:
                rescale += rescale_automatic_increment
    elif rescale_automatic_partial:
        rescale = 1
        for i in range(rescale_automatic_max_iter):
            samples, sample_lp, key_lp = generate_samples(key, np.array(lp),
                    int(rescale_automatic_partial_percentage*sample_size), rng,
                    rescale)
            predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = compute_rank(sample_lp,
                    key_lp, bootstrap, bootstrap_alpha)
            
            if abs(50 - weaker_percentage) < rescale_automatic_threshold:
                break
 
            if weaker_percentage > 50:
                rescale -= rescale_automatic_increment
            else:
                rescale += rescale_automatic_increment

        samples, sample_lp, key_lp = generate_samples(key, lp,
                sample_size, rng,
                rescale)
        predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = compute_rank(sample_lp,
                    key_lp, bootstrap, bootstrap_alpha)
 
    else:
        rescale = 1
        samples, sample_lp, key_lp = generate_samples(key, lp,
                sample_size, rng,
                rescale)
        predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = compute_rank(sample_lp,
                key_lp, bootstrap, bootstrap_alpha)
   
    end = time.perf_counter()
    print_result(weaker_n, weaker_percentage, key_lp, predicted_rank, sem,
            start, end, bs_lower, bs_upper, bootstrap_alpha)
    return mp.log(predicted_rank,2), weaker_percentage, sem, bs_lower, bs_upper, end-start
