#!/usr/bin/env python

"""
Test for MCRank with AES-256
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

import sys
sys.path.append('../mcrank')
sys.path.append('../utils')
sys.path.append('../sota')

import pickle

import numpy as np
import simulation
import ta
import argparse
from scipy.special import logsumexp
import time
import mcrank as mc
import gro18

def test(sample_size, seed_template, seed_attack, seed_mc, num_traces_template,
        num_traces_attack, scale, plot, runs, attack_type="ta", bootstrap=False,
        bootstrap_alpha=0.05, gr=False, gr_bins=2048):

    p_plaintexts, p_keys, p_r2states, p_traces = simulation.simulate256(
            n_traces=num_traces_template,
            fixed_key=False,
            loc=0,
            scale=scale,
            seed=seed_template
    )
    
    a_plaintexts, a_keys, a_r2states, a_traces = simulation.simulate256(
            n_traces=num_traces_attack,
            fixed_key=True,
            loc=0,
            scale=scale,
            seed=seed_attack
    )
    knownkey = a_keys[0]
    
    pois_0, means_0, covs_0 = ta.profile(
            p_plaintexts,
            p_keys[:,0:16],
            p_traces,
            plot=plot
    )
 
    if attack_type == "ta":
        lp_0, bestguess_0 = ta.attack_vec(
                pois_0,
                means_0,
                covs_0,
                a_plaintexts,
                a_keys[:,0:16],
                a_traces,
                printing=True
        )
    elif attack_type == "profiled_ca":
        lp_0, bestguess_0 = ta.profiled_corr_attack(
                pois_0,
                means_0,
                a_plaintexts,
                a_keys[:,0:16],
                a_traces,
                printing=True
        )
    
    p_r2states = np.array([simulation.round2(p, k) for p,k in
        zip(p_plaintexts, p_keys[:,0:16])])
    
    pois_1, means_1, covs_1 = ta.profile(
            p_r2states,
            p_keys[:,16:32],
            p_traces,
            plot=plot
    )
    
    results = []
    for run in range(runs):
        # First part of MC
        # Generate samples for key_0
        start = time.perf_counter()
        if seed_mc:
            seed_mc += run
        rng = np.random.default_rng(seed_mc)
        samples, sample_lp, key_lp = mc.generate_samples(
                a_keys[0,0:16], # known key_0
                lp_0,
                sample_size,
                rng
        )
        sorting = np.lexsort(samples)
        samples, sample_lp = samples.T[sorting], sample_lp[sorting]
        
        # Second part of MC
        # Run attack on second part, for bestguess and for all samples
        # And update probabilities
        
        # Run attack on key_1 using known key_0
        a_r2states = np.array([simulation.round2(p, a_keys[0,0:16]) for p in 
            a_plaintexts])
        
        if attack_type == "ta":
            lp_1, bestguess_1 = ta.attack_vec(
                    pois_1,
                    means_1,
                    covs_1,
                    a_r2states,
                    a_keys[:,16:32],
                    a_traces,
                    printing=True
            )
        elif attack_type == "profiled_ca":
            lp_1, bestguess_1 = ta.profiled_corr_attack(
                    pois_1,
                    means_1,
                    a_r2states,
                    a_keys[:,16:32],
                    a_traces,
                    printing=True
            )
        
        lp_g = np.concatenate((lp_0, lp_1), axis=0)
        print(np.shape(lp_g))
       
        lp_1 -= logsumexp(lp_1, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
        key_lp += mc.compute_key_lp(
                a_keys[0,16:32], # known key_1
                lp_1
        )
        
        # Run attack on all samples and sample again
        last_sample = None
        for i, k0_sample in enumerate(samples):
            if i % 100 == 0:
                print(i, k0_sample)
            
            # a_r2states = np.array([simulation.round2(p, bestguess_0) for p in 
                # a_plaintexts])
            a_r2states = np.array([simulation.round2(p, k0_sample) for p in 
                a_plaintexts])
        
            if (k0_sample != last_sample).any():
                lp = ta.attack_lp(pois_1, means_1, covs_1, a_r2states, a_traces)
                lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
                #the above is equivalent to prob /= prob.sum(axis=1)[:, np.newaxis] but numerically stable
                prob_sum = np.exp(lp).cumsum(axis=1)  # cumulative sum of probabilities
                assert np.isclose(prob_sum[:, -1], 1).all()  # probabilities sum up to 1
        
            # each element of sample_lp is the log-probability of a sample item
            for lp_row, sum_row in zip(lp, prob_sum):  # iterate over each key piece
                sample = sum_row.searchsorted(rng.random())
                sample_lp[i] += lp_row[sample]
            assert np.max(sample_lp) <= 0  # no probability is higher than 1
        
            last_sample = k0_sample
        
        predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = mc.compute_rank(sample_lp, key_lp, bootstrap, bootstrap_alpha)
        end = time.perf_counter()
         
        mc.print_result(weaker_n, weaker_percentage, key_lp, predicted_rank,
                sem, start, end, bs_lower, bs_upper, bootstrap_alpha)
     
        mc_rank, mc_percentage, mc_sem, mc_bs_lower, mc_bs_upper, mc_time = mc.estimate_rank(
                knownkey,
                lp_g,
                seed_mc,
                sample_size,
                bootstrap,
                bootstrap_alpha,
        )
 

        print("")
        if gr:
            gro18_rank_min, gro18_rank_max, gro18_time_rank = gro18.rank(
                    lp_g,
                    knownkey,
                    bins=gr_bins
            )

        if bootstrap:
            results.append([weaker_n, weaker_percentage, key_lp, predicted_rank,
                sem, bs_lower, bs_upper, start, end])
        else:
            results.append([weaker_n, weaker_percentage, key_lp, predicted_rank, sem, start, end])
    
    return results
 
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--seed-template', type=int)
    parser.add_argument('--seed-attack', type=int)
    parser.add_argument('--seed-mc', type=int)
    parser.add_argument('--num-traces-template', type=int, default=5000)
    parser.add_argument('--num-traces-attack', type=int, default=20)
    parser.add_argument('--scale', type=float, default=1.4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--bootstrap', type=bool, default=False)
    parser.add_argument('--bootstrap_alpha', type=float, default=0.003)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--attack-type', type=str, default="ta")
    parser.add_argument('--gr_bins', type=int, default=2048)
    parser.add_argument('--gr', type=bool, default=False)
    args = parser.parse_args()

    test(
        sample_size = args.sample_size,
        seed_template = args.seed_template,
        seed_attack = args.seed_attack,
        seed_mc = args.seed_mc,
        num_traces_template = args.num_traces_template,
        num_traces_attack = args.num_traces_attack,
        scale = args.scale,
        plot = args.plot,
        bootstrap = args.bootstrap,
        bootstrap_alpha = args.bootstrap_alpha,
        runs = args.runs,
        attack_type = args.attack_type,
        gr_bins = args.gr_bins,
        gr = args.gr
    )

if __name__ == "__main__":
    main()
