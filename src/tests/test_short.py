#!/usr/bin/env python

"""
Test for MCRank with toy example (dependent)
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
from myplots import plt
import mpmath as mp
import gro18

def test(sample_size, seed_template, seed_attack, seed_mc, num_traces_template,
        num_traces_attack, scale, plot, runs, attack_type="ta", savefig=False,
        outfile="./fig.pdf", gr=False, gr_bins=2048):

    p_plaintexts, p_keys, p_r2states, p_traces = simulation.simulate16(
            n_traces=num_traces_template,
            fixed_key=False,
            loc=0,
            scale=scale,
            seed=seed_template
    )
    
    a_plaintexts, a_keys, a_r2states, a_traces = simulation.simulate16(
            n_traces=num_traces_attack,
            fixed_key=True,
            loc=0,
            scale=scale,
            seed=seed_attack
    )
    knownkey = a_keys[0]
    
    pois_0, means_0, covs_0 = ta.profile(
            p_plaintexts,
            p_keys[:,0:1],
            p_traces,
            plot=False #plot
    )
    
    if attack_type == "ta":
        lp_0, bestguess_0 = ta.attack_vec(
                pois_0,
                means_0,
                covs_0,
                a_plaintexts,
                a_keys[:,0:1],
                a_traces,
                printing=True
        )
    elif attack_type == "profiled_ca":
       lp_0, bestguess_0 = ta.profiled_corr_attack(
               pois_0,
               means_0,
               a_plaintexts,
               a_keys[:,0:1],
               a_traces,
               printing=True
       )
    
    p_r2states = np.array([simulation.aes16_round2(p, k) for p,k in
        zip(p_plaintexts, p_keys[:,0:1])])
    
    pois_1, means_1, covs_1 = ta.profile(
            p_r2states,
            p_keys[:,1:2],
            p_traces,
            plot=False #plot
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
                a_keys[0,0:1], # known key_0
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
        a_r2states = np.array([simulation.aes16_round2(p, a_keys[0,0:1]) for p in 
            a_plaintexts])
        
        lp_1, bestguess_1 = ta.attack_vec(
                pois_1,
                means_1,
                covs_1,
                a_r2states,
                a_keys[:,1:2],
                a_traces,
                printing=True
        )

        lp_g = np.concatenate((lp_0, lp_1), axis=0)

        lp_1 -= logsumexp(lp_1, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
        key_lp += mc.compute_key_lp(
                a_keys[0,1:2], # known key_1
                lp_1
        )

        # Run attack on all samples and sample again
        lp_sampled = [np.inf for i in range(2**16)]
        last_sample = None
        for i, k0_sample in enumerate(samples):
            if i % 1000 == 0:
                print(i, k0_sample)

            a_r2states = np.array([simulation.aes16_round2(p, k0_sample) for p in 
                a_plaintexts])
            
            if (k0_sample != last_sample).any():
                lp = ta.attack_lp(pois_1, means_1, covs_1, a_r2states, a_traces)
                lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
                # the above is equivalent to prob /= prob.sum(axis=1)[:, np.newaxis] but numerically stable
                prob_sum = np.exp(lp).cumsum(axis=1)  # cumulative sum of probabilities
                assert np.isclose(prob_sum[:, -1], 1).all()  # probabilities sum up to 1
        
            # each element of sample_lp is the log-probability of a sample item
            for lp_row, sum_row in zip(lp, prob_sum):  # iterate over each key piece
                sample = sum_row.searchsorted(rng.random())
                sample_lp[i] += lp_row[sample]
                lp_sampled[k0_sample[0]+256*sample] = sample_lp[i]
            assert np.max(sample_lp) <= 0  # no probability is higher than 1
        
            last_sample = k0_sample
        
        lp_sampled = np.asarray(lp_sampled)
        
        predicted_rank, weaker_n, weaker_percentage, sem, bs_upper, bs_lower = mc.compute_rank(sample_lp, key_lp)
        end = time.perf_counter()
         
        mc.print_result(weaker_n, weaker_percentage, key_lp, predicted_rank, sem, start, end)
        
        lp_all = [0 for _ in range(2**16)]
        for byte_0 in range(256):
            a_r2states = np.array([simulation.aes16_round2(p, np.asarray([byte_0])) for p in 
                a_plaintexts])
            lp = ta.attack_lp(pois_1, means_1, covs_1, a_r2states, a_traces)
            lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities so they sum up to 1
            # the above is equivalent to prob /= prob.sum(axis=1)[:, np.newaxis] but numerically stable
            prob_sum = np.exp(lp).cumsum(axis=1)  # cumulative sum of probabilities
            assert np.isclose(prob_sum[:, -1], 1).all()  # probabilities sum up to 1
            # each element of sample_lp is the log-probability of a sample item
            
            for byte_1 in range(256):
                lp_all[byte_0+256*byte_1] = mc.compute_key_lp(np.asarray([byte_0]), lp_0)
                lp_all[byte_0+256*byte_1] += lp[0,byte_1]
                assert np.max(sample_lp) <= 0  # no probability is higher than 1
         
        lp_all = np.asarray(lp_all)
        key_lp_real = lp_all[a_keys[0,0]+256*a_keys[0,1]]
        real_rank = (lp_all > key_lp_real).sum()
        print(f"Real rank: {real_rank:,} (2^{np.log2(real_rank):.2f})")
        
        if plot or savefig:
            plt.hist(lp_all[lp_all > key_lp], bins=np.arange(-45,-5,0.1), 
                label='$W(K,k)$',
                edgecolor='None', alpha = 0.5)
            plt.hist(lp_all[lp_all <= key_lp], bins=np.arange(-45,-5,0.1),
                label='$K \\setminus W(K,k)$',
                edgecolor='None', alpha = 0.5)
            plt.xlabel("$\log(p(k))$")
            plt.ylabel("$\\textrm{count}$")

            plt.hist(lp_sampled[lp_sampled > key_lp], bins=np.arange(-45,-5,0.1), 
                label='$W(S,k)$',
                edgecolor='None', alpha = 1)
            plt.hist(lp_sampled[lp_sampled <= key_lp], bins=np.arange(-45,-5,0.1),
                label='$S \\setminus W(S,k)$',
                edgecolor='None', alpha = 1)

            plt.text(-44.5, 50,
                    "$|K|=%d$\n \
                    $R_{k}=|W(K,k)|=2^{%.2f}$\n \
                    $|S|=n=%d$\n \
                    $\\tilde R_{k}=\\frac{1}{n} \\sum_{s \\in W(S,k)}\\frac{1}{p(s)}=2^{%.2f}$ \n \
                    $CI_{99.7%s}=[2^{%.2f},2^{%.2f}]$ \n \
                    $Uncertainty=%.2f\\:bits$"%(len(lp_all),
                        np.log2(real_rank), len(sample_lp),
                        np.log2(float(predicted_rank)),
                        '\%',
                        np.log2(float(predicted_rank-3*sem)),
                        np.log2(float(predicted_rank+3*sem)),
                        np.log2(float((predicted_rank+3*sem)/(predicted_rank-3*sem)))),
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    fontsize=16)

            plt.legend()
 
            if savefig:
                plt.savefig(outfile, bbox_inches='tight')
                plt.cla()
            else:
                plt.show()
        
        print("")
        if gr:
            gro18_rank_min, gro18_rank_max, gro18_time_rank = gro18.rank(
                    lp_g,
                    knownkey,
                    bins=gr_bins
            )
            print(np.log2(real_rank), gro18_rank_min, gro18_rank_max)
            if np.log2(real_rank) > gro18_rank_min and np.log2(real_rank) < gro18_rank_max:
                print("Real rank falls in CARDIS18 bounds!")
            else:
                print("Real rank does not fall in the CARDIS18 bounds...")
              
        print("")
 
        rank_min = predicted_rank-3*sem
        rank_max = predicted_rank+3*sem
        if real_rank > rank_min and real_rank < rank_max:
            print("Real rank falls in the 99.7% confidence interval!")
        else:
            print("Real rank does not fall in the 99.7% confidence interval...")
        
        print("")
        
        results.append([weaker_n, weaker_percentage, key_lp, predicted_rank, sem, start, end, real_rank])
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=5000)
    parser.add_argument('--seed-template', type=int)
    parser.add_argument('--seed-attack', type=int)
    parser.add_argument('--seed-mc', type=int)
    parser.add_argument('--num-traces-template', type=int, default=5000)
    parser.add_argument('--num-traces-attack', type=int, default=20)
    parser.add_argument('--scale', type=float, default=3)
    parser.add_argument('--plot', type=bool, default=False)
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
        runs = 1,
        attack_type = args.attack_type,
        gr_bins = args.gr_bins,
        gr = args.gr
    )

if __name__ == "__main__":
    plt.rcParams['axes.grid'] = False
    main()
    plt.rcParams['axes.grid'] = True
