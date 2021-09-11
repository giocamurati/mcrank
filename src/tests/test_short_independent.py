#!/usr/bin/env python

"""
Test for MCRank with toy example (independent)
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
# from matplotlib import pyplot as plt
from myplots import plt

def test(sample_size, seed_template, seed_attack, seed_mc, num_traces_template,
        num_traces_attack, scale, plot, runs, bootstrap=False,
        bootstrap_alpha=0.03, savefig=False, outfile="./fig.pdf",
        rescale_automatic=False,
        rescale_automatic_threshold=5, rescale_automatic_increment=0.1,
        rescale_automatic_max_iter=200):

    nbytes = 2

    # simulate128 but then take only few bytes at the end
    p_plaintexts, p_keys, p_traces = simulation.simulate(
            n_traces=num_traces_template,
            fixed_key=False,
            loc=0,
            scale=scale,
            seed=seed_template
    )
    
    a_plaintexts, a_keys, a_traces = simulation.simulate(
            n_traces=num_traces_attack,
            fixed_key=True,
            loc=0,
            scale=scale,
            seed=seed_attack
    )
    
    knownkey = a_keys[0]
    
    pois, means, covs = ta.profile(
            p_plaintexts,
            p_keys,
            p_traces,
            plot=False #plot
    )
    
    lp, bestguess = ta.attack_vec(
            pois,
            means,
            covs,
            a_plaintexts,
            a_keys,
            a_traces,
            printing=True
    )
    
    # get back to few bytes
    lp = lp[0:nbytes]
    knownkey = knownkey[0:nbytes]

    results = []
    for run in range(runs):
        
        # MC
        print(f"Run {run}, seed_mc {seed_mc}")
 
        start = time.perf_counter()
        rng = np.random.default_rng(seed_mc)
        if not rescale_automatic:
            samples, sample_lp, key_lp = mc.generate_samples(knownkey, lp, sample_size, rng)
            predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = mc.compute_rank(sample_lp,
                key_lp, bootstrap, bootstrap_alpha)
        else:
            total_rescale = 1
            rescale = 1 #0.52
            for i in range(rescale_automatic_max_iter):
                total_rescale *= rescale
                samples, sample_lp, key_lp = mc.generate_samples(knownkey, lp,
                        sample_size, rng,
                        rescale)
                predicted_rank, weaker_n, weaker_percentage, sem, bs_lower, bs_upper = mc.compute_rank(sample_lp,
                        key_lp, bootstrap, bootstrap_alpha)
                
                if abs(50 - weaker_percentage) < rescale_automatic_threshold:
                    break
 
                if weaker_percentage > 50:
                    rescale -= rescale_automatic_increment
                else:
                    rescale += rescale_automatic_increment
            print(f"found rescale factor {total_rescale}")

        end = time.perf_counter()
        mc.print_result(weaker_n, weaker_percentage, key_lp, predicted_rank, sem,
            start, end, bs_lower, bs_upper, bootstrap_alpha)
        mc_time = end - start
        
        lp_all = [0 for _ in range(2**16)]
        for byte_0 in range(256):
            for byte_1 in range(256):
                lp_all[byte_0+256*byte_1] = lp[0,byte_0] + lp[1,byte_1]
         
        lp_all = np.asarray(lp_all)
        key_lp_real = lp_all[knownkey[0]+256*knownkey[1]]
        real_rank = (lp_all > key_lp_real).sum()
        print(f"Real rank: {real_rank:,} (2^{np.log2(real_rank):.2f})")
        
        if plot or savefig:
            plt.hist(lp_all[lp_all > key_lp], bins=np.arange(-48,-3,0.1), 
                label='$W(K,k^{*})$',
                edgecolor='None', alpha = 0.5)
            plt.hist(lp_all[lp_all <= key_lp], bins=np.arange(-48,-3,0.1),
                label='$K \\setminus W(K,k^{*})$',
                edgecolor='None', alpha = 0.5)
            plt.xlabel("$\log(p(k^{*}))$")
            plt.ylabel("$\\textrm{count}$")

            plt.hist(sample_lp[sample_lp > key_lp], bins=np.arange(-48,-3,0.1), 
                label='$W(S,k^{*})$',
                edgecolor='None', alpha = 1)
            plt.hist(sample_lp[sample_lp <= key_lp], bins=np.arange(-48,-3,0.1),
                label='$S \\setminus W(S,k^{*})$',
                edgecolor='None', alpha = 1)

            plt.text(-48, 50,
                    "$|K|=%d$\n \
                    $R_{k^{*}}=|W(K,k^{*})|=2^{%.2f}$\n \
                    $|S|=n=%d$\n \
                    $\\tilde R_{k^{*}}=\\frac{1}{n} \\sum_{s \\in W(S,k^{*})}\\frac{1}{p(s)}=2^{%.2f}$ \n \
                    $CI_{99.7%s}=[2^{%.2f},2^{%.2f}]$ \n \
                    $Uncertainty=%.2f\\:bits$"%(len(lp_all),
                        np.log2(float(real_rank)), sample_size,
                        np.log2(float(predicted_rank)),
                        '\%',
                        np.log2(float(predicted_rank-3*sem)),
                        np.log2(float(predicted_rank+3*sem)),
                        np.log2(float((predicted_rank+3*sem)/(predicted_rank-3*sem)))),
                    verticalalignment='bottom',
                    horizontalalignment='left',
                    fontsize=16)

            plt.text(-9.5,800,
                    "$p(k) > p(k^{*})$",
                    color='grey')

            plt.axvline(x=key_lp, color='grey', linestyle='--')  

            plt.legend(loc='upper left')
 
            # import IPython; IPython.embed()
            if savefig:
                plt.savefig(outfile, bbox_inches='tight')
                plt.cla()
            else:
                plt.show()
        
        rank_min = predicted_rank-3*sem
        rank_max = predicted_rank+3*sem
        if real_rank > rank_min and real_rank < rank_max:
            print("Real rank falls in the 99.7% confidence interval!")
        else:
            print("Real rank does not fall in the 99.7% confidence interval...")
        
        print("")
        
        results.append([weaker_percentage, key_lp,
            predicted_rank, sem, mc_time, real_rank])
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
    parser.add_argument('--rescale_automatic', type=bool, default=False)
    parser.add_argument('--rescale_automatic_threshold', type=float, default=5)
    parser.add_argument('--rescale_automatic_increment', type=float, default=0.1)
    parser.add_argument('--rescale_automatic_max_iter', type=int, default=200)
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
        rescale_automatic = args.rescale_automatic,
        rescale_automatic_threshold = args.rescale_automatic_threshold,
        rescale_automatic_increment = args.rescale_automatic_increment,
        rescale_automatic_max_iter = args.rescale_automatic_max_iter
    )

if __name__ == "__main__":
    plt.rcParams['axes.grid'] = False
    main()
    plt.rcParams['axes.grid'] = True
