#!/usr/bin/env python

"""
Test for MCRank with RSA
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

import numpy as np
import simulation
import ta
import argparse
import mcrank
import gro18

def test(num_key_bytes, sample_size, seed_template, seed_attack, seed_mc, num_traces_template,
        num_traces_attack, scale, plot, runs=1, gr=False, gr_bins=4096,
        gm=False, gm_symbolic=False, gm_digits=100,
        bootstrap=False, bootstrap_alpha=0.003,
        rescale_manual=False, rescale_manual_factor=1, rescale_automatic=False,
        rescale_automatic_threshold=5, rescale_automatic_increment=0.1,
        rescale_automatic_max_iter=200,
        rescale_automatic_partial=False,
        rescale_automatic_partial_percentage=0.1):


    p_keys, p_traces = simulation.simulate_rsa(
            n_key_bytes=num_key_bytes,
            n_traces=num_traces_template,
            fixed_key=False,
            loc=0,
            scale=scale,
            seed=seed_template
    )
    
    a_keys, a_traces = simulation.simulate_rsa(
            n_key_bytes=num_key_bytes,
            n_traces=num_traces_attack,
            fixed_key=True,
            loc=0,
            scale=scale,
            seed=seed_attack
    )
    knownkey = a_keys[0]
    
    pois, p_means, p_covs = ta.profile_rsa(
            p_keys,
            p_traces,
            plot=plot
    )
    
    lp, bestguess = ta.attack_vec_rsa(
            pois,
            p_means,
            p_covs,
            a_keys,
            a_traces,
            printing=True
    )

    results = []
    for run in range(runs):
        if seed_mc:
            seed_mc += run
        
        print(f"Run {run}, seed_mc {seed_mc}")
 
        mc_rank, mc_percentage, mc_sem, x, x, mc_time = mcrank.estimate_rank(
                knownkey,
                lp,
                seed_mc,
                sample_size,
                # batches=10
                bootstrap,
                bootstrap_alpha,
                rescale_manual,
                rescale_manual_factor,
                rescale_automatic,
                rescale_automatic_threshold,
                rescale_automatic_increment,
                rescale_automatic_max_iter,
                rescale_automatic_partial,
                rescale_automatic_partial_percentage
        )   
        
        if gm:
            import gmbounds
            gmlb, gmub, s_gmlb, s_gmub, gmtime = gmbounds.compute_gm_bounds(
                    lp,
                    int(gm_symbolic),
                    gm_digits
            )
        
        if gr:
            gro18_rank_min, gro18_rank_max, gro18_time_rank = gro18.rank(
                    lp,
                    knownkey,
                    bins=gr_bins
            )

        # hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank = hel.rank(
                # lp,
                # knownkey,
                # merge=hel_merge,
                # bins=hel_bins
        # )

        # if es:
            # import esrank
            # es_rank_lower, es_rank_upper, es_time = esrank.estimate_rank(
                    # knownkey,
                    # lp,
                    # gamma=es_gamma,
                    # b=es_b,
                    # d=16
            # )
        
        # if enum:
            # found = hel.enumeration(
                    # lp,
                    # a_plaintexts,
                    # knownkey,
                    # merge=hel_merge,
                    # bins=hel_bins,
                    # bit_bound_start=int(np.log2(mc_rank)-1),
                    # bit_bound_end=int(np.log2(mc_rank)+1)
            # )
        
        if bootstrap:
            results.append([
                mc_rank, mc_percentage, mc_sem, mc_bs_lower, mc_bs_upper, mc_time,
                0, 0, 0, 0
            ])
        else:
            if gm:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    0, 0, 0, 0,
                    gmlb, gmub, gmtime
                ])
            elif gr:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    0, 0, 0, 0,
                    gro18_rank_min, gro18_rank_max, gro18_time_rank
                ])
            else:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    0, 0, 0, 0
                ])

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-key-bytes', type=int, default=16)
    parser.add_argument('--sample-size', type=int, default=50000)
    parser.add_argument('--seed-template', type=int)
    parser.add_argument('--seed-attack', type=int)
    parser.add_argument('--seed-mc', type=int)
    parser.add_argument('--num-traces-template', type=int, default=5000)
    parser.add_argument('--num-traces-attack', type=int, default=20)
    parser.add_argument('--scale', type=float, default=1.4)
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--enum', type=bool, default=False)
    parser.add_argument('--gr', type=bool, default=False)
    parser.add_argument('--gr_bins', type=int, default=4096)
    parser.add_argument('--gm', type=bool, default=False)
    parser.add_argument('--gm_symbolic', type=bool, default=False)
    parser.add_argument('--gm_digits', type=int, default=100)
    # parser.add_argument('--es', type=bool, default=False)
    # parser.add_argument('--hel_merge', type=int, default=2)
    # parser.add_argument('--hel_bins', type=int, default=2048)
    # parser.add_argument('--es_gamma', type=float, default=1.025)
    # parser.add_argument('--es_b', type=float, default=41)
    parser.add_argument('--bootstrap', type=bool, default=False)
    parser.add_argument('--bootstrap_alpha', type=float, default=0.003)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--rescale_manual', type=bool, default=False)
    parser.add_argument('--rescale_manual_factor', type=float, default=1)
    parser.add_argument('--rescale_automatic', type=bool, default=False)
    parser.add_argument('--rescale_automatic_threshold', type=float, default=5)
    parser.add_argument('--rescale_automatic_increment', type=float, default=0.1)
    parser.add_argument('--rescale_automatic_max_iter', type=int, default=200)
    parser.add_argument('--rescale_automatic_partial', type=bool, default=False)
    parser.add_argument('--rescale_automatic_partial_percentage', type=float, default=0.1)
    args = parser.parse_args()

    test(
        num_key_bytes = args.num_key_bytes,
        sample_size = args.sample_size,
        seed_template = args.seed_template,
        seed_attack = args.seed_attack,
        seed_mc = args.seed_mc,
        num_traces_template = args.num_traces_template,
        num_traces_attack = args.num_traces_attack,
        scale = args.scale,
        plot = args.plot,
        runs = args.runs,
        bootstrap = args.bootstrap,
        bootstrap_alpha = args.bootstrap_alpha,
        gm = args.gm,
        gm_symbolic = args.gm_symbolic,
        gm_digits = args.gm_digits,
        gr = args.gr,
        gr_bins = args.gr_bins,
        rescale_manual = args.rescale_manual,
        rescale_manual_factor = args.rescale_manual_factor,
        rescale_automatic = args.rescale_automatic,
        rescale_automatic_threshold = args.rescale_automatic_threshold,
        rescale_automatic_increment = args.rescale_automatic_increment,
        rescale_automatic_max_iter = args.rescale_automatic_max_iter,
        rescale_automatic_partial = args.rescale_automatic_partial,
        rescale_automatic_partial_percentage = args.rescale_automatic_partial_percentage
    )

if __name__ == "__main__":
    main()
