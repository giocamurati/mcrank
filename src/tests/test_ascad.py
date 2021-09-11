#!/usr/bin/env python

import sys
sys.path.append('../mcrank')
sys.path.append('../utils')
sys.path.append('../sota')

import numpy as np
import simulation
import ta
import argparse
import mcrank
import hel
import gro18

def test(sample_size, seed_mc, lp_file, key_file,
        num_traces_attack, plot, enum, hel_merge, hel_bins, gr_bins=2048,
        attack_type="ta", gr=False, es=False,
        es_gamma=1.025, es_b=41, runs=1, gm=False, bootstrap=False,
        bootstrap_alpha=0.003,
        rescale_manual=False, rescale_manual_factor=1, rescale_automatic=False,
        rescale_automatic_threshold=5, rescale_automatic_increment=0.1,
        rescale_automatic_max_iter=200,
        rescale_automatic_partial=False,
        rescale_automatic_partial_percentage=0.1):



    knownkey = np.array(np.load(key_file), dtype=int)
    lps = np.load(lp_file)
    lp = lps[:,num_traces_attack,:]
 
    print([np.argmax(lp[i]) for i in range(16)])

    results = []
    for run in range(runs):
        if seed_mc:
            seed_mc += run
        
        print(f"Run {run}, seed_mc {seed_mc}")
 
        mc_rank, mc_percentage, mc_sem, mc_bs_lower, mc_bs_upper, mc_time = mcrank.estimate_rank(
                knownkey,
                lp,
                seed_mc,
                sample_size,
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
        
        hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank = hel.rank(
                lp,
                knownkey,
                merge=hel_merge,
                bins=hel_bins
        )

        if gr:
            gro18_rank_min, gro18_rank_max, gro18_time_rank = gro18.rank(
                    lp,
                    knownkey,
                    bins=gr_bins
            )

        if es:
            import esrank
            es_rank_lower, es_rank_upper, es_time = esrank.estimate_rank(
                    knownkey,
                    lp,
                    gamma=es_gamma,
                    b=es_b,
                    d=16
            )

        if gm:
            import gmbounds
            gmlb, gmub, s_gmlb, s_gmub, gmtime = gmbounds.compute_gm_bounds(lp)
        
        if bootstrap:
            results.append([
                mc_rank, mc_percentage, mc_sem, mc_bs_lower, mc_bs_upper, mc_time,
                hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank
            ])
        else:
            if gm:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank,
                    gmlb, gmub, gmtime
                ])
            elif gr:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank,
                    gro18_rank_min, gro18_rank_max, gro18_time_rank
                ])
            else:
                results.append([
                    mc_rank, mc_percentage, mc_sem, mc_time,
                    hel_rank_min, hel_rank_rounded, hel_rank_max, hel_time_rank
                ])

        if enum:
            found = hel.enumeration(
                    lp,
                    a_plaintexts,
                    knownkey,
                    merge=hel_merge,
                    bins=hel_bins,
                    bit_bound_start=int(mc_rank-1),
                    bit_bound_end=int(mc_rank+1)
            )
        
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=5500)
    parser.add_argument('--seed-mc', type=int)
    parser.add_argument('--num-traces-attack', type=int, default=20)
    parser.add_argument('--lp-file', type=str, default="../ascad/ASCAD/lp_multilabel_without_permind.npy")
    parser.add_argument('--key-file', type=str, default="../ascad/ASCAD/key_multilabel_without_permind.npy")
    parser.add_argument('--plot', type=bool, default=False)
    parser.add_argument('--enum', type=bool, default=False)
    parser.add_argument('--gr', type=bool, default=False)
    parser.add_argument('--es', type=bool, default=False)
    parser.add_argument('--gm', type=bool, default=False)
    parser.add_argument('--hel_merge', type=int, default=2)
    parser.add_argument('--hel_bins', type=int, default=2048)
    parser.add_argument('--gr_bins', type=int, default=2048)
    parser.add_argument('--es_gamma', type=float, default=1.025)
    parser.add_argument('--es_b', type=float, default=41)
    parser.add_argument('--bootstrap', type=bool, default=False)
    parser.add_argument('--bootstrap_alpha', type=float, default=0.003)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--attack-type', type=str, default="ta")
    parser.add_argument('--rescale_manual', type=bool, default=False)
    parser.add_argument('--rescale_manual_factor', type=float, default=1)
    parser.add_argument('--rescale_automatic', type=bool, default=False)
    parser.add_argument('--rescale_automatic_threshold', type=float, default=45)
    parser.add_argument('--rescale_automatic_increment', type=float, default=0.1)
    parser.add_argument('--rescale_automatic_max_iter', type=int, default=200)
    parser.add_argument('--rescale_automatic_partial', type=bool, default=True)
    parser.add_argument('--rescale_automatic_partial_percentage', type=float, default=0.02)
    args = parser.parse_args()

    test(
        sample_size = args.sample_size,
        seed_mc = args.seed_mc,
        num_traces_attack = args.num_traces_attack,
        lp_file = args.lp_file,
        key_file = args.key_file,
        plot = args.plot,
        enum = args.enum,
        hel_merge = args.hel_merge,
        hel_bins = args.hel_bins,
        gr_bins = args.gr_bins,
        gr = args.gr,
        es = args.es,
        gm = args.gm,
        es_gamma = args.es_gamma,
        es_b = args.es_b,
        runs = args.runs,
        bootstrap = args.bootstrap,
        bootstrap_alpha = args.bootstrap_alpha,
        attack_type = args.attack_type,
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
