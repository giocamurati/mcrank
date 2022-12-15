#!/usr/bin/env python

"""
MCrank evaluation
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

import sys
sys.path.append('../mcrank') # MCRank
sys.path.append('../tests') # Tests for toy, AES-128/256, RSA, ASCAD
sys.path.append('../utils') # Simulation of traces and attacks
sys.path.append('../sota') # State-of-the-art key rank estimation  tools

if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

from myplots import plt
import os
import shutil
from pathlib import Path
import pandas as pd
import test_short
import test_short_independent
import test_128
import test_256
import test_rsa
import test_ascad
import seaborn as sns
from scipy.stats import shapiro, anderson
import numpy as np
import math
from mpmath import mp
import argparse
from argparse import Action, ArgumentParser

def recreate_outdir(outdir, clean):
    """
    Create ouput folder.
    """
    if outdir.exists() and outdir.is_dir():
        if clean:
            shutil.rmtree(outdir)
            os.mkdir(outdir)
        else:
            pass
    else:
        os.mkdir(outdir)

def latex_init(outdir):
    """
    Init latex file.
    """
    with open(outdir / "all.tex", "w") as all_f:
        all_f.write("\\documentclass{article}\n")
        all_f.write("\\usepackage{booktabs}\n")
        all_f.write("\\usepackage{graphicx}\n")
        all_f.write("\\usepackage{placeins}\n")
        all_f.write("\\usepackage{caption}\n")
        all_f.write("\\begin{document}\n")

def latex_write(outdir, line):
    """
    Write a line in latex file.
    """
    with open(outdir / "all.tex", "a") as all_f:
        all_f.write(line+"\n")

def latex_add_fig(outdir, fig, caption=""):
    """
    Add a figure to latex file.
    """
    with open(outdir / "all.tex", "a") as all_f:
        all_f.write("\\begin{figure}[h!]\n")
        all_f.write("\\centering\n")
        all_f.write("\\includegraphics[scale=0.5]{%s}\n"%fig)
        all_f.write("\\caption*{%s}\n"%caption)
        all_f.write("\\end{figure}\n")

def latex_close(outdir):
    """
    End latex file.
    """
    with open(outdir / "all.tex", "a") as all_f:
        all_f.write("\\end{document}\n")

def run0(outdir):
    """
    Evaluation of Figures 1a, 1b, 1c
    """
 
    plt.rcParams['axes.grid'] = False
    results = test_short_independent.test(
       sample_size = 1500,
       seed_template = 10,
       seed_attack = 21,
       seed_mc = 33,
       num_traces_template = 50000,
       num_traces_attack = 30, #20,
       scale = 4,
       plot = False,
       runs = 1,
       savefig = True,
       outfile = outdir / "aes_short_independent_mc_illustrated_small.pdf"
    )
    results = test_short_independent.test(
       sample_size = 4000,
       seed_template = 10,
       seed_attack = 21,
       seed_mc = 33,
       num_traces_template = 50000,
       num_traces_attack = 30, #20,
       scale = 4,
       plot = False,
       runs = 1,
       savefig = True,
       outfile = outdir / "aes_short_independent_mc_illustrated_medium.pdf"
    )
    results = test_short_independent.test(
       rescale_automatic = True,
       rescale_automatic_threshold = 1,
       rescale_automatic_max_iter = 1000,
       rescale_automatic_increment = 0.1,
       sample_size = 4000,
       seed_template = 10,
       seed_attack = 21,
       seed_mc = 33,
       num_traces_template = 50000,
       num_traces_attack = 30, #20,
       scale = 4,
       plot = False,
       runs = 1,
       savefig = True,
       outfile = outdir / "aes_short_independent_mc_illustrated_medium_rescaling.pdf"
    )
    plt.rcParams['axes.grid'] = True

def plt0(outdir):
    """
    Plot of Figures 1a, 1b, 1c
    """

    latex_add_fig(
        outdir,
        "aes_short_independent_mc_illustrated_small.pdf",
        caption="Paper figure 1a."
    )
    latex_add_fig(
        outdir,
        "aes_short_independent_mc_illustrated_medium.pdf",
        caption="Paper figure 1b."
    )
    latex_add_fig(
        outdir,
        "aes_short_independent_mc_illustrated_medium_rescaling.pdf",
        caption="Paper figure 1c."
    )

def run1(outdir):
    """
    Evaluation of Figure 2a
    """

    columns=[
        "runs",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_bs_lower",
        "mc_bs_upper",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time"
    ]
    df = pd.DataFrame(columns=columns)
    sample_sizes = [i for i in range(200, 50000, 100)]
    for i, sample_size in enumerate(sample_sizes):
        results = test_128.test(
            sample_size = sample_size,
            seed_template = 0,
            seed_attack = 1,
            seed_mc = 2,
            num_traces_template = 5000,
            num_traces_attack = 2,
            scale = 2,
            plot = False,
            runs = 1,
            enum = False,
            hel_merge = 2,
            hel_bins = 2048,
            bootstrap = True,
            bootstrap_alpha = 0.003
        )
        for result in results:
            df2 = pd.DataFrame([[1, sample_size]+list(result)], columns=columns)
            df = pd.concat([df, df2])
    df.to_csv(outdir / "test1.csv", index=False)
 
def plt1(outdir):
    """
    Plot of Figure 2a
    """
    
    df = pd.read_csv(outdir / "test1.csv")
 
    df['mc_rank'] = 2**df['mc_rank']
    
    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    uncertainty = np.log2(upper/lower)

    plt.plot(df['sample_size'], np.log2(upper),
            linewidth=2,
            alpha=0.5,
            label='$\\log2(\\textrm{estimated rank} + 3 \cdot \\textrm{SEM})$')
    plt.plot(df['sample_size'], np.log2(lower),
            linewidth=2,
            alpha=0.5,
            label='$\\log2(\\textrm{estimated rank} - 3 \cdot \\textrm{SEM})$')
    
    plt.plot(df['sample_size'], np.log2(df['mc_bs_lower']), linewidth=0.01,
            label='$\\log2(\\textrm{bootstrap lower bound})$')
    plt.plot(df['sample_size'], np.log2(df['mc_bs_upper']), linewidth=0.01,
            label='$\\log2(\\textrm{boostrap upper bound})$')

    plt.title("Tightness of the bounds for increasing sample size")
    plt.ylabel("$\\log2(\\textrm{predicted rank})$")
    plt.xlabel("sample size")
    plt.legend()
    plt.savefig(outdir / "aes_128_tightness_by_sample_size_bootstrap.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_tightness_by_sample_size_bootstrap.pdf",
        "Paper figure 2a."
    )
 
def run2(outdir):
    """
    Evaluation of Figures 2b, 2c
    """
    columns=[
        "runs",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time"
    ]
    df = pd.DataFrame(columns=columns)
    sample_sizes = [i for i in range(200, 50000, 100)]
    for i, sample_size in enumerate(sample_sizes):
        results = test_128.test(
            sample_size = sample_size,
            seed_template = 0,
            seed_attack = 1,
            seed_mc = 2,
            num_traces_template = 5000,
            num_traces_attack = 2,
            scale = 2,
            plot = False,
            runs = 1,
            enum = False,
            hel_merge = 2,
            hel_bins = 2048
        )
        for result in results:
            df2 = pd.DataFrame([[1, sample_size]+list(result)], columns=columns)
            df = pd.concat([df, df2])
    df.to_csv(outdir / "test2.csv", index=False)

def plt2(outdir):
    """
    Plot of Figures 2b, 2c.
    """

    df = pd.read_csv(outdir / "test2.csv")
 
    df['mc_rank'] = 2**df['mc_rank']
    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    uncertainty = np.log2(upper/lower)

    plt.plot(df['sample_size'], 100*df['mc_sem']/df['mc_rank'], '*')
    plt.title("Relative SEM for increasing sample size")
    plt.ylabel("$100 \cdot \\textrm{SEM} / \\textrm{estimated rank} (\%)$")
    plt.xlabel("sample size")
    plt.savefig(outdir / "aes_128_relative_sem_by_sample_size.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_relative_sem_by_sample_size.pdf",
        "Paper figure 2b."
    )

    hel_uncertainty = np.log2(2**df['hel_rank_max'][0]/2**df['hel_rank_min'][0])
    crossing = df['sample_size'][np.argmax(uncertainty < hel_uncertainty)]
    plt.plot(df['sample_size'], uncertainty, '*')
    plt.title("Uncertainty for increasing sample size")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("sample size")
    plt.savefig(outdir / "aes_128_uncertainty_by_sample_size.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_uncertainty_by_sample_size.pdf",
        "Paper figure 2c"
    )

def run3(outdir):
    """
    Evaluation of Figure 2d
    """
    
    runs = 1000
    sample_sizes = [100, 1000, 10000]
    columns=[
        "runs",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time"
    ]
    df = pd.DataFrame(columns=columns)
    for i, sample_size in enumerate(sample_sizes):
        results = test_128.test(
            sample_size = sample_size,
            seed_template = 0,
            seed_attack = 1,
            seed_mc = 2,
            num_traces_template = 5000,
            num_traces_attack = 2,
            scale = 2,
            plot = False,
            runs = runs,
            enum = False,
            hel_merge = 2,
            hel_bins = 2048
        )
        for result in results:
            df2 = pd.DataFrame([[runs, sample_size]+list(result)], columns=columns)
            df = pd.concat([df, df2])
    df.to_csv(outdir / "test3.csv", index=False)

def plt3(outdir):
    """
    Plot of Figure 2d
    """
        
    df = pd.read_csv(outdir / "test3.csv")
    ranks_by_sample_size = df.groupby('sample_size')['mc_rank']

    for x, y in ranks_by_sample_size: 
        runs = len(y.values)
        plt.gca().set_aspect('auto')
        plt.hist(y.values, bins=int(runs/50), label=f"runs {runs}, size {x}"); 
        plt.xlabel("$\\log2(\\textrm{predicted rank})$") 
        plt.ylabel("count") 
    plt.plot([df.hel_rank_rounded[0], df.hel_rank_rounded[0]], [0, 120],
            'r--', label="HEL rank rounded")
    plt.legend()
    plt.savefig(outdir / "aes_128_ranks_by_sample_size.pdf", bbox_inches='tight') 
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_ranks_by_sample_size.pdf",
        "Paper figure 2d."
    )

def run4(outdir):
    """
    Evaluation of Figures 3a, 3b
    """

    columns=[
        "scale",
        "sample_size",
        "hel_bins",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time",
        "exp"
    ]
    df = pd.DataFrame(columns=columns)
    scales = [i/100.0 for i in range(30,300,50)]
    for exp in range(10):
        for i, scale in enumerate(scales):
            hel_uncertainty = 128
            mc_uncertainty = 128
            hel_bins = 64
            sample_size = 100
            n_runs = 10
            timeout = 0
            while(hel_uncertainty >= 1 or mc_uncertainty > hel_uncertainty):
                timeout += 1
                results = test_128.test(
                    sample_size = sample_size,
                    seed_template = 0 + exp,
                    seed_attack = 1 + exp,
                    seed_mc = 2,
                    num_traces_template = 5000,
                    num_traces_attack = 2,
                    scale = scale,
                    plot = False,
                    runs = n_runs,
                    enum = False,
                    hel_merge = 2,
                    hel_bins = hel_bins
                )

                results = [[float(r) for r in result] for result in results]
                hel_uncertainty = np.max(np.log2(2**np.asarray(results)[:,6]/2**np.asarray(results)[:,4]))
                mc_lower = 2**np.asarray(results)[:,0] - 3*np.asarray(results)[:,2]
                mc_upper = 2**np.asarray(results)[:,0] + 3*np.asarray(results)[:,2]
                mc_uncertainty = np.max(np.log2(mc_upper/mc_lower))
                if math.isnan(mc_uncertainty):
                    mc_uncertainty = 128
                if math.isnan(hel_uncertainty):
                    mc_uncertainty = 128
                if hel_uncertainty >= 20:
                    if(timeout > 10):
                        break
                    timeout += 1
                if hel_uncertainty >= 10 and scale < 0.5:
                    hel_bins += 4096
                    n_runs = 1
                elif hel_uncertainty >= 1.5 and scale < 0.5:
                    hel_bins += 512
                    n_runs = 1
                elif hel_uncertainty >= 1:
                    hel_bins += 64
                    n_runs = 1
                elif mc_uncertainty - hel_uncertainty > 1:
                    n_runs = 1
                    sample_size += 100
                elif mc_uncertainty > hel_uncertainty:
                    n_runs = 10
                    sample_size += 500

                if(hel_uncertainty < 1 and mc_uncertainty < hel_uncertainty):
                    for result in results:
                        df2 = pd.DataFrame([[scale, sample_size,
                            hel_bins]+list(result)+[exp]], columns=columns)
                        df = pd.concat([df, df2])
    df.to_csv(outdir / "test4.csv", index=False)

def plt4(outdir):
    """
    Plot of Figures 3a, 3b
    """
    
    df = pd.read_csv(outdir / "test4.csv")
        
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = 2**df['mc_rank']       

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = np.log2(upper/lower)
    df['hel_uncertainty'] = np.log2(2**df['hel_rank_max']/2**df['hel_rank_min'])
    
    for m, mu, ml, gu, gl in zip(df['mc_rank'], upper, lower, df['hel_rank_max'].tolist(),
            df['hel_rank_min'].tolist()):
        
        if math.isnan(gu): #gro18 not run
            continue 
        if float(mp.log(mu,2)) < gl or float(mp.log(ml,2)) > gu:
            print(f"Error {mp.log(m,2)}: {gu} {gl}")
 
    df['hel_time'] = df['hel_time']*1000
    df['mc_time'] = df['mc_time']*1000

    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_uncertainty', c='y', label='HEL')
    df.plot.scatter(x='mc_rank_log', y='mc_uncertainty', label='MCRank', c='b', ax=ax)
    plt.title("HEL-MCRank uncertainty")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.legend()
    plt.savefig(outdir / "aes_128_comparison_uncertainty_by_rank.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_comparison_uncertainty_by_rank.pdf",
        "Paper figure 3a."
    )

    df['hel_time'] = df['hel_time']*1000
    df['mc_time'] = df['mc_time']*1000
    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_time', c='y', label='HEL')
    plt.yscale('log')
    df.plot.scatter(x='mc_rank_log', y='mc_time', label='MCRank', c='b', ax=ax)
    plt.title("HEL-MCRank speed at uncertainty lower than 1 bit")
    plt.ylabel("execution time (ms)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    # plt.legend()
    plt.yscale('log')
    plt.savefig(outdir / "aes_128_comparison_execution_time_by_rank.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_comparison_execution_time_by_rank.pdf",
        "Paper figure 3b."
    )


def run5(outdir):
    """
    Evaluation of Figures 4a, 4b
    """
    
    columns=[
        "run",
        "sample_size",
        "weaker_n",
        "weaker_percentage", 
        "key_lp", 
        "predicted_rank",
        "sem", 
        "start",
        "end"
    ]
    df = pd.DataFrame(columns=columns)
    sample_sizes = [2]+[i for i in range(1000, 100000, 1000)]
    for i, sample_size in enumerate(sample_sizes):
        results = test_256.test(
            sample_size = sample_size,
            seed_template = 0,
            seed_attack = 1,
            seed_mc = 2,
            num_traces_template = 5000,
            num_traces_attack = 2,
            scale = 2,
            plot = False,
            runs = 1
        )
        for result in results:
            df2 = pd.DataFrame([[1, sample_size]+list(result)], columns=columns)
            df = pd.concat([df, df2])
    df.to_csv(outdir / "test5.csv", index=False)

def plt5(outdir):
    """
    Plot of Figures 4a, 4b
    """
    
    df = pd.read_csv(outdir / "test5.csv")
 
    lower = df['predicted_rank']-3*df['sem']
    upper = df['predicted_rank']+3*df['sem']
    uncertainty = np.log2(upper/lower)
  
    plt.plot(df['sample_size'], uncertainty, '*')
    plt.title("Uncertainty for increasing sample size")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("sample size")
    plt.savefig(outdir / "aes_256_uncertainty_by_sample_size.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_256_uncertainty_by_sample_size.pdf",
        "Paper figure 4a."
    )
    
    plt.plot(df['sample_size'], df['end']-df['start'], '*')
    plt.title("Execution time for increasing sample size")
    plt.ylabel("execution time (s)")
    plt.xlabel("sample size")
    plt.savefig(outdir / "aes_256_execution_time_by_sample_size.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_256_execution_time_by_sample_size.pdf",
        "Paper figure 4b."
    )
 
    #crossing = np.argmax(uncertainty < 0.1)
    #urint(np.log2(df['predicted_rank'][crossing]),
    #        df['end'][crossing]-df['start'][crossing],
    #        df['sample_size'][crossing],
    #        uncertainty[crossing])

def run6(outdir):
    """
    Evaluation of Figures 5a, 5b
    """
    
    columns=[
        "n_key_bytes",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time",
        "gro18_rank_min",
        "gro18_rank_max",
        "gro18_time_rank",
        "exp"
    ]
    df = pd.DataFrame(columns=columns)
    key_sizes = [i for i in [16, 32, 64, 128, 256, 512, 1024, 2048]]
    for exp in range(10):
        for i, n_key_bytes in enumerate(key_sizes):

            sample_size = 1000
            results = test_rsa.test(
                rescale_automatic = True,
                rescale_automatic_threshold = 25, #45,
                rescale_automatic_increment = 0.1,
                rescale_automatic_max_iter = 1000,
                gr = True,
                gr_bins = 4096, #65536, #4096,
                # gm = True,
                num_key_bytes = n_key_bytes,
                sample_size = sample_size,
                seed_template = 0 + exp,
                seed_attack = 1 + exp,
                seed_mc = 2 + exp,
                num_traces_template = 5000,
                num_traces_attack = 5,
                scale = 80,
                plot = False,
                runs = 11 #11
            )

            # skip first because first run is slower when calling matlab
            for result in results[1:]:
                df2 = pd.DataFrame([[n_key_bytes, sample_size]+list(result)+[exp]], columns=columns)
                df = pd.concat([df, df2])
    df.to_csv(outdir / "test6.csv", index=False)

def plt6(outdir):
    """
    Plot of Figures 5a, 5b
    """
    
    df = pd.read_csv(outdir / "test6.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = df['mc_rank'].apply(lambda x : mp.power(2,x))
    df['mc_sem'] = df['mc_sem'].apply(mp.mpmathify)

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = [float(mp.log(u/max(1,l),2)) for u,l in zip(upper, lower)]
    df['gro18_uncertainty'] = [float(mp.log(mp.power(2,ub)/mp.power(2,lb),2)) for ub, lb in
            zip(df['gro18_rank_max'].tolist(), df['gro18_rank_min'].tolist())]
 
    for m, mu, ml, gu, gl, i  in zip(df['mc_rank'], upper, lower, df['gro18_rank_max'].tolist(),
            df['gro18_rank_min'].tolist(), df['n_key_bytes'].tolist()):
        if float(mp.log(ml,2)) > gu or float(mp.log(mu,2)) < gl:
            print(f"Error {mp.log(m,2)} {float(mp.log(mu,2))} {float(mp.log(ml,2))} {gu} {gl}")

    df['gro18_time_rank'] = df['gro18_time_rank']*1000
    df['mc_time'] = df['mc_time']*1000

    ax = df.plot.scatter(x='n_key_bytes', y='gro18_uncertainty', c='y',
            label='[Gro18]')
    df.plot.scatter(x='n_key_bytes', y='mc_uncertainty', label='MCRank', c='b', ax=ax)
    plt.title("MCRank-[Gro18] uncertainty")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("key size (bytes)")
    plt.yscale('log')
    plt.legend()
    plt.savefig(outdir / "rsa_comparison_gro18_uncertainty_by_key_length.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gro18_uncertainty_by_key_length.pdf",
        "Paper figure 5a."
    )

    ax = df.plot.scatter(x='n_key_bytes', y='mc_time', label='MCRank', c='b')
    df.plot.scatter(x='n_key_bytes', y='gro18_time_rank', c='y',
            label='[Gro18]', ax=ax)
    plt.yscale('log')
    plt.title("MCRank-[Gro18] speed")
    plt.ylabel("execution time (ms)")
    plt.xlabel("key size (bytes)")
    plt.savefig(outdir /
            "rsa_comparison_gro18_execution_time_by_key_length.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gro18_execution_time_by_key_length.pdf",
        "Paper figure 5b."
    )

def run7(outdir):
    """
    Evaluation of Figures 5c, 5d
    """
    
    columns=[
        "n_key_bytes",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time",
        "gro18_rank_min",
        "gro18_rank_max",
        "gro18_time_rank",
        "exp"
    ]
    df = pd.DataFrame(columns=columns)
    key_sizes = [16, 32, 64, 128, 256, 512, 1024, 2048]
    sample_sizes = [1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000]
    bin_sizes = [2**i for i in range(13,21)]
    for exp in range(10):
        for i, n_key_bytes in enumerate(key_sizes):

            sample_size = sample_sizes[i]
            results = test_rsa.test(
                rescale_automatic = True,
                rescale_automatic_threshold = 25,
                rescale_automatic_increment = 0.1,
                rescale_automatic_max_iter = 1000,
                gr = True if i < 6 else 0,
                gr_bins = bin_sizes[i],
                num_key_bytes = n_key_bytes,
                sample_size = sample_size,
                seed_template = 0 + exp,
                seed_attack = 1 + exp,
                seed_mc = 2 + exp,
                num_traces_template = 5000,
                num_traces_attack = 5,
                scale = 80,
                plot = False,
                runs = 11 #11
            )

            if i >=6:
                for i in range(len(results)):
                    results[i].append(np.nan)
                    results[i].append(np.nan)
                    results[i].append(np.nan)
 
            # skip first because first run is slower when calling matlab
            for result in results[1:]:
                df2 = pd.DataFrame([[n_key_bytes, sample_size]+list(result)+[exp]], columns=columns)
                df = pd.concat([df, df2])
    df.to_csv(outdir / "test7.csv", index=False)

def plt7(outdir):
    """
    Plot of Figures 5c, 5d
    """
    
    df = pd.read_csv(outdir / "test7.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = df['mc_rank'].apply(lambda x : mp.power(2,x))
    df['mc_sem'] = df['mc_sem'].apply(mp.mpmathify)

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = [float(mp.log(u/max(1,l),2)) for u,l in zip(upper, lower)]
    df['gro18_uncertainty'] = [float(mp.log(mp.power(2,ub)/mp.power(2,lb),2)) for ub, lb in
            zip(df['gro18_rank_max'].tolist(), df['gro18_rank_min'].tolist())]
 
    for m, mu, ml, gu, gl, i  in zip(df['mc_rank'], upper, lower, df['gro18_rank_max'].tolist(),
            df['gro18_rank_min'].tolist(), df['n_key_bytes'].tolist()):
        if math.isnan(gu): #gro18 not run
            continue 
        if float(mp.log(mu,2)) < gl or float(mp.log(ml,2)) > gu:
            print(f"Error {mp.log(m,2)}: {gu} {gl}")

    df['gro18_time_rank'] = df['gro18_time_rank']*1000
    df['mc_time'] = df['mc_time']*1000
    
    ax = df.plot.scatter(x='n_key_bytes', y='mc_uncertainty', label='MCRank', c='b')
    ax = df.plot.scatter(x='n_key_bytes', y='gro18_uncertainty', c='y',
            label='[Gro18]', ax=ax)
    plt.title("MCRank-[Gro18] uncertainty")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("key size (bytes)")
    plt.yscale('log')
    plt.legend()
    plt.savefig(outdir / "rsa_comparison_gro18_uncertainty_by_key_length_2.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gro18_uncertainty_by_key_length_2.pdf",
        "Paper figure 5c."
    )

    ax = df.plot.scatter(x='n_key_bytes', y='gro18_time_rank', c='y',
            label='[Gro18]')
    df.plot.scatter(x='n_key_bytes', y='mc_time', label='MCRank', c='b',
            ax=ax)
    plt.yscale('log')
    plt.title("MCRank-[Gro18] speed")
    plt.ylabel("execution time (ms)")
    plt.xlabel("key size (bytes)")
    plt.savefig(outdir /
            "rsa_comparison_gro18_execution_time_by_key_length_2.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gro18_execution_time_by_key_length_2.pdf",
        "Paper figure 5d."
    )

def run8(outdir):
    """
    Evaluation of Figures 6a, 6b
    """
    
    columns=[
        "n_key_bytes",
        "sample_size",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time",
        "gmlb",
        "gmub",
        "gmtime",
        "exp"
    ]
    df = pd.DataFrame(columns=columns)
    key_sizes = [i for i in [16, 32, 64, 128, 256, 512, 1024, 2048]]
    symbolic = [False, False, False, False, True, True, True, True]
    digits = [100]*8
    for exp in range(10):
        for i, n_key_bytes in enumerate(key_sizes):

            sample_size = 1000
            results = test_rsa.test(
                rescale_automatic = True,
                rescale_automatic_threshold = 25, #45,
                rescale_automatic_increment = 0.1,
                rescale_automatic_max_iter = 1000,
                gm = True,
                gm_symbolic = symbolic[i],
                gm_digits = digits[i],
                num_key_bytes = n_key_bytes,
                sample_size = sample_size,
                seed_template = 0 + exp,
                seed_attack = 1 + exp,
                seed_mc = 2 + exp,
                num_traces_template = 5000,
                num_traces_attack = 5,
                scale = 80,
                plot = False,
                runs = 11 #11
            )

            # skip first because first run is slower when calling matlab
            for result in results[1:]:
                df2 = pd.DataFrame([[n_key_bytes, sample_size]+list(result)+[exp]], columns=columns)
                df = pd.concat([df, df2])
    df.to_csv(outdir / "test8.csv", index=False)

def plt8(outdir):
    """
    Plot of Figures 6a, 6b
    """
    
    df = pd.read_csv(outdir / "test8.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = df['mc_rank'].apply(lambda x : mp.power(2,x))
    df['mc_sem'] = df['mc_sem'].apply(mp.mpmathify)

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = [float(mp.log(u/max(1,l),2)) for u,l in zip(upper, lower)]
    df['gm_uncertainty'] = df['gmub']-df['gmlb']

    # No check on bounds here because we are measuring two different things
    # (rank vs ge)

    df['gm_time'] = df['gmtime']*1000
    df['mc_time'] = df['mc_time']*1000
    
    ax = df.plot.scatter(x='n_key_bytes', y='gm_uncertainty', c='y',
            label='GMBounds')
    df.plot.scatter(x='n_key_bytes', y='mc_uncertainty', label='MCRank', c='b', ax=ax)
    plt.title("HEL-GMBounds uncertainty")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("key size (bytes)")
    plt.legend()
    plt.yscale('log')
    plt.savefig(outdir / "rsa_comparison_gmbounds_uncertainty_by_key_length.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gmbounds_uncertainty_by_key_length.pdf",
        "Paper figure 6a."
    )

    ax = df.plot.scatter(x='n_key_bytes', y='gm_time', c='y',
            label='GMBounds')
    plt.yscale('log')
    df.plot.scatter(x='n_key_bytes', y='mc_time', label='MCRank', c='b', ax=ax)
    plt.title("GMBounds-MCRank speed")
    plt.ylabel("execution time (ms)")
    plt.xlabel("key size (bytes)")
    plt.yscale('log')
    plt.savefig(outdir /
            "rsa_comparison_gmbounds_execution_time_by_key_length.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./rsa_comparison_gmbounds_execution_time_by_key_length.pdf",
        "Paper figure 6b."
    )

def run9(outdir):
    """
    Evaluation of Figures 7a, 7b
    """
    
    columns=[
        "rescaling",
        "scale",
        "sample_size",
        "hel_bins",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time",
        "exp"
    ]
    df = pd.DataFrame(columns=columns)
    scales = [i/100.0 for i in range(30,300,50)]
    for rescaling in [False, True]:
        for exp in range(10):
            for i, scale in enumerate(scales):
                hel_uncertainty = 128
                mc_uncertainty = 128
                hel_bins = 64
                sample_size = 100
                n_runs = 10
                timeout = 0
                threshold = 45
                mc_timeout = False
                while((hel_uncertainty >= 1 or mc_uncertainty > hel_uncertainty) and not
                        mc_timeout):
                    timeout += 1
                    results = test_128.test(
                        attack_type = "profiled_ca",
                        rescale_automatic = rescaling,
                        rescale_automatic_threshold = max(1, threshold),
                        rescale_automatic_max_iter = 1000,
                        rescale_automatic_increment = 0.1,
                        sample_size = sample_size,
                        seed_template = 0 + exp,
                        seed_attack = 1 + exp,
                        seed_mc = 2,
                        num_traces_template = 5000,
                        num_traces_attack = 5,
                        scale = scale,
                        plot = False,
                        runs = n_runs,
                        enum = False,
                        hel_merge = 2,
                        hel_bins = hel_bins
                    )

                    results = [[float(r) for r in result] for result in results]
                    hel_uncertainty = np.max(np.log2(2**np.asarray(results)[:,6]/2**np.asarray(results)[:,4]))
                    mc_lower = 2**np.asarray(results)[:,0] - 3*np.asarray(results)[:,2]
                    mc_upper = 2**np.asarray(results)[:,0] + 3*np.asarray(results)[:,2]
                    mc_uncertainty = np.max(np.log2(mc_upper/mc_lower))
                    if math.isnan(mc_uncertainty):
                        mc_uncertainty = 128
                    if math.isnan(hel_uncertainty):
                        mc_uncertainty = 128
                    if hel_uncertainty >= 20:
                        if(timeout > 10):
                            break
                        timeout += 1
                    if hel_uncertainty >= 10 and scale < 0.5:
                        hel_bins += 4096
                        n_runs = 1
                    elif hel_uncertainty >= 1.5 and scale < 0.5:
                        hel_bins += 1024 #512
                        n_runs = 1
                    elif hel_uncertainty >= 1:
                        hel_bins += 64
                        n_runs = 1
                    elif mc_uncertainty - hel_uncertainty > 1:
                        n_runs = 1
                        sample_size += 100
                        if(mc_uncertainty == 128 and timeout > 20):
                            mc_timeout = True
                        timeout += 1
                    elif mc_uncertainty > hel_uncertainty:
                        n_runs = 10
                        sample_size += 100
                        if(mc_uncertainty == 128 and timeout > 20):
                            mc_timeout = True
                        timeout += 1
 
                    if(hel_uncertainty < 1 and mc_uncertainty < hel_uncertainty or
                            mc_timeout):
                        for result in results:
                            df2 = pd.DataFrame([[rescaling, scale, sample_size,
                                hel_bins]+list(result)+[exp]], columns=columns)
                            df = pd.concat([df, df2])
    df.to_csv(outdir / "test9.csv", index=False)

def plt9(outdir):
    """
    Plot of Figures 7a, 7b
    """
    
    df = pd.read_csv(outdir / "test9.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = 2**df['mc_rank']       

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = np.log2(upper/lower)
    df['hel_uncertainty'] = np.log2(2**df['hel_rank_max']/2**df['hel_rank_min'])

    for m, mu, ml, gu, gl, r in zip(df['mc_rank'], upper, lower, df['hel_rank_max'].tolist(),
            df['hel_rank_min'].tolist(), df.rescaling):
        if math.isnan(gu): #gro18 not run
            continue 
        if not r:
            continue
        if float(np.log2(mu)) < gl or float(np.log2(ml)) > gu:
            print(f"Error {mp.log(m,2)}: {gu} {gl}")

    df['hel_time'] = df['hel_time']*1000
    df['mc_time'] = df['mc_time']*1000

    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_uncertainty', c='y', label='HEL')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (rescaling)', c='b', ax=ax)
    df[df.rescaling==False].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank uncertainty") #\\\\(Profiled correlation attack)")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.legend()
    plt.savefig(outdir / "aes_128_comparison_uncertainty_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_comparison_uncertainty_by_rank_correlation.pdf",
        "Paper figure 7a."
    )

    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_time', c='y', label='HEL')
    plt.yscale('log')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_time', label='MCRank (rescaling)', c='b', ax=ax)
    df[df.rescaling==False].plot.scatter(x='mc_rank_log', y='mc_time',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank speed at uncertainty lower than 1 bit") #\\\\(Profiled correlation attack)")
    plt.ylabel("execution time (ms)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.yscale('log')
    plt.savefig(outdir /
            "aes_128_comparison_execution_time_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./aes_128_comparison_execution_time_by_rank_correlation.pdf",
        "Paper figure 7b."
    )

def run10(outdir):
    """
    Evaluation of Figures 8a, 8b
    """
    
    columns=[
        "rescaling",
        "num_traces_attack",
        "sample_size",
        "hel_bins",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time"
    ]
    df = pd.DataFrame(columns=columns)
    for rescaling in [True, False]:
        for num_traces_attack in range(10,400,10):
            hel_uncertainty = 128
            mc_uncertainty = 128
            hel_bins = 64
            sample_size = 100
            n_runs = 2 #100
            timeout = 0
            threshold = 45
            mc_timeout = False
            while((hel_uncertainty >= 1 or mc_uncertainty > hel_uncertainty) and not
                    mc_timeout):
                timeout += 1
                results = test_ascad.test(
                    key_file = "../ascad/ASCAD/key_multilabel.npy",
                    lp_file = "../ascad/ASCAD/lp_multilabel.npy",
                    rescale_automatic_partial = rescaling,
                    rescale_automatic_partial_percentage = 0.02,
                    rescale_automatic_threshold = max(1, threshold),
                    rescale_automatic_max_iter = 1000,
                    rescale_automatic_increment = 0.1,
                    sample_size = sample_size,
                    seed_mc = 2,
                    num_traces_attack = num_traces_attack,
                    plot = False,
                    runs = n_runs,
                    enum = False,
                    hel_merge = 2,
                    hel_bins = hel_bins
                )

                results = [[float(r) for r in result] for result in results]
                hel_uncertainty = np.max(np.log2(2**np.asarray(results)[:,6]/2**np.asarray(results)[:,4]))
                mc_lower = 2**np.asarray(results)[:,0] - 3*np.asarray(results)[:,2]
                mc_upper = 2**np.asarray(results)[:,0] + 3*np.asarray(results)[:,2]
                mc_uncertainty = np.max(np.log2(mc_upper/mc_lower))
                if math.isnan(mc_uncertainty):
                    mc_uncertainty = 128
                if math.isnan(hel_uncertainty):
                    mc_uncertainty = 128
                if hel_uncertainty >= 20:
                    if(timeout > 10):
                        break
                    timeout += 1
                if hel_uncertainty >= 10 and num_traces_attack < 10:
                    hel_bins += 4096
                    n_runs = 1
                elif hel_uncertainty >= 1.5 and num_traces_attack < 10:
                    hel_bins += 512
                    n_runs = 1
                elif hel_uncertainty >= 1:
                    hel_bins += 16 #64
                    n_runs = 1
                elif mc_uncertainty - hel_uncertainty > 1:
                    n_runs = 1
                    sample_size += 100
                    if(mc_uncertainty == 128 and timeout > 50):
                        mc_timeout = True
                    timeout += 1
                elif mc_uncertainty > hel_uncertainty:
                    n_runs = 10
                    sample_size += 500
                    if(mc_uncertainty == 128 and timeout > 50):
                        mc_timeout = True
                    timeout += 1
 
                if(hel_uncertainty < 1 and mc_uncertainty < hel_uncertainty or
                        mc_timeout):
                    for result in results:
                        df2 = pd.DataFrame([[rescaling, num_traces_attack, sample_size,
                            hel_bins]+list(result)], columns=columns)
                        df = pd.concat([df, df2])
    df.to_csv(outdir / "test10.csv", index=False)

def plt10(outdir):
    """
    Plot of Figures 8a, 8b
    """
    
    df = pd.read_csv(outdir / "test10.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = 2**df['mc_rank']       

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = np.log2(upper/lower)
    df['hel_uncertainty'] = np.log2(2**df['hel_rank_max']/2**df['hel_rank_min'])
    
    for m, mu, ml, gu, gl, r in zip(df['mc_rank'], upper, lower, df['hel_rank_max'].tolist(),
            df['hel_rank_min'].tolist(), df.rescaling):
        if math.isnan(gu): #gro18 not run
            continue 
        if not r or m == 0:
            continue
        if float(np.log2(mu)) < gl or float(np.log2(max(1,ml))) > gu:
            print(f"Error {mp.log(m,2)}: {gu} {gl}")
 
    df['hel_time'] = df['hel_time']*1000
    df['mc_time'] = df['mc_time']*1000
    
    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_uncertainty', c='y', label='HEL')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (rescaling)', c='b', ax=ax)
    df[df.rescaling==False].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank uncertainty") # (ASCAD multilabel)")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.legend()
    plt.savefig(outdir / "ascad_multilabel_comparison_uncertainty_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./ascad_multilabel_comparison_uncertainty_by_rank_correlation.pdf",
        "Paper figure 8a."
    )

    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_time', c='y', label='HEL')
    plt.yscale('log')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_time', label='MCRank (rescaling)', c='b', ax=ax)
    df2 = df[df.rescaling==False]
    df2 = df2[[not np.isnan(x) for x in df2.mc_uncertainty]]
    df2.plot.scatter(x='mc_rank_log', y='mc_time',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank speed at uncertainty lower than 1 bit") # (ASCAD multilabel)")
    plt.ylabel("execution time (ms)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.yscale('log')
    plt.savefig(outdir /
            "ascad_multilabel_comparison_execution_time_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./ascad_multilabel_comparison_execution_time_by_rank_correlation.pdf",
        "Paper figure 8b."
    )

def run11(outdir):
    """
    Evaluation of Figures 8c, 8d
    """
    
    columns=[
        "rescaling",
        "num_traces_attack",
        "sample_size",
        "hel_bins",
        "mc_rank",
        "mc_percentage",
        "mc_sem",
        "mc_time",
        "hel_rank_min",
        "hel_rank_rounded",
        "hel_rank_max",
        "hel_time"
    ]
    df = pd.DataFrame(columns=columns)
    for rescaling in [True, False]:
        for num_traces_attack in range(10,400,10):
            hel_uncertainty = 128
            mc_uncertainty = 128
            hel_bins = 64
            sample_size = 100
            n_runs = 2 #100
            timeout = 0
            threshold = 45
            mc_timeout = False
            while((hel_uncertainty >= 1 or mc_uncertainty > hel_uncertainty) and not
                    mc_timeout):
                timeout += 1
                results = test_ascad.test(
                    key_file = "../ascad/ASCAD/key_multilabel_without_permind.npy",
                    lp_file = "../ascad/ASCAD/lp_multilabel_without_permind.npy",
                    rescale_automatic_partial = rescaling,
                    rescale_automatic_partial_percentage = 0.02,
                    rescale_automatic_threshold = max(1, threshold),
                    rescale_automatic_max_iter = 1000,
                    rescale_automatic_increment = 0.1,
                    sample_size = sample_size,
                    seed_mc = 2,
                    num_traces_attack = num_traces_attack,
                    plot = False,
                    runs = n_runs,
                    enum = False,
                    hel_merge = 2,
                    hel_bins = hel_bins
                )

                results = [[float(r) for r in result] for result in results]
                hel_uncertainty = np.max(np.log2(2**np.asarray(results)[:,6]/2**np.asarray(results)[:,4]))
                mc_lower = 2**np.asarray(results)[:,0] - 3*np.asarray(results)[:,2]
                mc_upper = 2**np.asarray(results)[:,0] + 3*np.asarray(results)[:,2]
                mc_uncertainty = np.max(np.log2(mc_upper/mc_lower))
                if math.isnan(mc_uncertainty):
                    mc_uncertainty = 128
                if math.isnan(hel_uncertainty):
                    mc_uncertainty = 128
                if hel_uncertainty >= 20:
                    if(timeout > 10):
                        break
                    timeout += 1
                if hel_uncertainty >= 10 and num_traces_attack < 10:
                    hel_bins += 4096
                    n_runs = 1
                elif hel_uncertainty >= 1.5 and num_traces_attack < 10:
                    hel_bins += 512
                    n_runs = 1
                elif hel_uncertainty >= 1:
                    hel_bins += 16 #64
                    n_runs = 1
                elif mc_uncertainty - hel_uncertainty > 1:
                    n_runs = 1
                    sample_size += 100
                    if(mc_uncertainty == 128 and timeout > 50):
                        mc_timeout = True
                    timeout += 1
                elif mc_uncertainty > hel_uncertainty:
                    n_runs = 10
                    sample_size += 500
                    if(mc_uncertainty == 128 and timeout > 50):
                        mc_timeout = True
                    timeout += 1

                if(hel_uncertainty < 1 and mc_uncertainty < hel_uncertainty or
                        mc_timeout):
                    for result in results:
                        df2 = pd.DataFrame([[rescaling, num_traces_attack, sample_size,
                            hel_bins]+list(result)], columns=columns)
                        df = pd.concat([df, df2])
    df.to_csv(outdir / "test11.csv", index=False)

def plt11(outdir):
    """
    Plot of Figures 8c, 8d
    """
    
    df = pd.read_csv(outdir / "test11.csv")
    
    df['mc_rank_log'] = df['mc_rank']
    df['mc_rank'] = 2**df['mc_rank']       

    lower = df['mc_rank']-3*df['mc_sem']
    upper = df['mc_rank']+3*df['mc_sem']
    df['mc_uncertainty'] = np.log2(upper/lower)
    df['hel_uncertainty'] = np.log2(2**df['hel_rank_max']/2**df['hel_rank_min'])

    for m, mu, ml, gu, gl, r in zip(df['mc_rank'], upper, lower, df['hel_rank_max'].tolist(),
            df['hel_rank_min'].tolist(), df.rescaling):
        if math.isnan(gu): #gro18 not run
            continue 
        if not r:
            continue
        if float(np.log2(mu)) < gl or float(np.log2(max(1,ml))) > gu:
            print(f"Error {mp.log(m,2)}: {gu} {gl}")

    df['hel_time'] = df['hel_time']*1000
    df['mc_time'] = df['mc_time']*1000
    
    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_uncertainty', c='y', label='HEL')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (rescaling)', c='b', ax=ax)
    df[df.rescaling==False].plot.scatter(x='mc_rank_log', y='mc_uncertainty',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank uncertainty") # (ASCAD multilabel without permind)")
    plt.ylabel("uncertainty (bits)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.legend()
    plt.savefig(outdir / "ascad_multilabel_without_permind_comparison_uncertainty_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./ascad_multilabel_without_permind_comparison_uncertainty_by_rank_correlation.pdf",
        "Paper figure 8c."
    )

    ax = df.plot.scatter(x='hel_rank_rounded', y='hel_time', c='y', label='HEL')
    plt.yscale('log')
    df[df.rescaling==True].plot.scatter(x='mc_rank_log', y='mc_time', label='MCRank (rescaling)', c='b', ax=ax)
    df2 = df[df.rescaling==False]
    df2 = df2[[not np.isnan(x) for x in df2.mc_uncertainty]]
    df2.plot.scatter(x='mc_rank_log', y='mc_time',
            label='MCRank (no rescaling)', c='grey', ax=ax)
    plt.title("HEL-MCRank speed at uncertainty lower than 1 bit") # (ASCAD multilabel without permind)")
    plt.ylabel("execution time (ms)")
    plt.xlabel("$\\log2(\\textrm{predicted rank})$")
    plt.yscale('log')
    plt.savefig(outdir /
            "ascad_multilabel_without_permind_comparison_execution_time_by_rank_correlation.pdf", bbox_inches='tight')
    plt.cla()
    latex_add_fig(
        outdir,
        "./ascad_multilabel_without_permind_comparison_execution_time_by_rank_correlation.pdf",
        "Paper figure 8d."
    )

def parse_list(string, range_max):
    """
    Parse list argument
    """
    if string == "":
        return []

    if string == "a" or string == "all":
        return [i for i in range(range_max)]
    
    items = string.split(",")
    for item in items:
        if not item.isnumeric() or int(item) < 0 or int(item) >= range_max:
            raise argparse.ArgumentTypeError(f'Range values must be comma-separated numbers 0 <= x <= {range_max-1} or "a/all"')
    return [int(i) for i in items]

# List of all tests
tests = {
        0: {'run': run0,  'plt': plt0},  # Figures 1a, 1b, 1c
        1: {'run': run1,  'plt': plt1},  # Figure  2a
        2: {'run': run2,  'plt': plt2},  # Figures 2b, 2c
        3: {'run': run3,  'plt': plt3},  # Figure  2d
        4: {'run': run4,  'plt': plt4},  # Figure  3a, 3b
        5: {'run': run5,  'plt': plt5},  # Figure  4a, 4b
        6: {'run': run6,  'plt': plt6},  # Figures 5a, 5b 
        7: {'run': run7,  'plt': plt7},  # Figures 5c, 5d 
        8: {'run': run8,  'plt': plt8},  # Figure  6a, 6b
        9: {'run': run9,  'plt': plt9},  # Figure  7a, 7b
        10: {'run': run10,  'plt': plt10},  # Figure  8a, 8b
        11: {'run': run11,  'plt': plt11},  # Figure  8c, 8d
}

def main():
    """
    Run tests and plots.
    """

    # CLI
    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', type=str, default="")
    parser.add_argument('--plots', type=str, default="")
    parser.add_argument('--clean', type=bool, default=False)
    parser.add_argument('--outdir', type=str,
            default=str(Path.cwd() / "outdir"))
    args = parser.parse_args()

    # (Re)-create output directory
    outdir = Path(args.outdir)
    recreate_outdir(outdir, args.clean)
    
    # Run tests
    for test_run in parse_list(args.tests, len(tests)):
        tests[test_run]['run'](outdir)
    
    # Plots
    tests_run = parse_list(args.plots, len(tests))
    if len(tests_run) > 0:
        latex_init(outdir)
        for test_run in tests_run:
            tests[test_run]['plt'](outdir)
        latex_close(outdir)
        all_f = outdir / "all.tex"
        os.system(f"cd {outdir}; rubber -d all.tex")
        os.system(f"cd {outdir}; rubber --clean all.tex")

if __name__ == "__main__":
    main()
