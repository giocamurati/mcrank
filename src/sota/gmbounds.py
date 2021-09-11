#!/usr/bin/env python

"""
Python wrapper for gmbounds
"""

__author__ = "Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__copyright__ = "Copyright (C) 2022 ETH Zurich, University of Genoa, UC Louvain, \
                 Giovanni Camurati, Matteo Dell'Amico, François-Xavier Standaert"
__license__ = "GNU GPL v3"
__version__ = "1.0"

#https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html
#import ssl
#import matlab.engine
#eng = matlab.engine.start_matlab()
from oct2py import octave

import numpy as np
from scipy.special import logsumexp

#eng.addpath('~/mcrank/src/sota/gmbounds/')
octave.addpath('~/mcrank/src/sota/gmbounds/')

def compute_gm_bounds(lp: np.ndarray, dosym=0, sdigits=100):
    print("Running GMBounds")
    lp -= logsumexp(lp, axis=1)[:, np.newaxis]  # normalize probabilities
    prob_sum = np.exp(lp).cumsum(axis=1)  # cumulative sum of probabilities
    assert np.isclose(prob_sum[:, -1], 1).all()  # probabilities sum up to 1
    prob = np.exp(lp).T

    #output = eng.compute_gm_bounds(
    #        matlab.double(prob.tolist()),
    #        dosym,
    #        sdigits)

    if len(lp) > 500:
        print(f"Octave cannot handle key size > {len(lp)}")
        print(f"Use Matlab if you want to handle this case")
        return 0, 0, 0, 0, 0

    output = octave.compute_gm_bounds(
            octave.double(prob.tolist()),
            dosym,
            sdigits)

    gmlb = output[0][0]
    gmub = output[0][1]
    time = output[0][4]
    
    print(f"gmlb: 2^{gmlb}")
    print(f"gmub: 2^{gmub}")
    print(f"Accuracy {output[0][1]-output[0][0]} (bits)")
    print(f"Time: {time} (s)")
    print("")
    print("")

    return gmlb, gmub, 0, 0, time

if __name__ == "__main__":
    pass
