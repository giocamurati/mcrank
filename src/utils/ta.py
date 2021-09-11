#!/usr/bin/python3.6

import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal, norm, rankdata, pearsonr

# Utils from Chipwhisper
sbox=(
0x63,0x7c,0x77,0x7b,0xf2,0x6b,0x6f,0xc5,0x30,0x01,0x67,0x2b,0xfe,0xd7,0xab,0x76,
0xca,0x82,0xc9,0x7d,0xfa,0x59,0x47,0xf0,0xad,0xd4,0xa2,0xaf,0x9c,0xa4,0x72,0xc0,
0xb7,0xfd,0x93,0x26,0x36,0x3f,0xf7,0xcc,0x34,0xa5,0xe5,0xf1,0x71,0xd8,0x31,0x15,
0x04,0xc7,0x23,0xc3,0x18,0x96,0x05,0x9a,0x07,0x12,0x80,0xe2,0xeb,0x27,0xb2,0x75,
0x09,0x83,0x2c,0x1a,0x1b,0x6e,0x5a,0xa0,0x52,0x3b,0xd6,0xb3,0x29,0xe3,0x2f,0x84,
0x53,0xd1,0x00,0xed,0x20,0xfc,0xb1,0x5b,0x6a,0xcb,0xbe,0x39,0x4a,0x4c,0x58,0xcf,
0xd0,0xef,0xaa,0xfb,0x43,0x4d,0x33,0x85,0x45,0xf9,0x02,0x7f,0x50,0x3c,0x9f,0xa8,
0x51,0xa3,0x40,0x8f,0x92,0x9d,0x38,0xf5,0xbc,0xb6,0xda,0x21,0x10,0xff,0xf3,0xd2,
0xcd,0x0c,0x13,0xec,0x5f,0x97,0x44,0x17,0xc4,0xa7,0x7e,0x3d,0x64,0x5d,0x19,0x73,
0x60,0x81,0x4f,0xdc,0x22,0x2a,0x90,0x88,0x46,0xee,0xb8,0x14,0xde,0x5e,0x0b,0xdb,
0xe0,0x32,0x3a,0x0a,0x49,0x06,0x24,0x5c,0xc2,0xd3,0xac,0x62,0x91,0x95,0xe4,0x79,
0xe7,0xc8,0x37,0x6d,0x8d,0xd5,0x4e,0xa9,0x6c,0x56,0xf4,0xea,0x65,0x7a,0xae,0x08,
0xba,0x78,0x25,0x2e,0x1c,0xa6,0xb4,0xc6,0xe8,0xdd,0x74,0x1f,0x4b,0xbd,0x8b,0x8a,
0x70,0x3e,0xb5,0x66,0x48,0x03,0xf6,0x0e,0x61,0x35,0x57,0xb9,0x86,0xc1,0x1d,0x9e,
0xe1,0xf8,0x98,0x11,0x69,0xd9,0x8e,0x94,0x9b,0x1e,0x87,0xe9,0xce,0x55,0x28,0xdf,
0x8c,0xa1,0x89,0x0d,0xbf,0xe6,0x42,0x68,0x41,0x99,0x2d,0x0f,0xb0,0x54,0xbb,0x16)

hw = [bin(n).count("1") for n in range(256)]

def compute_cov(x, y):
    # Find the covariance between two 1D lists (x and y).
    # Note that var(x) = cov(x, x)
    return np.cov(x, y)[0][1]

def print_result(bestguess,knownkey,pge):
    print("Best Key Guess: ", end=""),
    for b in bestguess: print(" %02x "%b, end="")
    print("")
    
    print("Known Key:      ", end=""),
    for b in knownkey: print(" %02x "%b, end="")
    print("")
    
    print("PGE:            ", end="")
    for b in pge: print("%03d "%b, end="")
    print("")

    print("SUCCESS:        ", end="")
    tot = 0
    for g,r in zip(bestguess,knownkey):
        if(g==r):
            print("  1 ", end="")
            tot += 1
        else:
            print("  0 ", end="")
    print("")
    print("NUMBER OF CORRECT BYTES: %d"%tot)

# The profiling
def profile(P_PLAINTEXTS, P_KEYS, P_TRACES, plot=False):
    
    Nsa = len(P_TRACES[0])
    Nbytes = len(P_KEYS[0])

    # Cluster the profiling traces based on Sbox
    SETS = [[[] for _ in range(9)] for b in range(Nbytes)]
    for bnum in range(Nbytes):
        for p,k,trace in zip(P_PLAINTEXTS[:,bnum], P_KEYS[:,bnum], P_TRACES):
            cla = hw[sbox[p ^ k]]
            SETS[bnum][cla].append(trace)
    
    # Estimate mean, variance, and standard deviation for each class, and the
    # average trace for all traces
    MEANS = np.zeros((Nbytes, 9, Nsa))
    VARS = np.zeros((Nbytes, 9, Nsa))
    
    PROFILE_MEAN_TRACE = np.average(P_TRACES, axis=0)
    MEANS = np.zeros((Nbytes, 9, Nsa))
    VARS = np.zeros((Nbytes, 9, Nsa))
        
    for bnum in range(Nbytes):
        for cla in range(9):
            MEANS[bnum][cla] = np.average(SETS[bnum][cla], axis=0)
            VARS[bnum][cla] = np.var(SETS[bnum][cla], axis=0)
    
    # Estimate the side-channel SNR
    SNRS = np.zeros((Nbytes, Nsa))
    for bnum in range(Nbytes):
        SNRS[bnum] = np.var(MEANS[bnum], axis=0) / np.average(VARS[bnum], axis=0)
    
    # Find the POIs
    POIS = np.zeros((Nbytes, 1), dtype=int)
    for bnum in range(Nbytes):
        POIS[bnum,0] = np.argmax(SNRS[bnum])
    if plot:
        for bnum in range(Nbytes):
            plt.plot(SNRS[bnum], label="Byte %d"%bnum)
            plt.plot(POIS[bnum,0], SNRS[bnum][POIS[bnum,0]], '*')
        plt.ylabel("SNR")
        plt.xlabel("Sample")
        plt.legend()
        plt.show()
        
    # Estimate the profile
    PROFILE_MEANS = np.zeros((Nbytes, 9, 1))
    PROFILE_COVS = np.zeros((Nbytes, 9, 1, 1))
    
    for bnum in range(Nbytes):
        for cla in range(9):
            for i in range(1):
                PROFILE_MEANS[bnum][cla][i] = MEANS[bnum][cla][POIS[bnum][i]]
                for j in range(1):
                    PROFILE_COVS[bnum][cla][i][j] = compute_cov(
                            np.asarray(SETS[bnum][cla])[:, POIS[bnum][i]],
                            np.asarray(SETS[bnum][cla])[:, POIS[bnum][j]])
    
    return POIS, PROFILE_MEANS, PROFILE_COVS

hw_sbox = np.array([hw[sbox[i]] for i in range(256)])
arange_256 = np.arange(256)


def attack_lp(pois, profile_means, profile_covs, plaintexts, traces):
    """All parameters are numpy arrays."""

    # shapes:
    # pois (16, 1)
    # profile_means (16, 9, 1)
    # profile_cov (16, 9, 1, 1)
    # plaintexts (20, 16)
    # keys (20, 16)
    # traces (20, 32)

    profile_covs = np.squeeze(profile_covs, (2, 3))  # equivalent to  profile_covs[:, :, 0, 0]
    pois = np.squeeze(pois, 1)
    profile_means = np.squeeze(profile_means, 2)

    normals = norm(profile_means, np.sqrt(profile_covs))
    cla = hw_sbox[plaintexts[..., np.newaxis] ^ arange_256]  # shape (20, 16, 256)
    trace_poi = traces[:, pois, np.newaxis]  # shape (20, 16, 1)
    log_pdf = np.nan_to_num(normals.logpdf(trace_poi), neginf=0)  # shape (20, 16, 9)
    x = np.take_along_axis(log_pdf, cla, axis=2)  # shape (20, 16, 256)
    return x.sum(axis=0)  # shape (16, 256)


def attack_vec(pois, profile_means, profile_covs, plaintexts, keys, traces, plot=False, printing=False):
    # shapes:
    # pois (16, 1)
    # profile_means (16, 9, 1)
    # profile_cov (16, 9, 1, 1)
    # plaintexts (20, 16)
    # keys (20, 16)
    # traces (20, 32)

    known_key = keys[0]
    log_prob = attack_lp(pois, profile_means, profile_covs, plaintexts, traces)
    best_guess = log_prob.argmax(axis=1)

    if plot:
        plt.subplots_adjust(hspace=1)
        for b_num in range(16):
            plt.subplot(4, 4, b_num+1)
            plt.plot(log_prob[b_num], label="log_prob[guess]")
            plt.axvline(x=known_key[b_num], color='r', label="key")
            plt.plot(best_guess[b_num], log_prob[b_num][best_guess[b_num]],
                    'g*', label="best guess")
            plt.title(f"log_prob[guess] for byte {b_num}")
        plt.legend()
        plt.show()

    if printing:
        pge = [rankdata(-p_k, method='ordinal')[b_key] - 1 for p_k, b_key in zip(log_prob, known_key)]
        print_result(best_guess, keys[0], pge)

    return log_prob, best_guess


# The attack
def attack(POIS, PROFILE_MEANS, PROFILE_COVS, A_PLAINTEXTS, A_KEYS,
        A_TRACES, plot=False):
    
    A_knownkey = A_KEYS[0]
    Nsa = len(A_TRACES[0])
   
    # Attack
    LOG_PROBA = [[0 for r in range(256)] for bnum in range(16)]
    bestguess = [0]*16
    pge = [256]*16
     
    pooled_cov = False
    for bnum in range(16):
        if pooled_cov:
            covs = np.average(PROFILE_COVS[bnum,:,0,0], axis = 0)
        else:
            covs = PROFILE_COVS[bnum][:,0,0]
    
        print("Subkey %2d"%bnum)
        # Running total of log P_k
        P_k = np.zeros(256)
        for j, trace in enumerate(A_TRACES):
            P_k_tmp = np.zeros(256)
            # Test each key
            for k in range(256):
                # Find p_{k,j}
                p = A_PLAINTEXTS[j][bnum]
                cla = hw[sbox[p ^ k]]
                if pooled_cov:
                    cov = covs
                else:
                    cov = covs[cla]

                rv = multivariate_normal(PROFILE_MEANS[bnum][cla][0], cov)
                p_kj = rv.pdf(A_TRACES[j][POIS[bnum,0]])
                
                # multiply with the normalization constant
                # TODO only univariate for now
                # p_kj *= (cov * np.sqrt(2*np.pi))
    
                # Add it to running total
                x = np.log(p_kj)
                if x == -np.inf:
                    # print "inf"
                    continue
                P_k_tmp[k] += x
            
            P_k += P_k_tmp
    
            pge[bnum] = list(P_k.argsort()[::-1]).index(A_knownkey[bnum])
            # if j % 5 == 0:
            # print(j, "pge ", pge[bnum])
        LOG_PROBA[bnum] = P_k
        bestguess[bnum] = P_k.argsort()[-1]

    if plot:
        plt.subplots_adjust(hspace = 1)
        for bnum in range(16):
            plt.subplot(4,4,bnum+1)
            plt.plot(LOG_PROBA[bnum], label="logproba[guess]")
            plt.axvline(x=A_knownkey[bnum], color='r', label="key")
            plt.plot(bestguess[bnum], LOG_PROBA[bnum][bestguess[bnum]],
                    'g*', label="bestguess")
            plt.title("log_proba[guess] for byte %d"%bnum)
        plt.legend()
        plt.show()
    
    print_result(bestguess, A_knownkey, pge)
    return LOG_PROBA, bestguess

# The profiling
def profile_rsa(P_KEYS, P_TRACES, plot=False):
    Nsa = len(P_TRACES[0])
    Nbytes = len(P_KEYS[0])
    
    PROFILE_MEANS = np.zeros((Nbytes, 256, 1))
    PROFILE_COVS = np.zeros((Nbytes, 256, 1, 1))
    POIS = np.zeros((Nbytes,1), dtype=int)
 
    for bnum in range(Nbytes):
        # Cluster the profiling traces based on Sbox
        SETS = [[] for _ in range(256)]
        for k,trace in zip(P_KEYS[:,bnum], P_TRACES):
            cla = k
            SETS[cla].append(trace)

        # Mean/Var
        MEANS = np.zeros((256, Nsa))
        VARS = np.zeros((256, Nsa))
        
        for cla in range(256):
            MEANS[cla] = np.average(SETS[cla], axis=0)
            VARS[cla] = np.var(SETS[cla], axis=0)
    
        # Estimate the side-channel SNR
        SNRS = np.zeros(Nsa)
        SNRS = np.var(MEANS, axis=0) / np.average(VARS, axis=0)
    
        # Find the POIs
        POIS[bnum,0] = np.argmax(SNRS)
        
        # Estimate the profile
        for cla in range(256):
            for i in range(1):
                PROFILE_MEANS[bnum][cla][i] = MEANS[cla][POIS[bnum,i]]
                for j in range(1):
                    PROFILE_COVS[bnum][cla][i][j] = compute_cov(
                            np.asarray(SETS[cla])[:, POIS[bnum,i]],
                            np.asarray(SETS[cla])[:, POIS[bnum,j]])
    
    return POIS, PROFILE_MEANS, PROFILE_COVS

def attack_vec_rsa(pois, profile_means, profile_covs, keys, traces, plot=False, printing=False):
    # shapes:
    # pois (16, 1)
    # profile_means (16, 256, 1)
    # profile_cov (16, 256, 1, 1)
    # keys (20, 16)
    # traces (20, 32)

    known_key = keys[0]

    Nbytes = len(keys[0])
    
    profile_covs = np.squeeze(profile_covs, (2, 3))  # equivalent to  profile_covs[:, :, 0, 0]
    pois = np.squeeze(pois, 1)
    profile_means = np.squeeze(profile_means, 2)

    normals = norm(profile_means, np.sqrt(profile_covs))
    cla = np.zeros((np.shape(keys)[0],np.shape(keys)[1],256), dtype=int)
    for i in range(np.shape(keys)[0]):
        for j in range(np.shape(keys)[1]):
            cla[i,j,:] = arange_256
    trace_poi = traces[:, pois, np.newaxis]
    log_pdf = np.nan_to_num(normals.logpdf(trace_poi), neginf=0) 
    x = np.take_along_axis(log_pdf, cla, axis=2)
    log_prob = x.sum(axis=0)
   
    best_guess = log_prob.argmax(axis=1)

    if plot:
        plt.subplots_adjust(hspace=1)
        for b_num in range(Nbytes):
            plt.subplot(4, 4, b_num+1)
            plt.plot(log_prob[b_num], label="log_prob[guess]")
            plt.axvline(x=known_key[b_num], color='r', label="key")
            plt.plot(best_guess[b_num], log_prob[b_num][best_guess[b_num]],
                    'g*', label="best guess")
            plt.title(f"log_prob[guess] for byte {b_num}")
        plt.legend()
        plt.show()

    if printing:
        pge = [rankdata(-p_k, method='ordinal')[b_key] - 1 for p_k, b_key in zip(log_prob, known_key)]
        print_result(best_guess, keys[0], pge)

    return log_prob, best_guess

def profiled_corr_attack(pois, profile_means, plaintexts, keys, traces, plot=False, printing=False):
    # shapes:
    # pois (16, 1)
    # profile_means (16, 9, 1)
    # plaintexts (20, 16)
    # keys (20, 16)
    # traces (20, 32)

    known_key = keys[0]
 
    pois = np.squeeze(pois, 1)
    profile_means = np.squeeze(profile_means, 2) # shape (16, 9)
    trace_poi = traces[:, pois]  # shape (20, 16)
    
    log_prob = np.zeros((len(known_key), 256))
    corr = np.zeros((len(known_key), 256))
    for bnum in range(len(known_key)):
        for kguess in arange_256:
            cla = hw_sbox[plaintexts[:,bnum] ^ kguess]
            predicted = profile_means[bnum,cla]
            r, p = pearsonr(predicted, trace_poi[:,bnum])
            corr[bnum, kguess] = r 
    
    log_prob = np.nan_to_num(np.log(np.abs(corr)), neginf=0)
    best_guess = log_prob.argmax(axis=1)
    
    if plot:
        plt.subplots_adjust(hspace=1)
        for b_num in range(16):
            plt.subplot(4, 4, b_num+1)
            plt.plot(log_prob[b_num], label="log_prob[guess]")
            plt.axvline(x=known_key[b_num], color='r', label="key")
            plt.plot(best_guess[b_num], log_prob[b_num][best_guess[b_num]],
                    'g*', label="best guess")
            plt.title(f"log_prob[guess] for byte {b_num}")
        plt.legend()
        plt.show()

    if printing:
        pge = [rankdata(-p_k, method='ordinal')[b_key] - 1 for p_k, b_key in zip(log_prob, known_key)]
        print_result(best_guess, keys[0], pge)

    return log_prob, best_guess


