#!/usr/bin/python3.6

import numpy as np
from matplotlib import pyplot as plt
import aes

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

# A simple hamming weight model
def simulate(n_traces, fixed_key, loc, scale, seed):
    np.random.seed(seed)
    PLAINTEXTS = np.asarray([[np.random.randint(256) for i in range(16)] for j in
            range(n_traces)])
    if fixed_key:
        key = [np.random.randint(256) for i in range(16)]
        KEYS = np.asarray([key for j in range(n_traces)])
    else:
        KEYS = np.asarray([[np.random.randint(256) for i in range(16)] for j in
                range(n_traces)])
    TRACES = np.zeros((n_traces, 16))
    for i in range(n_traces):

        for bnum in range(16):
            p = PLAINTEXTS[i,bnum]
            k = KEYS[i,bnum]
            TRACES[i][bnum] = hw[sbox[p ^ k]] + np.random.normal(loc=loc,
                    scale=scale, size=None)

    return PLAINTEXTS, KEYS, TRACES

def depends(bnum):
    b = int(bnum / 4)
    if b == 0:
        return [0,5,10,15]
    elif b == 1:
        return [4,9,14,3]
    elif b == 2:
        return [8,13,2,7]
    else:
        return [12,1,6,11]

def round2(plaintext, key_0): #, key_1):

    # addroundkey(0)
    state = np.asarray(plaintext) ^ np.asarray(key_0)
    
    # subbytes
    state = np.asarray([sbox[s] for s in state])
    
    # shiftrows & mixcolumns
    # https://www.hindawi.com/journals/jece/2017/9828967/
    s = list(state)
    gm = lambda a,b : aes.AES().galois_multiplication(a,b)
    
    state[0] =  gm(2,s[0]) ^ gm(3,s[5]) ^ gm(1,s[10]) ^ gm(1,s[15])
    state[1] =  gm(1,s[0]) ^ gm(2,s[5]) ^ gm(3,s[10]) ^ gm(1,s[15])
    state[2] =  gm(1,s[0]) ^ gm(1,s[5]) ^ gm(2,s[10]) ^ gm(3,s[15])
    state[3] =  gm(3,s[0]) ^ gm(1,s[5]) ^ gm(1,s[10]) ^ gm(2,s[15])

    state[4] =  gm(2,s[4]) ^ gm(3,s[9]) ^ gm(1,s[14]) ^ gm(1,s[3])
    state[5] =  gm(1,s[4]) ^ gm(2,s[9]) ^ gm(3,s[14]) ^ gm(1,s[3])
    state[6] =  gm(1,s[4]) ^ gm(1,s[9]) ^ gm(2,s[14]) ^ gm(3,s[3])
    state[7] =  gm(3,s[4]) ^ gm(1,s[9]) ^ gm(1,s[14]) ^ gm(2,s[3])
    
    state[8] =  gm(2,s[8]) ^ gm(3,s[13]) ^ gm(1,s[2]) ^ gm(1,s[7])
    state[9] =  gm(1,s[8]) ^ gm(2,s[13]) ^ gm(3,s[2]) ^ gm(1,s[7])
    state[10] = gm(1,s[8]) ^ gm(1,s[13]) ^ gm(2,s[2]) ^ gm(3,s[7])
    state[11] = gm(3,s[8]) ^ gm(1,s[13]) ^ gm(1,s[2]) ^ gm(2,s[7])
    
    state[12] = gm(2,s[12]) ^ gm(3,s[1]) ^ gm(1,s[6]) ^ gm(1,s[11])
    state[13] = gm(1,s[12]) ^ gm(2,s[1]) ^ gm(3,s[6]) ^ gm(1,s[11])
    state[14] = gm(1,s[12]) ^ gm(1,s[1]) ^ gm(2,s[6]) ^ gm(3,s[11])
    state[15] = gm(3,s[12]) ^ gm(1,s[1]) ^ gm(1,s[6]) ^ gm(2,s[11])

    return state
    
    ## addroundkey(1)
    #state = state ^ np.array(key_1)
    #
    ## subbytes
    #state = np.asarray([sbox[s] for s in state])
    #print(state)

    #cipher, states = aes.AES().encrypt(plaintext, key_0+key_1, 32)
    #print(np.asarray(states[1]))

    #assert (state == states[1]).all()

def simulate256(n_traces, fixed_key, loc, scale, seed):
    np.random.seed(seed)
    PLAINTEXTS = np.asarray([[np.random.randint(256) for i in range(16)] for j in
            range(n_traces)])
    if fixed_key:
        key = [np.random.randint(256) for i in range(32)]
        KEYS = np.asarray([key for j in range(n_traces)])
    else:
        KEYS = np.asarray([[np.random.randint(256) for i in range(32)] for j in
                range(n_traces)])
    TRACES = []
    R2STATES = []
    n = lambda l, s : np.random.normal(loc=l, scale=s, size=None)
    for i, [plaintext, key] in enumerate(zip(PLAINTEXTS,KEYS)):
        s0 = plaintext ^ key[0:16]
        s1 = round2(plaintext, key[0:16])
        s2 = s1 ^ key[16:32]
    
        trace = [hw[sbox[x]] + n(loc,scale) for x in s0]
        trace += [hw[sbox[x]] + n(loc,scale) for x in s2]
        
        TRACES.append(trace)
        R2STATES.append(s1)
    TRACES = np.asarray(TRACES)
    R2STATES = np.asarray(R2STATES)

    return PLAINTEXTS, KEYS, R2STATES, TRACES

def aes16_round2(plaintext, key_0):

    # addroundkey(0)
    state = np.asarray(plaintext) ^ np.asarray(key_0)
    
    # subbytes
    state = np.asarray([sbox[s] for s in state])
    
    # shiftrows & mixcolumns
    # https://www.hindawi.com/journals/jece/2017/9828967/
    s = list(state)
    gm = lambda a,b : aes.AES().galois_multiplication(a,b)
    
    state[0] =  gm(2,s[0])
    
    return state
 
def simulate16(n_traces, fixed_key, loc, scale, seed):
    np.random.seed(seed)
    PLAINTEXTS = np.asarray([[np.random.randint(256) for i in range(1)] for j in
            range(n_traces)])
    if fixed_key:
        key = [np.random.randint(256) for i in range(2)]
        KEYS = np.asarray([key for j in range(n_traces)])
    else:
        KEYS = np.asarray([[np.random.randint(256) for i in range(2)] for j in
                range(n_traces)])
    TRACES = []
    R2STATES = []
    n = lambda l, s : np.random.normal(loc=l, scale=s, size=None)
    for i, [plaintext, key] in enumerate(zip(PLAINTEXTS,KEYS)):
        s0 = plaintext ^ key[0:1]
        s1 = aes16_round2(plaintext, key[0:1])
        s2 = s1 ^ key[1:2]
    
        trace = [hw[sbox[x]] + n(loc,scale) for x in s0]
        trace += [hw[sbox[x]] + n(loc,scale) for x in s2]
        
        TRACES.append(trace)
        R2STATES.append(s1)
    TRACES = np.asarray(TRACES)
    R2STATES = np.asarray(R2STATES)

    return PLAINTEXTS, KEYS, R2STATES, TRACES

def simulate_rsa(n_traces, n_key_bytes, fixed_key, loc, scale, seed):
    np.random.seed(seed)
    if fixed_key:
        key = [np.random.randint(256) for i in range(n_key_bytes)]
        KEYS = np.asarray([key for j in range(n_traces)])
    else:
        KEYS = np.asarray([[np.random.randint(256) for i in range(n_key_bytes)] for j in
                range(n_traces)])
    TRACES = []
    n = lambda l, s : np.random.normal(loc=l, scale=s, size=None)
    for i, key in enumerate(KEYS):
        trace = [x + n(loc,scale) for x in key]
        TRACES.append(trace)
    
    TRACES = np.asarray(TRACES)

    return KEYS, TRACES


