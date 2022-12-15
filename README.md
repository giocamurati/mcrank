# MCRank

## MCRank: Monte Carlo Key Rank Estimation for Side-Channel Security Evaluations

MCrank is a key ranking algorithm based on a Monte Carlo approach. Key scores/probabilities from a side channel attack inform the sampling policy, bringing fast convergence. Automated rescaling of the scores/probabilities can be used when the sampling policy is not balanced.
MCRank is fast, scalable to large keys, and it can handle the case of side channel attacks that return non-independent probability distributions for each subkey.
This release contains both our Python implementation of MCRank and all the necessary code and data to replicate the results of our CHES 2023 paper "MCRank: Monte Carlo Key Rank Estimation for Side-Channel Security Evaluations".

[Paper at CHES 2023](https://tches.iacr.org/index.php/TCHES/article/view/9953)

## Requirements

For the paper, we run out experiments natively on an HP ENVY laptop (Intel(R) Core(TM)
i7-4700MQ CPU @2.40GHz, 11GiB Memory, Ubuntu 22.04) in a Conda environment.
To replicate our environment, you can either install the code natively or run a Vagrant
script to generate a VirtualBox VM occupying 14GB, with 12GB memory, and 4 CPUs.

## Install 

Please refer to the official [Vagrant](https://www.vagrantup.com/) and 
[Virtualbox](https://www.virtualbox.org/) documentation for installing and using vagrant.
In short, on Ubuntu ```sudo apt install virtualbox vagrant```.

The Vagrant VM is configured using the ```vagrant/Vagrantfile``` and ```vagrant/bootstrap.sh``` files.
On a Linux machine you can follow the self-explanatory commands in ```vagrant/boostrap.sh``` for a native installation (too see which commands require root privileges, run ```grep -nr sudo``` in the mcrank folder). 
You can edit the ```vagrant/Vagrantfile``` file in order to allocate more
resources to the VM (the same can be done from the Virtual Box graphical interface).

Useful commands:

```cd mcrank/vagrant```
* ```vagrant up``` # Turn the VM on, the first time the VM is created (slow)
* ```vagrant up --provision``` # To reprovision ```bootstrap.sh```
* ```vagrant ssh``` # Connect to the VM
* ```vagrant reload``` # Reload the VM after changin the Vagrantfile
* ```vagrant halt``` # Halt the VM
* ```vagrant destroy``` # Destroy the VM
* ```VboxManage export mcrank -o mcrank.ova``` # Export .ova
* ```VboxManage import mcrank.ova``` # Import .ova

The login is:
* user ```vagrant```
* password ```vagrant```

## Usage

1. Replicate CHES2022 images from data:
   ```
   cd mcrank/src/eval_ches22/
   python3 evaluation.py --tests 0 --plots a --outdir data/
   zathura data/all.pdf
   ```
   Note that tests 0 runs again, wherease tests 1-11 create
   the figures from the .csv files pre-stored in data/
2. Replicate CHES2022 experiments and figures:
   ```
   # tmux # Useful if working over ssh
   cd mcrank/src/eval_ches22/
   python3 evaluation.py --tests a --plots a --outdir outdir
   zathura outdir/all.pdf
   ```
   Note that experiments can take a long time (e.g., 24h), especially
   on a Virtual Machine. Depending on the machine, timing results
   and memory limits might differ from the paper though the 
   overall behavior should be the same.

   There are 12 tests (0 to 11). If you want, you can run
   only one or some of the tests by specifying their number. For example:
   ```
   python3 evaluation.py --tests 0 --plots 0 --outdir outdir
   zathura outdir/all.pdf
   ```

3. Run single experiments manually:
   ```
   cd mcrank/src/tests/
   python3 test_128.py --help
   python3 test_256.py --help
   python3 test_rsa.py --help
   ```
   
4. Use mcrank directly in your code:
   ```
   import mcrank as mc
   ```
   Make sure ```mcrank/src/mcrank/``` is on the import path.
   In general, you can import the other files as modules as well.

***Important Note (Octave vs. Matlab)***: In order to make our work more
accessible, this release uses ```Octave``` instead of ```Matlab``` to run
```mcrank/src/sota/gmbounds/gmbounds.m``` in test 8.
(see ```mcrank/src/sota/gmbounds/gmbounds.py```).
Octave cannot handle the biggest keys.
If you have a Matlab license, you can simply:
1. Install Matlab
2. Install the [Matlab engine for Python](https://ch.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html)
3. In ```gmbounds.py``` comment the code to call Octave and and uncomment the code to call Matlab
Note that if you run ```python3 evaluation.py --tests 0 --plots a --outdir data/``` to generate the plots from
out pre-recorded data, these has been obtained with Matlab for test 8.

## More details

The more important part of this repository is the Python implementation of
MCRank available in ```mcrank/src/mcrank/mcrank.py```. The algorithm and its
parameters are described in the paper (Section3.1, Algorithm 1) including its
extension with rescaling (Algorithm 2).

In order to test MCRank, we need to run a side channel attack on some traces.
The code in ```mcrank/src/utils/simulation.py``` generate simulated traces for
both template and attack as explained in Section 4 (setup). Template and
profiled correlation attacks are implemented in ```mcrank/src/utils/ta.py```
and described in Section 2.

In addition to template/correlation attacks on simulated traces, we also run
deep-learning attack on real traces from ASCADv2, described in Section 2. The
script ```mcrank/src/ascad/prepare.sh```, which is invoked during installation,
download ASCADv2 from the public repository and runs the attacks.  It applies a
small patch that allows retrieving the necessary input for MCRank.

In order to compare MCRank with the sota, the installation script install
GMBounds, HEL, and GRO18, described in Section 2 (state-of-the-art algorithms)
and Section 4 (baseline). The first is implemented with Matlab, the others in
C++. We replace Matlab with Octave for portability, but moving back to Matlab
is trivial (see Important Note above).  The python wrappers for the three
algorithms are ```mcrank/src/sota/gmbounds.py```,
```mcrank/src/sota/python_hel.py```, and ```mcrank/src/sota/python_gro18.py```.

In Section 4, the paper evaluates MCRank over the following main cases:
template/correlation attacks on AES-128 template attacks on AES-256, template
attacks on RSA, deep learning attacks on real traces (ASCADv2).  For each of
these scenarios, the corresponding script in ```mcrank/src/tests``` does the
following: (1) generate the simulated traces and run the attack or collect the
attack results from ASCAD, (2) run MCRank, (3) run the corresponding SOTA
algorithm (if applicable).  The test ```mcrank/src/tests/test_aes256.py```
differs from the other because it implements the extension described in section
3.2 (Dependent Probability Distributions).  In addition, the test
```mcrank/src/tests/test_short_independent.py``` evaluates MCRank on a toy
example with only two bytes, used to generate the plots in Section 3.1
(Visualization with a toy example).  Each test can be run as an individual
command line tool, but their real purpose is to be called in the automated
evaluation.  Tests accept parameters to configure the simulation (e.g., number
of traces, seeds for pseudo-random), the attacks (e.g., templates vs
correlation), mcrank (e.g., sample size, rescaling), and SOTA (e.g., which
algorithm to run, its parameters).  Default values are chosen to be reasonable
for a quick manual test.  The scripts return the values of rank estimation
(e.g., rank, bounds, execution time) for MCRank and for SOTA:

All experiments described in the paper are implemented in
```mcrank/src/eval_ches22/```.
Each experiment is described in detail in Section 4 and is based on the scenarios
described above. Essentially, each experiment is a wrapper of a scenario (e.g.,
run the AES-128 scenario for increasing sample size).
The evaluation script is subdivided in 12 tests, from "test0" to "test11", each
implementing one experiment. Each test is split in two functions: (1) the function
"run" that executes the experiment and saves the results as .csv file in the output
directory, and (2) the function "plt" that generate the 
plot, annotates it, and saves it as pdf in the output folder.
At the end of the tests, a tex file including all the figures is generated and compiled,
thus creating a unique result ```outdir/output.pdf```.
Since execution and plotting are split, it is also possible to run only the plotting
starting from the pre-recorded experiment results stored in 
```mcrank/src/eval_ches22/data```.
The mapping between figures in the paper and tests, also reported in the code, is:

- test0:  Figures 1a, 1b, 1c (Toy example for increasing sample size and with rescaling)
- test1:  Figure  2a (AES-128, bounds on the rank for increasing sample size, with and without bootstrapping)
- test2:  Figures 2b, 2c (AES-128, SEM and uncertainty for increasing sample size)
- test3:  Figure  2d (AES-128, rank distribution for increasing samples size)
- test4:  Figure  3a, 3b (AES-128, comparison with HEL, execution time for comparable uncertainty)
- test5:  Figure  4a, 4b (AES-256, uncertainty and execution time for increasing sample size)
- test6:  Figures 5a, 5b (RSA, comparison with Gro18 for increasing key size, uncertainty for similar execution time)
- test7:  Figures 5c, 5d (RSA, comparison with Gro18 for increasing key size, execution time for similar uncertainty)
- test8:  Figure  6a, 6b (RSA, comparison with Gmbounds for increasing key size, uncertainty and execution time)
- test9:  Figure  7a, 7b (AES-128, repeated for correlation attacks)
- test10: Figure  8a, 8b (ASCADv2-1, comparison with HEL, execution time for similar uncertainty)
- test11: Figure  8c, 8d (ASCADv2-2, comparison with HEL, execution time for similar uncertainty)

## Files

### In this repo

```
.
├── LICENSE
├── README.md
├── conda
│   └── environment.yml # Conda env description
├── src
│   ├── ascad # Scripts to prepare ASCAD code/data
│   │   ├── ascad.patch
│   │   └── prepare.sh
│   ├── eval_ches22 
│   │   ├── data # Pre-computed evaluation data
│   │   │   ├── test1.csv
│   │   │   ├── test10.csv
│   │   │   ├── test11.csv
│   │   │   ├── test2.csv
│   │   │   ├── test3.csv
│   │   │   ├── test4.csv
│   │   │   ├── test5.csv
│   │   │   ├── test6.csv
│   │   │   ├── test7.csv
│   │   │   ├── test8.csv
│   │   │   └── test9.csv
│   │   └── evaluation.py # Eval script (data/figures)
│   ├── mcrank
│   │   └── mcrank.py # MCRank itself
│   ├── sota # Wrappers for SOTA and script to download code
│   │   ├── gmbounds.py
│   │   ├── gmbounds.patch
│   │   ├── gro18.py
│   │   ├── hel.py
│   │   └── prepare.sh
│   ├── tests # Tests for various scenarios
│   │   ├── test_128.py
│   │   ├── test_256.py
│   │   ├── test_ascad.py
│   │   ├── test_rsa.py
│   │   ├── test_short.py
│   │   └── test_short_independent.py
│   └── utils # Side channel attacks and plot utils
│       ├── aes.py
│       ├── myplots.py
│       ├── simulation.py
│       └── ta.py
└── vagrant # Vagrant Virtual Box VM config
    ├── Vagrantfile
    └── bootstrap.sh
```

### In the VM after installation

```
.
├── miniconda3 # Miniconda install with Python3.9 and enviroment
└── montecarlo-key-ranking
    ├── LICENSE
    ├── README.md
    ├── conda
    │   └── environment.yml
    ├── src
    │   ├── ascad
    │   │   ├── ASCAD # ASCAD data and attacks
    │   │   ├── ascad.patch
    │   │   └── prepare.sh
    │   ├── eval_ches22
    │   │   ├── evaluation.py
    │   │   └── data
    │   ├── mcrank
    │   │   └── mcrank.py
    │   ├── sota
    │   │   ├── gmbounds # GMBounds code
    │   │   ├── gmbounds.py
    │   │   ├── gmbounds.patch
    │   │   ├── gro18.py
    │   │   ├── hel.py
    │   │   ├── prepare.sh
    │   │   ├── python_gro18 # GRO18 code
    │   │   └── python_hel # HEL code
    │   ├── tests
    │   │   ├── test_128.py
    │   │   ├── test_256.py
    │   │   ├── test_ascad.py
    │   │   ├── test_rsa.py
    │   │   ├── test_short.py
    │   │   └── test_short_independent.py
    │   └── utils
    │       ├── aes.py
    │       ├── myplots.py
    │       ├── simulation.py
    │       └── ta.py
    └── vagrant
        ├── Vagrantfile
        └── bootstrap.sh
```

## Troubleshoot

1. ***Increase VM memory": change ```vb.memory = "12288"``` in the Vagrantfile and run ```vagrant reload``` or use the VirtualBox GUI.
2. ***Increase VM CPUs": change ```vb.cpus = "4"``` in the Vagrantfile and run ```vagrant reload``` or use the VirtualBox GUI.

## License

Copyright (C) 2022 ETH Zurich

Copyright (C) 2022 Giovanni Camurati

Copyright (C) 2022 University of Genoa

Copyright (C) 2022 Matteo Dell'Amico

Copyright (C) 2022 UC Louvain

Copyright (C) 2022 François-Xavier Standaert

This mcrank repository contains one possible implementation of the paper
"MCRank: Monte Carlo Key Rank Estimation for Side-Channel Security Evaluations".

Certain files in this project may have specific licenses or copyright restrictions, as this project uses multiple open-source projects. Files in this project without any specific license can be assumed to follow the following general clause:

mcrank is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

ghost-peak is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with mcrank. If not, see http://www.gnu.org/licenses/.
