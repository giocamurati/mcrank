# MCRank

## MCRank: Monte Carlo Key Rank Estimation for Side-Channel Security Evaluations

MCrank is a key ranking algorithm based on a Monte Carlo approach. Key scores/probabilities from a side channel attack inform the sampling policy, bringing fast convergence. Automated rescaling of the scores/probabilities can be used when the sampling policy is not balanced.
MCRank is fast, scalable to large keys, and it can handle the case of side channel attacks that return non-independent probability distributions for each subkey.
This release contains both our Python implementation of MCRank and all the necessary code and data to replicate the results of our CHES 2023 paper "MCRank: Monte Carlo Key Rank Estimation for Side-Channel Security Evaluations".

## Install 

We provide a Vagrant Virtual Machine to replicate our environment.
Please refer to the official [Vagrant](https://www.vagrantup.com/) and 
[Virtualbox](https://www.virtualbox.org/) documentation for installing and using vagrant.
In short, on Ubuntu ```sudo apt install virtualbox vagrant```.

The Vagrant VM is configured using the ```vagrant/Vagrantfile``` and ```vagrant/bootstrap.sh``` files.
On a Linux machine you can follow the self-explanatory commands in ```vagrant/boostrap.sh``` for a native installation.
You can edit the ```vagrant/Vagrantfile``` file in order to allocate more
resources to the VM (the same can be done from the Virtual Box graphical interface).

Useful commands:

```cd mcrank/vagrant```
* ```vagrant up``` # Turn the VM on, the first time the VM is created (slow)
* ```vagrant up --provision``` # To reprovision ```bootstrap.sh```
* ```vagrant ssh``` # Connect to the VM
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
3. Run single experiments manually:
   ```
   cd mcrank/src/eval_ches22/
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
