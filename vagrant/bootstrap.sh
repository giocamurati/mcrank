# Copyright (C) 2022 ETH Zurich
# Copyright (C) 2022 Giovanni Camurati
# 
# Copyright (C) 2022 University of Genoa
# Copyright (C) 2022 Matteo Dell'Amico
# 
# Copyright (C) 2022 UC Louvain
# Copyright (C) 2022 François-Xavier Standaert

set -e # Exit on first error.
set -x # Print commands

sudo apt-get update
sudo apt-get install -y vim git tig bzip2 build-essential texlive dvipng \
	texlive-latex-extra texlive-fonts-extra cm-super rubber zathura \
	octave

# Install Miniconda
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh
bash Miniconda3-py39_4.12.0-Linux-x86_64.sh -b
rm Miniconda3-py39_4.12.0-Linux-x86_64.sh
echo 'export PATH="~/miniconda3/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

# Get MCRank
cd ~
git clone https://github.com/giocamurati/mcrank
cd mcrank
git checkout 0aba3e7810eb33440941379638c7bdbf8378451c

# Create mcrank python environment
export PATH="~/miniconda3/bin:$PATH"
source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f=conda/environment.yml
conda activate mcrank

# Install SOTA
pushd .
cd src/sota/
bash prepare.sh
popd

# Install ASCAD
pushd .
cd src/ascad/
bash prepare.sh
popd

# Install octave symbolic package
cd ~/mcrank/src/sota/
wget https://octave.sourceforge.io/download.php?package=symbolic-3.0.1.tar.gz
echo "pkg install download.php\?package\=symbolic-3.0.1.tar.gz" > /tmp/sym.m
octave --no-gui /tmp/sym.m
rm -rf https://octave.sourceforge.io/download.php?package=symbolic-3.0.1.tar.gz

# Always activate conda env
conda init
echo "conda activate mcrank" >> ~/.bashrc
