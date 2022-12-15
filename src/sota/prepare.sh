# Copyright (C) 2022 ETH Zurich
# Copyright (C) 2022 Giovanni Camurati
# 
# Copyright (C) 2022 University of Genoa
# Copyright (C) 2022 Matteo Dell'Amico
# 
# Copyright (C) 2022 UC Louvain
# Copyright (C) 2022 François-Xavier Standaert

# Install gmbounds
git clone https://gitlab.cs.pub.ro/marios.choudary/gmbounds.git
pushd .
cd gmbounds/
git checkout 05da0a56a262f78c46ffd3ddb7e245e423200520
git apply ../gmbounds.patch
popd

# Install dependencies for HEL and gro18
sudo apt-get install -y libntl-dev
sudo apt-get install -y libgmp-dev

# Install library and python wrapper python_hel
git clone https://github.com/giocamurati/python_hel.git
pushd .
cd python_hel/
git checkout 5d1d13319e7ff6bb13839c997a08b977c26e5fb5
pushd .
cd hel_wrapper
#make AES_TYPE=aes_ni # Intel AES NI
make TYPE=aes_simple  # Software AES
sudo make install
sudo ldconfig
popd 
cd python_hel/
python3 setup.py install
python3 python_hel/hel.py
popd

# Install library and python wrapper for gro18
git clone https://github.com/giocamurati/python_gro18.git
pushd .
cd python_gro18
git checkout 01236c47ef2e03e75d44abb5e57f70a13d9375a3
pushd .
cd lib/
make
sudo make install
sudo ldconfig
popd 
cd python_gro18/
python3 setup.py install
python3 python_gro18/gro18.py
popd


