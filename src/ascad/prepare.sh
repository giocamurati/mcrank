# Copyright (C) 2022 ETH Zurich
# Copyright (C) 2022 Giovanni Camurati
# 
# Copyright (C) 2022 University of Genoa
# Copyright (C) 2022 Matteo Dell'Amico
# 
# Copyright (C) 2022 UC Louvain
# Copyright (C) 2022 François-Xavier Standaert

git clone https://github.com/ANSSI-FR/ASCAD

pushd .
cd ASCAD
git checkout 410b92bab3bc69502b43c5cc9ccdec74794870be
git apply ../ascad.patch

cd STM32_AES_v2
mkdir -p ASCAD_data/ASCAD_databases
cd ASCAD_data/ASCAD_databases
wget https://files.data.gouv.fr/anssi/ascadv2/ascadv2-extracted.h5
mkdir ../ASCAD_trained_models
cd ../ASCAD_trained_models
wget https://static.data.gouv.fr/resources/ascadv2/20210408-165909/ascadv2-multi-resnet-earlystopping.zip
wget https://static.data.gouv.fr/resources/ascadv2/20210409-105237/ascadv2-multi-resnet-wo-permind-earlystopping.zip
unzip ascadv2-multi-resnet-earlystopping.zip
unzip ascadv2-multi-resnet-wo-permind-earlystopping.zip
popd

pushd .
cd ASCAD
python ASCAD_test_models.py ./STM32_AES_v2/example_test_models_params
python ASCAD_test_models.py ./STM32_AES_v2/example_test_models_without_permind_params
popd
