# Install required tools

## Yosys (https://yosyshq.readthedocs.io/projects/yosys/en/latest/getting_started/installation.html)
git clone https://github.com/YosysHQ/yosys.git
cd yosys
git submodule update --init --recursive
sudo apt-get install gawk git make python3 lld bison clang flex \
   libffi-dev libfl-dev libreadline-dev pkg-config tcl-dev zlib1g-dev \
   graphviz xdot
sudo snap install cmake --classic
ccmake build

## SV to Verilog (https://github.com/zachjs/sv2v)
git clone https://github.com/zachjs/sv2v.git
cd sv2v
curl -sSL https://get.haskellstack.org/ | sh
make

## OpenROAD (To use OpenSTA)
Install using pre-built binaries: https://openroad-flow-scripts.readthedocs.io/en/latest/user/BuildWithPrebuilt.html


## Skywater PDK

## First install conda
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ./Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc

## Then get the liberty file (Try wwith  Python 3.11 if latest fails)
git clone https://github.com/google/skywater-pdk.git
SUBMODULE_VERSION=latest make submodules -j3 || make submodules -j1
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/
make timing
