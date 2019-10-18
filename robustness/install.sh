#!/bin/bash

sudo apt update 
sudo apt upgrade
sudo apt install libomp-dev git python3 python3-venv python3-pip cmake
sudo pip3 install numpy scipy cython


git clone --recursive https://github.com/oxfordcontrol/osqp
cd osqp
mkdir build
cd build
cmake -G "Unix Makefiles" ..
cmake --build .
sudo make install
cd ../..

sudo cp /usr/local/lib/libosqp.so /usr/lib
sudo cp /usr/local/lib/libqdldl.so /usr/lib

sudo cp /usr/local/include/osqp/* /usr/include
sudo cp /usr/local/include/qdldl/* /usr/include

if [ "$1" == "-gpu" ]
then
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget http://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get updatesudo apt-get -y install cuda
fi 

make
