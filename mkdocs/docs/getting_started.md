# Getting Started

## Building TLTk

TLTK can be built from source or run through a Docker container. Running through a Docker container is straighforward since the environment and the dependencies are automatically installed. 

In the following, we provide instructions for both alternatives.

### Running through Docker (Windows or Linux)

Instructions to run the tool using Docker. 

* Install Docker https://docs.docker.com/get-docker/ .
* In terminal or command proment, enter the following command to pull the TLTk docker image
``` Bash
docker pull bardhh/tltk
```
* Once the docker image is pulled, a container may be intialized, in interactive mode, with the command:
``` Bash
docker run -it --name tltk_cont bardhh/tltk bash 
```
* Alternatively, to execute a script without entering the container:
``` Bash
docker exec -it tltk_cont bash -c 'cd demos && python phi_1.py' 
```

*Other useful commands*

* To copy a file to the container: 
``` Bash
docker cp source_file tltk_cont:/usr/src/tltk/destination_file
```

If you have completed these steps, continue to the next section. 

### Building from Source (Linux Only)


#### Downloading TLTk

TLTk is currenlty hosted on bitbucket and is downloaded with the git clone command 

```Bash
git clone ~~~~~~~~~~~~~~~~~~~~~~~~~
```
#### Dependencies for robustness calculation
The following section describes how to install TLTk manually. There is a script that will do it automaticly skip to the bottom of the section for instructions on how to use the script
##### Operating System
TLTk is tested on Ubuntu linux. It can be installed on any linux distribution but is untested. This guide will be focused installing on the Ubuntu distribution of linux. 

##### Installing Git
To download TLTk source git is needed. If you do not have git it can be downloaded with the command:
```Bash
sudo apt install git 
```

##### CPU Compiler

TLTk has been tested with the gcc compiler. If gcc is not on your system it can be installed with:
```Bash
sudo apt install gcc
```

##### Installing python3
```Bash
sudo apt install python3
```
##### GPU Compiler

!!! warning 
    Not required unless you are using a GPU
    
    
To compile the gpu code you need the NVCC compiler. This compiler can be found:
[Here](https://developer.nvidia.com/cuda-downloads)


##### Installing python packages
Next, we need to install the python repositories we need. To do this we will use pip3, which we installed in the previous step.
The libaries that TLTk need are numpy, scipy, and cython. 
To install these, you can run the following command:
```Bash
pip3 install --user numpy scipy cython
```

##### Installing MATLAB for SimuLink model simulations

!!! warning
    Only needed if planing on using TLTk with simulink
    
Detailed steps can be found [here](https://www.scivision.dev/matlab-engine-callable-from-python-how-to-install-and-setup/)

The following two commands need to be executed (depending on the MATLAB version and directory structure) for Linux using python3:
```Bash
cd /usr/local/matlab/extern/engines/python/ 
python3 setup.py build --build-base=$(mktemp -d) install
```

##### Install script
There is a script that installs all the needed packages. At the start of the script it runs an apt update and upgrade.
The script can be found at
```Bash
tltk/robustness/install.sh
```

#### Compiling TLTk
Once all the dependencies are installed, TLTk needs to be compiled. To do this, there is a Make file in 
```Bash
tltk/robustness/make
```
This make file is uses GNU make which can be installed with
```Bash
sudo apt install make
```
To make the gpu code the make file can be ran like this
```Bash
make gpu
```


## Running Your First Script

### Docker

### Source
