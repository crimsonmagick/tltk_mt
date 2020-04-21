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

Instructions to install from source.

If you have completed these steps, continue to the next section. 

## Running Your First Script

### Docker

### Source