# Docker Commands

1. To build docker image:
```bash
sudo docker build -f <DOCKERFILE> -t <Your Dockerhub ID>/<Imagename>:<tag> .
```
2. To run The docker images:
```bash
sudo docker run -it -v /path/to/this/repo/w266_final_project:/workspace/w266_final_project -p 8888:8888 --gpus=all <Your Dockerhub ID>/<Imagename>:<tag>
```
## Dockerfiles:

There are three docker files to build different images. 

The first is `Dockerfile_torch`, and was used to conduct all source tuning, target tuning and evaluation experiments. It consists of a clone of both the prompt tuning and PETL repositories and is built on the NVIDIA official container registries for deep learning. All SPoT experiments were conducted in a containerised environment using this image. 

The dockerfile `Dockerfile_transformers` was used for full fine tuning. It simply contains a clone of the huggingface transformers repository, which was used as the repository for full fine-tuning BART on SuperGLUE tasks cast within the text-to-text framework. All attempts at replicating the full fine tuning experiments should be conducted from this container. 


The third file is a deprecated tensorflow dockerfile. 
