# Docker Commands

1. To build docker image:
```bash
sudo docker build -f Dockerfile_torch -t rbector/w266-final-project:latest .
```
2. To run docker image on cpu device:
```bash
sudo docker run -it -v /home/ubuntu/w266_final_project:/workspace/w266_final_project -p 8888:8888 --gpus=all rbector/w266-final-project:latest
```
## Dockerfiles:

There are three docker files to build different images. The first is `Dockerfile_torch`, and was used to conduct all source tuning, target tuning and evaluation experiments. It consists of a clone of both the prompt tuning and PETL repositories. The dockerfile `Dockerfile_transformers` was used for baseline tuning. It simply contains a clone of the huggingface transformers repository, which was used as the repository for full fine-tuning BART on SuperGLUE tasks cast within the text-to-text framework. The third file is a deprecated tensorflow dockerfile. 
