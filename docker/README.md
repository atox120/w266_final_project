# Docker Commands

1. To build docker image:
```bash
sudo docker build -f Dockerfile_torch -t rbector/w266-final-project:latest .
```
2. To run docker image on cpu device:
```bash
sudo docker run -it --rm -v /home/ubuntu/work/:/workspace/work rbector/w266-final-project:latest
```

## Tensorflow

Image based on the tensorflow 2.0 image (v22.01) from [nvidia container registry](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow). It comes with some packages installed, see requirements.txt

1. To build dockerimage
```bash
sudo docker build -f <your_dockerhub_namespace>/w266-fp-tf2:<version>
```
Locally i've tagged the image as ```atox120/w266-fp-tf2:0.1``` and pushed it to dockerhub so you can also just pull the image from my namespace directly. 

2. To run image
```bash
sudo docker run -it --rm -v /home/ubuntu/work:/workspace/work atox120/w266-fp-tf2:0.1
```
with gpus and ports:
```bash
sudo docker run -it --rm -v /home/ubuntu/work:/workspace/work -p 8888:8888 --gpus=all atox120/w266-fp-tf2:0.1

```
