# Docker Commands

1. To build docker image:
```bash
sudo docker build -t rbector/w266-final-project:latest .
```
2. To run docker image on cpu device:
```bash
sudo docker run -it --rm -v /home/ubuntu/work/:/workspace/work rbector/w266-final-project:latest
```
