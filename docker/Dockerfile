# NVIDIA Pytorch Base Container
FROM nvcr.io/nvidia/pytorch:22.01-py3

# Update apt and pip
RUN apt update
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Clone He repo and dependencies
RUN git clone https://github.com/jxhe/unify-parameter-efficient-tuning.git && \
    git clone git@github.com:jxhe/MAM-adapter.git && \
    git clone git@github.com:moses-smt/mosesdecoder.git

# Install other requirements
RUN cd MAM-adapter && \
    pip install -e . && \
    pip install datasets==1.11.0

VOLUME /home/ubuntu/work

    
