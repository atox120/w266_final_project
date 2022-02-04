# NVIDIA Pytorch Base Container
FROM nvcr.io/nvidia/pytorch:22.01-py3

# Update apt and pip
RUN apt update
RUN python3 -m pip install --no-cache-dir --upgrade pip

# Clone He repo and dependencies
RUN git clone https://github.com/jxhe/unify-parameter-efficient-tuning.git && \
    git clone https://github.com/google-research/prompt-tuning.git

# Install He repo requirements
RUN cd unify-parameter-efficient-tuning && \
    pip install -e . && \
    pip install datasets==1.11.0 && \
    cd ..

# VOLUME /home/ubuntu/work

    
