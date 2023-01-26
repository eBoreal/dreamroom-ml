# Must use a Cuda version 11+
FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

WORKDIR /

# Install git
RUN apt-get update && apt-get install -y git

# Install python packages
RUN pip3 install --upgrade pip
ADD requirements.txt requirements.txt
RUN pip3 install -r requirements.txt

# Add the banana boilerplate here
ADD server.py .

# Add  model weight files 
ADD download.py .
RUN python3 download.py


# Add the app code, init() and inference()
ADD app.py .
ADD utils.py .

# memory management
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

EXPOSE 8000

CMD python3 -u server.py
