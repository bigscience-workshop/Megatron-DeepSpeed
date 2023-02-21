FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-1.11-py38-cuda11.3-gpu
RUN pip install git+https://github.com/microsoft/DeepSpeed.git@olruwase/bf16-updates
RUN pip install transformers
RUN pip install setuptools_rust
RUN pip install datasets