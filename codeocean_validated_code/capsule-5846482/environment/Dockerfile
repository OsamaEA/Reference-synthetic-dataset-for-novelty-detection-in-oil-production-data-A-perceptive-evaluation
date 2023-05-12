# hash:sha256:50a6addd34c079a32659ecfb0693b75a6ff15f815890136926d5ac3034e593d0
FROM registry.codeocean.com/codeocean/miniconda3:4.12.0-python3.9-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

RUN conda install -y \
        jupyter==1.0.0 \
        python==3.9.12 \
    && conda clean -ya

RUN pip install -U --no-cache-dir \
    combo==0.1.3 \
    matplotlib==3.7.1 \
    numpy==1.23.5 \
    openpyxl==3.1.2 \
    pandas==2.0.1 \
    pyod==1.0.9 \
    scikit-learn==1.2.2 \
    scipy==1.6.0
