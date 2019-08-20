#
# NEURON Dockerfile
#

# Pull base image.
FROM continuumio/anaconda3

USER root

RUN wget https://neuron.yale.edu/ftp/neuron/versions/v7.7/nrn-7.7.x86_64-linux.deb
RUN dpkg -i nrn-7.7.x86_64-linux.deb

RUN pip install LFPy==2.0.3

# Switch back to non-root user privledges
WORKDIR $HOME
#USER main

