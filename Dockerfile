#
# NEURON Dockerfile
#

# Pull base image.
FROM continuumio/anaconda3

USER root

RUN conda config --add channels conda-forge
RUN conda install lfpy neuron=*=mpi*

# Switch back to non-root user privledges
WORKDIR $HOME
USER main