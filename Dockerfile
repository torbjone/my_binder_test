#
# NEURON Dockerfile
#

# Pull base image.
FROM continuumio/anaconda3

USER root


# Switch back to non-root user privledges
WORKDIR $HOME
USER main