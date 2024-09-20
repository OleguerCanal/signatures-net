#!/bin/sh
#$ -N ae_optimizer
#$ -cwd
#$ -j y
#$ -t 1-1
#$ -q short-centos79
#$ -l h_rt=02:00:00
#$ -l virtual_free=8G
#$ -o Cluster/aeclassifier.out

workon env_SigNet

python optimize_ae_classifier.py
