#!/bin/bash
set -e
set -x

# REQUIRES python 3.7 tensorflow 1.15 cudnn 7.6.5


cd bert/
wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
unzip cased_L-12_H-768_A-12.zip
cd -

