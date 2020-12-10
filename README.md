# cropyieldSingle

This repository includes codes that are used to reshape histogram files from Sentinel-2 single image time series into analysis ready format and 
build neural network models to predict crop yield.

## Python environment

Python version used is Python 3.8.6. The codes will probably also work on earlier versions as well. Tensorflow version used is 2.3.1.

The file requirements.txt may be used to create a virtual Python environment using:

conda create --name <env> --file requirements.txt

## processing

1. python/histo2stack.py

This code stacks all histogram files in a directory into one numpy array file.

2. python/stack2ARD.py

This code stacks all numpy array files for a given parameter (year) into one numpy array file.


3. python/run3DNeuralNet.py

This code trains LSTMsuperb and RNNsuperb models and calculates predictions for test set.
