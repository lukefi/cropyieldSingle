# cropyieldSingle

This repository includes codes that are used to reshape histogram files from Sentinel-2 single image time series into analysis ready format and 
build neural network models to predict crop yield.


## processing

1. python/histo2stack.py

This code stacks all histogram files in a directory into one numpy array file.

2. python/stack2ARD.py

This code stacks all numpy array files for a given parameter (year) into one numpy array file.


3. python/run3DNeuralNet.py

This code trains LSTMsuperb and RNNsuperb models and calculates predictions for test set.
