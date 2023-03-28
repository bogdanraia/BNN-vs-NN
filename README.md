# BNN-vs-NN
Bayesian Neural Net in Julia vs Normal Neural Net in Python

# Libraries
Used TensorFlow in Python, and [PyCall](https://github.com/JuliaPy/PyCall.jl) to import TensorFlow in Julia too. 
sklearn in Python for the datasets, [SyntheticDatasets](https://docs.juliahub.com/SyntheticDatasets/zXq9q/0.1.0/) in Julia. <br/>
[Gen](https://www.gen.dev/) for inference, and besides those, just basic stuff like PyPlot, Metrics, etc.

# Datasets
Circles contains a classic dataset of two concentric circles, which clearly aren't linearly separable, so the neural nets actually need to do their job. <br/>
N-class is an artificial dataset with some spread, in 3D, with 4 classes. <br/>

# Architecture
The Neural Networks are 2 layers-deep but with not that many neurons. The Bayesian Neural Networks use those to compute training, updating the weights via MCMC.

# Results
The classic NN outperforms the BNN, although not by much.
