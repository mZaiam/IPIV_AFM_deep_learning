# IPIV - Classification of AFM curves using Deep Learning
This repository has the codes involving the IPIV project involving the classification of AFM curves using Deep Learning models. The project was done by Matheus Zaia Monteiro and Maria Eduarda de Oliveira Crist, under advisors Bruno Focassio and Yasmin Watanabe de Moura at LNNano-CNPEM.

# Repository Organization

This GitHub is organized as follows:
- `mnist/`: contains codes for the initial phase of the project, involving the classification of MNIST images;
- `datasets_afm/`: contains initial exploration of the data, including plots of the wavelets and the curves;
- `models/`: contains the actual models used for the project.

In particular, the directory `models/` is organized with the following structure:
- `conv1d/`: models trained with AFM curves, using 1D convolution;
- `conv2d/`: models trained with the wavelets of the AFM curves, using 2D convolution.

Each one of the directories above have the following models:
- `autoencoders/`: the best models are stored in `best_models/`. Further analysis, such as `classification/`, `clustering/` and `annotated/` are also included.
- `gans/`: the best models are stored in `best_models/`. Other architectures, such as `cgans/` and `acgans/` have the same structure.
