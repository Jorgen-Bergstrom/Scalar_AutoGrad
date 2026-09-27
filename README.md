# Scalar_AutoGrad

A tiny reverse-mode automatic differentiation engine and a small neural-network
library, written from scratch in pure Python. A complete `Value` type, an MLP,
mini-batch SGD and a binary-classification example fit in a few hundred lines of Python code.

## Key Files
* `engine.py`: Python class (`Value`) for scalar forward/backward autograd.
* `test_autograd.py`: Short test that illustrates how to use the `engine.py` class.
* `nn.py`: Neuron, Lay, and MLP classes built on top of the `Value` class in `engine.py`.
* `cancer_prediction.py`: Binary classification of the Breast Cancer Wisconsin dataset
using the scalar autograd engine.
* `cancer_prediction_torch.py`: Same as `cancer_prediction.py` but using PyTorch instead of the autograd code.
* `cancer_prediction.cpp`: Single-file C++ port of `engine.py`, `nn.py` and `cancer_prediction.py`. Build with `g++ -O2 -std=c++17 cancer_prediction.cpp -o cancer_prediction`; it reads `breast_cancer.csv` and writes the training curves to `cancer_training_cpp.csv`.

The code is described [here](https://jorgen-bergstrom.github.io/posts/scalar_autograd/).
