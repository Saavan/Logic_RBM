# Code Repository for "Logically Synthesized, Hardware-Accelerated, Restricted Boltzmann Machines for Combinatorial Optimization and Integer Factorization"
Public Code Repository for Nature Electronics "Logically Synthesized, Hardware-Accelerated, Restricted Boltzmann Machines for Combinatorial Optimization and Integer Factorization"

This repository is broken into tjree folders, a hardware code base, a software code base, and plotting codes for reproducing figures. The hardware codebase includes verilog code, which has been scrubbed of Xilinx and Xillybus IP, to reproduce results from the paper. The software codebase contains basic RBM starter code and training algorithms to reproduce the testing and training of our system. The plotting code takes outputs from both the hardware and software codebase and recreates results from the paper. 

Necessary python packages:
python
PyTorch
Numpy
Matplotlib
seaborn
joblib
yaml

Any issues with the repository, or extra data required: please contact Saavan Patel (saavan@berkeley.edu). 
