# Stochastic Gradient Descent Methods and Uncertainty Quantification in Extended CLSNA Models

This repository supports the paper "Stochastic Gradient Descent Methods and Uncertainty Quantification in Extended CLSNA Models."

## Requirements

This repository requires the following libraries and packages to run the scripts:

- **Python**
- **PyTorch**: Install:
  ```bash
  pip install torch
  ```
- **NumPy**: Install:
  ```bash
  pip install numpy
  ```
- **SciPy**: Install:
  ```bash
  pip install scipy
  ```
- **Scikit-learn**: Install:
  ```bash
  pip install scikit-learn
  ```
- **Matplotlib**: Install:
  ```bash
  pip install matplotlib
  ```

To install all necessary packages, run:
```bash
pip install numpy scipy scikit-learn torch matplotlib
```

## Quick Start Guide

It is strongly recommended to run these scripts on a GPU for speed.

- **To run a simulation study on the extended CLSNA model**: Open `run_simulation.ipynb` in the `simulation/` folder using Jupyter Notebook and run all cells.
- **To analyze Twitter congressional hashtag networks with 207 nodes**: Open `207.ipynb` in the `X/207/` folder and run all cells. These nodes represent members consistently present throughout the study period.
- **To analyze Twitter congressional hashtag networks with all recorded nodes**: Open `550.ipynb` in the `X/550/` folder and run all cells.

## Code Overview

- `simulation/`: This directory includes `simulation.ipynb` and `run_simulation.ipynb`. 
  - `simulation.ipynb`: The primary script for simulating a dynamic network with changing membership and fitting the extended CLSNA model to the simulated data for inference.
  - `run_simulation.ipynb`: A batch script that automates the execution of the primary simulation for multiple iterations.

- `X/207/`: Contains the `207.ipynb` file, which is the primary script for analyzing Twitter congressional hashtag networks with 207 nodes. These nodes represent members who were consistently present throughout the study period.

- `X/550/`: Contains the `550.ipynb` file, analyzing the same Twitter congressional hashtag networks, but includes all nodes recorded during the study.

- `X/yearly/`: This directory contains raw network data files named `aggregated_network_hashtag_intersection_year***.csv`. Additionally, the `network_features.csv` file provides metadata about the actors in the network.

- `X/compare/`: The entry point here is `compare.ipynb`, which plots the trajectory of the mean latent positions of the members of each party, comparing the model fitting result from the reduced dataset in `207.ipynb` and the full dataset in `550.ipynb`.


## Implementation Details

Each notebook (`main.ipynb`, `simulate_congress.ipynb`, `207.ipynb`, and `550.ipynb`) can be executed from top to bottom. They call helper functions and classes from `utils.py` and `congress_utils.py`. Each notebook follows a unified structure with three primary steps:

1. **Initial CLSNA Model Fit**: Start with a CLSNA model with a higher-dimensional space than the intended dimension. The model is then fitted, utilizing the first `p` principal components of the fitted latent position as the initial values for the next step.

2. **Point Estimation**: A model with the targeted dimension is fitted. The outputs are point estimations for model parameters.

3. **Variance/Covariance Estimation**: Perform variance/covariance estimation for the parameters of interest.

Each step is unified under an SGD approach and utilizes a unified `nn.module` class from the PyTorch autodifferentiation library, defined in `utils.py` and `congress_utils.py`.

**Note:** The current implementation is specifically designed for a 2-party system.

## Reporting Bugs

To report bugs encountered while running the code, please contact Hancong Pan at hcpan@bu.edu.
