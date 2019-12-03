

## Installation

Create a conda environment

```` sh
conda create -n arena5 python=3.7 anaconda
conda activate arena5
pip install -e .
pip install stable_baselines tensorflow==1.14.0
conda install -c conda-forge mpi4py

````

## Examples

```` sh
cd examples/touchdown
mpiexec -n 7 python my_main_script.py
````

Inspect the results after training

```` sh
cd log_comms/policy_2/plot.png
````
