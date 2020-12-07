# AI Arena

version 1.5.0

## Installation

Create a conda environment

```` sh
conda create -n arena5 python=3.7 anaconda
conda activate arena5
pip install -e .
pip install stable_baselines tensorflow==1.14.0
conda install -c conda-forge mpi4py

````

## Quickstart guide

```` sh
cd examples/touchdown
mpiexec -n 7 python my_main_script.py
````

Inspect the results after training

```` sh
cd log_comms/policy_2/plot.png
````

## Additional Documentation
 - [DRL Overview](docs/01_DRL_Overview.md) 
 - [AI Arena Structure](docs/02_AI_Arena_Structure.md)
 - [Supporting the OpenAI Gym interface](docs/03_Gym_Interface.md)
 - [AI Arena interface](docs/04_Arena_Interface.md)
 - [Configuration and Parameters](docs/05_Parameters.md)
 - [Tutorial 1 - Touchdown](docs/Tut_01.md)
 - [Tutorial 2 - Atari](docs/Tut_02_Atari.md)
 - Additional examples available in the exmamples directory

## License
MIT
