# Tutorial 01: Learning to Play Touchdown

## */examples/touchdown*
In the first tutorial we will walk through using the AI Arena to train an agent play a game called Touchdown.  Touchdown is a very simple game that is provided with the Arena examples.  It can be set up to run with N agents per team, using discrete or continuous actions separately for each team, and using image or vector state observations separately for each team.  It also trains very quickly.

Touchdown is a very simple game.  Players from each teach start at opposing ends of a square field.  The players may move in any direction.  If a player makes it past the opposing end-line, the game is won for their team.  However, if players from opposing teams collide, both players are reset back to their respective starting positions.

![diagram 1](diagrams/Tut01D1.png "Diagram_1")

## Directory Setup

Create a project directory "touchdown" (or just read along with the existing directory in examples).  This directory can be anywhere you typically like to work: we will be importing the Arena into our python scripts.  Copy the file examples/touchdown/Touchdown.py into this directory if you are writing this out rather than reading along.  To provide the Arena with the necessary information to learn Touchdown, we will break our program into 3 files: **make\_env.py**, **my\_config.py**, and **my\_main\_script.py**.  These can all be one file, but for complex uses of the Arena it is probably easier to keep these separate, so it is a good habit to develop.

## make\_env.py
Here we define a python method that the Arena can use to build an instance of the environment.  Complex examples may use this method to reference environment-specific parameters or call external scripts, but Touchdown is entirely in python so we will just return it as an object here.  The complete script is:

```python

from Touchdown import TouchdownEnv

def make_env():
	return TouchdownEnv(1, blue_obs="image", blue_actions="discrete",
		red_obs="image", red_actions="discrete")

```

## my\_config.py
Here we define a few constants that the Arena needs to run.  Firstly, a log directory.  The Arena uses this directory to save policies and make plots of training progress.  It is important that if you are running on a cluster, there is a shared file system such that this directory is accessible from any process.

```python
import os
LOG_COMMS_DIR = os.getcwd()+"/log_comms/"
```


We also tell the Arena about our environment, specifically what the format of states and actions will be.  The easiest way to do this is temporarily make an instance of our environment and read the fields **observation\_spaces** and **action\_spaces**:

```python
from make_env import make_env
temp = make_env()
OBS_SPACES = temp.observation_spaces
ACT_SPACES = temp.action_spaces
del temp
```


## my\_main\_script.py
Finally, our main script.  Crucially, _this script will run on each process that the Arena uses._ However, only the root process will make it to most of the script, while other scripts are held in work loops waiting for instructions.  This is all handled with our first command:

```python
from arena5.core.stems import *
from arena5.core.utils import mpi_print
from arena5.core.policy_record import *
import my_config as cfg
from make_env import make_env

arena = make_stem(make_env, cfg.LOG_COMMS_DIR, cfg.OBS_SPACES, cfg.ACT_SPACES)
```

The above command builds an Arena object on every process that the Arena has been given to run.  We pass in a method to create our environment, a directory to write logs to, and the information about state and action data.

Note that this function call ONLY returns for the root process.  From this point forward, we can imagine the script as single-threaded.

Next, we define which policies will exist to train.  This is provided by a dictionary.  The keys of this dictionary are unique integer IDs, while the values are strings that explain the type of algorithm to use.  For this example, we will train policy #1 using PPO, and policy #2 will simply take random actions.  Our definition looks like:
```python
policy_types = {1:"ppo", 2:"random"}
```

Now we define a set of matches that we want the Arena to run.  In this case we only have one match to play: Policy 1 vs Policy 2.  This is defined by a list of matches, where each entry is a list of policy IDs.  The order of the IDs corresponds with the order of the entities in the environment:

```python
match_list = [ [1,2] ] #One match, policy 1 vs policy 2
```

Above we are saying: I would like one match (one entry in the top level list), in which entity 1 will be assigned to policy 1 and entity 2 will be assigned to policy 2 (as we have defined policies 1 and 2 above).  Some other options could be:

[ [2,1] ] - One match, Entity 1 played by policy 2, and entity 2 by policy 1
[ [1,2], [2,1] ] - Two matches, each policy plays either side of the game
[ [1,1] ] - One match, Policy 1 plays both sides (plays against itself)

This can get complicated with more entities or non-sequentially ordered assignments.  See further tutorials for advanced usage.

Finally, we will kickoff a round of training.  The arguments are the **match_list** and **policy_types** as defined above, as well as how many steps we want to run in each match.  Some additional options are render=[False/True - whether or not to render the environments] and scale=[False/True - whether or not to replicate matches across all available processes].  These options are False by default, but we will enable them here.

```python
arena.kickoff(match_list, policy_types, 15000, render=True, scale=True)
```

That's it!  Beside import statements, our script to build the arena, define policies, define matches, and kickoff training is only 4 lines.


## Running the Script
To run our script, we will need to use MPI and provide at least 4 processes (1 environment, 2 policies with one worker each, and 1 root process).  We can easily accomplish this by running:
```
mpiexec -n 4 python my_main_script.py
```
This says: Execute with mpi, use 4 processes, and on them run: "python my\_main\_script".

Note: Since we set scale=True in arena.kickoff, we can run with more than 4 process if desired.  Each new match will require 3 more processes: 1 environment and 2 workers.  Therefore, running with 7 processes would run 2 copies of our setup, and 10 processes will run 3 copies.  The Arena will simply ignore processes that it does not need.  If you have a nice desktop or cluster, you can safely provide some number greater than or equal to 4 and see what happens.

Subsequent tutorials will not go into as much detail, but rather show variations on this basic example.
