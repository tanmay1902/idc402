# Python scripts used in "Directed evolution of DNA strands effectively selects for reservoir computing networks capable of multiple tasks"

Set of python scripts used in "*Directed evolution of DNA strands effectively selects for reservoir computing networks capable of multiple tasks*".
Tanmay Pandey, Petro Feketa, Jan Steinkühler

## Setup

To run the code and test the systems, you'll need an experiment file. All files used in our experiments are provided in the `experiment/` directory. To use them, simply move the desired experiment file to the root folder.

## Versions of Packages and Softwares used

Python 3.10.15

Modules:

> argparse              1.1
>
> deap                    1.4.1
>
> matplotlib            3.9.2
>
> networkx              3.4.2
>
> numpy                 2.1.3
> pandas                2.2.3
> scipy                   1.14.1
> sklearn                1.5.2

Other Modules:

> copy
>
> datetime
>
> pickle
>
> random
>
> concurrent
>
> functools

## Repository Structure

The *main* repository contains the file used in the simulation.

> .....
>
> ```
> |--> dataset
>     |--> Task1.mat ; the task 1 dataset
>     |--> Task2.mat ; the task 2 dataset
>     |--> Task3.mat ; the task 3 dataset
> |--> experiment
>     contains the files required for each experiment. All the experiments done in the paper
>     were done by moving the file into main repository, and tweaking as we want experiments.
> |--> functions
>     contains the helper functions for the simulation setup.
> |--> Generators
>     contains generator file for mackey glass and volterra dataset.
> ```

### Helper Functions

1. ***change_to_networkx.py***  : contains function to convert the network directory to networkx
2. ***e_distance.py*** : calculate eucledian distance between two points.
3. ***load_data_pkl.py*** : load the pickle file generated during DEAP simulation.
4. ***mse.py*** : calculate mean-squared-error and normalized-mean-squared-error
5. ***ode.py*** : calculate position and velocity of beads by ODE
6. ***random_connections.py*** : make random connections in the network with a probability.
7. ***random.py*** : the file to draw random array from given range
8. ***wlc.py*** : calculate force by DNA strands using worm-like-chain model
