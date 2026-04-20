from Task1 import test_volterra_1 as t1
from Task2 import test_volterra_2 as t2
from Task3 import test_mackey as t3



from Task1 import test_volterra_genetic_all_params_brownian as t1b
from Task2 import test_volterra_genetic2_all_params_brownian as t2b
from Task3 import test_mackey_all_params_brownian as t3b


import numpy as np 
import pandas as pd 
from DEAP_functions import init_individual,global_network

dir_to_save = "./results/brownian"

individual =  init_individual(300)

t1_nmse = t1(global_network,individual,False)
t1b_nmse = t1b(global_network,individual)

print(f"Task 1::: W/O: {t1_nmse} | W/Brownian: {t1b_nmse}")

t2_nmse = t2(global_network,individual,False)
t2b_nmse = t2b(global_network,individual)

print(f"Task 2::: W/O: {t2_nmse} | W/Brownian: {t2b_nmse}")

t3_nmse = t3(global_network,individual,False)
t3b_nmse = t3b(global_network,individual)

print(f"Task 3::: W/O: {t3_nmse} | W/Brownian: {t3b_nmse}")