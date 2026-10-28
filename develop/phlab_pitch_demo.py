import os, sys
import numpy as np
import control as ct
import matplotlib.pyplot as plt

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, os.pardir))
for _p in (_BASE_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


import envs   # gym-like learning env
import envs.config
import signals  

T_max = 80 #s

env = envs.config.select_env("phlab_longitudinal")
env.set_eval_mode(t_max=T_max)





