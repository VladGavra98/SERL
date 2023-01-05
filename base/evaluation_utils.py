"""  Validation script utils """
import os
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch

# my modules
import signals

# current module
from core import genetic_agent


@dataclass
class Stats:
    nmae : np.float16
    nmae_sd: np.float16
    sm: np.float16
    sm_sd: np.float16


def gen_refs(t_max: int, amp_times: np.array, ampl_max: float, num_trails: int = 10) -> list:
    """ Generates a list of reference smoothened step signals.
        They start at 0 and end the simulation time t_mx.
    Args:
        t_max (int): Episode time.
        amp_times (np.array): Starting times of each new step block.
        ampl_max (float): Maximum amplitude 9the signal is symmetric wrt. zero)
        num_trails (int, optional): Nubmer of random references. Defaults to 10.

    Returns:
        list: List of reference smoothened step objects with length equal to num_trails.
    """
    refs_lst = []

    for _ in range(num_trails):
        # Possible choices
        ampl_choices = np.linspace(-ampl_max, ampl_max, 6)

        # Generate random amplitudes
        amplitudes = np.random.choice(ampl_choices, size=6, replace=True)
        amplitudes[0] = 0.0

        # Disturb starting times for each step
        amp_times = [amp_times[0]] + [
            t + np.random.uniform(-0.05, 0.05) for t in amp_times[1:]
        ]

        # Buil step object
        _step = signals.SmoothedStepSequence(
            amp_times, amplitudes, smooth_width=t_max//10)
        refs_lst.append(_step)

    return refs_lst


def find_logs_path(logs_name : str, root_dir : str = './logs/wandb/') -> str:
    cwd = os.getcwd()
    pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
    wandb = pwd / Path(root_dir)
    for _path in wandb.iterdir():
        if _path.is_dir():
            if _path.stem.lower().endswith(logs_name):
                return wandb / _path
    return None


def load_pop(model_path: str, args):
    """ Load evolutionary population"""
    model_path = model_path / Path('files/')
    actor_path = os.path.join(model_path, 'evo_nets.pkl')

    agents_pop = []
    checkpoint = torch.load(actor_path)

    for _, model in checkpoint.items():
        _agent = genetic_agent.GeneticAgent(args)
        _agent.actor.load_state_dict(model)
        agents_pop.append(_agent)

    print("Genetic actors loaded from: " + str(actor_path))

    return agents_pop


def load_rl_agent(model_path: str, args):
    """ Load RL actor from model path according to configuration."""
    model_path = model_path / Path('files/')
    actor_path = os.path.join(model_path, 'rl_net.pkl')

    checkpoint = torch.load(actor_path)
    rl_agent = genetic_agent.GeneticAgent(args)
    rl_agent.actor.load_state_dict(checkpoint)

    print("RL actor loaded from: " + str(actor_path))

    return rl_agent