""" Evaluation runner for the INDI attitude controller.

Reproduces the same reference trajectories, environment configurations and
metrics (nMAE, smoothness) as ``evaluate.py`` so that the INDI baseline can be
compared directly against the SERL / TD3 agents.

Usage:
    python3.8 ./src/evaluate_indi.py -env=PHlab_attitude_nominal -num_trails=5 -save_stats -save_plots
"""

import argparse
import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, os.pardir))
for _p in (_BASE_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.utils import calc_nMAE, calc_smoothness
from core.indi import INDIController
from evaluation_utils import Stats, gen_refs
from plotters.plot_utils import plot

import envs.config
import signals

parser = argparse.ArgumentParser()
parser.add_argument('-env', type=str, default='PHlab_attitude_nominal')
parser.add_argument('-seed', type=int, default=7)
parser.add_argument('-num_trails', type=int, default=1)
parser.add_argument('-save_stats', action='store_true')
parser.add_argument('-save_plots', action='store_true')
parser.add_argument('-verbose', action='store_true')

# controller tuning (optional overrides)
parser.add_argument('-K_theta', type=float, default=1.5)
parser.add_argument('-K_phi', type=float, default=0.6)
parser.add_argument('-K_beta', type=float, default=0.7)
parser.add_argument('-K_rate', type=float, default=3.0)
parser.add_argument('-K_rate_yaw', type=float, default=3.0)

cla, _ = parser.parse_known_args()

t_max = 80
env = envs.config.select_env(cla.env)
env.set_eval_mode(t_max=t_max)


def evaluate(ctrl, user_refs, **kwargs):
    """ Simulate one episode under the given reference signals. """
    done = False
    obs = env.reset(user_refs=user_refs)
    ctrl.reset()
    ctrl.identify(env)

    x_lst, u_lst, x_ctrl_lst, ref_lst, errors = [], [], [], [], []

    while not done:
        x_lst.append(env.x)
        u_lst.append(env.last_u)
        x_ctrl_lst.append(env.get_controlled_state())

        ref_value = np.deg2rad(
            np.array([ref(env.t) for ref in env.ref]).flatten())
        delta_cmd = ctrl.update(env.x, ref_value)
        action = np.clip(env.unscale_action(delta_cmd), -1.0, 1.0)
        obs, _, done, _ = env.step(action)

        if kwargs.get('verbose'):
            print(
                f't:{env.t:0.2f} theta:{env.theta:.03f} phi:{env.phi:.03f} '
                f'beta:{env.beta:.03f}  de:{np.rad2deg(env.last_u[0]):.02f} '
                f'da:{np.rad2deg(env.last_u[1]):.02f} dr:{np.rad2deg(env.last_u[2]):.02f}')

        ref_lst.append(ref_value)
        errors.append(ref_value - x_ctrl_lst[-1])

    env.finish()

    errors = np.asarray(errors)
    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions)
    nmae = calc_nMAE(errors)

    data = np.concatenate(
        (np.array(ref_lst), actions, np.array(x_lst),
         np.zeros((len(x_lst), 1))), axis=1)
    return data, nmae, smoothness


def main():
    env.seed(cla.seed)
    torch.manual_seed(cla.seed)
    np.random.seed(cla.seed)
    random.seed(cla.seed)

    _, _, fault_name = cla.env.split('_')

    time_array = np.linspace(0., t_max, 6)
    amp1 = [0, 12, 3, -4, -8, 2]
    base_ref_theta = signals.SmoothedStepSequence(
        time_array, amp1, smooth_width=t_max // 10)
    amp2 = [2, -2, 2, 10, 2, -6]
    base_ref_phi = signals.SmoothedStepSequence(
        time_array, amp2, smooth_width=t_max // 10)

    theta_refs = gen_refs(t_max, time_array, 12.0, num_trails=cla.num_trails)
    phi_refs = gen_refs(t_max, time_array, 10.0, num_trails=cla.num_trails)
    theta_refs.append(base_ref_theta)
    phi_refs.append(base_ref_phi)
    user_eval_refs = list(zip(theta_refs, phi_refs))

    ctrl = INDIController(env, K_theta=cla.K_theta, K_phi=cla.K_phi,
                          K_beta=cla.K_beta, K_rate=cla.K_rate,
                          K_rate_yaw=cla.K_rate_yaw)

    nmae_lst, sm_lst = [], []
    for i in range(len(user_eval_refs)):
        ref_tup = user_eval_refs[i]
        user_refs = {'theta_ref': ref_tup[0], 'phi_ref': ref_tup[1]}
        data, _nmae, _sm = evaluate(ctrl, user_refs, verbose=cla.verbose)
        nmae_lst.append(_nmae)
        sm_lst.append(_sm)

    nmae = np.average(nmae_lst)
    nmae_sd = np.std(nmae_lst)
    sm = np.average(sm_lst)
    sm_sd = np.std(sm_lst)
    print(f'INDI nMAE: {nmae:0.1f}% +/- {nmae_sd:0.1f}   '
          f'Smoothness: {sm:0.0f} +/- {sm_sd:0.1f}')

    if cla.save_plots:
        plot(data, name=f'nMAE: {nmae:0.1f}%      Smoothness: {sm:0.0f}',
             fault=fault_name, savefigure=f'indi_{fault_name}.png')

    if cla.save_stats:
        out = Path('tmp')
        out.mkdir(exist_ok=True)
        toml_path = out / 'indi_stats.toml'
        import toml
        stats = Stats(nmae, nmae_sd, sm, sm_sd)
        with open(toml_path, 'a+', encoding='utf-8') as f:
            toml.dump({fault_name: stats.__dict__}, f,
                      encoder=toml.TomlNumpyEncoder())
            f.write("\n\n")
        print(f'Stats appended to: {toml_path}')


if __name__ == '__main__':
    main()
