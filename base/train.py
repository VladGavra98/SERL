import os
import sys
import argparse
import time
import random
import numpy as np


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_BASE_DIR, os.pardir))

for _p in (_BASE_DIR, _ROOT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core import agent
import torch
from parameters import Parameters

from core.utils import load_config
import envs
import envs.config


parser = argparse.ArgumentParser()

parser.add_argument('-log','--should_log', help='Whether the TensorBoard loggers are used', action='store_true')
parser.add_argument('-name','--run_name', default='test', type=str)
parser.add_argument('-E','--env', help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',type=str, default='PHlab_attitude_nominal')
parser.add_argument('-F','--frames', help='Number of frames to learn from', type=int, required=True)

parser.add_argument('-N','--pop_size', help='Population size (if 0 only RL learns)', default=10, type=int)
parser.add_argument('--champion_target', help='Use champion actor as target policy for critic update.', action='store_true')
parser.add_argument('--seed', help='Random seed to be used',type=int, default=7)
parser.add_argument('--disable_cuda', help='Disables CUDA', action='store_true', default = True)
parser.add_argument('--use_caps', help='Use CPAS loss regularisation for smooth actions.', action='store_true', default=False)
parser.add_argument('--use_ounoise', help='Replace zero-mean Gaussian nosie with time-correletated OU noise', action='store_true')


parser.add_argument('--novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('--mut_type', help='Type of mutation operator', type = str, default='proximal')
parser.add_argument('--use_distil', help='Use distilation crossover', action='store_true', default=False)
parser.add_argument('--distil_type', help='Use distilation crossover. Choices: (novelty)(fitness) (distance)',
                    type=str, default='fitness')

parser.add_argument('--test_ea', help='Test the EA loop and deactivate RL.', default= False, action='store_true')
parser.add_argument('--verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover',help='Make crossovers verbose', action='store_true')
parser.add_argument('--use_ddpg', help='Wether to use DDPG in place of TD3 for the RL part.',action='store_true')
parser.add_argument('--opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('--test_operators', help='Test the variational operators', action='store_true')

parser.add_argument('--per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('--sync_period', help="How often to sync to population", type=int, default =1)
parser.add_argument('--save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('--next_save', help='Generation save frequency for save_periodic', type=int, default=1000)

parser.add_argument('--config_path', help='Generation save frequency for save_periodic',
                    type=str, default=None)
parser.add_argument('--smooth_fitness', help='Added negative smoothness penalty to the fitness.', action='store_true')

if __name__ == "__main__":
    cla = parser.parse_args()

    # Inject the cla arguments in the parameters object
    parameters = Parameters(cla)

    # Create Env
    env = envs.config.select_env(cla.env)
    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    if cla.config_path is not None:
        # Load config path:
        path = os.getcwd()
        pwd = os.path.abspath(os.path.join(path, os.pardir))
        config_path = pwd + cla.config_path
        config_dict = load_config(config_path)
        parameters.update_from_dict(config_dict)


    params_dict = parameters.__dict__
    # Start trackers
    if cla.should_log:
        from torch.utils.tensorboard import SummaryWriter
        print('\033[1;32m TensorBoard logging started')
        log_dir = os.path.join(_ROOT_DIR, 'logs', 'tensorboard', cla.run_name)
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)
        parameters.save_foldername = log_dir
        writer.add_text('run_name', cla.run_name)
        for key, value in params_dict.items():
            if isinstance(value, (int, float, bool)):
                writer.add_scalar('config/' + key, value)

    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)


    # Create Agent
    agent = agent.Agent(parameters, env)
    print('Running', parameters.env_name, ' State_dim:',
          parameters.state_dim, ' Action_dim:', parameters.action_dim)

    # Main training loop:
    start_time = time.time()

    while agent.num_frames <= parameters.num_frames:

        # evaluate over all episodes
        stats = agent.train()

        print('Epsiodes:', agent.num_episodes, 'Frames:', agent.num_frames,
              ' Train Max:', '%.2f' % stats['best_train_fitness'] if stats['best_train_fitness'] is not None else None,
              ' Test Max:', '%.2f' % stats['test_score'] if stats['test_score'] is not None else None,
              ' Test SD:', '%.2f' % stats['test_sd'] if stats['test_sd'] is not None else None,
              ' Population Avg:', '%.2f' % stats['pop_avg'],
              ' Weakest :', '%.2f' % stats['pop_min'],
              ' Novelty :', '%.2f' % stats['pop_novelty'],
              '\n',
              ' Avg. ep. len:', '%.2fs' % stats['avg_ep_len'],
              ' RL Reward:', '%.2f' % stats['rl_reward'],
              ' PG Objective:', '%.4f' % stats['PG_obj'],
              ' TD Loss:', '%.4f' % stats['TD_loss'],
              '\n')


        # Update loggers:
        stats['frames'] = agent.num_frames; stats['episodes'] = agent.num_episodes
        stats['time'] = time.time() - start_time
        if len(agent.pop):
            stats['rl_elite_fraction'] = agent.evolver.selection_stats['elite'] / \
                agent.evolver.selection_stats['total']
            stats['rl_selected_fraction'] = agent.evolver.selection_stats['selected'] / \
                agent.evolver.selection_stats['total']
            stats['rl_discarded_fraction'] = agent.evolver.selection_stats['discarded'] / \
                agent.evolver.selection_stats['total']

        if cla.should_log:
            for key, value in stats.items():
                if isinstance(value, (int, float, np.floating)) and value is not None:
                    writer.add_scalar(key, value, global_step=agent.num_frames)  # main call to tensorboard logger


    # Save final model:
    elite_index = stats['elite_index']
    agent.save_agent(parameters, elite_index)

    if cla.should_log:
        writer.close()
