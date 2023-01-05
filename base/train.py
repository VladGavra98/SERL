import os
import argparse
import time
import random
import numpy as np
from core import agent
import torch
from parameters import Parameters
import wandb

from core.utils import load_config
import envs.config


# -store_true means that it becomes true if I mention the argument
parser = argparse.ArgumentParser()

parser.add_argument('-should_log', help='Wether the WandB loggers are used', action='store_true')
parser.add_argument('-run_name', default='test', type=str)
parser.add_argument('-env', help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',type=str, default='PHlab_attitude_nominal')
parser.add_argument('-frames', help='Number of frames to learn from', type=int, required=True)

parser.add_argument('-pop_size', help='Population size (if 0 only RL learns)', default=10, type=int)
parser.add_argument('-champion_target', help='Use champion actor as target policy for critic update.', action='store_true')
parser.add_argument('-seed', help='Random seed to be used',type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA', action='store_true', default = True)
parser.add_argument('-use_caps', help='Use CPAS loss regularisation for smooth actions.', action='store_true', default=False)
parser.add_argument('-use_ounoise', help='Replace zero-mean Gaussian nosie with time-correletated OU noise', action='store_true')


parser.add_argument('-novelty', help='Use novelty exploration', action='store_true')
parser.add_argument('-mut_type', help='Type of mutation operator', type = str, default='proximal')
parser.add_argument('-use_distil', help='Use distilation crossover', action='store_true', default=False)
parser.add_argument('-distil_type', help='Use distilation crossover. Choices: (novelty)(fitness) (distance)',
                    type=str, default='fitness')

parser.add_argument('-test_ea', help='Test the EA loop and deactivate RL.', default= False, action='store_true')
parser.add_argument('-verbose_mut', help='Make mutations verbose', action='store_true')
parser.add_argument('-verbose_crossover',help='Make crossovers verbose', action='store_true')
parser.add_argument('-use_ddpg', help='Wether to use DDPG in place of TD3 for the RL part.',action='store_true')
parser.add_argument('-opstat', help='Store statistics for the variation operators', action='store_true')
parser.add_argument('-test_operators', help='Test the variational operators', action='store_true')

parser.add_argument('-per', help='Use Prioritised Experience Replay', action='store_true')
parser.add_argument('-sync_period', help="How often to sync to population", type=int, default =1)
parser.add_argument('-save_periodic', help='Save actor, critic and memory periodically', action='store_true')
parser.add_argument('-next_save', help='Generation save frequency for save_periodic', type=int, default=1000)

parser.add_argument('-config_path', help='Generation save frequency for save_periodic',
                    type=str, default=None)
parser.add_argument('-smooth_fitness', help='Added negative smoothness penalty to the fitness.', action='store_true')

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
    # strat trackers
    if cla.should_log:
        print('\033[1;32m WandB logging started')
        run = wandb.init(project="CAPS",
                        entity="vgavra",
                        dir='../logs',
                        name=cla.run_name,
                        config=params_dict)
        parameters.save_foldername = str(run.dir)
        wandb.config.update({"save_foldername": parameters.save_foldername,
                            "run_name": run.name}, allow_val_change=True)

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
            wandb.log(stats)                # main call to wandb logger


    # Save final model:
    elite_index = stats['elite_index']
    agent.save_agent(parameters, elite_index)

    if cla.should_log:
        run.finish()
