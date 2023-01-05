"""  Validation script """
import argparse
from copy import deepcopy, copy
import os
from pathlib import Path
import random
import toml
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import signals

from core.utils import calc_nMAE, calc_smoothness, load_config
from parameters import Parameters
from evaluation_utils import Stats, load_pop, load_rl_agent, gen_refs, find_logs_path
from plotters.plot_utils import plot

# my modules
import envs.config

# current module
parser = argparse.ArgumentParser()
parser.add_argument('-env', help='Environment Choices: (LunarLanderContinuous-v2) (PHLab)',
                    type=str, default='PHlab_attitude_nominal')
parser.add_argument('-seed', help='Random seed to be used',
                    type=int, default=7)
parser.add_argument('-disable_cuda', help='Disables CUDA',
                    action='store_true', default=True)
parser.add_argument('-agent_name', help='Path to the agent to be evaluated',type = str,
                    required = True)

parser.add_argument('-eval_pop', default=False, action='store_true')
parser.add_argument('-eval_actor', default=False, action='store_true')
parser.add_argument('-index', type=int)
parser.add_argument('-eval_rl', default=False, action='store_true')

parser.add_argument('-plot_hist', default=False, action='store_true')
parser.add_argument('-save_plots', default=False, action='store_true')
parser.add_argument('-save_stats', default=False, action='store_true')
parser.add_argument('-save_trajectory', default=False, action='store_true')
parser.add_argument('-plot_spectra', default=False, action='store_true')
parser.add_argument('-verbose', default=False, action='store_true')

parser.add_argument('-num_trails', default=1, type=int)

cla, unknown = parser.parse_known_args()

# Inject the cla arguments in the parameters object
parameters = Parameters(cla)

# Create Global Env in Validation mode
t_max = 80
env = envs.config.select_env(cla.env)
env.set_eval_mode(t_max=t_max)


def evaluate(actor, **kwargs: dict):
    """ Simulate one episode in the global environment. """
    # reset env
    done = False
    obs = env.reset(**kwargs)

    # NOTE total nubmer of steps is unknwon but at most Tmax//dt
    x_lst, rewards, u_lst = [], [], []
    x_ctrl_lst = []
    errors = []
    ref_lst = []

    while not done:
        x_lst.append(env.x)
        u_lst.append(env.last_u)
        x_ctrl_lst.append(env.get_controlled_state())

        # select action:
        action = actor.select_action(obs)

        # Simulate one step in environment
        action = np.clip(action, -1, 1)

        ref_value = np.deg2rad(
            np.array([ref(env.t) for ref in env.ref]).flatten())
        obs, reward, done, _ = env.step(action.flatten())
        next_obs = obs

        if kwargs.get('verbose'):
            print(
                f'Action: {np.rad2deg(action)} -> deflection: {np.rad2deg(env.last_u)}')
            print(
                f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} alpha:{env.alpha:.03f}   V:{env.V:.03f} H:{env.H:.03f}')
            print(
                f'Error: {np.rad2deg(obs[:env.n_actions])} Reward: {reward:0.03f} \n \n')

        # Update
        obs = next_obs

        # save
        ref_lst.append(ref_value)
        errors.append(ref_value - x_ctrl_lst[-1])
        rewards.append(reward)

    env.finish()

    # Control inputs
    errors = np.asarray(errors)

    # Compute scaled smoothness fitness
    actions = np.asarray(u_lst)
    smoothness = calc_smoothness(actions, **kwargs)

    # Format data
    rewards = np.asarray(rewards).reshape((-1, 1))
    ref_values = np.array(ref_lst)
    data = np.concatenate((ref_values, actions, x_lst, rewards), axis=1)

    # Calculate nMAE:
    nmae = calc_nMAE(errors)

    return data, nmae, smoothness


def validate_agent(gen_agent, user_refs_lst: list, num_trails: int = 1, **kwargs) -> tuple:
    """ Run evaluation over the user specified number of trails.
        The time traces from data come from he LAST episode played.
    """
    agent_nmae_lst, agent_sm_lst = [], []

    for i in tqdm(range(num_trails+1), total=num_trails):
        ref_tup = user_refs_lst[i]
        user_refs = {
            'theta_ref': ref_tup[0],
            'phi_ref': ref_tup[1],
        }
        data, _nmae, _smoothness = evaluate(
            gen_agent.actor, user_refs=user_refs, **kwargs)
        agent_nmae_lst.append(_nmae)
        agent_sm_lst.append(_smoothness)

    nmae = np.average(agent_nmae_lst[:])
    nmae_sd = np.std(agent_nmae_lst)
    smoothness = np.average(agent_sm_lst[:])
    sm_sd = np.std(agent_sm_lst[:])

    if kwargs.get('stdout'):
        print(f'nMAE: {nmae:0.1f}% with STD: {nmae_sd:0.1f}')
        print(f'Smoothness: {smoothness:0.0f} with STD: {sm_sd:0.1f}')

    stats = Stats(nmae, nmae_sd, smoothness, sm_sd)
    return data, stats



def main():
    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]

    # Identify fault:
    _, _, fault_name = cla.env.split('_')

    '''             Prepare References             '''
    # Time set-points
    time_array = np.linspace(0., t_max, 6)
    # time_array = [0,5,30,40,60,80]   # -- Perter's stall ref

    # Pitch set-points:
    amp1 = [0, 12, 3, -4, -8, 2]
    # amp1 = [0,30,45,15,0,0]   # -- Perter's stall ref
    base_ref_theta = signals.SmoothedStepSequence(
        time_array, amp1, smooth_width=t_max//10)


    # Roll set-points:
    amp2 = [2, -2, 2, 10, 2, -6]
    base_ref_phi = signals.SmoothedStepSequence(
        time_array, amp2, smooth_width=t_max//10)
    # base_ref_phi = signals.Const(0, t_max, value=0.0)

    # Build list of reference tuples from seed
    theta_refs = gen_refs(t_max, time_array, 12.0, num_trails=cla.num_trails)
    phi_refs = gen_refs(t_max, time_array, 10.0, num_trails=cla.num_trails)
    theta_refs.append(base_ref_theta)
    phi_refs.append(base_ref_phi)
    user_eval_refs = list(zip(theta_refs, phi_refs))


    # Load agent info
    logs_dir = find_logs_path(cla.agent_name)
    model_config = load_config(logs_dir)

    # build path to figures (might not be used)
    figpath = logs_dir / Path('figures')
    faultpath = figpath / Path(fault_name)

    # Update config
    parameters.update_from_dict(model_config)
    setattr(parameters, 'env_name', cla.env)
    parameters.stdout()

    '''              Load Population              '''
    if cla.eval_pop or cla.eval_actor:
        pop = load_pop(logs_dir, args=parameters)

    '''              Evaluation of one Actor              '''
    if cla.eval_actor:
        if cla.index is not None:
            idx = cla.index
        else:
            idx = int(
                input(f'Please provide index of the agent between 0 and {len(pop)}: '))

        data, stats = validate_agent(
            pop[idx], user_eval_refs, cla.num_trails, stdout=True, plot_spectra=cla.plot_spectra)
        _fig, _ = plot(
            data, name=f'nMAE: {stats.nmae:0.1f}%      Smoothness: {stats.sm:0.0f}', fault=fault_name)

        if cla.save_plots:
            if not os.path.exists(figpath):
                os.mkdir(figpath)
            if not os.path.exists(faultpath):
                os.mkdir(faultpath)
            figname = faultpath / Path(f'actor{idx}'+'_' + fault_name + '.png')
            _fig.savefig(fname=figname, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        # Save trajectories on base reference
        if cla.save_trajectory: save_trajecotry(faultpath, data)


    '''              Entire Population Evaluation                '''
    if cla.eval_pop:
        nmae_lst, sm_lst = [], []

        nmae_min = 500
        # smoothness_min = -10 if fault_name != 'jr' else -25

        for i, agent in enumerate(pop):
            print('Actor:', i)
            data, stats = validate_agent(
                agent, user_eval_refs, cla.num_trails, stdout=cla.verbose)

            # save for logging
            sm_lst.append(stats.sm)
            nmae_lst.append(stats.nmae)

            # check for champion
            if stats.nmae < nmae_min:
                nmae_min = stats.nmae
                champion_data = copy(data)
                champ_stats = copy(stats)
                idx = i

        # Plot the champion actor
        print('Champion:', idx)
        champ_fig, _ = plot(
            champion_data, name=rf'nMAE: {champ_stats.nmae:0.1f}%      Smoothness: {champ_stats.sm:0.0f}', fault=fault_name)

        if cla.save_plots:
            if not os.path.exists(figpath):
                os.mkdir(figpath)
            if not os.path.exists(faultpath):
                os.mkdir(faultpath)
            figname = faultpath / \
                Path('champion'+'_' + 'actor' +
                     f'{idx}' + '_' + fault_name + '.png')
            champ_fig.savefig(fname=figname, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        print("Popualtion stats:")
        print(
            f'Average nMAE: {np.average(nmae_lst):0.1f}  with SD: {np.std(nmae_lst):0.1f}')
        print(
            f'Average smoothness : {np.average(sm_lst):0.1f}  with SD: {np.std(sm_lst):0.1f}')

        '''   Save data to files '''
        if cla.save_stats:
            # sm and nmae for entire popualtion
            save_path = faultpath / Path('final_performance.csv')
            with open(save_path, 'w+',  encoding='utf-8') as fp:
                for _sm, _nmae in zip(sm_lst, nmae_lst):
                    fp.write(f"{_sm},{_nmae}\n")   # save  to csv
            fp.close()

            # full statistics to
            toml_path = logs_dir / Path('stats.toml')
            stats_dict = {fault_name:  {
                'champion_idx': idx,
                'champion': champ_stats.__dict__,
                'average': {
                    'nmae': np.average(nmae_lst),
                    'nmae_sd': np.std(nmae_lst),
                    'sm': np.average(sm_lst),
                    'sm_sd': np.std(sm_lst),
                }
            }
            }

            with open(toml_path, 'a+', encoding='utf-8') as f:
                toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())
                f.write("\n\n")
            f.close()

    '''              RL   Evaluation               '''
    if cla.eval_rl:
        # Update config
        rl_parameters = deepcopy(parameters)
        rl_parameters.update_from_dict(model_config)

        rl_agent = load_rl_agent(logs_dir, args=rl_parameters)
        data_rl, rl_stats = validate_agent(
            rl_agent, user_eval_refs, cla.num_trails, stdout=True, plot_spectra=cla.plot_spectra)

        # Plot the RL actor
        rl_fig, _ = plot(
            data_rl, name=f'nMAE: {rl_stats.nmae:0.1f}%      Smoothness: {rl_stats.sm:0.0f}', fault=fault_name)

        if cla.save_plots:
            if not os.path.exists(figpath):
                os.mkdir(figpath)
            if not os.path.exists(faultpath):
                os.mkdir(faultpath)
            figname = faultpath / Path('rl'+'_' + fault_name + '.png')
            rl_fig.savefig(fname=figname, dpi=300, format='png')
            plt.close()
        else:
            plt.show()

        '''   Save data to files '''
        if cla.save_stats:
            toml_path = logs_dir / Path('stats.toml')
            stats_dict = {fault_name: rl_stats.__dict__}

            with open(toml_path, 'a+', encoding='utf-8') as f:
                f.write("\n\n")
                toml.dump(stats_dict, f, encoder=toml.TomlNumpyEncoder())

            f.close()

        # Save trajectories on base reference
        if cla.save_trajectory: save_trajecotry(faultpath, data_rl)

def save_trajecotry(faultpath, data):
    save_path = faultpath / Path('nominal_trajectory.csv')
    with open(save_path, 'w+', encoding='utf-8') as fp:
        np.savetxt(fp, data)

    print(f'Trajcegory saved as: {save_path}')
    fp.close()


if __name__ == '__main__':
    main()
