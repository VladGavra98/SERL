"""  Validation script """
import argparse
from pathlib import Path
from pprint import pprint
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import signals

from core.utils import load_config
from core.operator_runner import OperatorRunner

from parameters import Parameters
from evaluation_utils import load_pop, gen_refs, find_logs_path

from plotters.mystyle import *

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

parser.add_argument('-save_plots', default=False, action='store_true')
parser.add_argument('-save_stats', default=False, action='store_true')
parser.add_argument('-verbose', default=False, action='store_true')

parser.add_argument('-num_trails', default=0, type=int)


cla, unknown = parser.parse_known_args()

# Inject the cla arguments in the parameters object
parameters = Parameters(cla)

# Create Global Env in Validation mode
t_max = 20
env = envs.config.select_env(cla.env)
env.set_eval_mode(t_max=t_max)


def mutation_plots(mut_data):
    diamond = dict(markerfacecolor='r', marker='D')
    green_diamond = dict(markerfacecolor='g', marker='D')
    fig, ax = plt.subplots()
    ax.set_title('Distributions of Relative Reward Improvement')
    ax.set_ylabel('Relative Change - Average Reward')
    ax.set_xlabel('Mutation Operator')
    ax.boxplot(mut_data[0].values(), flierprops=diamond,
                   labels=mut_data[0].keys(), meanline=True)


    fig, ax = plt.subplots()
    ax.set_title('Distribution of Average Cost')
    ax.set_ylabel('Relative Change - Average Cost')
    ax.set_xlabel('Mutation Operator')
    ax.boxplot(mut_data[1].values(), flierprops=green_diamond,
               labels=mut_data[1].keys(), meanline=True)

    plt.tight_layout()

    return fig, ax


def main():
    # Seed
    env.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    np.random.seed(parameters.seed)
    random.seed(parameters.seed)

    parameters.action_dim = env.action_space.shape[0]
    parameters.state_dim = env.observation_space.shape[0]


    '''             Prepare References             '''
    # Time set-points
    time_array = np.linspace(0., t_max, 6)

    # Pitch set-points:
    amp1 = [0, 12, 3, -4, -8, 2]
    base_ref_theta = signals.SmoothedStepSequence(
        time_array, amp1, smooth_width=t_max//10)

    # Roll set-points:
    amp2 = [2, -2, 2, 10, 2, -6]
    base_ref_phi = signals.SmoothedStepSequence(
        time_array, amp2, smooth_width=t_max//10)

    # Build list of reference tuples from seed
    theta_refs = gen_refs(t_max, time_array, 12.0, num_trails=cla.num_trails)
    phi_refs = gen_refs(t_max, time_array, 10.0, num_trails=cla.num_trails)
    theta_refs.append(base_ref_theta)
    phi_refs.append(base_ref_phi)
    user_eval_refs = list(zip(theta_refs, phi_refs))

    # Load agent info
    logs_dir = find_logs_path(cla.agent_name)
    print(logs_dir)
    model_config = load_config(logs_dir)


    # Update config
    parameters.update_from_dict(model_config)
    setattr(parameters, 'env_name', cla.env)
    pprint(parameters.__dict__)

    '''              Load Population              '''
    pop = load_pop(logs_dir, args=parameters)


    '''              Mutation Evaluation                '''
    op_runner = OperatorRunner(parameters, env, num_trails=cla.num_trails)
    print('Running', parameters.env_name, ' State_dim:',
          parameters.state_dim, ' Action_dim:', parameters.action_dim)

    # Sources for models:

    mut_data = op_runner.test_mutation(pop, user_eval_refs)
    mutation_plots(mut_data)

    pprint({
        'normal_r': np.mean(mut_data[0]['Normal']),
        'proximal_r': np.mean(mut_data[0]['Proximal']),
        'safe_r': np.mean(mut_data[0]['Safe']),
    })
    pprint({
        'normal_c': np.mean(mut_data[1]['Normal']),
        'proximal_c': np.mean(mut_data[1]['Proximal']),
        'safe_c': np.mean(mut_data[1]['Safe']),
    })

    plt.show()

    '''   Save data to files '''
    import toml
    if cla.save_stats:
        stats_reward = {'stats_reward': mut_data[0]}
        stats_cost = {'stats_cost': mut_data[1]}
        # full statistics to
        toml_path = logs_dir / Path('mutation_stats.toml')

        with open(toml_path, 'w+', encoding='utf-8') as f:
            toml.dump(stats_cost, f, encoder=toml.TomlNumpyEncoder())
            f.write('\n')
            toml.dump(stats_reward, f, encoder=toml.TomlNumpyEncoder())
        f.close()



if __name__ == '__main__':
    main()
