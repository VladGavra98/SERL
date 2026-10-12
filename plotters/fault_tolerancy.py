import os
import argparse
from pathlib import Path
import itertools
import toml
import matplotlib.pyplot as plt
import scipy
import numpy as np
from plotters import mystyle
from plotters.mystyle import *


parser = argparse.ArgumentParser()
parser.add_argument('-num_agents', default=5, type=int)
parser.add_argument('-stats_type', type = str, default = 'champion')

cla, unknown = parser.parse_known_args()

fault_names = [ r'Nominal',
                r'Ice on Wings',
                r'CG Shifted Aft',
                r'Saturated Aileron',
                r'Saturated Elevator',
                r'Broken Elevator' ,
                r'Jammed Rudder']


def plot_barchart(rewards: list, deviations: list, fault_names: list, agent_names: list, title: str = '', xlabel: str = 'Fault Cases'):
    deviations = list(deviations)
    n_agents = len(agent_names)
    n_faults = len(rewards)

    width = 0.9  # the width of the bars
    x = np.linspace(0,  1.1*n_faults * n_agents,
                    n_faults)  # the labels' locations

    # f6, r_f6, dev_f6 = [fault_names[-1][:]], [rewards[-1][:]], [deviations[-1][:]]

    if n_agents == 1:
        offsets = [0]
        c_lst = [color_serl50]
    elif n_agents == 2:
        offsets = [-width/2, width/2]
        c_lst = [color_td3, color_serl50]
    elif n_agents == 3:
        offsets = [-width, 0, width]
        c_lst = [color_td3, color_serl10, color_serl50]
    elif n_agents == 5:
        offsets = [-2*width, -width, 0, width, 2*width]
        c_lst = [color_td3, color_serl10,
                 color_serl10, color_serl50, color_serl50]

    fig, axs = plt.subplots(1,2, gridspec_kw={'width_ratios': [7, 1]})
    ax, f6_ax = axs
    lines = []
    for i in range(n_faults-1):
        for j, offset in enumerate(offsets):
            lines.append(ax.bar(x=x[i] + offset,
                                height=rewards[i][j],
                                width=width,
                                yerr=deviations[i][j],
                                color=c_lst[j],
                                capsize=5,
                                ecolor=(0, 0, 0, 0.7),
                                hatch='||' if 'td3' not in agent_names[j].lower(
            ) else None,
                linestyle='--',)
            )

    for j, offset in enumerate(offsets):
        f6_ax.bar(x=x[-1] + offset,
                height=rewards[-1][j],
                width=width,
                yerr=deviations[-1][j],
                color=c_lst[j],
                capsize=5,
                ecolor=(0, 0, 0, 0.7),
                hatch='||' if 'td3' not in agent_names[j].lower() else None,linestyle='--',)

    labels = agent_names

    # Legend
    ax.legend(handles=lines,
            labels=labels,
            loc='upper left',
            ncol=1)

    ax.set_ylabel('nMAE [%]')
    f6_ax.set_ylabel('nMAE [%]')
    f6_ax.yaxis.set_label_position("right")
    f6_ax.yaxis.tick_right()
    # ax.set_xlabel(xlabel)

    ax.set_xticks(x[:-1], fault_names[:-1])
    f6_ax.set_xticks([x[-1]], [fault_names[-1]])
    ax.xaxis.set_tick_params(labelsize=17)
    f6_ax.xaxis.set_tick_params(labelsize=17)
    # ax.set_title(title)

    fig.tight_layout()

    return fig, ax


def get_stats_list(path_logs : str, stats_type : str = None):
    cwd = os.getcwd()
    toml_ = cwd / Path(path_logs) / Path('stats.toml')
    stats_dict = toml.load(toml_)

    # lists
    nmae_lst, sd_lst, identified_faults = [],[],[]

    for i, fault_name  in enumerate(fault_names):
        name = fault_name.split()[0].lower()
        local_key = ''.join([_token[0] for _token in fault_name.split()]).lower()

        if name.lower() in stats_dict.keys():
            key = name.lower()
        elif local_key in stats_dict.keys():
            key = local_key
        else:
            print(f'Stats for {fault_name} are missing from {path_logs}')
            continue

        if stats_type is not None:
            stats = stats_dict[key][stats_type]
        else:
            stats = stats_dict[key]

        nmae_lst.append(stats['nmae']); sd_lst.append(stats['nmae_sd'])

        # reanme fault with F-identifier
        identified_faults.append(fault_name  + f'\n(F{i})'  if i else fault_name)

    return nmae_lst,sd_lst, identified_faults




def main():
    '''      Change path to logs folder from wand    '''
    path_logs_erl50 = 'logs/wandb/run-20220924_144643-1xzaqiba_LONG'
    path_logs_erl10 = 'logs/wandb/run-20220913_165505-12zowviu_GLAD_SAFER'
    # path_logs_zesty = 'logs/wandb/run-20220905_171125-1e2q2ljf_ZESTY'
    # path_logs_zesty  = 'logs/wandb/run-20221102_100106-18zl7d6h_ZESTY_CAPS'
    path_logs_zesty  = 'logs/wandb/run-20221102_144601-1dixcrrl_ZESTY_CAPS'
    path_logs_longrl = 'logs/wandb/run-20220924_144643-1xzaqiba_LONG_RL'
    path_logs_gladrl = 'logs/wandb/run-20220913_165505-12zowviu_GLAD_SAFER_RL'


    if cla.num_agents == 5:
        logs = [path_logs_zesty, path_logs_gladrl,  path_logs_erl10, path_logs_longrl, path_logs_erl50, ]   # full
        agent_names = ['TD3', 'SERL(10) - TD3', 'SERL(10)' , 'SERL(50) - TD3', 'SERL(50)']
    elif cla.num_agents == 3:
        logs = [path_logs_zesty,  path_logs_erl10, path_logs_erl50, ]   # full
        agent_names = ['TD3', 'SERL(10)' , 'SERL(50)']
    elif cla.num_agents == 2:
        logs = [path_logs_zesty, path_logs_erl50]            # small
        agent_names = ['TD3', 'SERL(50)']

    assert len(logs) == len(agent_names)

    rewards, deviations = [],[]
    for agent_name, log_path in zip(agent_names, logs):
        if 'td3' in agent_name.lower():
            _nmae, _sd, _case   = get_stats_list(log_path)
        else:
            _nmae, _sd, _case   = get_stats_list(log_path, stats_type = cla.stats_type)
        rewards.append((_nmae))
        deviations.append(_sd)

    fault_names = _case
    rewards = list(map(tuple, itertools.zip_longest(*rewards, fillvalue=None)))
    deviations = list(map(tuple, itertools.zip_longest(*deviations, fillvalue=None)))


    # Errors:
    print(agent_names)
    for i, fault in enumerate(fault_names):
        print('\n',fault.rstrip(), end='\t')

        # The fist agent is the base for t-test:
        for j in range(len(agent_names)):
            _nmae = rewards[i][j]
            _dev = deviations[i][j]
            print(f'{_nmae:0.2f}+/-{_dev:0.1f}%', end=' ')

    print('\n\n')
    # Statistica tests:
    for i, fault in enumerate(fault_names):
        print('\n',fault)
        # The fist agent is the base for t-test:
        base_nmae = rewards[i][0]
        base_dev = deviations[i][0]
        for j in range(1,len(agent_names)):
            _nmae = rewards[i][j]
            _dev = deviations[i][j]
            if _nmae < base_nmae:
                t, p = scipy.stats.ttest_ind_from_stats(_nmae,_dev, 20, base_nmae, base_dev, 20, alternative = 'less')
                print(f'{agent_names[j]:<15} < TD3 by {(base_nmae - _nmae)/base_nmae:0.1%} (p = {p:.1g})')
            else:
                t, p = scipy.stats.ttest_ind_from_stats(_nmae,_dev, 20, base_nmae, base_dev, 20, alternative = 'greater')
                print(f'{agent_names[j]:<15} > TD3 by {(_nmae - base_nmae)/base_nmae:0.1%} (p = {p:.1g})')


    plot_barchart(rewards, deviations, fault_names, agent_names= agent_names, title = f'Fault-tolerancy: Population {cla.stats_type.capitalize()}')
    # plot_barchart(r_f6, dev_f6, f6, agent_names= agent_names, title = f'Fault-tolerancy: Population {cla.stats_type.capitalize()}')
    plt.show()


if __name__ == '__main__':
    main()