import argparse
import os
from pathlib import Path
import toml
import matplotlib.pyplot as plt
import numpy as np

import mystyle
from mystyle import *


parser = argparse.ArgumentParser()
parser.add_argument('-num_agents', default=3, type=int)
parser.add_argument('-stats_type', type = str, default = 'champion')

cla, unknown = parser.parse_known_args()
def plot_barchart_validation(rewards: list, deviations: list, test_names: list, agent_names: list, stats_type: str = ''):
    n_agents = len(agent_names)
    n_cases = len(rewards[0])

    width = 0.3  # the width of the bars
    x = np.linspace(0, width * n_cases * n_agents,
                    n_agents)  # the label locations
    c_lst = [mystyle.color_serl10, mystyle.color_serl50, mystyle.color_td3]

    if n_cases == 2:
        offsets = [-width/2, width/2]
    elif n_cases == 3:
        offsets = [-width, 0, width]
    elif n_cases == 4:
        offsets = [-1.5*width, -0.5 * width, 0.5*width, 1.5*width]

    hatches = ['.', '/', '\\', '|']

    fig, ax = plt.subplots()
    lines = []
    for i in range(n_agents):
        for j in range(n_cases):
            lines.append(ax.bar(x[i] + offsets[j],
                                rewards[i][j],
                                width,
                                color='white',
                                hatch=hatches[j],
                                zorder=0,
                                # edgecolor = 'black' if 'nominal' in test_names[j].lower() else 'None',
                                ))
            ax.bar(x[i] + offsets[j],
                   rewards[i][j],
                   width,
                   yerr=deviations[i][j],
                   color=c_lst[i],
                   capsize=6,
                   ecolor=(0, 0, 0, 0.7),
                   # edgecolor = 'black' if 'nominal' in test_names[j].lower() else 'None',
                   hatch=hatches[j],
                   zorder=2,
                   lw=0.5)

    labels = test_names

    # Legend
    plt.legend(handles=lines,
               labels=labels,
               loc='upper center',
               ncol=1,
               frameon=True)

    # ax.set_title(f'Robustness - Population {stats_type.capitalize()}')
    ax.set_xlabel('Intelligent Controller')
    ax.set_xticks(x, agent_names)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.set_ylabel('nMAE [%]')

    fig.canvas.manager.set_window_title("validation_barchart_"+ stats_type)
    fig.tight_layout()

    return fig, ax



fault_names = [ 'Nominal',
                "High-q",
                "Low-q",
                "Gust"
                ]


def get_stats_list(path_logs : str, stats_type : str = None):
    cwd = os.getcwd()
    toml_ = cwd / Path(path_logs) / Path('stats.toml')
    stats_dict = toml.load(toml_)

    # lists
    nmae_lst, sd_lst, identified_faults = [],[],[]

    for _, fault_name  in enumerate(fault_names):
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
        identified_faults.append(fault_name)

    return nmae_lst,sd_lst, identified_faults



def main():
    '''      Change path to logs folder from wand    '''
    path_logs_erl50 = 'logs/wandb/run-20220924_144643-1xzaqiba_LONG'
    path_logs_erl10 = 'logs/wandb/run-20220913_165505-12zowviu_GLAD_SAFER'
    path_td3 = 'logs/wandb/run-20221102_144601-1dixcrrl_ZESTY_CAPS'


    if cla.num_agents == 1:
        logs = [path_logs_erl50]
        agent_names = ['SERL(50)']
    elif cla.num_agents == 2:
        logs = [path_logs_erl10, path_logs_erl50]
        agent_names = ['SERL(10)', 'SERL(50)']
    elif cla.num_agents == 3:
        logs = [path_logs_erl10, path_logs_erl50, path_td3]
        agent_names = ['SERL(10)', 'SERL(50)', 'TD3']

    assert len(logs) == len(agent_names)

    rewards, deviations = [],[]
    for agent_name, log_path in zip(agent_names, logs):
        if 'td3' in agent_name.lower():
            _nmae, _sd, _case   = get_stats_list(log_path)
        else:
            _nmae, _sd, _case   = get_stats_list(log_path, stats_type = cla.stats_type)
        rewards.append((_nmae))
        deviations.append(_sd)

    # regroup stats
    test_names = _case

    for  i,_ in enumerate(test_names):
        if '-q' in test_names[i].lower():
            test_names[i] = f'(R{i}) ' + test_names[i].replace('-q',' Dynamic Pressure')
            print(test_names[i])
        elif 'gust' in test_names[i].lower():
            test_names[i] = f'(R{i}) Vertical Gust + Sensor Noise'
            print(test_names[i])


    rewards = [tuple(_r) for _r in rewards]
    deviations = [tuple(_d) for _d in deviations]

    print(rewards, deviations)
    # t-test:
    import scipy
    i = 0
    for nmae_tup,dev_tup in zip(rewards, deviations):
        print('\n\n',agent_names[i])
        nominal_nmae = nmae_tup[0]
        nominal_dev = dev_tup[0]
        i+=1
        j=0
        for _nmae,_dev in zip(nmae_tup[1:], dev_tup[1:]):
            j+=1
            _, p = scipy.stats.ttest_ind_from_stats(_nmae,_dev, 20, nominal_nmae, nominal_dev, 20)
            print(f'{test_names[j]:<15} != Nominal by {abs(nominal_nmae - _nmae)/nominal_nmae:0.1%} (p = {p:.1g})')
            # print(f'{agent_names[j]:<15} > TD3 by {(_nmae - nominal_nmae)/nominal_nmae:0.1%} (p = {p:.1g})')

    plot_barchart_validation(rewards, deviations, test_names, agent_names= agent_names, stats_type = cla.stats_type)
    plt.show()




if __name__ == '__main__':
    main()