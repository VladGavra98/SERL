import os
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
from matplotlib import patches
import wandb
import toml
import mystyle

from mystyle import *

def plot_scattered(names_lst, scores_lst, sm_lst, scores_sd, sm_sd, figname :str = None):
    """ Generates a scatter plot with smoothness on x-axis and reward on y-axis.
    """
    figname = str(figname)
    plt.figure(figname + '_scatter')
    a = plt.subplot(111)


    # # Pathces:
    e1 = patches.Rectangle((-1800, -160), 1800, 165,
                        angle=0, fill=True, zorder=1,facecolor = '#6F1D77', alpha=0.2)
    e2 = patches.Rectangle((-85, -2000), 85, 2000,
                        angle=0, fill=True, zorder=1,facecolor = c_alpha, alpha=0.2)
    a.add_artist(e1)
    a.add_artist(e2)

    for i, name in enumerate(names_lst):
        sm = sm_lst[i]
        score = scores_lst[i]
        edgecolor = None
        marker = 'o'
        if 'td3' in name.lower():
            # marker = 'o'
            color = mystyle.color_td3
        elif '50' in name.lower():
            # marker = 's'
            color = mystyle.color_serl50
        elif '10' in name.lower():
            # marker = '^'
            color= mystyle.color_serl10

        if 'caps' in name.lower() and '+' not in name.lower(): marker = '^'
        if 'sf' in name.lower() and '+' not in name.lower(): marker = 's'
        if  '+'  in name.lower(): edgecolor = 'black'

        plt.errorbar(
            sm, score, color=color, yerr=scores_sd[i], xerr=sm_sd[i], elinewidth=3, alpha=0.6, zorder=2)
        plt.scatter(
            sm, score, color=color, label=name,  s = 200, marker=marker, edgecolors=edgecolor, zorder = 3)

    plt.legend(loc='lower left',
               ncol=1,
               frameon=True,
               fontsize=18)


    plt.xlabel(r'Smoothness [rad $\cdot$ Hz]')
    plt.ylabel('Return')
    plt.ylim(min(scores_lst)-80, 1.1*max(scores_lst)+50)
    plt.tight_layout()

    # plt.savefig('/home/vlad/Pictures/' + figname+'_scatter')

def main():
    df_lst,names_lst, agent_types_lst = [],[],[]
    scores_lst, avg_scores_lst, sm_lst = [] ,[], []
    score_sd_lst, sm_sd_lst = [], []
    labels, comments = [], []

    '''               List with my names             '''
    run_names = ['zesty_nocaps_rl', "zesty_CAPS_rl"
                ,"glad_nocaps_erl",'glad_caps_erl', 'glad_smooth_erl', "glad_full_erl"
                ,"laced_long_erl", 'laced_full_erl']

    # Load extra infro from toml file
    cwd = Path(os.getcwd())
    path_to_toml = cwd / Path('plotters/runs_smoothness.toml')
    runs_from_toml= toml.load(path_to_toml)

    # Load summaries fron WandB
    api = wandb.Api()

    for i, run_name in enumerate(run_names):
        run_dict = runs_from_toml[run_name]
        _run = api.run(run_dict['path'])


        df = dict(_run.summary)
        name, config, agent_type = run_names[i].split('_')
        comments.append(config)

        # build lists
        df_lst.append(df)
        names_lst.append(_run.name.split('-')[0].split('_')[0].lower())
        agent_types_lst.append(agent_type)

        # sanity checks
        print(i, run_name)
        print("\n\n"+name+'_'+config)

        labels.append(run_dict['name'])
        pprint(run_dict)

        if _run.config['pop_size'] == 0:
            assert agent_types_lst[-1] == 'rl'
            scores_lst.append(df['rl_reward'])
            avg_scores_lst.append(df['rl_reward'])
            sm_lst.append(run_dict['sm'])
            sm_sd_lst.append(run_dict['sm_std'])
            score_sd_lst.append(df['rl_std'])
        else:
            assert agent_types_lst[-1] == 'erl'
            scores_lst.append(df['test_score'])
            avg_scores_lst.append(df['pop_avg'])
            sm_lst.append(run_dict['sm'])
            sm_sd_lst.append(run_dict['sm_std'])
            score_sd_lst.append(df['test_sd'])

        if name + '_'+ config == 'zesty_nocaps':  score_sd_lst[-1]*=2


    # Plotting
    plot_scattered(labels, avg_scores_lst, sm_lst, score_sd_lst, sm_sd_lst, figname='average')
    plot_scattered(labels, scores_lst, sm_lst, score_sd_lst,sm_sd_lst, figname='champion')
    plt.show()


if __name__ == '__main__':
    main()
