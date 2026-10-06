import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sns

import mystyle
from mystyle import *


def plot_density(x: list, xlabel: str, **kwargs):
    plt.rcParams.update({'font.size': 30})
    fig, ax = plt.subplots()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    sns.histplot(data=x,
                color = '#00B8C8',
                 alpha=.5,
                 edgecolor=(1, 1, 1, .4),
                 **kwargs
                 )
    plt.xlabel(str(xlabel))
    plt.ylabel('Actor ' + kwargs['stat'].capitalize())
    plt.tight_layout()

    return fig, ax


'''    Change path to logs folder from wand    '''
path_logs = 'logs/wandb/run-20220924_144643-1xzaqiba_LONG'
fault_name = 'nominal'
statistics_type = 'count'
save_plots = False

cwd = os.getcwd()
pwd = Path(os.path.abspath(os.path.join(cwd, os.pardir)))
logs_dir = cwd / Path(path_logs)

figpath = logs_dir / Path('figures')
faultpath = figpath / Path(fault_name)


champ_nmae =  4.581009081951172
champ_sm = -2.810896612027241
rl_nmae = 9.798967831180565
rl_sm = -11.498465539858125


pop_stats_file  = faultpath / Path('final_performance.csv')
tab = np.genfromtxt(pop_stats_file, dtype = float, delimiter =',')
sm_lst = tab[:,0];nmae_lst = tab[:,1]

fig_sm, ax_sm = plot_density(sm_lst, xlabel=r'Smoothness [rad $\cdot$ Hz]', stat = statistics_type, binwidth = 1)
fig_nmae, ax_nmae = plot_density(nmae_lst, xlabel='nMAE [%]', stat = statistics_type, binwidth = 1)
ax_sm.axvline([champ_sm], linestyle= '--', color = c_state, label = 'Champion', linewidth = 3)
ax_sm.axvline([rl_sm], linestyle= '--', color = '#FFB81C', label = 'TD3 Policy', linewidth = 3)
ax_sm.legend(loc= 'upper left')

ax_nmae.axvline([champ_nmae], linestyle= '--', color = c_state, label = 'Champion', linewidth = 3)
ax_nmae.axvline([rl_nmae], linestyle= '--', color = '#FFB81C', label = 'TD3 Policy', linewidth = 3)
ax_nmae.legend(loc= 'upper right')


if save_plots:
    fig_sm.savefig(fname = faultpath / f'smoothness_hist_{fault_name}.png' )
    fig_nmae.savefig(fname = faultpath / f'nmae_hist_{fault_name}.png')
else:
    plt.show()