import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import re

from plotters.plot_utils import plot



def plot_epiosde_full (flst, ep_num_lst, idx, name : str = None, **kwargs):
    flst = [flst[i] for i in np.argsort(ep_num_lst)]
    ep_num_lst = np.sort(ep_num_lst)

    episode_file = open(logfolder / Path(flst[idx]),encoding = 'utf-8')

    # episode_num = episode_file.readline().strip('# ')
    data = np.genfromtxt(episode_file, skip_header=1)
    name = name + f' actor: episode {ep_num_lst[idx]} with R={np.sum(data[:,-1]):0.0f}'
    plot(data, name=name, **kwargs)

if __name__ == '__main__':
    savefig = True

    # Load state history data:
    logfolder = Path('../logs/wandb/run-20220924_144643-1xzaqiba_LONG_RL')
    logfolder = logfolder / Path('files/')
    
    flst,rl_flst ,ep_num_lst, rl_ep_num_lst = [], [], [], []

    for file in os.listdir(logfolder):
        if file.endswith(".txt") and 'requirements' not in file:
            ep_num = int(re.search(r'\d+', file).group())
            if 'rl' in file:
                rl_flst.append(file)
                rl_ep_num_lst.append(ep_num)
            else:
                flst.append(file)
                ep_num_lst.append(ep_num)


    idx = -1
    borders = {'top':0.935,
                'bottom':0.145,
                'left':0.07,
                'right':0.984,
                'hspace':0.19,
                'wspace':0.165}

    if len(flst):
        plot_epiosde_full(flst, ep_num_lst, idx, name = 'Champion', data_in_deg = True, borders = borders)

    plot_epiosde_full(rl_flst, rl_ep_num_lst, idx, name = 'RL', data_in_deg = True,borders = borders)


    plt.show()
