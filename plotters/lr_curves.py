
from pprint import pprint
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import wandb
import mystyle
from mystyle import *


stats_type = 'champion'
plot_long = False
types_dict = {
    'champion' : 'test_score',
    'average': 'pop_avg',
    'minimum': 'pop_min'
}


def plot_rewards(names_lst, comments, df_filtered, given_name_lst : list = None):
    use_names = False
    start_idx = 1
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams.update({'font.size': 30})

    fig,ax = plt.subplots()
    fig.canvas.manager.set_window_title("Reward versus frames")
    for i,df in enumerate(df_filtered):

        if use_names: name = names_lst[i]
        else: name = given_name_lst[i]
        comment = comments[i]
        frames = (df['frames'][start_idx:])
        end_idx = len(frames)
        if '10' in name:
            color = mystyle.color_serl10
        elif '50' in name:
            color = mystyle.color_serl50
        else:
            color = mystyle.color_td3
            end_idx = -1

        _frames = frames[:end_idx]
        score = df['score'][start_idx:]; score = score[:end_idx]
        sd = df['sd'][start_idx:];sd = sd[:end_idx]

        print(name+comment, score[-1], sd[-1])

        ax.plot(_frames, score, linestyle = '-',\
                    label =  name + comment, color = color)
        ax.fill_between(_frames, score - sd,\
                        score + sd, color=color, alpha=0.4)

    ax.set_title('Population '+ stats_type.capitalize())
    ax.set_ylabel("Return")
    ax.set_xlabel(r"Training Frames ")
    x = frames[-2]
    x -= x % -1_000_000
    ax.set_xlim(max(frames[0],20_000), x)
    ax.set_ylim(-1_000,-1)

    ax.legend(loc = 'lower right')
    fig.tight_layout()


def load_runs(run_paths, run_names, comments =  None):
    # Load runs hsitories:
    api = wandb.Api()

    df_lst, names_lst, types_lst = [],[],[]

    for idx, run_name in enumerate(run_paths):
        _run = api.run(run_name)
        df = pd.DataFrame(_run.history())
        _, run_type = run_names[idx].split('_')

        # build lists
        df_lst.append(df)
        names_lst.append(_run.name.split('-')[0].split('_')[0].lower())
        types_lst.append(run_type)

        # sanity checks
        print(idx, run_name)

        print(f"\n\n============= {names_lst[-1]} ===============")
        pprint(_run.config)

    return df_lst,names_lst,types_lst,comments


def main():
    given_name_lst, run_paths, run_names = [], [], []


    if not plot_long:
        given_name_lst.extend(['TD3','SERL(10)'])
        run_paths.extend(["vgavra/CAPS/3enkuysa", "vgavra/CAPS/20wft7fz", ])
        run_names.extend(["zesty_rl",  "safer_erl"])
    else:
        given_name_lst.extend(['SERL(50)'])
        run_paths.extend(["vgavra/CAPS/5jsfh71q"])


    df_lst, names_lst, types_lst, comments = load_runs(run_paths, run_names, comments = ['','','', ''])

    ref_run = 0
    df_filtered = []

    for i,df in enumerate(df_lst):
        if 'rl' == types_lst[i]:
            temp_dict = df.filter(['frames', 'rl_reward', 'rl_std'], axis= 1).to_dict('list')
            temp_dict['score'] = temp_dict.pop('rl_reward')
            temp_dict['sd'] = temp_dict.pop('rl_std')
        elif 'erl' == types_lst[i]:
            temp_dict = df.filter(['frames', 'test_score', 'test_sd', 'pop_avg', 'pop_min'], axis= 1).to_dict('list')
            temp_dict['score'] = temp_dict.pop(types_dict[stats_type])
            temp_dict['sd'] = temp_dict.pop('test_sd')
        else:
            Warning('Unknown run_type of run.')

        # replace data frame
        _dict = {k:np.asarray(v)[:] for k,v in temp_dict.items()}
        df_filtered.append(_dict)


    # smoothehing options
    kernel_size = 2
    kernel = np.ones(kernel_size) / kernel_size

    for i,df in enumerate(df_filtered):

        if i != ref_run:
            for col,v in df.items():
                if col != 'frames':
                    # f = interp1d(cur_frames, v, kind, fill_value = 'extrapolate')  # type: ignore
                    y = df[col]
                    df[col] = np.convolve(y, kernel, mode = 'same')
        else:
            for col,v in df.items():
                if col != 'frames':
                    df[col] = np.convolve(v, kernel, mode = 'same')

    # Plotting:
    do_plot = True

    if do_plot:
        plot_rewards(names_lst, comments, df_filtered, given_name_lst)
        plt.show()

    # # Statistical tests:
    # for comb in combinations(range(len(df_filtered)), 2):
    #     i,j = comb
    #     print(f'Test: mean of {given_name_lst[j]} > mean of {given_name_lst[i]} :')
    #     sample1 = df_filtered[j]['score']
    #     sample2 = df_filtered[i]['score']
    #     std1 = df_filtered[j]['sd']
    #     std2 = df_filtered[i]['sd']
    #     _, p = ttest_ind_from_stats(sample1, std1, 5, sample2, std2, 5, alternative='greater')

    #     print(f'{np.mean(p[-10:]):0.2%}')


if __name__ == '__main__':
    main()