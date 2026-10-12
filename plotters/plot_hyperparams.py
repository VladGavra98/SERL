import pprint
import wandb
import pandas as pd

import plotly.express as px


def plot_parallel(all_df, labels, colormap = 'test_score'):
    fig = px.parallel_coordinates(all_df, color=colormap,
                                labels = labels,
                                color_continuous_scale=px.colors.sequential.Turbo_r,
                             range_color = [-250,-70],
                             dimensions = all_df,
                             color_continuous_midpoint=100)
    fig.update_layout(
    font=dict(
        family="sans-serif",
        size=22,  # Set the font size here
        color="black"
        ) )

    return fig



def get_run_data(runs, pop_size : int = 10):
    summary_list, config_list, name_list = [], [], []
    print(len(runs))
    for run in runs:
        if 'bad' not in run.name.lower() and run.config['pop_size'] == pop_size:
            # run.summary are the output key/values like accuracy.
            # We call ._json_dict to omit large files
            summary_list.append(run.summary._json_dict)

            # run.config is the input metrics.
            # We remove special values that start with _.
            config = {k: v for k, v in run.config.items()
                    if not k.startswith('_')}
            config_list.append(config)

            # run.name is the name of the run.
            name_list.append(run.name)


    return summary_list, config_list, name_list


def lst2df(summary_list, config_list, name_list):
    name_df = pd.DataFrame({'name': name_list})
    summary_df = pd.DataFrame.from_records(summary_list)
    config_df = pd.DataFrame.from_records(config_list)
    return name_df, summary_df, config_df


def filter_data_erl(summary_list, config_list, name_list):
    name_df, summary_df, config_df = lst2df(
        summary_list, config_list, name_list)

    labels = {'gamma': 'Discount Factor',
              'buffer_size': 'Buffer Size',
              'batch_size': 'Batch Size',
              'hidden_size': 'Hidden Size',
              'lr': 'Learning Rate',
              'mutation_mag': 'Mutation Mag.',
              'noise_sd': 'Exploratory Noise Mag.'}

    params = labels.keys()

    config_df = config_df[params]
    all_df = pd.concat([name_df, config_df, summary_df['test_score']], axis=1)
    all_df = all_df.loc[2:, :]

    pprint.pprint(all_df)
    labels['test_score'] = r'Return'
    return labels, all_df


def filter_data_td3(summary_list, config_list, name_list):
    name_df, summary_df, config_df = lst2df(
        summary_list, config_list, name_list)

    labels = {'gamma': 'Discount Factor',
              'buffer_size': 'Buffer Size',
              'batch_size': 'Batch Size',
              'hidden_size': 'Hidden Size',
              'lr': 'Learning Rate',
              'noise_sd': 'Exploratory Noise Mag.'}

    params = labels.keys()

    config_df = config_df[params]
    all_df = pd.concat([name_df, config_df, summary_df['rl_reward']], axis=1)
    all_df = all_df.loc[:, :]

    pprint.pprint(all_df)
    labels['rl_reward'] = 'Return'
    return labels, all_df


def main():
    api = wandb.Api()
    # Project is specified by <entity/project-name>

    """               Change THESE               """
    project = "vgavra/sweeps-erl"  # "vgavra/sweeps-erl"
    pop_size = 50

    runs = api.runs(project)
    summary_list, config_list, name_list = get_run_data(runs, pop_size=pop_size)

    filter_data = filter_data_erl
    colormap = 'test_score'

    if 'td3' in project.lower():
        filter_data = filter_data_td3
        colormap = 'rl_reward'


    labels, all_df = filter_data(summary_list, config_list, name_list)

    fig = plot_parallel(all_df, labels, colormap)
    fig.show()


if __name__ == '__main__':
    main()
