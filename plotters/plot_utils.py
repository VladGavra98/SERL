import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import plotters.mystyle as mystyle

from plotters.mystyle import *


def plot(data, name='Actor', data_in_deg: bool = False, **kwargs):
    # hange these here
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    # smoothening options
    skip_index = 1
    kernel_size = 2
    kernel = np.ones(kernel_size) / kernel_size
    for idx in range(data.shape[1]):
        data[:, idx] = np.convolve(data[:, idx], kernel, mode='same')
    _start = kernel_size//2 + 1

    # demu data table
    ref_signals = data[:, :3]
    u_lst = data[:, 3:6]
    x_lst = data[:, 6:-1]
    rewards = data[:, -1]
    time = np.linspace(
        0., data.shape[0] * mystyle.dt, data.shape[0] - skip_index - 1)
    x_lst = x_lst[0:-skip_index-1, :]
    u_lst = u_lst[0:-skip_index-1, :]
    ref_signals = ref_signals[0:-skip_index-1, :]
    rewards = rewards[0:-skip_index-1]
    u_lst[0, :] = 0.0

    # define figure
    fig, axs = plt.subplots(3, 2)
    name = name + r' rad$\cdot$Hz'
    fig.suptitle(name)

    # covner rad/deg if needed
    _convertor = np.rad2deg(1.)
    if data_in_deg:
        _convertor = 1.

    # longitudinal
    line2, = axs[0, 0].plot(time, np.rad2deg(
        x_lst[:, 1]), label=r'$q$', linestyle=':', color=mystyle.c_rate)
    line1, = axs[0, 0].plot(time, np.rad2deg(
        x_lst[:, 7]), label=r'$\theta$', color=mystyle.c_state)
    line_alpha, = axs[0, 0].plot(time[_start:], np.rad2deg(
        x_lst[_start:, 4]), label=r'$\alpha$', color=mystyle.c_alpha, linestyle='dashdot')
    axs[0, 0].legend(loc='upper right', handles=[line_alpha],
                     fontsize=19, frameon=False, labels=[r'$\alpha$ [deg]'])
    line3, = axs[0, 0].plot(time, ref_signals[:, 0] * _convertor,
                            linestyle=mystyle.style_ref, label=r'$\theta_{ref}$', color=mystyle.c_ref)
    axs[0, 0].set_ylabel(r'$\theta,q$')

    # lateral
    axs[1, 0].plot(time, np.rad2deg(x_lst[:, 0]),
                   label=r'$p$', linestyle=':', color=mystyle.c_rate)
    axs[1, 0].plot(time, np.rad2deg(x_lst[:, 6]),
                   label=r'$\phi$', color=mystyle.c_state)
    axs[1, 0].plot(time, ref_signals[:, 1] * _convertor,
                   linestyle=mystyle.style_ref, label=r'$\phi_{ref}$', color=mystyle.c_ref)
    axs[1, 0].set_ylabel(r'$\phi,p$')

    axs[2, 0].plot(time, np.rad2deg(x_lst[:, 5]),
                   label=r'$\beta$', color=mystyle.c_state)
    axs[2, 0].plot(time, ref_signals[:, 2] * _convertor,
                   linestyle=mystyle.style_ref, label=r'$\beta_{ref}$', color=mystyle.c_ref)
    axs[2, 0].set_ylabel(r'$\beta$')

    # plot actions
    line_de, = axs[0, 1].plot(time, np.rad2deg(
        u_lst[:, 0]), linestyle='-', color=mystyle.c_command)
    axs[0, 1].set_ylabel(r'$\delta_e$')
    axs[1, 1].plot(time, np.rad2deg(u_lst[:, 1]),
                   linestyle='-', color=mystyle.c_command)
    axs[1, 1].set_ylabel(r'$\delta_a$')
    axs[2, 1].plot(time, np.rad2deg(u_lst[:, 2]),
                   linestyle='-', color=mystyle.c_command)
    axs[2, 1].set_ylabel(r'$\delta_r$')

    # Label time axis
    axs[-1, 0].set_xlabel(r'Time $[s]$')
    axs[-1, 1].set_xlabel(r'Time $[s]$')

    # Legend:
    labels = [r'Tracked State [deg]', r'Tracked State Rate [deg/s]',
              r'Reference [deg]', r'Actuator Deflection [deg]']

    legend = fig.legend(handles=[line1, line2, line3, line_de, line_alpha],
                        labels=labels,
                        ncol=len(labels),
                        loc="lower center",
                        borderaxespad=0.1,
                        fontsize=20)

    # align legend text
    for t in legend.get_texts():
        t.set_ha('center')

    # axis settings
    for ax in axs:
        ax[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
        ax[0].locator_params(axis='y', nbins=5)
        ax[0].locator_params(axis='x', nbins=8)
        ax[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
        ax[1].locator_params(axis='y', nbins=5)
        ax[0].set_xlim(-0.1, time[-1])
        ax[1].set_xlim(-0.1, time[-1])
        # ax[0].tick_params(axis='both',which='minor',length=2,width=2, direction = 'in')
        # ax[0].tick_params(axis='both',which='major',length=6,width=2, direction = 'in')

    plt.tight_layout()

    if 'borders' in kwargs:
        plt.subplots_adjust(**kwargs['borders'])
    else:
        plt.subplots_adjust(top=0.92,
                            bottom=0.157,
                            left=0.079,
                            right=0.981,
                            hspace=0.176,
                            wspace=0.175)

    # additions for faults:
    if 'fault' in kwargs:
        if kwargs['fault'] == 'sa':
            hline = axs[1, 1].hlines(
                y=[1, -1], color=colors[2], linestyle=constraint_style, xmin=-0.1, xmax=time[-1] + 0.1, alpha=0.8,)
            axs[1, 1].legend(loc='upper left', handles=[hline], fontsize=19, frameon=False,
                             labels=['Saturation Limit'])

        if kwargs['fault'] == 'jr':
            hline = axs[2, 1].hlines(
                y=[15], color=colors[2], linestyle=constraint_style, xmin=-0.1, xmax=time[-1] + 0.1, alpha=0.8,)
            axs[2, 1].legend(loc='upper left', handles=[hline], fontsize=19, frameon=False,
                             labels=['Jammed Rudder Deflection'])

        if kwargs['fault'] == 'se':
            hline = axs[0, 1].hlines(y=[2.5, -2.5], color=colors[2],
                                     linestyle=constraint_style, xmin=-0.1, xmax=time[-1] + 0.1, alpha=0.8)
            axs[0, 1].legend(loc='upper left', handles=[hline], fontsize=19, frameon=False,
                             edgecolor='black', labels=['Saturation Limit'])

    return fig, axs


def plot_long(data, name='Actor', data_in_deg: bool = False, **kwargs):
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18
    # smoothehing options
    skip_index = 1
    if kwargs.get('kernel_size'):
        kernel_size = kwargs['kernel_size']
    else:
        kernel_size = 2

    kernel = np.ones(kernel_size) / kernel_size
    for idx in range(data.shape[1]):
        data[:, idx] = np.convolve(data[:, idx], kernel, mode='same')
    _start = kernel_size//2 + 1

    # demu data table
    ref_signals = data[:, :3]
    u_lst = data[:, 3:6]
    x_lst = data[:, 6:-1]
    rewards = data[:, -1]
    time = np.linspace(0., data.shape[0] * dt, data.shape[0] - skip_index - 1)
    x_lst = x_lst[0:-skip_index-1, :]
    u_lst = u_lst[0:-skip_index-1, :]
    ref_signals = ref_signals[0:-skip_index-1, :]
    rewards = rewards[0:-skip_index-1]
    u_lst[0, :] = 0.0

    # define figure
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(name)
    fig.set_size_inches(16, 4.8)

    # covner rad/deg if needed
    _convertor = np.rad2deg(1.)
    if data_in_deg:
        _convertor = 1.

    # longitudinal
    line2, = axs[0].plot(time, np.rad2deg(x_lst[:, 1]),
                         label=r'$q$', linestyle=':', color=mystyle.c_rate)
    line1, = axs[0].plot(time, np.rad2deg(x_lst[:, 7]),
                         label=r'$\theta$', color=mystyle.c_state)
    line_alpha, = axs[0].plot(time[_start:], np.rad2deg(
        x_lst[_start:, 4]), label=r'$\alpha$', color=mystyle.c_alpha, linestyle='dashdot')
    line3, = axs[0].plot(time, ref_signals[:, 0] * _convertor,
                         linestyle=mystyle.style_ref, label=r'$\theta_{ref}$', color=mystyle.c_ref)
    axs[0].set_ylabel(r'$\theta,q, \alpha$')

    # plot actions
    line_de, = axs[1].plot(time, np.rad2deg(u_lst[:, 0]),
                           linestyle='-', color=mystyle.c_command)
    axs[1].set_ylabel(r'$\delta_e$')

    # Label time axis
    axs[0].set_xlabel(r'Time $[s]$')
    axs[1].set_xlabel(r'Time $[s]$')

    # Legend:
    labels = [r'Controlled State $[deg]$', r'Controlled State Rate $[deg/s]$',
              r'Reference $[deg]$', r'Actuator Command $[deg]$', r'Angle of Attack $[deg]$']

    fig.legend(handles=[line1, line2, line3, line_de, line_alpha],
               ncol=3,
               labels=labels,
               loc="lower center",
               borderaxespad=0.1,
               frameon=True)

    # axis settings
    axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
    axs[0].locator_params(axis='y', nbins=5)
    axs[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
    axs[1].locator_params(axis='y', nbins=5)
    axs[0].set_xlim(-0.1, time[-1])
    axs[1].set_xlim(-0.1, time[-1])

    plt.tight_layout()

    plt.subplots_adjust(top=0.89,
                        bottom=0.23,
                        left=0.06,
                        right=0.979,
                        hspace=0.175,
                        wspace=0.17)

    # additions for faults:
    if 'fault' in kwargs:
        if kwargs['fault'] == 'se':
            hline = axs[0, 1].hlines(y=[2.5, -2.5], color=mystyle.colors[2],
                                     linestyle=mystyle.constraint_style, xmin=-0.1, xmax=time[-1] + 0.1, alpha=0.8)
            axs[0, 1].legend(loc='upper left', handles=[hline], fontsize=19, frameon=False,
                             edgecolor='black', labels=['Saturation Limit'])

    return fig, axs


def plot_diff(data: np.ndarray, old_data: np.ndarray, name='Actor', data_in_deg: bool = False, **kwargs):
    # demu data table
    ref_signals = data[:, :3]
    u_lst = data[:, 3:6]
    x_lst = data[:, 6:-1]
    u_lst_old = old_data[:, 3:6]
    x_lst_old = old_data[:, 6:-1]
    time = np.linspace(0., data.shape[0] * mystyle.dt, data.shape[0])
    u_lst[0, :] = 0.0

    # define figure
    fig, axs = plt.subplots(1, 2)
    fig.suptitle(name)
    fig.set_size_inches(16, 5)

    # covner rad/deg if needed
    _convertor = np.rad2deg(1.)
    if data_in_deg:
        _convertor = 1.

    # gust area
    axs[0].axvspan(20, 23, alpha=0.1, color=mystyle.c_command)
    axs[1].axvspan(20, 23, alpha=0.1, color=mystyle.c_command)

    # longitudinal
    line_alpha, = axs[0].plot(time[:], np.rad2deg(x_lst[:, 4]),
                              color=mystyle.c_alpha, linestyle='dashdot')
    line2, = axs[0].plot(time, np.rad2deg(x_lst[:, 1]),
                         linestyle=':', color=mystyle.c_rate)
    theta_old, = axs[0].plot(time, np.rad2deg(
        x_lst_old[:, 7]), color='black', alpha=0.75)
    line1, = axs[0].plot(time, np.rad2deg(x_lst[:, 7]), color=mystyle.c_state)
    line3, = axs[0].plot(time, ref_signals[:, 0] *
                         _convertor, linestyle=mystyle.style_ref, color=mystyle.c_ref)
    axs[0].set_ylabel(r'$\theta,q, \alpha$')

    # plot actions
    line_de, = axs[1].plot(time, np.rad2deg(u_lst[:, 0]),
                           linestyle='-', color=mystyle.c_command, label='')
    line_de_old, = axs[1].plot(time, np.rad2deg(
        u_lst_old[:, 0]), linestyle='dashdot', color='black')
    axs[1].set_ylabel(r'$\delta_e$')

    # Label time axis
    axs[0].set_xlabel(r'Time $[s]$')
    axs[1].set_xlabel(r'Time $[s]$')

    # Legend:
    labels = [r'Pitch $[deg]$',
              r'Pitch - nominal case $[deg]$',
              r'Pitch Rate $[deg/s]$',
              r'Pitch Reference $[deg]$',
              r'Elevator Deflection $[deg]$',
              r'Elevator Deflection - nominal case $[deg]$',
              r'Angle of Attack $[deg]$',
              ]

    fig.legend(handles=[line1, theta_old, line2, line3, line_de, line_de_old, line_alpha, ],
               ncol=4,
               labels=labels,
               loc="lower center",
               borderaxespad=0.1,
               fontsize=18,
               frameon=True,
               fancybox=True,
               facecolor='white',
               edgecolor='black')

    # axis settings
    axs[0].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
    axs[0].locator_params(axis='y', nbins=5)
    axs[1].yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:2.1f}"))
    axs[1].locator_params(axis='y', nbins=5)
    # axs[0].set_xlim(18, 26)
    # axs[1].set_xlim(18, 26)
    # axs[0].set_ylim(-0.5, 15)
    # axs[1].set_ylim(-5.2, -1.2)

    plt.tight_layout()

    plt.subplots_adjust(top=0.895,
                        bottom=0.345,
                        left=0.08,
                        right=0.979,
                        hspace=0.16,
                        wspace=0.175)

    # additions for faults:
    if 'fault' in kwargs:
        if kwargs['fault'] == 'se':
            hline = axs[0, 1].hlines(y=[2.5, -2.5], color=mystyle.colors[2],
                                     linestyle=mystyle.constraint_style, xmin=-0.1, xmax=time[-1] + 0.1, alpha=0.8)
            axs[0, 1].legend(loc='upper left', handles=[hline],
                             edgecolor='black', labels=['Saturation Limit'])

    return fig, axs




def plot_ref(data, **kwargs):
    # hange these here
    plt.rcParams['xtick.labelsize'] = 18
    plt.rcParams['ytick.labelsize'] = 18

    # demu data table
    ref_signals = data[:, :3]
    time = np.linspace(0., data.shape[0] * mystyle.dt, data.shape[0])


    # define figure
    fig, axs = plt.subplots(3, 1)
    fig.suptitle('Reference Signals')

    # longitudinal
    axs[0].plot(time, ref_signals[:, 0] ,
                            linestyle=mystyle.style_ref, label=r'$\theta_{ref}$', color=mystyle.c_ref)
    axs[0].set_ylabel(r'$\theta,q$')

    # lateral
    axs[1].plot(time, ref_signals[:, 1],
                   linestyle=mystyle.style_ref, label=r'$\phi_{ref}$', color=mystyle.c_ref)
    axs[1].set_ylabel(r'$\phi,p$')
    axs[2].plot(time, ref_signals[:, 2],
                   linestyle=mystyle.style_ref, label=r'$\beta_{ref}$', color=mystyle.c_ref)
    axs[2].set_ylabel(r'$\beta$')

    # Label time axis
    axs[-1].set_xlabel(r'Time $[s]$')


    plt.tight_layout()

    if 'borders' in kwargs:
        plt.subplots_adjust(**kwargs['borders'])
    else:
        plt.subplots_adjust(top=0.92,
                            bottom=0.157,
                            left=0.079,
                            right=0.981,
                            hspace=0.176,
                            wspace=0.175)



    return fig, axs
