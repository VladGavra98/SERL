# import envs.citation as citation
import copy
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from signals.sequences import SmoothedStepSequence
import config
from  envs.models import Model
from envs.models import RLS


t_start = 0.01
t_play = 5  # play in the future every t_play secs
t_hor  = 5  # play in the future for t_hor secs


# plot style
style = 'seaborn-whitegrid'
mpl.style.use(style.lower())
mpl.rcParams.update({'font.size': 22})
mpl.rcParams['figure.figsize'] = (16, 9)
mpl.rcParams["axes.edgecolor"] = "0.15"
mpl.rcParams['lines.linewidth'] = 2.5
mpl.rcParams["axes.linewidth"] = 1.3
mpl.rcParams['axes.labelpad'] = 6
mpl.rcParams["axes.xmargin"] = 0
mpl.rcParams["axes.ymargin"] = 0.1
mpl.rcParams["axes.labelpad"] = 4.0

# mpl.rcParams["figure.autolayout"] = True
# grid
mpl.rcParams["axes.grid"] = True
mpl.rcParams["grid.color"] = "C1C1C1"
mpl.rcParams["grid.linestyle"] = ":"

# legend
mpl.rcParams['legend.frameon']  = True
mpl.rcParams["legend.framealpha"] = 1.0
mpl.rcParams["legend.edgecolor"] = "k"
mpl.rcParams["legend.fancybox"] = False
mpl.rcParams["xtick.minor.visible"] = True
mpl.rcParams["ytick.minor.visible"] = True

# saving
mpl.rcParams["savefig.pad_inches"] = 0.003
mpl.rcParams['savefig.format']= 'png'
mpl.rcParams['savefig.dpi']= 300
mpl.rcParams["savefig.bbox"] = "tight"

# colours
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
colors = sns.color_palette("Paired", 10)[::-1]
color_serl50 = colors[6]
color_serl10 = colors[2]
color_td3 = colors[0]
c_ref = 'black'
c_state = colors[4]
c_rate = colors[5]
c_command = colors[8]
c_alpha = colors[6]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

states = ['p','q','r','V', 'alpha', 'beta', 'theta', 'phi', 'psi', 'H']

class PID:
    def __init__(self, P_gain, I_gain, D_gain, dt : float = 0.01) -> None:
        self.p = P_gain
        self.d = D_gain
        self.i = I_gain

        self.der : np.float64 = 0.
        self.int : np.float64 = 0.

        self.dt : np.float64 = dt

    def select_action(self, state) -> np.ndarray:
        state = state[:3]
        action = -(self.p * state + self.i * self.int + self.d * self.der)
        action[-1]*=-1.5  # rudder needs some scaling

        return action

    def update(self, prev_state : np.ndarray, state : np.ndarray) -> None:
        self.int += state * self.dt
        self.der  = (state - prev_state) / self.dt

    def clone_from_dict(self, pid_dict : dict):
        for k in self.__dict__:
            self.__dict__[k] = copy.copy(pid_dict[k])



def evaluate (actor, env, model : Model = None, verbose : bool = False, **kwargs):
    """ Simulate one episode """
    # reset env
    done = False

    obs = env.reset(**kwargs)
    model.reset()

    last_obs = obs

    x_lst, rewards,u_lst, = [], [], []
    x_ctrl_lst = []
    error = []
    pred_lst = []

    times_lst = []
    pred_rewards_lst = []
    theta_lst = []
    ref_values_lst = []

    while not done:
        x_ctrl_lst.append(env.get_controlled_state())

        # select action:
        action = actor.select_action(obs[:env.n_actions])

        # Simulate one step in environment
        action = np.clip(action,-1,1)
        # action[0] = -5/57.3 if env.t > 5 else -np.deg2rad(8)

        if verbose:
            print(f'Action: {np.rad2deg(action)} -> deflection: {np.rad2deg(env.last_u)}')
            print(f't:{env.t:0.2f} theta:{env.theta:.03f} q:{env.q:.03f} alpha:{env.alpha:.03f}   V:{env.V:.03f} H:{env.H:.03f}')
            print(np.rad2deg(env.obs))

        x = env.x # current state.
        # Base env step:
        next_obs, reward, done, _ = env.step(action.flatten())
        next_x = env.x

        # Update model:
        model.update(x, action.flatten(), next_x)
        theta_lst.append(np.abs(model.epsilon))


        if model.t > t_start and round(env.t,2) % t_play == 0:
            playing_actor = PID(10,8,5)

            # playing_actor = actor
            playing_actor.clone_from_dict(actor.__dict__)

            # synchronise
            model.sync_env(env)

            pred_rewards, times = model.predictive_control(playing_actor, initial_obs = obs, t_horizon =t_hor, last_obs = obs)

            # print('Expected reward over horizon:', sum(pred_rewards))
            # print(len(pred_rewards))
            pred_rewards_lst.extend(pred_rewards)
            times_lst.extend(times)


        if verbose:
            print(f'Error: {obs[:env.n_actions]} Reward: {reward:0.03f} \n \n')

        assert next_obs[3] == env.p
        assert next_obs[4] == env.q
        assert next_obs[5] == env.r

        # Update
        actor.update(last_obs[:env.n_actions], obs[:env.n_actions] )

        # NOTE this can change places and it DOES have an effect
        x_lst.append(env.x)
        last_obs = obs
        obs = next_obs

        # save
        u_lst.append(env.last_u)
        error.append(env.error)
        rewards.append(reward)
        ref_values_lst.append(env.ref_values)


    # BCs:
    x_ctrl = np.asarray(x_ctrl_lst)
    x_lst = np.asarray(x_lst)
    pred_lst = np.asarray(pred_lst)
    theta_array = np.asarray(theta_lst).reshape(-1,10)
    ref_values_lst = np.asanyarray(ref_values_lst).reshape(-1,3)

    env.finish()
    mae = np.mean(np.absolute(error), axis = 0)

    _range = np.max(x_ctrl, axis = 0) - np.min(x_ctrl, axis = 0)
    nmae = np.mean(mae/_range) * 100

    return x_lst,rewards,np.asarray(u_lst),ref_values_lst, nmae, pred_rewards_lst, times_lst, theta_array


def main():
    env_name = 'phlab_attitude_nominal'
    # init env an actor
    base_env = config.select_env(env_name)
    env_id = config.select_env(env_name)

    # PID actor init
    p, i, d = 10, 8., 5.            # gains
    pid_actor = PID(p, i, d, dt = base_env.dt)

    # Inc RLS model
    model_config = {"seed": 7,
                    "gamma": 1.,
                    "cov0": 1000.0,
                    "eps_thresh": np.inf,
                    "state_size": 10}
    inc_model = RLS(model_config, env_id)


    """      Select Model     """
    _model = inc_model

    # evaluation mode settings
    t_eval = 80
    time_array = np.linspace(0., t_eval, 6)
    amplitudes = [0,15,3,-4,-8,2]
    step_ref = SmoothedStepSequence(time_array, amplitudes, smooth_width = 2.)
    user_eval_refs = {
        'theta_ref' : step_ref,
        'phi_ref' : step_ref,
    }

    base_env.set_eval_mode(t_eval)
    env_id.set_eval_mode(t_eval)

    trials = 1
    fitness_lst = []
    nmae_lst = []

    # Reconstruct data
    for _ in tqdm(range(trials)):
        x_lst, rewards, u_lst, ref_values, nmae, pred_rewards, times, theta_lst = \
                evaluate(pid_actor, base_env, model = _model, user_refs = user_eval_refs)

        fitness_lst.append(sum(rewards))
        nmae_lst.append(nmae)

    print('Fitness: ', np.mean(fitness_lst), 'SD:', np.std(fitness_lst))
    print(f'nMAE: {np.mean(nmae_lst)} %')

    avg_de =np.rad2deg(np.average(u_lst[:,0]))
    print(f'Average delta_e: {avg_de:0.01f} deg')


    do_plot = True
    if do_plot:
        plot(base_env, x_lst, u_lst, ref_values)


    base_env.finish()
    env_id.finish()


def plot_step(base_env, x_lst, u_lst, ref_values):

    time = np.linspace(0., base_env.t , x_lst.shape[0])

    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(18.5, 10.5)


    line_alpha, = axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$', linestyle = '-.', color = c_alpha)
    axs[0, 0].legend(loc='lower right', handles=[line_alpha],
                     fontsize=19, frameon=False, labels=[r'$\alpha$ [deg]'])
    l_s, = axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$', color = c_state)
    l_r, = axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), linestyle=':', label = r'$q$', color = c_rate)
    axs[0, 0].set_ylabel(r'$\theta,q$')

    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = r'$\phi$', color = c_state)
    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), linestyle=':', label = r'$p$', color = c_rate)
    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = r'$\beta$', color = c_state)
    axs[1,0].set_ylabel(r'$\phi,p$')
    axs[2,0].set_ylabel(r'$\beta$')


    axs[3,0].plot(time,x_lst[:,9]/1000, label = 'H [m]', color = c_alpha)
    axs[3,1].plot(time,x_lst[:,3], label = 'V [m/s]', color = c_alpha)
    axs[3,0].set_ylabel(r'$H$ [km]')
    axs[3,1].set_ylabel(r'$V_{tas}$ [m/s]')

        # plot actions
    l_c, = axs[0,1].plot(time,np.rad2deg(u_lst[:,0])+1.4, linestyle = '-',label = r'$\delta_e$ [deg]', color =c_command)
    axs[1,1].plot(time,np.rad2deg(u_lst[:,1]), linestyle = '-',label = r'$\delta_a$ [deg]', color =c_command)
    axs[2,1].plot(time,np.rad2deg(u_lst[:,2]), linestyle = '-',label = r'$\delta_r$ [deg]', color =c_command)
    axs[0, 1].set_ylabel(r'$\delta_e$')
    axs[1, 1].set_ylabel(r'$\delta_a$')
    axs[2, 1].set_ylabel(r'$\delta_r$')


        # Legend:
    labels = ['State [deg]', 'Attitude Rate [deg/s]', 'Actuator Command [deg]']
    fig.legend(handles = [ l_s, l_r, l_c],
                    ncol= len(labels),
                    labels=labels,
                    loc="lower center",
                    borderaxespad=0.1)

        # Label time axis
    axs[-1, 0].set_xlabel(r'Time $[s]$')
    axs[-1, 1].set_xlabel(r'Time $[s]$')

        # axis settings
    for ax in axs:
        ax[0].yaxis.set_major_formatter("{x:2.1f}")
        ax[0].locator_params(axis='y', nbins=5)
        ax[0].locator_params(axis='x', nbins=8)
        ax[1].yaxis.set_major_formatter("{x:2.1f}")
        ax[1].locator_params(axis='y', nbins=5)
        ax[0].set_xlim(-0.1, time[-1])
        ax[1].set_xlim(-0.1, time[-1])

    axs[-1,1].yaxis.set_major_formatter("{x}")

    axs[1,0].set_ylim(0, 0)
    axs[2,0].set_ylim(0, 0)
    plt.tight_layout()

    plt.subplots_adjust(top=0.96,
                            bottom=0.157,
                            left=0.079,
                            right=0.981,
                            hspace=0.26,
                            wspace=0.175)


    plt.show()

def plot(base_env, x_lst, u_lst, ref_values):
    time = np.linspace(0., base_env.t , x_lst.shape[0])

    fig, axs = plt.subplots(4,2)
    fig.set_size_inches(18.5, 10.5)
    l_ref, = axs[0,0].plot(time,np.rad2deg(ref_values[:,0]), linestyle = '--',label = r'$\theta_{ref}$', color= c_ref)
    axs[1,0].plot(time,np.rad2deg(ref_values[:,1]),linestyle = '--' ,label = r'$\phi_{ref}$', color= c_ref)
    axs[2,0].plot(time,np.rad2deg(ref_values[:,2]), linestyle = '--',label = r'$\beta_{ref}$', color= c_ref)

        # inc model
        # axs[0,0].plot(time, np.rad2deg(pred_lst[:,7]), linestyle = '-.',label = r'$\theta_{pred}$')

    line_alpha, = axs[0,0].plot(time,np.rad2deg(x_lst[:,4]), label = r'$\alpha$', linestyle = '-.', color = c_alpha)
    axs[0, 0].legend(loc='upper right', handles=[line_alpha],
                     fontsize=19, frameon=False, labels=[r'$\alpha$ [deg]'])
    l_s, = axs[0,0].plot(time,np.rad2deg(x_lst[:,7]), label = r'$\theta$', color = c_state)
    l_r, = axs[0,0].plot(time,np.rad2deg(x_lst[:,1]), linestyle=':', label = r'$q$', color = c_rate)
    axs[0, 0].set_ylabel(r'$\theta,q$')

    axs[1,0].plot(time,np.rad2deg(x_lst[:,6]), label = r'$\phi$', color = c_state)
    axs[1,0].plot(time,np.rad2deg(x_lst[:,0]), linestyle=':', label = r'$p$', color = c_rate)
    axs[2,0].plot(time,np.rad2deg(x_lst[:,5]), label = r'$\beta$', color = c_state)
    axs[1, 0].set_ylabel(r'$\phi,p$')
    axs[2, 0].set_ylabel(r'$\beta$')


    axs[3,0].plot(time,x_lst[:,9]/1000, label = 'H [m]',  color = c_alpha)
    axs[3,1].plot(time,x_lst[:,3], label = 'V [m/s]', color = c_alpha)
    axs[3,0].set_ylabel(r'$H$ [km]')
    axs[3,1].set_ylabel(r'$V_{tas}$ [m/s]')

        # plot actions
    l_c, = axs[0,1].plot(time,np.rad2deg(u_lst[:,0]), linestyle = '-',label = r'$\delta_e$ [deg]', color =c_command)
    axs[1,1].plot(time,np.rad2deg(u_lst[:,1]), linestyle = '-',label = r'$\delta_a$ [deg]', color =c_command)
    axs[2,1].plot(time,np.rad2deg(u_lst[:,2]), linestyle = '-',label = r'$\delta_r$ [deg]', color =c_command)
    axs[0, 1].set_ylabel(r'$\delta_e$')
    axs[1, 1].set_ylabel(r'$\delta_a$')
    axs[2, 1].set_ylabel(r'$\delta_r$')


        # Legend:
    labels = ['Referecne [deg]', 'State [deg]', 'Attitude Rate [deg/s]', 'Actuator Command [deg]']
    fig.legend(handles = [l_ref, l_s, l_r, l_c],
                    ncol= len(labels),
                    labels=labels,
                    loc="lower center",
                    borderaxespad=0.1)

        # Label time axis
    axs[-1, 0].set_xlabel(r'Time $[s]$')
    axs[-1, 1].set_xlabel(r'Time $[s]$')

        # axis settings
    for ax in axs:
        ax[0].yaxis.set_major_formatter("{x:2.1f}")
        ax[0].locator_params(axis='y', nbins=5)
        ax[0].locator_params(axis='x', nbins=8)
        ax[1].yaxis.set_major_formatter("{x:2.1f}")
        ax[1].locator_params(axis='y', nbins=5)
        ax[0].set_xlim(-0.1, time[-1])
        ax[1].set_xlim(-0.1, time[-1])

    axs[-1,1].yaxis.set_major_formatter("{x}")
    plt.tight_layout()

    plt.subplots_adjust(top=0.96,
                            bottom=0.157,
                            left=0.079,
                            right=0.981,
                            hspace=0.26,
                            wspace=0.175)


    plt.show()


if __name__=='__main__':
    main()
