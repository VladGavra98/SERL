from pprint import pprint
import os
import torch


class Parameters:
    def __init__(self, cla, init = True):
        if not init:
            return

        # Set the device to run on CUDA or CPU
        if not cla.disable_cuda and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        print('Current device:', self.device)

        # Render episodes
        self.env_name = cla.env
        self.save_periodic = True if hasattr(cla, 'save_periodic') else False

        # Number of Frames to Run
        if hasattr(cla, 'frames'):
            self.num_frames = cla.frames
        else:
            self.num_frames = 800_000

        # Synchronization
        if hasattr(cla, 'sync_period'):
            self.rl_to_ea_synch_period = cla.sync_period
        else:
            self.rl_to_ea_synch_period = 1

        # Model save frequency if save is active
        self.next_save = cla.next_save if hasattr(cla, 'next_save') else 1000

        # ==================================  RL (TD3) Params =============================================
        self.test_ea = cla.test_ea if hasattr(cla, 'test_ea') else False
        if self.test_ea:
            self.frac_frames_train = 0.
        else:
            self.frac_frames_train = 1.  # default training

        self.batch_size = 86
        self.buffer_size = 100_000

        self.lr = 0.0004335
        self.gamma = 0.98
        self.noise_sd = 0.2962183114680794

        self.use_done_mask = True
        self.use_ounoise = cla.use_ounoise if hasattr(
            cla, 'use_ounoise') else False
        self.tau = 0.005
        self.seed = cla.seed

        # hidden layer
        self.num_layers = 3
        self.hidden_size = 72
        self.activation_actor = 'tanh'
        self.activation_critic = 'elu'

        self.learn_start = 10_000       # frames accumulated before grad updates
        # Prioritised Experience Replay
        self.per = cla.per if hasattr(cla, 'per') else False
        if self.per:
            self.replace_old = True
            self.alpha = 0.7
            self.beta_zero = 0.5

        # CAPS
        self.use_caps = cla.use_caps if hasattr(cla, 'use_caps') else True

        # ==================================    TD3 Params  =============================================
        self.policy_update_freq = 3      # minimum for TD3 is 2
        self.noise_clip = 0.5                # default for TD3

        # =================================   NeuroEvolution Params =====================================
        # Number of actors in the population
        self.pop_size = cla.pop_size if hasattr(cla, 'pop_size') else 10

        # champion is target actor
        self.use_champion_target = cla.champion_target if hasattr(
            cla, 'champion_target') else False

        # Genetic memory size
        self.individual_bs = 10_000

        if self.pop_size:
            self.smooth_fitness = cla.smooth_fitness if hasattr(
                cla, 'smooth_fitness') else False

            # # increase buffer size for more experiences
            self.buffer_size = 800_000

            # # decrease lr
            self.lr = 0.00018643512599969097

            # Num. of trials during evaluation step
            self.num_evals = 3

            # Elitism Rate - % of elites
            self.elite_fraction = 0.2

            # Mutation and crossover
            self.mutation_prob = 0.9
            self.mutation_mag = 0.0247682869654

            self.mutation_batch_size = self.batch_size
            self.mut_type = cla.mut_type if hasattr(
                cla, 'mut_type') else 'proximal'
            self.distil_crossover = True
            self.distil_type = cla.distil_type if hasattr(
                cla, 'distil_type') else 'distance'
            self.crossover_prob = 0.0
            self._verbose_mut = cla.verbose_mut if hasattr(
                cla, 'verbose_mut') else False
            self._verbose_crossover = cla.verbose_crossover if hasattr(
                cla, 'verbose_crossover') else False

        # Save Results
        self.state_dim = None   # To be initialised externally
        self.action_dim = None  # To be initialised externally
        self.save_foldername = './tmp/'
        self.should_log = cla.should_log if hasattr(
            cla, 'should_log') else False

        if not os.path.exists(self.save_foldername):
            os.makedirs(self.save_foldername)


    def write_params(self, stdout=False) -> dict:
        """ Transfer parmaters obejct to a state dictionary.
            Args:
                stdout (bool, optional): Print. Defaults to True.

        Returns:
            dict: Parameters dict
        """
        if stdout:
            params = pprint.pformat(vars(self), indent=4)
            print(params)

        return self.__dict__

    def update_from_dict(self, new_config_dict: dict):
        self.__dict__.update(new_config_dict)

    def stdout(self) -> None:
        keys = ['save_foldername', 'seed', 'batch_size',
                'buffer_size', 'lr', 'gamma', 'noise_sd',
                'num_layers', 'hidden_size', 'activation_actor',
                'activation_critic', 'use_caps', 'pop_size',
                'use_champion_target', 'smooth_fitness', 'num_evals',
                'elite_fraction']
        _dict = {}
        for k in keys:
            _dict[k] = self.__dict__[k]

        pprint(_dict)
