
import os
from typing import List, Dict, Tuple
from core import genetic_agent, mod_utils, replay_memory
from core import mod_neuro_evo as utils_ne
from core import ddpg
from core import td3
from core.utils import Episode, calc_smoothness
from parameters import Parameters
from tqdm import tqdm
import numpy as np
import torch

class Agent:
    """ Inteligent controller agent. It manages both the population and the RL parts. """

    def __init__(self, args: Parameters, environment):
        self.args = args
        self.env = environment

        # Init population
        self.pop: List = []
        self.pop = [genetic_agent.GeneticAgent(
            args) for _ in range(args.pop_size)]

        # Define RL Agent
        self.rl_agent = td3.TD3(args)

        # Define Memory Buffer:
        if args.per:
            self.replay_buffer = replay_memory.PrioritizedReplayMemory(args.buffer_size, args.device,
                                                                       beta_frames=self.args.num_frames)
        else:
            self.replay_buffer = replay_memory.ReplayMemory(
                args.buffer_size, args.device)

        # Define noise process:
        if args.use_ounoise:
            self.noise_process = mod_utils.OUNoise(args.action_dim)
        else:
            self.noise_process = mod_utils.GaussianNoise(
                args.action_dim, sd=args.noise_sd)

        # Initialise evolutionary loop
        if not self.pop:
            self.evolver = utils_ne.SSNE(
                self.args, self.rl_agent.critic, self.evaluate)

        # Testing
        self.validation_tests = 5

        # Trackers
        self.num_episodes = 0
        self.num_frames = 0
        self.iterations = 0
        self.gen_frames = None
        self.rl_history = None
        self.rl_iteration = 0            # for TD3 delyed policy updates
        self.champion: genetic_agent.GeneticAgent = None
        self.champion_actor: genetic_agent.Actor = None
        self.champion_history: np.ndarray = None

    def evaluate(self,
                 agent: genetic_agent.GeneticAgent or ddpg.DDPG or td3.TD3,
                 is_action_noise: bool,
                 store_transition: bool) -> Episode:
        """ Play one game to evaluate the agent.

        Args:
            agent (GeneticAgentor): Agent class.
            is_action_noise (bool): Add Gaussian/OU noise to action.
            store_transition (bool, optional): Add frames to memory buffer for training. Defaults to True.

        Returns:
            Episode: data class with the episode stats
        """
        # init states, env and
        rewards, state_lst, action_lst = [], [], []
        obs = self.env.reset()
        done = False

        # set actor to evaluation mode
        agent.actor.eval()

        while not done:
            # select  actor ation
            action = agent.actor.select_action(obs)

            # add exploratory noise
            if is_action_noise:
                clipped_noise = np.clip(self.args.noise_sd * np.random.randn(action.shape[0]),
                                        - self.args.noise_clip, self.args.noise_clip)
                action = np.clip(action + clipped_noise, -1.0, 1.0)

            # Simulate one step in environment
            next_obs, reward, done, info = self.env.step(action.flatten())
            rewards.append(reward)
            action_lst.append(self.env.last_u)  # actuator deflection

            # Add experiences to buffer:
            if store_transition:
                # store for training
                transition = (obs, action, next_obs, reward, float(done))
                self.replay_buffer.add(*transition)
                agent.buffer.add(*transition)

                # check cost -> enter (pre)stall/high-bank dynamics
                if info['cost']:
                    agent.critical_buffer.add(*transition)

                self.num_frames += 1
                self.gen_frames += 1
            else:
                # save for future validation
                state_lst.append(self.env.x)

            # update agent obs
            obs = next_obs

        # End env
        self.env.finish()

        # updated episodes if is done
        if store_transition:
            self.num_episodes += 1

        # Compute smoothness and fitness:
        actions = np.asarray(action_lst)
        smoothness = calc_smoothness(actions, plot_spectra=False)
        fitness = np.sum(rewards)

        # Use smoothness-based fitness
        if self.args.smooth_fitness:
            fitness += smoothness

        return Episode(fitness=fitness, smoothness=smoothness, length=info['t'],
                       state_history=state_lst, ref_signals=info['ref'],
                       actions=actions, reward_lst=rewards)

    def rl_to_evo(self, rl_agent: ddpg.DDPG or td3.TD3, evo_net: genetic_agent.GeneticAgent):
        for target_param, param in zip(evo_net.actor.parameters(), rl_agent.actor.parameters()):
            target_param.data.copy_(param.data)
        evo_net.buffer.reset()
        evo_net.buffer.add_content_of(rl_agent.buffer)
        evo_net.critical_buffer.reset()
        evo_net.critical_buffer.add_content_of(rl_agent.critical_buffer)

    def evo_to_rl(self, rl_net, evo_net):
        for target_param, param in zip(rl_net.parameters(), evo_net.parameters()):
            target_param.data.copy_(param.data)

    def get_pop_novelty(self, bcs: np.array):
        return np.sum(np.std(bcs, axis=0))/bcs.shape[1]

    def train_rl(self, rl_transitions: int) -> Dict[float, float]:
        """ Train the RL agent on the same number of frames seens by the entire actor populaiton during the last generation.
            The frames are sampled from the common buffer.
        """

        pgs_obj, TD_loss = [], []

        if len(self.replay_buffer) > self.args.learn_start:
            # prepare for training
            self.rl_agent.actor.train()

            # select target policy
            if self.args.use_champion_target:
                if self.champion_actor is not None:  # safeguard for RL-only runs
                    self.evo_to_rl(self.rl_agent.actor_target,
                                   self.champion_actor)

            # train over generation experiences
            for _ in tqdm(range(int(rl_transitions * self.args.frac_frames_train)),
                         desc='Train RL', colour = 'blue'):
                self.rl_iteration += 1

                batch = self.replay_buffer.sample(self.args.batch_size)
                pgl, TD = self.rl_agent.update_parameters(
                    batch, self.rl_iteration, self.args.use_champion_target)

                if pgl is not None:
                    pgs_obj.append(-pgl)
                if TD is not None:
                    TD_loss.append(TD)

        return {'PG_obj': np.mean(pgs_obj), 'TD_loss': np.median(TD_loss)}

    def validate_agent(self, agent: genetic_agent.Actor) -> Tuple[float, float, float, float, Episode]:
        """ Evaluate the  given actor and do NOT store these trials.
        """
        test_scores, episode_lengths, smoothness_lst = [], [], []

        for _ in range(self.validation_tests):
            last_episode = self.evaluate(agent,
                                         is_action_noise=False,
                                         store_transition=False)
            test_scores.append(np.sum(last_episode.reward_lst))
            episode_lengths.append(last_episode.length)
            smoothness_lst.append(last_episode.smoothness)

        test_score = np.mean(test_scores)
        test_sd = np.std(test_scores)
        ep_len = np.mean(episode_lengths)
        ep_len_sd = np.std(episode_lengths)
        sm = np.median(smoothness_lst)
        sm_sd = np.std(smoothness_lst)


        return test_score, test_sd, ep_len, ep_len_sd, last_episode, sm, sm_sd

    def train(self):
        self.iterations += 1

        # Initialise local trackers:
        self.gen_frames = 0
        best_train_fitness = 1.
        worst_train_fitness = 1.
        population_avg = 1.
        test_score = 1.
        test_sd = -1.
        sm = 1.
        sm_sd = -1
        elite_index = -1.
        pop_novelty = -1.
        lengths = []

        '''++++++++++++++++++++++++++++++   Evolution   ++++++++++++++++++++++++++++++++++'''
        if not self.pop:
            fitness_lst = np.zeros((self.args.num_evals, self.args.pop_size))
            smoothness_lst = []

            # Evaluate genomes/individuals
            # >>> loop over population AND store experiences
            for j, net in tqdm(enumerate(self.pop), total = len(self.pop), desc = 'Population evaluation', colour="green"):
                for i in range(self.args.num_evals):
                    episode = self.evaluate(net,
                                            is_action_noise=False,
                                            store_transition=(i == self.args.num_evals-1))
                    smoothness_lst.append(episode.smoothness)
                    lengths.append(episode.length)
                    fitness_lst[i, j] = episode.fitness


            # take average stats
            pop_fitness = np.mean(fitness_lst, axis=0)
            sm = np.mean(smoothness_lst)
            sm_sd = np.std(smoothness_lst)
            ep_len_avg = np.mean(lengths)
            ep_len_sd = np.std(lengths)

            # get popualtion stas
            best_train_fitness = np.max(pop_fitness)      # champion - highest fitness
            worst_train_fitness = np.min(pop_fitness)
            population_avg = np.average(pop_fitness)       # population_avg fitness
            self.champion = self.pop[np.argmax(pop_fitness)]
            self.champion_actor = self.champion.actor         # unpack for RL critic updates

            # Validation test for NeuroEvolution
            test_score, test_sd, _, _, last_episode, _, _ = self.validate_agent(
                self.champion)
            if self.args.should_log:
                self.champion_history = last_episode.get_history()

            # NeuroEvolution's probabilistic selection and recombination step
            elite_index = self.evolver.epoch(self.pop, pop_fitness)

        ''' +++++++++++++++++++++++++++++++   RL  ++++++++++++++++++++++++++++++++++++++'''
        # Collect extra experience for RL training
        self.evaluate(self.rl_agent, is_action_noise=True, store_transition=True)
                # Gradient updates of RL actor and critic:
        rl_train_scores = self.train_rl(self.gen_frames)

        # Validate RL actor separately:
        rl_reward, rl_std, rl_ep_len, rl_ep_std, rl_episode, rl_sm, rl_sm_sd = self.validate_agent(
            self.rl_agent)

        if self.args.pop_size == 0:
            ep_len_avg = rl_ep_len
            ep_len_sd = rl_ep_std

        if self.args.should_log:
            self.rl_history = rl_episode.get_history()

        ''' ++++++++++++++++++++++++++++ Actor Injection ++++++++++++++++++++++++++++++'''
        if self.args.pop_size and self.iterations % self.args.rl_to_ea_synch_period == 0:
            # Replace any index different from the new elite
            replace_index = np.argmin(pop_fitness)

            if replace_index == elite_index:
                replace_index = (replace_index + 1) % len(self.pop)

            self.rl_to_evo(self.rl_agent, self.pop[replace_index])
            self.evolver.rl_policy = replace_index
            print('Sync from RL --> Evolution')

        ''' ++++++++++++++++++++++++++ Collect Statistics +++++++++++++++++++++++++++++'''
        return {
            'best_train_fitness': best_train_fitness,
            'test_score':         test_score,
            'test_sd':            test_sd,
            'pop_avg':            population_avg,
            'pop_min':            worst_train_fitness,
            'elite_index':        elite_index,
            'avg_smoothness':     sm,
            'smoothness_sd':      sm_sd,
            'rl_reward':          rl_reward,
            'rl_smoothness':      rl_sm,
            'rl_smoothness_std':  rl_sm_sd,
            'rl_std':             rl_std,
            'avg_ep_len':         ep_len_avg,
            'ep_len_sd':          ep_len_sd,
            'PG_obj':             rl_train_scores['PG_obj'],
            'TD_loss':            rl_train_scores['TD_loss'],
            'pop_novelty':        pop_novelty,
        }

    def save_agent(self, parameters: object, elite_index: int = None) -> None:
        """ Save the trained agent(s).

        Args:
            parameters (object): Container class of the trainign hyperparameters.
            elite_index (int): Index of the best performing agent i.e. the champion.
                        Defaults to None.
        """
        # Save gentic popualtion
        if not self.pop.isEmpty():
            pop_dict = {}
            for i, ind in enumerate(self.pop):
                pop_dict[f'actor_{i}'] = ind.actor.state_dict()

            torch.save(pop_dict, os.path.join(
                parameters.save_foldername, 'evo_nets.pkl'))

            # Save best performing agent separately:
            torch.save(self.pop[elite_index].actor.state_dict(),
                       os.path.join(parameters.save_foldername, 'elite_net.pkl'))

            # Save state history of the champion
            filename = 'statehistory_episode' + str(self.num_episodes) + '.txt'
            np.savetxt(os.path.join(parameters.save_foldername, filename),
                       self.champion_history, header=str(self.num_episodes))

        # Save RL actor separately:
        torch.save(self.rl_agent.actor.state_dict(),
                   os.path.join(parameters.save_foldername, 'rl_net.pkl'))

        filename = 'rl_statehistory_episode' + str(self.num_episodes) + '.txt'
        np.savetxt(os.path.join(parameters.save_foldername, filename),
                   self.rl_history, header=str(self.num_episodes))

        # NOTE might want to save RL state-history for future cheks
        print('> Saved state history in ' + str(filename) + '\n')
