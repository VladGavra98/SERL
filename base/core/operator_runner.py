from dataclasses import dataclass
import numpy as np
from tqdm import tqdm

from core import genetic_agent
from core import mod_neuro_evo
from core.utils import calc_smoothness


@dataclass
class Stats:
    score: np.float16
    score_sd: np.float16
    sm: np.float16
    sm_sd: np.float16
    cost: np.float16 = 0
    cost_sd: np.float16 = 0


class OperatorRunner:
    def __init__(self, args, env, num_trails):
        self.env = env
        self.args = args
        self.num_trails = num_trails

    def evaluate(self, agent, **kwargs: dict):
        """ Simulate one episode in the global environment. """
        # reset env
        done = False
        obs = self.env.reset(**kwargs)

        x_lst, rewards, u_lst = [], [], []
        x_ctrl_lst = []
        errors = []
        ref_lst = []
        trial_cost = 0

        while not done:
            x_lst.append(self.env.x)
            u_lst.append(self.env.last_u)
            x_ctrl_lst.append(self.env.get_controlled_state())

            # select action:
            action = agent.actor.select_action(obs)

            # Simulate one step in environment
            action = np.clip(action, -1, 1)

            ref_value = np.deg2rad(
                np.array([ref(self.env.t) for ref in self.env.ref]).flatten())

            obs, reward, done, info = self.env.step(action.flatten())
            next_obs = obs

            if kwargs.get('save_transitions'):
                # save transition
                transition = (obs, action, next_obs, reward, float(done))
                agent.buffer.add(*transition)

                if info['cost']:
                    agent.critical_buffer.add(*transition)

            # check cost:
            trial_cost += info['cost']

            # Update
            obs = next_obs

            # save
            ref_lst.append(ref_value)
            errors.append(ref_value - x_ctrl_lst[-1])
            rewards.append(reward)

        self.env.finish()

        # Control inputs
        errors = np.asarray(errors)

        # Compute scaled smoothness fitness
        actions = np.asarray(u_lst)
        smoothness = calc_smoothness(actions, **kwargs)

        # Format data
        rewards = np.asarray(rewards).reshape((-1, 1))
        ref_values = np.array(ref_lst)
        data = np.concatenate((ref_values, actions, x_lst, rewards), axis=1)

        return data, np.sum(rewards), smoothness, trial_cost

    def validate_agent(self, gen_agent, user_refs_lst: list, num_trails: int = 1, **kwargs) -> tuple:
        """ Run evaluation over the user specified number of trails.
            The time traces from data come from he LAST episode played.
        """
        agent_score_lst, agent_sm_lst, cost_lst = [], [], []

        for i in tqdm(range(num_trails+1), total=num_trails):
            ref_tup = user_refs_lst[i]
            user_refs = {
                'theta_ref': ref_tup[0],
                'phi_ref': ref_tup[1],
            }
            data, _score, _smoothness, _cost = self.evaluate(
                gen_agent, user_refs=user_refs, **kwargs)
            agent_score_lst.append(_score)
            agent_sm_lst.append(_smoothness)
            cost_lst.append(_cost)

        score = np.average(agent_score_lst[:])
        score_sd = np.std(agent_score_lst)
        smoothness = np.average(agent_sm_lst[:])
        sm_sd = np.std(agent_sm_lst[:])
        cost = np.average(cost_lst)
        cost_sd = np.average(cost_lst)

        if kwargs.get('stdout'):
            print(f'Score: {score:0.1f}% with STD: {score_sd:0.1f}')
            print(f'Smoothness: {smoothness:0.0f} with STD: {sm_sd:0.1f}')

        stats = Stats(score, score_sd, smoothness, sm_sd, cost, cost_sd)
        return data, stats

    def test_mutation(self, pop: list, user_eval_refs: list):
        N_models = len(pop)
        models = list(range(N_models))
        num_all_runs = self.num_trails + 1

        pr, nmr, smr, pmr = np.zeros((N_models, num_all_runs)), np.zeros((N_models, num_all_runs)), np.zeros(
            (N_models, num_all_runs)), np.zeros((N_models, num_all_runs))  # reward arrays
        eff_sm, eff_nm, eff_pm = np.zeros((N_models, num_all_runs)), np.zeros(
            (N_models, num_all_runs)), np.zeros((N_models, num_all_runs))
        cost_sm, cost_nm, cost_pm = np.zeros((N_models, num_all_runs)), np.zeros(
            (N_models, num_all_runs)), np.zeros((N_models, num_all_runs))
        pc, nmc, smc, pmc = np.zeros((N_models, num_all_runs)), np.zeros(
            (N_models, num_all_runs)), np.zeros((N_models, num_all_runs)), np.zeros((N_models, num_all_runs))

        ssne = mod_neuro_evo.SSNE(self.args, None, None)
        for i, model in enumerate(models):
            print(f"========== Mutation for {model} ==============")
            agent = pop[model]
            _, stats = self.validate_agent(
                agent, user_eval_refs, self.num_trails, save_transitions=True)
            pr[i, :] = stats.score
            pc[i, :] = stats.cost

            # normal child
            normal_child = genetic_agent.GeneticAgent(self.args)
            ssne.clone(agent, normal_child)
            ssne.mutate_inplace(normal_child, self.args.mutation_mag)
            _, stats = self.validate_agent(
                normal_child, user_eval_refs, self.num_trails, )
            nmr[i, :] = stats.score
            nmc[i, :] = stats.cost

            # proximal child
            proximal_child = genetic_agent.GeneticAgent(self.args)
            ssne.clone(agent, proximal_child)
            ssne.proximal_mutate(proximal_child, self.args.mutation_mag)
            _, stats = self.validate_agent(
                proximal_child, user_eval_refs, self.num_trails)
            pmr[i, :] = stats.score
            pmc[i, :] = stats.cost

            # safe child
            safe_child = genetic_agent.GeneticAgent(self.args)
            ssne.clone(agent, safe_child)
            ssne.safe_mutate(safe_child, self.args.mutation_mag)
            _, stats = self.validate_agent(
                safe_child, user_eval_refs, self.num_trails)
            smr[i, :] = stats.score
            smc[i, :] = stats.cost

            # print averages over evaluatory trials
            print(
                f"Parent {np.mean(pr[i,:]):0.1f}, Cost: {np.mean(pc[i,:]):0.1f}")
            print(
                f"Normal {np.mean(nmr[i,:]):0.1f}  Cost: {np.mean(nmc[i,:]):0.1f} ")
            print(
                f"Safe {np.mean(smr[i,:]):0.1f}  Cost: {np.mean(smc[i,:]):0.1f}")
            print(
                f"Proximal {np.mean(pmr[i,:]):0.1f}  Cost: {np.mean(pmc[i,:]):0.1f}")

            # effects:
            eff_nm[i, :] = 1 - nmr[i, :] / pr[i, :]
            eff_sm[i, :] = 1 - smr[i, :] / pr[i, :]
            eff_pm[i, :] = 1 - pmr[i, :] / pr[i, :]

            cost_nm[i, :] = nmc[i, :] / pc[i, :] - 1
            cost_pm[i, :] = pmc[i, :] / pc[i, :] - 1
            cost_sm[i, :] = smc[i, :] / pc[i, :] - 1
            # print(np.average(eff_nm[i, :]), np.average(
            #     eff_sm[i, :]), np.average(eff_pm[i, :]))

        stats_reward = {
            'Normal': eff_nm.flatten(),
            'Proximal': eff_pm.flatten(),
            'Safe': eff_sm.flatten(),
        }

        stats_cost = {
            # 'Parent': pc.flatten(),
            'Normal': cost_nm.flatten(),
            'Proximal': cost_pm.flatten(),
            'Safe': cost_sm.flatten(),

        }
        return stats_reward, stats_cost

        # Ablation for safe mutation
        # ablation_mag = [0.0, 0.01, 0.05, 0.1, 0.2]

        # ablr = []
        # abls = []
        # for mag in ablation_mag:
        #     dchild = ddpg.GeneticAgent(self.self.args)
        #     ssne.clone(agent, dchild)
        #     ssne.proximal_mutate(dchild, mag)

        #     sm_reward, sm_states = self.self.evaluate(dchild)
        #     ablr.append(sm_reward)
        #     abls.append(sm_states)

        # save_file = 'visualise/mutation'
        # np.savez(save_file, pr=pr, nmr=nmr, smr=smr, ps=ps, nms=nms, sms=sms, ablr=ablr, abls=abls,
        #          abl_mag=ablation_mag)
