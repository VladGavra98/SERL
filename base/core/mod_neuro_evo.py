import random
import numpy as np
from typing import List, Tuple, Dict
import random
import torch
import torch.distributions as dist
from core.mod_utils import hard_update, soft_update
from parameters import Parameters
import os
import math
from core.genetic_agent import GeneticAgent
from core.mod_utils import is_lnorm_key

class SSNE:
    def __init__(self, args: Parameters, critic : torch.nn, evaluate : callable):
        self.current_gen = 0
        self.args = args
        self.critic = critic
        self.population_size = self.args.pop_size
        self.num_elitists = max(int(self.args.elite_fraction * args.pop_size),1)
        self.evaluate = evaluate

        # testing stats
        self.stats = PopulationStats(self.args)
        self.mut_change = 0.0

        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded':0, 'total':0.0000001}

        # Mutation type:
        if self.args.mut_type == 'normal' or self.args.mut_type == 'inplace':
            self.mutate = self.mutate_inplace
        elif self.args.mut_type == 'proximal':
            self.mutate = self.proximal_mutate
        elif self.args.mut_type == 'safe':
            self.mutate = self.safe_mutate
        else:
            raise ValueError('Mutation type is unknown!')

    def selection_tournament(self, index_rank : List[int], num_offsprings : int, tournament_size : int ) -> List[GeneticAgent]:
        """ Returns a list of non-elite offsprings.
        """
        total_choices = len(index_rank)
        offsprings = []
        for _ in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[random.randint(0,len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        weight = torch.clamp(weight, -mag, mag)
        return weight

    def crossover_inplace(self, gene1: GeneticAgent, gene2: GeneticAgent):
        # Evaluate the parents
        trials = 5
        for param1, param2 in zip(gene1.actor.parameters(), gene2.actor.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2: #Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randint(0,num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0,W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                    else:
                        ind_cr = random.randint(0,W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]

            elif len(W1.shape) == 1: #Bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = random.randint(0,num_variables)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = random.randint(0,W1.shape[0])  #
                        W1[ind_cr] = W2[ind_cr]
                    else:
                        ind_cr = random.randint(0,W1.shape[0])  #
                        W2[ind_cr] = W1[ind_cr]

        # Evaluate the children

        if self.args.test_ea and self.args._verbose_crossover:
            test_score_p1 = 0
            for _ in range(trials):
                episode = self.evaluate(gene1, is_action_noise=False, store_transition=False)
                test_score_p1 += episode.reward
            test_score_p1 /= trials

            test_score_p2 = 0
            for _ in range(trials):
                episode = self.evaluate(gene2, is_action_noise=False, store_transition=False)
                test_score_p2 += episode.reward
            test_score_p2 /= trials

            test_score_c1 = 0
            for _ in range(trials):
                episode = self.evaluate(gene1, is_action_noise=False, store_transition=False)
                test_score_c1 += episode.reward

            test_score_c2 = 0
            for _ in range(trials):
                episode = self.evaluate(gene1, is_action_noise=False, store_transition=False)
                test_score_c2 += episode.reward
            test_score_c2 /= trials


            print("==================== Classic Crossover ======================")
            print(f"Parent 1: {test_score_p1:0.1f}")
            print(f"Parent 2: {test_score_p2:0.1f}")
            print(f"Child1 performance: {test_score_c1:0.2f}")
            print(f"Child2 performance: {test_score_c2:0.2f}")
            print(f"Benefit1: {test_score_c1 - max(test_score_p1,test_score_p2) :0.2f}")
            print(f"Benefit1: {test_score_c2 - max(test_score_p1,test_score_p2) :0.2f}")


    def distilation_crossover(self, gene1: GeneticAgent, gene2: GeneticAgent) -> GeneticAgent:
        new_agent = GeneticAgent(self.args)
        new_agent.buffer.add_latest_from(gene1.buffer, self.args.individual_bs // 2)
        new_agent.buffer.add_latest_from(gene2.buffer, self.args.individual_bs // 2)
        new_agent.buffer.shuffle()

        hard_update(new_agent.actor, gene2.actor)
        batch_size = min(128, len(new_agent.buffer))
        iters = len(new_agent.buffer) // batch_size
        losses = []
        for _ in range(12):
            for _ in range(iters):
                batch = new_agent.buffer.sample(batch_size)
                losses.append(new_agent.update_parameters(batch, gene1.actor, gene2.actor, self.critic))

        # test and print
        if self.args.test_ea and self.args._verbose_crossover:
            test_score_p1 = 0
            trials = 5
            for _ in range(trials):
                episode = self.evaluate(gene1, is_action_noise=False, store_transition=False)
                test_score_p1 += episode.reward
            test_score_p1 /= trials

            test_score_p2 = 0
            for _ in range(trials):
                episode = self.evaluate(gene2, is_action_noise=False, store_transition=False)
                test_score_p2 += episode.reward
            test_score_p2 /= trials

            test_score_c = 0
            for _ in range(trials):
                episode = self.evaluate(new_agent,  is_action_noise=False, store_transition=False)
                test_score_c += episode.reward
            test_score_c /= trials


            print("==================== Distillation Crossover ======================")
            print(f"MSE Loss: {np.mean(losses[-40:]):0.4f}")
            print(f"Parent 1: {test_score_p1:0.1f}")
            print(f"Parent 2: {test_score_p2:0.1f}")
            print(f"Child performance: {test_score_c:0.2f}")
            print(f"Benefit: {test_score_c - min(test_score_p1,test_score_p2) :0.2f} (>0 is better)")

            # self.stats.add({
            #     'cros_parent1_fit': test_score_p1,
            #     'cros_parent2_fit': test_score_p2,
            #     'cros_child_fit': test_score_c,
            # })

        return new_agent

    def proximal_mutate(self, gene: GeneticAgent, mag : float ):
        # Based on code from https://github.com/uber-research/safemutations
        model = gene.actor
        # sample mutation batch
        batch = gene.buffer.sample(min(self.args.mutation_batch_size, len(gene.buffer)))
        state, _, _, _, _ = batch
        output = model(state)

        params = model.extract_parameters()
        tot_size = model.count_parameters()
        num_outputs = output.size()[1]

        # initial perturbation
        normal = dist.Normal(torch.zeros_like(params), torch.ones_like(params) * mag)
        delta = normal.sample()

        # we want to calculate a jacobian of derivatives
        # of each output's sensitivity to each parameter
        jacobian = torch.zeros(num_outputs, tot_size).to(self.args.device)
        grad_output = torch.zeros(output.size()).to(self.args.device)

        # do a backward pass for each output
        for i in range(num_outputs):
            model.zero_grad()
            grad_output.zero_()
            grad_output[:, i] = 1.0

            output.backward(grad_output, retain_graph=True)
            jacobian[i] = model.extract_grad()

        # summed gradients sensitivity
        scaling = torch.sqrt((jacobian**2).sum(0))

        lam_max = 0.01
        scaling[scaling == 0] = 1.0
        scaling[scaling < lam_max] = lam_max
        delta /= scaling

        #update child actor net
        new_params = params + delta
        model.inject_parameters(new_params)

        # test
        if self.args.test_ea and self.args._verbose_mut:
            trials = 5
            test_score_p = 0
            for _ in range(trials):
                episode = self.evaluate(gene, is_action_noise=False, store_transition=False)
                test_score_p += episode.reward
            test_score_p /= trials

            test_score_c = 0
            for _ in range(trials):
                episode = self.evaluate(gene, is_action_noise=False, store_transition=False)
                test_score_c += episode.reward
            test_score_c /= trials

            self.stats.add({
                'mut_parent_fit': test_score_p,
                'mut_child_fit': test_score_c,
            })
            self.mut_change = 0.1*self.mut_change + 0.9*(test_score_c - test_score_p)/(-test_score_p)*100
            print("==================== Mutation ======================")
            print(f"Parent: {test_score_p:0.1f}")
            print(f"Child: {test_score_c:0.1f}")
            print(f'Delta: {torch.mean(delta).item()}')
            print(f'Average mutation change: {self.mut_change:0.2f} %')
            print(f"Mean mutation change: from {torch.mean(torch.abs(params)).item():0.3f} /\
                to {torch.mean(torch.abs(new_params)).item():0.3f} /\
                by {torch.mean(torch.abs(new_params - params)).item():0.3f}")

    def safe_mutate(self, gene: GeneticAgent, mag : float ):
        # Based on safety-infromed mutation from E. Marchesini.
        model = gene.actor

        if len(gene.critical_buffer) > 1:
            batch = gene.critical_buffer.sample(min(self.args.mutation_batch_size, len(gene.critical_buffer)))
        else:
            batch = gene.buffer.sample(min(self.args.mutation_batch_size, len(gene.buffer)))

        state, _, _, _, _ = batch
        output = model(state)

        params = model.extract_parameters()
        tot_size = model.count_parameters()
        num_outputs = output.size()[1]

        # initial perturbation
        normal = dist.Normal(torch.zeros_like(params), torch.ones_like(params) * mag)
        delta = normal.sample()

        # we want to calculate a jacobian of derivatives
        # of each output's sensitivity to each parameter
        jacobian = torch.zeros(num_outputs, tot_size).to(self.args.device)
        grad_output = torch.zeros(output.size()).to(self.args.device)

        # do a backward pass for each output
        for i in range(num_outputs):
            model.zero_grad()
            grad_output.zero_()
            grad_output[:, i] = 1.0

            output.backward(grad_output, retain_graph=True)
            jacobian[i] = model.extract_grad()

        # summed gradients sensitivity
        scaling = torch.sqrt((jacobian**2).sum(0))

        lam_max = 0.01
        scaling[scaling == 0] = 1.0
        scaling[scaling < lam_max] = lam_max
        delta /= scaling

        #update child actor net
        new_params = params + delta
        model.inject_parameters(new_params)

        # test
        if self.args.test_ea and self.args._verbose_mut:
            trials = 5
            test_score_p = 0
            for _ in range(trials):
                episode = self.evaluate(gene, is_action_noise=False, store_transition=False)
                test_score_p += episode.reward
            test_score_p /= trials

            test_score_c = 0
            for _ in range(trials):
                episode = self.evaluate(gene, is_action_noise=False, store_transition=False)
                test_score_c += episode.reward
            test_score_c /= trials

            self.stats.add({
                'mut_parent_fit': test_score_p,
                'mut_child_fit': test_score_c,
            })
            self.mut_change = 0.1*self.mut_change + 0.9*(test_score_c - test_score_p)/(-test_score_p)*100
            print("==================== Safe Mutation ======================")
            print(f"Parent: {test_score_p:0.1f}")
            print(f"Child: {test_score_c:0.1f}")
            print(f'Delta: {torch.mean(delta).item()}')
            print(f'Average mutation change: {self.mut_change:0.2f} %')
            print(f"Mean mutation change: from {torch.mean(torch.abs(params)).item():0.3f} /\
                to {torch.mean(torch.abs(new_params)).item():0.3f} /\
                by {torch.mean(torch.abs(new_params - params)).item():0.3f}")

    def mutate_inplace(self, gene: GeneticAgent, mag : float ):
        # normal mutation
        num_mutation_frac = 0.1

        # super mutation
        super_mut_strength = 10 * mag
        super_mut_prob = 0.05
        reset_prob = super_mut_prob + 0.05

        num_params = len(list(gene.actor.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.actor.state_dict()

        for i, key in enumerate(model_params): #Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2: # Weights, no bias

                num_weights= W.shape[0]*W.shape[1]
                ssne_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_mutations = random.randint(0,int(math.ceil(num_mutation_frac * num_weights)))  # Number of mutation instances
                    for _ in range(num_mutations):
                        ind_dim1 = random.randint(0,W.shape[0])
                        ind_dim2 = random.randint(0,W.shape[-1])
                        random_num = random.random()

                        if random_num < super_mut_prob:  # Super Mutation probability
                            W[ind_dim1, ind_dim2] += random.gauss(0, super_mut_strength * W[ind_dim1, ind_dim2])
                        elif random_num < reset_prob:  # Reset probability
                            W[ind_dim1, ind_dim2] = random.gauss(0, 1)
                        else:  # mutation even normal
                            W[ind_dim1, ind_dim2] += random.gauss(0, mag *W[ind_dim1, ind_dim2])

                        # Regularization hard limit
                        W[ind_dim1, ind_dim2] = self.regularize_weight(W[ind_dim1, ind_dim2], 1000000)

    def clone(self, master: GeneticAgent, replacee: GeneticAgent):  # Replace the replacee individual with master
        """ Copy weights and memories from master to replacee.
        """
        # clone params
        for target_param, source_param in zip(replacee.actor.parameters(), master.actor.parameters()):
            target_param.data.copy_(source_param.data)
        # clone memoery buffer
        replacee.buffer.reset()
        replacee.buffer.add_content_of(master.buffer)
        # clone buffeer of selected critical states
        replacee.critical_buffer.reset()
        replacee.critical_buffer.add_content_of(master.critical_buffer)

    def reset_genome(self, gene: GeneticAgent):
        for param in (gene.actor.parameters()):
            param.data.copy_(param.data)

    @staticmethod
    def sort_groups_by_fitness(genomes, fitness):
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                if fitness[first] < fitness[second]:
                    groups.append((second, first, fitness[first] + fitness[second]))
                else:
                    groups.append((first, second, fitness[first] + fitness[second]))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    @staticmethod
    def get_novelty(bcs : np.ndarray, first : int, second : int) -> np.float64:
        return np.linalg.norm(bcs[first,:] - bcs[second,:], axis = -1, ord =2)

    @staticmethod
    def sort_groups_by_novelty(genomes, bcs):
        groups = []
        for i,first in enumerate(genomes):
            for  _,second in enumerate(genomes[i+1:]):
                groups.append((second, first, SSNE.get_novelty(bcs, first, second)))
        return sorted(groups, key=lambda group: group[2], reverse=True)

    @staticmethod
    def get_distance(gene1: GeneticAgent, gene2: GeneticAgent):
        batch_size = min(256, min(len(gene1.buffer), len(gene2.buffer)))
        batch_gene1 = gene1.buffer.sample_from_latest(batch_size, 1000)
        batch_gene2 = gene2.buffer.sample_from_latest(batch_size, 1000)

        return gene1.actor.get_novelty(batch_gene2) + gene2.actor.get_novelty(batch_gene1)

    # @staticmethod
    # def get_novelty(gene1: GeneticAgent, gene2: GeneticAgent):
    #     """ Average action over a batch """
    #     batch_size = min(256, min(len(gene1.buffer), len(gene2.buffer)))
    #     batch_gene1 = gene1.buffer.sample_from_latest(batch_size, 1000)
    #     batch_gene2 = gene2.buffer.sample_from_latest(batch_size, 1000)

    #     return gene1.actor.get_novelty(batch_gene2) + gene2.actor.get_novelty(batch_gene1)

    @staticmethod
    def sort_groups_by_distance(genomes, pop):
        """ Adds all posssible parent-pairs to a group,
        then sorts them based on distance from largest to smallest.

        Args:
            genomes (_type_): Parent wieghts.
            pop (_type_): List of genetic actors.

        Returns:
            list : sorted groups from most different to msot similar
        """
        groups = []
        for i, first in enumerate(genomes):
            for second in genomes[i+1:]:
                groups.append((second, first, SSNE.get_distance(pop[first], pop[second])))

        return sorted(groups, key=lambda group: group[2], reverse=True)

    def epoch (self, pop: List[GeneticAgent], fitness_evals : np.array or List[float], bcs_evals : np.array = None):
        """ One generation update. Entire epoch is handled with indices;
            Index ranks  nets by fitness evaluation - 0 is the best after reversing.
        Args:
            pop (List[GeneticAgent]):List of gentic actors.
            fitness_evals (np.arrayorList[float]): List of fitness values of each actor.
            bcs_evals (np.array, optional): List of behavioural characteristics (tuples) of each actor. Defaults to None.

        Raises:
            NotImplementedError: Unknwon operator use for crossover or mutation.
        """
        # NOTE fitness and bcs arrays remain unsorted

        index_rank = np.argsort(fitness_evals)[::-1]
        elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard -- first indeces
        # print('Elites:', elitist_index)

        '''    Selection   '''
        # offsprings are kept for crossover and mutation together with elites
        offsprings = self.selection_tournament(index_rank,
                                               num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        new_elitists = []
        unselects = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # COMPUTE RL_SELECTION RATE
        if self.rl_policy is not None: # RL Transfer happened
            self.selection_stats['total'] += 1.0

            if self.rl_policy in elitist_index: self.selection_stats['elite'] += 1.0
            elif self.rl_policy in offsprings: self.selection_stats['selected'] += 1.0
            elif self.rl_policy in unselects: self.selection_stats['discarded'] += 1.0
            self.rl_policy = None

        # Elitism
        # >> assigning elite candidates to some unselects
        for i in elitist_index:
            try: replacee = unselects.pop(0)
            except: replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replacee=pop[replacee])

        ''' Crossover '''
        # >> between elite and offsprings for the unselected genes with 100 percent probability
        # >> offspring gets an empty buffer
        if self.args.distil_crossover:
            if 'fitness' in self.args.distil_type.lower():
                sorted_groups = SSNE.sort_groups_by_fitness(new_elitists + offsprings, fitness_evals)
            elif 'dist' in self.args.distil_type.lower():
                sorted_groups = SSNE.sort_groups_by_distance(new_elitists + offsprings, pop)
            # elif 'novelty' in self.args.distil_type.lower() and bcs_evals is not None:
            #     print('BC distil crossover')
                sorted_groups = SSNE.sort_groups_by_novelty(new_elitists + offsprings, bcs_evals)
            else:
                raise NotImplementedError('Unknown distilation type')

            for i, unselected_actor in enumerate(unselects):
                first, second, _ = sorted_groups[i % len(sorted_groups)]
                if fitness_evals[first] < fitness_evals[second]:
                    first, second = second, first
                offspring = self.distilation_crossover(pop[first], pop[second])
                self.clone(offspring, pop[unselected_actor])
        else:
            if len(unselects) % 2 != 0:  # Number of unselects left should be even
                unselects.append(unselects[random.randint(0,len(unselects))])
            for i, j in zip(unselects[0::2], unselects[1::2]):
                off_i = random.choice(new_elitists)
                off_j = random.choice(offsprings)
                self.clone(master=pop[off_i], replacee=pop[i])
                self.clone(master=pop[off_j], replacee=pop[j])
                self.crossover_inplace(pop[i], pop[j])

        # Crossover for selected offsprings
        if self.args.crossover_prob > 0.01:  # so far this is not called
            for i in offsprings:
                if random.random() < self.args.mutation_prob:
                    others = offsprings.copy()
                    others.remove(i)
                    off_j = random.choice(others)
                    self.clone(self.distilation_crossover(pop[i], pop[off_j]), pop[i])

        '''   Mutation  '''
        #  Mutate all genes in the population  EXCEPT the new elitists
        # -> buffer is kept
        for i in index_rank[self.num_elitists:]:
            if random.random() < self.args.mutation_prob:
                self.mutate(pop[i], mag=self.args.mutation_mag)

        self.stats.reset()

        return new_elitists[0]


def unsqueeze(array, axis=1):
    if axis == 0: return np.reshape(array, (1, len(array)))
    elif axis == 1: return np.reshape(array, (len(array), 1))


class PopulationStats:
    def __init__(self, args: Parameters, file='population.csv'):
        self.data = {}
        self.args = args
        self.save_path = os.path.join(args.save_foldername, file)
        self.generation = 0

        if not os.path.exists(args.save_foldername):
            os.makedirs(args.save_foldername)

    def add(self, res):
        for k, v in res.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def log(self):
        with open(self.save_path, 'a+') as f:
            if self.generation == 0:
                f.write('generation,')
                for i, k in enumerate(self.data):
                    if i > 0:
                        f.write(',')
                    f.write(k)
                f.write('\n')

            f.write(str(self.generation))
            f.write(',')
            for i, k in enumerate(self.data):
                if i > 0:
                    f.write(',')
                f.write(str(np.mean(self.data[k])))
            f.write('\n')

    def should_log(self):
        return self.generation % self.args.opstat_freq == 0 and self.args.opstat

    def reset(self):
        for k in self.data:
            self.data[k] = []
        self.generation += 1
