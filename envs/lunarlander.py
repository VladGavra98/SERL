import numpy as np
import typing
import gym.wrappers

# class NormalizedActions(gym.ActionWrapper):
#     """ Wrapper class to resclae the gym env actions.
#         from [-1, 1] => [0, 1]

#     Args:
#         gym (module): Gym environment
#     """
#     def action(self, action):
#         action = (action + 1) / 2  # [-1, 1] => [0, 1]
#         action *= (self.action_space.high - self.action_space.low)
#         action += self.action_space.low
#         return action

  


class LunarLanderWrapper(gym.Wrapper):
    env = gym.make('LunarLanderContinuous-v2')

    def __init__(self, base_env = env):
        super().__init__(base_env)
        self.env = base_env

    def simulate (self, actor : object,
                render : bool = False,
                broken_engine : bool = False, 
                state_noise : bool = False, 
                noise_intensity : float = 0.05) -> tuple:
        """ Wrapper function for the gym LunarLander enviornment. It can include the faulty cases:
                -> broken main engine
                -> faulty navigation snesors (i.e., nosiy position)

        Args:
            actor (object): Actor class that has the select_action() method
            env (object): Environment with OpenAI Gym API (make(), reset(),step())
            render (bool, optional): Should render the video env. Defaults to False.
            broken_engine (bool, optional): Clip the main engine to 75% (fault cases for evaluation only). Defaults to False.
            state_noise (bool, optional): Add zero-mean Gaussian noise to the (x,y) observations (fault cases for evaluation only). Defaults to False.                               
            noise_intensity (float,opptional): Intensity (SD) of the Gaussian noise added to the state. Defaults to 0.05.                                
        Returns:
            tuple: Reward (float), Imapct x-position (float), impact y-velocity (float)
        """
        
        total_reward = 0.0
        impact_x_pos = None
        impact_y_vel = None
        all_y_vels = []

        # reset env
        done = False
        state = self.env.reset()

        while not done:
            if render:
                self.env.render()

            # Actor:
            action = self.scale_action(actor.select_action(np.array(state)))

            # Simulate one step in environment
            # fault -- break main engine by cliping it to 75%
            if broken_engine:
                action[0] = np.clip(action[0], -1., 0.5)
            # Step
            next_state, reward, done, info = self.env.step(action.flatten())

            # Update
            total_reward += reward
            state = next_state

            ## afault -- noisy position estimation
            if state_noise:
                noise = np.random.normal(0,noise_intensity,2)
                state[:2] = state[:2] + noise

            
            # Boudnary characteristics (BCs):
            x_pos = state[0]
            y_vel = state[3]
            leg0_touch = bool(state[6])
            leg1_touch = bool(state[7])
            all_y_vels.append(y_vel)

            # Check if the lunar lander is impacting for the first time.
            if impact_x_pos is None and (leg0_touch or leg1_touch):
                impact_x_pos = x_pos
                impact_y_vel = y_vel

        # if no impact edge case
        if impact_x_pos is None:
            impact_x_pos = x_pos
            impact_y_vel = min(all_y_vels)

        return {'total_reward':total_reward, 'bcs': (impact_x_pos,impact_y_vel)}

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        """Scale the actions from [-1, 1] to the appropriate scale of the action space."""
        low, high = self.env.action_space.low, self.env.action_space.high
        action = low + 0.5 * (action + 1.0) * (high - low)
        return action
