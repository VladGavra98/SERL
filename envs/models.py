import copy
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
import gym
from envs.phlabenv import printGreen


class Model (gym.Env):
    """
     Base class for online model identification.
    """

    def __init__(self, config_dict: dict, env: gym.Env):
        self.config = config_dict

        # Compatibility for RL use:
        self.env = env

        # Dimensions
        self.state_size = min(10, self.config['state_size'], env.n_obs_full)
        self.action_size = env.action_space.shape[0]

        # Time
        self.dt = self.env.dt

    def reset(self):
        raise NotImplementedError

    def get_X(self, state, action):
        """ Format and concatenate state and action vectors.

        Args:
            state (np.ndarray): State.
            action (np.ndarray): Action

        Returns:
            np.ndarray: Input data array. Shape: 1 x state_size
        """
        state = state[:self.state_size]
        X = np.hstack([state, action])

        return X.reshape(-1, self.state_size + self.action_size)

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """
        Update RLS parameters based on one state-action sample.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.
            next_state (np.ndarray): Next state.
        """
        raise NotImplementedError

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """   Predict next state based on RLS model

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.

        Returns:
            np.ndarray: Predicted state.
        """
        raise NotImplementedError

    def step(self, action: np.ndarray) -> tuple:
        """ Simualate one step in the local environment.

        Args:
            action (np.ndarray): Un-sclaled action in the interval [-1,1] to be taken.

        Returns:
            Tuple[np.ndarray, float, bool, dict]: new_state, obtained reward, is_done mask, inof:{refference signal value, time, current state}
        """
        is_done = False

        # scale action to actuator rate bounds
        action = self.env.scale_action(action)   # scaled to actuator limits

        # model step
        x = copy.copy(self.env.x)

        x = np.divide(x, self.scale)
        pred_x = self.predict(x, action).flatten()
        pred_x = np.hstack((pred_x, np.array([0., 0.])))
        pred_x = np.multiply(pred_x, self.scale[:pred_x.size])
        self.env.x = copy.copy(pred_x)

        # Reward using clipped error
        reward = self.env.get_reward()

        # Calcualte cost:
        cost = self.env.get_cost()

        # Update observation based on perfect observations & actuator state
        self.env.obs = np.hstack(
            (self.env.error, self.env.x[self.env.obs_idx]))
        self.env.last_u = action

        # Check for bounds and addd corresponding penalty for dying early
        is_done, penalty = self.env.check_bounds()
        reward += penalty

        # Step time
        self.env.t += self.env.dt
        self.t += self.env.dt
        self.env.last_obs = self.env.obs

        # info:
        info = {
            "ref": self.env.ref,
            "x":   self.env.x,
            "t":   self.env.t,
            "cost": cost,
        }
        return self.env.obs, reward, is_done, info

    def predictive_control(self, controller: object, t_horizon: float = 5.0, **kwargs) -> tuple:
        """ Predicts the trajectory of a controller (i.e., actor)
            for a finite horizon in the future.

        Args:
            controller (object): Actor object to be used during prediction.
            t_horizon (float, optional): Prediction horizon. Defaults to 5.

        Returns:
            tuple: List of rewards, list of time-steps played.
        """
        t_hor = t_horizon
        done = False
        rewards, times = [], []

        # Initliase local progress:
        imagine_obs = self.env.obs
        t_start = self.env.t

        # play the future time-horizon
        while self.env.t < t_start + t_hor and not done:

            # select action:
            action = controller.select_action(imagine_obs)

            # Simulate one step in environment
            action = np.clip(action, -1, 1)

            # step
            next_obs, reward, done, _ = self.step(action.flatten())
            imagine_obs = next_obs

            # save
            rewards.append(reward)
            times.append(self.env.t)

        return rewards, times

    def sync_env(self, source_env: gym.Env, **kwargs) -> None:
        """ Synchronize local environment state with another environment object (base/main).

        Args:
            new_env (gym.Env): Source environment to be copied from.
        """
        if kwargs.get('verbose'):
            printGreen(
                '\nState of online learning environment is synched with global env.\n')
        self.env.__dict__ = source_env.__dict__.copy()

    def save_weights(self, save_dir):
        raise NotImplementedError

    def load_weights(self, save_dir):
        raise NotImplementedError

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                           Recursive Least Squares
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class RLS (Model):
    """
    Recursive Least Squares (RLS) incremental environment model
    """

    def __init__(self, config_dict: dict, env: gym.Env):
        super().__init__(config_dict, env)

        # Specific config passing
        self.gamma = self.config["gamma"]

        # Initialize covariance matrix
        self.Cov0 = self.config["cov0"] * \
            np.identity(self.state_size + self.action_size)

        # Low pass constant:
        self.tau = 0.0

        # Initial innovation (prediction error)
        self.epsilon_thresh = np.ones(
            self.state_size) * self.config["eps_thresh"]
        self.Cov_reset = False

        # Scale:
        self.scale = np.array([10, 20, 10, 100, 1, 1, 1, 1, 1, 2000, 1, 1])

    @property
    def F(self):
        return np.float32(self.Theta[: self.state_size, :].T)

    @property
    def G(self):
        return np.float32(self.Theta[self.state_size:, :].T)

    def reset(self):
        # Initialize measurement matrix
        self.X = np.ones((self.state_size + self.action_size, 1))

        # Initialise parameter matrix
        self.Theta = np.zeros(
            (self.state_size + self.action_size, self.state_size))

        # State vector:
        self.x: np.array = np.zeros(self.state_size)

        # Error vector and Covarience matrix:
        self.epsilon = np.zeros((1, self.state_size))
        self.Cov = self.Cov0

        # Time
        self.t = 0.

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """
        Update RLS parameters based on one state-action sample.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.
            next_state (np.ndarray): Next state.
        """
        state = state[:self.state_size]
        next_state = next_state[:self.state_size]

        state = np.divide(state, self.scale[:state.size])
        next_state = np.divide(next_state, self.scale[:state.size])

        # Predict next state
        next_state_pred = self.predict(state, action)

        # Error
        self.epsilon = (np.array(next_state)[np.newaxis].T - next_state_pred).T

        # Intermediate computations
        CovX = self.Cov @ self.X
        XCov = self.X.T @ self.Cov
        gammaXCovX = self.gamma + XCov @ self.X

        # Update parameter matrix
        _theta = self.Theta + (CovX @ self.epsilon) / gammaXCovX
        self.Theta = self.tau * self.Theta + (1 - self.tau) * _theta

        # Update covariance matrix
        self.Cov = (self.Cov - (CovX @ XCov) / gammaXCovX) / self.gamma

        # Check if Cov needs reset
        if self.Cov_reset is False:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 1:
                self.Cov_reset = True
                self.Cov = self.Cov0
        elif self.Cov_reset is True:
            if np.sum(np.greater(np.abs(self.epsilon), self.epsilon_thresh)) == 0:
                self.Cov_reset = False

        self.t += self.dt

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """        Predict next state based on RLS model

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.

        Returns:
            np.ndarray: Predicted state.
        """
        state = state[:self.state_size]

        # Set measurement matrix
        self.X = self.get_X(state, action).T

        # Predict next state
        next_state_pred = (self.X.T @ self.Theta).T

        return next_state_pred

    def save_weights(self, save_dir):
        """
        Save current weights
        """

        np.savez(
            save_dir + ".npz",
            x=self.X,
            theta=self.Theta,
            cov=self.Cov,
            epsilon=self.epsilon,
        )

    def load_weights(self, save_dir):
        """
        Load weights
        """

        # Weights npz
        npzfile = np.load(save_dir + ".npz")

        # Load weights
        self.X = npzfile["x"]
        self.Theta = npzfile["theta"]
        self.Cov = npzfile["cov"]
        self.epsilon = npzfile["epsilon"]


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                               Gaussian Process
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class GP(Model, GaussianProcessRegressor):
    def __init__(self, config: dict, env: gym.Env):
        super().__init__(config, env)

        self.kernel = config['kernel']
        self.random_state = config['seed']

        self.t = 0.
        self.dt = self.env.dt
        # Dimensions
        self.state_size = min(self.config['state_size'], env.n_obs_full)
        self.action_size = env.action_space.shape[0]

        self.sk_model = GaussianProcessRegressor(
            kernel=self.kernel, random_state=self.random_state)

    def reset(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray):
        X = self.get_X(state, action)
        y = next_state[:self.state_size].reshape(-1, self.state_size)

        self.sk_model.fit(X, y)
        print('Reset GPR')

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """
            Update RLS parameters based on one state-action sample.

            Args:
                state (np.ndarray): Current state.
                action (np.ndarray): Current action.
                next_state (np.ndarray): Next state.
        """
        self.kernel.set_params(**(self.sk_model.kernel_.get_params()))
        self.sk_model = GaussianProcessRegressor(
            kernel=self.kernel, random_state=self.random_state)
        X = self.get_X(state, action)
        y = next_state[:self.state_size].reshape(-1, self.state_size)
        self.sk_model.fit(X, y)

        self.t += self.dt

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """   Predict next state based on RLS model

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.

        Returns:
            np.ndarray: Predicted state.
        """
        x = self.get_X(state, action)
        y_pred, _ = self.sk_model.predict(x, return_std=True)

        return y_pred


# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#                                    MLP
# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


class MLP(Model):
    def __init__(self, config: dict, env: gym.Env):
        super().__init__(config, env)

        self.random_state = config['seed']

        self.t = 0.
        self.dt = self.env.dt

        # Dimensions
        self.state_size = min(self.config['state_size'], env.n_obs_full)
        self.action_size = env.action_space.shape[0]

        self.sk_model = MLPRegressor(hidden_layer_sizes=(50, 50, 50), activation='relu',
                                     solver='adam',
                                     alpha=0.0001,
                                     batch_size='auto',
                                     learning_rate='adaptive',
                                     learning_rate_init=0.001,
                                     power_t=0.5, max_iter=200,
                                     shuffle=True, random_state=self.random_state)

    def update(self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray) -> None:
        """
            Update RLS parameters based on one state-action sample.

            Args:
                state (np.ndarray): Current state.
                action (np.ndarray): Current action.
                next_state (np.ndarray): Next state.
        """
        # format input data
        X = self.get_X(state, action)
        y = next_state[:self.state_size].reshape(1, -1)

        # parital fit
        self.sk_model.partial_fit(X, y)

        self.t += self.dt

    def predict(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """   Predict next state based on RLS model

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Current action.

        Returns:
            np.ndarray: Predicted state.
        """
        x = self.get_X(state, action)
        y_pred = self.sk_model.predict(x)

        return y_pred
