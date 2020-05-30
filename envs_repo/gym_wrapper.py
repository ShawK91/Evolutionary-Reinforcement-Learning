
import numpy as np
import gym


class GymWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, frameskip=None):
        """
        A base template for all environment wrappers.
        """
        self.env_name = env_name
        self.env = gym.make(env_name)
        self.frameskip=frameskip
        self.is_discrete = self.is_discrete(self.env)

        #State and Action Parameters
        self.state_dim = self.env.observation_space.shape[0]
        if self.is_discrete:
            self.action_dim = self.env.action_space.n
        else:
            self.action_dim = self.env.action_space.shape[0]
            self.action_low = float(self.env.action_space.low[0])
            self.action_high = float(self.env.action_space.high[0])
        self.test_size = 10

    def reset(self):
        """Method overloads reset
            Parameters:
                None

            Returns:
                next_obs (list): Next state
        """
        state = self.env.reset()
        return np.expand_dims(state, axis=0)

    def step(self, action): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                action (ndarray): action to take in the env

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                done (bool): Simulation done?
                info (None): Template from OpenAi gym (doesnt have anything)
        """
        if  self.is_discrete:
           action = action[0]
        else:
            #Assumes action is in [-1, 1] --> Hyperbolic Tangent Activation
            action = self.action_low + (action + 1.0) / 2.0 * (self.action_high - self.action_low)

        reward = 0
        for _ in range(self.frameskip):
            next_state, rew, done, info = self.env.step(action)
            reward += rew
            if done: break

        next_state = np.expand_dims(next_state, axis=0)
        return next_state, reward, done, info

    def render(self):
        self.env.render()

    def is_discrete(self, env):
        try:
            k = env.action_space.n
            return True
        except:
            return False

