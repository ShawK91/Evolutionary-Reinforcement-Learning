from envs_repo.gym_wrapper import GymWrapper

class EnvConstructor:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            env_name (str): Env name


    """
    def __init__(self, env_name, frameskip):
        """
        A general Environment Constructor
        """
        self.env_name = env_name
        self.frameskip = frameskip

        #Dummy env to get some macros
        dummy_env = self.make_env()
        self.is_discrete = dummy_env.is_discrete
        self.state_dim = dummy_env.state_dim
        self.action_dim = dummy_env.action_dim


    def make_env(self, **kwargs):
        """
        Generate and return an env object
        """
        env = GymWrapper(self.env_name, self.frameskip)
        return env



