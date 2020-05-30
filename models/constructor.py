import torch

class ModelConstructor:

    def __init__(self, state_dim, action_dim, hidden_size, actor_seed=None, critic_seed=None):
        """
        A general Environment Constructor
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.actor_seed = actor_seed
        self.critic_seed = critic_seed


    def make_model(self, type, seed=False):
        """
        Generate and return an model object
        """

        if type == 'Gaussian_FF':
            from models.continous_models import Gaussian_FF
            model = Gaussian_FF(self.state_dim, self.action_dim, self.hidden_size)
            if seed:
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)


        elif type == 'Tri_Head_Q':
            from models.continous_models import Tri_Head_Q
            model = Tri_Head_Q(self.state_dim, self.action_dim, self.hidden_size)
            if seed:
                model.load_state_dict(torch.load(self.critic_seed))
                print('Critic seeded from', self.critic_seed)

        elif type == 'GumbelPolicy':
            from models.discrete_models import GumbelPolicy
            model = GumbelPolicy(self.state_dim, self.action_dim, self.hidden_size)

        elif type == 'CategoricalPolicy':
            from models.discrete_models import CategoricalPolicy
            model = CategoricalPolicy(self.state_dim, self.action_dim, self.hidden_size)


        else:
            AssertionError('Unknown model type')


        return model



