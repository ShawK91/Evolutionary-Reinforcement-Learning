
import torch
import torch.nn as nn
from torch.distributions import Normal, RelaxedOneHotCategorical
from core.utils import weights_init_
import torch.nn.functional as F

def weights_init_(m, lin_gain=1.0, bias_gain=0.1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=lin_gain)
        torch.nn.init.constant_(m.bias, bias_gain)


class Gaussian_FF(nn.Module):

    def __init__(self, num_inputs, num_actions, hidden_size):
        super(Gaussian_FF, self).__init__()

        self.num_actions = num_actions

        #Shared FF
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.mean_linear = nn.Linear(hidden_size, num_actions)

        self.log_std_linear = nn.Linear(hidden_size, num_actions)

        # SAC SPECIFIC
        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.epsilon = 1e-6



    def clean_action(self, state, return_only_action=True):

        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)

        if return_only_action: return torch.tanh(mean)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.LOG_SIG_MIN, max=self.LOG_SIG_MAX)
        return mean, log_std


    def noisy_action(self, state,return_only_action=True):
        mean, log_std = self.clean_action(state, return_only_action=False)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        action = torch.tanh(x_t)

        if return_only_action:
            return action

        log_prob = normal.log_prob(x_t)

        # Enforcing Action Bound
        log_prob -= torch.log(1 - action.pow(2) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        return action, log_prob, None,None,torch.tanh(mean)




    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean

class Tri_Head_Q(nn.Module):


    def __init__(self, state_dim, action_dim, hidden_size):
        super(Tri_Head_Q, self).__init__()

        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim, hidden_size)

        # Hidden Layer 2
        self.q1f2 = nn.Linear(hidden_size, hidden_size)

        # Out
        self.q1out = nn.Linear(hidden_size, 1)

        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim, hidden_size)

        # Hidden Layer 2
        self.q2f2 = nn.Linear(hidden_size, hidden_size)

        # Out
        self.q2out = nn.Linear(hidden_size, 1)

    def forward(self, obs, action):


        #Concatenate observation+action as critic state
        state = torch.cat([obs, action], 1)

        ###### Q1 HEAD ####
        q1 = F.relu(self.q1f1(state))
        #q1 = self.q1ln1(q1)
        q1 = F.relu(self.q1f2(q1))
        #q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = F.relu(self.q2f1(state))
        #q2 = self.q2ln1(q2)
        q2 = F.relu(self.q2f2(q2))
        #q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)


        return q1, q2, None





