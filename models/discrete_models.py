import torch, random
import torch.nn as nn
from torch.distributions import  Normal, RelaxedOneHotCategorical, Categorical
import torch.nn.functional as F

class CategoricalPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, hidden_size):
        super(CategoricalPolicy, self).__init__()
        self.action_dim = action_dim


        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.f1 = nn.Linear(state_dim, hidden_size)
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Value
        self.val = nn.Linear(hidden_size, 1)

        #Advantages
        self.adv = nn.Linear(hidden_size, action_dim)



    def clean_action(self, obs, return_only_action=True):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        ###### Feature ####
        info = torch.relu(self.f1(obs))
        info = torch.relu(self.f2(info))

        val = self.val(info)
        adv = self.adv(info)

        logits = val + adv - adv.mean()

        if return_only_action:
            return logits.argmax(1)

        return None, None, logits

    def noisy_action(self, obs, return_only_action=True):
        _, _, logits = self.clean_action(obs, return_only_action=False)

        dist = Categorical(logits=logits)
        action = dist.sample()
        action = action

        if return_only_action:
            return action

        return action, None, logits



class GumbelPolicy(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, hidden_size, epsilon_start, epsilon_end, epsilon_decay_frames):
        super(GumbelPolicy, self).__init__()
        self.action_dim = action_dim


        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.f1 = nn.Linear(state_dim, hidden_size)
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(hidden_size, hidden_size)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Value
        self.val = nn.Linear(hidden_size, 1)

        #Advantages
        self.adv = nn.Linear(hidden_size, action_dim)

        #Temperature
        self.log_temp = torch.nn.Linear(hidden_size, 1)

        self.LOG_TEMP_MAX = 2
        self.LOG_TEMP_MIN = -10


    def clean_action(self, obs, return_only_action=True):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        ###### Feature ####
        info = torch.relu(self.f1(obs))
        info = torch.relu(self.f2(info))

        val = self.val(info)
        adv = self.adv(info)

        logits = val + adv - adv.mean()

        if return_only_action:
            return logits.argmax(1)
        else:
            log_temp = self.log_temp(info)
            log_temp = torch.clamp(log_temp, min=self.LOG_TEMP_MIN, max=self.LOG_TEMP_MAX)

            return logits.argmax(1), log_temp, logits

    def noisy_action(self, obs, return_only_action=True):
        _, log_temp, logits = self.clean_action(obs, return_only_action=False)

        temp = log_temp.exp()
        dist = RelaxedOneHotCategorical(temperature=temp, probs=F.softmax(logits, dim=1))
        action = dist.rsample()

        if return_only_action:
            return action.argmax(1)

        log_prob = dist.log_prob(action)
        log_prob = torch.diagonal(log_prob, offset=0).unsqueeze(1)


        return action.argmax(1), log_prob, logits



