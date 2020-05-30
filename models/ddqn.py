import torch, random
import torch.nn as nn
from torch.distributions import Categorical, Normal
from core.utils import GumbelSoftmax



class DDQN(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim, epsilon_start, epsilon_end, epsilon_decay_frames):
        super(DDQN, self).__init__()
        self.action_dim = action_dim
        l1 = 128; l2 = 128



        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.f1 = nn.Linear(state_dim, l1)
        #self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        #self.q1ln2 = nn.LayerNorm(l2)

        #Value
        self.val = nn.Linear(l2, 1)


        #Advantages
        self.adv = nn.Linear(l2, action_dim)

        #self.half()
        #weights_init_(self, lin_gain=1.0, bias_gain=0.1)

        #Epsilon Decay
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_rate = (epsilon_start - epsilon_end) / epsilon_decay_frames



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
        info = torch.selu(self.f1(obs))
        #q1 = self.q1ln1(q1)
        info = torch.selu(self.f2(info))
        #q1 = self.q1ln2(q1)


        val = self.val(info)
        adv = self.adv(info)

        logits = val + adv - adv.mean()


        if return_only_action:
            return logits.argmax(1)
        else:
            return logits.argmax(1), None, logits

    def noisy_action(self, obs, return_only_action=True):
        _, _, logits = self.clean_action(obs, return_only_action=False)

        # dist = GumbelSoftmax(temperature=1, logits=q)
        # action = dist.sample()
        # #action = q.argmax(1)

        if random.random() < self.epsilon:
            action = torch.Tensor([random.randint(0, self.action_dim-1)]).int()
            if self.epsilon > self.epsilon_end:
                self.epsilon -= self.epsilon_decay_rate
        else:
            action = logits.argmax(1)


        if return_only_action:
            return action

        #log_prob = dist.log_prob(action)

        #print(action[0].detach().item(), log_prob[0].detach().item())
        return action, None, logits