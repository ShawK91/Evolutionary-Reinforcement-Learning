# ******************************************************************************
# Copyright 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ******************************************************************************

import torch, random
import torch.nn as nn
from torch.distributions import Categorical, Normal
from core.utils import GumbelSoftmax

#SAC SPECIFIC
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6


class QActor(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions, policy_type):
        super(QActor, self).__init__()

        self.policy_type = policy_type; self.num_actions = num_actions

        h1=128; h2 =128

        #Shared FF
        self.linear1 = nn.Linear(num_inputs, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.mean_linear = nn.Linear(h2, num_actions)

        #Knet
        #self.knet = KNet(num_inputs+g1, h1, 5, h2)

        if self.policy_type == 'GaussianPolicy':
            self.log_std_linear = nn.Linear(h2, num_actions)

        elif self.policy_type == 'DeterministicPolicy':
            self.noise = Normal(0, 0.01)

        self.apply(weights_init_)



    def clean_action(self, state, return_only_action=True):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        #x = self.knet(state)
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        mean = self.mean_linear(x)

        if return_only_action or self.policy_type == 'DeterministicPolicy': return mean.argmax(1)

        elif self.policy_type == 'GaussianPolicy':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            return mean, log_std


    def noisy_action(self, state,return_only_action=True):

        if self.policy_type == 'GaussianPolicy':
            mean, log_std = self.clean_action(state, return_only_action=False)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            action = x_t.argmax(1)

            if return_only_action:
                return action
            log_prob = normal.log_prob(x_t)



            return action, log_prob, x_t

        elif self.policy_type == 'DeterministicPolicy':
            mean = self.clean_action(state)
            action = mean + torch.clamp(self.noise.sample((len(state), self.num_actions)), min=-0.5, max=0.5)
            Exception('Not Implemented Noisy Action for deterministic QActor')

            if return_only_action: return action
            else: return action, None, None


    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean


class Actor(nn.Module):
    """Actor model
        Parameters:
              args (object): Parameter class
    """

    def __init__(self, num_inputs, num_actions, policy_type):
        super(Actor, self).__init__()

        self.policy_type = policy_type; self.num_actions = num_actions

        h1=128; h2 =128

        #Shared FF
        self.linear1 = nn.Linear(num_inputs, h1)
        self.linear2 = nn.Linear(h1, h2)
        self.mean_linear = nn.Linear(h2, num_actions)

        #Knet
        #self.knet = KNet(num_inputs+g1, h1, 5, h2)

        if self.policy_type == 'GaussianPolicy':
            self.log_std_linear = nn.Linear(h2, num_actions)

        elif self.policy_type == 'DeterministicPolicy':
            self.noise = Normal(0, 0.01)

        self.apply(weights_init_)



    def clean_action(self, state, return_only_action=True):
        """Method to forward propagate through the actor's graph
            Parameters:
                  input (tensor): states
            Returns:
                  action (tensor): actions
        """
        #x = self.knet(state)
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))
        mean = self.mean_linear(x)

        if return_only_action or self.policy_type == 'DeterministicPolicy': return torch.tanh(mean)

        elif self.policy_type == 'GaussianPolicy':
            log_std = self.log_std_linear(x)
            log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
            return mean, log_std


    def noisy_action(self, state,return_only_action=True):

        if self.policy_type == 'GaussianPolicy':
            mean, log_std = self.clean_action(state, return_only_action=False)
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            action = torch.tanh(x_t)


            if return_only_action:
                return action

            log_prob = normal.log_prob(x_t)

            # Enforcing Action Bound
            log_prob -= torch.log(1 - action.pow(2) + epsilon)
            log_prob = log_prob.sum(1, keepdim=True)

            # print(' Mean', mean[0,:].detach().item(), ' Mean+Activ', torch.tanh(mean[0,:]).detach().item(), ' Action', action[0,:].detach().item(), ' Log_STD',log_std[0,:].detach().item(),
            #       ' STD', std[0,:].detach().item(),  ' Log_Prob', log_prob[0,:].detach().item())

            return action, log_prob, None,None,torch.tanh(mean)

        elif self.policy_type == 'DeterministicPolicy':
            mean = self.clean_action(state)
            action = mean + torch.clamp(self.noise.sample((len(state), self.num_actions)), min=-0.5, max=0.5)


            if return_only_action: return action
            else: return action, None, None, None, None


    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean

class Critic(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        l1 = 128; l2 = 128



        ######################## Q1 Head ##################
        # Construct Hidden Layer 1 with state
        self.q1f1 = nn.Linear(state_dim + action_dim, l1)
        self.q1ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q1f2 = nn.Linear(l1, l2)
        self.q1ln2 = nn.LayerNorm(l2)

        #Out
        self.q1out = nn.Linear(l2, 1)


        ######################## Q2 Head ##################
        # Construct Hidden Layer 1 with state
        self.q2f1 = nn.Linear(state_dim + action_dim, l1)
        self.q2ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.q2f2 = nn.Linear(l1, l2)
        self.q2ln2 = nn.LayerNorm(l2)

        #Out
        self.q2out = nn.Linear(l2, 1)

        #self.half()
        self.apply(weights_init_)


        # ######################## Value Head ##################  [NOT USED IN CERL]
        # # Construct Hidden Layer 1 with
        # self.vf1 = nn.Linear(state_dim, l1)
        # self.vln1 = nn.LayerNorm(l1)
        #
        # # Hidden Layer 2
        # self.vf2 = nn.Linear(l1, l2)
        # self.vln2 = nn.LayerNorm(l2)
        #
        # # Out
        # self.vout = nn.Linear(l2, 1)





    def forward(self, obs,action):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states
                   input (tensor): actions

             Returns:
                   Q1 (tensor): Qval 1
                   Q2 (tensor): Qval 2
                   V (tensor): Value



         """

        #Concatenate observation+action as critic state
        state = torch.cat([obs, action], 1)

        ###### Q1 HEAD ####
        q1 = torch.relu(self.q1f1(state))
        #q1 = self.q1ln1(q1)
        q1 = torch.relu(self.q1f2(q1))
        #q1 = self.q1ln2(q1)
        q1 = self.q1out(q1)

        ###### Q2 HEAD ####
        q2 = torch.relu(self.q2f1(state))
        #q2 = self.q2ln1(q2)
        q2 = torch.relu(self.q2f2(q2))
        #q2 = self.q2ln2(q2)
        q2 = self.q2out(q2)

        # ###### Value HEAD ####
        # v = torch.tanh(self.vf1(obs))
        # v = self.vln1(v)
        # v = torch.tanh(self.vf2(v))
        # v = self.vln2(v)
        # v = self.vout(v)

        #self.half()

        return q1, q2, None




# Initialize Policy weights
def weights_init_(m, lin_gain=1.0, bias_gain=0.1):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=lin_gain)
        torch.nn.init.constant_(m.bias, bias_gain)




class KNet(nn.Module):
    """Actor model

        Parameters:
              args (object): Parameter class
    """

    def __init__(self, input_dim, neuron_dim, col_dim, output_dim):
        super(KNet, self).__init__()

        #Core Network
        self.linear1 = nn.Linear(input_dim, neuron_dim)
        self.linear2 = nn.Linear(neuron_dim, neuron_dim)
        self.linear_out = nn.Linear(neuron_dim, output_dim)


        #Column Distribution
        self.map1, self.entropy1 = self.init_allocation(neuron_dim, col_dim)
        self.map2, self.entropy2 = self.init_allocation(neuron_dim, col_dim)

        self.apply(weights_init_)


    def forward(self, input):
        """Method to forward propagate through the actor's graph

            Parameters:
                  input (tensor): [batch_size, entry]

            Returns:
                  action (tensor): [batch_size, entry]


        """
        #First Layer
        out = self.linear1(input)
        out = self.apply_map(out, self.map1, self.entropy1)

        #Second Layer
        out = self.linear2(out)
        out = self.apply_map(out, self.map2, self.entropy2)

        #Final Layer
        out = self.linear_out(out)

        return out

    def get_norm_stats(self):
        minimum = min([torch.min(param).item() for param in self.parameters()])
        maximum = max([torch.max(param).item() for param in self.parameters()])
        means = [torch.mean(torch.abs(param)).item() for param in self.parameters()]
        mean = sum(means)/len(means)

        return minimum, maximum, mean

    def apply_map(self, input, map, entropy):

        ls = []
        for i, col in enumerate(map):
            local_out = torch.softmax(input[:,col], dim=1)
            #nput[:,col] = local_out
            ls.append(local_out)

            #Entropy
            entropy[i] = (local_out * torch.log(local_out)).sum()

        out = torch.cat(ls, axis=1)
        return out


    def map_update(self):
        if random.random<0.05:
            max_ent_col = self.entropy1.index(max([v.item() for v in self.entropy1]))
            min_ent_col = self.entropy1.index(min([v.item() for v in self.entropy1]))
            if self.map1[max_ent_col] > 2:
                node = self.map1[max_ent_col].pop(0)
                self.map1[min_ent_col].append(node)

        if random.random<0.05:
            max_ent_col = self.entropy2.index(max([v.item() for v in self.entropy2]))
            min_ent_col = self.entropy2.index(min([v.item() for v in self.entropy2]))
            if self.map2[max_ent_col] > 2:
                node = self.map2[max_ent_col].pop(0)
                self.map2[min_ent_col].append(node)

    def init_allocation(self, neurone_dim, col_dim):
        allocation = []
        ration = int(neurone_dim/col_dim)
        counter = 0
        for i in range(col_dim):
            allocation.append([])
            for _ in range(ration):
                allocation[-1].append(counter)
                counter+=1

        #Put all remaining to last entry
        while counter < neurone_dim:
            allocation[-1].append(counter)
            counter += 1

        entropy = [None for _ in range(col_dim)]

        return allocation, entropy








