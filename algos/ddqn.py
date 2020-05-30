import os, random
import torch
import torch.nn.functional as F
from torch.optim import Adam
from core.utils import soft_update, hard_update


class DDQN(object):
    def __init__(self, args, model_constructor):

        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.actor_lr)

        self.actor_target = model_constructor.make_model('CategoricalPolicy').to(device=self.device)
        hard_update(self.actor_target, self.actor)

        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.softmax = torch.nn.Softmax(dim=1)
        self.num_updates = 0

    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch):

        state_batch = state_batch.to(self.device)
        next_state_batch=next_state_batch.to(self.device)
        action_batch=action_batch.to(self.device)
        reward_batch=reward_batch.to(self.device)
        done_batch=done_batch.to(self.device)

        action_batch = action_batch.long().unsqueeze(1)
        with torch.no_grad():
            na = self.actor.clean_action(next_state_batch,  return_only_action=True)
            _, _, ns_logits = self.actor_target.noisy_action(next_state_batch, return_only_action=False)
            next_entropy = -(F.softmax(ns_logits, dim=1) * F.log_softmax(ns_logits, dim=1)).mean(1).unsqueeze(1)

            ns_logits = ns_logits.gather(1, na.unsqueeze(1))

            next_target = ns_logits + self.alpha * next_entropy
            next_q_value = reward_batch + (1-done_batch) * self.gamma * next_target


        _, _, logits  = self.actor.noisy_action(state_batch, return_only_action=False)
        entropy = -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).mean(1).unsqueeze(1)
        q_val = logits.gather(1, action_batch)

        q_loss = (next_q_value - q_val)**2
        q_loss -= self.alpha*entropy
        q_loss = q_loss.mean()


        self.actor_optim.zero_grad()
        q_loss.backward()
        self.actor_optim.step()

        self.num_updates += 1
        soft_update(self.actor_target, self.actor, self.tau)

