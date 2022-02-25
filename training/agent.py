from typing import Optional

import numpy as np
import torch
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from torch import nn
from torch.nn import Linear, Sequential, ReLU

from earl_pytorch.util.util import mlp
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from training.parser import NectoActionTEST


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, features=32, layers=2, actions=None):
        super().__init__()
        if actions is None:
            self.actions = torch.from_numpy(NectoActionTEST.make_lookup_table()).float()
        else:
            self.actions = torch.from_numpy(actions).float()
        self.net = mlp(8, features, layers)
        self.emb_convertor = nn.Linear(in_features, features)

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None):
        if actions is None:
            actions = self.actions
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions)
        if act_emb.ndim == 3:
            return torch.einsum("bad,bpd->bpa", act_emb, player_emb)
        return torch.einsum("ad,bpd->bpa", act_emb, player_emb)


class Necto(nn.Module):  # Wraps earl + an output and takes only a single input
    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.relu = ReLU()
        self.output = output

    def forward(self, inp):
        q, kv, m = inp
        res = self.earl(q, kv, m)
        res = self.output(self.relu(res))
        if isinstance(res, tuple):
            return tuple(r for r in res)
        return res


def get_critic():
    return Necto(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                 Linear(128, 1))


def get_actor():
    # split = (3, 3, 2, 2, 2)
    split = (90,)
    return DiscretePolicy(Necto(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                                ControlsPredictorDot(128)), split)


def get_agent(actor_lr, critic_lr=None):
    actor = get_actor()
    critic = get_critic()
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": actor_lr},
        {"params": critic.parameters(), "lr": critic_lr if critic_lr is not None else actor_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    return agent


if __name__ == '__main__':
    d = DiscretePolicy(Necto(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
                             ControlsPredictorDot(128)), (90,))
    # d = DiscretePolicy(Necto(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
    #                          ControlsPredictorDiscrete(128)))
    q, kv, m = (torch.ones((4, 1, 32)), torch.ones((4, 41, 24)), torch.zeros((4, 41)))
    dist = d.get_action_distribution((q, kv, m))
    kv[:, 0, 0] = 0
    q[0, :, :] = 0
    kv[1, :, :] = 0
    m[2, 0] = 1

    dist = d.get_action_distribution((q, kv, m))
    # print(torch.all((dist.logits == dist2.logits), dim=2))
    act = d.sample_action(dist)
    act[:] = act[0]
    lp = d.log_prob(dist, act)
    ent = d.entropy(dist, act)

    print(act)
    print(lp)
    print(ent)
