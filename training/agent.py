from typing import Optional

import numpy as np
import torch
from torch.nn.init import xavier_uniform_

from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from torch import nn
from torch.nn import Linear, Sequential, ReLU

from earl_pytorch.util.util import mlp
from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy
from training.parser import NectoAction


class ControlsPredictorDot(nn.Module):
    def __init__(self, in_features, features=32, layers=1, actions=None):
        super().__init__()
        if actions is None:
            self.actions = torch.from_numpy(NectoAction.make_lookup_table()).float()
        else:
            self.actions = torch.from_numpy(actions).float()
        self.net = mlp(8, in_features, layers, features)  # Default 8->256->32
        self.emb_convertor = nn.Linear(in_features, features)

    def forward(self, player_emb: torch.Tensor, actions: Optional[torch.Tensor] = None):
        if actions is None:
            actions = self.actions
        player_emb = self.emb_convertor(player_emb)
        act_emb = self.net(actions.to(player_emb.device))

        if act_emb.ndim == 2:
            return torch.einsum("ad,bpd->bpa", act_emb, player_emb)

        return torch.einsum("bad,bpd->bpa", act_emb, player_emb)


class Necto(nn.Module):  # Wraps earl + an output and takes only a single input
    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.relu = ReLU()
        self.output = output
        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model. Taken from PyTorch Transformer impl"""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)

    def forward(self, inp):
        q, kv, m = inp
        res = self.earl(q, kv, m)
        weights = None
        if isinstance(res, tuple):
            res, weights = res
        res = self.output(self.relu(res))
        if isinstance(res, tuple):
            res = tuple(r[:, 0, :] for r in res)
        else:
            res = res[:, 0, :]
        if weights is None:
            return res
        return res, weights


def get_critic():
    return Necto(EARLPerceiver(256, 4, 8, 1, query_features=36, key_value_features=25 + 30),
                 Linear(256, 1))


def get_actor():
    split = (90,)
    return DiscretePolicy(Necto(EARLPerceiver(256, 4, 8, 1, query_features=36, key_value_features=25 + 30),
                                ControlsPredictorDot(256)), split)


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
    necto = torch.jit.load("../src/necto-model-base.pt")
    earl = EARLPerceiver(128, 1, 4, 1, query_features=32, key_value_features=24)
    splits = (90,)  # (3,) * 2 + (2,) * 3
    d = DiscretePolicy(Necto(earl,
                             ControlsPredictorDot(128)), splits)
    # d = DiscretePolicy(Necto(EARLPerceiver(128, 2, 4, 1, query_features=32, key_value_features=24),
    #                          ControlsPredictorDiscrete(128)))
    q, kv, m, a = (torch.normal(0, 1, size=(4, 1, 32)),
                   torch.normal(0, 1, size=(4, 41, 24)),
                   torch.zeros((4, 41)),
                   torch.normal(0, 1, size=(90, 8)).unsqueeze(0).repeat(4, 1, 1))
    dist = d.get_action_distribution((q, kv, m))
    # kv[:, 0, 0] = 0
    # q[0, :, :] = 0
    # kv[1, :, :] = 0
    # m[2, 0] = 1

    dist2 = d.get_action_distribution((q[:1], kv[:1], m[:1]))
    # print(torch.all((dist.logits == dist2.logits), dim=2))
    act = d.sample_action(dist)
    act[:] = act[0]
    lp = d.log_prob(dist, act)
    ent = d.entropy(dist, act)

    print(act)
    print(d.env_compatible(act))
    print(lp)
    print(ent)
