import numpy as np
import torch
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from torch import nn
from torch.nn import Linear, Sequential, ReLU

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy


class Necto(nn.Module):  # Wraps earl + an output and takes only a single input
    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.relu = ReLU()
        self.output = output

    def forward(self, inp):
        q, kv, m = inp
        res = self.output(self.relu(self.earl(q, kv, m)))
        if isinstance(res, tuple):
            return tuple(r for r in res)
        return res


def get_critic():
    # return Sequential(
    #     Linear(107, 256), ReLU(),
    #     Linear(256, 256), ReLU(),
    #     Linear(256, 256), ReLU(),
    #     Linear(256, 256), ReLU(),
    #     Linear(256, 1))
    return Necto(EARLPerceiver(128, 1, 4, 1, query_features=32, key_value_features=24),
                 Linear(128, 1))


def get_actor():
    # return DiscretePolicy(Sequential(Linear(107, 256), ReLU(),
    #                                  Linear(256, 256), ReLU(),
    #                                  Linear(256, 256), ReLU(),
    #                                  Linear(256, 256), ReLU(),
    #                                  ControlsPredictorDiscrete(256)))

    return DiscretePolicy(Necto(EARLPerceiver(128, 1, 4, 1, query_features=32, key_value_features=24),
                                ControlsPredictorDiscrete(128, splits=(3, 3, 2, 2, 2))),
                          index_action_map=np.array([-1, 0, 1], [-1, 0, 1], [0, 1], [0, 1], [0, 1]))


def get_agent(actor_lr, critic_lr=None):
    actor = get_actor()
    critic = get_critic()
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": actor_lr},
        {"params": critic.parameters(), "lr": critic_lr if critic_lr is not None else actor_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    return agent
