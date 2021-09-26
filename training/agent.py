import torch
from earl_pytorch import EARLPerceiver, ControlsPredictorDiscrete
from torch import nn
from torch.nn import Linear

from rocket_learn.agent.actor_critic_agent import ActorCriticAgent
from rocket_learn.agent.discrete_policy import DiscretePolicy


class Necto(nn.Module):  # Wraps earl + an output and takes only a single input
    def __init__(self, earl, output):
        super().__init__()
        self.earl = earl
        self.output = output

    def forward(self, inp):
        q, kv, m = inp
        res = self.output(self.earl(q, kv, m))
        if isinstance(res, tuple):
            return tuple(r for r in res)
        return res


def get_critic():
    return Necto(EARLPerceiver(128, 1, 4, 1, query_features=32, key_value_features=24),
                 Linear(128, 1))


def get_actor():
    return DiscretePolicy(Necto(EARLPerceiver(256, 1, 4, 1, query_features=32, key_value_features=24),
                                ControlsPredictorDiscrete(256)))


def get_agent(actor_lr, critic_lr=None):
    actor = get_actor()
    critic = get_critic()
    optim = torch.optim.Adam([
        {"params": actor.parameters(), "lr": actor_lr},
        {"params": critic.parameters(), "lr": critic_lr if critic_lr is not None else actor_lr}
    ])

    agent = ActorCriticAgent(actor=actor, critic=critic, optimizer=optim)
    return agent
