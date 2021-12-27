# import os
import os

import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F


class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        self.actor = torch.load(os.path.join(cur_dir, "necto-model.pt"))

    def act(self, state):
        # Evaluate your model here
        state = tuple(torch.from_numpy(s).float() for s in state)
        with torch.no_grad():
            out = self.actor(state)

        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ]
        ).swapdims(0, 1).squeeze()

        dist = Categorical(logits=logits)
        actions = dist.sample().numpy()

        actions = actions.reshape((-1, 5))
        actions[:, 0] = actions[:, 0] - 1
        actions[:, 1] = actions[:, 1] - 1

        parsed = np.zeros((actions.shape[0], 8))
        parsed[:, 0] = actions[:, 0]  # throttle
        parsed[:, 1] = actions[:, 1]  # steer
        parsed[:, 2] = actions[:, 0]  # pitch
        parsed[:, 3] = actions[:, 1] * (1 - actions[:, 4])  # yaw
        parsed[:, 4] = actions[:, 1] * actions[:, 4]  # roll
        parsed[:, 5] = actions[:, 2]  # jump
        parsed[:, 6] = actions[:, 3]  # boost
        parsed[:, 7] = actions[:, 4]  # handbrake

        return parsed[0]
