import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


class Agent:
    def __init__(self):
        cur_dir = os.path.dirname(os.path.realpath(__file__))
        with open(os.path.join(cur_dir, "nexto-model.pt"), 'rb') as f:
            self.actor = torch.jit.load(f)
        torch.set_num_threads(1)
        self._lookup_table = self.make_lookup_table()
        self.state = None

    @staticmethod
    def make_lookup_table():
        actions = []
        # Ground
        for throttle in (-1, 0, 1):
            for steer in (-1, 0, 1):
                for boost in (0, 1):
                    for handbrake in (0, 1):
                        if boost == 1 and throttle != 1:
                            continue
                        actions.append([throttle or boost, steer, 0, steer, 0, 0, boost, handbrake])
        # Aerial
        for pitch in (-1, 0, 1):
            for yaw in (-1, 0, 1):
                for roll in (-1, 0, 1):
                    for jump in (0, 1):
                        for boost in (0, 1):
                            if jump == 1 and yaw != 0:  # Only need roll for sideflip
                                continue
                            if pitch == roll == jump == 0:  # Duplicate with ground
                                continue
                            # Enable handbrake for potential wavedashes
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def act(self, state, beta):
        state = tuple(torch.from_numpy(s).float() for s in state)

        with torch.no_grad():
            out, weights = self.actor(state)
        self.state = state

        out = (out,)
        max_shape = max(o.shape[-1] for o in out)
        logits = torch.stack(
            [
                l
                if l.shape[-1] == max_shape
                else F.pad(l, pad=(0, max_shape - l.shape[-1]), value=float("-inf"))
                for l in out
            ],
            dim=1
        )

        # beta = 0.5
        if beta == 1:
            actions = np.argmax(logits, axis=-1)
        elif beta == -1:
            actions = np.argmin(logits, axis=-1)
        else:
            if beta == 0:
                logits[torch.isfinite(logits)] = 0
            else:
                logits *= math.log((beta + 1) / (1 - beta), 3)
            dist = Categorical(logits=logits)
            actions = dist.sample()

        # print(Categorical(logits=logits).sample())
        parsed = self._lookup_table[actions.numpy().item()]

        return parsed, weights
