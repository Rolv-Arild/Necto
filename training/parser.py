from typing import Any

import gym.spaces
import numpy as np
from gym.spaces import Discrete
from rlgym.utils.action_parsers import ActionParser
from rlgym.utils.gamestates import GameState


# from rlgym_tools.extra_action_parsers.kbm_act import KBMAction

# NectoActionOLD = KBMAction


class NectoAction(ActionParser):
    def __init__(self):
        super().__init__()
        self._lookup_table = self.make_lookup_table()

    @staticmethod
    def make_lookup_table() -> np.array:
        """
        Defines possible actions

            throttle(int) -> [-1, 1]
            steer(int) -> [-1, 1]
            pitch(int) -> [-1, 1]
            yaw(int) -> [-1, 1]
            roll(int) -> [-1, 1]

            jump(bool) -> [0, 1]
            boost(bool) -> [0, 1]
            handbrake(bool) -> [0, 1]

        - Conditions
            if boost == 1
                -> throttle = 1 (prevents braking and boosting)
            if jump == 1
                -> yaw = 0 (Only need roll for sideflip)
            if pitch == roll == jump == 0
                -> don't add action (duplicated with ground action)
            if jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                -> handbrake = 1 (enables possible wavedashing)

        - Array format (one action)
            [throttle (or boost), steer (or yaw), pitch, yaw (or steer), roll, jump, boost, handbrake]

        Returns:
            Numpy array with possible actions
        """
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
                            if jump == 1 and yaw != 0:
                                continue
                            if pitch == roll == jump == 0:
                                continue
                            handbrake = jump == 1 and (pitch != 0 or yaw != 0 or roll != 0)
                            actions.append([boost, yaw, pitch, yaw, roll, jump, boost, handbrake])
        actions = np.array(actions)
        return actions

    def get_action_space(self) -> gym.spaces.Space:
        """
        Retrieves action space

        Returns:
            Discrete action space with size of possible actions
        """
        return Discrete(len(self._lookup_table))

    def parse_actions(self, actions: Any, state: GameState) -> np.ndarray:
        """
        Pass through that allows both multiple types of agent actions while still parsing Nectos

        Strip out fillers, pass through 8sets, get look up table values, recombine

        Args:
            actions (Any): Array of actions
            state (GameState)

        Returns:
            Numpy ndarray with parsed actions
        """
        parsed_actions = []
        for action in actions:
            # support reconstruction
            if action.size != 8:
                if action.shape == 0:
                    action = np.expand_dims(action, axis=0)
                # to allow different action spaces, pad out short ones (assume later unpadding in parser)
                action = np.pad(action.astype('float64'), (0, 8 - action.size), 'constant', constant_values=np.NAN)

            if np.isnan(action).any():  # its been padded, delete to go back to original
                stripped_action = (action[~np.isnan(action)]).squeeze().astype('int')
                parsed_actions.append(self._lookup_table[stripped_action])
            else:
                parsed_actions.append(action)

        return np.asarray(parsed_actions)


if __name__ == '__main__':
    ap = NectoAction()
    print(ap.get_action_space())
