import random
from typing import Any

import numpy as np
from gym import Space
from gym.spaces import Tuple, Box
from numba import njit
from rlgym.gym import Gym
from rlgym.utils import ObsBuilder
from rlgym.utils.action_parsers import DefaultAction
from rlgym.utils.common_values import BOOST_LOCATIONS
from rlgym.utils.gamestates import GameState, PlayerData

from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder
from rocket_learn.utils.gamestate_encoding import encode_gamestate
from rocket_learn.utils.gamestate_encoding import StateConstants as SC

# from training.scoreboard import Scoreboard
from rocket_learn.utils.scoreboard import Scoreboard

IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP, HAS_JUMP = range(20, 25)
ACTIONS = range(25, 33)
GOAL_DIFF, TIME_LEFT, IS_OVERTIME = range(33, 36)


# BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
# ACTIONS = range(24, 32)


class NectoObsBuilder(BatchedObsBuilder):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 5 + [1] * 30)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1, 10, 1, 1, 1] + [1] * 30)

    def __init__(self, scoreboard: Scoreboard, env: Gym = None, n_players=6, tick_skip=8):
        super().__init__(scoreboard)
        self.env = env
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip

    def _reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(self.n_players or len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        if self.scoreboard is not None and self.scoreboard.random_resets and self.env is not None:
            self.env.update_settings(boost_consumption=random.random() > 0.02)

    def pre_step(self, state: GameState):
        if state != self.current_state:
            if self.scoreboard is not None and self.scoreboard.random_resets and self.env is not None:
                boost_consumption_rate = self.env._match._boost_consumption  # noqa blame Rangler
                if boost_consumption_rate <= 0:
                    for player in state.players:
                        player.boost_amount = float("inf")
                else:
                    for player in state.players:
                        player.boost_amount /= boost_consumption_rate
        super(NectoObsBuilder, self).pre_step(state)

    def get_obs_space(self) -> Space:
        players = self.n_players or 6
        entities = 1 + players + len(self._boost_locations)
        return Tuple((
            Box(-np.inf, np.inf, (1, len(self._invert) - 30 + 8)),
            Box(-np.inf, np.inf, (entities, len(self._invert))),
            Box(-np.inf, np.inf, (entities,)),
        ))

    @staticmethod
    def _quats_to_rot_mtx(quats: np.ndarray) -> np.ndarray:
        # From rlgym.utils.math.quat_to_rot_mtx
        w = -quats[:, 0]
        x = -quats[:, 1]
        y = -quats[:, 2]
        z = -quats[:, 3]

        theta = np.zeros((quats.shape[0], 3, 3))

        norm = np.einsum("fq,fq->f", quats, quats)

        sel = norm != 0

        w = w[sel]
        x = x[sel]
        y = y[sel]
        z = z[sel]

        s = 1.0 / norm[sel]

        # front direction
        theta[sel, 0, 0] = 1.0 - 2.0 * s * (y * y + z * z)
        theta[sel, 1, 0] = 2.0 * s * (x * y + z * w)
        theta[sel, 2, 0] = 2.0 * s * (x * z - y * w)

        # left direction
        theta[sel, 0, 1] = 2.0 * s * (x * y - z * w)
        theta[sel, 1, 1] = 1.0 - 2.0 * s * (x * x + z * z)
        theta[sel, 2, 1] = 2.0 * s * (y * z + x * w)

        # up direction
        theta[sel, 0, 2] = 2.0 * s * (x * z + y * w)
        theta[sel, 1, 2] = 2.0 * s * (y * z - x * w)
        theta[sel, 2, 2] = 1.0 - 2.0 * s * (x * x + y * y)

        return theta

    @staticmethod
    def convert_to_relative(q, kv):
        kv[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
        # kv[..., POS] -= q[..., POS]
        forward = q[..., FW]
        theta = np.arctan2(forward[..., 0], forward[..., 1])
        theta = np.expand_dims(theta, axis=-1)
        ct = np.cos(theta)
        st = np.sin(theta)
        xs = kv[..., POS.start:ANG_VEL.stop:3]
        ys = kv[..., POS.start + 1:ANG_VEL.stop:3]
        # Use temp variables to prevent modifying original array
        nx = ct * xs - st * ys
        ny = st * xs + ct * ys
        kv[..., POS.start:ANG_VEL.stop:3] = nx  # x-components
        kv[..., POS.start + 1:ANG_VEL.stop:3] = ny  # y-components

    @staticmethod
    def add_relative_components(q, kv):
        forward = q[..., FW]
        up = q[..., UP]
        left = np.cross(up, forward)

        pitch = np.arctan2(forward[..., 2], np.sqrt(forward[..., 0] ** 2 + forward[..., 1] ** 2))
        yaw = np.arctan2(forward[..., 1], forward[..., 0])
        roll = np.arctan2(left[..., 2], up[..., 2])

        pitch = np.expand_dims(pitch, axis=-1)
        yaw = np.expand_dims(yaw, axis=-1)
        roll = np.expand_dims(roll, axis=-1)

        cr = np.cos(roll)
        sr = np.sin(roll)
        cp = np.cos(pitch)
        sp = np.sin(pitch)
        cy = np.cos(yaw)
        sy = np.sin(yaw)

        # Each of these holds 5 values for each player for each tick
        vals = kv[..., POS.start:ANG_VEL.stop]
        vals[..., POS.start:LIN_VEL.stop] -= q[..., POS.start:LIN_VEL.stop]
        xs = vals[..., 0::3]
        ys = vals[..., 1::3]
        zs = vals[..., 2::3]

        # Rotation matrix with only yaw
        flip_relative_xs = cy * xs - sy * ys
        flip_relative_ys = sy * xs + cy * ys
        flip_relative_zs = zs

        # Now full rotation matrix
        car_relative_xs = cp * cy * xs + (sr * sp * cy - cr * sy) * ys - (cr * sp * cy + sr * sy) * zs
        car_relative_ys = cp * sy * xs + (sr * sp * sy + cr * cy) * ys - (cr * sp * sy - sr * cy) * zs
        car_relative_zs = sp * xs - cp * sr * ys + cp * cr * zs

        all_rows = np.concatenate(
            (flip_relative_xs, flip_relative_ys, flip_relative_zs,
             car_relative_xs, car_relative_ys, car_relative_zs), axis=-1)
        kv[..., ACTIONS.start:] = all_rows

    @staticmethod
    @njit
    def _update_timers(boost_timers, self_boost_locations, demo_timers, self_tick_skip,
                       boost_states: np.ndarray, demo_states: np.ndarray):
        for i in range(1, boost_timers.shape[0]):
            for b in range(boost_states.shape[1]):
                if boost_states[i - 1, b] == 0:  # Not available
                    prev_timer = boost_timers[i - 1, b]
                    if prev_timer > 0:
                        boost_timers[i, b] = max(0, prev_timer - self_tick_skip / 120)
                    elif self_boost_locations[b, 2] > 72:
                        boost_timers[i, b] = 10
                    else:
                        boost_timers[i, b] = 4
                else:  # Available
                    boost_timers[i, b] = 0

        for i in range(1, demo_timers.shape[0]):
            for b in range(demo_states.shape[1]):
                if demo_states[i - 1, b] == 1:  # Demoed
                    prev_timer = demo_timers[i - 1, b]
                    if prev_timer > 0:
                        demo_timers[i, b] = max(0, prev_timer - self_tick_skip / 120)
                    else:
                        demo_timers[i, b] = 3
                else:  # Not demoed
                    demo_timers[i, b] = 0

    def batched_build_obs(self, encoded_states: np.ndarray):
        if self.boost_timers is None or self.demo_timers is None:
            # if obs is being rebuilt, need to generate timers
            #
            self.demo_timers = np.zeros(self.n_players)
            self.boost_timers = np.zeros(34)

        ball_start_index = 3 + GameState.BOOST_PADS_LENGTH
        players_start_index = ball_start_index + GameState.BALL_STATE_LENGTH
        player_length = GameState.PLAYER_INFO_LENGTH

        n_players = (encoded_states.shape[1] - players_start_index) // player_length

        # need to give the same num of max players as workers
        lim_players = n_players if self.n_players is None else self.n_players
        n_entities = lim_players + 1 + 34  # Includes player+ball+boosts

        teams = encoded_states[0, players_start_index + 1::player_length]

        # Update boost and demo timers
        # Need to create them here since numba does not support array creation
        boost_states = encoded_states[:, 3:3 + 34]
        boost_timers = np.zeros((boost_states.shape[0] + 1, self.boost_timers.shape[0]))
        boost_timers[0, :] = self.boost_timers

        demo_states = encoded_states[:, players_start_index + 33::player_length]
        demo_timers = np.zeros((demo_states.shape[0] + 1, self.demo_timers.shape[0]))
        demo_timers[0, :] = self.demo_timers
        self._update_timers(boost_timers, self._boost_locations,
                            demo_timers, self.tick_skip,
                            boost_states, demo_states)
        boost_timers = boost_timers[1:]
        demo_timers = demo_timers[1:]
        self.boost_timers = boost_timers[-1, :]
        self.demo_timers = demo_timers[-1, :]

        # SELECTORS
        sel_players = slice(0, lim_players)
        sel_ball = sel_players.stop
        sel_boosts = slice(sel_ball + 1, None)

        # MAIN ARRAYS
        q = np.zeros((n_players, encoded_states.shape[0], 1, 25 + 8 + 3))
        kv = np.zeros((n_players, encoded_states.shape[0], n_entities, 25 + 30))
        m = np.zeros((n_players, encoded_states.shape[0], n_entities))  # Mask is shared

        # SCOREBOARD
        blue_score = encoded_states[:, SC.BALL_ANGULAR_VELOCITY.start + 9]
        orange_score = encoded_states[:, SC.BALL_ANGULAR_VELOCITY.start + 10]
        ticks_left = encoded_states[:, SC.BALL_ANGULAR_VELOCITY.start + 11]

        is_overtime = (ticks_left > 0) & np.isinf(ticks_left)
        goal_diff = np.clip(blue_score - orange_score, -5, 5) / 5
        time_left = (~is_overtime) * np.clip(ticks_left, 0, 300) / (120 * 60 * 5)
        q[teams == 0, :, 0, GOAL_DIFF] = goal_diff
        q[teams == 1, :, 0, GOAL_DIFF] = -goal_diff
        q[:, :, 0, TIME_LEFT] = time_left
        q[:, :, 0, IS_OVERTIME] = is_overtime

        # BALL
        kv[:, :, sel_ball, 3] = 1
        kv[:, :, sel_ball, np.r_[POS, LIN_VEL, ANG_VEL]] = encoded_states[:, ball_start_index: ball_start_index + 9]

        # BOOSTS
        # big_boost_mask = self._boost_locations[:, 2] > 72
        kv[:, :, sel_boosts, IS_BOOST] = 1
        kv[:, :, sel_boosts, POS] = self._boost_locations  # [big_boost_mask]
        kv[:, :, sel_boosts, BOOST] = 1
        kv[:, :, sel_boosts, DEMO] = boost_timers

        # PLAYERS
        kv[:, :, :n_players, IS_MATE] = 1 - teams  # Default team is blue
        kv[:, :, :n_players, IS_OPP] = teams
        for i in range(n_players):
            encoded_player = encoded_states[:,
                             players_start_index + i * player_length: players_start_index + (i + 1) * player_length]

            kv[i, :, i, IS_SELF] = 1
            kv[:, :, i, POS] = encoded_player[:, SC.CAR_POS_X.start: SC.CAR_POS_Z.start + 1]
            kv[:, :, i, LIN_VEL] = encoded_player[:, SC.CAR_LINEAR_VEL_X.start: SC.CAR_LINEAR_VEL_Z.start + 1]
            quats = encoded_player[:, SC.CAR_QUAT_W.start: SC.CAR_QUAT_Z.start + 1]
            rot_mtx = self._quats_to_rot_mtx(quats)
            kv[:, :, i, FW] = rot_mtx[:, :, 0]
            kv[:, :, i, UP] = rot_mtx[:, :, 2]
            kv[:, :, i, ANG_VEL] = encoded_player[:, SC.CAR_ANGULAR_VEL_X.start: SC.CAR_ANGULAR_VEL_Z.start + 1]
            kv[:, :, i, BOOST] = np.clip(encoded_player[:, SC.BOOST_AMOUNT.start], 0, 2)
            kv[:, :, i, DEMO] = demo_timers[:, i]
            kv[:, :, i, ON_GROUND] = encoded_player[:, SC.ON_GROUND.start]
            kv[:, :, i, HAS_FLIP] = encoded_player[:, SC.HAS_FLIP.start]
            kv[:, :, i, HAS_JUMP] = encoded_player[:, SC.HAS_JUMP.start]

        kv[teams == 1] *= self._invert
        kv[np.argwhere(teams == 1), ..., (IS_MATE, IS_OPP)] = kv[
            np.argwhere(teams == 1), ..., (IS_OPP, IS_MATE)]  # Swap teams

        kv /= self._norm

        for i in range(n_players):
            q[i, :, 0, :HAS_JUMP + 1] = kv[i, :, i, :HAS_JUMP + 1]

        self.add_relative_components(q, kv)

        # MASK
        m[:, :, n_players: lim_players] = 1

        return [(q[i], kv[i], m[i]) for i in range(n_players)]

    def add_actions(self, obs: Any, previous_actions: np.ndarray, player_index=None):
        if player_index is None:
            for (q, kv, m), act in zip(obs, previous_actions):
                q[:, 0, ACTIONS] = act
        else:
            q, kv, m = obs[player_index]
            q[:, 0, ACTIONS] = previous_actions


if __name__ == '__main__':
    import rlgym


    class CombinedObs(ObsBuilder):
        def __init__(self, *obsbs):
            super().__init__()
            self.obsbs = obsbs

        def reset(self, initial_state: GameState):
            for obsb in self.obsbs:
                obsb.reset(initial_state)

        def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
            obss = []
            for obsb in self.obsbs:
                obss.append(obsb.build_obs(player, state, previous_action))
            return obss


    env = rlgym.make(use_injector=True, self_play=True, team_size=3,
                     obs_builder=CombinedObs(NectoObsBuilder(Scoreboard(), n_players=6), NectoObsOLD()))

    states = []
    actions = [[np.zeros(8)] for _ in range(6)]
    done = False
    obs, info = env.reset(return_info=True)
    obss = [[o] for o in obs]
    states.append(info["state"])
    while not done:
        act = [env.action_space.sample() for _ in range(6)]
        for a, arr in zip(act, actions):
            arr.append(a)
        obs, reward, done, info = env.step(act)
        for os, o in zip(obss, obs):
            os.append(o)
        states.append(info["state"])

    obs_b = NectoObsBuilder(Scoreboard(), n_players=6)

    enc_states = np.array([encode_gamestate(s) for s in states])
    actions = np.array(actions)

    # FIXME ensure obs corresponds to old obs
    # FIXME ensure reconstructed obs is *exactly* the same as obs
    # reconstructed_obs = obs_b.reset(GameState(enc_states[0].tolist()))
    reconstructed_obs = obs_b.batched_build_obs(enc_states)
    ap = DefaultAction()
    obs_b.add_actions(reconstructed_obs, ap.parse_actions(actions.reshape(-1, 8), None).reshape(actions.shape))

    formatted_obss = []
    for player_obs in obss:
        transposed = tuple(zip(*player_obs))
        obs_tensor = tuple(np.vstack(t) for t in transposed)
        formatted_obss.append(obs_tensor)

    for o0, o1 in zip(formatted_obss, reconstructed_obs):
        for arr0, arr1 in zip(o0, o1):
            if not np.all(arr0 == arr1):
                print("Error")

    print("Hei")
