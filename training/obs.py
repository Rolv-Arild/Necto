from typing import Any

import numpy as np
from rlgym.utils import ObsBuilder
from rlgym.utils.common_values import BOOST_LOCATIONS, BLUE_TEAM, ORANGE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData

from rocket_learn.utils.batched_obs_builder import BatchedObsBuilder


class NectoObsBuilder(ObsBuilder):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, n_players=6, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip

    def reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(self.n_players)
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.current_state = initial_state

    def _maybe_update_obs(self, state: GameState):
        if state == self.current_state:  # No need to update
            return

        if self.boost_timers is None:
            self.reset(state)
        else:
            self.current_state = state

        qkv = np.zeros((1, 1 + self.n_players + len(state.boost_pads), 24))  # Ball, players, boosts

        # Add ball
        n = 0
        ball = state.ball
        qkv[0, 0, 3] = 1  # is_ball
        qkv[0, 0, 5:8] = ball.position
        qkv[0, 0, 8:11] = ball.linear_velocity
        qkv[0, 0, 17:20] = ball.angular_velocity

        # Add players
        n += 1
        demos = np.zeros(self.n_players)  # Which players are currently demoed
        for player in state.players:
            if player.team_num == BLUE_TEAM:
                qkv[0, n, 1] = 1  # is_teammate
            else:
                qkv[0, n, 2] = 1  # is_opponent
            car_data = player.car_data
            qkv[0, n, 5:8] = car_data.position
            qkv[0, n, 8:11] = car_data.linear_velocity
            qkv[0, n, 11:14] = car_data.forward()
            qkv[0, n, 14:17] = car_data.up()
            qkv[0, n, 17:20] = car_data.angular_velocity
            qkv[0, n, 20] = player.boost_amount
            #             qkv[0, n, 21] = player.is_demoed
            demos[n - 1] = player.is_demoed  # Keep track for demo timer
            qkv[0, n, 22] = player.on_ground
            qkv[0, n, 23] = player.has_flip
            n += 1

        # Add boost pads
        n = 1 + self.n_players
        boost_pads = state.boost_pads
        qkv[0, n:, 4] = 1  # is_boost
        qkv[0, n:, 5:8] = self._boost_locations
        qkv[0, n:, 20] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)  # Boost amount
        #         qkv[0, n:, 21] = boost_pads

        # Boost and demo timers
        new_boost_grabs = (boost_pads == 1) & (self.boost_timers == 0)  # New boost grabs since last frame
        self.boost_timers[new_boost_grabs] = 0.4 + 0.6 * (self._boost_locations[new_boost_grabs, 2] > 72)
        self.boost_timers *= boost_pads  # Make sure we have zeros right
        qkv[0, 1 + self.n_players:, 21] = self.boost_timers
        self.boost_timers -= self.tick_skip / 1200  # Pre-normalized, 120 fps for 10 seconds
        self.boost_timers[self.boost_timers < 0] = 0

        new_demos = (demos == 1) & (self.demo_timers == 0)
        self.demo_timers[new_demos] = 0.3
        self.demo_timers *= demos
        qkv[0, 1: 1 + self.n_players, 21] = self.demo_timers
        self.demo_timers -= self.tick_skip / 1200
        self.demo_timers[self.demo_timers < 0] = 0

        # Store results
        self.current_qkv = qkv / self._norm
        mask = np.zeros((1, qkv.shape[1]))
        mask[0, 1 + len(state.players):1 + self.n_players] = 1
        self.current_mask = mask

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> Any:
        if self.boost_timers is None:
            return np.zeros(0)  # Obs space autodetect, make Aech happy
        self._maybe_update_obs(state)
        invert = player.team_num == ORANGE_TEAM

        qkv = self.current_qkv.copy()
        mask = self.current_mask.copy()

        main_n = state.players.index(player) + 1
        qkv[0, main_n, 0] = 1  # is_main
        if invert:
            qkv[0, :, (1, 2)] = qkv[0, :, (2, 1)]  # Swap blue/orange
            qkv *= self._invert  # Negate x and y values

        # TODO left-right normalization (always pick one side)

        q = qkv[0, main_n, :]
        q = np.expand_dims(np.concatenate((q, previous_action), axis=0), axis=(0, 1))
        # kv = np.delete(qkv, main_n, axis=0)  # Delete main? Watch masking
        kv = qkv

        # With EARLPerceiver we can use relative coords+vel(+more?) for key/value tensor, might be smart
        kv[0, :, 5:11] -= q[0, 0, 5:11]
        return q, kv, mask


IS_SELF, IS_MATE, IS_OPP, IS_BALL, IS_BOOST = range(5)
POS = slice(5, 8)
LIN_VEL = slice(8, 11)
FW = slice(11, 14)
UP = slice(14, 17)
ANG_VEL = slice(17, 20)
BOOST, DEMO, ON_GROUND, HAS_FLIP = range(20, 24)
ACTIONS = range(24, 32)


class NectoObsTEST(BatchedObsBuilder):
    _boost_locations = np.array(BOOST_LOCATIONS)
    _invert = np.array([1] * 5 + [-1, -1, 1] * 5 + [1] * 4)
    _norm = np.array([1.] * 5 + [2300] * 6 + [1] * 6 + [5.5] * 3 + [1] * 4)

    def __init__(self, n_players=None, tick_skip=8):
        super().__init__()
        self.n_players = n_players
        self.demo_timers = None
        self.boost_timers = None
        self.current_state = None
        self.current_qkv = None
        self.current_mask = None
        self.tick_skip = tick_skip

    def _reset(self, initial_state: GameState):
        self.demo_timers = np.zeros(len(initial_state.players))
        self.boost_timers = np.zeros(len(initial_state.boost_pads))
        # self.current_state = initial_state

    #     def encode_gamestate(state: GameState):
    #     state_vals = [0, state.blue_score, state.orange_score]
    #     state_vals += state.boost_pads.tolist()
    #
    #     for bd in (state.ball, state.inverted_ball):
    #         state_vals += bd.position.tolist()
    #         state_vals += bd.linear_velocity.tolist()
    #         state_vals += bd.angular_velocity.tolist()
    #
    #     for p in state.players:
    #         state_vals += [p.car_id, p.team_num]
    #         for cd in (p.car_data, p.inverted_car_data):
    #             state_vals += cd.position.tolist()
    #             state_vals += cd.quaternion.tolist()
    #             state_vals += cd.linear_velocity.tolist()
    #             state_vals += cd.angular_velocity.tolist()
    #         state_vals += [
    #             p.match_goals,
    #             p.match_saves,
    #             p.match_shots,
    #             p.match_demolishes,
    #             p.boost_pickups,
    #             p.is_demoed,
    #             p.on_ground,
    #             p.ball_touched,
    #             p.has_flip,
    #             p.boost_amount
    #         ]
    #     return state_vals

    def batched_build_obs(self, encoded_states: np.ndarray):
        n_players = (encoded_states.shape[1] - 46) // 25
        lim_players = n_players if self.n_players is None else self.n_players
        n_entities = lim_players + 1 + 34

        # SELECTORS
        sel_players = slice(0, lim_players)
        sel_ball = sel_players.stop
        sel_boosts = slice(sel_ball + 1, None)

        # MAIN ARRAYS
        q = np.zeros((n_players, encoded_states.shape[0], 1, 32))
        kv = np.zeros((n_players, encoded_states.shape[0], n_entities, 24))  # Keys and values are (mostly) shared
        m = np.zeros((n_players, encoded_states.shape[0], n_entities))  # Mask is shared

        # PLAYERS
        for i in range(n_players):
            kv[i, :, i, IS_SELF] = 1
            encoded_player = encoded_states[:, 46 + i * 25: 46 + (i + 1) * 25]
            kv[:, :, i, POS] = encoded_player[:, 2: 5]
            kv[:, :, i, LIN_VEL] = encoded_player[:, 5: 8]
            quat = encoded_player[:, 8, 12]
            # from rlgym.utils.math import quat_to_rot_mtx
            # kv[:, i, FW] = encoded_states[:, ei: ei + 3]  # FIXME convert to forward and up
            # kv[:, i, UP] = encoded_states[:, ei: ei + 3]
            kv[:, :, i, ANG_VEL] = encoded_player[:, 12: 15]
            kv[:, :, i, BOOST] = encoded_player[:, 24]
            kv[:, :, i, DEMO] = encoded_player[:, 20]  # FIXME demo timer
            kv[:, :, i, ON_GROUND] = encoded_player[:, 21]
            kv[:, :, i, HAS_FLIP] = encoded_player[:, 23]
            q[i, :, 0, :] = kv[i, :, i, :]

        # BALL
        kv[:, :, sel_ball, 3] = 1
        kv[:, :, sel_ball, np.r_[POS, LIN_VEL, ANG_VEL]] = encoded_states[:, 37:46]

        # BOOSTS
        kv[:, :, sel_boosts, IS_BOOST] = 1
        kv[:, :, sel_boosts, POS] = self._boost_locations
        kv[:, :, sel_boosts, BOOST] = 0.12 + 0.88 * (self._boost_locations[:, 2] > 72)
        kv[:, :, sel_boosts, DEMO] = encoded_states[:, 3:3 + 34]  # FIXME boost timer

        # MASK
        m[:, :, n_players, lim_players] = 1

        # FIXME normalize

        return [(q[i], kv[i], m[i]) for i in range(n_players)]

    def add_action(self, previous_actions: np.ndarray, player_index=None):
        if player_index is None:
            for (q, kv, m), act in zip(self.current_obs, previous_actions):
                q[:, 0, ACTIONS] = act
        else:
            q, kv, m = self.current_obs[player_index]
            q[:, 0, ACTIONS] = previous_actions
