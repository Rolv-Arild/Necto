import os
import random

import numpy as np
from redis import Redis
from rlgym.utils import StateSetter
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED
from rlgym.utils.gamestates import GameState
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters import DefaultState, StateWrapper

from rlgym_tools.extra_state_setters.goalie_state import GoaliePracticeState
from rlgym_tools.extra_state_setters.hoops_setter import HoopsLikeSetter
from rlgym_tools.extra_state_setters.replay_setter import ReplaySetter
from rlgym_tools.extra_state_setters.symmetric_setter import KickoffLikeSetter
from rlgym_tools.extra_state_setters.wall_state import WallPracticeState

from rocket_learn.rollout_generator.redis.utils import _unserialize

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi

GOAL_X_MAX = 800.0
GOAL_X_MIN = -800.0

PLACEMENT_BOX_X = 5000
PLACEMENT_BOX_Y = 2000
PLACEMENT_BOX_Y_OFFSET = 3000

GOAL_LINE = 5100

YAW_MAX = np.pi


class BetterRandom(StateSetter):  # Random state with some triangular distributions
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99.9% chance of below ball max speed
        ball_speed = np.random.exponential(-BALL_MAX_SPEED / np.log(1 - 0.999))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            # On average 1 second at max speed away from ball
            ball_dist = np.random.exponential(BALL_MAX_SPEED)
            ball_car = rand_vec3(ball_dist)
            car_pos = state_wrapper.ball.position + ball_car
            if abs(car_pos[0]) < LIM_X \
                    and abs(car_pos[1]) < LIM_Y \
                    and 0 < car_pos[2] < LIM_Z:
                car.set_pos(*car_pos)
            else:  # Fallback on fully random
                car.set_pos(
                    x=np.random.uniform(-LIM_X, LIM_X),
                    y=np.random.uniform(-LIM_Y, LIM_Y),
                    z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
                )

            vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_SPEED))
            car.set_lin_vel(*vel)

            car.set_rot(
                pitch=np.random.triangular(-PITCH_LIM, 0, PITCH_LIM),
                yaw=np.random.uniform(-YAW_LIM, YAW_LIM),
                roll=np.random.triangular(-ROLL_LIM, 0, ROLL_LIM),
            )

            ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL))
            car.set_ang_vel(*ang_vel)
            car.boost = np.random.uniform(0, 1)


class NectoReplaySetter(ReplaySetter):
    def generate_probabilities(self):
        ball_heights = self.states[:, 2]
        player_heights = self.states[:, 9 + 2::13]
        weights = 1 + 10 * (ball_heights + player_heights.sum(axis=-1)) / CEILING_Z
        return weights / weights.sum()


class NectoStateSetter(StateSetter):
    def __init__(
            self, redis: Redis, *,
            replay_prob=0.7,
            random_prob=0.08,
            kickoff_prob=0.04,
            kickofflike_prob=0.04,
            goalie_prob=0.05,
            hoops_prob=0.04,
            wall_prob=0.05
    ):  # add goalie_prob/shooting/dribbling?
        super().__init__()
        self.redis = redis
        self.replay_setters = [
            NectoReplaySetter(replay_array)
            for replay_array in _unserialize(redis.get("replay-arrays"))
        ]
        self.setters = [
            BetterRandom(),
            DefaultState(),
            KickoffLikeSetter(),
            GoaliePracticeState(first_defender_in_goal=True, allow_enemy_interference=True),
            HoopsLikeSetter(),
            WallPracticeState()
        ]
        self.probs = np.array(
            [replay_prob, random_prob, kickoff_prob, kickofflike_prob, goalie_prob, hoops_prob, wall_prob])
        assert self.probs.sum() == 1, "Probabilities must sum to 1"

    # def build_wrapper(self, max_team_size: int, spawn_opponents: bool) -> StateWrapper:
    #     assert max_team_size >= 3, "Env has to support 3 players per team"
    #     assert spawn_opponents, "Env has to spawn opponents"
    #     gamemode_counts = self.redis.hgetall(EXPERIENCE_COUNTER_KEY)
    #     mode = min(gamemode_counts, key=gamemode_counts.get)
    #     team_size = int(mode[0])
    #     return StateWrapper(blue_count=team_size, orange_count=team_size)

    def reset(self, state_wrapper: StateWrapper):
        # counts = self.redis.hgetall(EXPERIENCE_COUNTER_KEY)
        # gamemode = int(min(counts, key=counts.get)[:1])
        # # FIXME: Generate state wrapper from gamemode
        i = np.random.choice(1 + len(self.setters), p=self.probs)
        if i == 0:
            self.replay_setters[len(state_wrapper.cars) // 2 - 1].reset(state_wrapper)
        else:
            self.setters[i - 1].reset(state_wrapper)
        for car in state_wrapper.cars:  # In case of 0 boost consumption rate we want it to be able to boost
            car.boost = max(car.boost, 0.01)
