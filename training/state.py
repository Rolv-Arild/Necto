import random

import numpy as np
from rlgym.utils import StateSetter
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters import DefaultState, StateWrapper

LIM_X = SIDE_WALL_X - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Y = BACK_WALL_Y - 1152 / 2 - BALL_RADIUS * 2 ** 0.5
LIM_Z = CEILING_Z - BALL_RADIUS

PITCH_LIM = np.pi / 2
YAW_LIM = np.pi
ROLL_LIM = np.pi


class BetterRandom(StateSetter):  # Random state with some triangular distributions
    def __init__(self):
        super().__init__()

    def reset(self, state_wrapper: StateWrapper):
        state_wrapper.ball.set_pos(
            x=np.random.uniform(-LIM_X, LIM_X),
            y=np.random.uniform(-LIM_Y, LIM_Y),
            z=np.random.triangular(BALL_RADIUS, BALL_RADIUS, LIM_Z),
        )

        # 99% chance of below car max speed
        ball_speed = np.random.exponential(-CAR_MAX_SPEED / np.log(1 - 0.99))
        vel = rand_vec3(min(ball_speed, BALL_MAX_SPEED))
        state_wrapper.ball.set_lin_vel(*vel)

        ang_vel = rand_vec3(np.random.triangular(0, 0, CAR_MAX_ANG_VEL + 0.5))
        state_wrapper.ball.set_ang_vel(*ang_vel)

        for car in state_wrapper.cars:
            for _ in range(10):  # 10 retries
                ball_dist = np.random.exponential(2300)
                ball_car = rand_vec3(ball_dist)
                car_pos = state_wrapper.ball.position + ball_car
                if abs(car_pos[0]) < LIM_X \
                        and abs(car_pos[1]) < LIM_Y \
                        and 0 < car_pos[2] < LIM_Z:
                    car.set_pos(*car_pos)
                    break
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


# class KickoffLike(StateSetter): TODO
#     def reset(self, state_wrapper: StateWrapper):
#         state_wrapper.ball.set_pos(
#             x=np.random.uniform(-LIM_X, LIM_X)
#         )


class NectoStateSetter(StateSetter):
    def __init__(self, kickoff_prob=0.01):
        super().__init__()
        # TODO sample from SSL replays, kickoff-like
        self.kickoff_prob = kickoff_prob
        self.default = DefaultState()
        self.random = BetterRandom()

    def reset(self, state_wrapper: StateWrapper):
        if np.random.random() < self.kickoff_prob:
            self.default.reset(state_wrapper)
        else:
            self.random.reset(state_wrapper)
