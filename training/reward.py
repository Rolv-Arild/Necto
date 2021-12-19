import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CEILING_Z, BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_TEAM, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BALL_RADIUS, ORANGE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import cosine_similarity
from numpy import exp
from numpy.linalg import norm


# class NectoRewardFunction(RewardFunction):
#     def __init__(
#             self,
#             team_spirit=0.3,
#             goal_w=10,
#             shot_w=5,
#             save_w=5,
#             demo_w=5,
#             boost_w=0.5,
#             touch_height_w=0.5,
#             touch_accel_w=1,
#             car_accel_w=0.03,
#             cb_accel_w=0.05
#     ):
#         self.team_spirit = team_spirit
#         self.last_state = None
#         self.current_state = None
#         self.rewards = None
#         self.blue_rewards = None
#         self.orange_rewards = None
#         self.n = 0
#         self.goal_w = goal_w
#         self.shot_w = shot_w
#         self.save_w = save_w
#         self.demo_w = demo_w
#         self.boost_w = boost_w
#         self.touch_height_w = touch_height_w
#         self.touch_accel_w = touch_accel_w
#         self.car_accel_w = car_accel_w
#         self.cb_accel_w = cb_accel_w
#
#     def reset(self, initial_state: GameState):
#         self.last_state = None
#         self.current_state = initial_state
#         self.rewards = np.zeros(len(initial_state.players))
#
#     def _maybe_update_rewards(self, state: GameState):
#         if state == self.current_state:
#             return
#         self.n = 0
#         self.last_state = self.current_state
#         self.current_state = state
#         rewards = np.zeros(len(state.players))
#         blue_mask = np.zeros_like(rewards, dtype=bool)
#         orange_mask = np.zeros_like(rewards, dtype=bool)
#         i = 0
#
#         d_blue = state.blue_score - self.last_state.blue_score
#         d_orange = state.orange_score - self.last_state.orange_score
#
#         for old_p, new_p in zip(self.last_state.players, self.current_state.players):
#             assert old_p.car_id == new_p.car_id
#             # d_goal = new_p.match_goals - old_p.match_goals
#             rew = (  # self.goal_w * d_goal +
#                     self.shot_w * (new_p.match_shots - old_p.match_shots) +
#                     self.save_w * (new_p.match_saves - old_p.match_saves) +
#                     self.demo_w * (new_p.match_demolishes - old_p.match_demolishes) +
#                     self.boost_w * max(new_p.boost_amount - old_p.boost_amount, 0))
#             # Some napkin math: going around edge of field picking up 100 boost every second and gamma 0.995, skip 8
#             # Discounted future reward in limit would be (0.5 / (1 * 15)) / (1 - 0.995) = 6.67 as a generous estimate
#             # Pros are generally around maybe 400 bcpm, which would be 0.44 limit
#             if new_p.ball_touched:
#                 # target = np.array(ORANGE_GOAL_BACK if new_p.team_num == BLUE_TEAM else BLUE_GOAL_BACK)
#                 curr_vel = self.current_state.ball.linear_velocity
#                 last_vel = self.last_state.ball.linear_velocity
#                 # On ground it gets about 0.05 just for touching, as well as some extra for the speed it produces
#                 # Close to 20 in the limit with ball on top, but opponents should learn to challenge way before that
#                 rew += (self.touch_height_w * state.ball.position[2] / CEILING_Z +
#                         self.touch_accel_w * norm(curr_vel - last_vel) / BALL_MAX_SPEED)
#
#             diff_abs_vel = (norm(new_p.car_data.linear_velocity)
#                             - norm(old_p.car_data.linear_velocity))
#             diff_vel = (new_p.car_data.linear_velocity
#                         - old_p.car_data.linear_velocity)
#             ball_dir = self.current_state.ball.position - new_p.car_data.position
#             ball_dir = ball_dir / norm(ball_dir)
#             accel_ball = np.dot(diff_vel, ball_dir)
#
#             rew += (self.car_accel_w * diff_abs_vel / CAR_MAX_SPEED +
#                     self.cb_accel_w * accel_ball / CAR_MAX_SPEED)
#
#             rewards[i] = rew
#             if new_p.team_num == BLUE_TEAM:
#                 blue_mask[i] = True
#                 # d_blue -= d_goal
#             else:
#                 orange_mask[i] = True
#                 # d_orange -= d_goal
#             i += 1
#
#         # Handle goals with no scorer for critic consistency,
#         # random state could send ball straight into goal
#         if d_blue > 0:
#             rewards[blue_mask] = d_blue * self.goal_w
#         if d_orange > 0:
#             rewards[orange_mask] = d_orange * self.goal_w
#
#         blue_rewards = rewards[blue_mask]
#         orange_rewards = rewards[orange_mask]
#         blue_mean = np.nan_to_num(blue_rewards.mean())
#         orange_mean = np.nan_to_num(orange_rewards.mean())
#         self.rewards = np.zeros_like(rewards)
#         self.rewards[blue_mask] = (1 - self.team_spirit) * blue_rewards + self.team_spirit * blue_mean - orange_mean
#         self.rewards[orange_mask] = (1 - self.team_spirit) * orange_rewards + self.team_spirit * orange_mean - blue_mean
#
#     def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#         self._maybe_update_rewards(state)
#         rew = self.rewards[self.n]
#         self.n += 1
#         return rew / 1.6  # Approximate std at init is 1.6, helps the critic a little


class NectoRewardFunction(RewardFunction):
    BLUE_GOAL = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
    ORANGE_GOAL = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2

    def __init__(
            self,
            team_spirit=0.3,
            goal_w=10,
            demo_w=5,
            dist_w=1,
            align_w=0.5,
            boost_w=0.5,
            spacing_w=5,
            touch_height_w=0.5,
            touch_accel_w=1,
    ):
        self.team_spirit = team_spirit
        self.current_state = None
        self.last_state = None
        self.n = 0
        self.goal_w = goal_w
        self.demo_w = demo_w
        self.dist_w = dist_w
        self.align_w = align_w
        self.boost_w = boost_w
        self.spacing_w = spacing_w
        self.touch_height_w = touch_height_w
        self.touch_accel_w = touch_accel_w
        self.state_quality = None
        self.player_qualities = None
        self.rewards = None

    def _state_qualities(self, state: GameState):
        ball_pos = state.ball.position
        state_quality = self.goal_w * (exp(-norm(self.ORANGE_GOAL - ball_pos) / CAR_MAX_SPEED)
                                       - exp(-norm(self.BLUE_GOAL - ball_pos) / CAR_MAX_SPEED))
        player_qualities = np.zeros(len(state.players))
        for i, player in enumerate(state.players):
            pos = player.car_data.position

            # Align player->ball and player->net vectors
            alignment = 0.5 * (cosine_similarity(ball_pos - pos, ORANGE_GOAL_BACK - pos)
                               - cosine_similarity(ball_pos - pos, BLUE_GOAL_BACK - pos))
            if player.team_num == ORANGE_TEAM:
                alignment *= -1
            liu_dist = exp(-norm(ball_pos - pos) / 1410)  # Max driving speed
            player_qualities[i] = (self.dist_w * liu_dist + self.align_w * alignment
                                   + self.boost_w * player.boost_amount)
            # for j in range(i + 1, len(state.players)):
            #     mate = state.players[j]
            #     if mate.team_num == player.team_num:
            #         # Don't get too close to teammates
            #         player_qualities[[i, j]] -= self.spacing_w * exp(-norm(pos - mate.car_data.position)
            #                                                          / (2 * BALL_RADIUS))

        return state_quality, player_qualities

    def _calculate_rewards(self, state: GameState):
        # Calculate rewards, positive for blue, negative for orange
        state_quality, player_qualities = self._state_qualities(state)
        player_rewards = np.zeros_like(player_qualities)

        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            if player.ball_touched:
                curr_vel = self.current_state.ball.linear_velocity
                last_vel = self.last_state.ball.linear_velocity
                # On ground it gets about 0.05 just for touching, as well as some extra for the speed it produces
                # Close to 20 in the limit with ball on top, but opponents should learn to challenge way before that
                player_rewards[i] += (self.touch_height_w * state.ball.position[2] / CEILING_Z +
                                      self.touch_accel_w * norm(curr_vel - last_vel) / BALL_MAX_SPEED)

            if player.is_demoed and not last.is_demoed:
                player_rewards[i] -= self.demo_w / 2
            if player.match_demolishes > last.match_demolishes:
                player_rewards[i] += self.demo_w / 2

        mid = len(player_rewards) // 2
        blue = player_rewards[:mid]
        orange = player_rewards[mid:]
        bm = np.nan_to_num(blue.mean())
        om = np.nan_to_num(orange.mean())

        player_rewards += player_qualities - self.player_qualities
        player_rewards[:mid] += state_quality - self.state_quality
        player_rewards[mid:] -= state_quality - self.state_quality

        self.player_qualities = player_qualities
        self.state_quality = state_quality

        # Handle goals with no scorer for critic consistency,
        # random state could send ball straight into goal
        d_blue = state.blue_score - self.last_state.blue_score
        d_orange = state.orange_score - self.last_state.orange_score
        if d_blue > 0:
            player_rewards[:mid] = d_blue * self.goal_w
        if d_orange > 0:
            player_rewards[mid:] = d_orange * self.goal_w

        player_rewards[:mid] = (1 - self.team_spirit) * blue + self.team_spirit * bm - om
        player_rewards[mid:] = (1 - self.team_spirit) * orange + self.team_spirit * om - bm

        self.last_state = state
        self.rewards = player_rewards

    def reset(self, initial_state: GameState):
        self.n = 0
        self.last_state = None
        self.rewards = None
        self.current_state = initial_state
        self.state_quality, self.player_qualities = self._state_qualities(initial_state)

    def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self._calculate_rewards(state)
            self.n = 0
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)
