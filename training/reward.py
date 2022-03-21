import numpy as np
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CEILING_Z, BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_TEAM, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BALL_RADIUS, ORANGE_TEAM
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import cosine_similarity
from numpy import exp
from numpy.linalg import norm


class NectoRewardFunction(RewardFunction):
    BLUE_GOAL = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
    ORANGE_GOAL = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2

    def __init__(
            self,
            team_spirit=0.5,  # 0.3 -> 0.5
            goal_w=10,
            goal_dist_w=10,
            goal_speed_bonus_w=2.5,
            goal_dist_bonus_w=2.5,
            demo_w=5,
            dist_w=0.5,  # 0.75 -> 0.5
            align_w=0.5,
            boost_gain_w=1,
            boost_lose_w=0.5,
            touch_grass_w=0.005,
            touch_height_w=1,  # 1 -> ~1.1 (changed normalization factor)
            touch_accel_w=0.25,
            opponent_punish_w=1
    ):
        self.team_spirit = team_spirit
        self.current_state = None
        self.last_state = None
        self.n = 0
        self.goal_w = goal_w
        self.goal_dist_w = goal_dist_w
        self.goal_speed_bonus_w = goal_speed_bonus_w
        self.goal_dist_bonus_w = goal_dist_bonus_w
        self.demo_w = demo_w
        self.dist_w = dist_w
        self.align_w = align_w
        self.boost_gain_w = boost_gain_w
        self.boost_lose_w = boost_lose_w
        self.touch_grass_w = touch_grass_w
        self.touch_height_w = touch_height_w
        self.touch_accel_w = touch_accel_w
        self.opponent_punish_w = opponent_punish_w
        self.state_quality = None
        self.player_qualities = None
        self.rewards = None

    def _state_qualities(self, state: GameState):
        ball_pos = state.ball.position

        state_quality = 0.5 * self.goal_dist_w * (exp(-norm(self.ORANGE_GOAL - ball_pos) / CAR_MAX_SPEED)
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
            player_qualities[i] = (self.dist_w * liu_dist + self.align_w * alignment)

            # TODO use only dist of closest player for entire team?

        # Half state quality because it is applied to both teams, thus doubling it in the reward distributing
        return state_quality / 2, player_qualities

    def _calculate_rewards(self, state: GameState):
        # Calculate rewards, positive for blue, negative for orange
        state_quality, player_qualities = self._state_qualities(state)
        player_rewards = np.zeros_like(player_qualities)

        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            if player.ball_touched:
                curr_vel = self.current_state.ball.linear_velocity
                last_vel = self.last_state.ball.linear_velocity

                # On ground it gets about 0.04 just for touching, as well as some extra for the speed it produces
                # Ball is pretty close to z=150 when on top of car, so 1 second of dribbling is 1 reward
                # Close to 20 in the limit with ball on top, but opponents should learn to challenge way before that
                player_rewards[i] += self.touch_height_w * state.ball.position[2] / CEILING_Z

                # Changing speed of ball from standing still to supersonic (~83kph) is 1 reward
                player_rewards[i] += self.touch_accel_w * norm(curr_vel - last_vel) / CAR_MAX_SPEED

            # Encourage collecting and saving boost, sqrt to weight boost more the less it has
            boost_diff = np.sqrt(player.boost_amount) - np.sqrt(last.boost_amount)
            if boost_diff >= 0:
                player_rewards[i] += self.boost_gain_w * boost_diff
            else:
                player_rewards[i] += self.boost_lose_w * boost_diff

            # Encourage being in the air (slightly)
            player_rewards[i] -= player.on_ground * self.touch_grass_w

            if player.is_demoed and not last.is_demoed:
                player_rewards[i] -= self.demo_w / 2
            if player.match_demolishes > last.match_demolishes:
                player_rewards[i] += self.demo_w / 2

        mid = len(player_rewards) // 2

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
            goal_speed = norm(self.last_state.ball.linear_velocity)
            distances = norm(
                np.stack([p.car_data.position for p in state.players[mid:]])
                - self.last_state.ball.position,
                axis=-1
            )
            player_rewards[mid:] = -self.goal_dist_bonus_w * (1 - exp(-distances / CAR_MAX_SPEED))
            player_rewards[:mid] = (self.goal_w * d_blue
                                    + self.goal_dist_bonus_w * goal_speed / BALL_MAX_SPEED)
        if d_orange > 0:
            goal_speed = norm(self.last_state.ball.linear_velocity)
            distances = norm(
                np.stack([p.car_data.position for p in state.players[:mid]])
                - self.last_state.ball.position,
                axis=-1
            )
            player_rewards[:mid] = -self.goal_dist_bonus_w * (1 - exp(-distances / CAR_MAX_SPEED))
            player_rewards[mid:] = (self.goal_w * d_orange
                                    + self.goal_dist_bonus_w * goal_speed / BALL_MAX_SPEED)

        blue = player_rewards[:mid]
        orange = player_rewards[mid:]
        bm = np.nan_to_num(blue.mean())
        om = np.nan_to_num(orange.mean())

        player_rewards[:mid] = ((1 - self.team_spirit) * blue + self.team_spirit * bm
                                - self.opponent_punish_w * om)
        player_rewards[mid:] = ((1 - self.team_spirit) * orange + self.team_spirit * om
                                - self.opponent_punish_w * bm)

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
        return float(rew)  # / 3.2  # Divide to get std of expected reward to ~1 at start, helps value net a little
