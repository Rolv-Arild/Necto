import numpy as np
from numpy import exp
from numpy.linalg import norm
from rlgym.utils import RewardFunction
from rlgym.utils.common_values import CEILING_Z, BALL_MAX_SPEED, CAR_MAX_SPEED, BLUE_GOAL_BACK, \
    BLUE_GOAL_CENTER, ORANGE_GOAL_BACK, ORANGE_GOAL_CENTER, BALL_RADIUS, ORANGE_TEAM, GOAL_HEIGHT, CAR_MAX_ANG_VEL
from rlgym.utils.gamestates import GameState, PlayerData
from rlgym.utils.math import cosine_similarity

from rocket_learn.utils.scoreboard import win_prob


class NectoRewardFunction(RewardFunction):
    BLUE_GOAL = (np.array(BLUE_GOAL_BACK) + np.array(BLUE_GOAL_CENTER)) / 2
    ORANGE_GOAL = (np.array(ORANGE_GOAL_BACK) + np.array(ORANGE_GOAL_CENTER)) / 2

    def __init__(
            self,
            team_spirit=0.6,
            goal_w=10,
            win_prob_w=10,
            goal_dist_w=10,
            goal_speed_bonus_w=2.5,
            goal_dist_bonus_w=2.5,
            demo_w=8,
            dist_w=0.25,
            align_w=0.25,
            boost_gain_w=1.5,
            boost_lose_w=0.8,
            ang_vel_w=0.005,
            touch_grass_w=0.005,
            touch_height_w=3,
            touch_accel_w=0.5,
            flip_reset_w=10,
            opponent_punish_w=1
    ):
        self.team_spirit = team_spirit
        self.current_state = None
        self.last_state = None
        self.n = 0
        self.goal_w = goal_w
        self.win_prob_w = win_prob_w
        self.goal_dist_w = goal_dist_w
        self.goal_speed_bonus_w = goal_speed_bonus_w
        self.goal_dist_bonus_w = goal_dist_bonus_w
        self.demo_w = demo_w
        self.dist_w = dist_w
        self.align_w = align_w
        self.boost_gain_w = boost_gain_w
        self.boost_lose_w = boost_lose_w
        self.ang_vel_w = ang_vel_w
        self.touch_grass_w = touch_grass_w
        self.touch_height_w = touch_height_w
        self.touch_accel_w = touch_accel_w
        self.flip_reset_w = flip_reset_w
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

        blue, orange, ticks_left = state.inverted_ball.angular_velocity
        diff = blue - orange
        prob = win_prob(len(state.players) // 2,
                        [ticks_left / 120],
                        np.clip([diff], -5, 5))[0]
        if np.isinf(ticks_left):  # Goal scored at 0 seconds / in overtime
            if diff > 0:
                prob = 1
            elif diff < 0:
                prob = 0
            else:
                prob = 0.5
        state_quality += self.win_prob_w * prob

        # Half state quality because it is applied to both teams, thus doubling it in the reward distributing
        return state_quality / 2, player_qualities

    @staticmethod
    def _height_activation(z):
        return np.cbrt((float(z) - 150) / CEILING_Z)  # 150 is approximate dribble height

    @staticmethod
    def dist_to_closest_wall(x, y):
        dist_side_wall = abs(4096 - abs(x))
        dist_back_wall = abs(5120 - abs(y))

        # From https://stackoverflow.com/questions/849211/shortest-distance-between-a-point-and-a-line-segment
        x1, y1, x2, y2 = 4096 - 1152, 5120, 4096, 5120 - 1152  # Line segment for corner
        A = abs(x) - x1
        B = abs(y) - y1
        C = x2 - x1
        D = y2 - y1

        dot = A * C + B * D
        len_sq = C * C + D * D
        param = -1
        if len_sq != 0:  # in case of 0 length line
            param = dot / len_sq

        if param < 0:
            xx = x1
            yy = y1
        elif param > 1:
            xx = x2
            yy = y2
        else:
            xx = x1 + param * C
            yy = y1 + param * D

        dx = abs(x) - xx
        dy = abs(y) - yy
        dist_corner_wall = np.sqrt(dx * dx + dy * dy)

        return min(dist_side_wall, dist_back_wall, dist_corner_wall)

    def pre_step(self, state: GameState):
        # Calculate rewards, positive for blue, negative for orange
        if state != self.current_state:
            self.last_state = self.current_state
            self.current_state = state
            self.n = 0
        state_quality, player_qualities = self._state_qualities(state)
        player_rewards = np.zeros_like(player_qualities)
        ball_height = state.ball.position[2]

        for i, player in enumerate(state.players):
            last = self.last_state.players[i]

            car_height = player.car_data.position[2]

            if player.ball_touched:
                curr_vel = self.current_state.ball.linear_velocity
                last_vel = self.last_state.ball.linear_velocity

                # On ground it gets about 0.04 just for touching, as well as some extra for the speed it produces
                # Ball is pretty close to z=150 when on top of car, so 1 second of dribbling is 1 reward
                # Close to 20 in the limit with ball on top, but opponents should learn to challenge way before that
                avg_height = 0.5 * (car_height + ball_height)
                h0 = self._height_activation(0)
                h1 = self._height_activation(CEILING_Z)
                hx = self._height_activation(avg_height)
                height_factor = ((hx - h0) / (h1 - h0)) ** 2
                wall_dist_factor = 1 - np.exp(-self.dist_to_closest_wall(*player.car_data.position[:2]) / CAR_MAX_SPEED)
                player_rewards[i] += self.touch_height_w * height_factor * (1 + wall_dist_factor)
                if player.has_flip and not last.has_flip \
                        and player.car_data.position[2] > 3 * BALL_RADIUS \
                        and np.linalg.norm(state.ball.position - player.car_data.position) < 2 * BALL_RADIUS \
                        and cosine_similarity(state.ball.position - player.car_data.position,
                                              -player.car_data.up()) > 0.9:
                    player_rewards[i] += self.flip_reset_w

                # Changing speed of ball from standing still to supersonic (~83kph) is 1 reward
                player_rewards[i] += self.touch_accel_w * (1 - height_factor) * norm(
                    curr_vel - last_vel) / CAR_MAX_SPEED

            # Encourage collecting and saving boost, sqrt to weight boost more the less it has
            boost_diff = (np.sqrt(np.clip(player.boost_amount, 0, 1))
                          - np.sqrt(np.clip(last.boost_amount, 0, 1)))
            if boost_diff >= 0:
                player_rewards[i] += self.boost_gain_w * boost_diff
            elif car_height < GOAL_HEIGHT:
                player_rewards[i] += self.boost_lose_w * boost_diff * (1 - car_height / GOAL_HEIGHT)

            # Encourage spinning (slightly), helps it not stop flipping at the start of training
            # and (hopefully) explore rotating in the air
            ang_vel_norm = np.linalg.norm(player.car_data.angular_velocity) / CAR_MAX_ANG_VEL
            player_rewards[i] += ang_vel_norm * self.ang_vel_w

            if player.on_ground and car_height < BALL_RADIUS:
                player_rewards[i] -= self.touch_grass_w

            # Divide demo reward equally between demoer (positive) and demoee (negative)
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
        blue, orange, ticks_left = state.inverted_ball.angular_velocity
        if d_blue > 0 or d_orange > 0 \
                or ticks_left < 0 and np.isinf(ticks_left):
            home = slice(None, mid)
            away = slice(mid, None)
            new_diff = blue - orange
            d_home = d_blue
            if d_orange > 0:
                home, away = away, home
                new_diff *= -1
                d_home = d_orange

            goal_speed = d_home * norm(self.last_state.ball.linear_velocity)
            distances = d_home * norm(
                np.stack([p.car_data.position for p in state.players[away]])
                - self.last_state.ball.position,
                axis=-1
            )

            # TODO Want to find something better, this could promote waiting to score when losing
            # print(f"{importance_reward=}, {old_prob=}, {new_prob=}, {ticks_left=}, "
            #       f"{old_diff=}, {new_diff=}, {blue=}, {orange=}")
            # assert new_prob >= old_prob and (old_prob >= (0.5 - 1e-10) or new_prob <= (0.5 + 1e-10))
            player_rewards[away] -= self.goal_dist_bonus_w * (1 - exp(-distances / CAR_MAX_SPEED))
            player_rewards[home] += (self.goal_w * d_home
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
        rew = self.rewards[self.n]
        self.n += 1
        return float(rew)  # / 3.2  # Divide to get std of expected reward to ~1 at start, helps value net a little
