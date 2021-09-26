import math
import numpy as np
from rlgym_compat import common_values
from rlgym_compat import PlayerData, GameState


class DefaultObs:
    POS_STD = 2300
    ANG_STD = math.pi

    def __init__(self):
        super().__init__()

    def reset(self, initial_state: GameState):
        pass

    def build_obs(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> np.ndarray:
        players = state.players
        if player.team_num == common_values.ORANGE_TEAM:
            player_car = player.inverted_car_data
            ball = state.inverted_ball
        else:
            player_car = player.car_data
            ball = state.ball

        ob = []

        ob.append(ball.position / self.POS_STD)
        ob.append(ball.linear_velocity / self.POS_STD)
        ob.append(ball.angular_velocity / self.ANG_STD)

        ob.append(player_car.position / self.POS_STD)
        ob.append(player_car.linear_velocity / self.POS_STD)
        ob.append(player_car.angular_velocity / self.ANG_STD)
        ob.append([player.boost_amount,
                   int(player.has_flip),
                   int(player.on_ground)])

        for other in players:
            if other.car_id == player.car_id:
                continue

            if player.team_num == common_values.ORANGE_TEAM:
                car_data = other.inverted_car_data
            else:
                car_data = other.car_data

            ob.append(car_data.position / self.POS_STD)
            ob.append(car_data.linear_velocity / self.POS_STD)
            ob.append(car_data.angular_velocity / self.ANG_STD)
            
        return np.concatenate(ob)
