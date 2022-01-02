import random

import numpy as np
from rlgym.utils import StateSetter
from rlgym.utils.common_values import CAR_MAX_SPEED, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z, BALL_RADIUS, CAR_MAX_ANG_VEL, \
    BALL_MAX_SPEED
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters import DefaultState, StateWrapper

#from rlgym_utils.extra_state_setters.goalie_state import GoaliePracticeState

from numpy import random as rand

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

class GoaliePracticeState(StateSetter):

    def __init__(self, aerial_only=False, allow_enemy_interference=False, first_defender_in_goal=False, reset_to_max_boost=True):
        """
        GoaliePracticeState constructor.

        :param aerial_only: Boolean indicating whether the shots will only be in the air.
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        super().__init__()
        self.team_turn = 0 #swap every reset who's getting shot at
        
        self.aerial_only = aerial_only
        self.allow_enemy_interference = allow_enemy_interference
        self.first_defender_in_goal = first_defender_in_goal
        self.reset_to_max_boost = reset_to_max_boost

    def reset(self, state_wrapper: StateWrapper):
        """
        Modifies the StateWrapper to set a new shot

        :param state_wrapper: StateWrapper object to be modified with desired state values.
        """
        self._reset_ball(state_wrapper, self.team_turn, self.aerial_only)
        self._reset_cars(state_wrapper, self.team_turn, self.first_defender_in_goal, self.allow_enemy_interference, self.reset_to_max_boost)
        
        #which team will recieve the next incoming shot
        self.team_turn = (self.team_turn + 1) % 2
    

    def _reset_cars(self, state_wrapper: StateWrapper, team_turn, first_defender_in_goal, allow_enemy_interference, reset_to_max_boost):
        """
        Function to set cars in preparation for an incoming shot

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param allow_enemy_interference: Boolean indicating whether opponents will spawn close enough to easily affect the play
        :param first_defender_in_goal: Boolean indicating whether the first defender will spawn in the goal
        :param reset_to_max_boost: Boolean indicating whether the cars will start each episode with 100 boost or keep from last episode
        """
        first_set = False
        for car in state_wrapper.cars:
            # set random position and rotation for all cars based on pre-determined ranges
            
            if car.team_num == team_turn and not first_set:   
                if first_defender_in_goal:
                    y_pos = -GOAL_LINE if car.team_num == 0 else GOAL_LINE
                    car.set_pos(0, y_pos, z=17)
                    first_set = True
                else:
                    self._place_car_in_box_area(car, team_turn)
                    
            else:
                if allow_enemy_interference:
                    self._place_car_in_box_area(car, team_turn)
                
                else:
                    self._place_car_in_box_area(car, car.team_num)                    
            
            
            if reset_to_max_boost:
                car.boost = 100
            
            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX/2, 0) 
            
            
            
    def _place_car_in_box_area(self, car, team_delin):
        """
        Function to place a car in an allowed areaI 

        :param car: car to be modified
        :param team_delin: team number delinator to look at when deciding where to place the car
        """
    
        y_pos = (PLACEMENT_BOX_Y - (rand.random() * PLACEMENT_BOX_Y)) 
        
        if team_delin == 0:
            y_pos -= PLACEMENT_BOX_Y_OFFSET
        else:
            y_pos += PLACEMENT_BOX_Y_OFFSET
    
        car.set_pos(rand.random() * PLACEMENT_BOX_X - PLACEMENT_BOX_X/2, y_pos, z=17) 
        
    def _reset_ball(self, state_wrapper: StateWrapper, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal

        :param state_wrapper: StateWrapper object to be modified.
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """
        
        pos, lin_vel, ang_vel = self._get_shot_parameters(team_turn, aerial_only)
        state_wrapper.ball.set_pos(pos[0], pos[1], pos[2])
        state_wrapper.ball.set_lin_vel(lin_vel[0], lin_vel[1], lin_vel[2])
        state_wrapper.ball.set_ang_vel(ang_vel[0], ang_vel[1], ang_vel[2])
        
    def _get_shot_parameters(self, team_turn, aerial_only):
        """
        Function to set a new ball towards a goal
        
        :param team_turn: team who's getting shot at
        :param aerial_only: Boolean indicating whether should shots only be from the air
        """
        
        # *** Magic numbers are from manually calibrated shots ***
        # *** They are unrelated to numbers in other functions ***
    
        shotpick = random.randrange(4)        
        INVERT_IF_BLUE = (-1 if team_turn == 0 else 1) #invert shot for orang
        
        #random pick x value of target in goal
        x_pos = random.uniform(GOAL_X_MIN, GOAL_X_MAX)
        
        #if its not an air shot, we can randomize the shot speed
        shot_randomizer = 1 if aerial_only else (random.uniform(0,1) )
        
        y_vel = (3000 * INVERT_IF_BLUE) if aerial_only else (3000 * shot_randomizer * INVERT_IF_BLUE)
        if shotpick == 0: #long range shot 
             
            z_pos = 1500 if aerial_only else random.uniform(100, 1500) 
            
            pos = np.array([x_pos,  -3300 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 600])
        elif shotpick == 1: #medium range shot
            z_pos =  1550 if aerial_only else random.uniform(100,  1550)  
            
            pos = np.array([x_pos, -500 * INVERT_IF_BLUE, z_pos])
            lin_vel = np.array([0, y_vel, 100])
            
        elif shotpick == 2: #angled shot    
            z_pos =  1500 if aerial_only else random.uniform(100,  1500) 
            x_pos += 3200 #add offset to start the shot from the side
            
            pos = np.array([x_pos,  0, z_pos])
            lin_vel = np.array([-1900 * shot_randomizer, y_vel, 0])
            
        elif shotpick == 3: # opposite angled shot    
            z_pos =  1500 if aerial_only else random.uniform(100,  1500) 
            x_pos -= 3200 #add offset to start the shot from the other side
            
            pos = np.array([x_pos,  0, z_pos])
            lin_vel = np.array([1900 * shot_randomizer, y_vel, 0])
        else:
            print("FAULT")
        
        ang_vel = np.array([0, 0, 0])
        
        return pos, lin_vel, ang_vel

    
    
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
                ball_dist = np.random.exponential(BALL_MAX_SPEED)
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
    def __init__(self, kickoff_prob=0.01, goalie_prob=0.03):
        super().__init__()
        # TODO sample from SSL replays, kickoff-like
        self.kickoff_prob = kickoff_prob
        self.goalie_prob = goalie_prob
        
        self.default = DefaultState()
        self.random = BetterRandom()
        self.goalie = GoaliePracticeState(first_defender_in_goal=True)

    def reset(self, state_wrapper: StateWrapper):
        if np.random.random() < self.kickoff_prob:
            self.default.reset(state_wrapper)
        elif np.random.random() < self.goalie_prob:
            self.goalie.reset(state_wrapper)
        else:
            self.random.reset(state_wrapper)
