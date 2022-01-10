import time

import numpy as np
from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.utils.structures.game_data_struct import GameTickPacket
from rlgym_compat import GameState

from agent import Agent
from obs.necto_obs import NectoObsBuilder


class RLGymExampleBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)

        # FIXME Hey, botmaker. Start here:
        # Swap the obs builder if you are using a different one, RLGym's AdvancedObs is also available
        self.obs_builder = NectoObsBuilder()
        # Your neural network logic goes inside the Agent class, go take a look inside src/agent.py
        self.agent = Agent()
        # Adjust the tickskip if your agent was trained with a different value
        self.tick_skip = 8

        self.game_state: GameState = None
        self.controls = None
        self.action = None
        self.update_action = True
        self.ticks = 0
        self.prev_time = 0
        print('RLGymExampleBot Ready - Index:', index)

    def initialize_agent(self):
        # Initialize the rlgym GameState object now that the game is active and the info is available
        self.game_state = GameState(self.get_field_info())
        self.ticks = self.tick_skip  # So we take an action the first tick
        self.prev_time = 0
        self.controls = SimpleControllerState()
        self.action = np.zeros(8)
        self.update_action = True

    # def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
    #     game_time = packet.game_info.seconds_elapsed
    #     fps120 = self.tick_skip * 0.0083333333333333
    #     self.game_state.decode(packet)
    #
    #     if (game_time - self.prev_time) > fps120:
    #
    #         player = self.game_state.players[self.index]
    #         teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
    #         opponents = [p for p in self.game_state.players if p.team_num != self.team]
    #
    #         self.game_state.players = [player] + teammates + opponents
    #
    #         previous_action = np.copy(self.action)
    #         obs = self.obs_builder.build_obs(player, self.game_state, previous_action)
    #         self.action = self.agent.act(obs)
    #
    #         self.update_controls(self.action)
    #         self.prev_time = game_time
    #     #     if self.index == 0:
    #     #         print(packet.game_info.seconds_elapsed, 'update')
    #     # else:
    #     #     if self.index == 0:
    #     #         print(packet.game_info.seconds_elapsed, 'no update')
    #
    #     return self.controls

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        cur_time = time.perf_counter()
        delta = cur_time - self.prev_time
        self.prev_time = cur_time

        ticks_elapsed = delta * 120  # Smaller than 1/120 on purpose
        self.ticks += ticks_elapsed
        self.game_state.decode(packet, ticks_elapsed)

        if self.update_action:
            self.update_action = False

            player = self.game_state.players[self.index]
            teammates = [p for p in self.game_state.players if p.team_num == self.team and p != player]
            opponents = [p for p in self.game_state.players if p.team_num != self.team]

            # Maybe draw some stuff based on attention scores?
            # self.renderer.draw_string_3d(closest_op.car_data.position, 2, 2, "CLOSEST", self.renderer.white())

            self.game_state.players = [player] + teammates + opponents

            obs = self.obs_builder.build_obs(player, self.game_state, self.action)
            self.action = self.agent.act(obs)

        if self.ticks >= self.tick_skip:
            self.ticks = 0
            self.update_controls(self.action)
            self.update_action = True

        return self.controls

    def update_controls(self, action):
        self.controls.throttle = action[0]
        self.controls.steer = action[1]
        self.controls.pitch = action[2]
        self.controls.yaw = action[3]
        self.controls.roll = action[4]
        self.controls.jump = action[5] > 0
        self.controls.boost = action[6] > 0
        self.controls.handbrake = action[7] > 0
