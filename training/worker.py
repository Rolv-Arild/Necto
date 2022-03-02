import sys
from distutils.util import strtobool
import argparse

import torch
from redis import Redis
from rlgym.envs import Match
from rlgym.utils.action_parsers import DiscreteAction
from rlgym.utils.reward_functions.common_rewards import ConstantReward
from rlgym_tools.extra_action_parsers.kbm_act import KBMAction
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter


from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker, _unserialize
from rocket_learn.utils.util import ExpandAdvancedObs
from training.learner import WORKER_COUNTER
from training.obs import NectoObsBuilder, NectoObsTEST
from training.parser import NectoAction, NectoActionTEST
from training.reward import NectoRewardFunction
from training.state import NectoStateSetter
from training.terminal import NectoTerminalCondition


def get_match(r, force_match_size, replay_arrays, game_speed=100):
    order = (1, 2, 3, 1, 1, 2, 1, 1, 3, 2, 1)  # Close as possible number of agents
    # order = (1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3)  # Close as possible with 1s >= 2s >= 3s
    # order = (1,)
    team_size = order[r % len(order)]
    if force_match_size:
        team_size = force_match_size

    return Match(
        # reward_function=CombinedReward.from_zipped(
        #     (DiffReward(LiuDistancePlayerToBallReward()), 0.05),
        #     (DiffReward(LiuDistanceBallToGoalReward()), 10),
        #     (EventReward(touch=0.05, goal=10)),
        # ),
        # reward_function=NectoRewardFunction(goal_w=0, shot_w=0, save_w=0, demo_w=0, boost_w=0),
        reward_function=NectoRewardFunction(goal_w=1, team_spirit=0, opponent_punish_w=0, boost_lose_w=0, ),
        terminal_conditions=NectoTerminalCondition(),
        obs_builder=NectoObsBuilder(),
        action_parser=NectoActionTEST(),  # NectoActionTEST(),  # KBMAction()
        state_setter=AugmentSetter(NectoStateSetter(replay_arrays[team_size - 1])),
        self_play=True,
        team_size=team_size,
        game_speed=game_speed,
    )


def make_worker(host, name, password, limit_threads=True, send_gamestates=False, force_match_size=None,
                is_streamer=False, is_human_match=False):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password=password)
    w = r.incr(WORKER_COUNTER) - 1

    current_prob = .8
    eval_prob = .01
    game_speed = 100
    if is_streamer:
        current_prob = 1
        eval_prob = 0
        game_speed = 1

    replay_arrays = _unserialize(r.get("replay-arrays"))

    return RedisRolloutWorker(r, name,
                              match=get_match(w, force_match_size,
                                              game_speed=game_speed, replay_arrays=replay_arrays),
                              past_version_prob=1-current_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_gamestates,
                              display_only=is_streamer)


def main():
    assert len(sys.argv) >= 4

    parser = argparse.ArgumentParser(description='Launch Necto worker')

    parser.add_argument('name', type=ascii,
                        help='<required> who is doing the work?')
    parser.add_argument('ip', type=ascii,
                        help='<required> learner ip')
    parser.add_argument('password', type=ascii,
                        help='<required> learner password')
    parser.add_argument('--compress', action='store_true',
                        help='compress sent data')
    parser.add_argument('--display_only', action='store_true',
                        help='Start a streamer match, dont learn with this instance')
    parser.add_argument('--force_match_size', type=int, nargs='?', metavar='match_size',
                        help='Force a 1s, 2s, or 3s game')
    parser.add_argument('--human_match', action='store_true',
                        help='A required integer positional argument')

    parser.print_help()
    args = parser.parse_args()

    print("Argument values:")
    print(args.name)
    print(args.ip)
    print(args.password)
    print(args.compress)
    print(args.display_only)
    print(args.force_match_size)
    print(args.human_match)

    name = args.name
    ip = args.ip
    password = args.password
    compress = args.compress
    stream_state = args.display_only
    force_match_size = args.force_match_size

    if force_match_size is not None and (force_match_size < 1 or force_match_size > 3):
        parser.error("Match size must be between 1 and 3")

    try:
        worker = make_worker(ip, name, password,
                             limit_threads=True,
                             send_gamestates=compress,
                             force_match_size=force_match_size,
                             is_streamer=stream_state)
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")


if __name__ == '__main__':
    main()
