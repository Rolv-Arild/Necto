import sys
from distutils.util import strtobool

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
        reward_function=NectoRewardFunction(),
        terminal_conditions=NectoTerminalCondition(),
        obs_builder=NectoObsTEST(6),
        action_parser=NectoActionTEST(),  # NectoActionTEST(),  # KBMAction()
        state_setter=AugmentSetter(NectoStateSetter(replay_arrays[team_size - 1])),
        self_play=True,
        team_size=team_size,
        game_speed=game_speed,
    )


def make_worker(host, name, password, limit_threads=True, send_gamestates=False, force_match_size=None,
                is_streamer=False):
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
                              match=get_match(w, force_match_size, game_speed=game_speed, replay_arrays=replay_arrays),
                              current_version_prob=current_prob,
                              evaluation_prob=eval_prob,
                              send_gamestates=send_gamestates,
                              display_only=is_streamer)


def main():
    # if not torch.cuda.is_available():
    #     sys.exit("Unable to train on your hardware, perhaps due to out-dated drivers or hardware age.")

    assert len(sys.argv) >= 4  # last is optional to force match size

    force_match_size = None

    print(len(sys.argv))

    # import argparse
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-n", "--name", required=True)
    # parser.add_argument("--ip", required=True)
    # parser.add_argument("-p", "--password", required=True)
    # parser.add_argument("-c", "--compress", default=True)
    # parser.add_argument("-d", "--display-only", default=True)
    # parser.add_argument("-g", "--gamemode", default=None)
    # parser.add_argument("-p", "--password", required=True)
    # parser.add_argument("-p", "--password", required=True)
    #
    # compress = bool(strtobool(sys.argv[4])) if len(sys.argv) >= 5 else True
    # display_only = bool(strtobool(sys.argv[5])) if len(sys.argv) >= 6 else False
    # force_match_size = int(sys.argv[6]) if len(sys.argv) >= 7 else None
    # current_version_prob = float(sys.argv[7]) if len(sys.argv) >= 8 else 0.8
    # evaluation_prob = float(sys.argv[8]) if len(sys.argv) >= 9 else 0.01

    if len(sys.argv) == 5:
        _, name, ip, password, compress = sys.argv
        stream_state = False
    elif len(sys.argv) == 6:
        _, name, ip, password, compress, is_stream = sys.argv

        # atm, adding an extra arg assumes you're trying to stream
        stream_state = True
        force_match_size = int(2)

    elif len(sys.argv) == 7:
        _, name, ip, password, compress, is_stream, force_match_size = sys.argv

        # atm, adding an extra arg assumes you're trying to stream
        stream_state = True
        force_match_size = int(force_match_size)

        if not (1 <= force_match_size <= 3):
            force_match_size = None
    else:
        raise ValueError

    try:
        worker = make_worker(ip, name, password,
                             limit_threads=True,
                             send_gamestates=bool(strtobool(compress)),
                             force_match_size=force_match_size,
                             is_streamer=stream_state)
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")


if __name__ == '__main__':
    main()
