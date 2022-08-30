import argparse
import random
import sys

import torch
from redis import Redis
from redis.retry import Retry
from redis.exceptions import TimeoutError, ConnectionError
from redis.backoff import EqualJitterBackoff
from rlgym.envs import Match
from rlgym_tools.extra_state_setters.augment_setter import AugmentSetter

from rocket_learn.rollout_generator.redis.redis_rollout_worker import RedisRolloutWorker
from rocket_learn.rollout_generator.redis.utils import _unserialize
from rocket_learn.utils.scoreboard import Scoreboard

try:
    from rocket_learn.agent.pretrained_agents.human_agent import HumanAgent
except ImportError:
    pass

from training.obs import NectoObsBuilder
from training.parser import NectoAction
from training.reward import NectoRewardFunction
from training.state import NectoStateSetter
from training.terminal import NectoTerminalCondition, NectoHumanTerminalCondition


def get_match(r, force_match_size, scoreboard, game_speed=100, human_match=False):
    if force_match_size is not None:
        team_size = force_match_size  # TODO
    else:
        team_size = 3

    terminals = NectoTerminalCondition
    if human_match:
        terminals = NectoHumanTerminalCondition

    return Match(
        reward_function=NectoRewardFunction(),
        terminal_conditions=NectoTerminalCondition(),
        obs_builder=NectoObsBuilder(scoreboard, None, 6),
        action_parser=NectoAction(),  # NectoActionTEST(),  # KBMAction()
        state_setter=AugmentSetter(NectoStateSetter(r)),
        team_size=3,
        spawn_opponents=True,
        game_speed=game_speed,
    )


def make_worker(host, name, password, limit_threads=True, send_obs=True,
                send_gamestates=True, force_match_size=None,
                is_streamer=False, deterministic=False, human_match=False):
    if limit_threads:
        torch.set_num_threads(1)

    r = Redis(host=host, password=password, socket_timeout=300, health_check_interval=30,
              retry_on_error=(ConnectionError, TimeoutError), retry=Retry(EqualJitterBackoff(cap=10, base=1), retries=-1))

    agents = None
    human = None

    dynamic_gm = True if force_match_size is None else False
    past_prob = .2
    eval_prob = .01
    game_speed = 100

    if is_streamer:
        past_prob = 0
        eval_prob = 0
        game_speed = 1
        is_streamer += deterministic

    if human_match:
        past_prob = 0
        eval_prob = 0
        game_speed = 1
        human = HumanAgent()

    scoreboard = Scoreboard()

    # replay_arrays = _unserialize(r.get("replay-arrays"))
    worker = RedisRolloutWorker(r, name,
                                match=get_match(r, force_match_size,
                                                scoreboard=scoreboard,
                                                game_speed=game_speed,
                                                human_match=human_match),
                                dynamic_gm=dynamic_gm,
                                past_version_prob=past_prob,
                                evaluation_prob=eval_prob,
                                send_gamestates=send_gamestates,
                                send_obs=send_obs,
                                streamer_mode=is_streamer,
                                pretrained_agents=agents,
                                human_agent=human,
                                scoreboard=scoreboard)
    worker.env._match._obs_builder.env = worker.env  # noqa hack for infinite boost
    return worker


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
    parser.add_argument('--streamer_mode', action='store_true',
                        help='Start a streamer match, dont learn with this instance')
    parser.add_argument('--deterministic', action='store_true',
                        help='Deterministic streamer mode')
    parser.add_argument('--force_match_size', type=int, nargs='?', metavar='match_size',
                        help='Force a 1s, 2s, or 3s game')
    parser.add_argument('--human_match', action='store_true',
                        help='Play a human match against Necto')

    args = parser.parse_args()

    name = args.name.replace("'", "")
    ip = args.ip.replace("'", "")
    password = args.password.replace("'", "")
    compress = args.compress
    stream_state = args.streamer_mode
    deterministic = args.deterministic
    force_match_size = args.force_match_size
    human_match = args.human_match

    if force_match_size is not None and (force_match_size < 1 or force_match_size > 3):
        parser.error("Match size must be between 1 and 3")
    if deterministic and not stream_state:
        parser.error("Deterministic mode is only available in streamer mode")

    worker = make_worker(ip, name, password,
                         limit_threads=True,
                         send_obs=not compress,
                         send_gamestates=True,
                         force_match_size=force_match_size,
                         is_streamer=stream_state,
                         deterministic=deterministic,
                         human_match=human_match)

    try:
        worker.run()
    finally:
        print("Problem Detected. Killing Worker...")
        worker.env.close()


if __name__ == '__main__':
    main()
