import sys

import torch
from redis import Redis
from rlgym.envs import Match

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutWorker
from training.learner import WORKER_COUNTER
from training.obs import NectoObsBuilder
from training.reward import NectoRewardFunction
from training.state import NectoStateSetter
from training.terminal import NectoTerminalCondition


def get_match(r):
    order = (1, 2, 3, 1, 1, 2, 1, 1, 3, 2, 1)  # Close as possible number of agents
    # order = (1, 1, 2, 1, 1, 2, 3, 1, 1, 2, 3)  # Close as possible with 1s >= 2s >= 3s
    team_size = order[r % len(order)]
    return Match(
        reward_function=NectoRewardFunction(),
        terminal_conditions=NectoTerminalCondition(),
        obs_builder=NectoObsBuilder(),
        state_setter=NectoStateSetter(),
        self_play=True,
        team_size=team_size,
    )


def make_worker(host, name, password, limit_threads=True):
    if limit_threads:
        torch.set_num_threads(1)
    r = Redis(host=host, password=password)
    w = r.incr(WORKER_COUNTER) - 1
    return RedisRolloutWorker(r, name, get_match(w), current_version_prob=.9)


def main():
    if not torch.cuda.is_available():
        sys.exit("Unable to train on your hardware, perhaps due to out-dated drivers or hardware age.")

    assert len(sys.argv) == 4
    _, name, ip, password = sys.argv

    worker = make_worker(ip, name, password, limit_threads=True)
    worker.run()


if __name__ == '__main__':
    main()
