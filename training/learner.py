import os
import sys

import wandb
from redis import Redis
from rlgym.utils.reward_functions import CombinedReward
from rlgym.utils.reward_functions.common_rewards import LiuDistancePlayerToBallReward, EventReward
from rlgym_tools.extra_rewards.diff_reward import DiffReward

from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs
from training.agent import get_agent
from training.obs import NectoObsBuilder
from training.reward import NectoRewardFunction

WORKER_COUNTER = "worker-counter"

config = dict(
    actor_lr=1e-4,
    critic_lr=1e-4,
    n_steps=1_00_000,
    batch_size=4_000,
    minibatch_size=1_000,
    epochs=25,
    gamma=0.99,
    iterations_per_save=10
)

if __name__ == "__main__":
    run_id = None

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild", id=run_id, config=config)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0


    def obs():
        return ExpandAdvancedObs()


    def rew():
        return CombinedReward.from_zipped(
            (DiffReward(LiuDistancePlayerToBallReward()), 1),
            (EventReward(touch=10)),
        )
        # return NectoRewardFunction()


    rollout_gen = RedisRolloutGenerator(redis, obs, rew,
                                        save_every=10, logger=logger, clear=run_id is None)

    agent = get_agent(actor_lr=1e-4, critic_lr=1e-4)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        logger=logger,
    )

    # if run_id is not None:
    #     alg.load("ppos/rocket-learn_1634138943.7612503/rocket-learn_60/checkpoint.pt")

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=10, save_dir="ppos")
