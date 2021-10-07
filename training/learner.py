import os
import sys

import wandb
from redis import Redis

from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from training.agent import get_agent

WORKER_COUNTER = "worker-counter"

if __name__ == "__main__":
    run_id = None

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild", id=run_id)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0
    rollout_gen = RedisRolloutGenerator(redis, save_every=10, logger=logger, clear=run_id is None)

    agent = get_agent(actor_lr=1e-4, critic_lr=1e-4)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=1_000_000,
        batch_size=40_000,
        minibatch_size=10_000,
        epochs=25,
        gamma=0.995,
        logger=logger,
    )

    alg.load("ppos/rocket-learn_1633471635.9134436/rocket-learn_150/checkpoint.pt")

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=10, save_dir="ppos")
