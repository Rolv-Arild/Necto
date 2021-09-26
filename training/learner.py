import os

from redis import Redis

import wandb
from rocket_learn.ppo import PPO
from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator

from training.agent import get_agent

WORKER_COUNTER = "worker-counter"

if __name__ == "__main__":
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild")

    redis = Redis(password="rocket-learn")
    redis.delete(WORKER_COUNTER)  # Reset to 0
    rollout_gen = RedisRolloutGenerator(redis, save_every=10, logger=logger)

    agent = get_agent(actor_lr=1e-5, critic_lr=1e-5)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=1_00_000,
        batch_size=20_000,
        minibatch_size=10_000,
        epochs=10,
        gamma=0.995,
        logger=logger,
    )

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(epochs_per_save=10, save_dir="ppos")
