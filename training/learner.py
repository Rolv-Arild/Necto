import os
import sys

import torch
import wandb
from redis import Redis

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from training.agent import get_agent
from training.obs import NectoObsBuilder, NectoObsTEST
from training.parser import NectoAction, NectoActionTEST
from training.reward import NectoRewardFunction

WORKER_COUNTER = "worker-counter"

config = dict(
    seed=123,
    actor_lr=1e-4,
    critic_lr=1e-4,
    n_steps=1_000_000,
    batch_size=100_000,
    minibatch_size=25_000,
    epochs=30,
    gamma=0.995,
    iterations_per_save=10
)

if __name__ == "__main__":
    from rocket_learn.ppo import PPO
    run_id = None

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(project="rocket-learn", entity="rolv-arild", id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis, lambda: NectoObsTEST(6), NectoRewardFunction, NectoActionTEST,
                                        save_every=logger.config.iterations_per_save,
                                        logger=logger, clear=run_id is None)

    agent = get_agent(actor_lr=logger.config.actor_lr, critic_lr=logger.config.critic_lr)

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

    if run_id is not None:
        alg.load("ppos/rocket-learn_1641320591.2569141/rocket-learn_9180/checkpoint.pt")
        # alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        # alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
