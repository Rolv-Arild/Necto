import os
import sys

import torch
import wandb
from redis import Redis
from rocket_learn.rollout_generator.redis.redis_rollout_generator import RedisRolloutGenerator

from training.agent import get_agent
from training.obs import NectoObsBuilder
from training.parser import NectoAction
from training.reward import NectoRewardFunction

from rocket_learn.utils.stat_trackers.common_trackers import Speed, Demos, TimeoutRate, Touch, EpisodeLength, Boost, \
    BehindBall, TouchHeight, DistToBall

config = dict(
    seed=123,
    actor_lr=1e-4,
    critic_lr=1e-4,
    n_steps=1_000_000,
    batch_size=100_000,
    minibatch_size=10_000,
    epochs=30,
    gamma=0.995,
    iterations_per_save=10,
    iterations_per_model=100,
    ent_coef=0.01,
)

if __name__ == "__main__":
    from rocket_learn.ppo import PPO

    run_id = "t50zfr9d"

    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    wandb_key = os.environ["WANDB_KEY"]
    redis_password = os.environ["REDIS_PASSWORD"]
    _, ip = sys.argv
    wandb.login(key=wandb_key)
    logger = wandb.init(name="necto-v3", project="necto", entity="rolv-arild", id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=redis_password)

    stat_trackers = [
        Speed(), Demos(), TimeoutRate(), Touch(), EpisodeLength(), Boost(), BehindBall(), TouchHeight(), DistToBall()
    ]
    rollout_gen = RedisRolloutGenerator("tecko",
                                        redis,
                                        lambda: NectoObsBuilder(None, 6),
                                        lambda: NectoRewardFunction(),
                                        NectoAction,
                                        save_every=logger.config.iterations_per_save,
                                        model_every=logger.config.iterations_per_model,
                                        logger=logger,
                                        clear=run_id is None,
                                        max_age=1,
                                        stat_trackers=stat_trackers
                                        )

    agent = get_agent(actor_lr=logger.config.actor_lr, critic_lr=logger.config.critic_lr)

    alg = PPO(
        rollout_gen,
        agent,
        n_steps=logger.config.n_steps,
        batch_size=logger.config.batch_size,
        minibatch_size=logger.config.minibatch_size,
        epochs=logger.config.epochs,
        gamma=logger.config.gamma,
        ent_coef=logger.config.ent_coef,
        logger=logger,
    )

    if run_id is not None:
        alg.load("ppos/necto_1660212070.3225243/necto_12300/checkpoint.pt")
        alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
