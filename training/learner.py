import os
import sys

import torch
import wandb
from redis import Redis
from rlgym.utils.action_parsers import DiscreteAction

from rocket_learn.rollout_generator.redis_rollout_generator import RedisRolloutGenerator
from rocket_learn.utils.util import ExpandAdvancedObs
from training.agent import get_agent
from training.obs import NectoObsOLD, NectoObsBuilder
from training.parser import NectoActionOLD, NectoAction
from training.reward import NectoRewardFunction

WORKER_COUNTER = "worker-counter"

config = dict(
    seed=123,
    actor_lr=5e-5,
    critic_lr=5e-5,
    n_steps=2_000_000,
    batch_size=200_000,
    minibatch_size=20_000,
    epochs=30,
    gamma=0.9975,
    iterations_per_save=5,
    ent_coef=0.007,
)

if __name__ == "__main__":
    from rocket_learn.ppo import PPO

    run_id = "1xtehclu"

    _, ip, password = sys.argv
    wandb.login(key=os.environ["WANDB_KEY"])
    logger = wandb.init(name="necto-v2", project="necto", entity="rolv-arild", id=run_id, config=config)
    torch.manual_seed(logger.config.seed)

    redis = Redis(host=ip, password=password)
    redis.delete(WORKER_COUNTER)  # Reset to 0

    rollout_gen = RedisRolloutGenerator(redis,
                                        lambda: NectoObsBuilder(6),
                                        lambda: NectoRewardFunction(),
                                        # lambda: NectoRewardFunction(goal_w=1, team_spirit=0., opponent_punish_w=0., boost_lose_w=0),
                                        NectoAction,
                                        save_every=logger.config.iterations_per_save,
                                        logger=logger, clear=run_id is None,
                                        max_age=1, min_sigma=2)

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
        alg.load("ppos/necto_1652298396.28676/necto_15350/checkpoint.pt")
        alg.agent.optimizer.param_groups[0]["lr"] = logger.config.actor_lr
        alg.agent.optimizer.param_groups[1]["lr"] = logger.config.critic_lr

    log_dir = "E:\\log_directory\\"
    repo_dir = "E:\\repo_directory\\"

    alg.run(iterations_per_save=logger.config.iterations_per_save, save_dir="ppos")
