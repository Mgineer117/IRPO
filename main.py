import datetime
import os
import random
import uuid

import torch

import wandb
from utils.functions import (
    call_env,
    concat_csv_columnwise_and_delete,
    seed_all,
    setup_logger,
)
from utils.get_args import get_args, override_args

# it suppresses the wandb printing when it logs data
os.environ["WANDB_SILENT"] = "true"


def run(args, seed, unique_id, exp_time):
    # fix seed
    seed_all(seed)

    # get env
    env = call_env(args)
    logger, writer = setup_logger(args, unique_id, exp_time, seed)

    # run algorithm
    if args.algo_name == "ppo":
        from algorithms.ppo import PPO_Algorithm

        algo = PPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "irpo":
        from algorithms.irpo import IRPO_Algorithm

        algo = IRPO_Algorithm(env=env, logger=logger, writer=writer, args=args)
    elif args.algo_name == "hrl":
        from algorithms.hrl import HRL

        algo = HRL(env=env, logger=logger, writer=writer, args=args)
    else:
        raise NotImplementedError(f"{args.algo_name} is not implemented.")

    algo.begin_training()

    # ✅ Memory cleanup
    del algo, env, logger, writer  # delete large references
    torch.cuda.empty_cache()  # release unreferenced GPU memory
    wandb.finish()


if __name__ == "__main__":
    torch.set_default_dtype(torch.float32)

    init_args = get_args()
    unique_id = str(uuid.uuid4())[:4]
    exp_time = datetime.datetime.now().strftime("%m-%d_%H-%M-%S.%f")

    random.seed(init_args.seed)
    seeds = [random.randint(1, 10_000) for _ in range(init_args.num_runs)]
    print(f"-------------------------------------------------------")
    print(f"      Running ID: {unique_id}")
    print(f"      Running Seeds: {seeds}")
    print(f"      Time Begun   : {exp_time}")
    print(f"-------------------------------------------------------")

    for seed in seeds:
        args = override_args(init_args)
        args.seed = seed
        args.unique_id = unique_id

        run(args, seed, unique_id, exp_time)
    concat_csv_columnwise_and_delete(folder_path=args.logdir)
