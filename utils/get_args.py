import argparse
import json
from copy import deepcopy

import torch


def override_args(init_args):
    # copy args
    args = deepcopy(init_args)
    env_name, version = args.env_name.split("-")
    file_path = f"config/{env_name}/{args.algo_name}.json"
    current_params = load_hyperparams(file_path=file_path)

    # use pre-defined params if no pram given as args
    for k, v in current_params.items():
        if getattr(args, k) is None:
            setattr(args, k, v)

    return args


def load_hyperparams(file_path):
    """Load hyperparameters for a specific environment from a JSON file."""
    try:
        with open(file_path, "r") as f:
            hyperparams = json.load(f)
            return hyperparams  # .get({})
    except FileNotFoundError:
        print(f"No file found at {file_path}. Returning default empty dictionary.")
        return {}


def get_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument(
        "--project", type=str, default="Exp", help="WandB project classification"
    )
    parser.add_argument(
        "--logdir", type=str, default="log/train_log", help="name of the logging folder"
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="Global folder name for experiments with multiple seed tests.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help='Seed-specific folder name in the "group" folder.',
    )
    parser.add_argument(
        "--env-name",
        type=str,
        default="pointmaze-v0",
        help="Name of the environment to run.",
    )
    parser.add_argument(
        "--algo-name", type=str, default="ppo", help="Name of the algorithm to run."
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility."
    )
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs.")
    parser.add_argument(
        "--num-options",
        type=int,
        default=None,
        help="Number of options (i.e, sub-policies or intrinsic rewards).",
    )

    parser.add_argument(
        "--actor-lr",
        type=float,
        default=1e-4,
        help="Base learning rate for the actor for baselines.",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=1e-4,
        help="Base learning rate for the critic for all algorithms.",
    )
    parser.add_argument(
        "--eps-clip", type=float, default=None, help="PPO clipping parameter."
    )
    parser.add_argument(
        "--actor-fc-dim",
        type=list,
        default=[256, 256],
        help="actor fc layer dimensions.",
    )
    parser.add_argument(
        "--critic-fc-dim",
        type=list,
        default=[256, 256],
        help="critic fc layer dimensions.",
    )
    parser.add_argument(
        "--outer-level-update-mode",
        type=str,
        default="trpo",
        help="IRPO outer-level update mode (e.g., 'trpo', 'sgd').",
    )
    parser.add_argument(
        "--outer-actor-lr",
        type=float,
        default=None,
        help="IRPO: outer-level actor lr when sgd is used.",
    )
    parser.add_argument(
        "--inner-actor-lr", type=float, default=None, help="IRPO: inner-level actor lr."
    )
    parser.add_argument(
        "--num-inner-updates",
        type=int,
        default=None,
        help="IRPO: number of inner updates.",
    )
    parser.add_argument(
        "--weight-option",
        type=str,
        default="softmax",
        help="Weighting method for gradient aggregation: softmax vs argmax.",
    )
    parser.add_argument(
        "--extractor-epochs",
        type=int,
        default=200000,
        help="ALLO: number of training epochs.",
    )
    parser.add_argument(
        "--extractor-lr",
        type=float,
        default=3e-4,
        help="ALLO: learning rate for the extractor.",
    )
    parser.add_argument(
        "--discount-sampling-factor",
        type=float,
        default=None,
        help="ALLO: discount sampling factor.",
    )
    parser.add_argument(
        "--lr-barrier-coeff",
        type=float,
        default=1.0,
        help="ALLO: parameter for prioritizing the orthogonal loss.",
    )
    parser.add_argument(
        "--positional-indices",
        type=list,
        default=None,
        help="ALLO & DRND: The state indices to focus on.",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=10, help="ALLO: feature dimension."
    )
    parser.add_argument(
        "--hl-timesteps",
        type=int,
        default=None,
        help="HRL: Number of high-level policy timesteps.",
    )
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Number of training timesteps."
    )

    parser.add_argument(
        "--log-interval", type=int, default=200, help="Interval for logging results."
    )
    parser.add_argument(
        "--eval-num", type=int, default=10, help="Number of evaluation episodes."
    )
    parser.add_argument("--num-minibatch", type=int, default=None, help="")
    parser.add_argument("--minibatch-size", type=int, default=None, help="")
    parser.add_argument("--batch-size", type=int, default=None, help="")
    parser.add_argument("--K-epochs", type=int, default=None, help="")
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="Target KL constraint.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="Generalized Advantage Estimation (GAE) factor.",
    )
    parser.add_argument(
        "--entropy-scaler", type=float, default=1e-3, help="Base learning rate."
    )
    parser.add_argument(
        "--intrinsic-reward-mode",
        type=str,
        default=None,
        help="Mode for intrinsic rewards (e.g., 'allo', 'drnd', 'allo-drnd').",
    )
    parser.add_argument("--gamma", type=float, default=None, help="Discount factor.")
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="Enable rendering of the environment.",
    )

    parser.add_argument(
        "--gpu-idx", type=int, default=0, help="Index of the GPU to use."
    )

    args = parser.parse_args()
    args.device = select_device(args.gpu_idx)

    return args


def select_device(gpu_idx=0, verbose=True):
    if verbose:
        print(
            "============================================================================================"
        )
        # set device to cpu or cuda
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print(
            "============================================================================================"
        )
    else:
        device = torch.device("cpu")
        if torch.cuda.is_available() and gpu_idx is not None:
            device = torch.device("cuda:" + str(gpu_idx))
            torch.cuda.empty_cache()
    return device
