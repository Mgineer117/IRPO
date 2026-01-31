import argparse
import json
from copy import deepcopy

import torch


def override_args(init_args):
    """
    Overrides the initial arguments with parameters found in a JSON config file.
    The config file path is derived from the environment and algorithm names.
    """
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
    parser = argparse.ArgumentParser(
        description="Argument parser for RL training, including IRPO, PPO, and ALLO configurations."
    )

    # === WANDB / LOGGING SETTINGS === #
    parser.add_argument(
        "--project",
        type=str,
        default="Exp",
        help="The name of the Weights & Biases (WandB) project where these experimental runs will be logged.",
    )
    parser.add_argument(
        "--logdir",
        type=str,
        default="log/train_log",
        help="The local directory path where training logs, checkpoints, and event files will be saved.",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        help="A unique string identifier to group multiple experimental runs (e.g., different seeds) together in WandB dashboard.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="A specific name for this individual run (often used to distinguish seeds within a group).",
    )

    # === ENVIRONMENT & ALGORITHM === #
    parser.add_argument(
        "--env-name",
        type=str,
        default="pointmaze-v0",
        help="The unique ID of the gymnasium/MuJoCo environment to execute (e.g., 'Ant-v4', 'pointmaze-v0').",
    )
    parser.add_argument(
        "--algo-name",
        type=str,
        default="ppo",
        help="The name of the reinforcement learning algorithm to use (e.g., 'ppo', 'irpo', 'trpo').",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="The random seed used for initializing networks and environment reproducibility.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="The total number of independent training sessions to run sequentially (useful for statistical significance).",
    )
    parser.add_argument(
        "--num-options",
        type=int,
        default=None,
        help="The quantity of intrinsic rewards or sub-policies (options) to generate. Represents the size of the intrinsic vector.",
    )

    # === NETWORK ARCHITECTURE & OPTIMIZATION === #
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=5e-5,
        help="The learning rate for the Actor network optimizer (used in standard baselines like PPO).",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=5e-5,
        help="The learning rate for the Critic network optimizer (used across all algorithms).",
    )
    parser.add_argument(
        "--eps-clip",
        type=float,
        default=None,
        help="The clipping parameter 'epsilon' used in the PPO objective function to constrain policy updates (typically 0.1 to 0.2).",
    )
    parser.add_argument(
        "--actor-fc-dim",
        type=int,
        default=[64, 64],
        nargs="+",
        help="List of integers defining the number of neurons in each fully-connected layer of the Actor network.",
    )
    parser.add_argument(
        "--critic-fc-dim",
        type=list,
        default=[256, 256],
        help="List of integers defining the number of neurons in each fully-connected layer of the Critic network.",
    )

    # === IRPO (INTRINSIC REWARD POLICY OPTIMIZATION) PARAMETERS === #
    parser.add_argument(
        "--irpo-type",
        type=str,
        default="irpo",
        help="Specifies the variant of the IRPO algorithm to use. Choices: {'irpo', 'irpo_lr', 'irpo_pick', 'irpo_blend', 'irpo_is'}.",
    )
    parser.add_argument(
        "--outer-level-update-mode",
        type=str,
        default="trpo",
        help="The optimization strategy for the meta-update (outer loop). 'trpo' uses Trust Region updates, 'sgd' uses standard gradient descent.",
    )
    parser.add_argument(
        "--outer-actor-lr",
        type=float,
        default=None,
        help="The learning rate for the outer-level actor update (ONLY required if outer-level-update-mode is 'sgd').",
    )
    parser.add_argument(
        "--inner-actor-lr",
        type=float,
        default=None,
        help="The learning rate used for updating the exploratory policies within the inner loop optimization.",
    )
    parser.add_argument(
        "--num-inner-updates",
        type=int,
        default=None,
        help="The number of gradient updates performed on the exploratory policies per iteration of the inner loop.",
    )
    parser.add_argument(
        "--intrinsic-reward-mode",
        type=str,
        default=None,
        help="Defines the nature of the intrinsic reward signal (e.g., 'allo' for feature-based, 'random' for noise-based).",
    )

    # === ALLO (FEATURE EXTRACTOR) PARAMETERS === #
    parser.add_argument(
        "--extractor-epochs",
        type=int,
        default=50000,
        help="The total number of epochs to train the ALLO feature extractor.",
    )
    parser.add_argument(
        "--extractor-lr",
        type=float,
        default=3e-4,
        help="The learning rate used for the ALLO feature extractor optimizer.",
    )
    parser.add_argument(
        "--discount-sampling-factor",
        type=float,
        default=None,
        help="The discount factor used when sampling data for the feature extractor, prioritizing recent experiences.",
    )
    parser.add_argument(
        "--lr-barrier-coeff",
        type=float,
        default=1.0,
        help="Coefficient scaling the orthogonal loss term in ALLO to prevent feature collapse.",
    )
    parser.add_argument(
        "--positional-indices",
        type=list,
        default=None,
        help="A list of specific state vector indices that the extractor should focus on (or ignore).",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=10,
        help="The dimensionality of the latent feature space learned by the ALLO extractor.",
    )
    parser.add_argument(
        "--hl-timesteps",
        type=int,
        default=None,
        help="Hierarchical RL: The number of timesteps the high-level policy operates before switching.",
    )

    # === TRAINING LOOP CONFIGURATION === #
    parser.add_argument(
        "--timesteps",
        type=int,
        default=None,
        help="The total number of environment interaction steps (samples) to collect during training.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=100,
        help="The frequency (in steps) at which training metrics are logged to console and WandB.",
    )
    parser.add_argument(
        "--eval-num",
        type=int,
        default=10,
        help="The number of full episodes to run during the evaluation phase to calculate average performance.",
    )
    parser.add_argument(
        "--num-minibatch",
        type=int,
        default=None,
        help="The number of mini-batches to split the collected buffer into for PPO updates.",
    )
    parser.add_argument(
        "--minibatch-size",
        type=int,
        default=None,
        help="The size of each mini-batch used during the gradient update step (Alternative to num-minibatch).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="The total number of samples collected per rollout before performing an update.",
    )
    parser.add_argument(
        "--K-epochs",
        type=int,
        default=None,
        help="The number of epochs (passes over the collected batch) to perform during one PPO update cycle.",
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=None,
        help="The maximum allowed KL divergence between the old and new policy. Used for early stopping in PPO or constraint in TRPO.",
    )
    parser.add_argument(
        "--gae",
        type=float,
        default=0.95,
        help="The Lambda parameter for Generalized Advantage Estimation (GAE), balancing bias and variance.",
    )
    parser.add_argument(
        "--entropy-scaler",
        type=float,
        default=1e-3,
        help="The coefficient for the entropy regularization term in the loss function (encourages exploration).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="The discount factor for future rewards (typically 0.99).",
    )

    # === SYSTEM / RENDER === #
    parser.add_argument(
        "--rendering",
        action="store_true",
        help="If set, the environment will be rendered visually during training/evaluation (slows down training).",
    )
    parser.add_argument(
        "--gpu-idx",
        type=int,
        default=0,
        help="The integer index of the GPU device to use (e.g., 0 for cuda:0, 1 for cuda:1).",
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
