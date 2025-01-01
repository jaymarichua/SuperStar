#!/usr/bin/env python3
# play.py
#
# SuperStar Play Script
# - Checks StarCraft II installation paths on Windows/macOS
# - Ensures maps are copied to the correct location
# - Accepts model arguments for agent vs agent/bot/human
# - Launches the DI-Star Actor

import argparse
import os
import shutil
import sys
import torch
import warnings

from distar.actor import Actor
from distar.ctools.utils import read_config

warnings.filterwarnings(
    "ignore",
    message="Setting attributes on ParameterList is not supported."
)


def get_args():
    """
    Command line parser for specifying:
      --model1, --model2: model filenames minus .pth
      --cpu: forcibly use CPU
      --game_type: agent_vs_agent, agent_vs_bot, or human_vs_agent
    """
    parser = argparse.ArgumentParser(description="SuperStar play script for SC2.")
    parser.add_argument(
        "--model1",
        type=str,
        default=None,
        help="Name of the first model (minus '.pth')."
    )
    parser.add_argument(
        "--model2",
        type=str,
        default=None,
        help="Name of the second model (minus '.pth')."
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Use CPU for inference, ignoring CUDA checks."
    )
    parser.add_argument(
        "--game_type",
        type=str,
        default="human_vs_agent",
        choices=["agent_vs_agent", "agent_vs_bot", "human_vs_agent"],
        help="Select match style (agent vs agent/bot/human)."
    )
    return parser.parse_args()


def detect_sc2_install():
    """
    Locates the StarCraft II installation path on Windows or macOS.
    Raises an error if not found.
    """
    # Windows default
    if os.path.exists(r"C:\Program Files (x86)\StarCraft II"):
        return r"C:\Program Files (x86)\StarCraft II"

    # macOS default
    if os.path.exists("/Applications/StarCraft II"):
        return "/Applications/StarCraft II"

    # Fallback to SC2PATH environment variable
    sc2path = os.environ.get("SC2PATH", "")
    if not sc2path:
        raise EnvironmentError(
            "Please set SC2PATH to your StarCraft II installation."
        )

    if not os.path.isdir(sc2path):
        raise NotADirectoryError(
            f"SC2PATH={sc2path} does not exist. Ensure correct install directory."
        )
    return sc2path


def check_sc2_version(sc2path):
    """
    Optional: Attempt to detect SC2 version from 'Versions' subfolder.
    Logs a message if found. This is a simple demonstration.
    """
    versions_dir = os.path.join(sc2path, "Versions")
    if not os.path.isdir(versions_dir):
        print(f"[INFO] 'Versions' directory not found in {sc2path}, skipping version check.")
        return

    # E.g., find subfolder naming: "Base75689", "Base81009", etc.
    subfolders = [
        d for d in os.listdir(versions_dir)
        if os.path.isdir(os.path.join(versions_dir, d)) and d.startswith("Base")
    ]
    if subfolders:
        # Example: "Base75689"
        subfolders.sort()
        print(f"[INFO] SC2 Versions found: {', '.join(subfolders)}")
        print(f"[INFO] Using: {subfolders[-1]} as the newest base.")
    else:
        print(f"[INFO] No SC2 base versions found under {versions_dir}.")


if __name__ == "__main__":
    args = get_args()

    # Attempt SC2 detection
    sc2path = detect_sc2_install()
    assert os.path.exists(sc2path), f"SC2PATH: {sc2path} does not exist!"
    print(f"[INFO] StarCraft II installation detected at: {sc2path}")

    # Optional SC2 version check
    check_sc2_version(sc2path)

    # Copy or remove the Ladder2019Season2 map folder
    maps_dir = os.path.join(sc2path, "Maps", "Ladder2019Season2")
    if os.path.exists(maps_dir):
        shutil.rmtree(maps_dir)
    if not os.path.exists(maps_dir):
        local_map_src = os.path.join(os.path.dirname(__file__), "../envs/maps/Ladder2019Season2")
        shutil.copytree(local_map_src, maps_dir)
        print(f"[INFO] Copied Ladder2019Season2 maps to {maps_dir}")

    # Load user config
    user_config = read_config(os.path.join(os.path.dirname(__file__), "user_config.yaml"))
    user_config.actor.job_type    = "eval_test"
    user_config.common.type       = "play"
    user_config.actor.episode_num = 1
    user_config.env.realtime      = True

    # Resolve model paths
    script_dir = os.path.dirname(__file__)
    default_model_path = os.path.join(script_dir, "rl_model.pth")

    if args.model1:
        path_model1 = os.path.join(script_dir, args.model1 + ".pth")
        user_config.actor.model_paths["model1"] = path_model1
    else:
        path_model1 = user_config.actor.model_paths["model1"]
    if path_model1 == "default":
        path_model1 = default_model_path
        user_config.actor.model_paths["model1"] = path_model1

    if args.model2:
        path_model2 = os.path.join(script_dir, args.model2 + ".pth")
        user_config.actor.model_paths["model2"] = path_model2
    else:
        path_model2 = user_config.actor.model_paths["model2"]
    if path_model2 == "default":
        path_model2 = default_model_path
        user_config.actor.model_paths["model2"] = path_model2

    assert os.path.exists(path_model1), f"Model1 file: {path_model1} not found."
    assert os.path.exists(path_model2), f"Model2 file: {path_model2} not found."

    # Decide CPU vs CUDA
    if not args.cpu:
        if not torch.cuda.is_available():
            raise EnvironmentError("CUDA not available despite --cpu=False. Install CUDA or use --cpu.")
        user_config.actor.use_cuda = True
        print("[INFO] Using CUDA for model inference.")
    else:
        user_config.actor.use_cuda = False
        print("[WARNING] Using CPU. Performance may degrade.")

    # Handle game_type logic
    valid_types = ["agent_vs_agent", "agent_vs_bot", "human_vs_agent"]
    if args.game_type not in valid_types:
        raise ValueError(f"Invalid game_type={args.game_type}. Must be one of: {valid_types}")

    # Setup env/player IDs
    base1 = os.path.basename(path_model1).split(".")[0]
    base2 = os.path.basename(path_model2).split(".")[0]

    if args.game_type == "agent_vs_agent":
        user_config.env.player_ids = [base1, base2]
    elif args.game_type == "agent_vs_bot":
        user_config.actor.player_ids = ["model1"]  # The local agent
        bot_level = "bot10"
        if args.model2 and "bot" in args.model2:
            bot_level = args.model2
        user_config.env.player_ids = [base1, bot_level]
    elif args.game_type == "human_vs_agent":
        user_config.actor.player_ids = ["model1"]
        user_config.env.player_ids   = [base1, "human"]

    # Initialize Actor and start
    actor = Actor(user_config)
    actor.run()

    print("[INFO] SuperStar play script finished.")