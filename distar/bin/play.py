import argparse
import os
import shutil
import torch
import warnings

# These imports assume your updated actor.py is in 'distar.actor'
# If your structure differs, update accordingly
from distar.actor import Actor
from distar.ctools.utils import read_config

warnings.filterwarnings("ignore", message="Setting attributes on ParameterList is not supported.")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model1', type=str,
                        default=None,
                        help='Name (without .pth) of the first model file.')
    parser.add_argument('--model2', type=str,
                        default=None,
                        help='Name (without .pth) of the second model file.')
    parser.add_argument('--cpu', action="store_true",
                        help='Use CPU for inference (instead of CUDA).')
    parser.add_argument('--game_type', type=str,
                        default='human_vs_agent',
                        choices=['agent_vs_agent', 'agent_vs_bot', 'human_vs_agent'],
                        help='Type of game to run.')
    return parser.parse_args()


def main():
    # Attempt SC2 path detection
    if os.path.exists(r'C:\Program Files (x86)\StarCraft II'):
        sc2path = r'C:\Program Files (x86)\StarCraft II'
    elif os.path.exists('/Applications/StarCraft II'):
        sc2path = '/Applications/StarCraft II'
    else:
        # If user hasn't specified SC2PATH in environment, raise an error
        assert 'SC2PATH' in os.environ, (
            'Please set your SC2 installation path in SC2PATH or place StarCraft II in a known location.'
        )
        sc2path = os.environ['SC2PATH']

    assert os.path.exists(sc2path), f"SC2PATH does not exist: {sc2path}"

    # Ensure map files exist
    ladder_path = os.path.join(sc2path, 'Maps', 'Ladder2019Season2')
    script_dir = os.path.dirname(__file__)
    local_map_dir = os.path.join(script_dir, '../envs/maps/Ladder2019Season2')
    if os.path.exists(ladder_path):
        shutil.rmtree(ladder_path)
    if not os.path.exists(ladder_path):
        shutil.copytree(local_map_dir, ladder_path)

    # Load user config
    user_config = read_config(os.path.join(script_dir, 'user_config.yaml'))
    user_config.actor.job_type = 'eval_test'
    user_config.common.type = 'play'
    user_config.actor.episode_num = 1
    user_config.env.realtime = True

    # Parse command line arguments
    args = get_args()

    # By default, if no custom model is passed, we fallback to 'rl_model.pth'
    default_model_path = os.path.join(script_dir, 'rl_model.pth')

    # If the user specified model1 from command line, build path. Otherwise use config.
    if args.model1 is not None:
        model1 = os.path.join(script_dir, args.model1 + '.pth')
        user_config.actor.model_paths['model1'] = model1
    else:
        model1 = user_config.actor.model_paths['model1']

    # If config says 'default', set it to default_model_path
    if user_config.actor.model_paths['model1'] == 'default':
        user_config.actor.model_paths['model1'] = default_model_path
        model1 = default_model_path

    # Same for model2
    if args.model2 is not None:
        model2 = os.path.join(script_dir, args.model2 + '.pth')
        user_config.actor.model_paths['model2'] = model2
    else:
        model2 = user_config.actor.model_paths['model2']
    if user_config.actor.model_paths['model2'] == 'default':
        user_config.actor.model_paths['model2'] = default_model_path
        model2 = default_model_path

    # Check that the model files exist
    assert os.path.exists(model1), f"Model1 file does not exist: {model1}"
    assert os.path.exists(model2), f"Model2 file does not exist: {model2}"

    # Decide CPU vs. CUDA
    if not args.cpu:
        assert torch.cuda.is_available(), 'CUDA is not available, please install it or pass --cpu.'
        user_config.actor.use_cuda = True
    else:
        user_config.actor.use_cuda = False
        print('Warning: CPU-only mode will degrade agent performance.')

    # game_type can be agent_vs_agent, agent_vs_bot, or human_vs_agent
    # Adjust config accordingly
    gtype = args.game_type
    if gtype == 'agent_vs_agent':
        # We have two models => e.g. model1 vs. model2
        user_config.env.player_ids = [
            os.path.basename(model1).split('.')[0],
            os.path.basename(model2).split('.')[0]
        ]
    elif gtype == 'agent_vs_bot':
        # The second "model" is actually a bot level
        user_config.actor.player_ids = ['model1']  # single agent
        bot_level = 'bot10'  # default bot
        if args.model2 is not None and 'bot' in args.model2:
            bot_level = args.model2
        user_config.env.player_ids = [
            os.path.basename(model1).split('.')[0],
            bot_level
        ]
    elif gtype == 'human_vs_agent':
        # Player1 is a model, player2 is a human
        user_config.actor.player_ids = ['model1']
        user_config.env.player_ids = [
            os.path.basename(model1).split('.')[0],
            'human'
        ]

    # Create Actor and run the environment
    actor = Actor(user_config)
    actor.run()

    return actor


if __name__ == '__main__':
    actor = main()

    # Optionally parse logs if actor has parse_logs
    log_file = './path_to_log_file.log'
    if hasattr(actor, 'parse_logs'):
        spam_events, toxic_events = actor.parse_logs(log_file)
        print("Spam Events:", spam_events)
        print("Toxic Events:", toxic_events)
    else:
        print("[INFO] parse_logs not found on actor; skipping.")

    # Optionally summarize results if actor has summarize_results
    result_file = 'path_to_result_file.json'
    if hasattr(actor, 'summarize_results'):
        actor.summarize_results(result_file)
    else:
        print("[INFO] summarize_results not found on actor; skipping.")