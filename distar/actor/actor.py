import os
import time
import traceback
import uuid
import random
import json
import platform
import collections
from collections import defaultdict, deque

import torch
import torch.multiprocessing as mp

from distar.agent.import_helper import import_module
from distar.ctools.utils import read_config, deep_merge_dicts
from distar.ctools.utils.log_helper import TextLogger, VariableRecord
from distar.ctools.worker.actor.actor_comm import ActorComm
from distar.ctools.utils.dist_helper import dist_init
from distar.envs.env import SC2Env
from distar.ctools.worker.league.player import FRAC_ID

# The default actor config is loaded here, then partially merged with user-provided configs
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))

class RollingRewardHackingMonitor:
    """
    RollingRewardHackingMonitor:
      - Newly introduced to replace the former static 5-second reset approach.
      - Monitors action repetition within a short rolling window.
    """
    def __init__(self, loop_threshold=10, window_seconds=3.0, warn_interval=10):
        # loop_threshold: max allowed repeats in rolling window
        # window_seconds: rolling window length
        # warn_interval: steps before logging spam again
        self.loop_threshold = loop_threshold
        self.window_seconds = window_seconds
        self.warn_interval = warn_interval
        self.action_history = defaultdict(deque)
        self.steps_since_warn = 0

    def record_action(self, action_id):
        """
        This function now logs each action timestamp
        into a deque for the given action_id.
        """
        now = time.time()
        self.action_history[action_id].append(now)
        # Remove timestamps that exceed current_time - window_seconds
        while self.action_history[action_id] and (now - self.action_history[action_id][0]) > self.window_seconds:
            self.action_history[action_id].popleft()

    def detect_spam_loops(self, logger=None):
        """
        Compares all action counts within the rolling window to the loop_threshold.
        Logs a warning if threshold is exceeded, subject to warn_interval frequency.
        """
        self.steps_since_warn += 1
        if self.steps_since_warn < self.warn_interval:
            return

        suspicious_actions = []
        for act_id, timestamps in self.action_history.items():
            if len(timestamps) >= self.loop_threshold:
                suspicious_actions.append((act_id, len(timestamps)))

        if suspicious_actions and logger:
            for act_id, count in suspicious_actions:
                logger.info(
                    f"[RollingRewardHackingMonitor] Potential spam: Action '{act_id}' repeated "
                    f"{count} times in the last {self.window_seconds:.1f}s."
                )
            self.steps_since_warn = 0

class ToxicStrategyMonitor:
    """
    Tracks potential micro-aggressions or 'toxic' strategies:
      - Frequent worker harass in early game
      - Overly frequent 'cheese' expansions or proxy structures
      - Too many forced worker kills without direct macro follow-up
    Correlate counts with user feedback logs to confirm toxicity.
    """
    def __init__(self, early_game_cutoff=300, max_worker_harass=5, cheese_threshold=2):
        """
        early_game_cutoff: Time (in SC2 game seconds) considered 'early game.'
        max_worker_harass: # of quick worker kills or harass actions beyond which we flag suspicion.
        cheese_threshold: # of unusual expansions/proxy buildings in early game to treat as cheese.
        """
        self.early_game_cutoff = early_game_cutoff
        self.max_worker_harass = max_worker_harass
        self.cheese_threshold = cheese_threshold

        self.worker_harass_count = 0
        self.early_expansion_or_proxy_count = 0
        self.toxic_strategic_events = 0

    def update_toxic_strategies(self, raw_ob, current_game_time, logger=None):
        """
        Called on each environment step or specified interval to check for potential micro-aggressions.
        raw_ob: typically next_obs[idx]['raw_obs']
        current_game_time: SC2 internal time, or your own measure of elapsed game time in seconds/frames.
        """
        # 1) Detect worker harass: repeated kills or direct attacks on worker units (SCVs, Drones, Probes).
        self._check_worker_harass(raw_ob, current_game_time)

        # 2) Check early expansions or proxies if the agent is building expansions in unusual quantity/time.
        self._check_early_cheese_expansions(raw_ob, current_game_time)

        # Example: increment overall toxic events if thresholds exceeded
        if (self.worker_harass_count > self.max_worker_harass
                or self.early_expansion_or_proxy_count > self.cheese_threshold):
            self.toxic_strategic_events += 1
            if logger:
                logger.info(
                    f"[ToxicStrategyMonitor] Potential toxic strategy event at {current_game_time}s: "
                    f"harass_count={self.worker_harass_count}, cheeses={self.early_expansion_or_proxy_count}"
                )

    def _check_worker_harass(self, raw_ob, current_game_time):
        """
        Simple logic: if a large number of enemy worker units (alliance=4 for enemy) have died recently,
        or the agent is sending repeated targeted attacks at worker lines, increment the harass counter.
        """
        # A naive approach might check unit_tags or last hits for low-HP worker units:
        for u in raw_ob.observation.raw_data.units:
            # Suppose these IDs represent SCV=45, Drone=104, Probe=84 (examples only!)
            if u.alliance == 4 and u.unit_type in [45, 104, 84]:
                # If this unit is in critical condition or near death from agent attacks:
                if u.health < 10 and u.weapon_cooldown > 0:
                    self.worker_harass_count += 1

    def _check_early_cheese_expansions(self, raw_ob, current_game_time):
        """
        Identifies expansions or proxy structures built unnaturally early.
        In some StarCraft II contexts, building expansions before 2 minutes (120s) could be suspicious if repeated.
        """
        if current_game_time <= self.early_game_cutoff:
            expansions_built = 0
            for unit in raw_ob.observation.raw_data.units:
                # Example: Checking for building progress on expansions: Hatchery=86, Nexus=59, CommandCenter=18
                if (unit.alliance == 1
                        and unit.unit_type in [86, 59, 18]
                        and unit.build_progress < 1.0):
                    expansions_built += 1

            if expansions_built > 0:
                self.early_expansion_or_proxy_count += expansions_built

    def summarize_toxic_strategies(self):
        """
        Return a dictionary summarizing toxic or micro-aggressive strategic events that occurred.
        You can merge this with user feedback logs or central result data.
        """
        return {
            "worker_harass_count": self.worker_harass_count,
            "early_expansion_or_proxy_count": self.early_expansion_or_proxy_count,
            "toxic_strategic_events": self.toxic_strategic_events
        }

class Actor:
    """
    The Actor class manages SC2Env interactions for a StarCraft II environment.
    It tracks:
      - Episode resets, steps, and done conditions
      - Communication with a training coordinator in 'train' mode
      - RollingRewardHackingMonitor (updated for real-time spam detection)
      - Partial reward ratio computation at game’s end
      - ToxicStrategyMonitor for detecting aggressive strategies
    """
    def __init__(self, cfg):
        # Merge any user config with default actor settings
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = self._cfg.job_type
        self._league_job_type = self._cfg.get('league_job_type', 'train')
        self._actor_uid = str(uuid.uuid1())
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)

        # Logging setup
        self._logger = TextLogger(
            path=os.path.join(
                os.getcwd(),
                'experiments',
                self._whole_cfg.common.experiment_name,
                'actor_log'
            ),
            name=self._actor_uid
        )

        if self._job_type == 'train':
            self._comm = ActorComm(self._whole_cfg, self._actor_uid, self._logger)
            interval = self._whole_cfg.communication.actor_ask_for_job_interval
            self.max_job_duration = interval * random.uniform(0.7, 1.3)

        # For APM (actions per minute) calculations
        self._bot_action_timestamps = {}

        # Updated to RollingRewardHackingMonitor for continuous short-interval checks
        self._reward_hacking_monitor = RollingRewardHackingMonitor(
            loop_threshold=self._cfg.get('loop_threshold', 10),
            window_seconds=self._cfg.get('spam_window_seconds', 3.0),
            warn_interval=self._cfg.get('warn_interval', 10)
        )

        # Instantiate the ToxicStrategyMonitor
        self._toxic_strategy_monitor = ToxicStrategyMonitor()

        self._setup_agents()

    def _setup_agents(self):
        """
        Loads agents, including their models if necessary.
        In 'train' mode, obtains job setup from coordinator.
        """
        self.agents = []
        if self._job_type == 'train':
            self._comm.ask_for_job(self)
        else:
            self.models = {}
            map_names = []

            for idx, player_id in enumerate(self._cfg.player_ids):
                if 'bot' in player_id:
                    continue

                AgentClass = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = AgentClass(self._whole_cfg)
                agent.player_id = player_id
                agent.side_id = idx
                self.agents.append(agent)

                if agent.HAS_MODEL:
                    if player_id not in self.models:
                        if self._cfg.use_cuda:
                            agent.model = agent.model.cuda()
                        else:
                            agent.model = agent.model.eval().share_memory()

                        if not self._cfg.fake_model:
                            loaded_state = torch.load(self._cfg.model_paths[player_id], map_location='cpu')
                            if 'map_name' in loaded_state:
                                map_names.append(loaded_state['map_name'])
                                agent._fake_reward_prob = loaded_state['fake_reward_prob']
                                agent._z_path = loaded_state['z_path']
                                agent.z_idx = loaded_state['z_idx']

                            state_dict = {
                                k: v for k, v in loaded_state['model'].items()
                                if 'value_networks' not in k
                            }
                            agent.model.load_state_dict(state_dict, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]

            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            elif len(map_names) == 2:
                if not (map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'

    def _inference_loop(self, env_id=0, job=None, result_queue=None, pipe_c=None):
        """
        The main loop for environment interactions:
          - Resets the environment
          - Steps through the game until done
          - Tracks partial rewards and calculates partial_reward_ratio
          - Updates ToxicStrategyMonitor for aggressive strategy detection
        """
        if job is None:
            job = {}
        torch.set_num_threads(1)

        frac_ids = job.get('frac_ids', [])
        env_info = job.get('env_info', {})
        races = []
        for frac_id in frac_ids:
            races.append(random.choice(FRAC_ID[frac_id]))
        if races:
            env_info['races'] = races

        merged_cfg = deep_merge_dicts(self._whole_cfg, {'env': env_info})
        self._env = SC2Env(merged_cfg)

        iter_count = 0
        if env_id == 0:
            variable_record = VariableRecord(self._cfg.print_freq)
            variable_record.register_var('agent_time')
            variable_record.register_var('agent_time_per_agent')
            variable_record.register_var('env_time')
            if 'train' in self._job_type:
                variable_record.register_var('post_process_time')
                variable_record.register_var('post_process_per_agent')
                variable_record.register_var('send_data_time')
                variable_record.register_var('send_data_per_agent')
                variable_record.register_var('update_model_time')

        bot_target_apm = self._cfg.get('bot_target_apm', 900)
        action_cooldown = 60.0 / bot_target_apm
        last_bot_action_time = {}
        NO_OP_ACTION = [{
            'func_id': 0,
            'queued': 0,
            'skip_steps': 0,
            'unit_tags': [],
            'target_unit_tag': 0,
            'location': (0, 0)
        }]

        episode_count = 0
        with torch.no_grad():
            while episode_count < self._cfg.episode_num:
                try:
                    game_start = time.time()
                    game_iters = 0
                    observations, game_info, map_name = self._env.reset()

                    # Setting partial_reward_sum for each agent (added lines for partial sum tracking)
                    for idx in observations.keys():
                        self.agents[idx].env_id = env_id
                        race = self._whole_cfg.env.races[idx]
                        self.agents[idx].reset(map_name, race, game_info[idx], observations[idx])
                        setattr(self.agents[idx], "partial_reward_sum", 0.0)

                        pid = self.agents[idx].player_id
                        if 'bot' in pid or 'model' in pid:
                            last_bot_action_time[pid] = 0.0
                            self._bot_action_timestamps[pid] = deque()

                    while True:
                        if pipe_c is not None and pipe_c.poll():
                            cmd = pipe_c.recv()
                            if cmd == 'reset':
                                break
                            if cmd == 'close':
                                self._env.close()
                                return

                        agent_start = time.time()
                        agent_count = 0
                        actions = {}
                        players_obs = observations

                        for player_index, obs in players_obs.items():
                            agent = self.agents[player_index]
                            pid = agent.player_id

                            if self._job_type == 'train':
                                agent._model_last_iter = self._comm.model_last_iter_dict[pid].item()

                            # APM-limited action for bots/models
                            if 'bot' in pid or 'model' in pid:
                                now_time = time.time()
                                if (now_time - last_bot_action_time[pid]) < action_cooldown:
                                    actions[player_index] = NO_OP_ACTION
                                else:
                                    real_action = agent.step(obs)
                                    actions[player_index] = real_action
                                    last_bot_action_time[pid] = now_time
                                    self._bot_action_timestamps[pid].append(now_time)
                                    while (self._bot_action_timestamps[pid] and
                                           (now_time - self._bot_action_timestamps[pid][0]) > 60):
                                        self._bot_action_timestamps[pid].popleft()

                                    apm_now = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player {pid}: {apm_now} (last 60s)")

                                    # Rolling monitor checks each action for potential spam
                                    if isinstance(real_action, list):
                                        for a_dict in real_action:
                                            if 'func_id' in a_dict:
                                                self._reward_hacking_monitor.record_action(a_dict['func_id'])
                            else:
                                actions[player_index] = agent.step(obs)

                            agent_count += 1

                        agent_time = time.time() - agent_start
                        env_start = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start

                        # Real-time spam detection in the new rolling window
                        self._reward_hacking_monitor.detect_spam_loops(logger=self._logger)

                        # Update toxic strategy monitor
                        for p_idx, obs_data in next_obs.items():
                            # Assuming SC2 game time or another measure is used for `current_game_time`
                            current_game_time = observations[p_idx]['game_loop'] / 22.4  # Example conversion
                            self._toxic_strategy_monitor.update_toxic_strategies(obs_data['raw_obs'], current_game_time, self._logger)

                        if 'train' in self._job_type:
                            post_process_time = 0
                            post_process_count = 0
                            send_data_time = 0
                            send_data_count = 0

                            for p_idx, obs_data in next_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or self.agents[p_idx].player_id in self._comm.job['send_data_players']
                                )
                                # Accumulate partial reward each step (additional line)
                                self.agents[p_idx].partial_reward_sum += reward[p_idx]

                                if store_data:
                                    t0 = time.time()
                                    traj_data = self.agents[p_idx].collect_data(
                                        next_obs[p_idx],
                                        reward[p_idx],
                                        done,
                                        p_idx
                                    )
                                    post_process_time += (time.time() - t0)
                                    post_process_count += 1

                                    if traj_data is not None and self._job_type == 'train':
                                        t1 = time.time()
                                        self._comm.send_data(traj_data, self.agents[p_idx].player_id)
                                        send_data_time += (time.time() - t1)
                                        send_data_count += 1
                                else:
                                    self.agents[p_idx].update_fake_reward(next_obs[p_idx])

                        iter_count += 1
                        game_iters += 1

                        if env_id == 0:
                            if 'train' in self._job_type:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time
                                })
                                if post_process_count > 0:
                                    variable_record.update_var({
                                        'post_process_time': post_process_time,
                                        'post_process_per_agent': post_process_time / post_process_count
                                    })
                                if send_data_count > 0:
                                    variable_record.update_var({
                                        'send_data_time': send_data_time,
                                        'send_data_per_agent': send_data_time / send_data_count
                                    })
                            else:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time
                                })
                            self.iter_after_hook(iter_count, variable_record)

                        if not done:
                            observations = next_obs
                            continue

                        if self._job_type == 'train':
                            rand_pid = random.sample(observations.keys(), 1)[0]
                            game_steps = observations[rand_pid]['raw_obs'].observation.game_loop
                            result_info = defaultdict(dict)

                            for idx2 in range(len(self.agents)):
                                pid2 = self.agents[idx2].player_id
                                side_id2 = self.agents[idx2].side_id
                                race2 = self.agents[idx2].race
                                agent_iters = self.agents[idx2].iter_count
                                final_reward = reward[idx2]

                                partial_sum = getattr(self.agents[idx2], "partial_reward_sum", 0.0)
                                ratio = None
                                if abs(final_reward) > 1e-6:
                                    ratio = partial_sum / abs(final_reward)

                                result_info[side_id2]['race'] = race2
                                result_info[side_id2]['player_id'] = pid2
                                result_info[side_id2]['opponent_id'] = self.agents[idx2].opponent_id
                                result_info[side_id2]['winloss'] = final_reward
                                result_info[side_id2]['agent_iters'] = agent_iters
                                result_info[side_id2]['partial_reward_sum'] = partial_sum
                                if ratio is not None:
                                    result_info[side_id2]['partial_reward_ratio'] = ratio
                                result_info[side_id2].update(self.agents[idx2].get_unit_num_info())
                                result_info[side_id2].update(self.agents[idx2].get_stat_data())

                            game_duration = time.time() - game_start
                            result_info['game_steps'] = game_steps
                            result_info['game_iters'] = game_iters
                            result_info['game_duration'] = game_duration

                            toxic_summary = self._toxic_strategy_monitor.summarize_toxic_strategies()
                            result_info.update(toxic_summary)

                            self._comm.send_result(result_info)

                        break

                    episode_count += 1

                except Exception as e:
                    print('[EPISODE LOOP ERROR]', e, flush=True)
                    print(''.join(traceback.format_tb(e.__traceback__)), flush=True)
                    episode_count += 1
                    self._env.close()

            self._env.close()
            if result_queue is not None:
                print(os.getpid(), 'done')
                result_queue.put('done')
            else:
                return

            def parse_logs(self, log_file):
                """
                Parses the log file to extract spam and toxic events.
                """
                with open(log_file, 'r') as file:
                    logs = file.readlines()
                spam_events = [line for line in logs if 'RollingRewardHackingMonitor' in line]
                toxic_events = [line for line in logs if 'ToxicStrategyMonitor' in line]
                return spam_events, toxic_events

            def summarize_results(self, result_file):
                """
                Summarizes and prints the results from the result file.
                """
                with open(result_file, 'r') as file:
                    results = json.load(file)
                print("Partial Reward Ratios:", results.get('partial_reward_ratio', 'N/A'))
                print("Toxic Strategy Summary:", results.get('toxic_strategy_summary', 'N/A'))