#!/usr/bin/env python3
# actor.py
#
# Actor code for DI-Star, preserving old model shapes while:
#  - Limiting APM for bot/model players
#  - Detecting spam loops via RollingRewardHackingMonitor
#  - Flagging potentially 'toxic' strategies with improved checks

import os
import time
import traceback
import uuid
import random
import json
import platform
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

# Load and merge default actor config
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))

###############################################################################
# A reference mapping from action IDs to descriptive names.
# Adjust or expand for your environment.
###############################################################################
ACTION_ID_NAME_MAP = {
    0:  "NoOp",
    1:  "Move_screen",
    2:  "Select_rect",
    3:  "Select_point",
    4:  "Attack_pt",
    5:  "Attack_unit",
    6:  "Smart_pt",
    7:  "Smart_unit",
    8:  "Train_Drone_quick",
    9:  "Train_Zergling_quick",
    10: "Build_Hatchery_pt",
    503:"Cancel_quick",
    # Add more as needed.
}

class RollingRewardHackingMonitor:
    """
    RollingRewardHackingMonitor:
      Tracks repeated identical actions in a short time window.
      Logs potential spam.
    """
    def __init__(self, loop_threshold=10, window_seconds=3.0, warn_interval=10):
        self.loop_threshold = loop_threshold
        self.window_seconds = window_seconds
        self.warn_interval = warn_interval
        self.action_history = defaultdict(deque)
        self.steps_since_warn = 0

    def record_action(self, action_id: int) -> None:
        now = time.time()
        self.action_history[action_id].append(now)
        while self.action_history[action_id] and (now - self.action_history[action_id][0]) > self.window_seconds:
            self.action_history[action_id].popleft()

    def detect_spam_loops(self, logger=None) -> None:
        self.steps_since_warn += 1
        if self.steps_since_warn < self.warn_interval:
            return
        suspicious = []
        for act_id, times in self.action_history.items():
            if len(times) >= self.loop_threshold:
                suspicious.append((act_id, len(times)))
        if suspicious and logger:
            for (act_id, count) in suspicious:
                # Provide short descriptor if known
                action_name = ACTION_ID_NAME_MAP.get(act_id, f"Unknown({act_id})")
                logger.info(
                    f"[RollingRewardHackingMonitor] Potential spam: action={action_name} repeated "
                    f"{count} times in {self.window_seconds:.1f}s."
                )
        self.steps_since_warn = 0

class ToxicStrategyMonitor:
    """
    Monitors possible 'toxic' strategies:
      - Worker harass
      - Early expansions / proxy
      - Zergling rush
    With some improved checks to reduce duplicates.
    """

    WORKER_TYPES     = [45, 104, 84]  # SCV, Drone, Probe
    EXPANSION_TYPES  = [86, 59, 18]   # Hatch, Nexus, CC
    ZERGLING_TYPE    = 105           # Zergling

    def __init__(self,
                 early_game_cutoff=180,    # 3:00 for expansions
                 max_worker_harass=5,
                 cheese_threshold=2,
                 zergling_rush_count=6,
                 zergling_rush_time=300):  # 5:00 for lings
        self.early_game_cutoff      = early_game_cutoff
        self.max_worker_harass      = max_worker_harass
        self.cheese_threshold       = cheese_threshold
        self.zergling_rush_count    = zergling_rush_count
        self.zergling_rush_time     = zergling_rush_time

        self.worker_harass_count    = 0
        self.early_expansion_count  = 0
        self.seen_expansion_tags    = set()
        self.zergling_rush_flag     = False
        self.toxic_strategic_events = 0

        # Track workers recently damaged to avoid repeated increments
        self.recently_damaged_workers = {}
        self.worker_damage_window = 10.0  # seconds of real-time to hold the "damaged" record

    def update_toxic_strategies(self, raw_ob, current_game_time: float, logger=None) -> None:
        """
        Called each step. current_game_time is in SC2 seconds.
        We'll convert that to H:MM:SS in logs for clarity.
        """
        hhmmss = self._format_game_time(current_game_time)
        self._check_worker_harass(raw_ob, hhmmss, logger)
        self._check_early_expansions(raw_ob, current_game_time, hhmmss, logger)
        self._check_zergling_rush(raw_ob, current_game_time, hhmmss, logger)

        # If any condition is above threshold => log once
        if (self.worker_harass_count > self.max_worker_harass
            or self.early_expansion_count > self.cheese_threshold
            or self.zergling_rush_flag):
            self.toxic_strategic_events += 1
            if logger:
                logger.info(
                    f"[ToxicStrategyMonitor] Potentially toxic: harass={self.worker_harass_count}, "
                    f"expansions={self.early_expansion_count}, zrush={self.zergling_rush_flag}"
                )

        # Remove expired worker-damage records
        self._cleanup_damaged_workers()

    def _check_worker_harass(self, raw_ob, hhmmss, logger) -> None:
        """
        Increments count if we find an enemy worker below 50% HP,
        ignoring repeats for the same worker within a short time window.
        """
        for u in raw_ob.observation.raw_data.units:
            if u.alliance == 4 and u.unit_type in self.WORKER_TYPES:
                if u.health < (u.health_max * 0.5) and u.health > 0:
                    # Only increment once if not recently flagged
                    if u.tag not in self.recently_damaged_workers:
                        self.worker_harass_count += 1
                        self.recently_damaged_workers[u.tag] = time.time()
                        if logger:
                            logger.info(
                                f"[ToxicStrategyMonitor] WorkerHarass: Found enemy worker tag={u.tag} "
                                f"HP={u.health:.1f}/{u.health_max:.1f} at {hhmmss} (<50%)"
                            )

    def _check_early_expansions(self, raw_ob, current_game_time, hhmmss, logger) -> None:
        """
        Increments count if expansions are started before early_game_cutoff (in SC2 seconds).
        """
        if current_game_time <= self.early_game_cutoff:
            for unit in raw_ob.observation.raw_data.units:
                if (unit.alliance == 1 and
                    unit.unit_type in self.EXPANSION_TYPES and
                    unit.build_progress < 1.0):
                    if unit.tag not in self.seen_expansion_tags:
                        self.seen_expansion_tags.add(unit.tag)
                        self.early_expansion_count += 1
                        if logger:
                            logger.info(
                                f"[ToxicStrategyMonitor] EarlyExpansion: Found new expansion tag={unit.tag} "
                                f"type={unit.unit_type} at {hhmmss} (<{self.early_game_cutoff}s)."
                            )

    def _check_zergling_rush(self, raw_ob, current_game_time, hhmmss, logger) -> None:
        """
        If time < zergling_rush_time and zerglings >= zergling_rush_count => rush.
        """
        if not self.zergling_rush_flag and current_game_time <= self.zergling_rush_time:
            ling_count = sum(
                1 for u in raw_ob.observation.raw_data.units
                if (u.alliance == 1 and u.unit_type == self.ZERGLING_TYPE)
            )
            if ling_count >= self.zergling_rush_count:
                self.zergling_rush_flag = True
                if logger:
                    logger.info(
                        f"[ToxicStrategyMonitor] ZerglingRush: Found {ling_count} zerglings by {hhmmss} "
                        f"(threshold={self.zergling_rush_count})."
                    )

    def _cleanup_damaged_workers(self) -> None:
        """
        Removes worker entries older than worker_damage_window seconds of real-time.
        """
        now = time.time()
        remove_list = []
        for tag, stamp in self.recently_damaged_workers.items():
            if (now - stamp) > self.worker_damage_window:
                remove_list.append(tag)
        for r in remove_list:
            del self.recently_damaged_workers[r]

    def _format_game_time(self, sc2_seconds: float) -> str:
        """
        Returns HH:MM:SS format from SC2 'seconds'.
        Typically in SC2, 22.4 frames = 1 second if 'faster' speed, but
        we assume sc2_seconds is the time in real seconds for clarity.
        """
        hrs  = int(sc2_seconds // 3600)
        mins = int((sc2_seconds % 3600) // 60)
        secs = int(sc2_seconds % 60)
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"

class Actor:
    """
    Actor orchestrates SC2Env interactions with improved 'toxic' checks:
      - APM-limiting
      - Spam detection
      - More robust worker-harass, expansions, zergling rush logic
    """
    def __init__(self, cfg):
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = self._cfg.job_type
        self._league_job_type = self._cfg.get('league_job_type', 'train')
        self._actor_uid = str(uuid.uuid1())
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)

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

        self._bot_action_timestamps = {}
        self._reward_hacking_monitor = RollingRewardHackingMonitor(
            loop_threshold=self._cfg.get('loop_threshold', 10),
            window_seconds=self._cfg.get('spam_window_seconds', 3.0),
            warn_interval=self._cfg.get('warn_interval', 10)
        )
        self._toxic_strategy_monitor = ToxicStrategyMonitor(
            early_game_cutoff=180,
            max_worker_harass=5,
            cheese_threshold=2,
            zergling_rush_count=12,
            zergling_rush_time=299
        )

        self._setup_agents()

    def _setup_agents(self) -> None:
        self.agents = []
        if self._job_type == 'train':
            self._comm.ask_for_job(self)
        else:
            self.models = {}
            map_names = []
            for idx, player_id in enumerate(self._cfg.player_ids):
                if 'bot' in player_id:
                    continue
                AgentCls = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = AgentCls(self._whole_cfg)
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
                            net_state = {
                                k: v for k, v in loaded_state['model'].items()
                                if 'value_networks' not in k
                            }
                            agent.model.load_state_dict(net_state, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]
            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            elif len(map_names) == 2:
                if not (map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'

    def _inference_loop(self, env_id=0, job=None, result_queue=None, pipe_c=None):
        if job is None:
            job = {}
        torch.set_num_threads(1)

        frac_ids = job.get('frac_ids', [])
        env_info = job.get('env_info', {})
        chosen_races = []
        for fid in frac_ids:
            chosen_races.append(random.choice(FRAC_ID[fid]))
        if chosen_races:
            env_info['races'] = chosen_races

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

                    for idx in observations.keys():
                        self.agents[idx].env_id = env_id
                        race_str = self._whole_cfg.env.races[idx]
                        self.agents[idx].reset(map_name, race_str, game_info[idx], observations[idx])
                        setattr(self.agents[idx], "partial_reward_sum", 0.0)

                        pid = self.agents[idx].player_id
                        if ('bot' in pid) or ('model' in pid):
                            last_bot_action_time[pid] = 0.0
                            self._bot_action_timestamps[pid] = deque()

                    while True:
                        if pipe_c and pipe_c.poll():
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

                        for player_idx, obs_data in players_obs.items():
                            ag = self.agents[player_idx]
                            pid = ag.player_id

                            if self._job_type == 'train':
                                ag._model_last_iter = self._comm.model_last_iter_dict[pid].item()

                            if ('bot' in pid) or ('model' in pid):
                                now_t = time.time()
                                if (now_t - last_bot_action_time[pid]) < action_cooldown:
                                    actions[player_idx] = NO_OP_ACTION
                                else:
                                    real_act = ag.step(obs_data)
                                    actions[player_idx] = real_act
                                    last_bot_action_time[pid] = now_t
                                    self._bot_action_timestamps[pid].append(now_t)
                                    while self._bot_action_timestamps[pid] and (now_t - self._bot_action_timestamps[pid][0]) > 60:
                                        self._bot_action_timestamps[pid].popleft()
                                    apm_now = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player {pid}: {apm_now} (last 60s)")

                                    # Record action for spam detection
                                    if isinstance(real_act, list):
                                        for dct in real_act:
                                            if 'func_id' in dct:
                                                self._reward_hacking_monitor.record_action(dct['func_id'])
                            else:
                                actions[player_idx] = ag.step(obs_data)

                            agent_count += 1

                        agent_time = time.time() - agent_start
                        env_start = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start

                        self._reward_hacking_monitor.detect_spam_loops(logger=self._logger)

                        # Toxic detection
                        for p_idx, nxt_data in next_obs.items():
                            if 'raw_obs' in nxt_data:
                                gl_val = nxt_data['raw_obs'].observation.game_loop
                                # Convert game_loop frames to SC2 seconds if needed.
                                # If gl_val is already in seconds, just pass it directly.
                                current_time_s = gl_val  / 22.4
                                self._toxic_strategy_monitor.update_toxic_strategies(
                                    nxt_data['raw_obs'], current_time_s, logger=self._logger
                                )

                        # If training => gather data
                        if 'train' in self._job_type:
                            post_t = 0; post_c = 0
                            send_t = 0; send_c = 0
                            for p_idx, nxt_p_obs in next_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or self.agents[p_idx].player_id in self._comm.job['send_data_players']
                                )
                                self.agents[p_idx].partial_reward_sum += reward[p_idx]
                                if store_data:
                                    t0 = time.time()
                                    traj_data = self.agents[p_idx].collect_data(nxt_p_obs, reward[p_idx], done, p_idx)
                                    post_t += (time.time() - t0)
                                    post_c += 1

                                    if traj_data is not None and self._job_type == 'train':
                                        t1 = time.time()
                                        self._comm.send_data(traj_data, self.agents[p_idx].player_id)
                                        send_t += (time.time() - t1)
                                        send_c += 1
                                else:
                                    self.agents[p_idx].update_fake_reward(nxt_p_obs)

                        iter_count += 1
                        game_iters += 1

                        if env_id == 0:
                            if 'train' in self._job_type:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time / (agent_count + 1e-6),
                                    'env_time': env_time
                                })
                                if post_c > 0:
                                    variable_record.update_var({
                                        'post_process_time': post_t,
                                        'post_process_per_agent': post_t / post_c
                                    })
                                if send_c > 0:
                                    variable_record.update_var({
                                        'send_data_time': send_t,
                                        'send_data_per_agent': send_t / send_c
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

                        # Episode end => gather final data if training
                        if self._job_type == 'train':
                            rand_pid = random.sample(observations.keys(), 1)[0]
                            game_steps_done = observations[rand_pid]['raw_obs'].observation.game_loop
                            result_data = defaultdict(dict)

                            for i2 in range(len(self.agents)):
                                pid2    = self.agents[i2].player_id
                                side2   = self.agents[i2].side_id
                                race2   = self.agents[i2].race
                                iters2  = self.agents[i2].iter_count
                                rew2    = reward[i2]
                                partial2= getattr(self.agents[i2], "partial_reward_sum", 0.0)
                                ratio2  = None
                                if abs(rew2) > 1e-6:
                                    ratio2 = partial2 / abs(rew2)

                                result_data[side2]['race'] = race2
                                result_data[side2]['player_id'] = pid2
                                result_data[side2]['opponent_id'] = self.agents[i2].opponent_id
                                result_data[side2]['winloss'] = rew2
                                result_data[side2]['agent_iters'] = iters2
                                result_data[side2]['partial_reward_sum'] = partial2
                                if ratio2 is not None:
                                    result_data[side2]['partial_reward_ratio'] = ratio2
                                result_data[side2].update(self.agents[i2].get_unit_num_info())
                                result_data[side2].update(self.agents[i2].get_stat_data())

                            duration = time.time() - game_start
                            result_data['game_steps']    = game_steps_done
                            result_data['game_iters']    = game_iters
                            result_data['game_duration'] = duration

                            # Summarize toxic data
                            tox_data = self._toxic_strategy_monitor.summarize_toxic_strategies()
                            result_data.update(tox_data)

                            self._comm.send_result(result_data)

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
                time.sleep(9999999)
            else:
                return

    def _gpu_inference_loop(self):
        _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)
        for ag in self.agents:
            ag.model = ag.model.cuda()
            if 'train' in self._job_type:
                ag.teacher_model = ag.teacher_model.cuda()
        st = time.time()
        done_count = 0
        with torch.no_grad():
            while True:
                if self._job_type == 'train':
                    self._comm.async_update_model(self)
                    if time.time() - st > self.max_job_duration:
                        self.close()
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self.close()
                            break
                elif self._job_type == 'eval':
                    if self._result_queue.qsize():
                        self._result_queue.get()
                        done_count += 1
                        if done_count == len(self._processes):
                            self._close_processes()
                            break
                for ag in self.agents:
                    ag.gpu_batch_inference()
                    if 'train' in self._job_type:
                        ag.gpu_batch_inference(teacher=True)

    def _start_multi_inference_loop(self):
        self._close_processes()
        self._processes = []
        job = self._comm.job if hasattr(self, '_comm') else {}
        self.pipes = []
        ctx_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        mp_ctx = mp.get_context(ctx_str)
        self._result_queue = mp_ctx.Queue()
        for env_id in range(self._cfg.env_num):
            pipe_p, pipe_c = mp_ctx.Pipe()
            p = mp_ctx.Process(
                target=self._inference_loop,
                args=(env_id, job, self._result_queue, pipe_c),
                daemon=True
            )
            self.pipes.append(pipe_p)
            self._processes.append(p)
            p.start()

    def reset_env(self):
        for p in self.pipes:
            p.send('reset')

    def run(self):
        try:
            if 'test' in self._job_type:
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        st = time.time()
                        while True:
                            if time.time() - st > self.max_job_duration:
                                self.reset()
                            self._comm.update_model(self)
                            time.sleep(1)
                if self._job_type == 'eval':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        for _ in range(len(self._processes)):
                            self._result_queue.get()
                        self._close_processes()
        except Exception as e:
            print('[MAIN LOOP ERROR]', e, flush=True)
            print(''.join(traceback.format_tb(e.__traceback__)), flush=True)

    def reset(self):
        self._logger.info('Actor reset multi-process.')
        self._close_processes()
        if hasattr(self, '_comm'):
            self._comm.ask_for_job(self)
        self._start_multi_inference_loop()

    def close(self):
        self._logger.info('Actor close.')
        time.sleep(2)
        if hasattr(self, '_comm'):
            self._comm.close()
        self._close_processes()
        time.sleep(1)
        os._exit(0)

    def _close_processes(self):
        if hasattr(self, '_processes'):
            for pp in self.pipes:
                pp.send('close')
            for proc in self._processes:
                proc.join()

    def iter_after_hook(self, iter_count, variable_record):
        if iter_count % self._cfg.print_freq == 0:
            if hasattr(self, '_comm'):
                variable_record.update_var({'update_model_time': self._comm._avg_update_model_time.item()})
            self._logger.info(
                'ACTOR({}):\n{}TimeStep{}{} {}'.format(
                    self._actor_uid,
                    '=' * 35,
                    iter_count,
                    '=' * 35,
                    variable_record.get_vars_text()
                )
            )

    def parse_logs(self, log_file) -> tuple:
        if not os.path.exists(log_file):
            return [], []
        with open(log_file, 'r') as f:
            lines = f.readlines()
        spam_events  = [ln for ln in lines if 'RollingRewardHackingMonitor' in ln]
        toxic_events = [ln for ln in lines if 'ToxicStrategyMonitor' in ln]
        return spam_events, toxic_events

    def summarize_results(self, result_file) -> None:
        if not os.path.exists(result_file):
            self._logger.info(f"No such result file: {result_file}")
            return
        with open(result_file, 'r') as f:
            data = json.load(f)
        self._logger.info(f"[Result Summary] => keys: {list(data.keys())}")

if __name__ == '__main__':
    actor = Actor(cfg={})
    actor.run()