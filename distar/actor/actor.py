#!/usr/bin/env python3
# actor.py
#
# DI-Star/SC2 Actor:
# - Preserves original model I/O shapes for legacy compatibility
# - Enforces APM limit on bot/model players
# - Flags repeated actions as potential spam
# - Identifies possible 'toxic' strategies (harass, cheese, rush)
# - Logs concise, confident messages aligned with SC2 knowledge

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

# Default actor config
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))

###############################################################################
# Illustrative mapping of SC2 actions for spam logs. Adjust as required.
###############################################################################
ACTION_ID_NAME_MAP = {
    0:   "NoOp",
    1:   "Move_screen",
    2:   "Attack_pt",
    3:   "Attack_unit",
    12:  "HoldPosition",
    19:  "Rally_Workers_pt",
    4:   "Train_Zergling_quick",
    45:  "Train_Drone_quick",
    88:  "Morph_Lair_quick",
    503: "Effect_InjectLarva_unit",
    13:  "Build_SpawningPool",
    14:  "Build_RoachWarren",
    15:  "Build_BanelingNest",
    16:  "Build_EvolutionChamber",
    17:  "Build_Spire",
    110: "Build_Barracks",
    111: "Build_SupplyDepot",
    112: "Build_Refinery",
    113: "Build_Gateway",
    114: "Build_Pylon",
    115: "Build_Assimilator",
    101: "Research_Burrow",
    120: "Train_Overlord_quick",
    999: "Some_Future_Action"
}

def format_game_time(seconds: float) -> str:
    """Converts a float game-time in seconds to HH:MM:SS."""
    hh = int(seconds // 3600)
    mm = int((seconds % 3600) // 60)
    ss = int(seconds % 60)
    return f"{hh:02d}:{mm:02d}:{ss:02d}"

class RollingRewardHackingMonitor:
    """Tracks repeated identical actions in short windows to flag spam."""

    def __init__(self, loop_threshold=10, window_seconds=3.0, warn_interval=10):
        self.loop_threshold   = loop_threshold
        self.window_seconds   = window_seconds
        self.warn_interval    = warn_interval
        self.action_history   = defaultdict(deque)
        self.steps_since_warn = 0
        self.last_game_time_s = 0.0

    def record_action(self, action_id: int) -> None:
        """Append the current time for this action, removing old timestamps."""
        now = time.time()
        self.action_history[action_id].append(now)
        while self.action_history[action_id] and (now - self.action_history[action_id][0]) > self.window_seconds:
            self.action_history[action_id].popleft()

    def detect_spam_loops(self, logger=None) -> None:
        """Checks if any action repeats >= loop_threshold within window_seconds."""
        self.steps_since_warn += 1
        if self.steps_since_warn < self.warn_interval:
            return

        suspicious_actions = []
        for act_id, times in self.action_history.items():
            if len(times) >= self.loop_threshold:
                suspicious_actions.append((act_id, len(times)))

        if suspicious_actions and logger:
            gt = format_game_time(self.last_game_time_s)
            for (act_id, count) in suspicious_actions:
                act_name = ACTION_ID_NAME_MAP.get(act_id, f"ActionID_{act_id}")
                logger.info(
                    f"[RollingRewardHackingMonitor] @GameTime {gt}: "
                    f"'{act_name}' repeated {count} times in {self.window_seconds:.1f}s."
                )
        self.steps_since_warn = 0

class ToxicStrategyMonitor:
    """Flags worker harass, early expansions, and Zergling rush conditions."""

    WORKER_TYPES    = [45, 104, 84]
    EXPANSION_TYPES = [86, 59, 18]
    ZERGLING_TYPE   = 105

    def __init__(self, early_game_cutoff=420, max_worker_harass=5,
                 cheese_threshold=2, zergling_rush_count=10, zergling_rush_time=420):
        self.early_game_cutoff     = early_game_cutoff
        self.max_worker_harass     = max_worker_harass
        self.cheese_threshold      = cheese_threshold
        self.zergling_rush_count   = zergling_rush_count
        self.zergling_rush_time    = zergling_rush_time

        self.worker_harass_count   = 0
        self.early_expansion_count = 0
        self.seen_expansion_tags   = set()
        self.zergling_rush_flag    = False
        self.toxic_strategic_events= 0

    def update_toxic_strategies(self, raw_ob, game_time_s: float, logger=None) -> None:
        """Analyzes SC2 state for harass, expansions, or early Zergling masses."""
        self._check_worker_harass(raw_ob, game_time_s, logger)
        self._check_early_expansions(raw_ob, game_time_s, logger)
        self._check_zergling_rush(raw_ob, game_time_s, logger)

        triggers = []
        if self.worker_harass_count > self.max_worker_harass:
            triggers.append(f"HarassCount={self.worker_harass_count}")
        if self.early_expansion_count > self.cheese_threshold:
            triggers.append(f"Expansions={self.early_expansion_count}")
        if self.zergling_rush_flag:
            triggers.append("ZerglingRush=TRUE")

        if triggers:
            self.toxic_strategic_events += 1
            if logger:
                gt = format_game_time(game_time_s)
                logger.info(
                    f"[ToxicStrategyMonitor] @GameTime {gt}: Potentially Toxic => {', '.join(triggers)}"
                )

    def _check_worker_harass(self, raw_ob, game_time_s: float, logger=None) -> None:
        """Increments harass count if an enemy worker is <50% HP."""
        for u in raw_ob.observation.raw_data.units:
            if u.alliance == 4 and u.unit_type in self.WORKER_TYPES:
                if u.health < (u.health_max * 0.5) and u.health > 0:
                    self.worker_harass_count += 1
                    if logger:
                        gt = format_game_time(game_time_s)
                        logger.info(
                            f"[ToxicStrategyMonitor] @GameTime {gt}: WorkerHarass - "
                            f"WorkerType={u.unit_type}, HP={u.health}/{u.health_max}, "
                            f"TotalHarass={self.worker_harass_count}"
                        )

    def _check_early_expansions(self, raw_ob, game_time_s: float, logger=None) -> None:
        """Each new expansion is recorded once if built early."""
        if game_time_s <= self.early_game_cutoff:
            for unit in raw_ob.observation.raw_data.units:
                if unit.alliance == 1 and unit.unit_type in self.EXPANSION_TYPES and unit.build_progress < 1.0:
                    if unit.tag not in self.seen_expansion_tags:
                        self.seen_expansion_tags.add(unit.tag)
                        self.early_expansion_count += 1
                        if logger:
                            gt = format_game_time(game_time_s)
                            logger.info(
                                f"[ToxicStrategyMonitor] @GameTime {gt}: EarlyExpansion - "
                                f"UnitType={unit.unit_type}, Tag={unit.tag}, "
                                f"ExpansionsSoFar={self.early_expansion_count}"
                            )

    def _check_zergling_rush(self, raw_ob, game_time_s: float, logger=None) -> None:
        """If game_time_s <= zergling_rush_time and enough lings exist => rush flag."""
        if (not self.zergling_rush_flag) and (game_time_s <= self.zergling_rush_time):
            zerglings_seen = sum(
                1 for u in raw_ob.observation.raw_data.units
                if (u.alliance == 1 and u.unit_type == self.ZERGLING_TYPE)
            )
            if zerglings_seen >= self.zergling_rush_count:
                self.zergling_rush_flag = True
                if logger:
                    gt = format_game_time(game_time_s)
                    logger.info(
                        f"[ToxicStrategyMonitor] @GameTime {gt}: ZerglingRush => {zerglings_seen} lings."
                    )

    def summarize_toxic_strategies(self) -> dict:
        """Returns final stats on harass, expansions, and zergling rush flags."""
        return {
            "worker_harass_count": self.worker_harass_count,
            "early_expansion_or_proxy_count": self.early_expansion_count,
            "zergling_rush_flag": self.zergling_rush_flag,
            "toxic_strategic_events": self.toxic_strategic_events
        }

class Actor:
    """Manages SC2Env interaction, APM limit, spam detection, and toxic logs."""

    def __init__(self, cfg):
        cfg = deep_merge_dicts(default_config, cfg)
        self._whole_cfg       = cfg
        self._cfg             = cfg.actor
        self._job_type        = self._cfg.job_type
        self._league_job_type = self._cfg.get('league_job_type', 'train')
        self._actor_uid       = str(uuid.uuid1())
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
            t = self._whole_cfg.communication.actor_ask_for_job_interval
            self.max_job_duration = t * random.uniform(0.7, 1.3)

        self._bot_action_timestamps = {}
        self._reward_hacking_monitor = RollingRewardHackingMonitor(
            loop_threshold=self._cfg.get('loop_threshold', 10),
            window_seconds=self._cfg.get('spam_window_seconds', 3.0),
            warn_interval=self._cfg.get('warn_interval', 10)
        )
        self._toxic_strategy_monitor = ToxicStrategyMonitor(
            early_game_cutoff=420,
            max_worker_harass=5,
            cheese_threshold=2,
            zergling_rush_count=10,
            zergling_rush_time=420
        )

        self._setup_agents()

    def _setup_agents(self):
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
                agent.side_id   = idx
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
                                agent._z_path           = loaded_state['z_path']
                                agent.z_idx             = loaded_state['z_idx']

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
        selected_races = []
        for fid in frac_ids:
            selected_races.append(random.choice(FRAC_ID[fid]))
        if selected_races:
            env_info['races'] = selected_races

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

        bot_target_apm  = self._cfg.get('bot_target_apm', 900)
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
                        race = self._whole_cfg.env.races[idx]
                        self.agents[idx].reset(map_name, race, game_info[idx], observations[idx])
                        setattr(self.agents[idx], "partial_reward_sum", 0.0)

                        pid = self.agents[idx].player_id
                        if 'bot' in pid or 'model' in pid:
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
                        actions     = {}
                        players_obs = observations

                        for p_idx, obs_data in players_obs.items():
                            ag  = self.agents[p_idx]
                            pid = ag.player_id

                            if self._job_type == 'train':
                                ag._model_last_iter = self._comm.model_last_iter_dict[pid].item()

                            # APM limit
                            if 'bot' in pid or 'model' in pid:
                                now_t = time.time()
                                if (now_t - last_bot_action_time[pid]) < action_cooldown:
                                    actions[p_idx] = NO_OP_ACTION
                                else:
                                    real_act = ag.step(obs_data)
                                    actions[p_idx] = real_act
                                    last_bot_action_time[pid] = now_t
                                    self._bot_action_timestamps[pid].append(now_t)
                                    while self._bot_action_timestamps[pid] and (now_t - self._bot_action_timestamps[pid][0]) > 60:
                                        self._bot_action_timestamps[pid].popleft()

                                    apm_current = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player={pid}, APM(last60s)={apm_current}")

                                    if isinstance(real_act, list):
                                        for dct in real_act:
                                            if 'func_id' in dct:
                                                self._reward_hacking_monitor.record_action(dct['func_id'])
                            else:
                                actions[p_idx] = ag.step(obs_data)
                            agent_count += 1

                        agent_time = time.time() - agent_start
                        env_start  = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start

                        # SC2 game time from a sample next_obs
                        sample_idx  = next(iter(next_obs.keys()), None)
                        sc2_time_s  = 0.0
                        if sample_idx is not None and 'raw_obs' in next_obs[sample_idx]:
                            gl_val     = next_obs[sample_idx]['raw_obs'].observation.game_loop
                            sc2_time_s = gl_val / 22.4
                            self._reward_hacking_monitor.last_game_time_s = sc2_time_s

                        # Check spam
                        self._reward_hacking_monitor.detect_spam_loops(logger=self._logger)

                        # Check toxic
                        for px, nobs in next_obs.items():
                            if 'raw_obs' in nobs:
                                loop_val   = nobs['raw_obs'].observation.game_loop
                                game_ts    = loop_val / 22.4
                                self._toxic_strategy_monitor.update_toxic_strategies(
                                    nobs['raw_obs'],
                                    game_ts,
                                    logger=self._logger
                                )

                        # Data collection if training
                        if 'train' in self._job_type:
                            post_proc_time  = 0
                            post_proc_count = 0
                            send_data_time  = 0
                            send_data_count = 0

                            for px, nobs_ in next_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or self.agents[px].player_id in self._comm.job['send_data_players']
                                )
                                self.agents[px].partial_reward_sum += reward[px]

                                if store_data:
                                    t0 = time.time()
                                    traj_data = self.agents[px].collect_data(nobs_, reward[px], done, px)
                                    post_proc_time += (time.time() - t0)
                                    post_proc_count+=1

                                    if traj_data is not None and self._job_type == 'train':
                                        t1 = time.time()
                                        self._comm.send_data(traj_data, self.agents[px].player_id)
                                        send_data_time += (time.time() - t1)
                                        send_data_count+=1
                                else:
                                    self.agents[px].update_fake_reward(nobs_)

                        iter_count += 1
                        game_iters += 1

                        if env_id == 0:
                            if 'train' in self._job_type:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time/(agent_count+1e-6),
                                    'env_time': env_time
                                })
                                if post_proc_count>0:
                                    variable_record.update_var({
                                        'post_process_time': post_proc_time,
                                        'post_process_per_agent': post_proc_time/post_proc_count
                                    })
                                if send_data_count>0:
                                    variable_record.update_var({
                                        'send_data_time': send_data_time,
                                        'send_data_per_agent': send_data_time/send_data_count
                                    })
                            else:
                                variable_record.update_var({
                                    'agent_time': agent_time,
                                    'agent_time_per_agent': agent_time/(agent_count+1e-6),
                                    'env_time': env_time
                                })
                            self.iter_after_hook(iter_count, variable_record)

                        if not done:
                            observations = next_obs
                            continue

                        # Episode done => final result if training
                        if self._job_type == 'train':
                            rand_pid    = random.sample(observations.keys(), 1)[0]
                            final_steps = observations[rand_pid]['raw_obs'].observation.game_loop
                            result_info = defaultdict(dict)

                            for i2 in range(len(self.agents)):
                                pid2 = self.agents[i2].player_id
                                s2   = self.agents[i2].side_id
                                r2   = self.agents[i2].race
                                it2  = self.agents[i2].iter_count
                                rw2  = reward[i2]
                                pr2  = getattr(self.agents[i2], "partial_reward_sum", 0.0)
                                ratio2 = None
                                if abs(rw2) > 1e-6:
                                    ratio2 = pr2 / abs(rw2)

                                result_info[s2]['race']                = r2
                                result_info[s2]['player_id']           = pid2
                                result_info[s2]['opponent_id']         = self.agents[i2].opponent_id
                                result_info[s2]['winloss']             = rw2
                                result_info[s2]['agent_iters']         = it2
                                result_info[s2]['partial_reward_sum']  = pr2
                                if ratio2 is not None:
                                    result_info[s2]['partial_reward_ratio'] = ratio2
                                result_info[s2].update(self.agents[i2].get_unit_num_info())
                                result_info[s2].update(self.agents[i2].get_stat_data())

                            result_info['game_steps']    = final_steps
                            result_info['game_iters']    = game_iters
                            result_info['game_duration'] = time.time() - game_start

                            tox_data = self._toxic_strategy_monitor.summarize_toxic_strategies()
                            result_info.update(tox_data)

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
                time.sleep(9999999)
            else:
                return

    def _gpu_inference_loop(self):
        """Manages GPU-batch inference if configured, preserving shape."""
        _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)
        for ag in self.agents:
            ag.model = ag.model.cuda()
            if 'train' in self._job_type:
                ag.teacher_model = ag.teacher_model.cuda()
        start_t   = time.time()
        done_count= 0
        with torch.no_grad():
            while True:
                if self._job_type == 'train':
                    self._comm.async_update_model(self)
                    if time.time() - start_t > self.max_job_duration:
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
        ctx_str = 'spawn' if platform.system().lower()=='windows' else 'fork'
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
        """Optional environment reset from external calls."""
        for p in self.pipes:
            p.send('reset')

    def run(self):
        """Initiates the main actor flow, single or multi-process."""
        try:
            if 'test' in self._job_type:
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        start_t = time.time()
                        while True:
                            if time.time() - start_t > self.max_job_duration:
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
        """Forcibly reset environment processes if training job times out or ends."""
        self._logger.info("Actor reset multi-process.")
        self._close_processes()
        if hasattr(self, '_comm'):
            self._comm.ask_for_job(self)
        self._start_multi_inference_loop()

    def close(self):
        """Graceful environment shutdown."""
        self._logger.info("Actor close.")
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
        """Logs relevant iteration stats, reminiscent of SC2 debug hooks."""
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

    def parse_logs(self, log_file: str) -> tuple:
        """Searches logs for spam/toxic lines, returning them if found."""
        if not os.path.exists(log_file):
            return [], []
        with open(log_file, 'r') as f:
            lines = f.readlines()
        spam_lines  = [ln for ln in lines if 'RollingRewardHackingMonitor' in ln]
        toxic_lines = [ln for ln in lines if 'ToxicStrategyMonitor' in ln]
        return spam_lines, toxic_lines

    def summarize_results(self, result_file: str) -> None:
        """If a JSON result file is available, logs its top-level keys."""
        if not os.path.exists(result_file):
            self._logger.info(f"No such result file: {result_file}")
            return
        with open(result_file, 'r') as f:
            data = json.load(f)
        self._logger.info(f"[Result Summary] => keys: {list(data.keys())}")


if __name__ == '__main__':
    # Demonstration usage with empty config.
    actor = Actor(cfg={})
    actor.run()