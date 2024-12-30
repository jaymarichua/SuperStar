#!/usr/bin/env python3
# =============================================================================
# actor.py
#
# This module defines the Actor class that controls environment interaction
# and data collection routines for DI-Star.
# Below, it includes a short APM measurement mechanism for each bot or "model"
# agent, using timestamps of real actions to compute a rolling 60-second APM.
# =============================================================================

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


# =============================================================================
# Default configuration is loaded and merged with user cfg.
# =============================================================================
default_config = read_config(os.path.join(os.path.dirname(__file__), 'actor_default_config.yaml'))


class Actor(object):
    """
    The Actor class oversees SC2Env interactions, data collection,
    and optional APM tracking for bot agents.
    """

    def __init__(self, cfg):
        """
        Merges user config with defaults, sets up logging and job communication
        if in training mode, initializes agents, and prepares APM tracking.
        """
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

        # Holds timestamps of real actions for each bot to measure APM
        self._bot_action_timestamps = {}

        self._setup_agents()

    def _setup_agents(self):
        """
        If in training mode, obtains job details from communication.
        Otherwise, loads agent and teacher models if needed.
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

                Agent = import_module(self._cfg.agents.get(player_id, 'default'), 'Agent')
                agent = Agent(self._whole_cfg)
                agent.player_id = player_id
                agent.side_id = idx
                self.agents.append(agent)

                if agent.HAS_MODEL:
                    if player_id not in self.models:
                        if self._cfg.use_cuda:
                            assert 'test' in self._job_type, 'Only test mode supports GPU usage'
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
                                k: v
                                for k, v in loaded_state['model'].items()
                                if 'value_networks' not in k
                            }
                            agent.model.load_state_dict(state_dict, strict=False)
                        self.models[player_id] = agent.model
                    else:
                        agent.model = self.models[player_id]

            if len(map_names) == 1:
                self._whole_cfg.env.map_name = map_names[0]
            if len(map_names) == 2:
                if not (map_names[0] == 'random' and map_names[1] == 'random'):
                    self._whole_cfg.env.map_name = 'NewRepugnancy'

            if self._job_type == 'train_test':
                teacher_models = {}
                for idx, teacher_player_id in enumerate(self._cfg.teacher_player_ids):
                    if 'bot' in self._cfg.player_ids[idx]:
                        continue
                    agent = self.agents[idx]
                    agent.teacher_player_id = teacher_player_id
                    if agent.HAS_TEACHER_MODEL:
                        if teacher_player_id not in teacher_models:
                            if self._cfg.use_cuda:
                                agent.teacher_model = agent.teacher_model.cuda()
                            else:
                                agent.teacher_model = agent.teacher_model.eval()
                            if not self._cfg.fake_model:
                                loaded_teacher = torch.load(
                                    self._cfg.teacher_model_paths[teacher_player_id],
                                    map_location='cpu'
                                )
                                t_state = {
                                    k: v for k, v in loaded_teacher['model'].items()
                                    if 'value_networks' not in k
                                }
                                agent.teacher_model.load_state_dict(t_state)
                            teacher_models[teacher_player_id] = agent.teacher_model
                        else:
                            agent.teacher_model = teacher_models[teacher_player_id]

    def _inference_loop(self, env_id=0, job=None, result_queue=None, pipe_c=None):
        """
        Initializes the SC2 environment, runs episodes, and manages data
        collection or APM tracking as configured.
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

        bot_target_apm = self._cfg.get('bot_target_apm', 5)
        action_cooldown = 60.0 / bot_target_apm
        last_bot_action_time = {}

        # Minimal no-op dictionary for bots on cooldown
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

                        pid = self.agents[idx].player_id
                        if 'bot' in pid or 'model' in pid:
                            last_bot_action_time[pid] = 0.0
                            # Set up a fresh deque for APM tracking
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

                            if 'bot' in pid or 'model' in pid:
                                now_time = time.time()
                                if (now_time - last_bot_action_time[pid]) < action_cooldown:
                                    actions[player_index] = NO_OP_ACTION
                                else:
                                    real_action = agent.step(obs)
                                    actions[player_index] = real_action
                                    last_bot_action_time[pid] = now_time

                                    # Record timestamp for APM measurement
                                    self._bot_action_timestamps[pid].append(now_time)
                                    # Remove timestamps older than 60s
                                    while (
                                        self._bot_action_timestamps[pid]
                                        and (now_time - self._bot_action_timestamps[pid][0]) > 60
                                    ):
                                        self._bot_action_timestamps[pid].popleft()

                                    # Optionally log APM every so often
                                    apm_now = len(self._bot_action_timestamps[pid])
                                    self._logger.info(f"[APM] Player {pid}: {apm_now} (last 60s)")
                            else:
                                actions[player_index] = agent.step(obs)

                            agent_count += 1

                        agent_time = time.time() - agent_start
                        env_start = time.time()
                        next_obs, reward, done = self._env.step(actions)
                        env_time = time.time() - env_start
                        next_players_obs = next_obs

                        if 'train' in self._job_type:
                            post_process_time = 0
                            post_process_count = 0
                            send_data_time = 0
                            send_data_count = 0
                            for p_idx, obs_data in next_players_obs.items():
                                store_data = (
                                    self._job_type == 'train_test'
                                    or self.agents[p_idx].player_id in self._comm.job['send_data_players']
                                )
                                if store_data:
                                    t0 = time.time()
                                    traj_data = self.agents[p_idx].collect_data(
                                        next_obs[p_idx], reward[p_idx], done, p_idx
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

                        if (
                            'test' in self._whole_cfg
                            and self._whole_cfg.test.get('tb_stat', False)
                        ):
                            if not os.path.exists(self._env._result_dir):
                                os.makedirs(self._env._result_dir)
                            data = self.agents[0].get_stat_data()
                            path_file = '{}_{}_{}_.json'.format(env_id, episode_count, player_index)
                            full_path = os.path.join(self._env._result_dir, path_file)
                            with open(full_path, 'w') as f:
                                json.dump(data, f)

                        if self._job_type == 'train':
                            rand_pid = random.sample(observations.keys(), 1)[0]
                            game_steps = observations[rand_pid]['raw_obs'].observation.game_loop
                            result_info = defaultdict(dict)
                            for idx2 in range(len(self.agents)):
                                pid2 = self.agents[idx2].player_id
                                side_id2 = self.agents[idx2].side_id
                                race2 = self.agents[idx2].race
                                agent_iters = self.agents[idx2].iter_count
                                result_info[side_id2]['race'] = race2
                                result_info[side_id2]['player_id'] = pid2
                                result_info[side_id2]['opponent_id'] = self.agents[idx2].opponent_id
                                result_info[side_id2]['winloss'] = reward[idx2]
                                result_info[side_id2]['agent_iters'] = agent_iters
                                result_info[side_id2].update(self.agents[idx2].get_unit_num_info())
                                result_info[side_id2].update(self.agents[idx2].get_stat_data())

                            game_duration = time.time() - game_start
                            result_info['game_steps'] = game_steps
                            result_info['game_iters'] = game_iters
                            result_info['game_duration'] = game_duration
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

    def _gpu_inference_loop(self):
        """
        Batch inference loop for GPU usage. Periodically checks for done signals
        and updates model if in training mode.
        """
        _, _ = dist_init(method='single_node')
        torch.set_num_threads(1)

        for agent in self.agents:
            agent.model = agent.model.cuda()
            if 'train' in self._job_type:
                agent.teacher_model = agent.teacher_model.cuda()

        start_time = time.time()
        done_count = 0

        with torch.no_grad():
            while True:
                if self._job_type == 'train':
                    self._comm.async_update_model(self)
                    if time.time() - start_time > self.max_job_duration:
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

                for agent in self.agents:
                    agent.gpu_batch_inference()
                    if 'train' in self._job_type:
                        agent.gpu_batch_inference(teacher=True)

    def _start_multi_inference_loop(self):
        """
        Spawns child processes for environment stepping in parallel.
        """
        self._close_processes()
        self._processes = []
        job = self._comm.job if hasattr(self, '_comm') else {}

        self.pipes = []
        context_str = 'spawn' if platform.system().lower() == 'windows' else 'fork'
        mp_context = mp.get_context(context_str)
        self._result_queue = mp_context.Queue()

        for env_id in range(self._cfg.env_num):
            pipe_p, pipe_c = mp_context.Pipe()
            p = mp_context.Process(
                target=self._inference_loop,
                args=(env_id, job, self._result_queue, pipe_c),
                daemon=True
            )
            self.pipes.append(pipe_p)
            self._processes.append(p)
            p.start()

    def reset_env(self):
        """
        Signals all child processes to reset their environments,
        effectively restarting the episodes.
        """
        for p in self.pipes:
            p.send('reset')

    def run(self):
        """
        Determines whether to run a single-env test or multi-env train/eval flow,
        then possibly engages GPU batch inference as configured.
        """
        try:
            if 'test' in self._job_type:
                self._inference_loop()
            else:
                if self._job_type == 'train':
                    self._start_multi_inference_loop()
                    if self._gpu_batch_inference:
                        self._gpu_inference_loop()
                    else:
                        start_time = time.time()
                        while True:
                            if time.time() - start_time > self.max_job_duration:
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
        """
        Closes existing processes, obtains a new job if in training,
        and restarts multi-env inference if applicable.
        """
        self._logger.info('Actor reset multi-process.')
        self._close_processes()
        if hasattr(self, '_comm'):
            self._comm.ask_for_job(self)
        self._start_multi_inference_loop()

    def close(self):
        """
        Gracefully shuts down the actor, processes, and any open resources,
        then exits.
        """
        self._logger.info('Actor close.')
        time.sleep(2)
        if hasattr(self, '_comm'):
            self._comm.close()
        self._close_processes()
        time.sleep(1)
        os._exit(0)

    def _close_processes(self):
        """
        Sends 'close' to all child processes and waits for them to join,
        ensuring a clean shutdown.
        """
        if hasattr(self, '_processes'):
            for p in self.pipes:
                p.send('close')
            for p in self._processes:
                p.join()

    def iter_after_hook(self, iter_count, variable_record):
        """
        Periodically logs iteration stats such as agent or environment times,
        and can also track model update times if in training mode.
        """
        if iter_count % self._cfg.print_freq == 0:
            if hasattr(self, '_comm'):
                variable_record.update_var({
                    'update_model_time': self._comm._avg_update_model_time.item()
                })
            self._logger.info(
                'ACTOR({}):\n{}TimeStep{}{} {}'.format(
                    self._actor_uid,
                    '=' * 35,
                    iter_count,
                    '=' * 35,
                    variable_record.get_vars_text()
                )
            )


if __name__ == '__main__':
    actor = Actor(cfg={})
    actor.run()