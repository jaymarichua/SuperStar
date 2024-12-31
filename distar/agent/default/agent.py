import copy
import json
import os
import random
import time

import torch
from copy import deepcopy
from collections import deque, defaultdict
from functools import partial
from torch.utils.data._utils.collate import default_collate

# Imports below assume your file structure matches DI-Star's conventions.
# Adjust if your layout differs.
from .model.model import Model
from .lib.actions import (
    NUM_CUMULATIVE_STAT_ACTIONS, ACTIONS, BEGINNING_ORDER_ACTIONS,
    CUMULATIVE_STAT_ACTIONS, UNIT_ABILITY_TO_ACTION, QUEUE_ACTIONS,
    UNIT_TO_CUM, UPGRADE_TO_CUM
)
from .lib.features import (
    Features, SPATIAL_SIZE, BEGINNING_ORDER_LENGTH, compute_battle_score,
    fake_step_data, fake_model_output
)
from .lib.stat import Stat, cum_dict
from distar.ctools.torch_utils.metric import (
    levenshtein_distance, hamming_distance, l2_distance
)
from distar.pysc2.lib.units import get_unit_type
from distar.pysc2.lib.static_data import UNIT_TYPES, NUM_UNIT_TYPES
from distar.ctools.torch_utils import to_device

RACE_DICT = {
    1: 'terran',
    2: 'zerg',
    3: 'protoss',
    4: 'random',
}


def copy_input_data(shared_step_data, step_data, data_idx):
    """
    Copies observation data into a shared buffer for GPU batch inference.
    Uses entity_num and selected_units_num to decide how much to copy.
    """
    entity_num = step_data['entity_num']
    selected_units_num = step_data.get('selected_units_num', 0)

    for key, val in step_data.items():
        if key == 'hidden_state':
            for i in range(len(val)):
                shared_step_data['hidden_state'][i][0][data_idx].copy_(val[i][0])
                shared_step_data['hidden_state'][i][1][data_idx].copy_(val[i][1])
            continue

        if key == 'value_feature':
            continue

        if isinstance(val, torch.Tensor):
            shared_step_data[key][data_idx].copy_(val)
        elif isinstance(val, dict):
            for sub_k, sub_v in val.items():
                if key == 'action_info' and sub_k == 'selected_units':
                    if selected_units_num > 0:
                        shared_step_data[key][sub_k][data_idx, :selected_units_num].copy_(sub_v)
                elif key == 'entity_info':
                    shared_step_data[key][sub_k][data_idx, :entity_num].copy_(sub_v)
                elif key == 'spatial_info':
                    if 'effect' in sub_k:
                        shared_step_data[key][sub_k][data_idx].copy_(sub_v)
                    else:
                        h, w = sub_v.shape
                        shared_step_data[key][sub_k][data_idx] *= 0
                        shared_step_data[key][sub_k][data_idx, :h, :w].copy_(sub_v)
                else:
                    shared_step_data[key][sub_k][data_idx].copy_(sub_v)


def copy_output_data(shared_step_data, step_data, data_indexes):
    """
    Copies model inference results back from shared memory to agent-specific data.
    """
    data_indexes = data_indexes.nonzero().squeeze(dim=1)
    for key, val in step_data.items():
        if key == 'hidden_state':
            for i in range(len(val)):
                shared_step_data['hidden_state'][i][0].index_copy_(
                    0, data_indexes, val[i][0][data_indexes].cpu()
                )
                shared_step_data['hidden_state'][i][1].index_copy_(
                    0, data_indexes, val[i][1][data_indexes].cpu()
                )
        elif isinstance(val, dict):
            for sub_k, sub_v in val.items():
                if len(sub_v.shape) == 3:
                    _, s1, s2 = sub_v.shape
                    shared_step_data[key][sub_k][:, :s1, :s2].index_copy_(
                        0, data_indexes, sub_v[data_indexes].cpu()
                    )
                elif len(sub_v.shape) == 2:
                    _, s1 = sub_v.shape
                    shared_step_data[key][sub_k][:, :s1].index_copy_(
                        0, data_indexes, sub_v[data_indexes].cpu()
                    )
                elif len(sub_v.shape) == 1:
                    shared_step_data[key][sub_k].index_copy_(
                        0, data_indexes, sub_v[data_indexes].cpu()
                    )
        elif isinstance(val, torch.Tensor):
            shared_step_data[key].index_copy_(0, data_indexes, val[data_indexes].cpu())


class Agent:
    """
    Agent class that manages:
      - An LSTM policy model
      - Building-order and cumulative-stat shaping
      - (Optional) teacher and successive models for training
      - GPU batch inference if configured
    """

    HAS_MODEL = True
    HAS_TEACHER_MODEL = True
    HAS_SUCCESSIVE_MODEL = False

    def __init__(self, cfg=None, env_id=0):
        self._whole_cfg = cfg
        self._cfg = cfg.actor
        self._job_type = self._cfg.job_type
        self._env_id = env_id

        learner_cfg = self._whole_cfg.get('learner', {})
        self._only_cum_action_kl = learner_cfg.get('only_cum_action_kl', False)
        self._z_path = self._whole_cfg.agent.z_path
        self._bo_norm = self._whole_cfg.get('learner', {}).get('bo_norm', 20)
        self._cum_norm = self._whole_cfg.get('learner', {}).get('cum_norm', 30)
        self._battle_norm = self._whole_cfg.get('learner', {}).get('battle_norm', 30)
        self._zero_z_value = self._whole_cfg.get('feature', {}).get('zero_z_value', 1.0)
        self._zero_z_exceed_loop = self._whole_cfg.agent.get('zero_z_exceed_loop', False)
        self._extra_units = self._whole_cfg.agent.get('extra_units', False)
        self._bo_zergling_num = self._whole_cfg.agent.get('bo_zergling_num', 8)
        self._fake_reward_prob = self._whole_cfg.agent.get('fake_reward_prob', 1.0)
        self._use_value_feature = self._whole_cfg.get('learner', {}).get('use_value_feature', False)
        self._clip_bo = self._whole_cfg.agent.get('clip_bo', True)
        self._cum_type = self._whole_cfg.agent.get('cum_type', 'action')

        self.model = Model(cfg)
        self._num_layers = self.model.cfg.encoder.core_lstm.num_layers
        self._hidden_size = self.model.cfg.encoder.core_lstm.hidden_size

        self._player_id = None
        self._race = None
        self._iter_count = 0
        self._model_last_iter = 0
        self._success_iter_count = 0
        self._enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)

        if 'train' in self._job_type and self.HAS_TEACHER_MODEL:
            self.teacher_model = Model(cfg)
        else:
            self.teacher_model = None

        self._use_dapo = learner_cfg.get('use_dapo', False)
        if 'train' in self._job_type and self._use_dapo and self.HAS_SUCCESSIVE_MODEL:
            self.successive_model = Model(cfg)
            self._successive_hidden_state = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
        else:
            self.successive_model = None
            self._successive_hidden_state = None

        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]
        self.z_idx = None
        self._gpu_batch_inference = self._cfg.get('gpu_batch_inference', False)
        if self._gpu_batch_inference:
            batch_size = self._cfg.env_num
            self._shared_input = fake_step_data(
                share_memory=True,
                batch_size=batch_size,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers,
                train=False
            )
            self._shared_output = fake_model_output(
                batch_size=batch_size,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers,
                teacher=False
            )
            self._signals = torch.zeros(batch_size).share_memory_()

            if 'train' in self._job_type and self.teacher_model is not None:
                self._teacher_shared_input = fake_step_data(
                    share_memory=True,
                    batch_size=batch_size,
                    hidden_size=self._hidden_size,
                    hidden_layer=self._num_layers,
                    train=True
                )
                self._teacher_shared_output = fake_model_output(
                    batch_size=batch_size,
                    hidden_size=self._hidden_size,
                    hidden_layer=self._num_layers,
                    teacher=True
                )
                self._teacher_signals = torch.zeros(batch_size).share_memory_()

        if self._whole_cfg.env.realtime:
            init_data = fake_step_data(
                share_memory=True, batch_size=1,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers,
                train=False
            )
            if self._cfg.use_cuda:
                init_data = to_device(init_data, torch.cuda.current_device())
                self.model = self.model.cuda()
            with torch.no_grad():
                _ = self.model.compute_logp_action(**init_data)

    @property
    def env_id(self):
        return self._env_id

    @env_id.setter
    def env_id(self, val):
        self._env_id = val

    @property
    def player_id(self):
        return self._player_id

    @player_id.setter
    def player_id(self, val):
        self._player_id = val

    @property
    def race(self):
        return self._race

    @property
    def iter_count(self):
        return self._iter_count

    def reset(self, map_name, race, game_info, obs):
        self._race = race
        self._map_name = map_name
        self._iter_count = 0
        self._model_last_iter = 0
        self._success_iter_count = 0
        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]
        self._last_action_type = torch.tensor(0, dtype=torch.long)
        self._last_delay = torch.tensor(0, dtype=torch.long)
        self._last_queued = torch.tensor(0, dtype=torch.long)
        self._last_selected_unit_tags = None
        self._last_target_unit_tag = None
        self._last_location = None
        self._exceed_flag = True
        self._game_step = 0
        self._enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)

        self._behaviour_building_order = []
        self._behaviour_bo_location = []
        self._bo_zergling_count = 0
        self._behaviour_cumulative_stat = [0] * NUM_CUMULATIVE_STAT_ACTIONS

        if 'train' in self._job_type:
            self._hidden_state_backup = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
            if self.teacher_model is not None:
                self._teacher_hidden_state = [
                    (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                    for _ in range(self._num_layers)
                ]
            if self.successive_model is not None:
                self._successive_hidden_state = [
                    (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                    for _ in range(self._num_layers)
                ]
            self._data_buffer = deque(maxlen=self._cfg.traj_len)
            self._push_count = 0

        self._stat_api = Stat(race)
        self._feature = Features(game_info, obs['raw_obs'], self._whole_cfg)

        raw_ob = obs['raw_obs']
        base_locs = []
        for unit in raw_ob.observation.raw_data.units:
            if unit.unit_type in [59, 18, 86]:
                base_locs.append([unit.pos.x, unit.pos.y])
        assert len(base_locs) == 1, "Expected exactly one base location."

        self._born_location = deepcopy(base_locs[0])
        self._born_location[0] = int(self._born_location[0])
        self._born_location[1] = int(self._feature.map_size.y - self._born_location[1])
        born_loc_str = str(self._born_location[0] + self._born_location[1] * 160)

        z_file_path = os.path.join(os.path.dirname(__file__), 'lib', self._z_path)
        with open(z_file_path, 'r') as z_f:
            self._z_data = json.load(z_f)

        player_race_id = self._feature.requested_races[raw_ob.observation.player_common.player_id]
        opp_id = 1 if raw_ob.observation.player_common.player_id == 2 else 2
        opp_race_id = self._feature.requested_races[opp_id]
        self_race_str = RACE_DICT[player_race_id]
        opp_race_str = RACE_DICT[opp_race_id]
        if self_race_str == opp_race_str:
            mix_race = self_race_str
        else:
            # fallback if partial coverage
            mix_race = self_race_str + opp_race_str
            if mix_race not in self._z_data.get(map_name, {}):
                mix_race = self_race_str

        z_type = None
        if (self.z_idx is not None and
                self._map_name in self.z_idx and
                mix_race in self.z_idx[self._map_name] and
                born_loc_str in self.z_idx[self._map_name][mix_race]):
            pick_list = self.z_idx[self._map_name][mix_race][born_loc_str]
            idx, z_type = random.choice(pick_list)
            z = self._z_data[self._map_name][mix_race][born_loc_str][idx]
        else:
            if born_loc_str not in self._z_data.get(map_name, {}).get(mix_race, {}):
                print("[WARNING] No advanced Z data found => skipping.")
                return
            picks = self._z_data[map_name][mix_race][born_loc_str]
            z = random.choice(picks)

        if len(z) == 5:
            bo_list, cum_list, bo_loc_list, self._target_z_loop, z_type = z
        else:
            bo_list, cum_list, bo_loc_list, self._target_z_loop = z
        self.use_cum_reward = True
        self.use_bo_reward = True
        if z_type is not None:
            if z_type == 2 or z_type == 3:
                self.use_cum_reward = False
            if z_type == 1 or z_type == 3:
                self.use_bo_reward = False
        if random.random() > self._fake_reward_prob:
            self.use_cum_reward = False
        if random.random() > self._fake_reward_prob:
            self.use_bo_reward = False

        print(f"[Z] z_type={z_type}, use_cum={self.use_cum_reward}, use_bo={self.use_bo_reward}")

        bo_tensor = torch.tensor(bo_list, dtype=torch.long)
        bo_loc_tensor = torch.tensor(bo_loc_list, dtype=torch.long)
        cum_tensor = torch.zeros(NUM_CUMULATIVE_STAT_ACTIONS, dtype=torch.float)
        if len(cum_list) > 0:
            cum_idx = torch.tensor(cum_list, dtype=torch.long)
            cum_tensor.scatter_(0, cum_idx, 1.0)

        self._target_building_order = bo_tensor
        self._target_bo_location = bo_loc_tensor
        self._target_cumulative_stat = cum_tensor
        self._bo_norm = len(bo_list)
        self._cum_norm = len(cum_list)

        if not self._whole_cfg.env.realtime:
            if not self._clip_bo:
                init_bo_dist = -levenshtein_distance(
                    torch.as_tensor(self._behaviour_building_order, dtype=torch.long),
                    bo_tensor
                ) / (self._bo_norm if self._bo_norm else 1.0)
                self._old_bo_reward = init_bo_dist
            else:
                self._old_bo_reward = torch.tensor(0.0)

            init_cum_dist = -hamming_distance(
                torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.float),
                cum_tensor
            ) / (self._cum_norm if self._cum_norm else 1.0)
            self._old_cum_reward = init_cum_dist
            self._total_bo_reward = torch.tensor(0.0)
            self._total_cum_reward = torch.tensor(0.0)

    def _pre_process(self, obs):
        if self._use_value_feature:
            agent_obs = self._feature.transform_obs(
                obs['raw_obs'],
                padding_spatial=True,
                opponent_obs=obs.get('opponent_obs', None)
            )
        else:
            agent_obs = self._feature.transform_obs(obs['raw_obs'], padding_spatial=True)

        self._game_info = agent_obs.pop('game_info')
        self._game_step = self._game_info['game_loop']

        if self._zero_z_exceed_loop and self._game_step > self._target_z_loop:
            self._exceed_flag = False
            self._target_z_loop = 99999999

        ent_num = agent_obs['entity_num']
        last_sel_vec = torch.zeros(ent_num, dtype=torch.int8)
        last_tgt_vec = torch.zeros(ent_num, dtype=torch.int8)
        tag_list = self._game_info['tags']

        if self._last_selected_unit_tags is not None:
            for t_val in self._last_selected_unit_tags:
                if t_val in tag_list:
                    idx_val = tag_list.index(t_val)
                    last_sel_vec[idx_val] = 1

        if self._last_target_unit_tag is not None and self._last_target_unit_tag in tag_list:
            idx_val = tag_list.index(self._last_target_unit_tag)
            last_tgt_vec[idx_val] = 1

        agent_obs['entity_info']['last_selected_units'] = last_sel_vec
        agent_obs['entity_info']['last_targeted_unit'] = last_tgt_vec

        agent_obs['hidden_state'] = self._hidden_state
        agent_obs['scalar_info']['last_delay'] = self._last_delay
        agent_obs['scalar_info']['last_action_type'] = self._last_action_type
        agent_obs['scalar_info']['last_queued'] = self._last_queued

        new_enemy_bool = (self._enemy_unit_type_bool | agent_obs['scalar_info']['enemy_unit_type_bool'])
        agent_obs['scalar_info']['enemy_unit_type_bool'] = new_enemy_bool.to(torch.uint8)

        use_bo = (self.use_bo_reward and self._exceed_flag)
        agent_obs['scalar_info']['beginning_order'] = self._target_building_order * int(use_bo)
        agent_obs['scalar_info']['bo_location'] = self._target_bo_location * int(use_bo)

        if self.use_cum_reward and self._exceed_flag:
            agent_obs['scalar_info']['cumulative_stat'] = self._target_cumulative_stat
        else:
            agent_obs['scalar_info']['cumulative_stat'] = self._target_cumulative_stat * 0 + self._zero_z_value

        self._observation = agent_obs

        if self._cfg.use_cuda:
            agent_obs = to_device(agent_obs, 'cuda:0')

        if self._gpu_batch_inference:
            copy_input_data(self._shared_input, agent_obs, data_idx=self._env_id)
            self._signals[self._env_id] += 1
            return None
        else:
            return default_collate([agent_obs])

    def step(self, observation):
        if 'eval' in self._job_type and self._iter_count > 0 and not self._whole_cfg.env.realtime:
            self._update_fake_reward(self._last_action_type, self._last_location, observation)

        model_input = self._pre_process(observation)
        self._stat_api.update(
            self._last_action_type,
            observation['action_result'][0],
            self._observation,
            self._game_step
        )

        if not self._gpu_batch_inference:
            model_output = self.model.compute_logp_action(**model_input)
        else:
            while True:
                if self._signals[self._env_id] == 0:
                    model_output = self._shared_output
                    break
                time.sleep(0.01)

        final_action = self._post_process(model_output)
        self._iter_count += 1
        return final_action

    def decollate_output(self, output, k=None, batch_idx=None):
        if isinstance(output, torch.Tensor):
            if batch_idx is None:
                return output.squeeze(dim=0)
            else:
                return output[batch_idx].clone().cpu()
        if k == 'hidden_state':
            if batch_idx is None:
                return [
                    (output[i][0].squeeze(dim=0), output[i][1].squeeze(dim=0))
                    for i in range(len(output))
                ]
            else:
                return [
                    (
                        output[i][0][batch_idx].clone().cpu(),
                        output[i][1][batch_idx].clone().cpu()
                    )
                    for i in range(len(output))
                ]
        if isinstance(output, dict):
            new_data = {}
            for sub_k, sub_v in output.items():
                new_data[sub_k] = self.decollate_output(sub_v, k, batch_idx)
            if batch_idx is not None and k is None:
                ent_num = new_data['entity_num']
                sel_units_num = new_data['selected_units_num']
                new_data['logit']['selected_units'] = (
                    new_data['logit']['selected_units'][:sel_units_num, :ent_num + 1]
                )
                new_data['logit']['target_unit'] = new_data['logit']['target_unit'][:ent_num]
                if 'action_info' in new_data:
                    new_data['action_info']['selected_units'] = (
                        new_data['action_info']['selected_units'][:sel_units_num]
                    )
                    new_data['action_logp']['selected_units'] = (
                        new_data['action_logp']['selected_units'][:sel_units_num]
                    )
            return new_data
        return output

    def _post_process(self, output):
        if self._gpu_batch_inference:
            out_data = self.decollate_output(output, batch_idx=self._env_id)
        else:
            out_data = self.decollate_output(output)

        self._hidden_state = out_data['hidden_state']
        self._last_queued = out_data['action_info']['queued']
        self._last_action_type = out_data['action_info']['action_type']
        self._last_delay = out_data['action_info']['delay']
        self._last_location = out_data['action_info']['target_location']
        self._output = out_data

        action_dict = {}
        action_idx = out_data['action_info']['action_type'].item()
        action_dict['func_id'] = ACTIONS[action_idx]['func_id']
        action_dict['skip_steps'] = out_data['action_info']['delay'].item()
        action_dict['queued'] = out_data['action_info']['queued'].item()
        action_dict['unit_tags'] = []

        sel_num = out_data['selected_units_num']
        for i in range(sel_num - 1):
            try:
                tag_idx = out_data['action_info']['selected_units'][i].item()
                action_dict['unit_tags'].append(self._game_info['tags'][tag_idx])
            except:
                print("[WARN] Missing selected unit index in _post_process")

        if self._extra_units:
            extra_tensor = out_data.get('extra_units', [])
            if isinstance(extra_tensor, torch.Tensor):
                ex_units = torch.nonzero(extra_tensor, as_tuple=False)
                if ex_units.numel() > 0:
                    for e_idx in ex_units.squeeze(dim=1).tolist():
                        action_dict['unit_tags'].append(self._game_info['tags'][e_idx])

        target_u_idx = out_data['action_info']['target_unit'].item()
        if ACTIONS[action_idx]['target_unit']:
            action_dict['target_unit_tag'] = self._game_info['tags'][target_u_idx]
            self._last_target_unit_tag = action_dict['target_unit_tag']
        else:
            action_dict['target_unit_tag'] = 0
            self._last_target_unit_tag = None

        loc_item = out_data['action_info']['target_location'].item()
        x_val = loc_item % SPATIAL_SIZE[1]
        y_val = loc_item // SPATIAL_SIZE[1]
        inv_y = max(self._feature.map_size.y - y_val, 0)
        action_dict['location'] = (x_val, inv_y)

        if 'test' in self._job_type:
            self._print_action(out_data['action_info'], [x_val, y_val], out_data['action_logp'])

        return [action_dict]

    def get_unit_num_info(self):
        return {'unit_num': self._stat_api.unit_num}

    def _print_action(self, action_info, location_xy, logp):
        a_type_idx = action_info['action_type'].item()
        a_name = ACTIONS[a_type_idx]['name']
        selected_units_list = ''
        su_len = len(action_info['selected_units'])
        if ACTIONS[a_type_idx]['selected_units']:
            for i, su_idx in enumerate(action_info['selected_units'][:-1].tolist()):
                unit_type_idx = self._observation['entity_info']['unit_type'][su_idx]
                ut_str = str(get_unit_type(UNIT_TYPES[unit_type_idx])).split('.')[-1]
                prob_v = torch.exp(logp['selected_units'][i]).item()
                selected_units_list += f" {ut_str}({prob_v:.2f})"
            final_su_prob = torch.exp(logp['selected_units'][-1]).item()
            selected_units_list += f" end({final_su_prob:.2f})"

        target_unit_str = ''
        if ACTIONS[a_type_idx]['target_unit']:
            t_u_idx = action_info['target_unit'].item()
            t_utype = self._observation['entity_info']['unit_type'][t_u_idx]
            target_unit_str = str(get_unit_type(UNIT_TYPES[t_utype])).split('.')[-1]

        main_logp = torch.exp(logp['action_type']).item()
        delay_logp = torch.exp(logp['delay']).item()
        loc_logp = torch.exp(logp['target_location']).item()
        t_unit_logp = torch.exp(logp['target_unit']).item()

        debug_line = (
            f"{self.player_id}, game_step:{self._game_step}, "
            f"at:{a_name}({main_logp:.2f}), delay:{action_info['delay']}({delay_logp:.2f}), "
            f"su:({su_len}){selected_units_list}, tu:{target_unit_str}({t_unit_logp:.2f}), "
            f"lo:{location_xy}({loc_logp:.2f})"
        )
        print(debug_line)

    def get_stat_data(self):
        data = self._stat_api.get_stat_data()

        bo_dist = levenshtein_distance(
            torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
            self._target_building_order
        ).item()

        bo_loc_dist = levenshtein_distance(
            torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
            self._target_building_order,
            torch.as_tensor(self._behaviour_bo_location, dtype=torch.int),
            self._target_bo_location,
            partial(l2_distance, spatial_x=SPATIAL_SIZE[1])
        ).item()

        cum_dist = hamming_distance(
            torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool),
            self._target_cumulative_stat.to(torch.bool)
        ).item()

        stats_dict = {
            'race_id': self._race,
            'step': self._game_step,
            'dist/bo': bo_dist,
            'dist/bo_location': bo_loc_dist - bo_dist,
            'dist/cum': cum_dist,
            'bo_reward': self._total_bo_reward.item(),
            'cum_reward': self._total_cum_reward.item(),
            'bo_len': len(self._behaviour_building_order)
        }

        z_type_0, z_type_1 = 0, 0
        if not self.use_bo_reward:
            stats_dict['dist/bo'] = None
            stats_dict['bo_reward'] = None
            stats_dict['bo_len'] = None
            stats_dict['dist/bo_location'] = None
            z_type_0 = 1
        if not self.use_cum_reward:
            stats_dict['dist/cum'] = None
            stats_dict['cum_reward'] = None
            z_type_1 = 1
        stats_dict['z_type'] = 2 * z_type_1 + z_type_0
        data.update(stats_dict)

        cum_in = defaultdict(int)
        cum_out = defaultdict(int)
        for i in range(len(self._behaviour_cumulative_stat)):
            if self._race not in cum_dict[i]['race']:
                continue
            c_name = cum_dict[i]['name']
            if self._target_cumulative_stat[i] < 1e-3:
                if self._behaviour_cumulative_stat[i] >= 1:
                    cum_out['cum_out/' + c_name] = 1
                else:
                    cum_out['cum_out/' + c_name] = 0
            else:
                if self._behaviour_cumulative_stat[i] >= 1:
                    cum_in['cum_in/' + c_name] = 1
                else:
                    cum_in['cum_in/' + c_name] = 0
        data.update(cum_in)
        data.update(cum_out)
        return data

    def collect_data(self, next_obs, reward, done, idx):
        action_result = False if next_obs is None else ('Success' in next_obs['action_result'])
        if action_result:
            self._success_iter_count += 1

        behavior_z = self.get_behavior_z()
        bo_reward, cum_reward, battle_reward = self.update_fake_reward(next_obs)
        agent_obs = self._observation

        teacher_logits = None
        if self.teacher_model is not None:
            teacher_obs = {
                'spatial_info': agent_obs['spatial_info'],
                'entity_info': agent_obs['entity_info'],
                'scalar_info': agent_obs['scalar_info'],
                'entity_num': agent_obs['entity_num'],
                'hidden_state': self._teacher_hidden_state if self.teacher_model else None,
                'selected_units_num': self._output['selected_units_num'],
                'action_info': self._output['action_info']
            }
            if self._cfg.use_cuda:
                teacher_obs = to_device(teacher_obs, 'cuda:0')
            if self._gpu_batch_inference:
                copy_input_data(self._teacher_shared_input, teacher_obs, data_idx=self._env_id)
                self._teacher_signals[self._env_id] += 1
                while True:
                    if self._teacher_signals[self._env_id] == 0:
                        teacher_output = self._teacher_shared_output
                        teacher_output = self.decollate_output(
                            teacher_output, batch_idx=self._env_id
                        )
                        break
                    time.sleep(0.01)
            else:
                teacher_input = default_collate([teacher_obs])
                t_out = self.teacher_model.compute_teacher_logit(**teacher_input)
                teacher_output = self.decollate_output(t_out)
            self._teacher_hidden_state = teacher_output['hidden_state']
            teacher_logits = teacher_output['logit']

        successive_logits = None
        if self.successive_model is not None:
            s_obs = deepcopy(agent_obs)
            s_obs['hidden_state'] = self._successive_hidden_state
            s_obs['selected_units_num'] = self._output['selected_units_num']
            s_obs['action_info'] = self._output['action_info']
            s_input = default_collate([s_obs])
            s_out = self.successive_model.compute_teacher_logit(**s_input)
            s_out = self.decollate_output(s_out)
            self._successive_hidden_state = s_out['hidden_state']
            if 'logit' in s_out:
                successive_logits = s_out['logit']

        action_info = deepcopy(self._output['action_info'])
        action_mask = copy.deepcopy({
            k: v for k, v in ACTIONS[action_info['action_type'].item()].items()
            if k not in ['name', 'goal', 'func_id', 'general_ability_id', 'game_id']
        })
        mask = {}
        mask['actions_mask'] = {}
        for ak, av in action_mask.items():
            mask['actions_mask'][ak] = torch.tensor(av, dtype=torch.long)

        if self._only_cum_action_kl:
            mask['cum_action_mask'] = torch.tensor(0.0, dtype=torch.float)
        else:
            mask['cum_action_mask'] = torch.tensor(1.0, dtype=torch.float)

        if self.use_bo_reward:
            mask['build_order_mask'] = torch.tensor(1.0, dtype=torch.float)
        else:
            mask['build_order_mask'] = torch.tensor(0.0, dtype=torch.float)

        if self.use_cum_reward:
            mask['built_unit_mask'] = torch.tensor(1.0, dtype=torch.float)
            mask['cum_action_mask'] = torch.tensor(1.0, dtype=torch.float)
        else:
            mask['built_unit_mask'] = torch.tensor(0.0, dtype=torch.float)

        step_data = {
            'map_name': self._map_name,
            'spatial_info': agent_obs['spatial_info'],
            'model_last_iter': torch.tensor(self._model_last_iter, dtype=torch.float),
            'entity_info': agent_obs['entity_info'],
            'scalar_info': agent_obs['scalar_info'],
            'entity_num': agent_obs['entity_num'],
            'selected_units_num': self._output['selected_units_num'],
            'hidden_state': self._hidden_state_backup,
            'action_info': action_info,
            'behaviour_logp': self._output['action_logp'],
            'teacher_logit': teacher_logits,
            'reward': {
                'winloss': torch.tensor(reward, dtype=torch.float),
                'build_order': bo_reward,
                'built_unit': cum_reward,
                'battle': battle_reward
            },
            'step': torch.tensor(self._game_step, dtype=torch.float),
            'mask': mask
        }
        if successive_logits is not None:
            step_data['successive_logit'] = successive_logits

        if self._use_value_feature and 'value_feature' in agent_obs:
            step_data['value_feature'] = agent_obs['value_feature']
            step_data['value_feature'].update(behavior_z)

        self._hidden_state_backup = self._hidden_state
        self._data_buffer.append(step_data)
        self._push_count += 1

        if self._push_count == self._cfg.traj_len or done:
            if not done:
                if not next_obs['raw_obs'].observation:
                    return None
                self._pre_process(next_obs)
                n_obs = deepcopy(self._observation)
            else:
                n_obs = deepcopy(self._observation)

            final_step_data = {
                'map_name': self._map_name,
                'spatial_info': n_obs['spatial_info'],
                'entity_info': n_obs['entity_info'],
                'scalar_info': n_obs['scalar_info'],
                'entity_num': n_obs['entity_num'],
                'hidden_state': self._hidden_state
            }
            if self._use_value_feature and 'value_feature' in n_obs:
                final_step_data['value_feature'] = n_obs['value_feature']
                final_step_data['value_feature'].update(self.get_behavior_z())

            buf_list = list(self._data_buffer)
            buf_list.append(final_step_data)
            self._push_count = 0
            return buf_list
        else:
            return None

    def get_behavior_z(self):
        bo_len_pad = BEGINNING_ORDER_LENGTH - len(self._behaviour_building_order)
        bo_loc_len_pad = BEGINNING_ORDER_LENGTH - len(self._behaviour_bo_location)
        bo_seq = self._behaviour_building_order + [0] * bo_len_pad
        bo_loc_seq = self._behaviour_bo_location + [0] * bo_loc_len_pad
        return {
            'beginning_order': torch.as_tensor(bo_seq, dtype=torch.long),
            'bo_location': torch.as_tensor(bo_loc_seq, dtype=torch.long),
            'cumulative_stat': torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool).long()
        }

    def update_fake_reward(self, next_obs):
        bo_r, cum_r, battle_r = self._update_fake_reward(self._last_action_type, self._last_location, next_obs)
        return bo_r, cum_r, battle_r

    def _update_fake_reward(self, action_type, location, next_obs):
        bo_reward = torch.zeros(size=(), dtype=torch.float)
        cum_reward = torch.zeros(size=(), dtype=torch.float)

        agent_battle = compute_battle_score(next_obs['raw_obs'])
        opp_battle = compute_battle_score(next_obs['opponent_obs'])
        net_battle = (agent_battle - self._game_info['battle_score']) - (opp_battle - self._game_info['opponent_battle_score'])
        battle_reward = torch.tensor(net_battle, dtype=torch.float) / self._battle_norm

        if not self._exceed_flag:
            return bo_reward, cum_reward, battle_reward

        if action_type in BEGINNING_ORDER_ACTIONS and next_obs['action_result'][0] == 1:
            if action_type == 322:
                self._bo_zergling_count += 1
                if self._bo_zergling_count > self._bo_zergling_num:
                    return bo_reward, cum_reward, battle_reward

            order_idx = BEGINNING_ORDER_ACTIONS.index(action_type)
            if order_idx == 39 and 39 not in self._target_building_order:
                return bo_reward, cum_reward, battle_reward

            if len(self._behaviour_building_order) < len(self._target_building_order):
                self._behaviour_building_order.append(order_idx)
                if ACTIONS[action_type]['target_location']:
                    self._behaviour_bo_location.append(location.item())
                else:
                    self._behaviour_bo_location.append(0)

                if self.use_bo_reward:
                    if self._clip_bo:
                        partial_bo = self._target_building_order[:len(self._behaviour_building_order)]
                        partial_bo_loc = self._target_bo_location[:len(self._behaviour_building_order)]
                    else:
                        partial_bo = self._target_building_order
                        partial_bo_loc = self._target_bo_location
                    new_bo_dist = -levenshtein_distance(
                        torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
                        torch.as_tensor(partial_bo, dtype=torch.int),
                        torch.as_tensor(self._behaviour_bo_location, dtype=torch.int),
                        torch.as_tensor(partial_bo_loc, dtype=torch.int),
                        partial(l2_distance, spatial_x=SPATIAL_SIZE[1])
                    ) / (self._bo_norm if self._bo_norm else 1.0)
                    bo_reward = new_bo_dist - self._old_bo_reward
                    self._old_bo_reward = new_bo_dist

        cum_flag = False
        if self._cum_type == 'observation':
            cum_flag = True
            for unit_obj in next_obs['raw_obs'].observation.raw_data.units:
                if unit_obj.alliance == 1 and unit_obj.build_progress == 1:
                    idx_val = UNIT_TO_CUM.get(unit_obj.unit_type, -1)
                    if idx_val != -1:
                        # ignore initial base
                        if unit_obj.unit_type in [59, 18, 86]:
                            if (int(unit_obj.pos.x) == self._born_location[0]
                                    and int(unit_obj.pos.y) == (self._feature.map_size.y - self._born_location[1])):
                                continue
                        self._behaviour_cumulative_stat[idx_val] = 1

            for up_id in next_obs['raw_obs'].observation.raw_data.player.upgrade_ids:
                up_idx = UPGRADE_TO_CUM.get(up_id, -1)
                if up_idx != -1:
                    self._behaviour_cumulative_stat[up_idx] = 1

        elif self._cum_type == 'action':
            cum_flag = False
            a_name = ACTIONS[action_type]['name']
            action_info = self._output['action_info']
            if a_name in ['Cancel_quick', 'Cancel_Last_quick']:
                unit_idx = action_info['selected_units'][0].item()
                o_len = self._observation['entity_info']['order_length'][unit_idx]
                if o_len == 0:
                    act_idx = 0
                elif o_len == 1:
                    act_idx = UNIT_ABILITY_TO_ACTION.get(
                        self._observation['entity_info']['order_id_0'][unit_idx].item(), 0
                    )
                else:
                    key_str = f'order_id_{o_len - 1}'
                    queue_idx = self._observation['entity_info'][key_str][unit_idx].item() - 1
                    if queue_idx >= 0:
                        act_idx = QUEUE_ACTIONS[queue_idx]
                    else:
                        act_idx = 0

                if act_idx in CUMULATIVE_STAT_ACTIONS:
                    cum_flag = True
                    c_idx = CUMULATIVE_STAT_ACTIONS.index(act_idx)
                    self._behaviour_cumulative_stat[c_idx] = max(0, self._behaviour_cumulative_stat[c_idx] - 1)

            if action_type in CUMULATIVE_STAT_ACTIONS:
                cum_flag = True
                c_idx = CUMULATIVE_STAT_ACTIONS.index(action_type)
                self._behaviour_cumulative_stat[c_idx] += 1
        else:
            raise NotImplementedError(f"Unrecognized cum_type: {self._cum_type}")

        if self.use_cum_reward and cum_flag and (
            self._cum_type == 'observation' or next_obs['action_result'][0] == 1
        ):
            new_cum_dist = -hamming_distance(
                torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool),
                self._target_cumulative_stat.to(torch.bool)
            ) / (self._cum_norm if self._cum_norm else 1.0)
            c_reward = (new_cum_dist - self._old_cum_reward) * self._get_time_factor(self._game_step)
            cum_reward = c_reward
            self._old_cum_reward = new_cum_dist

        self._total_bo_reward += bo_reward
        self._total_cum_reward += cum_reward
        return bo_reward, cum_reward, battle_reward

    def gpu_batch_inference(self, teacher=False):
        if not teacher:
            indices = self._signals.clone().bool()
            count_num = indices.sum()
            if count_num == 0:
                return

            model_inp = to_device(self._shared_input, torch.cuda.current_device())
            outp = self.model.compute_logp_action(**model_inp)
            copy_output_data(self._shared_output, outp, indices)
            self._signals[indices] *= 0
        else:
            if self.teacher_model is None:
                return
            indices = self._teacher_signals.clone().bool()
            count_num = indices.sum()
            if count_num == 0:
                return

            model_inp = to_device(self._teacher_shared_input, torch.cuda.current_device())
            outp = self.teacher_model.compute_teacher_logit(**model_inp)
            copy_output_data(self._teacher_shared_output, outp, indices)
            self._teacher_signals[indices] *= 0

    @staticmethod
    def _get_time_factor(game_step):
        if game_step < 10000:
            return 1.0
        elif game_step < 20000:
            return 0.5
        elif game_step < 30000:
            return 0.25
        return 0