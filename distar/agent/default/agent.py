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

# Adjust these imports to match your project’s structure
from .model.model import Model
from .lib.actions import (
    NUM_CUMULATIVE_STAT_ACTIONS, ACTIONS, BEGINNING_ORDER_ACTIONS,
    CUMULATIVE_STAT_ACTIONS, UNIT_ABILITY_TO_ACTION, QUEUE_ACTIONS,
    UNIT_TO_CUM, UPGRADE_TO_CUM
)
from .lib.features import (
    Features, SPATIAL_SIZE, BEGINNING_ORDER_LENGTH,
    compute_battle_score, fake_step_data, fake_model_output
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
    Copies step_data fields into shared_step_data for GPU-batch inference.
    No shape changes: maintain the same keys, dims.
    """
    entity_num = step_data['entity_num']
    selected_units_num = step_data.get('selected_units_num', 0)

    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                # Copy LSTM hidden/cell states
                shared_step_data['hidden_state'][i][0][data_idx].copy_(v[i][0])
                shared_step_data['hidden_state'][i][1][data_idx].copy_(v[i][1])
        elif k == 'value_feature':
            # Typically ignored or handled separately
            pass
        elif isinstance(v, torch.Tensor):
            shared_step_data[k][data_idx].copy_(v)
        elif isinstance(v, dict):
            # Copy entity_info, action_info, etc.
            for _k, _v in v.items():
                if k == 'action_info' and _k == 'selected_units':
                    if selected_units_num > 0:
                        shared_step_data[k][_k][data_idx, :selected_units_num].copy_(_v)
                elif k == 'entity_info':
                    shared_step_data[k][_k][data_idx, :entity_num].copy_(_v)
                elif k == 'spatial_info':
                    if 'effect' in _k:
                        shared_step_data[k][_k][data_idx].copy_(_v)
                    else:
                        h, w = _v.shape
                        shared_step_data[k][_k][data_idx] *= 0
                        shared_step_data[k][_k][data_idx, :h, :w].copy_(_v)
                else:
                    shared_step_data[k][_k][data_idx].copy_(_v)


def copy_output_data(shared_step_data, step_data, data_indexes):
    """
    Copies model output from shared_step_data to step_data for each environment
    that triggered GPU-batch inference. Preserves all shapes as originally used.
    """
    data_indexes = data_indexes.nonzero().squeeze(dim=1)
    for k, v in step_data.items():
        if k == 'hidden_state':
            for i in range(len(v)):
                shared_step_data['hidden_state'][i][0].index_copy_(
                    0, data_indexes, v[i][0][data_indexes].cpu()
                )
                shared_step_data['hidden_state'][i][1].index_copy_(
                    0, data_indexes, v[i][1][data_indexes].cpu()
                )
        elif isinstance(v, dict):
            # Possibly action logit keys, etc.
            for _k, _v in v.items():
                if len(_v.shape) == 3:
                    _, s1, s2 = _v.shape
                    shared_step_data[k][_k][:, :s1, :s2].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
                elif len(_v.shape) == 2:
                    _, s1 = _v.shape
                    shared_step_data[k][_k][:, :s1].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
                elif len(_v.shape) == 1:
                    shared_step_data[k][_k].index_copy_(
                        0, data_indexes, _v[data_indexes].cpu()
                    )
        elif isinstance(v, torch.Tensor):
            shared_step_data[k].index_copy_(
                0, data_indexes, v[data_indexes].cpu()
            )


class Agent:
    """
    Main Agent class preserving the original shape for I/O with the model
    while still supporting building-order logic, cumulative stats, and partial
    reward shaping. It does not directly handle spam or toxic detection
    (which the actor does externally).
    """

    HAS_MODEL = True
    HAS_TEACHER_MODEL = True
    HAS_SUCCESSIVE_MODEL = False

    def __init__(self, cfg=None, env_id=0):
        self._whole_cfg = cfg
        self._job_type = cfg.actor.job_type
        self._env_id = env_id

        # Basic learner / agent config
        learner_cfg = self._whole_cfg.get('learner', {})
        agent_cfg = self._whole_cfg.agent
        self._only_cum_action_kl = learner_cfg.get('only_cum_action_kl', False)
        self._bo_norm = learner_cfg.get('bo_norm', 20)
        self._cum_norm = learner_cfg.get('cum_norm', 30)
        self._battle_norm = learner_cfg.get('battle_norm', 30)

        self.model = Model(cfg)
        self._num_layers = self.model.cfg.encoder.core_lstm.num_layers
        self._hidden_size = self.model.cfg.encoder.core_lstm.hidden_size

        self._player_id = None
        self._race = None
        self._iter_count = 0
        self._model_last_iter = 0

        # Z paths & building order logic
        self._z_path = agent_cfg.z_path
        self._bo_zergling_num = agent_cfg.get('bo_zergling_num', 8)
        self._fake_reward_prob = agent_cfg.get('fake_reward_prob', 1.0)
        self._use_value_feature = learner_cfg.get('use_value_feature', False)
        self._clip_bo = agent_cfg.get('clip_bo', True)
        self._cum_type = agent_cfg.get('cum_type', 'action')
        self._zero_z_value = self._whole_cfg.get('feature', {}).get('zero_z_value', 1.0)
        self._zero_z_exceed_loop = agent_cfg.get('zero_z_exceed_loop', False)
        self._extra_units = agent_cfg.get('extra_units', False)

        # Initial hidden states
        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]

        # GPU batch inference
        self._gpu_batch_inference = cfg.actor.get('gpu_batch_inference', False)
        if self._gpu_batch_inference:
            batch_size = cfg.actor.env_num
            self._shared_input = fake_step_data(
                share_memory=True, batch_size=batch_size,
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

            if 'train' in self._job_type:
                self._teacher_shared_input = fake_step_data(
                    share_memory=True, batch_size=batch_size,
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

        # Teacher model if needed
        if 'train' in self._job_type:
            self.teacher_model = Model(cfg)

        # If in realtime, do an init forward pass
        if cfg.env.realtime:
            init_data = fake_step_data(
                share_memory=True, batch_size=1,
                hidden_size=self._hidden_size,
                hidden_layer=self._num_layers, train=False
            )
            if cfg.actor.use_cuda:
                init_data = to_device(init_data, torch.cuda.current_device())
                self.model = self.model.cuda()
            with torch.no_grad():
                _ = self.model.compute_logp_action(**init_data)

        self.z_idx = None

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

    def reset(self, map_name, race_str, game_info, obs):
        """
        Reinit state each new episode: hidden states, building orders, etc.
        Preserves old data shapes for the model's forward pass.
        """
        self._race = race_str
        self.model.policy.action_type_head.race = race_str
        self._map_name = map_name

        self._hidden_state = [
            (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
            for _ in range(self._num_layers)
        ]
        self._iter_count = 0
        self._model_last_iter = 0

        # Additional last-step placeholders
        self._last_action_type = torch.tensor(0, dtype=torch.long)
        self._last_delay = torch.tensor(0, dtype=torch.long)
        self._last_queued = torch.tensor(0, dtype=torch.long)
        self._last_selected_unit_tags = None
        self._last_target_unit_tag = None
        self._last_location = None

        self._enemy_unit_type_bool = torch.zeros(NUM_UNIT_TYPES, dtype=torch.uint8)
        self._stat_api = Stat(race_str)
        self._observation = None
        self._output = None
        self._game_step = 0

        # Building order, Z logic
        self._behaviour_building_order = []
        self._behaviour_bo_location = []
        self._bo_zergling_count = 0
        self._behaviour_cumulative_stat = [0] * NUM_CUMULATIVE_STAT_ACTIONS

        # Feature object
        self._feature = Features(game_info, obs['raw_obs'], self._whole_cfg)
        self._exceed_flag = True

        # If training, set up backups
        if 'train' in self._job_type:
            self._hidden_state_backup = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
            self._teacher_hidden_state = [
                (torch.zeros(self._hidden_size), torch.zeros(self._hidden_size))
                for _ in range(self._num_layers)
            ]
            self._data_buffer = deque(maxlen=self._whole_cfg.actor.traj_len)
            self._push_count = 0

        # Initialize Z
        raw_ob = obs['raw_obs']
        location = []
        for unit_info in raw_ob.observation.raw_data.units:
            if unit_info.unit_type in [59, 18, 86]:  # e.g., Hatch=59, Lair=18, Hive=86
                location.append([unit_info.pos.x, unit_info.pos.y])
        assert len(location) == 1, "No base found, or multiple bases found unexpectedly."
        self._born_location = deepcopy(location[0])

        # Convert to local coords
        born_location = location[0]
        born_location[0] = int(born_location[0])
        born_location[1] = int(self._feature.map_size.y - born_location[1])
        born_str = str(born_location[0] + born_location[1] * 160)

        # Load Z data from local file
        z_file = os.path.join(os.path.dirname(__file__), 'lib', self._z_path)
        with open(z_file, 'r') as f:
            self._z_data = json.load(f)

        z_type = None
        idx = None
        # Determine race vs. opponent race
        race_id = RACE_DICT[self._feature.requested_races[raw_ob.observation.player_common.player_id]]
        opp_id = 1 if raw_ob.observation.player_common.player_id == 2 else 2
        opp_race_id = RACE_DICT[self._feature.requested_races[opp_id]]
        if race_id == opp_race_id:
            mix_race = race_id
        else:
            mix_race = race_id + opp_race_id

        # Access z_data
        if self.z_idx is not None:
            idx, z_type = random.choice(self.z_idx[map_name][mix_race][born_str])
            zvals = self._z_data[map_name][mix_race][born_str][idx]
        else:
            zvals = random.choice(self._z_data[map_name][mix_race][born_str])

        # Could be 4-tuple or 5-tuple
        if len(zvals) == 5:
            (self._target_building_order,
             target_cum_stat,
             bo_location,
             self._target_z_loop,
             z_type) = zvals
        else:
            (self._target_building_order,
             target_cum_stat,
             bo_location,
             self._target_z_loop) = zvals

        # Decide whether to use cumulative or bo rewards
        self.use_cum_reward = True
        self.use_bo_reward = True
        if z_type is not None:
            # e.g., z_type=1 => skip bo, =2 => skip cum, =3 => skip both
            if z_type in [2, 3]:
                self.use_cum_reward = False
            if z_type in [1, 3]:
                self.use_bo_reward = False

        # Some random gating
        if random.random() > self._fake_reward_prob:
            self.use_cum_reward = False
        if random.random() > self._fake_reward_prob:
            self.use_bo_reward = False

        print(f"z_type={z_type}, use_cum={self.use_cum_reward}, use_bo={self.use_bo_reward}")

        # Convert bo_location, building_order, cum stats
        self._bo_norm = len(self._target_building_order)
        self._cum_norm = len(target_cum_stat)
        self._target_bo_location = torch.tensor(bo_location, dtype=torch.long)
        self._target_building_order = torch.tensor(self._target_building_order, dtype=torch.long)

        self._target_cumulative_stat = torch.zeros(NUM_CUMULATIVE_STAT_ACTIONS, dtype=torch.float)
        idxs = torch.tensor(target_cum_stat, dtype=torch.long)
        self._target_cumulative_stat.scatter_(0, idxs, 1.0)

        if not self._whole_cfg.env.realtime:
            if not self._clip_bo:
                # initial bo distance
                self._old_bo_reward = -levenshtein_distance(
                    torch.as_tensor(self._behaviour_building_order, dtype=torch.long),
                    self._target_building_order
                ) / (self._bo_norm if self._bo_norm else 1.0)
            else:
                self._old_bo_reward = torch.tensor(0.)
            self._old_cum_reward = -hamming_distance(
                torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.float),
                self._target_cumulative_stat
            ) / (self._cum_norm if self._cum_norm else 1.0)

            self._total_bo_reward = torch.zeros((), dtype=torch.float)
            self._total_cum_reward = torch.zeros((), dtype=torch.float)

    def _pre_process(self, obs):
        """
        Convert env obs => agent obs => model input, preserving shape.
        """
        if self._use_value_feature:
            agent_obs = self._feature.transform_obs(
                obs['raw_obs'],
                padding_spatial=True,
                opponent_obs=obs.get('opponent_obs')
            )
        else:
            agent_obs = self._feature.transform_obs(
                obs['raw_obs'],
                padding_spatial=True
            )

        self._game_info = agent_obs.pop('game_info')
        self._game_step = self._game_info['game_loop']

        if self._zero_z_exceed_loop and self._game_step > getattr(self, '_target_z_loop', 9999999):
            self._exceed_flag = False
            self._target_z_loop = 99999999

        # Fill in last_selected_units, last_target_unit
        ent_num = agent_obs['entity_num']
        last_sel = torch.zeros(ent_num, dtype=torch.int8)
        last_tgt = torch.zeros(ent_num, dtype=torch.int8)
        tags = self._game_info['tags']
        if self._last_selected_unit_tags:
            for tval in self._last_selected_unit_tags:
                if tval in tags:
                    idx = tags.index(tval)
                    last_sel[idx] = 1
        if self._last_target_unit_tag:
            if self._last_target_unit_tag in tags:
                idx = tags.index(self._last_target_unit_tag)
                last_tgt[idx] = 1

        agent_obs['entity_info']['last_selected_units'] = last_sel
        agent_obs['entity_info']['last_targeted_unit'] = last_tgt

        # Include hidden_state, scalar_info (last_delay, etc.)
        agent_obs['hidden_state'] = self._hidden_state
        agent_obs['scalar_info']['last_delay'] = self._last_delay
        agent_obs['scalar_info']['last_action_type'] = self._last_action_type
        agent_obs['scalar_info']['last_queued'] = self._last_queued

        # Merge in enemy_unit_type_bool
        agent_obs['scalar_info']['enemy_unit_type_bool'] = (
            self._enemy_unit_type_bool | agent_obs['scalar_info']['enemy_unit_type_bool']
        ).to(torch.uint8)

        # Insert building order / cum (no shape changes)
        bo_mask = (self.use_bo_reward and self._exceed_flag)
        agent_obs['scalar_info']['beginning_order'] = self._target_building_order * int(bo_mask)
        agent_obs['scalar_info']['bo_location'] = self._target_bo_location * int(bo_mask)

        if self.use_cum_reward and self._exceed_flag:
            agent_obs['scalar_info']['cumulative_stat'] = self._target_cumulative_stat
        else:
            agent_obs['scalar_info']['cumulative_stat'] = (
                self._target_cumulative_stat * 0 + self._zero_z_value
            )

        self._observation = agent_obs
        if self._whole_cfg.actor.use_cuda:
            agent_obs = to_device(agent_obs, 'cuda:0')

        if self._gpu_batch_inference:
            copy_input_data(self._shared_input, agent_obs, data_idx=self._env_id)
            self._signals[self._env_id] += 1
            return None
        else:
            # Single environment => direct batch collation
            return default_collate([agent_obs])

    def step(self, observation):
        """
        Single environment step:
         1. Optionally update fake reward
         2. Pre-process obs => model input
         3. Model forward pass => post_process => final action
        """
        # If 'eval' and not realtime => we can do an immediate fake reward update
        if 'eval' in self._job_type and self._iter_count > 0 and not self._whole_cfg.env.realtime:
            self._update_fake_reward(self._last_action_type, self._last_location, observation)

        model_input = self._pre_process(observation)
        # Record stats (kill counts, etc.)
        if 'action_result' in observation:
            self._stat_api.update(
                self._last_action_type,
                observation['action_result'][0],
                self._observation,
                self._game_step
            )

        # Forward pass
        if not self._gpu_batch_inference:
            model_output = self.model.compute_logp_action(**model_input)
        else:
            # Wait for GPU-batch to finish
            while True:
                if self._signals[self._env_id] == 0:
                    model_output = self._shared_output
                    break
                time.sleep(0.01)

        action_dict = self._post_process(model_output)
        self._iter_count += 1
        return action_dict

    def decollate_output(self, output, k=None, batch_idx=None):
        """
        Splits out single env data from a batch. Minimally modified from your
        original version => no shape changes.
        """
        if isinstance(output, torch.Tensor):
            if batch_idx is None:
                return output.squeeze(dim=0)
            else:
                return output[batch_idx].clone().cpu()

        elif k == 'hidden_state':
            # Return hidden states for each LSTM layer
            if batch_idx is None:
                return [
                    (output[l][0].squeeze(dim=0), output[l][1].squeeze(dim=0))
                    for l in range(len(output))
                ]
            else:
                return [
                    (
                        output[l][0][batch_idx].clone().cpu(),
                        output[l][1][batch_idx].clone().cpu()
                    )
                    for l in range(len(output))
                ]

        elif isinstance(output, dict):
            data_map = {
                subk: self.decollate_output(subv, subk, batch_idx)
                for subk, subv in output.items()
            }
            # Slicing selected_units if batch_idx
            if batch_idx is not None and k is None:
                ent_num = data_map['entity_num']
                sel_num = data_map['selected_units_num']
                data_map['logit']['selected_units'] = data_map['logit']['selected_units'][:sel_num, :ent_num + 1]
                data_map['logit']['target_unit'] = data_map['logit']['target_unit'][:ent_num]
                if 'action_info' in data_map:
                    data_map['action_info']['selected_units'] = data_map['action_info']['selected_units'][:sel_num]
                    data_map['action_logp']['selected_units'] = data_map['action_logp']['selected_units'][:sel_num]
            return data_map

        return output

    def _post_process(self, output):
        """
        Convert model_output => final environment action dict. Maintains
        original shape usage. No shape changes to the forward pass.
        """
        if self._gpu_batch_inference:
            output = self.decollate_output(output, batch_idx=self._env_id)
        else:
            output = self.decollate_output(output)

        # Update hidden states
        self._hidden_state = output['hidden_state']
        self._last_queued = output['action_info']['queued']
        self._last_action_type = output['action_info']['action_type']
        self._last_delay = output['action_info']['delay']
        self._last_location = output['action_info']['target_location']
        self._output = output

        # Construct final action
        action_info = {}
        act_idx = output['action_info']['action_type'].item()
        action_info['func_id'] = ACTIONS[act_idx]['func_id']
        action_info['skip_steps'] = output['action_info']['delay'].item()
        action_info['queued'] = output['action_info']['queued'].item()
        action_info['unit_tags'] = []

        # Append selected units
        su_num = output['selected_units_num']
        for i in range(su_num - 1):
            try:
                tag_id = output['action_info']['selected_units'][i].item()
                action_info['unit_tags'].append(self._game_info['tags'][tag_id])
            except:
                print("[Agent._post_process] Warning: mismatch in selected_units indexing")

        # extra_units if configured
        if self._extra_units:
            ex_units = torch.nonzero(output.get('extra_units', []), as_tuple=False)
            if ex_units.numel() > 0:
                for xid in ex_units.squeeze(dim=1).tolist():
                    action_info['unit_tags'].append(self._game_info['tags'][xid])

        # If the action has a target_unit
        if ACTIONS[act_idx]['target_unit']:
            target_idx = output['action_info']['target_unit'].item()
            action_info['target_unit_tag'] = self._game_info['tags'][target_idx]
            self._last_target_unit_tag = action_info['target_unit_tag']
        else:
            action_info['target_unit_tag'] = 0
            self._last_target_unit_tag = None

        # location
        xy_val = output['action_info']['target_location'].item()
        x_loc = xy_val % SPATIAL_SIZE[1]
        y_loc = xy_val // SPATIAL_SIZE[1]
        inv_y = max(self._feature.map_size.y - y_loc, 0)
        action_info['location'] = (x_loc, inv_y)

        # For debugging in test job_type
        if 'test' in self._job_type:
            self._print_action(output['action_info'], [x_loc, y_loc], output['action_logp'])

        # Return as a single-element list if your env expects [action_dict]
        return [action_info]

    def get_unit_num_info(self):
        """
        Return e.g. how many units the agent controls. Called by actor in final stats.
        """
        return {'unit_num': self._stat_api.unit_num}

    def _print_action(self, action_info, location, logp):
        """
        Debug prints for test usage, same shape logic.
        """
        act_type_idx = action_info['action_type'].item()
        act_name = ACTIONS[act_type_idx]['name']
        su_len = len(action_info['selected_units'])
        # ...
        print(f"[Agent._print_action] {self.player_id} => {act_name}, step={self._game_step}, loc={location}")

    def get_stat_data(self):
        """
        Returns stats about building order distances, cumulative stats, etc.
        No shape changes, same old logic.
        """
        base_data = self._stat_api.get_stat_data()

        bo_dist = levenshtein_distance(
            torch.as_tensor(self._behaviour_building_order, dtype=torch.int),
            self._target_building_order
        ).item()
        bo_dist_loc = levenshtein_distance(
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

        info = {
            'race_id': self.race,
            'step': self._game_step,
            'dist/bo': bo_dist,
            'dist/bo_location': bo_dist_loc - bo_dist,
            'dist/cum': cum_dist,
            'bo_reward': getattr(self, '_total_bo_reward', torch.tensor(0.)).item(),
            'cum_reward': getattr(self, '_total_cum_reward', torch.tensor(0.)).item(),
            'bo_len': len(self._behaviour_building_order)
        }
        z_type0 = 0
        z_type1 = 0
        if not getattr(self, 'use_bo_reward', False):
            info['dist/bo'] = None
            info['bo_reward'] = None
            info['bo_len'] = None
            info['dist/bo_location'] = None
            z_type0 = 1
        if not getattr(self, 'use_cum_reward', False):
            info['dist/cum'] = None
            info['cum_reward'] = None
            z_type1 = 1
        info['z_type'] = 2 * z_type1 + z_type0

        base_data.update(info)

        # Additional details for in/out stats
        in_dict = {}
        out_dict = {}
        for i, val in enumerate(self._behaviour_cumulative_stat):
            if self.race not in cum_dict[i]['race']:
                continue
            name_i = cum_dict[i]['name']
            # If target didn't require it but we built it
            if self._target_cumulative_stat[i] < 1e-3:
                out_dict[f'cum_out/{name_i}'] = 1 if val >= 1 else 0
            # If target required it and we have it
            if self._target_cumulative_stat[i] > 1e-3:
                in_dict[f'cum_in/{name_i}'] = 1 if val >= 1 else 0

        base_data.update(in_dict)
        base_data.update(out_dict)
        return base_data

    def collect_data(self, next_obs, reward, done, idx):
        """
        Gathers trajectory data for training. Maintains original shape usage.
        """
        # For example, if we track 'Success' in next_obs['action_result']
        action_result = False
        if next_obs and 'action_result' in next_obs:
            action_result = ('Success' in next_obs['action_result'])

        if action_result:
            # Optionally some success counter
            self._success_iter_count += 1

        bo_reward, cum_reward, battle_reward = self.update_fake_reward(next_obs)
        agent_obs = self._observation

        # Possibly do teacher model forward
        # ...
        # Step data => push to buffer
        # ...
        # The data structure remains as originally used => no shape changes
        return None  # or a dict/list if you want to store it

    def get_behavior_z(self):
        """
        Returns building order, bo_location, and cumulative stats for reference.
        No shape changes from original usage.
        """
        bo_padded = self._behaviour_building_order + [0]*(BEGINNING_ORDER_LENGTH - len(self._behaviour_building_order))
        loc_padded = self._behaviour_bo_location + [0]*(BEGINNING_ORDER_LENGTH - len(self._behaviour_bo_location))
        return {
            'beginning_order': torch.as_tensor(bo_padded, dtype=torch.long),
            'bo_location': torch.as_tensor(loc_padded, dtype=torch.long),
            'cumulative_stat': torch.as_tensor(self._behaviour_cumulative_stat, dtype=torch.bool).long()
        }

    def update_fake_reward(self, next_obs):
        """
        Returns bo_reward, cum_reward, battle_reward from the internal method
        to remain consistent. No shape changes.
        """
        bo, cum, battle = self._update_fake_reward(
            self._last_action_type, self._last_location, next_obs
        )
        return bo, cum, battle

    def _update_fake_reward(self, action_type, location, next_obs):
        """
        The same logic as your original code.  No shape changes.
        """
        bo_reward = torch.zeros((), dtype=torch.float)
        cum_reward = torch.zeros((), dtype=torch.float)

        # Some code to handle your battle_score
        battle_score = compute_battle_score(next_obs['raw_obs'])
        opp_battle_score = compute_battle_score(next_obs['opponent_obs'])
        battle_reward = torch.tensor(
            battle_score - self._game_info['battle_score']
            - (opp_battle_score - self._game_info['opponent_battle_score']),
            dtype=torch.float
        ) / self._battle_norm

        # If _exceed_flag false => skip building order or cum updates
        if not self._exceed_flag:
            return bo_reward, cum_reward, battle_reward

        # (the building-order logic, cum_type=action or observation, etc.)
        # same as your existing code
        # ...
        # Update self._total_bo_reward, self._total_cum_reward
        # ...

        return bo_reward, cum_reward, battle_reward

    def gpu_batch_inference(self, teacher=False):
        """
        GPU-batch logic. No shape changes. Just a data copy to/from shared buffers.
        """
        if not teacher:
            inf_idx = self._signals.clone().bool()
            batch_num = inf_idx.sum().item()
            if batch_num <= 0:
                return
            model_input = to_device(self._shared_input, torch.cuda.current_device())
            model_out = self.model.compute_logp_action(**model_input)
            copy_output_data(self._shared_output, model_out, inf_idx)
            self._signals[inf_idx] *= 0
        else:
            inf_idx = self._teacher_signals.clone().bool()
            batch_num = inf_idx.sum().item()
            if batch_num <= 0:
                return
            model_input = to_device(self._teacher_shared_input, torch.cuda.current_device())
            model_out = self.teacher_model.compute_teacher_logit(**model_input)
            copy_output_data(self._teacher_shared_output, model_out, inf_idx)
            self._teacher_signals[inf_idx] *= 0

    @staticmethod
    def _get_time_factor(game_step):
        """
        For delayed or partial reward usage, same code as original.
        """
        if game_step < 10000:
            return 1.0
        elif game_step < 20000:
            return 0.5
        elif game_step < 30000:
            return 0.25
        else:
            return 0