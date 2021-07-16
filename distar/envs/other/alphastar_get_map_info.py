"""
Script to extract map_size from maps
Usage: Run me on a computer with ctools.pysc2 environment with ladder map packs installed
       The LOCALIZED_BNET_NAME_TO_ctools.pysc2_NAME_LUT can be used to convert any kind of map name to ctools.pysc2 name
       Output will be written to map_info.py
"""
import os
import re
from absl import app
from ctools.pysc2 import run_configs
from ctools.pysc2 import maps
from s2clientprotocol import sc2api_pb2 as sc_pb

header = '# flake8: noqa\n\
\"\"\"\n\
Generated by map_size_gen.py, containing information for each map in MAPS dict\n\
key: map_name (formatted, without spaces and LE)\n\
value: (battle_net, map_path, map_size (cropped to game boundary)(x,y),\
uncropped_map_size, localized_names, localized_names_without_spaces)\n\
\"\"\"\n\n\n\
MAPS = {\n'

trailer = '}\n\
\n\n\
def get_map_size(map_name, cropped=True):\n\
    return MAPS[map_name][2 if cropped else 3]\n\n\
def get_localized_map_name(map_name, cleared=True):\n\
    return MAPS[map_name][5 if cleared else 4]\n\n\
def get_inverse_tran_table(table, indices):\n\
    inv_tab = {}\n\
    for k, v in table.items():\n\
        inv_tab[k] = k\n\
    for index in indices:\n\
        for k, v in table.items():\n\
            if v[index]:\n\
                inv_tab[v[index]] = k\n\
    return inv_tab\n\n\
LOCALIZED_BNET_NAME_TO_ctools.pysc2_NAME_LUT = get_inverse_tran_table(MAPS, [0, 4, 5, 6])'

print(header + trailer)


def main(unused_argv):
    with open('map_info.py', 'w', encoding='utf-8') as of:
        run_config = run_configs.get()
        interface = sc_pb.InterfaceOptions(raw=True, score=False, raw_crop_to_playable_area=True)
        interface_nocrop = sc_pb.InterfaceOptions(raw=True, score=False, raw_crop_to_playable_area=False)
        of.write(header)
        formated_name_list = []
        with run_config.start(want_rgb=False) as controller:
            mps = maps.get_maps()
            print(mps)
            for fnm, mp in mps.items():
                print('Creating game')
                try:
                    mp_i = mp()
                    create = sc_pb.RequestCreateGame(realtime=True)
                    create.player_setup.add(type=sc_pb.Participant)
                    create.player_setup.add(type=sc_pb.Computer)
                    print(mp_i.path)
                    create.local_map.map_path = mp_i.path
                    join = sc_pb.RequestJoinGame(options=interface, race=2, player_name="SenseStar")
                    controller.create_game(create)
                    controller.join_game(join)
                    map_size = controller.game_info().start_raw.map_size
                    join = sc_pb.RequestJoinGame(options=interface_nocrop, race=2, player_name="SenseStar")
                    controller.create_game(create)
                    controller.join_game(join)
                    map_size_nocrop = controller.game_info().start_raw.map_size
                    try:
                        bnet_localized_map_name = controller.game_info().map_name
                        bnet_localized_map_name_cleaned = re.sub(r" ", "", bnet_localized_map_name)
                    except Exception:
                        bnet_localized_map_name = None
                        bnet_localized_map_name_cleaned = None
                except Exception as e:
                    print(e)
                    print('Starting game failed, check map file existence')
                    continue
                if fnm in formated_name_list:
                    print('Duplicate, skipping')
                    continue
                else:
                    formated_name_list.append(fnm)
                print(
                    '{}: ({}, {}, {}, {}, {}, {}, {})'.format(
                        repr(fnm), repr(mp_i.battle_net), repr(mp_i.path), repr([map_size.x, map_size.y]),
                        repr([map_size_nocrop.x, map_size_nocrop.y]), repr(bnet_localized_map_name),
                        repr(bnet_localized_map_name_cleaned),
                        repr(re.sub(r"[ ']", "", mp_i.battle_net) if mp_i.battle_net else None)
                    )
                )
                of.write(
                    '    {}: ({}, {}, {}, {}, {}, {}, {}),\n'.format(
                        repr(fnm), repr(mp_i.battle_net), repr(mp_i.path), repr([map_size.x, map_size.y]),
                        repr([map_size_nocrop.x, map_size_nocrop.y]), repr(bnet_localized_map_name),
                        repr(bnet_localized_map_name_cleaned),
                        repr(re.sub(r"[ ']", "", mp_i.battle_net) if mp_i.battle_net else None)
                    )
                )
                of.flush()
        of.write(trailer)


if __name__ == '__main__':
    app.run(main)
