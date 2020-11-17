# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
util_common
"""
import os
import json
from te.utils.error_manager import error_manager_util
from te import platform as tbe_platform


PAD_MIN = 0
# the dim of most parameters in conv3d is 5
CONV3D_SHAPE_COMMON_DIM = 5


def ceil(x_1, x_2):
    """
    do ceiling division

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return (x_1 + x_2 - 1) // x_2


def check_pads_value_3d(pads):
    pad_head, pad_tail, pad_up, pad_down, pad_left, pad_right = pads
    if pad_head < PAD_MIN or pad_tail < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
        'param_name': 'pad D',
        'expected_value': 'non_negative vlaue',
        'input_value': 'pad_d[0] = {}, pad_d[1] = {}'.format(pad_head, pad_tail)
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if pad_up < PAD_MIN or pad_down < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pad H',
            'expected_value': 'non_negative vlaue',
            'input_value': 'pad_h[0] = {}, pad_h[1] = {}'.format(pad_up, pad_down)
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
    if pad_left < PAD_MIN or pad_right < PAD_MIN:
        dict_args = {
            'errCode': 'E60000',
            'param_name': 'pad W',
            'expected_value': 'non_negative vlaue',
            'input_value': 'pad_w[0] = {}, pad_w[1] = {}'.format(pad_left, pad_right)
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))


def align(x_1, x_2):
    """
    align x_1 with x_2

    Parameters
    ----------
    x_1: int
    x_2: int
    Returns
    -------
    result
    """
    if x_2 == 0:
        dict_args = {
            'errCode': 'E62502',
            'first_operand': str(x_1),
            'second_operand': str(x_2),
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return ((x_1 + x_2 - 1) // x_2) * x_2


def write_code(wkspace_dict, kernel_name):
    """
    write workspaces to json file

    """
    fname = tbe_platform.cce_conf.get_kernel_meta_dir() + "/" + kernel_name + ".json"
    fname = os.path.realpath(fname)
    if fname.startswith(os.getcwd()):
        if os.path.exists(fname):
            with open(fname, "r") as f_var:
                load_dict = json.load(f_var)
            load_dict.update(wkspace_dict)
            with open(fname, "w") as f_var:
                json.dump(load_dict, f_var, sort_keys=True,
                          indent=4, separators=(',', ':'))


def lcm(param1, param2):
    """
    calculate least common multiple
    """
    temp = param1 * param2
    while param1 % param2 != 0:
        param1, param2 = param2, param1 % param2

    return temp // param2


def calculate_group(fmap_c, cout, groups, cout0, cin0):
    """
    calculate groups parameter
    """
    mag_factor0 = lcm(fmap_c // groups, cin0) // (fmap_c // groups)
    mag_factor1 = lcm(cout // groups, cout0) // (cout // groups)
    mag_factor = min(lcm(mag_factor0, mag_factor1), groups)

    cin1_g = (mag_factor * fmap_c // groups + cin0 - 1) // cin0
    cout_g = (mag_factor * cout // groups + cout0 - 1) // cout0 * cout0

    group_dict = {"real_g": (groups + mag_factor - 1) // mag_factor,
                  "mag_factor": mag_factor,
                  "cin1_g": cin1_g,
                  "cout_g": cout_g,
                  "cin_ori": fmap_c,
                  "cout_ori": cout}

    return group_dict
