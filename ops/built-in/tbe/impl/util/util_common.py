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
import itertools
import math
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
    # check group
    if groups <= 0 or groups > fmap_c or groups > cout:
        dict_args = {
            'errCode': 'E60038',
            'desc': "Group must not be larger than x channel and filter channel"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

    if fmap_c % groups != 0 or cout % groups != 0:
        dict_args = {
            'errCode': 'E60038',
            'desc': "Feature map's or filter's channel must be divisible by group"
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))

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


def update_axis_for_other_format(ori_shape, axis, input_format, ori_format, reduce_mode=False):
    """
    update_axis_for_other_format: when format is changed, the axis will be updated
    """
    if input_format in ("NDC1HWC0", "NC1HWC0"):
        axis = axis % len(ori_shape)
        # ex: ori axis with N, axis = 0
        # ex: ori axis with D, axis = 1
        # ex: ori axis with C, axis = 1 (NC1HWC0) 2(NDC1HWC0)
        # ex: ori axis with H, axis = 2 (NC1HWC0) 3(NDC1HWC0)
        # ex: ori axis with W, axis = 3 (NC1HWC0) 4(NDC1HWC0)
        offset_6hd = 1 if input_format == "NDC1HWC0" else 0
        format_c_axis = 1 + offset_6hd if not reduce_mode else [1 + offset_6hd, 4 + offset_6hd]
        format_axis_map = {
            "N": 0,
            "C": format_c_axis,
            "H": 2 + offset_6hd,
            "W": 3 + offset_6hd,
            "D": 1
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    if input_format in ("FRACTAL_NZ",):
        axis = axis % len(ori_shape)
        # when FRACTAL_NZ, mean: [A, B, C, D] -> [A, B, ceil(D//16), ceil(C//16), 16, 16]
        # update axis as follow:
        # ex: ori axis with last one dim, axis = the dim of ceil(D//16)
        # ex: ori axis with last second dim, axis = the dim of ceil(C//16)
        if axis == len(ori_shape) - 1:
            axis = len(ori_shape) - 2 if not reduce_mode else [len(ori_shape) - 2, len(ori_shape) + 1]
        elif axis == len(ori_shape) - 2:
            axis = len(ori_shape) - 1 if not reduce_mode else [len(ori_shape) - 1, len(ori_shape) + 0]

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        axis = axis % len(ori_shape)
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        offset_3d = 1 if input_format == "FRACTAL_Z_3D" else 0
        format_c_axis = 0 + offset_3d if not reduce_mode else [0 + offset_3d, 5 + offset_3d]
        format_n_axis = 3 + offset_3d if not reduce_mode else [3 + offset_3d, 4 + offset_3d]
        format_axis_map = {
            "N": format_n_axis,
            "C": format_c_axis,
            "H": 1 + offset_3d,
            "W": 2 + offset_3d,
            "D": 0
        }
        concat_dim_name = ori_format[axis]
        axis = format_axis_map[concat_dim_name]

    return axis


def update_shape_base_other_format(input_dict):
    """
    update_axis_for_other_format: when format is changed, the axis will be updated
    """
    ori_shape = input_dict.get("ori_shape")
    ori_format = input_dict.get("ori_format")
    input_shape = input_dict.get("shape")
    input_format = input_dict.get("format")

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        if len(input_shape) == 4:
            # fe will reshape the C1HWNiNoC0/DC1HWNiNoC0 to 4s = (C1HW)NiNoC0/(DC1HW)NiNoC0
            # now will reshape to 6d/7d = C1HWNiNoC0/DC1HWNiNoC0
            dict_zip_shape = dict(zip(list(ori_format), ori_shape))
            shape_h_dim = dict_zip_shape["H"]
            shape_w_dim = dict_zip_shape["W"]

            shape_c1_dim = input_shape[0] // (shape_h_dim * shape_w_dim)
            new_shape = [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])
            if input_format == "FRACTAL_Z_3D":
                shape_d_dim = dict_zip_shape["D"]
                shape_c1_dim = new_shape[0] // shape_d_dim
                new_shape = [shape_d_dim] + [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            input_dict["shape"] = new_shape

    return input_dict


# pylint: disable=too-many-locals
def update_shape_base_other_format_dynamic(input_dict):
    """
    update_axis_for_other_format_dynamic: when format is changed, the axis will be updated
    """
    ori_shape = input_dict.get("ori_shape")
    ori_format = input_dict.get("ori_format")
    input_shape = input_dict.get("shape")
    input_format = input_dict.get("format")
    input_range = input_dict.get("range")

    if input_format in ("FRACTAL_Z", "FRACTAL_Z_3D"):
        # when FRACTAL_Z, mean: C1HWNiNoC0
        # when FRACTAL_Z_3D, mean: DC1HWNiNoC0
        if len(input_shape) == 4:
            # fe will reshape the C1HWNiNoC0/DC1HWNiNoC0 to 4s = (C1HW)NiNoC0/(DC1HW)NiNoC0
            # now will reshape to 6d/7d = C1HWNiNoC0/DC1HWNiNoC0
            dict_zip_shape = dict(zip(list(ori_format), ori_shape))
            shape_h_dim = dict_zip_shape["H"]
            shape_w_dim = dict_zip_shape["W"]

            if shape_h_dim <= 0 or shape_w_dim <= 0 or input_shape[0] <= 0:
                shape_c1_dim = -1
                temp_range = [(1, None)]
                if shape_h_dim > 0 and shape_w_dim > 0:
                    upper = None if input_range[0][1] is None else int(
                        math.ceil(input_range[0][1] / (shape_h_dim * shape_w_dim)))
                    lower = 1 if int(math.floor(input_range[0][0] / (shape_h_dim * shape_w_dim))) == 0 else int(
                        math.floor(input_range[0][0] / (shape_h_dim * shape_w_dim)))
                    temp_range = [(lower, upper)]
            else:
                shape_c1_dim = input_shape[0] // (shape_h_dim * shape_w_dim)
                temp_range = [(shape_c1_dim, shape_c1_dim)]

            for dim in [shape_h_dim, shape_w_dim]:
                temp_range.append((1, None) if dim == -1 else (dim, dim))
            input_range = temp_range + list(input_range[1:])

            new_shape = [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            if input_format == "FRACTAL_Z_3D":
                shape_d_dim = dict_zip_shape["D"]
                if shape_d_dim <= 0 or new_shape[0] <= 0:
                    shape_c1_dim = -1
                    temp_range = [(1, None)]
                    if shape_d_dim > 0:
                        upper = None if input_range[0][1] is None else int(math.ceil(input_range[0][1] / shape_d_dim))
                        lower = 1 if int(math.floor(input_range[0][0] / shape_d_dim)) == 0 else int(
                            math.floor(input_range[0][0] /shape_d_dim))
                        temp_range = [(lower, upper)]
                else:
                    shape_c1_dim = new_shape[0] // shape_d_dim
                    temp_range = [(shape_c1_dim, shape_c1_dim)]

                temp_range.insert(0, (1, None) if shape_d_dim == -1 else (shape_d_dim, shape_d_dim))
                input_range = temp_range + input_range[1:]
                new_shape = [shape_d_dim] + [shape_c1_dim, shape_h_dim, shape_w_dim] + list(input_shape[1:])

            input_dict["shape"] = new_shape
            input_dict["range"] = input_range

    return input_dict


def get_fused_format_str(format_char_list):
    """
    get_fused_format from char
    ex:
        input  ["N", "C", "H"]
        putput ["NCH", "NHC", "CNH", "CHN", "HNC", "HCN"]
    """
    format_iter = itertools.permutations(format_char_list, len(format_char_list))
    format_char_list = list(format_iter)
    format_str_list = []
    for _, char_list in enumerate(format_char_list):
        format_str_list.append(''.join(list(char_list)))

    return format_str_list


def is_dynamic_input(_inputs):
    """
    is_dynamic_input: check whether the shape contain -1
        contain -1 return True else False

    Parameters
    ----------
    _inputs: list of dict/tuple of dict/dict

    Returns
    -------
    bool
    """
    if not isinstance(_inputs, list) and not isinstance(_inputs, tuple):
        _inputs = [_inputs]

    for _, _input in enumerate(_inputs):
        if -1 in _input.get("shape"):
            return True

    return False
