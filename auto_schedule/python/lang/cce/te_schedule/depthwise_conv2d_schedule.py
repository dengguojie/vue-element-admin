# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
Schedule of depthwise conv2d.
"""
# pylint: disable=too-many-lines
from te.utils.error_manager import error_manager_util
from te.domain.tiling.get_tiling import get_tiling
from te.domain.tiling.tiling_query import tiling_query
from te.tvm.schedule import create_schedule
from te.tvm import api as tvm
from te.platform import cce_params
from te.platform import cce_conf
from te.lang.cce.te_schedule.util import L1CommonParam
from te.lang.cce.te_compute.depthwise_conv2d_compute import DepthwiseConv2dParam

TILING_HO_TIMES = 16

BLOCK_SIZE = cce_params.BLOCK_REDUCE

# fp16 dtype size 2
FP16_SIZE = 2
# l1 and l0, ping pong read/write mechanism
DOUBLE_BUFFER = 2
# l1 memory size 1M byte
L1_MEM_LIMIT = cce_conf.get_soc_spec(cce_conf.L1_SIZE)
# tilling batch in l1
RESHAPE_BATCH_SIZE_IN_L1 = 1
# l1 include 1 batch
BATCH_SIZE_IN_L1 = 1
# Split ho and wo for mad_cc as 1.
SPLIT_SLOT_TO_BE_FILL_BY_COMPUTE_AT = 1
# L1 size cell
CUBE_M_SIZE_CELL = cce_conf.get_soc_spec(cce_conf.L0B_SIZE) // DOUBLE_BUFFER // FP16_SIZE // BLOCK_SIZE

# fmap h/w 14, N,C fusion tiling
SMALL_FEATURE_MAP_SIZE = 14
# N,C fusion tiling, N max 32
BATCH_TILING_FACTOR = 32

# tiling check
TILING_AL1_SHAPWE_DIM = 4
TILING_BL1_SHAPWE_DIM = 4
TILING_AL0_MATRIX_DIM = 6
TILING_BL0_MATRIX_DIM = 6
TILING_CL0_MATRIX_DIM = 6
TILING_CUB_MATRIX_DIM = 6
TILING_BLOCK_DIM_DIM = 4
TILING_FLOAT16_M = 16
TILING_FLOAT16_K = 16
TILING_FLOAT16_N = 16
TILING_INT8_M = 16
TILING_INT8_K = 32
TILING_INT8_N = 16

C0_32 = TILING_INT8_K
C0_16 = TILING_FLOAT16_K


def _ceil(num):
    """
        Return the least multiple of 16 integer number
        which is greater than or equal to num.
    """
    return ((num + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE


def set_pragma_for_cache_read_mode(is_overload, stage, first_axis):
    """
    set pragma on the first axis for cache read mode

    Parameters
    ----------
    is_overload: True means overload

    stage: a Stage represents schedule for one operation

    first_axis: axis to set flag

    Returns
    -------
    """
    cache_read_mode = 0 if is_overload else 1
    stage.pragma(first_axis, "json_info_cache_read_mode", cache_read_mode)


def _check_shape_valid(tiling, keyname):
    """
    check tiling shape and valid
    """
    return (tiling[keyname] != []) and (tiling[keyname] is not None)


def _check_shape(tiling, keyname, length, force_check=False):
    """
    check keyname shape
    """
    if _check_shape_valid(tiling, keyname) or force_check:
        if len(tiling[keyname]) != length:
            dict_args = {
                'errCode': 'E60030',
                'param_name': 'tiling[{}]'.format(keyname),
                'op_name': 'depthwise_conv2d',
                'expected_length': str(length),
                'length': str(len(tiling[keyname]))
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_keyname_type(tiling, keyname, data_type):
    """
    check the type of keyname is bool
    """
    if tiling[keyname] == 0:
        tiling[keyname] = False
    if not isinstance(tiling[keyname], data_type):
        dict_args = {
            'errCode': 'E60032',
            'param_name': 'tiling[{}]'.format(keyname),
            'op_name': 'depthwise_conv2d',
            'expected_data_type_list': str(data_type),
            'data_type': type(tiling[keyname])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_pingpong_key(tiling, keyname):
    """
    check tiling keys
    """
    if keyname not in tiling["manual_pingpong_buffer"].keys():
        dict_args = {'errCode': 'E60115', 'op_name': 'depthwise_conv2d'}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_tiling_al0_matrix(tiling):
    """
    check the field AL0_matrix in tiling
    """
    if tiling["AL0_matrix"][0] != tiling["CL0_matrix"][1]:
        dict_args = {
            'errCode': 'E61301',
            'op_name': 'depthwise_conv2d',
            'param_name_1': tiling["AL0_matrix"][0],
            'param_name_2': tiling["CL0_matrix"][1]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["AL0_matrix"][2] != TILING_FLOAT16_M:
        dict_args = {
            'errCode': 'E62305',
            'op_name': 'depthwise_conv2d',
            'param_name': 'tiling["AL0_matrix"][2]',
            'expect_value': TILING_FLOAT16_M,
            'value': tiling["AL0_matrix"][2]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["AL0_matrix"][3] != TILING_FLOAT16_K:
        dict_args = {
            'errCode': 'E62305',
            'op_name': 'depthwise_conv2d',
            'param_name': 'tiling["AL0_matrix"][3]',
            'expect_value': TILING_FLOAT16_K,
            'value': tiling["AL0_matrix"][3]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_tiling_bl0_matrix(tiling):
    """
    check the field BL0_matrix in tiling
    """
    if tiling["BL0_matrix"][1] != tiling["CL0_matrix"][0]:
        dict_args = {
            'errCode': 'E61301',
            'op_name': 'depthwise_conv2d',
            'param_name_1': tiling["BL0_matrix"][1],
            'param_name_2': tiling["CL0_matrix"][0]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["BL0_matrix"] != []:
        if tiling["BL0_matrix"][2] != TILING_FLOAT16_N:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["BL0_matrix"][2]',
                'expect_value': TILING_FLOAT16_N,
                'value': tiling["BL0_matrix"][2]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if tiling["BL0_matrix"][3] != TILING_FLOAT16_K:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["BL0_matrix"][3]',
                'expect_value': TILING_FLOAT16_K,
                'value': tiling["BL0_matrix"][3]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _common_tiling_check(tiling):
    """
    check common tiling

    Parameters
    ----------
    tiling: buffer split

    Returns
    -------
    """
    _check_shape(tiling, "AL1_shape", TILING_AL1_SHAPWE_DIM)
    _check_shape(tiling, "BL1_shape", TILING_BL1_SHAPWE_DIM)
    _check_shape(tiling, "AL0_matrix", TILING_AL0_MATRIX_DIM)
    _check_shape(tiling, "BL0_matrix", TILING_BL0_MATRIX_DIM)
    _check_shape(tiling, "CL0_matrix", TILING_CL0_MATRIX_DIM, True)
    _check_shape(tiling, "CUB_matrix", TILING_CUB_MATRIX_DIM, True)
    _check_shape(tiling, "block_dim", TILING_BLOCK_DIM_DIM, True)

    _check_keyname_type(tiling, "n_bef_batch_flag", bool)
    _check_keyname_type(tiling, "n_bef_group_flag", bool)
    _check_keyname_type(tiling, "A_overhead_opt_flag", bool)
    _check_keyname_type(tiling, "B_overhead_opt_flag", bool)
    _check_keyname_type(tiling, "manual_pingpong_buffer", dict)

    _check_pingpong_key(tiling, "AL1_pbuffer")
    _check_pingpong_key(tiling, "BL1_pbuffer")
    _check_pingpong_key(tiling, "AL0_pbuffer")
    _check_pingpong_key(tiling, "BL0_pbuffer")
    _check_pingpong_key(tiling, "CL0_pbuffer")
    _check_pingpong_key(tiling, "CUB_pbuffer")

    if _check_shape_valid(tiling, "AL0_matrix"):
        _check_tiling_al0_matrix(tiling)
    if _check_shape_valid(tiling, "BL0_matrix"):
        _check_tiling_bl0_matrix(tiling)
    if _check_shape_valid(tiling, "AL0_matrix") and _check_shape_valid(tiling, "BL0_matrix"):
        if tiling["AL0_matrix"][1] != tiling["BL0_matrix"][0]:
            dict_args = {
                'errCode': 'E61301',
                'op_name': 'depthwise_conv2d',
                'param_name_1': tiling["AL0_matrix"][1],
                'param_name_2': tiling["BL0_matrix"][0]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["CL0_matrix"][2] != TILING_FLOAT16_M:
        dict_args = {
            'errCode': 'E62305',
            'op_name': 'depthwise_conv2d',
            'param_name': 'tiling["CL0_matrix"][2]',
            'expect_value': TILING_FLOAT16_M,
            'value': tiling["CL0_matrix"][2]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["CL0_matrix"][3] != TILING_FLOAT16_N:
        dict_args = {
            'errCode': 'E62305',
            'op_name': 'depthwise_conv2d',
            'param_name': 'tiling["CL0_matrix"][3]',
            'expect_value': TILING_FLOAT16_N,
            'value': tiling["CL0_matrix"][3]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["BL1_shape"] is None:
        dict_args = {'errCode': 'E67004', 'op_name': 'depthwise_conv2d_backprop_filter', 'BL1_shape': 'None'}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_type_bool(input_type):
    """
    check the type of keyname is bool
    """
    if not isinstance(input_type, bool):
        dict_args = {
            'errCode': 'E60032',
            'param_name': 'input_type',
            'op_name': 'depthwise_conv2d',
            'expected_data_type_list': "bool",
            'data_type': type(input_type)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_tiling_valid_rasie_error(tiling, tilling_fractor_k):
    """
    check tiling valid and raise error
    """
    if not isinstance(tiling["manual_pingpong_buffer"], dict):
        dict_args = {
            'errCode': 'E60032',
            'param_name': 'tiling["manual_pingpong_buffer"]',
            'op_name': 'depthwise_conv2d',
            'expected_data_type_list': "dict",
            'data_type': type(tiling["manual_pingpong_buffer"])
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if tiling["AL0_matrix"][0] != tiling["CL0_matrix"][1]:
        dict_args = {
            'errCode': 'E61301',
            'op_name': 'depthwise_conv2d',
            'param_name_1': tiling["AL0_matrix"][0],
            'param_name_2': tiling["CL0_matrix"][1]
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if tiling["BL0_matrix"] != []:
        if tiling["AL0_matrix"][1] != tiling["BL0_matrix"][0]:
            dict_args = {
                'errCode': 'E61301',
                'op_name': 'depthwise_conv2d',
                'param_name_1': tiling["AL0_matrix"][1],
                'param_name_2': tiling["BL0_matrix"][0]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["BL0_matrix"][1] != tiling["CL0_matrix"][0]:
            dict_args = {
                'errCode': 'E61301',
                'op_name': 'depthwise_conv2d',
                'param_name_1': tiling["BL0_matrix"][1],
                'param_name_2': tiling["CL0_matrix"][0]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["AL0_matrix"][2] != TILING_FLOAT16_M:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["AL0_matrix"][2]',
                'expect_value': TILING_FLOAT16_M,
                'value': tiling["AL0_matrix"][2]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["AL0_matrix"][3] != tilling_fractor_k:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["AL0_matrix"][3]',
                'expect_value': tilling_fractor_k,
                'value': tiling["AL0_matrix"][3]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["BL0_matrix"] != []:
            if tiling["BL0_matrix"][2] != TILING_FLOAT16_N:
                dict_args = {
                    'errCode': 'E62305',
                    'op_name': 'depthwise_conv2d',
                    'param_name': 'tiling["BL0_matrix"][2]',
                    'expect_value': TILING_FLOAT16_N,
                    'value': tiling["BL0_matrix"][2]
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            if tiling["BL0_matrix"][3] != tilling_fractor_k:
                dict_args = {
                    'errCode': 'E62305',
                    'op_name': 'depthwise_conv2d',
                    'param_name': 'tiling["BL0_matrix"][3]',
                    'expect_value': tilling_fractor_k,
                    'value': tiling["BL0_matrix"][3]
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["CL0_matrix"][2] != TILING_FLOAT16_M:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["CL0_matrix"][2]',
                'expect_value': TILING_FLOAT16_M,
                'value': tiling["CL0_matrix"][2]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if tiling["CL0_matrix"][3] != TILING_FLOAT16_N:
            dict_args = {
                'errCode': 'E62305',
                'op_name': 'depthwise_conv2d',
                'param_name': 'tiling["CL0_matrix"][3]',
                'expect_value': TILING_FLOAT16_N,
                'value': tiling["CL0_matrix"][3]
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def check_tiling_raise_error(tiling, dtype):
    """
    check tiling and raise error
    """
    if dtype == "int8":
        tilling_fractor_k = TILING_INT8_K
    elif dtype == "float16":
        tilling_fractor_k = TILING_FLOAT16_K
    if tiling["AL1_shape"] != []:
        if len(tiling["AL1_shape"]) != TILING_AL1_SHAPWE_DIM:
            dict_args = {
                'errCode': 'E60030',
                'param_name': 'tiling["AL1_shape"]',
                'op_name': 'depthwise_conv2d',
                'expected_length': str(TILING_AL1_SHAPWE_DIM),
                'length': str(len(tiling["AL1_shape"]))
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    chek_bl1_shape = (tiling["BL1_shape"] != []) and (tiling["BL1_shape"] is not None)
    if chek_bl1_shape:
        if len(tiling["BL1_shape"]) != TILING_BL1_SHAPWE_DIM:
            dict_args = {
                'errCode': 'E60030',
                'param_name': 'tiling["BL1_shape"]',
                'op_name': 'depthwise_conv2d',
                'expected_length': str(TILING_BL1_SHAPWE_DIM),
                'length': str(len(tiling["BL1_shape"]))
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if len(tiling["AL0_matrix"]) != TILING_AL0_MATRIX_DIM:
        dict_args = {
            'errCode': 'E60030',
            'param_name': 'tiling["AL0_matrix"]',
            'op_name': 'depthwise_conv2d',
            'expected_length': str(TILING_AL0_MATRIX_DIM),
            'length': str(len(tiling["AL0_matrix"]))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if tiling["BL0_matrix"] != []:
        if len(tiling["BL0_matrix"]) != TILING_BL0_MATRIX_DIM:
            dict_args = {
                'errCode': 'E60030',
                'param_name': 'tiling["BL0_matrix"]',
                'op_name': 'depthwise_conv2d',
                'expected_length': str(TILING_BL0_MATRIX_DIM),
                'length': str(len(tiling["BL0_matrix"]))
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if len(tiling["CL0_matrix"]) != TILING_CL0_MATRIX_DIM:
        dict_args = {
            'errCode': 'E60030',
            'param_name': 'tiling["CL0_matrix"]',
            'op_name': 'depthwise_conv2d',
            'expected_length': str(TILING_CL0_MATRIX_DIM),
            'length': str(len(tiling["CL0_matrix"]))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if len(tiling["CUB_matrix"]) != TILING_CUB_MATRIX_DIM:
        dict_args = {
            'errCode': 'E60030',
            'param_name': 'tiling["CUB_matrix"]',
            'op_name': 'depthwise_conv2d',
            'expected_length': str(TILING_CUB_MATRIX_DIM),
            'length': str(len(tiling["CUB_matrix"]))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if len(tiling["block_dim"]) != TILING_BLOCK_DIM_DIM:
        dict_args = {
            'errCode': 'E60030',
            'param_name': 'tiling["block_dim"]',
            'op_name': 'depthwise_conv2d',
            'expected_length': str(TILING_BLOCK_DIM_DIM),
            'length': str(len(tiling["block_dim"]))
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    _check_type_bool(tiling["n_bef_batch_flag"])
    _check_type_bool(tiling["n_bef_group_flag"])
    _check_type_bool(tiling["batch_bef_group_flag"])
    _check_type_bool(tiling["A_overhead_opt_flag"])
    _check_type_bool(tiling["B_overhead_opt_flag"])

    _check_tiling_valid_rasie_error(tiling, tilling_fractor_k)


def check_tiling(tiling, dtype):
    """
    check tiling dtype
    """
    if tiling["AL0_matrix"][2] == 32:
        return False
    check_tiling_raise_error(tiling, dtype)
    return True


def tiling_new_check_empty(input_module, string, place):
    """
    check tiling empty
    """
    if input_module[string] != []:
        return input_module[string][0:place]
    return []


def tiling_new_check_none(input_module, string, place):
    """
    check tiling none
    """
    if input_module[string] is not None:
        return input_module[string][0:place]
    return input_module[string]


def tiling_new_check_none_empty(input_module, string, place):
    """
    check tiling none and empty
    """
    if input_module[string] is not None and input_module[string] != []:
        return input_module[string][0:place]
    return input_module[string]


def get_tiling_dict_first(tiling_new):
    """
    get tiling dictionary first
    """
    tiling = {}
    tiling["AL0_matrix"] = tiling_new["AL0_matrix"][0:6]
    tiling["CL0_matrix"] = tiling_new["CL0_matrix"][0:6]
    tiling["CUB_matrix"] = tiling_new["CUB_matrix"][0:6]
    tiling["BL0_matrix"] = tiling_new_check_empty(tiling_new, "BL0_matrix", 6)
    tiling["manual_pingpong_buffer"] = tiling_new["manual_pingpong_buffer"]
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["AUB_shape"] = tiling_new_check_none(tiling_new, "AUB_shape", 4)
    tiling["AL1_shape"] = tiling_new_check_empty(tiling_new, "AL1_shape", 4)
    tiling["BL1_shape"] = tiling_new_check_none_empty(tiling_new, "BL1_shape", 5)
    tiling["block_dim"] = tiling_new["block_dim"][0:4]

    tiling["scale_drq_split_flag"] = False
    tiling["bias_split_flag"] = False
    tiling["n_bef_batch_flag"] = tiling_new["n_bef_batch_flag"]
    tiling["n_bef_group_flag"] = tiling_new["n_bef_group_flag"]
    tiling["batch_bef_group_flag"] = tiling_new["batch_bef_group_flag"]
    tiling["A_overhead_opt_flag"] = tiling_new["A_overhead_opt_flag"]
    tiling["B_overhead_opt_flag"] = tiling_new["B_overhead_opt_flag"]

    return tiling


def get_tiling_dict_second(tiling, shape_input, padding, stride, dtype):
    """
    get tiling dictionary second
    """
    if not check_tiling(tiling, dtype):
        tiling = {}
        m_bit_length = {"float32": 32, "float16": 16, "uint8": 8, "int8": 8, "uint4": 4, "int4": 4}
        m_bit_ratio = {"int32": 4, "float32": 4, "float16": 2, "uint8": 1, "int8": 1, "uint4": 1.0 / 2, "int4": 1.0 / 2}
        wo_shape = (shape_input[0][3] + (2 * padding[0]) - shape_input[1].shape[2]) // stride[1] + 1
        gen_m_target = 0
        for m_target in range(32, 0, -1):
            tmp1 = ((m_target * m_bit_length['float16']) + wo_shape - 1) // wo_shape
            tmp2 = ((tmp1 * padding[1]) + shape_input[1].shape[1]) * shape_input[0][3]
            max_feature_map = tmp2 * 2 * m_bit_ratio[dtype]
            if int(max_feature_map) < L1_MEM_LIMIT:
                gen_m_target = m_target
                break

        m_value = gen_m_target
        tiling["AL1_shape"] = [1, 1, 1, 1]
        tiling["BL1_shape"] = None
        tiling["AL0_matrix"] = [m_value, 1, 16, 16, 1, 1]
        tiling["BL0_matrix"] = [1, 2, 16, 16, 1, 1]
        tiling["CL0_matrix"] = [2, m_value, 16, 16, 1, 1]
        tiling["CUB_matrix"] = [2, m_value, 16, 16, 1, 1]
        tiling["manual_pingpong_buffer"] = {
            'AL1_pbuffer': 1,
            'BL1_pbuffer': 1,
            'AL0_pbuffer': 1,
            'BL0_pbuffer': 1,
            'CL0_pbuffer': 1,
            'CUB_pbuffer': 1,
            'UBG_pbuffer': 1,
        }

        tiling["scale_drq_split_flag"] = True
        tiling["bias_split_flag"] = True
        tiling["block_dim"] = [1, 1, 1, 1]

    return tiling


def _get_fused_double_operand_num(tensor_dict):
    """
    get fused double operand num
    """
    fused_double_operand_num = tensor_dict["fused_double_operand_num"]
    not_fused_flag = False
    if tensor_dict["flag_is_dequant2"] or (tensor_dict["flag_is_dequant_quant"]
                                           and tensor_dict["flag_is_dequant_sqrt"]):
        fused_double_operand_num = 3
    elif tensor_dict["flag_is_dequant"] or (tensor_dict["flag_is_dequant_quant"]
                                            and not tensor_dict["flag_is_dequant_sqrt"]):
        fused_double_operand_num = 2
    elif tensor_dict["flag_is_requant"]:
        fused_double_operand_num = 2
    elif tensor_dict["flag_is_dequant_sigmoid_mul"] or tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        fused_double_operand_num = 4 if tensor_dict["flag_is_dequant2_sigmoid_mul"] else 3
    else:
        not_fused_flag = True

    return fused_double_operand_num, not_fused_flag


def _tiling_fetch_all(fmap_shape,
                      shape_w,
                      group_num,
                      padding,
                      stride,
                      mad_dtype,
                      fused_num,
                      input_memory_type,
                      output_memory_type,
                      l1_fusion_type,
                      fusion_type_new,
                      fm_l1_valid_size,
                      bias=False,
                      c_dtype="int8",
                      not_fused_flag=False,
                      kernel_name="depthwise_fused_tiling_fetch"):
    """
    tiling fetch all
    """
    if len(fmap_shape) == 6:
        fmap_shape_nc1hwco = fmap_shape[0], fmap_shape[1], fmap_shape[3], fmap_shape[4], fmap_shape[5]
    else:
        fmap_shape_nc1hwco = fmap_shape[0], fmap_shape[1], fmap_shape[2], fmap_shape[3], fmap_shape[4]

    shape_w_nc1hwco = shape_w.shape[3], shape_w.shape[0], shape_w.shape[1], shape_w.shape[2], shape_w.shape[4]
    if mad_dtype == "int32":
        dtype = "int8"
        res_dtype = "int32"
    else:
        dtype = "float16"
        res_dtype = "float16"
        fmap_shape_nc1hwco = list(map(int, fmap_shape_nc1hwco))
    if not_fused_flag:
        c_dtype = res_dtype

    fmap_shape_nc1hwco = list(map(int, fmap_shape_nc1hwco))
    shape_w_nc1hwco = list(map(int, shape_w_nc1hwco))
    if input_memory_type == 0 and output_memory_type == 0:
        temp_k = 0
    elif input_memory_type == 0 and output_memory_type == 1:
        temp_k = 1
    elif input_memory_type == 1 and output_memory_type == 0:
        temp_k = 2
    elif input_memory_type == 1 and output_memory_type == 1:
        temp_k = 3
    else:
        temp_k = 4
    fusion_type_new = fusion_type_new * 10 + temp_k
    info_dict = {
        "op_type": 'depthwise_conv2d_forward',
        "A_shape": fmap_shape_nc1hwco,
        "B_shape": shape_w_nc1hwco,
        "A_dtype": dtype,
        "B_dtype": dtype,
        "C_dtype": c_dtype,
        "mad_dtype": mad_dtype,
        "padu": padding[0],
        "padd": padding[1],
        "padl": padding[2],
        "padr": padding[3],
        "strideH": stride[0],
        "strideW": stride[1],
        "dilationH": 1,
        "dilationW": 1,
        "group": group_num,
        "bias_flag": bias,
        "fused_double_operand_num": fused_num,
        "in_fm_memory_type": [input_memory_type],
        "out_fm_memory_type": [output_memory_type],
        "l1_fusion_type": l1_fusion_type,
        "fusion_type": fusion_type_new,
        "fm_l1_valid_size": fm_l1_valid_size,
        "kernel_name": kernel_name.value
    }

    tiling = get_tiling(info_dict)
    return tiling


def _get_tiling_fetch(mad_dtype, tensor_dict):
    """
    get tiling fetch
    """
    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])
    stride_w = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][1])
    stride_h = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][0])
    kernel_name = tensor_dict["kernel_name"]

    bias_flag = tensor_dict["flag_is_dequant_bias"] or tensor_dict["flag_is_requant_bias"] or tensor_dict["bias_flag"]
    fused_double_operand_num, not_fused_flag = _get_fused_double_operand_num(tensor_dict)

    fmap_shape = tensor_dict["fmap"].shape
    if tensor_dict["fmap_valid_shape"]:
        fmap_shape = tensor_dict["fmap_valid_shape"]
    tiling = _tiling_fetch_all(fmap_shape, tensor_dict["filter_buf"], tensor_dict["group_num"],
                               [pad_top, pad_bottom, pad_left, pad_right], [stride_h, stride_w], mad_dtype,
                               fused_double_operand_num, tensor_dict["input_memory_type"],
                               tensor_dict["output_memory_type"], tensor_dict["l1_fusion_type"],
                               tensor_dict["fusion_type_new"], tensor_dict["fm_l1_valid_size"], bias_flag,
                               tensor_dict["fused_c_dtype"], not_fused_flag, kernel_name)
    return tiling


def _set_a_cbuf_row_major(mad_dtype, a_cbuf_row_major, wo_shape, sch):
    """
    set schedule a_cbuf_row_major
    """
    if mad_dtype == "int32":
        sch[a_cbuf_row_major].buffer_align((1, 1), (1, 1), (wo_shape, wo_shape), (1, 1), (1, 1), (1, 1), (1, 32))
    else:
        sch[a_cbuf_row_major].buffer_align((1, 1), (1, 1), (wo_shape, wo_shape), (1, 1), (1, 1), (1, 1),
                                           (1, BLOCK_SIZE))
    return sch


def _set_common_flag():
    """
    set common flag
    """
    tensor_dict = {}
    tensor_dict["bias_flag"] = False
    tensor_dict["flag_is_dequant"] = False
    tensor_dict["flag_is_dequant_bias"] = False
    tensor_dict["flag_is_requant_bias"] = False
    tensor_dict["flag_is_dequant_sqrt"] = False
    tensor_dict["flag_is_dequant2"] = False
    tensor_dict["flag_is_dequant_quant"] = False
    tensor_dict["flag_is_quant_sqrt"] = False
    tensor_dict["flag_is_quant_relu6_dequant"] = False
    tensor_dict["flag_is_quant_mul_dequant"] = False
    tensor_dict["flag_is_dequant_mul"] = False
    tensor_dict["flag_is_dequant2_mul"] = False
    tensor_dict["flag_is_dequant_sigmoid_mul"] = False
    tensor_dict["flag_is_dequant2_sigmoid_mul"] = False
    tensor_dict["flag_is_requant"] = False
    tensor_dict["flag_is_sigmoid_mul"] = False
    tensor_dict["flag_is_eltwisce_case"] = False
    tensor_dict["flag_is_dequant_power_eltwise"] = False
    tensor_dict["flag_is_dequant2_power_eltwise"] = False
    tensor_dict["flag_is_write_select"] = False
    tensor_dict["flag_is_broadcast"] = False
    tensor_dict["fusion_type_new"] = 0

    tensor_dict["fused_double_operand_num"] = 0
    tensor_dict["fused_c_dtype"] = "int8"
    # group_num is used to distin pre_relu
    tensor_dict["group_num"] = 1

    return tensor_dict


def _deq_scalar_mode(tensor_dict):
    """
    set scalar vector flag
    """
    if "deq_reg" in tensor_dict:
        if int(tensor_dict["deq_reg"].shape[1]) == 1:
            tensor_dict["sca_vec_flag"] = 0
        else:
            tensor_dict["sca_vec_flag"] = 1
    if "vreq_reg" in tensor_dict:
        if int(tensor_dict["vreq_reg"].shape[1]) == 1:
            tensor_dict["sca_vec_flag"] = 0
        else:
            tensor_dict["sca_vec_flag"] = 1
    return tensor_dict


def _set_tensor_by_op_tag(out):
    """
    set tensor by op_tag
    """
    tensor_dict = _set_common_flag()
    if out.op.tag == "dequant2_remove_pad":
        tensor_dict = _dequant2_remove_pad(out, tensor_dict)
    elif out.op.tag == "dequant_remove_pad":
        tensor_dict = _dequant_remove_pad(out, tensor_dict)
    elif out.op.tag == "quant":
        tensor_dict = _quant(out, tensor_dict)
    elif out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        tensor_dict = _elewise_single_relu(out, tensor_dict)
    elif out.op.tag == "elewise_binary_mul":
        tensor_dict = _elewise_binary_mul(out, tensor_dict)
    elif out.op.tag == "elewise_single_VS_min":
        tensor_dict = _elewise_single_vs_min(out, tensor_dict)
    elif out.op.tag == "depthwise_conv2d":
        tensor_dict = _depthwise_conv2d(out, tensor_dict)
    elif out.op.tag == "requant_remove_pad":
        tensor_dict = _requant_remove_pad(out, tensor_dict)
    elif out.op.tag == "write_select":
        tensor_dict = _write_select(out, tensor_dict)
    else:
        dict_args = {'errCode': 'E67001', 'op_name': 'depthwise_conv2d', 'prama_name': 'out.op.tag', 'tag': out.op.tag}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if out.op.tag == "depthwise_conv2d":
        tensor_dict["kernel_name"] = out.op.attrs["kernel_name"]
    else:
        tensor_dict["kernel_name"] = tensor_dict["depthwise_res"].op.attrs['kernel_name']
    tensor_dict = _deq_scalar_mode(tensor_dict)

    return tensor_dict


def _l1_fusion_check(tensor_dict):
    """
    l1 fusion check
    """
    offset = DepthwiseConv2dParam.fusion_para.get("slice_offset")
    valid_shape = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    input_mem_type = int(DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    if offset and valid_shape:
        if input_mem_type == 1:
            tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
        else:
            tensor_dict["fusion_fmap_select"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
            tensor_dict["fmap"] = tensor_dict["fusion_fmap_select"].op.input_tensors[0]
    else:
        tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _dequant2_remove_pad(out, tensor_dict):
    """
    dequant2 remove pad
    """
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant2"] = out.op.input_tensors[0]
    tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_dequant2"] = True
    tensor_dict["fusion_type_new"] = 6

    return tensor_dict


def _dequant_remove_pad(out, tensor_dict):
    """
    dequant remove pad
    """
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant1"] = out.op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_dequant"] = True
    tensor_dict["fusion_type_new"] = 5

    return tensor_dict


def _requant_remove_pad(out, tensor_dict):
    """
    requant remove pad
    """
    tensor_dict["data_transfer"] = out.op.input_tensors[0]
    tensor_dict["requant"] = tensor_dict["data_transfer"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["requant"].op.input_tensors[0]
    tensor_dict["vreq_reg"] = tensor_dict["requant"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_requant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]

    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]

    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict = _l1_fusion_check(tensor_dict)
    tensor_dict["flag_is_requant"] = True

    return tensor_dict


def _quant_dequant2(tensor_dict):
    """
    quant dequant2
    """
    tensor_dict["flag_is_dequant_sqrt"] = True
    if ("max" in tensor_dict and tensor_dict["max"].op.input_tensors[0].name == "dequant2_remove_pad"):
        tensor_dict["quant_remove_pad"] = tensor_dict["max"].op.input_tensors[0]
    elif "mul_res" in tensor_dict and tensor_dict["mul_res"].op.input_tensors[0].name == "dequant2_remove_pad":
        tensor_dict["quant_remove_pad"] = tensor_dict["mul_res"].op.input_tensors[0]
    elif "power0_mul" in tensor_dict and tensor_dict["power0_mul"].op.input_tensors[0].name == "dequant2_remove_pad":
        tensor_dict["quant_remove_pad"] = tensor_dict["power0_mul"].op.input_tensors[0]

    else:
        tensor_dict["quant_remove_pad"] = tensor_dict["input_ub"].op.input_tensors[0]

    tensor_dict["dequant2"] = tensor_dict["quant_remove_pad"].op.input_tensors[0]

    tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]

    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["bias_flag"] = True
        if tensor_dict["depthwise_res"].op.attrs['dsl_flag'].value == 1:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
            tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
            tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
            tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
            tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
            tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
            tensor_dict = _l1_fusion_check(tensor_dict)
            tensor_dict["bias_flag"] = False
    else:
        tensor_dict["flag_is_dequant_bias"] = False
        tensor_dict["mad_ubuf_ori"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_ubuf_ori"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
        tensor_dict = _l1_fusion_check(tensor_dict)

    return tensor_dict


def _quant_dequant1(tensor_dict):
    """
    quant dequant1
    """
    tensor_dict["flag_is_dequant_sqrt"] = False
    if "max" in tensor_dict and tensor_dict["max"].op.input_tensors[0].name == "dequant_remove_pad":
        tensor_dict["quant_remove_pad"] = tensor_dict["max"].op.input_tensors[0]
    elif "mul_res" in tensor_dict and tensor_dict["mul_res"].op.input_tensors[0].name == "dequant_remove_pad":
        tensor_dict["quant_remove_pad"] = tensor_dict["mul_res"].op.input_tensors[0]
    elif "power0_mul" in tensor_dict and tensor_dict["power0_mul"].op.input_tensors[0].name == "dequant_remove_pad":
        tensor_dict["quant_remove_pad"] = tensor_dict["power0_mul"].op.input_tensors[0]
    else:
        tensor_dict["quant_remove_pad"] = tensor_dict["input_ub"].op.input_tensors[0]
    tensor_dict["dequant1"] = tensor_dict["quant_remove_pad"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["bias_flag"] = True
        if tensor_dict["depthwise_res"].op.attrs['dsl_flag'].value == 1:
            tensor_dict["flag_is_dequant_bias"] = True
            tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
            tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
            tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
            tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
            tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
            tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
            tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
            tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
            tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
            tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
            tensor_dict = _l1_fusion_check(tensor_dict)
            tensor_dict["bias_flag"] = False
    else:
        tensor_dict["flag_is_dequant_bias"] = False
        tensor_dict["mad_ubuf_ori"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_ubuf_ori"].op.input_tensors[0]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
        tensor_dict = _l1_fusion_check(tensor_dict)

    return tensor_dict


def _quant(out, tensor_dict):
    """
    quant
    """
    tensor_dict["flag_is_dequant_quant"] = True
    tensor_dict["fusion_type_new"] = 7
    if out.op.attrs['scale'].value == 1 and out.op.attrs['sqrt_mode'].value == 0:
        tensor_dict["flag_is_quant_sqrt"] = False
        tensor_dict["cast_i8_ub"] = out.op.input_tensors[0]
        tensor_dict["reform_by_vadds"] = tensor_dict["cast_i8_ub"].op.input_tensors[0]

        tensor_dict["input_ub"] = tensor_dict["reform_by_vadds"].op.input_tensors[0]

        # eltwise_case
        if "eltwise_case" in tensor_dict["input_ub"].op.input_tensors[0].op.attrs:

            def _last_node_vmul_quant(tensor_dict):
                # case 0 last node is vmul
                if tensor_dict["input_ub"].op.input_tensors[0].op.attrs["eltwise_case"] == "eltwise_case_0":
                    tensor_dict["mul_res"] = tensor_dict["input_ub"].op.input_tensors[0]
                    tensor_dict["power1_add"] = tensor_dict["mul_res"].op.input_tensors[1]
                    tensor_dict["flag_is_eltwisce_case"] = "0"

                elif tensor_dict["input_ub"].op.input_tensors[0].op.attrs["eltwise_case"] == "eltwise_case_2":
                    tensor_dict["eltwise_max"] = tensor_dict["input_ub"].op.input_tensors[0]
                    tensor_dict["power1_add"] = tensor_dict["eltwise_max"].op.input_tensors[1]
                    tensor_dict["flag_is_eltwisce_case"] = "2"

                elif tensor_dict["input_ub"].op.input_tensors[0].op.attrs["eltwise_case"] == "eltwise_case_1_2":
                    tensor_dict["eltwise_add"] = tensor_dict["input_ub"].op.input_tensors[0]
                    tensor_dict["eltwise_mul_left"] = tensor_dict["eltwise_add"].op.input_tensors[0]
                    tensor_dict["eltwise_mul_right"] = tensor_dict["eltwise_add"].op.input_tensors[1]
                    tensor_dict["power1_add"] = tensor_dict["eltwise_mul_right"].op.input_tensors[0]
                    tensor_dict["flag_is_eltwisce_case"] = "1_2"

                elif tensor_dict["input_ub"].op.input_tensors[0].op.attrs["eltwise_case"] == "eltwise_case_1_1":
                    tensor_dict["eltwise_add"] = tensor_dict["input_ub"].op.input_tensors[0]
                    tensor_dict["power1_add"] = tensor_dict["eltwise_add"].op.input_tensors[1]
                    tensor_dict["flag_is_eltwisce_case"] = "1_1"
                return tensor_dict

            tensor_dict = _last_node_vmul_quant(tensor_dict)
            tensor_dict["power1_mul"] = tensor_dict["power1_add"].op.input_tensors[0]
            tensor_dict["relu_min"] = tensor_dict["power1_mul"].op.input_tensors[0]
            tensor_dict["relu_max"] = tensor_dict["relu_min"].op.input_tensors[0]
            tensor_dict["power0_add"] = tensor_dict["relu_max"].op.input_tensors[0]
            tensor_dict["power0_mul"] = tensor_dict["power0_add"].op.input_tensors[0]

            if tensor_dict["power0_mul"].op.input_tensors[0].name == "dequant2_remove_pad":
                tensor_dict = _quant_dequant2(tensor_dict)
                tensor_dict["flag_is_dequant2_power_eltwise"] = True
            else:
                tensor_dict = _quant_dequant1(tensor_dict)
                tensor_dict["flag_is_dequant_power_eltwise"] = True
            tensor_dict["flag_is_dequant_quant"] = False
        elif "min" in tensor_dict["input_ub"].op.input_tensors[0].name:
            tensor_dict["flag_is_quant_relu6_dequant"] = True
            tensor_dict["fusion_type_new"] = 8
            tensor_dict["min"] = tensor_dict["input_ub"].op.input_tensors[0]
            if "max" in tensor_dict["min"].op.input_tensors[0].name:
                tensor_dict["max"] = tensor_dict["min"].op.input_tensors[0]
        elif "mul" in tensor_dict["input_ub"].op.input_tensors[0].name:
            tensor_dict["flag_is_quant_mul_dequant"] = True
            tensor_dict["fusion_type_new"] = 9
            tensor_dict["mul_res"] = tensor_dict["input_ub"].op.input_tensors[0]

            def _check_broadcast_in_mul_res(tensor_dict):
                if "broadcast" in tensor_dict["mul_res"].op.input_tensors[1].name:
                    tensor_dict["broadcast_tensor_0"] = tensor_dict["mul_res"].op.input_tensors[1]
                    tensor_dict["float16_mul_input_tensor"] = tensor_dict["broadcast_tensor_0"].op.input_tensors[0]
                    tensor_dict["flag_is_broadcast"] = True
                else:
                    tensor_dict["float16_mul_input_tensor"] = tensor_dict["mul_res"].op.input_tensors[1]
                return tensor_dict

            tensor_dict = _check_broadcast_in_mul_res(tensor_dict)

        if tensor_dict["input_ub"].op.input_tensors[0].name == "dequant2_remove_pad" or (
                "max" in tensor_dict and tensor_dict["max"].op.input_tensors[0].name == "dequant2_remove_pad"):
            tensor_dict = _quant_dequant2(tensor_dict)
        else:
            tensor_dict = _quant_dequant1(tensor_dict)
    else:
        dict_args = {
            'errCode': 'E67002',
            'op_name': 'depthwise_conv2d',
            'param_name_1': "out.op.attrs['scale'].value",
            'param_name_2': "out.op.attrs['sqrt_mode'].value",
            'expect_param1_value': 1,
            'expect_param2_value': 0,
            'param1_value': out.op.attrs['scale'].value,
            'param2_value': out.op.attrs['sqrt_mode'].value,
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    return tensor_dict


def _elewise_single_relu(out, tensor_dict):
    """
    elewise single relu
    """
    tensor_dict["depthwise_res"] = out.op.input_tensors[0]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[0].name == "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict["depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 1
    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _elewise_deq_sigmiod_mul2(tensor_dict):
    """
    elewise_deq_sigmiod_mul sub part
    """
    if tensor_dict["muls"].op.input_tensors[0].op.tag == "dequant_remove_pad":
        tensor_dict["dequant_remove_pad"] = tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["dequant1"] = tensor_dict["dequant_remove_pad"].op.input_tensors[0]
        tensor_dict["flag_is_dequant_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 12
    elif tensor_dict["muls"].op.input_tensors[0].op.tag == "dequant2_remove_pad":
        tensor_dict["dequant2_remove_pad"] = tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["dequant2"] = tensor_dict["dequant2_remove_pad"].op.input_tensors[0]
        tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
        tensor_dict["flag_is_dequant2_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 13
    if tensor_dict["muls"].op.input_tensors[0].op.tag == "depthwise_conv2d":
        tensor_dict["depthwise_res"] = tensor_dict["muls"].op.input_tensors[0]
        tensor_dict["flag_is_sigmoid_mul"] = True
        tensor_dict["fusion_type_new"] = 3
    else:
        tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
        tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[0].name == "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict["depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 2
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    elif tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    return tensor_dict


def _elewise_deq_sigmiod_mul(out, tensor_dict):
    """
    elewise_deq_sigmiod_mul
    """
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["rec_7"] = out.op.input_tensors[1]
    tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[0]
    tensor_dict["rec_6"] = tensor_dict["rec_7"].op.input_tensors[0]
    tensor_dict["rec_5"] = tensor_dict["rec_6"].op.input_tensors[0]
    tensor_dict["rec_4"] = tensor_dict["rec_5"].op.input_tensors[0]
    tensor_dict["add_2"] = tensor_dict["rec_4"].op.input_tensors[0]
    tensor_dict["rec_3"] = tensor_dict["rec_4"].op.input_tensors[1]
    if cce_conf.is_v200_version_new():
        tensor_dict["rec_2"] = tensor_dict["rec_3"].op.input_tensors[0]
        tensor_dict["rec_1"] = tensor_dict["rec_2"].op.input_tensors[0]
        tensor_dict["rec_0"] = tensor_dict["rec_1"].op.input_tensors[0]
        tensor_dict["rec_n"] = tensor_dict["rec_0"].op.input_tensors[1]
    tensor_dict["exp"] = tensor_dict["add_2"].op.input_tensors[0]
    tensor_dict["muls"] = tensor_dict["exp"].op.input_tensors[0]
    tensor_dict = _elewise_deq_sigmiod_mul2(tensor_dict)
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _elewise_deq_mul(out, tensor_dict):
    """
    elewise_deq_mul
    """
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant_remove_pad"] = out.op.input_tensors[0]
    if "broadcast" in out.op.input_tensors[1].name:
        tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
        tensor_dict["float16_mul_input_tensor"] = tensor_dict["broadcast_tensor_0"].op.input_tensors[0]
        tensor_dict["flag_is_broadcast"] = True
    else:
        tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
    tensor_dict["dequant1"] = tensor_dict["dequant_remove_pad"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    tensor_dict["flag_is_dequant_mul"] = True
    tensor_dict["fusion_type_new"] = 10

    return tensor_dict


def _elewise_deq2_mul(out, tensor_dict):
    """
    _elewise_deq2_mul
    """
    tensor_dict["fused_c_dtype"] = "float16"
    tensor_dict["dequant2_remove_pad"] = out.op.input_tensors[0]
    if "broadcast" in out.op.input_tensors[1].name:
        tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
        tensor_dict["float16_mul_input_tensor"] = tensor_dict["broadcast_tensor_0"].op.input_tensors[0]
        tensor_dict["flag_is_broadcast"] = True
    else:
        tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
    tensor_dict["dequant2"] = tensor_dict["dequant2_remove_pad"].op.input_tensors[0]
    tensor_dict["dequant1"] = tensor_dict["dequant2"].op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["dequant1"].op.input_tensors[0]
    tensor_dict["deq_reg"] = tensor_dict["dequant1"].op.input_tensors[1]
    tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        tensor_dict["flag_is_dequant_bias"] = True
        tensor_dict["mad_after_bias"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["mad_bias"] = tensor_dict["mad_after_bias"].op.input_tensors[0]
        tensor_dict["mad"] = tensor_dict["mad_after_bias"].op.input_tensors[1]
        tensor_dict["mad_bias_ub_brc"] = tensor_dict["mad_bias"].op.input_tensors[0]
        tensor_dict["bias_gm"] = tensor_dict["mad_bias_ub_brc"].op.input_tensors[0]
    else:
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    tensor_dict["flag_is_dequant2_mul"] = True
    tensor_dict["fusion_type_new"] = 11
    return tensor_dict


def _elewise_binary_mul(out, tensor_dict):
    """
    elewise_binary_mul
    """
    if out.op.input_tensors[1].op.name[:3] == "rec":
        tensor_dict = _elewise_deq_sigmiod_mul(out, tensor_dict)
    elif out.op.input_tensors[0].op.tag == 'dequant_remove_pad':
        tensor_dict = _elewise_deq_mul(out, tensor_dict)
    elif out.op.input_tensors[0].op.tag == 'dequant2_remove_pad':
        tensor_dict = _elewise_deq2_mul(out, tensor_dict)
    else:
        tensor_dict["fusion_type_new"] = 2
        tensor_dict["depthwise_res"] = out.op.input_tensors[0]

        if "broadcast" in out.op.input_tensors[1].name:
            tensor_dict["broadcast_tensor_0"] = out.op.input_tensors[1]
            tensor_dict["float16_mul_input_tensor"] = tensor_dict["broadcast_tensor_0"].op.input_tensors[0]
            tensor_dict["flag_is_broadcast"] = True
        else:
            tensor_dict["float16_mul_input_tensor"] = out.op.input_tensors[1]
        tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]

        if tensor_dict["depthwise_res"].op.input_tensors[0].name == "bias_add_vector_cc":
            tensor_dict["bias_add"] = tensor_dict["depthwise_res"].op.input_tensors[0]
            tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
            tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[1]
            tensor_dict["bias_flag"] = True
            tensor_dict["fused_double_operand_num"] = 1
        tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
        tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
        tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
        tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
        tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
        tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _depthwise_conv2d(out, tensor_dict):
    """
    depthwise conv2d
    """
    tensor_dict["mad_ubuf"] = out.op.input_tensors[0]
    if tensor_dict["mad_ubuf"].dtype == "float16" and out.op.attrs['bias_flag'].value == 1 or (
            tensor_dict["mad_ubuf"].dtype != "float16" and out.op.attrs['bias_flag'].value == 1
            and out.op.attrs['dsl_flag'].value == 0):
        tensor_dict["bias_add"] = out.op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[1]
        tensor_dict["bias_flag"] = True

    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        tensor_dict["group_num"] = 2
        tensor_dict["relu_0"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
        tensor_dict["fmap"] = tensor_dict["relu_0"].op.input_tensors[0]
    return tensor_dict


def _elewise_single_vs_min(out, tensor_dict):
    """
    elewise_single_VS_min
    """
    tensor_dict["max_0"] = out.op.input_tensors[0]
    tensor_dict["depthwise_res"] = tensor_dict["max_0"].op.input_tensors[0]
    if tensor_dict["depthwise_res"].op.input_tensors[0].name == "bias_add_vector_cc":
        tensor_dict["bias_add"] = tensor_dict["depthwise_res"].op.input_tensors[0]
        tensor_dict["mad_ubuf"] = tensor_dict["bias_add"].op.input_tensors[0]
        tensor_dict["bias_tensor"] = tensor_dict["bias_add"].op.input_tensors[1]
        tensor_dict["bias_flag"] = True
        tensor_dict["fused_double_operand_num"] = 1
    else:
        tensor_dict["mad_ubuf"] = tensor_dict["depthwise_res"].op.input_tensors[0]
    tensor_dict["mad"] = tensor_dict["mad_ubuf"].op.input_tensors[0]
    tensor_dict["filter_reshape"] = tensor_dict["mad"].op.input_tensors[1]
    tensor_dict["im2col_fractal"] = tensor_dict["mad"].op.input_tensors[0]
    tensor_dict["filter_buf"] = tensor_dict["filter_reshape"].op.input_tensors[0]
    tensor_dict["im2col_row_major"] = tensor_dict["im2col_fractal"].op.input_tensors[0]
    tensor_dict["fmap"] = tensor_dict["im2col_row_major"].op.input_tensors[0]
    return tensor_dict


def _write_select(out, tensor_dict):
    """
    write select
    """
    tensor_dict["flag_is_write_select"] = True
    tensor_dict["write_select"] = out.op.input_tensors[0]

    if out.op.input_tensors[0].op.tag == "quant":
        tensor_dict = _quant(out.op.input_tensors[0], tensor_dict)
    elif out.op.input_tensors[0].op.tag == "dequant2_remove_pad":
        tensor_dict = _dequant2_remove_pad(out.op.input_tensors[0], tensor_dict)
    elif out.op.input_tensors[0].op.tag == "dequant_remove_pad":
        tensor_dict = _dequant_remove_pad(out.op.input_tensors[0], tensor_dict)
    elif out.op.input_tensors[0].op.tag == "requant_remove_pad":
        tensor_dict = _requant_remove_pad(out.op.input_tensors[0], tensor_dict)
    elif out.op.input_tensors[0].op.tag == "elewise_binary_mul":
        tensor_dict = _elewise_binary_mul(out.op.input_tensors[0], tensor_dict)

    else:
        dict_args = {
            'errCode': 'E67001',
            'op_name': 'depthwise_conv2d',
            'prama_name': 'out.op.input_tensors[0].op.tag',
            'tag': out.op.input_tensors[0].op.tag
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    return tensor_dict


def _check_broadcast(tensor_dict, sch, attrs_dict):
    """
    check broadcast
    """
    if tensor_dict["flag_is_broadcast"]:
        float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                [tensor_dict["broadcast_tensor_0"]])
        sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
    else:
        float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                [attrs_dict["mul_ubuf"]])
    return float16_mul_input_ubuf, sch


def _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch):
    """
    set_sch_int32_phase1_dequant
    """
    dequant_ubuf, deq_reg_ubuf, requant_ubuf, req_reg_ubuf, bias_ub = buf
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["write_select"]].compute_inline()
    if tensor_dict["flag_is_dequant"]:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf, tensor_dict["dequant1"])
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
    elif tensor_dict["flag_is_dequant2"]:
        sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf,
                                      (tensor_dict["dequant1"], tensor_dict["dequant2"]))
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant1"]].compute_inline()
    elif tensor_dict["flag_is_requant"]:
        sch[tensor_dict["data_transfer"]].set_scope(cce_params.scope_ubuf)
        req_reg_ubuf = sch.cache_read(tensor_dict["vreq_reg"], cce_params.scope_ubuf, tensor_dict["requant"])
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        sch[tensor_dict["requant"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["requant"]].compute_inline()

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
        sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
        sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
    if tensor_dict["flag_is_requant"]:
        return deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch
    return deq_reg_ubuf, bias_ub, dequant_ubuf, sch


def _set_sch_int32_phase1_deq(tensor_dict, attrs_dict, sch, bias_ub):
    """
    set_sch_int32_phase1_dequant_quant flag is dequant_sqrt
    """
    deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf,
                                  (tensor_dict["dequant1"], tensor_dict["dequant2"]))
    sch[tensor_dict["cast_i8_ub"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["reform_by_vadds"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["input_ub"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["input_ub"]].compute_inline()
    sch[tensor_dict["quant_remove_pad"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["quant_remove_pad"]].compute_inline()
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["write_select"]].compute_inline()
    if tensor_dict["flag_is_quant_relu6_dequant"]:
        sch[tensor_dict["min"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["min"]].compute_inline()
        sch[tensor_dict["max"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["max"]].compute_inline()
    elif tensor_dict["flag_is_quant_mul_dequant"]:
        sch[tensor_dict["mul_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mul_res"]].compute_inline()
        mul_ubuf = sch.cache_write(tensor_dict["mul_res"], cce_params.scope_ubuf)
        attrs_dict["mul_ubuf"] = mul_ubuf
        float16_mul_input_ubuf, sch = _check_broadcast(tensor_dict, sch, attrs_dict)

        tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
    sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["dequant1"]].compute_inline()
    sch[tensor_dict["dequant2"]].compute_inline()
    sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["depthwise_res"]].compute_inline()
    sch[tensor_dict["mad_ubuf"]].compute_inline()
    if tensor_dict["flag_is_dequant_bias"]:
        bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
        sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
        sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
    return deq_reg_ubuf, sch, bias_ub


def _set_sch_int32_phase1_dequant_quant(tensor_dict, attrs_dict, buf, sch):
    """
    set_sch_int32_phase1_dequant_quant
    """
    dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
    if tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        deq_reg_ubuf, sch, bias_ub = _set_sch_int32_phase1_deq(tensor_dict, attrs_dict, sch, bias_ub)
    elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf, tensor_dict["dequant1"])
        sch[tensor_dict["cast_i8_ub"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["reform_by_vadds"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["input_ub"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["quant_remove_pad"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["quant_remove_pad"]].compute_inline()
        if tensor_dict["flag_is_write_select"]:
            sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["write_select"]].compute_inline()
        if tensor_dict["flag_is_quant_relu6_dequant"]:
            sch[tensor_dict["min"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["min"]].compute_inline()
            sch[tensor_dict["max"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["max"]].compute_inline()
        elif tensor_dict["flag_is_quant_mul_dequant"]:
            sch[tensor_dict["mul_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mul_res"]].compute_inline()
            mul_ubuf = sch.cache_write(tensor_dict["mul_res"], cce_params.scope_ubuf)
            attrs_dict["mul_ubuf"] = mul_ubuf
            float16_mul_input_ubuf, sch = _check_broadcast(tensor_dict, sch, attrs_dict)

            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant1"]].compute_inline()
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        sch[tensor_dict["mad_ubuf"]].compute_inline()
        if tensor_dict["flag_is_dequant_bias"]:
            bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
            sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
            sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
    else:
        dict_args = {
            'errCode': 'E67003',
            'op_name': 'depthwise_conv2d',
            'prama_name': 'tensor_dict["flag_is_quant_sqrt"]'
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    return deq_reg_ubuf, bias_ub, dequant_ubuf, sch


def _set_sch_int32_phase1_dequant_mul(tensor_dict, attrs_dict, out, buf, sch):
    """
    set_sch_int32_phase1_dequant_mul
    """
    dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
    if tensor_dict["flag_is_write_select"] is True:
        sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["write_select"]].compute_inline()
        out = out.op.input_tensors[0]

    if tensor_dict["flag_is_dequant_mul"]:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf, tensor_dict["dequant1"])
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        sch[mul_ubuf].compute_inline()

        attrs_dict["mul_ubuf"] = mul_ubuf
        if tensor_dict["flag_is_broadcast"]:
            float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                    [tensor_dict["broadcast_tensor_0"]])
            sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
        else:
            float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                    [attrs_dict["mul_ubuf"]])
        tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
        sch[tensor_dict["dequant_remove_pad"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant_remove_pad"]].compute_inline()
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_ubuf"]].compute_inline()
        dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        sch[dequant_ubuf].compute_inline()
    elif tensor_dict["flag_is_dequant2_mul"]:
        sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf,
                                      (tensor_dict["dequant1"], tensor_dict["dequant2"]))
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        sch[mul_ubuf].compute_inline()
        attrs_dict["mul_ubuf"] = mul_ubuf

        if tensor_dict["flag_is_broadcast"]:
            float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                    [tensor_dict["broadcast_tensor_0"]])
            sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
        else:
            float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                    [attrs_dict["mul_ubuf"]])

        tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
        sch[tensor_dict["dequant2_remove_pad"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant2_remove_pad"]].compute_inline()
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_ubuf"]].compute_inline()
        sch[tensor_dict["dequant1"]].compute_inline()
        sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant2"]].compute_inline()
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
        sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
        sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)

    return deq_reg_ubuf, bias_ub, dequant_ubuf, sch


def _set_sch_int32_phase1_dequant2_sigmoid_mul(tensor_dict, attrs_dict, out, sch):
    """
    set_sch_int32_phase1_dequant2_sigmoid_mul
    """
    sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
    deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf,
                                  (tensor_dict["dequant1"], tensor_dict["dequant2"]))
    sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["depthwise_res"]].compute_inline()
    mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
    sch[mul_ubuf].compute_inline()
    attrs_dict["mul_ubuf"] = mul_ubuf
    float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                            [attrs_dict["mul_ubuf"]])
    tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
    sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["rec_7"]].compute_inline()
    sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["rec_6"]].compute_inline()
    sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["rec_5"]].compute_inline()
    sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["rec_4"]].compute_inline()
    sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["rec_3"]].compute_inline()
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["rec_2"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_2"]].compute_inline()
        sch[tensor_dict["rec_1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_1"]].compute_inline()
        sch[tensor_dict["rec_0"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_0"]].compute_inline()
        sch[tensor_dict["rec_n"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_n"]].compute_inline()
    sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["muls"]].compute_inline()
    sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["exp"]].compute_inline()
    sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["add_2"]].compute_inline()
    sch[tensor_dict["dequant2_remove_pad"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["dequant2_remove_pad"]].compute_inline()
    sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["mad_ubuf"]].compute_inline()
    sch[tensor_dict["dequant1"]].compute_inline()
    sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["dequant2"]].compute_inline()
    return deq_reg_ubuf, tensor_dict, sch


def _set_sch_int32_phase1_dequant_sigmoid_mul(tensor_dict, attrs_dict, out, buf, sch):
    """
    set_sch_int32_phase1_dequant_sigmoid_mul
    """
    dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
    if tensor_dict["flag_is_dequant_sigmoid_mul"]:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf, tensor_dict["dequant1"])
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        sch[mul_ubuf].compute_inline()
        attrs_dict["mul_ubuf"] = mul_ubuf
        float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                [attrs_dict["mul_ubuf"]])
        tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
        sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_7"]].compute_inline()
        sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_6"]].compute_inline()
        sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_5"]].compute_inline()
        sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_4"]].compute_inline()
        sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["rec_3"]].compute_inline()
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["rec_2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_2"]].compute_inline()
            sch[tensor_dict["rec_1"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_1"]].compute_inline()
            sch[tensor_dict["rec_0"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_0"]].compute_inline()
            sch[tensor_dict["rec_n"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_n"]].compute_inline()
        sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["muls"]].compute_inline()
        sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["exp"]].compute_inline()
        sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["add_2"]].compute_inline()
        sch[tensor_dict["dequant_remove_pad"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant_remove_pad"]].compute_inline()
        sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_ubuf"]].compute_inline()
        dequant_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        sch[dequant_ubuf].compute_inline()
    elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        deq_reg_ubuf, tensor_dict, sch = _set_sch_int32_phase1_dequant2_sigmoid_mul(tensor_dict, attrs_dict, out, sch)
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
        sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
        sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)
    return deq_reg_ubuf, bias_ub, dequant_ubuf, sch


def _set_sch_int32_phase1_deqaunt_power_eltwise(tensor_dict, buf, sch):
    dequant_ubuf, deq_reg_ubuf, _, _, bias_ub = buf
    sch[tensor_dict["cast_i8_ub"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["reform_by_vadds"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["input_ub"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["input_ub"]].compute_inline()
    sch[tensor_dict["quant_remove_pad"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["quant_remove_pad"]].compute_inline()
    if tensor_dict["flag_is_eltwisce_case"] == "0":
        sch[tensor_dict["mul_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mul_res"]].compute_inline()
    elif tensor_dict["flag_is_eltwisce_case"] == "1":
        sch[tensor_dict["eltwise_max"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["eltwise_max"]].compute_inline()
    elif tensor_dict["flag_is_eltwisce_case"] == "1_2":
        sch[tensor_dict["eltwise_add"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["eltwise_add"]].compute_inline()
        sch[tensor_dict["eltwise_mul_left"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["eltwise_mul_left"]].compute_inline()
        sch[tensor_dict["eltwise_mul_right"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["eltwise_mul_right"]].compute_inline()
    else:
        sch[tensor_dict["eltwise_add"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["eltwise_add"]].compute_inline()
    sch[tensor_dict["power1_add"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["power1_add"]].compute_inline()
    sch[tensor_dict["power1_mul"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["power1_mul"]].compute_inline()
    sch[tensor_dict["relu_min"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["relu_min"]].compute_inline()
    sch[tensor_dict["relu_max"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["relu_max"]].compute_inline()
    sch[tensor_dict["power0_add"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["power0_add"]].compute_inline()
    sch[tensor_dict["power0_mul"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["power0_mul"]].compute_inline()

    if tensor_dict["flag_is_dequant2_power_eltwise"]:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf,
                                      (tensor_dict["dequant1"], tensor_dict["dequant2"]))
        sch[tensor_dict["dequant2"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["dequant2"]].compute_inline()

    else:
        deq_reg_ubuf = sch.cache_read(tensor_dict["deq_reg"], cce_params.scope_ubuf, tensor_dict["dequant1"])

    sch[tensor_dict["dequant1"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["dequant1"]].compute_inline()
    sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["depthwise_res"]].compute_inline()
    sch[tensor_dict["mad_ubuf"]].set_scope(cce_params.scope_ubuf)
    sch[tensor_dict["mad_ubuf"]].compute_inline()
    if tensor_dict["flag_is_dequant_bias"]:
        bias_ub = sch.cache_read(tensor_dict["bias_gm"], cce_params.scope_ubuf, [tensor_dict["mad_bias_ub_brc"]])
        sch[tensor_dict["mad_bias_ub_brc"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["mad_bias"]].set_scope(cce_params.scope_cc)
        sch[tensor_dict["mad_after_bias"]].set_scope(cce_params.scope_cc)

    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["write_select"]].compute_inline()
    return deq_reg_ubuf, bias_ub, dequant_ubuf, sch


# phase1, set scope
def _set_sch_int32_phase1(tensor_dict, attrs_dict, out, sch):
    """
    set schedule int32 and phase1
    """
    dequant_ubuf = None
    deq_reg_ubuf = None
    requant_ubuf = None
    req_reg_ubuf = None
    bias_ub = None
    buf = (dequant_ubuf, deq_reg_ubuf, requant_ubuf, req_reg_ubuf, bias_ub)

    if tensor_dict["flag_is_dequant"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant2"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant(tensor_dict, out, buf, sch)
    elif tensor_dict["flag_is_requant"]:
        deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch = _set_sch_int32_phase1_dequant(
            tensor_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant_quant"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant_quant(
            tensor_dict, attrs_dict, buf, sch)
    elif tensor_dict["flag_is_dequant_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant_mul(
            tensor_dict, attrs_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant2_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant_mul(
            tensor_dict, attrs_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant_sigmoid_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant_sigmoid_mul(
            tensor_dict, attrs_dict, out, buf, sch)
    elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_dequant_sigmoid_mul(
            tensor_dict, attrs_dict, out, buf, sch)
    elif tensor_dict["flag_is_eltwisce_case"]:
        deq_reg_ubuf, bias_ub, dequant_ubuf, sch = _set_sch_int32_phase1_deqaunt_power_eltwise(
            tensor_dict, buf, sch)
    return deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch


# phase2, compute at
def _sch_flag_is_dequant_power_eltwise(sch, tensor_dict, attrs_dict, res_cut_dict):
    """
    schedule flag is dequant power relu6 power eltwise quant
    """
    sch[tensor_dict["cast_i8_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["reform_by_vadds"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["input_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["flag_is_eltwisce_case"] == "0":
        sch[tensor_dict["mul_res"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    elif tensor_dict["flag_is_eltwisce_case"] == "1":
        sch[tensor_dict["eltwise_max"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    elif tensor_dict["flag_is_eltwisce_case"] == "1_2":
        sch[tensor_dict["eltwise_add"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["eltwise_mul_left"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["eltwise_mul_right"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    elif tensor_dict["flag_is_eltwisce_case"] == "1_1":
        sch[tensor_dict["eltwise_add"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])

    sch[tensor_dict["power1_add"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["power1_mul"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["relu_min"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["relu_max"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["power0_add"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["power0_mul"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["flag_is_dequant2_power_eltwise"]:
        sch[tensor_dict["dequant2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])

    sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(sch[tensor_dict["mad_bias"]].op.axis[0],
                                             sch[tensor_dict["mad_bias"]].op.axis[1],
                                             sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                                             sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
    if tensor_dict["flag_is_write_select"] is True:
        sch[tensor_dict["write_select"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["flag_is_dequant2_power_eltwise"]:
        sch[tensor_dict["dequant2"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    else:
        sch[tensor_dict["dequant1"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    return a2_axis, a3_axis, sch


def _sch_flag_is_dequant2(sch, tensor_dict, attrs_dict, res_cut_dict):
    """
    schedule flag is dequant2
    """
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(sch[tensor_dict["mad_bias"]].op.axis[0],
                                             sch[tensor_dict["mad_bias"]].op.axis[1],
                                             sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                                             sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
    if tensor_dict["flag_is_dequant2_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_7"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_6"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_5"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_4"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_3"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["rec_2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_0"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_n"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["muls"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["add_2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["exp"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant2"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))

    return a2_axis, a3_axis, sch


def _sch_flag_is_dequant(sch, tensor_dict, attrs_dict, res_cut_dict):
    """
    schedule flag is dequant
    """
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if not tensor_dict["flag_is_dequant_sigmoid_mul"] and not tensor_dict[
            "flag_is_dequant2_sigmoid_mul"] and not tensor_dict["flag_is_dequant_mul"]:
        sch[attrs_dict["dequant_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(sch[tensor_dict["mad_bias"]].op.axis[0],
                                             sch[tensor_dict["mad_bias"]].op.axis[1],
                                             sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                                             sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        if tensor_dict["flag_is_dequant_mul"]:
            sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))
    if tensor_dict["flag_is_dequant_sigmoid_mul"]:
        sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_7"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_6"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_5"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_4"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["rec_3"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["rec_2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_0"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["rec_n"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["muls"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["add_2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["exp"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))

    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["dequant1"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))

    return a2_axis, a3_axis, sch


def _sch_flag_is_dequant_quant(sch, double_buffer_flag, tensor_dict, attrs_dict, res_cut_dict):
    """
    schedule flag is dequant_quant
    """
    a2_axis = None
    a3_axis = None

    def _sch_deqaunt_quant_compute_at():
        if tensor_dict["flag_is_dequant_bias"]:
            a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(sch[tensor_dict["mad_bias"]].op.axis[3], 16)
            sch[tensor_dict["mad_bias"]].reorder(sch[tensor_dict["mad_bias"]].op.axis[0],
                                                 sch[tensor_dict["mad_bias"]].op.axis[1],
                                                 sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                                                 sch[tensor_dict["mad_bias"]].op.axis[4])
            sch[tensor_dict["mad_after_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])

            sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
            sch[tensor_dict["mad_bias_ub_brc"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])

            sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_cccut_i"])
            sch[attrs_dict["deq_reg_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_cccut_i"])

            sch[tensor_dict["mad_bias_ub_brc"]].double_buffer()
            sch[tensor_dict["mad_bias_ub_brc"]].preload()
            if double_buffer_flag["CL0_pbuffer"] == 2:
                sch[tensor_dict["mad_bias"]].double_buffer()
                sch[tensor_dict["mad_bias"]].preload()

            sch[tensor_dict["cast_i8_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["reform_by_vadds"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            sch[tensor_dict["input_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["max"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            if tensor_dict["flag_is_quant_mul_dequant"]:
                sch[attrs_dict["mul_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[attrs_dict["out"]],
                                                                      res_cut_dict["res_mcut_iio"])
                sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, 16), (1, BLOCK_SIZE))
        else:
            dict_args = {
                'errCode': 'E67003',
                'op_name': 'depthwise_conv2d',
                'prama_name': 'tensor_dict["flag_is_dequant_bias"]'
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        return a2_axis, a3_axis, sch

    if tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        a2_axis, a3_axis, sch = _sch_deqaunt_quant_compute_at()
        sch[tensor_dict["dequant2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant2"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        a2_axis, a3_axis, sch = _sch_deqaunt_quant_compute_at()
        sch[tensor_dict["dequant1"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["dequant1"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    else:
        dict_args = {
            'errCode': 'E67003',
            'op_name': 'depthwise_conv2d',
            'prama_name': 'tensor_dict["flag_is_quant_sqrt"]'
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    return a2_axis, a3_axis, sch


def _sch_flag_is_requant(sch, tensor_dict, attrs_dict, res_cut_dict):
    """
    schedule flag is requant
    """
    a2_axis = None
    a3_axis = None
    sch[attrs_dict["req_reg_ubuf"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["data_transfer"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])

    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        a2_axis, a3_axis = sch[tensor_dict["mad_bias"]].split(sch[tensor_dict["mad_bias"]].op.axis[3], 16)
        sch[tensor_dict["mad_bias"]].reorder(sch[tensor_dict["mad_bias"]].op.axis[0],
                                             sch[tensor_dict["mad_bias"]].op.axis[1],
                                             sch[tensor_dict["mad_bias"]].op.axis[2], a2_axis, a3_axis,
                                             sch[tensor_dict["mad_bias"]].op.axis[4])
        sch[tensor_dict["mad_after_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[tensor_dict["mad_bias_ub_brc"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
        sch[attrs_dict["bias_ub"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_io"])
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
    sch[tensor_dict["data_transfer"]].buffer_align((1, 1), (1, 1), (1, 1), (1, TILING_INT8_M), (1, BLOCK_SIZE))
    return a2_axis, a3_axis, sch


def _set_sch_int32_phase2(mad_dtype, double_buffer_flag, tensor_dict, attrs_dict, res_cut_dict, sch):
    """
    set schedule int32 and phase2
    """
    a2_axis = None
    a3_axis = None
    if mad_dtype == "int32":
        if tensor_dict["flag_is_dequant2"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant_quant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant_quant(sch, double_buffer_flag, tensor_dict, attrs_dict,
                                                               res_cut_dict)
        elif tensor_dict["flag_is_dequant_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant_sigmoid_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_dequant2_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_requant"]:
            a2_axis, a3_axis, sch = _sch_flag_is_requant(sch, tensor_dict, attrs_dict, res_cut_dict)

        elif tensor_dict["flag_is_dequant2_sigmoid_mul"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant2(sch, tensor_dict, attrs_dict, res_cut_dict)
        elif tensor_dict["flag_is_eltwisce_case"]:
            a2_axis, a3_axis, sch = _sch_flag_is_dequant_power_eltwise(sch, tensor_dict, attrs_dict, res_cut_dict)
    return a2_axis, a3_axis, sch


def _emit_insn_dequant1(tensor_dict, sch):
    """
    emit_insn_dequant1
    """
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[3], 'dma_copy')
    else:
        sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')


# phase 3, emit insn phase
def _flag_is_dequant_sqrt_bias(tensor_dict, sch, attrs_dict):
    """
    flag is dequant sqrt
    """
    sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
    sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
    sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    _emit_insn_dequant1(tensor_dict, sch)
    sch[tensor_dict["dequant2"]].emit_insn(sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].emit_insn(sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
    if tensor_dict["flag_is_quant_relu6_dequant"]:
        sch[tensor_dict["min"]].emit_insn(sch[tensor_dict["min"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["max"]].emit_insn(sch[tensor_dict["max"]].op.axis[0], 'vector_auto')
    if tensor_dict["flag_is_quant_mul_dequant"]:
        sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(tensor_dict["float16_mul_input_ubuf"].op.axis[0],
                                                             'dma_copy')
        sch[attrs_dict["mul_ubuf"]].emit_insn(sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[tensor_dict["input_ub"]].emit_insn(sch[tensor_dict["input_ub"]].op.axis[0], 'dma_padding')

    ndim = len(sch[tensor_dict["reform_by_vadds"]].op.axis)
    factor = 16
    coo, _ = sch[tensor_dict["reform_by_vadds"]].split(sch[tensor_dict["reform_by_vadds"]].op.axis[ndim - 1], factor)
    axis_list = sch[tensor_dict["reform_by_vadds"]].op.axis[0:ndim - 1]
    sch[tensor_dict["reform_by_vadds"]].reorder(coo, *axis_list)
    sch[tensor_dict["reform_by_vadds"]].emit_insn(sch[tensor_dict["reform_by_vadds"]].op.axis[3], 'vector_auto')

    sch[tensor_dict["cast_i8_ub"]].emit_insn(sch[tensor_dict["cast_i8_ub"]].op.axis[0], 'vector_conv')
    return sch


def _flag_is_dequant_quant(tensor_dict, sch, attrs_dict):
    """
    flag is dequant_quant
    """
    if tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        if tensor_dict["flag_is_dequant_bias"]:
            sch = _flag_is_dequant_sqrt_bias(tensor_dict, sch, attrs_dict)
    elif not tensor_dict["flag_is_dequant_sqrt"] and not tensor_dict["flag_is_quant_sqrt"]:
        if tensor_dict["flag_is_dequant_bias"]:
            sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
            sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
            sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')

            sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
            if tensor_dict["flag_is_write_select"]:
                sch[tensor_dict["write_select"]].emit_insn(sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
            if tensor_dict["flag_is_quant_relu6_dequant"]:
                sch[tensor_dict["min"]].emit_insn(sch[tensor_dict["min"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["max"]].emit_insn(sch[tensor_dict["max"]].op.axis[0], 'vector_auto')
            if tensor_dict["flag_is_quant_mul_dequant"]:
                sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(tensor_dict["float16_mul_input_ubuf"].op.axis[0],
                                                                     'dma_copy')
                sch[attrs_dict["mul_ubuf"]].emit_insn(sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
            _emit_insn_dequant1(tensor_dict, sch)
            sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
            sch[tensor_dict["input_ub"]].emit_insn(sch[tensor_dict["input_ub"]].op.axis[0], 'dma_padding')

            ndim = len(sch[tensor_dict["reform_by_vadds"]].op.axis)
            factor = 16
            coo, _ = sch[tensor_dict["reform_by_vadds"]].split(sch[tensor_dict["reform_by_vadds"]].op.axis[ndim - 1],
                                                               factor)
            axis_list = sch[tensor_dict["reform_by_vadds"]].op.axis[0:ndim - 1]
            sch[tensor_dict["reform_by_vadds"]].reorder(coo, *axis_list)
            sch[tensor_dict["reform_by_vadds"]].emit_insn(sch[tensor_dict["reform_by_vadds"]].op.axis[3], 'vector_auto')

            sch[tensor_dict["cast_i8_ub"]].emit_insn(sch[tensor_dict["cast_i8_ub"]].op.axis[0], 'vector_conv')
    else:
        dict_args = {
            'errCode': 'E67003',
            'op_name': 'depthwise_conv2d',
            'prama_name': 'tensor_dict["flag_is_quant_sqrt"]'
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    return sch


def _flag_is_dequant_sigmoid_mul(tensor_dict, sch, attrs_dict):
    """
    flag is dequant_sigmoid_mul
    """
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
    else:
        if tensor_dict['sca_vec_flag'] == 0:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
        else:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        sch[tensor_dict["dequant2"]].emit_insn(sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_7"]].emit_insn(sch[tensor_dict["rec_7"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_6"]].emit_insn(sch[tensor_dict["rec_6"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_5"]].emit_insn(sch[tensor_dict["rec_5"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_4"]].emit_insn(sch[tensor_dict["rec_4"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["rec_3"]].emit_insn(sch[tensor_dict["rec_3"]].op.axis[0], 'vector_auto')
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["rec_2"]].emit_insn(sch[tensor_dict["rec_2"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["rec_1"]].emit_insn(sch[tensor_dict["rec_1"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["rec_0"]].emit_insn(sch[tensor_dict["rec_0"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["rec_n"]].emit_insn(sch[tensor_dict["rec_n"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["muls"]].emit_insn(sch[tensor_dict["muls"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["add_2"]].emit_insn(sch[tensor_dict["add_2"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["exp"]].emit_insn(sch[tensor_dict["exp"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(tensor_dict["float16_mul_input_ubuf"].op.axis[0], 'dma_copy')
    sch[attrs_dict["mul_ubuf"]].emit_insn(sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    return sch


def _flag_is_dequant_mul(tensor_dict, sch, attrs_dict):
    """
    flag is dequant_mul
    """
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
    else:
        if tensor_dict['sca_vec_flag'] == 0:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
        else:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_dequant2_mul"]:
        sch[tensor_dict["dequant2"]].emit_insn(sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].emit_insn(sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
    sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(tensor_dict["float16_mul_input_ubuf"].op.axis[0], 'dma_copy')
    sch[attrs_dict["mul_ubuf"]].emit_insn(sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
    return sch


def _flag_is_dequant_power_eltwise(tensor_dict, sch, attrs_dict):
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')

    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
    else:
        if tensor_dict['sca_vec_flag'] == 0:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
        else:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_dequant2_power_eltwise"]:
        sch[tensor_dict["dequant2"]].emit_insn(sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')

    if tensor_dict["flag_is_eltwisce_case"] == "0":
        sch[tensor_dict["mul_res"]].emit_insn(sch[tensor_dict["mul_res"]].op.axis[0], 'vector_auto')
    elif tensor_dict["flag_is_eltwisce_case"] == "1":
        sch[tensor_dict["eltwise_max"]].emit_insn(sch[tensor_dict["eltwise_max"]].op.axis[0], 'vector_auto')
    elif tensor_dict["flag_is_eltwisce_case"] == "1_2":
        sch[tensor_dict["eltwise_add"]].emit_insn(sch[tensor_dict["eltwise_add"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["eltwise_mul_left"]].emit_insn(sch[tensor_dict["eltwise_mul_left"]].op.axis[0], 'vector_auto')
        sch[tensor_dict["eltwise_mul_right"]].emit_insn(sch[tensor_dict["eltwise_mul_right"]].op.axis[0], 'vector_auto')
    elif tensor_dict["flag_is_eltwisce_case"] == "1_1":
        sch[tensor_dict["eltwise_add"]].emit_insn(sch[tensor_dict["eltwise_add"]].op.axis[0], 'vector_auto')

    sch[tensor_dict["power1_add"]].emit_insn(sch[tensor_dict["power1_add"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["power1_mul"]].emit_insn(sch[tensor_dict["power1_mul"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["relu_min"]].emit_insn(sch[tensor_dict["relu_min"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["relu_max"]].emit_insn(sch[tensor_dict["relu_max"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["power0_add"]].emit_insn(sch[tensor_dict["power0_add"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["power0_mul"]].emit_insn(sch[tensor_dict["power0_mul"]].op.axis[0], 'vector_auto')
    sch[tensor_dict["input_ub"]].emit_insn(sch[tensor_dict["input_ub"]].op.axis[0], 'dma_padding')
    ndim = len(sch[tensor_dict["reform_by_vadds"]].op.axis)
    factor = 16
    coo, _ = sch[tensor_dict["reform_by_vadds"]].split(sch[tensor_dict["reform_by_vadds"]].op.axis[ndim - 1], factor)
    axis_list = sch[tensor_dict["reform_by_vadds"]].op.axis[0:ndim - 1]
    sch[tensor_dict["reform_by_vadds"]].reorder(coo, *axis_list)
    sch[tensor_dict["reform_by_vadds"]].emit_insn(sch[tensor_dict["reform_by_vadds"]].op.axis[3], 'vector_auto')
    sch[tensor_dict["cast_i8_ub"]].emit_insn(sch[tensor_dict["cast_i8_ub"]].op.axis[0], 'vector_conv')

    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].emit_insn(sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')

    return sch


def _phase3_avoid_complexity(tensor_dict, sch, attrs_dict):
    """
    phase3 avoid complexity
    """
    sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
    if cce_conf.is_v200_version_new():
        sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
    else:
        if tensor_dict['sca_vec_flag'] == 0:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
        else:
            sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    sch[tensor_dict["dequant2"]].emit_insn(sch[tensor_dict["dequant2"]].op.axis[0], 'vector_auto')

    return sch


def _emit_insn_phase3(tensor_dict, sch, attrs_dict):
    """
    emit_insn_phase3
    """
    if tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1:
        sch[tensor_dict["mad_after_bias"]].emit_insn(tensor_dict["mad_after_bias"].op.axis[0], 'phony_insn')
        sch[tensor_dict["mad_bias"]].emit_insn(attrs_dict["a2_axis"], 'dma_copy')
        sch[tensor_dict["mad_bias_ub_brc"]].emit_insn(sch[tensor_dict["mad_bias_ub_brc"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["bias_ub"]].emit_insn(sch[attrs_dict["bias_ub"]].op.axis[0], 'dma_copy')
    if tensor_dict["flag_is_dequant"]:
        sch[attrs_dict["dequant_ubuf"]].emit_insn(sch[attrs_dict["dequant_ubuf"]].op.axis[0], 'dma_copy')
        sch[attrs_dict["deq_reg_ubuf"]].emit_insn(sch[attrs_dict["deq_reg_ubuf"]].op.axis[0], 'dma_copy')
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["dequant1"]].emit_insn(sch[tensor_dict["dequant1"]].op.axis[0], 'dma_copy')
        else:
            if tensor_dict['sca_vec_flag'] == 0:
                sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'scalar')
            else:
                sch[tensor_dict["dequant1"]].pragma(sch[tensor_dict["dequant1"]].op.axis[3], 'deq_scale', 'vector')
    elif tensor_dict["flag_is_dequant2"]:
        sch = _phase3_avoid_complexity(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_requant"]:
        sch[attrs_dict["req_reg_ubuf"]].emit_insn(sch[attrs_dict["req_reg_ubuf"]].op.axis[0], 'dma_copy')
        if cce_conf.is_v200_version_new():
            sch[tensor_dict["data_transfer"]].emit_insn(sch[tensor_dict["data_transfer"]].op.axis[3], 'dma_copy')
        else:
            if tensor_dict['sca_vec_flag'] == 0:
                sch[tensor_dict["requant"]].pragma(sch[tensor_dict["requant"]].op.axis[3], 'deq_scale', 'scalar')
            else:
                sch[tensor_dict["requant"]].pragma(sch[tensor_dict["requant"]].op.axis[3], 'deq_scale', 'vector')
    if tensor_dict["flag_is_write_select"]:
        sch[tensor_dict["write_select"]].emit_insn(sch[tensor_dict["write_select"]].op.axis[0], 'dma_copy')
    return sch


def _set_sch_int32_phase3(tensor_dict, sch, attrs_dict, res_cut_dict, out):
    """
    set schedule int32 and phase3
    """
    if tensor_dict["flag_is_dequant2"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_requant"]:
        sch = _emit_insn_phase3(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_quant"]:
        sch = _flag_is_dequant_quant(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_mul"] or tensor_dict["flag_is_dequant2_mul"]:
        sch = _flag_is_dequant_mul(tensor_dict, sch, attrs_dict)
    elif tensor_dict["flag_is_dequant_sigmoid_mul"] or tensor_dict["flag_is_dequant2_sigmoid_mul"]:
        sch = _flag_is_dequant_sigmoid_mul(tensor_dict, sch, attrs_dict)
    elif attrs_dict["out"].op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        sch[attrs_dict["relu_ubuf"]].emit_insn(sch[attrs_dict["relu_ubuf"]].op.axis[0], 'vector_auto')
    elif attrs_dict["out"].op.tag == "elewise_binary_mul":

        def attrs_dict_tag_elewise(tensor_dict, sch, attrs_dict):
            if tensor_dict["flag_is_sigmoid_mul"]:
                sch[tensor_dict["rec_7"]].emit_insn(sch[tensor_dict["rec_7"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["rec_6"]].emit_insn(sch[tensor_dict["rec_6"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["rec_5"]].emit_insn(sch[tensor_dict["rec_5"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["rec_4"]].emit_insn(sch[tensor_dict["rec_4"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["rec_3"]].emit_insn(sch[tensor_dict["rec_3"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["muls"]].emit_insn(sch[tensor_dict["muls"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["add_2"]].emit_insn(sch[tensor_dict["add_2"]].op.axis[0], 'vector_auto')
                sch[tensor_dict["exp"]].emit_insn(sch[tensor_dict["exp"]].op.axis[0], 'vector_auto')
            else:
                sch[tensor_dict["float16_mul_input_ubuf"]].emit_insn(tensor_dict["float16_mul_input_ubuf"].op.axis[0],
                                                                     'dma_copy')
            sch[attrs_dict["mul_ubuf"]].emit_insn(sch[attrs_dict["mul_ubuf"]].op.axis[0], 'vector_auto')
            return sch

        sch = attrs_dict_tag_elewise(tensor_dict, sch, attrs_dict)
    elif attrs_dict["out"].op.tag == "elewise_single_VS_min":
        sch[tensor_dict["max_0"]].emit_insn(sch[tensor_dict["max_0"]].op.axis[0], 'vector_auto')
        sch[attrs_dict["relu_ubuf"]].emit_insn(sch[attrs_dict["relu_ubuf"]].op.axis[0], 'vector_auto')
    elif tensor_dict["flag_is_eltwisce_case"]:
        sch = _flag_is_dequant_power_eltwise(tensor_dict, sch, attrs_dict)
    # STRIDE WRITE
    if out.op.tag == "write_select":
        align_length = int(out.op.attrs["HWC0"])
        sch[out].bind_buffer(out.op.axis[1], align_length, 0)
    sch[out].emit_insn(res_cut_dict["res_mcut_iii"], 'dma_copy')

    return sch


def _dequant_out_cg(mad_dtype, attrs_dict, block_dim_tiling):
    """
    dequant out cg
    """
    if mad_dtype == "int32":
        if attrs_dict["deq_reg_ubuf"] is not None:
            _, dequant_out_cg, _, _, _ = (int(i.value) for i in attrs_dict["deq_reg_ubuf"].shape)
            if dequant_out_cg % 2 != 0:
                block_dim_tiling[3] = 1
        elif attrs_dict["req_reg_ubuf"] is not None:
            _, requant_out_cg, _, _, _ = (int(i.value) for i in attrs_dict["req_reg_ubuf"].shape)
            if requant_out_cg % 2 != 0:
                block_dim_tiling[3] = 1

    return block_dim_tiling


def _avoid_complexity_mul(out, tensor_dict, attrs_dict, sch):
    """
    avoid complexity mul
    """
    if not tensor_dict["flag_is_dequant_sigmoid_mul"] and not tensor_dict[
            "flag_is_dequant2_sigmoid_mul"] and not tensor_dict["flag_is_dequant_mul"] and not tensor_dict[
                "flag_is_dequant2_mul"]:
        if tensor_dict["flag_is_sigmoid_mul"]:
            sch[tensor_dict["rec_7"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_7"]].compute_inline()
            sch[tensor_dict["rec_6"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_6"]].compute_inline()
            sch[tensor_dict["rec_5"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_5"]].compute_inline()
            sch[tensor_dict["rec_4"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_4"]].compute_inline()
            sch[tensor_dict["rec_3"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["rec_3"]].compute_inline()
            sch[tensor_dict["muls"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["muls"]].compute_inline()
            sch[tensor_dict["exp"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["exp"]].compute_inline()
            sch[tensor_dict["add_2"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["add_2"]].compute_inline()
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            attrs_dict["mul_ubuf"] = mul_ubuf
        else:
            sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
            sch[tensor_dict["depthwise_res"]].compute_inline()
            mul_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
            attrs_dict["mul_ubuf"] = mul_ubuf
            if tensor_dict["flag_is_broadcast"]:
                float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                        [tensor_dict["broadcast_tensor_0"]])
                sch[tensor_dict["broadcast_tensor_0"]].compute_inline()
            else:
                float16_mul_input_ubuf = sch.cache_read(tensor_dict["float16_mul_input_tensor"], cce_params.scope_ubuf,
                                                        [attrs_dict["mul_ubuf"]])

            tensor_dict["float16_mul_input_ubuf"] = float16_mul_input_ubuf
    return tensor_dict, attrs_dict, sch


def _l1_fusion_phase1(sch, tensor_dict):
    """
    l1 fusion phase1
    """
    input_mem_type = int(DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    valid_shape = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    l1_fusion_type = int(DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))

    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])

    if valid_shape:
        tensor_dict["fmap_valid_shape"] = valid_shape
    if int(input_mem_type) == 1:
        sch[tensor_dict["fmap"]].set_scope(cce_params.scope_cbuf_fusion)
        a_cbuf_nc1hwc0 = sch.cache_read(tensor_dict["fmap"], cce_params.scope_cbuf_fusion,
                                        [tensor_dict["im2col_row_major"]])

        if valid_shape:
            if len(tensor_dict["fmap"].op.shape) == 6:
                sch[a_cbuf_nc1hwc0].buffer_tile((None, None), (None, None), (None, None),
                                                (-pad_top, tensor_dict["fmap"].op.shape[3] + pad_top + pad_bottom),
                                                (-pad_left, tensor_dict["fmap"].op.shape[4] + pad_left + pad_right),
                                                (None, None))
            else:
                sch[a_cbuf_nc1hwc0].buffer_tile(
                    (None, None), (None, None), (-pad_top, tensor_dict["fmap"].op.shape[2] + pad_top + pad_bottom),
                    (-pad_left, tensor_dict["fmap"].op.shape[3] + pad_left + pad_right), (None, None))
    else:
        # need L1 fusion buffer to storage L1 data
        if l1_fusion_type in (0, 1):
            # DDR in and select
            if valid_shape:
                sch[tensor_dict["fusion_fmap_select"]].set_scope(cce_params.scope_cbuf_fusion)
                a_cbuf_nc1hwc0 = tensor_dict["fusion_fmap_select"]
            else:
                a_cbuf_nc1hwc0 = sch.cache_read(tensor_dict["fmap"], cce_params.scope_cbuf_fusion,
                                                [tensor_dict["im2col_row_major"]])
        else:
            # DDR in and not select, not fusion
            a_cbuf_nc1hwc0 = sch.cache_read(tensor_dict["fmap"], cce_params.scope_cbuf,
                                            [tensor_dict["im2col_row_major"]])
    return sch, a_cbuf_nc1hwc0


def _save_workspace(tensor_dict, a_cbuf_nc1hwc0, sch):
    """
    save workspace
    """
    input_mem_type = int(DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    l1_fusion_type = int(DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    fmap_l1_addr_flag = int(DepthwiseConv2dParam.fusion_para.get("fmap_l1_addr_flag"))
    fmap_l1_valid_size = int(DepthwiseConv2dParam.fusion_para.get("fmap_l1_valid_size"))
    l1_tensor_map = {}
    if fmap_l1_addr_flag == "nothing":
        l1_tensor_map = None
    else:
        if int(input_mem_type) in (0, 2) and int(l1_fusion_type) in (0, 1):
            l1_tensor_map[tensor_dict["fmap"]] = a_cbuf_nc1hwc0
            if fmap_l1_valid_size > 0:
                sch[a_cbuf_nc1hwc0].set_storage_bound(fmap_l1_valid_size)
        else:
            l1_tensor_map = None

    return l1_tensor_map


def _fmp_emit_insn(sch, a_cbuf_nc1hwc0):
    """
    fmp_emit_insn
    """
    input_mem_type = int(DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    l1_fusion_type = int(DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    valid_shape = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    # no l1fusion
    if l1_fusion_type == -1:
        sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'dma_copy')
    # L1 in, do nothing
    if input_mem_type == 1:
        sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'phony_insn')
    else:
        if valid_shape:
            sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'dma_copy', {"mem_align": 1})
            sch[a_cbuf_nc1hwc0].pragma(a_cbuf_nc1hwc0.op.axis[0], 'jump_data', 1)
        else:
            sch[a_cbuf_nc1hwc0].emit_insn(a_cbuf_nc1hwc0.op.axis[0], 'dma_copy')
            sch[a_cbuf_nc1hwc0].pragma(a_cbuf_nc1hwc0.op.axis[0], 'jump_data', 1)
    return sch


def _set_sch_ph1(out, sch, tensor_dict, attrs_dict, mad_dtype):
    """
    set schedule
    """
    if "addr_type" in out.op.attrs:
        out_addr_type = int(out.op.attrs["addr_type"])
        if out_addr_type == 1:
            sch[out].set_scope(cce_params.scope_cbuf_fusion)
            # when ub fusion, depthwise out may not be final out
            tensor_dict["output_memory_type"] = 1
    if True in [tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"], tensor_dict["flag_is_requant"]]:
        sch[tensor_dict["mad_ubuf"]].compute_inline()
    if tensor_dict["bias_flag"]:
        bias_ubuf = sch.cache_read(tensor_dict["bias_tensor"], cce_params.scope_ubuf, [tensor_dict["bias_add"]])
        attrs_dict["bias_ubuf"] = bias_ubuf
        sch[bias_ubuf].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["bias_add"]].set_scope(cce_params.scope_ubuf)
    if out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].compute_inline()
        relu_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        attrs_dict["relu_ubuf"] = relu_ubuf
    if out.op.tag == "elewise_binary_mul":
        tensor_dict, attrs_dict, sch = _avoid_complexity_mul(out, tensor_dict, attrs_dict, sch)
    if out.op.tag == "elewise_single_VS_min":
        sch[tensor_dict["max_0"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["depthwise_res"]].set_scope(cce_params.scope_ubuf)
        sch[tensor_dict["max_0"]].compute_inline()
        sch[tensor_dict["depthwise_res"]].compute_inline()
        relu_ubuf = sch.cache_write(out, cce_params.scope_ubuf)
        attrs_dict["relu_ubuf"] = relu_ubuf
    c_0 = C0_16
    if mad_dtype == "int32":
        deq_reg_ubuf, req_reg_ubuf, bias_ub, dequant_ubuf, requant_ubuf, sch = _set_sch_int32_phase1(
            tensor_dict, attrs_dict, out, sch)
        attrs_dict["bias_ub"] = bias_ub
        attrs_dict["deq_reg_ubuf"] = deq_reg_ubuf
        attrs_dict["dequant_ubuf"] = dequant_ubuf
        attrs_dict["req_reg_ubuf"] = req_reg_ubuf
        attrs_dict["requant_ubuf"] = requant_ubuf
        c_0 = C0_32
    attrs_dict["c_0"] = c_0
    return sch, tensor_dict, attrs_dict


def get_shape_type():
    """
    get relative shape, type and size
    """
    is_overload = False
    offset = DepthwiseConv2dParam.fusion_para.get("slice_offset")
    valid_shape = DepthwiseConv2dParam.fusion_para.get("valid_shape")
    input_mem_type = int(DepthwiseConv2dParam.fusion_para.get("input_memory_type"))
    output_mem_type = int(DepthwiseConv2dParam.fusion_para.get("output_memory_type"))
    l1_fusion_type = int(DepthwiseConv2dParam.fusion_para.get("l1_fusion_type"))
    l1_valid_size = int(DepthwiseConv2dParam.fusion_para.get("fmap_l1_valid_size"))
    return is_overload, offset, valid_shape, input_mem_type, output_mem_type, l1_fusion_type, l1_valid_size


def prepare_tensor_attrs(out, input_mem_type, output_mem_type, l1_fusion_type, l1_valid_size):
    """
    Prepare tensors and attrs
    """
    tensor_dict = _set_tensor_by_op_tag(out)
    attrs_dict = {}
    attrs_dict["out"] = out
    tensor_dict["input_memory_type"] = input_mem_type
    tensor_dict["output_memory_type"] = output_mem_type
    tensor_dict["l1_fusion_type"] = l1_fusion_type
    tensor_dict["fm_l1_valid_size"] = l1_valid_size
    tensor_dict["fmap_valid_shape"] = None
    return attrs_dict, tensor_dict


def get_sch_cache(sch, tensor_dict):
    """
    get schedule cache
    """
    a_cbuf_row_major = sch.cache_write(tensor_dict["im2col_row_major"], cce_params.scope_cbuf)
    sch[tensor_dict["im2col_row_major"]].compute_inline()
    a_ca = sch.cache_write(tensor_dict["im2col_fractal"], cce_params.scope_ca)
    sch[tensor_dict["im2col_fractal"]].compute_inline()

    b_cbuf = sch.cache_read(tensor_dict["filter_buf"], cce_params.scope_cbuf, [tensor_dict["filter_reshape"]])
    b_cb = sch.cache_write(tensor_dict["filter_reshape"], cce_params.scope_cb)
    sch[tensor_dict["filter_reshape"]].compute_inline()

    mad_cc = sch.cache_write(tensor_dict["mad"], cce_params.scope_cc)
    sch[tensor_dict["mad"]].compute_inline()

    mad_dtype = mad_cc.dtype
    sch[tensor_dict["mad_ubuf"]].set_scope(cce_params.scope_ubuf)
    return a_cbuf_row_major, a_ca, b_cbuf, b_cb, mad_cc, mad_dtype, sch


def _default_tiling(tensor_dict, fmap_w, pad_top, pad_bottom, kernel_w, kernel_h, stride_w):
    """
    default tiling
    """
    tiling = {}
    if tensor_dict["fused_c_dtype"] == "int32":
        dtype = "int8"
    else:
        dtype = "float16"
    m_bit_length = {"float32": 32, "float16": 16, "uint8": 8, "int8": 8, "uint4": 4, "int4": 4}
    m_bit_ratio = {"int32": 4, "float32": 4, "float16": 2, "uint8": 1, "int8": 1, "uint4": 1.0 / 2, "int4": 1.0 / 2}
    wo_shape = (fmap_w + (2 * pad_top) - kernel_w) // stride_w + 1
    gen_m_target = 0
    for m_target in range(32, 0, -1):
        tmp1 = ((m_target * m_bit_length['float16']) + wo_shape - 1) // wo_shape
        tmp2 = ((tmp1 * pad_bottom) + kernel_h) * fmap_w
        max_feature_map = tmp2 * 2 * m_bit_ratio[dtype]
        if int(max_feature_map) < L1_MEM_LIMIT:
            gen_m_target = m_target
            break

    tiling["AL1_shape"] = [1, 1, 1, 1]
    tiling["BL1_shape"] = None
    tiling["AL0_matrix"] = [gen_m_target, 1, 16, 16, 1, 1]
    tiling["BL0_matrix"] = [1, 2, 16, 16, 1, 1]
    tiling["CL0_matrix"] = [2, gen_m_target, 16, 16, 1, 1]
    tiling["CUB_matrix"] = [2, gen_m_target, 16, 16, 1, 1]
    tiling["manual_pingpong_buffer"] = {
        'AL1_pbuffer': 1,
        'BL1_pbuffer': 1,
        'AL0_pbuffer': 1,
        'BL0_pbuffer': 1,
        'CL0_pbuffer': 1,
        'CUB_pbuffer': 1,
        'UBG_pbuffer': 1
    }
    tiling["AUB_channel_wise_flag"] = None
    tiling["BUB_channel_wise_flag"] = None
    tiling["A_overhead_opt_flag"] = 0
    tiling["B_overhead_opt_flag"] = 0
    tiling["batch_bef_group_flag"] = 0
    tiling["n_bef_batch_flag"] = 0
    tiling["n_bef_group_flag"] = 0
    tiling["block_dim"] = [1, 1, 1, 1]

    return tiling


def _relu_mul_handle(out, sch, attrs_dict, tensor_dict, res_cut_dict):
    """
    relu multi handle
    """
    res_mcut_iio = res_cut_dict["res_mcut_iio"]
    if out.op.tag in ["elewise_single_relu", "elewise_single_lrelu"]:
        sch[attrs_dict["relu_ubuf"]].compute_at(sch[out], res_mcut_iio)
        sch[attrs_dict["relu_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
    if out.op.tag == "elewise_binary_mul":
        if not tensor_dict["flag_is_dequant_sigmoid_mul"] and not tensor_dict["flag_is_dequant2_sigmoid_mul"]:
            if tensor_dict["flag_is_sigmoid_mul"]:
                sch[tensor_dict["rec_7"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["rec_6"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["rec_5"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["rec_4"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["rec_3"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["muls"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["add_2"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
                sch[tensor_dict["exp"]].compute_at(sch[attrs_dict["out"]], res_cut_dict["res_mcut_iio"])
            else:
                sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["mul_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))

        elif not tensor_dict["flag_is_dequant_mul"] and not tensor_dict["flag_is_dequant2_mul"]:
            sch[attrs_dict["mul_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[tensor_dict["float16_mul_input_ubuf"]].compute_at(sch[out], res_mcut_iio)
            sch[attrs_dict["mul_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
    if out.op.tag == "elewise_single_VS_min":
        sch[tensor_dict["max_0"]].compute_at(sch[out], res_mcut_iio)
        sch[attrs_dict["relu_ubuf"]].compute_at(sch[out], res_mcut_iio)
        sch[attrs_dict["relu_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
    return sch


def _set_double_buffer_flag(tiling):
    """
    set double buffer flag
    """
    double_buffer_flag = {
        'AL1_pbuffer': False,
        'BL1_pbuffer': False,
        'AL0_pbuffer': False,
        'BL0_pbuffer': False,
        'CL0_pbuffer': False,
        'CUB_pbuffer': False,
        'UBG_pbuffer': False,
    }

    def _db_flag_handle(tiling):
        """
        db flag handle
        """
        if "manual_pingpong_buffer" in tiling:
            double_buffer_flag = tiling["manual_pingpong_buffer"]
        return double_buffer_flag

    double_buffer_flag = _db_flag_handle(tiling)
    return double_buffer_flag


def _set_res_cut_dict(sch, out, tensor_dict, a_l1_tiling, a_l0_tiling, b_l1_tiling, b_l0_tiling, c_ub_tiling,
                      c_l0_tiling, block_dim_tiling):
    """
    set res_cut_dict
    """
    res_cut_dict = {}
    res_bcut_o, res_bcut_i = sch[out].split(out.op.axis[0], factor=a_l1_tiling[2])
    res_bcut_io, res_bcut_ii = sch[out].split(res_bcut_i, factor=a_l0_tiling[4])
    res_bcut_iio, res_bcut_iii = sch[out].split(res_bcut_ii, factor=c_ub_tiling[4])
    res_cut_dict["res_bcut_o"] = res_bcut_o
    res_cut_dict["res_bcut_i"] = res_bcut_i
    res_cut_dict["res_bcut_io"] = res_bcut_io
    res_cut_dict["res_bcut_ii"] = res_bcut_ii
    res_cut_dict["res_bcut_iio"] = res_bcut_iio
    res_cut_dict["res_bcut_iii"] = res_bcut_iii

    if tensor_dict["flag_is_requant"]:
        requant_o, requant_i = sch[tensor_dict["data_transfer"]].split(tensor_dict["data_transfer"].op.axis[-1],
                                                                       factor=16)
        sch[tensor_dict["data_transfer"]].reorder(tensor_dict["data_transfer"].op.axis[0],
                                                  tensor_dict["data_transfer"].op.axis[1],
                                                  tensor_dict["data_transfer"].op.axis[2], requant_o,
                                                  tensor_dict["data_transfer"].op.axis[-2], requant_i)

    # m
    res_mcut_o, res_mcut_i = sch[out].split(out.op.axis[3], factor=a_l1_tiling[1] * c_l0_tiling[1] * a_l0_tiling[2])
    res_mcut_io, res_mcut_ii = sch[out].split(res_mcut_i, factor=a_l0_tiling[0] * a_l0_tiling[2])
    res_mcut_iio, res_mcut_iii = sch[out].split(res_mcut_ii, factor=c_ub_tiling[1] * c_ub_tiling[2])
    res_cut_dict["res_mcut_o"] = res_mcut_o
    res_cut_dict["res_mcut_i"] = res_mcut_i
    res_cut_dict["res_mcut_io"] = res_mcut_io
    res_cut_dict["res_mcut_ii"] = res_mcut_ii
    res_cut_dict["res_mcut_iio"] = res_mcut_iio
    res_cut_dict["res_mcut_iii"] = res_mcut_iii
    # n
    res_ncut_o, res_ncut_i = sch[out].split(out.op.axis[2], factor=b_l1_tiling[1])
    res_ncut_io, res_ncut_ii = sch[out].split(res_ncut_i, factor=b_l0_tiling[1])
    res_ncut_iio, res_ncut_iii = sch[out].split(res_ncut_ii, factor=c_ub_tiling[0])
    res_cut_dict["res_ncut_o"] = res_ncut_o
    res_cut_dict["res_ncut_i"] = res_ncut_i
    res_cut_dict["res_ncut_io"] = res_ncut_io
    res_cut_dict["res_ncut_ii"] = res_ncut_ii
    res_cut_dict["res_ncut_iio"] = res_ncut_iio
    res_cut_dict["res_ncut_iii"] = res_ncut_iii

    sch[out].reorder(out.op.axis[1], res_bcut_o, res_ncut_o, res_mcut_o, res_bcut_io, res_ncut_io, res_mcut_io,
                     res_bcut_iio, res_ncut_iio, res_mcut_iio, res_bcut_iii, res_ncut_iii, res_mcut_iii, out.op.axis[4])
    res_bbcut_o, res_bbcut_i = sch[out].split(res_bcut_o, nparts=block_dim_tiling[0])
    res_nncut_o, res_nncut_i = sch[out].split(res_ncut_o, nparts=block_dim_tiling[1])
    res_mmcut_o, res_mmcut_i = sch[out].split(res_mcut_o, nparts=block_dim_tiling[2])
    res_cccut_o, res_cccut_i = sch[out].split(out.op.axis[1], nparts=block_dim_tiling[3])
    res_cut_dict["res_bbcut_o"] = res_bbcut_o
    res_cut_dict["res_bbcut_i"] = res_bbcut_i
    res_cut_dict["res_nncut_o"] = res_nncut_o
    res_cut_dict["res_nncut_i"] = res_nncut_i
    res_cut_dict["res_mmcut_o"] = res_mmcut_o
    res_cut_dict["res_mmcut_i"] = res_mmcut_i
    res_cut_dict["res_cccut_o"] = res_cccut_o
    res_cut_dict["res_cccut_i"] = res_cccut_i
    sch[out].reorder(res_cccut_o, res_bbcut_o, res_nncut_o, res_mmcut_o, res_cccut_i, res_bbcut_i, res_nncut_i,
                     res_mmcut_i)
    return res_cut_dict


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=too-many-branches
def depthwise_conv2d_schedule(out):
    """
    depthwise conv2d schedule
    """
    is_overload, offset, valid_shape, input_mem_type, output_mem_type, l1_fusion_type, l1_valid_size = get_shape_type()
    sch = create_schedule(out.op)
    # Prepare tensors.
    attrs_dict, tensor_dict = prepare_tensor_attrs(out, input_mem_type, output_mem_type, l1_fusion_type, l1_valid_size)

    # set data flow
    if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        pre_relu_ubuf = sch.cache_read(tensor_dict["fmap"], cce_params.scope_ubuf, [tensor_dict["relu_0"]])
        pre_relu_cbuf = sch.cache_read(tensor_dict["relu_0"], cce_params.scope_cbuf, [tensor_dict["im2col_row_major"]])
        sch[tensor_dict["relu_0"]].set_scope(cce_params.scope_ubuf)
        fmp_shape = tensor_dict["fmap"].op.shape
    else:
        sch, a_cbuf_nc1hwc0 = _l1_fusion_phase1(sch, tensor_dict)
        L1CommonParam.l1_fusion_tensors_map = _save_workspace(tensor_dict, a_cbuf_nc1hwc0, sch)
        fmp_shape = a_cbuf_nc1hwc0.shape

    a_cbuf_row_major, a_ca, b_cbuf, b_cb, mad_cc, mad_dtype, sch = get_sch_cache(sch, tensor_dict)
    # out to L1
    sch, tensor_dict, attrs_dict = _set_sch_ph1(out, sch, tensor_dict, attrs_dict, mad_dtype)

    if len(fmp_shape) == 6:
        _, _, _, fmap_h, fmap_w, fmap_c0 = (int(i.value) for i in fmp_shape)
    else:
        _, _, fmap_h, fmap_w, fmap_c0 = (int(i.value) for i in fmp_shape)
    pad_top = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][0])
    pad_right = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][3])
    pad_left = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][2])
    pad_bottom = (int)(tensor_dict["mad_ubuf"].op.attrs['padding'][1])
    kernel_w = (int)(tensor_dict["mad_ubuf"].op.attrs['kernel_w'])
    kernel_h = (int)(tensor_dict["mad_ubuf"].op.attrs['kernel_h'])
    stride_w = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][1])
    stride_h = (int)(tensor_dict["mad_ubuf"].op.attrs['stride'][0])
    howo_one_flag = (tensor_dict["mad_ubuf"].op.attrs['howo_one_flag'])
    if hasattr(howo_one_flag, "value"):
        howo_one_flag = howo_one_flag.value
    wo_shape = (fmap_w + pad_left + pad_right - kernel_w) // stride_w + 1
    ho_shape = (fmap_h + pad_top + pad_bottom - kernel_h) // stride_h + 1
    # get tiling params
    tiling = _get_tiling_fetch(mad_dtype, tensor_dict)

    if tiling["AL0_matrix"][2] == 32 or not isinstance(tiling["AL1_shape"], list):
        tiling = _default_tiling(tensor_dict, fmap_w, pad_top, pad_bottom, kernel_w, kernel_h, stride_w)

    a_l1_tiling = tiling['AL1_shape']
    b_l1_tiling = tiling['BL1_shape']
    a_l0_tiling = tiling['AL0_matrix']
    b_l0_tiling = tiling['BL0_matrix']
    c_l0_tiling = tiling['CL0_matrix']
    c_ub_tiling = tiling['CUB_matrix']
    block_dim_tiling = tiling['block_dim']
    block_dim_tiling = _dequant_out_cg(mad_dtype, attrs_dict, block_dim_tiling)

    def _tiling_handle(a_l1_tiling, b_l1_tiling, b_l0_tiling, is_overload):
        """
        tiling handle
        """
        if block_dim_tiling[1] > 1 or (block_dim_tiling[2] > 1 and (stride_h < kernel_h or stride_w < kernel_w)):
            is_overload = True

        if a_l1_tiling == []:
            a_l1_tiling = [
                fmap_c0 * kernel_w * kernel_h,
                (ho_shape * wo_shape + (c_l0_tiling[1] * TILING_INT8_M) - 1) // (c_l0_tiling[1] * TILING_INT8_M), 1
            ]

        if b_l1_tiling == [] or b_l1_tiling is None:
            b_l1_tiling = [fmap_c0 * kernel_w * kernel_h, fmap_c0 // fmap_c0, 1]

        if b_l0_tiling == []:
            b_l0_tiling = [a_l0_tiling[1], 1, TILING_INT8_N, TILING_INT8_N, 1, a_l0_tiling[5]]
        return is_overload, a_l1_tiling, b_l1_tiling, b_l0_tiling

    is_overload, a_l1_tiling, b_l1_tiling, b_l0_tiling = _tiling_handle(a_l1_tiling, b_l1_tiling, b_l0_tiling,
                                                                        is_overload)

    # --------------------------double buffer------------------------
    double_buffer_flag = _set_double_buffer_flag(tiling)
    # L0C
    # batch
    mad_cc_bcut_o, mad_cc_bcut_ii = sch[mad_cc].split(mad_cc.op.axis[0], factor=a_l0_tiling[4])

    # m
    mad_cc_mcut_o, mad_cc_mcut_ii = sch[mad_cc].split(mad_cc.op.axis[3], factor=a_l0_tiling[0] * a_l0_tiling[2])

    # n
    mad_cc_ncut_o, mad_cc_ncut_ii = sch[mad_cc].split(mad_cc.op.axis[2], factor=b_l0_tiling[1])

    # k
    mad_cc_kcut_o, mad_cc_kcut_ii = sch[mad_cc].split(mad_cc.op.reduce_axis[0], factor=b_l0_tiling[0])

    sch[mad_cc].reorder(mad_cc_bcut_o, mad_cc.op.axis[1], mad_cc_ncut_o, mad_cc_mcut_o, mad_cc_kcut_o, mad_cc_bcut_ii,
                        mad_cc_ncut_ii, mad_cc_mcut_ii, mad_cc.op.axis[4], mad_cc_kcut_ii, mad_cc.op.reduce_axis[1])
    sch[a_ca].compute_at(sch[mad_cc], mad_cc_kcut_o)
    sch = _set_a_cbuf_row_major(mad_dtype, a_cbuf_row_major, wo_shape, sch)
    # batch
    res_cut_dict = _set_res_cut_dict(sch, out, tensor_dict, a_l1_tiling, a_l0_tiling, b_l1_tiling, b_l0_tiling,
                                     c_ub_tiling, c_l0_tiling, block_dim_tiling)
    blocks = block_dim_tiling[0] * block_dim_tiling[1] * block_dim_tiling[2] * block_dim_tiling[3]

    batch_cout_fused = sch[out].fuse(res_cut_dict["res_cccut_o"], res_cut_dict["res_bbcut_o"],
                                     res_cut_dict["res_nncut_o"], res_cut_dict["res_mmcut_o"])
    noo_true, _ = sch[out].split(batch_cout_fused, nparts=blocks)
    block = tvm.thread_axis("blockIdx.x")
    sch[out].bind(noo_true, block)
    sch[b_cbuf].compute_at(sch[out], res_cut_dict["res_cccut_i"])

    def _spe_handle():
        """
        spe handle
        """
        if tiling['BL0_matrix'] == [] and howo_one_flag:
            sch[b_cb].compute_at(sch[out], res_cut_dict["res_cccut_i"])
        else:
            sch[b_cb].compute_at(sch[mad_cc], mad_cc_kcut_o)

        if out.op.tag == "elewise_single_VS_min":
            sch[tensor_dict["bias_add"]].mem_unique()
            sch[tensor_dict["max_0"]].mem_unique()
            sch[attrs_dict["relu_ubuf"]].mem_unique()

        if tiling['BL1_shape'] is None:
            sch[b_cbuf].compute_inline()
        if tensor_dict["bias_flag"]:
            sch[attrs_dict["bias_ubuf"]].compute_at(sch[out], res_cut_dict["res_cccut_i"])
            sch[tensor_dict["bias_add"]].compute_at(sch[out], res_cut_dict["res_mcut_iio"])
        if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
            sch[pre_relu_ubuf].compute_at(sch[out], res_cut_dict["res_mmcut_i"])
            sch[tensor_dict["relu_0"]].compute_at(sch[out], res_cut_dict["res_mmcut_i"])
            sch[pre_relu_cbuf].compute_at(sch[out], res_cut_dict["res_mmcut_i"])
        else:
            sch[a_cbuf_nc1hwc0].compute_at(sch[out], res_cut_dict["res_mmcut_i"])
        sch[a_cbuf_row_major].compute_at(sch[out], res_cut_dict["res_mmcut_i"])

        sch[mad_cc].compute_at(sch[out], res_cut_dict["res_mcut_io"])
        return sch

    def _int32_spe_handle():
        """
        int32_spe_handle
        """
        if mad_dtype == "int32":
            if double_buffer_flag["AL0_pbuffer"] == 2:
                sch[a_cbuf_row_major].double_buffer()
            if "fmap" in tensor_dict and tensor_dict["fmap"].dtype == out.dtype:
                if double_buffer_flag["BL1_pbuffer"] == 2:
                    sch[b_cbuf].preload()
                if double_buffer_flag["AL0_pbuffer"] == 2:
                    sch[a_cbuf_row_major].preload()
                if "relu" not in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
                    if double_buffer_flag["AL1_pbuffer"] == 2:
                        sch[a_cbuf_nc1hwc0].preload()
            if tensor_dict["flag_is_quant_relu6_dequant"] or tensor_dict["flag_is_dequant_sigmoid_mul"] or tensor_dict[
                    "flag_is_dequant2_sigmoid_mul"]:
                sch[attrs_dict["deq_reg_ubuf"]].double_buffer()
                sch[attrs_dict["deq_reg_ubuf"]].preload()
                sch[attrs_dict["bias_ub"]].double_buffer()
                sch[attrs_dict["bias_ub"]].preload()
        return sch

    sch = _spe_handle()
    sch = _int32_spe_handle()

    if ([
            tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"], tensor_dict["flag_is_dequant_quant"],
            tensor_dict["flag_is_requant"], tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
            tensor_dict["flag_is_dequant_sigmoid_mul"], tensor_dict["flag_is_dequant2_sigmoid_mul"],
            tensor_dict["flag_is_eltwisce_case"]
    ] == [False] * 9):
        sch[tensor_dict["mad_ubuf"]].compute_at(sch[out], res_cut_dict["res_mcut_iio"])
        sch[tensor_dict["mad_ubuf"]].buffer_align((1, 1), (1, 1), (1, 1), (1, attrs_dict["c_0"]), (1, BLOCK_SIZE))
    a2_axis, a3_axis, sch = _set_sch_int32_phase2(mad_dtype, double_buffer_flag, tensor_dict, attrs_dict, res_cut_dict,
                                                  sch)
    attrs_dict["a2_axis"] = a2_axis
    attrs_dict["a3_axis"] = a3_axis

    sch = _relu_mul_handle(out, sch, attrs_dict, tensor_dict, res_cut_dict)

    def _set_double_buffer(input_module, sch_to_deal):
        """
        set double buffer
        """
        if input_module == 2:
            sch_to_deal.double_buffer()

    # al1
    if "relu" not in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
        _set_double_buffer(double_buffer_flag["AL1_pbuffer"], sch[a_cbuf_nc1hwc0])
    _set_double_buffer(double_buffer_flag["BL1_pbuffer"], sch[b_cbuf])
    _set_double_buffer(double_buffer_flag["AL0_pbuffer"], sch[a_ca])
    _set_double_buffer(double_buffer_flag["BL0_pbuffer"], sch[b_cb])
    if [
            tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"], tensor_dict["flag_is_dequant_quant"],
            tensor_dict["flag_is_requant"], tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
            tensor_dict["flag_is_dequant_sigmoid_mul"], tensor_dict["flag_is_dequant2_sigmoid_mul"],
            tensor_dict["flag_is_eltwisce_case"]
    ] == [False] * 9:
        _set_double_buffer(double_buffer_flag["CUB_pbuffer"], sch[tensor_dict["mad_ubuf"]])
    setfmatrix_dict = {
        "conv_kernel_h": tensor_dict["mad_ubuf"].op.attrs['kernel_h'],
        "conv_kernel_w": tensor_dict["mad_ubuf"].op.attrs['kernel_w'],
        "conv_padding_top": tensor_dict["mad_ubuf"].op.attrs['padding'][0],
        "conv_padding_bottom": tensor_dict["mad_ubuf"].op.attrs['padding'][1],
        "conv_padding_left": tensor_dict["mad_ubuf"].op.attrs['padding'][2],
        "conv_padding_right": tensor_dict["mad_ubuf"].op.attrs['padding'][3],
        "conv_stride_h": tensor_dict["mad_ubuf"].op.attrs['stride'][0],
        "conv_stride_w": tensor_dict["mad_ubuf"].op.attrs['stride'][1],
        "conv_fm_c": fmap_c0,
        "conv_fm_h": fmap_h,
        "conv_fm_w": fmap_w,
        "conv_dilation_h": tensor_dict["mad_ubuf"].op.attrs['dilation'][0],
        "conv_dilation_w": tensor_dict["mad_ubuf"].op.attrs['dilation'][1]
    }

    def _valid_shape_handle():
        if valid_shape:
            setfmatrix_dict["conv_fm_h"] = valid_shape[2]
            if input_mem_type == 1:
                setfmatrix_dict["conv_fm_offset_h"] = offset[2]
        return setfmatrix_dict

    setfmatrix_dict = _valid_shape_handle()

    # emit insn
    def _bias_relu_emit_insn(sch):
        if tensor_dict["bias_flag"]:
            sch[attrs_dict["bias_ubuf"]].emit_insn(sch[attrs_dict["bias_ubuf"]].op.axis[0], 'dma_copy')
            sch[tensor_dict["bias_add"]].emit_insn(sch[tensor_dict["bias_add"]].op.axis[0], 'vector_auto')

        if "relu" in tensor_dict["im2col_row_major"].op.input_tensors[0].name:
            sch[tensor_dict["relu_0"]].emit_insn(tensor_dict["relu_0"].op.axis[0], 'vector_auto')
            sch[pre_relu_ubuf].emit_insn(pre_relu_ubuf.op.axis[0], 'dma_copy')
            sch[pre_relu_cbuf].emit_insn(pre_relu_cbuf.op.axis[0], 'dma_copy')
        else:
            sch = _fmp_emit_insn(sch, a_cbuf_nc1hwc0)
        return sch

    sch = _bias_relu_emit_insn(sch)

    sch[a_cbuf_row_major].emit_insn(a_cbuf_row_major.op.axis[1], 'set_fmatrix', setfmatrix_dict)
    sch[a_ca].emit_insn(a_ca.op.axis[1], 'im2col')
    sch[b_cbuf].emit_insn(b_cbuf.op.axis[0], 'dma_copy')
    sch[b_cb].emit_insn(b_cb.op.axis[0], 'dma_copy')
    mad_dict = {"mad_pattern": cce_params.CONV_MODE, "k_outer": [mad_cc_kcut_o]}
    if ((True in [
            tensor_dict["flag_is_dequant2"], tensor_dict["flag_is_dequant"], tensor_dict["flag_is_requant"],
            tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant_sigmoid_mul"],
            tensor_dict["flag_is_dequant2_mul"], tensor_dict["flag_is_dequant2_sigmoid_mul"],
            tensor_dict["flag_is_eltwisce_case"]
    ] and tensor_dict["depthwise_res"].op.attrs['bias_flag'].value == 1) or tensor_dict["flag_is_dequant_bias"]):
        mad_dict["init_bias"] = 1
        sch[tensor_dict["mad_bias"]].reused_by(tensor_dict["mad_after_bias"], mad_cc)
        if double_buffer_flag["CL0_pbuffer"] == 2:
            sch[tensor_dict["mad_bias"]].double_buffer()
            sch[mad_cc].double_buffer()
            sch[tensor_dict["mad_after_bias"]].double_buffer()
    sch[mad_cc].emit_insn(mad_cc_bcut_ii, 'mad', mad_dict)
    if [
            tensor_dict["flag_is_dequant"], tensor_dict["flag_is_dequant2"], tensor_dict["flag_is_dequant_quant"],
            tensor_dict["flag_is_requant"], tensor_dict["flag_is_dequant_mul"], tensor_dict["flag_is_dequant2_mul"],
            tensor_dict["flag_is_dequant_sigmoid_mul"], tensor_dict["flag_is_dequant2_sigmoid_mul"],
            tensor_dict["flag_is_eltwisce_case"]
    ] == [False] * 9:
        sch[tensor_dict["mad_ubuf"]].emit_insn(sch[tensor_dict["mad_ubuf"]].op.axis[0], 'dma_copy')
    attrs_dict["out"] = out
    attrs_dict["mad_dtype"] = mad_dtype
    sch = _set_sch_int32_phase3(tensor_dict, sch, attrs_dict, res_cut_dict, out)

    set_pragma_for_cache_read_mode(is_overload, sch[out], res_cut_dict["res_mmcut_i"])

    return sch


def pragma_overload_filter(condition0, condition1, stage, first_axis):
    """
    pragma overload filter
    """
    is_overload = False

    if condition0 > 1 or condition1 > 1:
        is_overload = True

    set_pragma_for_cache_read_mode(is_overload, stage, first_axis)


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=too-many-branches
def depthwise_conv2d_backprop_filter_d_schedule(depthwise_dfilter_res):
    """
    schedule of depthwise conv2d backprop filter

    dout->dout_transpose->dout_fractal--|
                                         |->mad_res->depthwise_dfilter
    fmap->fmap_transpose->feature_col--  /                   |
    ----->feature_col_pad---------------/          depthwise_dfilter_res
    reduce : K=NHoWo M=16 N=HwWw*16

    Parameters
    ----------
    depthwise_dfilter_res : tvm tensor
        output tensor(dfilter) in tvm.

    Returns
    -------
    s: tvm schedule
        the tensor of output.
    """
    sch = create_schedule(depthwise_dfilter_res.op)

    # get tensors form input
    depthwise_dfilter = depthwise_dfilter_res.op.input_tensors[0]
    mad_res = depthwise_dfilter.op.input_tensors[0]
    dout_fractal = mad_res.op.input_tensors[0]
    dout_transpose = dout_fractal.op.input_tensors[0]
    dout = dout_transpose.op.input_tensors[0]
    feature_col_pad = mad_res.op.input_tensors[1]
    feature_col = feature_col_pad.op.input_tensors[0]
    fmap_transpose = feature_col.op.input_tensors[0]
    fmap = fmap_transpose.op.input_tensors[0]

    # define stages
    fmap_cbuf_nc1hwc0 = sch.cache_write(fmap_transpose, cce_params.scope_cbuf)
    sch[fmap_transpose].compute_inline()
    fmap_cbuf_row_major = sch.cache_write(feature_col, cce_params.scope_cbuf)
    sch[feature_col].compute_inline()
    fmap_cb = sch.cache_write(feature_col_pad, cce_params.scope_cb)
    sch[feature_col_pad].compute_inline()
    dout_cbuf = sch.cache_write(dout_transpose, cce_params.scope_cbuf)
    sch[dout_transpose].compute_inline()
    dout_ca = sch.cache_write(dout_fractal, cce_params.scope_ca)
    sch[dout_fractal].compute_inline()
    mad_ubuf = sch.cache_write(mad_res, cce_params.scope_ubuf)
    sch[mad_res].compute_inline()
    mad_cc = sch.cache_write(mad_ubuf, cce_params.scope_cc)
    depthwise_dfilter_ubuf = sch.cache_write(depthwise_dfilter, cce_params.scope_ubuf)
    sch[depthwise_dfilter].compute_inline()

    # get shape values
    fmap_shape = [int(i.value) for i in fmap.shape]
    dout_shape = [int(i.value) for i in dout.shape]

    # NCgC1HiWiC0 (C1 = 1)
    fmap_n, fmap_cg, _, fmap_h, fmap_w, fmap_c0 = fmap_shape
    # NCgC1HoWoC0 (C1 = 1)
    dout_n, dout_cg, dout_c1, dout_h, dout_w, dout_c0 = dout_shape
    filter_h = depthwise_dfilter_res.op.attrs['kernel_h'].value
    filter_w = depthwise_dfilter_res.op.attrs['kernel_w'].value
    stride_h = int(depthwise_dfilter_res.op.attrs['stride'][0].value)
    stride_w = int(depthwise_dfilter_res.op.attrs['stride'][1].value)
    dilation_h = int(depthwise_dfilter_res.op.attrs['dilations'][0].value)
    dilation_w = int(depthwise_dfilter_res.op.attrs['dilations'][1].value)
    pad_top = int(depthwise_dfilter_res.op.attrs['padding'][0].value)
    pad_bottom = int(depthwise_dfilter_res.op.attrs['padding'][1].value)
    pad_left = int(depthwise_dfilter_res.op.attrs['padding'][2].value)
    pad_right = int(depthwise_dfilter_res.op.attrs['padding'][3].value)
    kernel_name = depthwise_dfilter_res.op.attrs['kernel_name']

    dout_shape_5hd = (dout_n, dout_cg, dout_h, dout_w, dout_c0)
    fmap_shape_5hd = (fmap_n, fmap_cg, fmap_h, fmap_w, fmap_c0)
    filter_5hd = (BLOCK_SIZE, dout_c1, filter_h, filter_w, BLOCK_SIZE)
    dout_shape_5hd = list(map(int, dout_shape_5hd))
    fmap_shape_5hd = list(map(int, fmap_shape_5hd))
    filter_5hd = list(map(int, filter_5hd))
    tiling = tiling_query(a_shape=dout_shape_5hd,
                          b_shape=fmap_shape_5hd,
                          c_shape=filter_5hd,
                          a_dtype=dout.dtype,
                          b_dtype=fmap.dtype,
                          c_dtype=depthwise_dfilter_res.dtype,
                          mad_dtype="float32",
                          padl=pad_left,
                          padr=pad_right,
                          padu=pad_top,
                          padd=pad_bottom,
                          strideh=stride_h,
                          stridew=stride_w,
                          strideh_expand=1,
                          stridew_expand=1,
                          dilationh=dilation_h,
                          dilationw=dilation_w,
                          group=1,
                          fused_double_operand_num=0,
                          bias_flag=0,
                          op_tag="depthwise_bp_filter",
                          kernel_name=kernel_name)
    _common_tiling_check(tiling)
    if tiling["BL1_shape"] is None:
        dict_args = {'errCode': 'E67004', 'op_name': 'depthwise_conv2d_backprop_filter', 'BL1_shape': 'None'}
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    # Cg,C1HwWw,C1C0,C0 (zN in L0C, Cg, N1, M, N0)
    mad_cc_axis_cg, mad_cc_axis_n1, mad_cc_axis_m, mad_cc_axis_n0 = mad_cc.op.axis
    n1_l0_factor = tiling["CL0_matrix"][0]  # tiling["BL0_matrix"][1]
    m0_l0_factor = tiling["CL0_matrix"][2]
    m1_l0_factor = tiling["CL0_matrix"][1]  # or tiling["AL0_matrix"][0]
    al1_shape_invalid = tiling["AL1_shape"] == [] or tiling["AL1_shape"] is None
    if al1_shape_invalid:
        m_l1_factor = BLOCK_SIZE
        k_hw_al1_factor = dout_h * dout_w
    else:
        m_l1_factor = tiling["AL1_shape"][1] * m1_l0_factor * m0_l0_factor
        k_hw_al1_factor = tiling["AL1_shape"][0]

    if not tiling["BL1_shape"]:
        n1_l1_factor = filter_h * filter_w * BLOCK_SIZE
        k_hw_bl1_factor = dout_h * dout_w
    else:
        n1_l1_factor = tiling["BL1_shape"][1] * n1_l0_factor
        k_hw_bl1_factor = tiling["BL1_shape"][0]

    if tiling["AL0_matrix"] == [] or tiling["AL0_matrix"] is None:
        k_hw_k0_factor = BLOCK_SIZE
        k_hw_l0_factor = (dout_h * dout_w + BLOCK_SIZE - 1) // BLOCK_SIZE
    else:
        k_hw_k0_factor = tiling["AL0_matrix"][3]  # or tiling["BL0_matrix"][3]
        k_hw_l0_factor = tiling["AL0_matrix"][1]  # or tiling["BL0_matrix"][0]
    block_batch_nparts = tiling["block_dim"][0]
    block_n_nparts = tiling["block_dim"][1]
    block_m_nparts = tiling["block_dim"][2]
    block_cg_nparts = tiling["block_dim"][3]
    if "AUB_shape" not in tiling.keys() or tiling["AUB_shape"] == [] or tiling["AUB_shape"] is None:
        block_h_nparts = 1
    else:
        block_h_nparts = tiling["AUB_shape"][0]
    if tiling["AL1_shape"] is None:
        sch[dout_cbuf].compute_inline()
    # N
    mad_cc_n1_l1o, mad_cc_n1_l1i = sch[mad_cc].split(mad_cc_axis_n1, n1_l1_factor)
    mad_cc_n1_l0o, mad_cc_n1_l0i = sch[mad_cc].split(mad_cc_n1_l1i, n1_l0_factor)
    # M
    mad_cc_m1_l1o, mad_cc_m_l1i = sch[mad_cc].split(mad_cc_axis_m, m_l1_factor)
    mad_cc_m1, mad_cc_m0 = sch[mad_cc].split(mad_cc_m_l1i, m0_l0_factor)
    mad_cc_m1_l0o, mad_cc_m1_l0i = sch[mad_cc].split(mad_cc_m1, m1_l0_factor)
    # K
    mad_cc_kn_l1o = mad_cc.op.reduce_axis[0]
    block_batch_o, block_batch_i = sch[mad_cc].split(mad_cc_kn_l1o, nparts=block_batch_nparts)
    mad_cc_axis_k = mad_cc.op.reduce_axis[1]
    block_h_o, block_h_i = sch[mad_cc].split(mad_cc_axis_k, nparts=block_h_nparts)

    if k_hw_al1_factor >= k_hw_bl1_factor:
        mad_cc_ak1_l1o, mad_cc_ak1_l1i = sch[mad_cc].split(block_h_i, k_hw_al1_factor)
        mad_cc_bk1_l1o, mad_cc_k1_l1i = sch[mad_cc].split(mad_cc_ak1_l1i, k_hw_bl1_factor)
        mad_cc_max_k1_l1o = mad_cc_ak1_l1o
        mad_cc_min_k1_l1o = mad_cc_bk1_l1o
    else:
        mad_cc_bk1_l1o, mad_cc_bk1_l1i = sch[mad_cc].split(block_h_i, k_hw_bl1_factor)
        mad_cc_ak1_l1o, mad_cc_k1_l1i = sch[mad_cc].split(mad_cc_bk1_l1i, k_hw_al1_factor)
        mad_cc_max_k1_l1o = mad_cc_bk1_l1o
        mad_cc_min_k1_l1o = mad_cc_ak1_l1o

    mad_cc_k1, mad_cc_k0 = sch[mad_cc].split(mad_cc_k1_l1i, k_hw_k0_factor)
    mad_cc_k1_l0o, mad_cc_k1_l0i = sch[mad_cc].split(mad_cc_k1, k_hw_l0_factor)

    sch[mad_cc].reorder(mad_cc_axis_cg, block_batch_o, block_h_o, mad_cc_n1_l1o, mad_cc_m1_l1o, block_batch_i,
                        mad_cc_max_k1_l1o, mad_cc_min_k1_l1o, mad_cc_n1_l0o, mad_cc_m1_l0o, mad_cc_k1_l0o,
                        mad_cc_m1_l0i, mad_cc_n1_l0i, mad_cc_axis_n0, mad_cc_m0, mad_cc_k1_l0i, mad_cc_k0)
    sch[dout_ca].compute_at(sch[mad_cc], mad_cc_k1_l0o)
    sch[fmap_cb].compute_at(sch[mad_cc], mad_cc_k1_l0o)
    sch[dout_cbuf].compute_at(sch[mad_cc], mad_cc_ak1_l1o)
    sch[fmap_cbuf_nc1hwc0].compute_at(sch[mad_cc], mad_cc_bk1_l1o)
    sch[fmap_cbuf_row_major].compute_at(sch[mad_cc], mad_cc_bk1_l1o)

    dw_axis_cg, dw_axis_n1, dw_axis_m, dw_axis_n0 = sch[depthwise_dfilter_res].op.axis
    n1_ub_factor = tiling["CUB_matrix"][0]
    m1_ub_factor = tiling["CUB_matrix"][1]
    # Block tiling
    block_cg_o, block_cg_i = sch[depthwise_dfilter_res].split(dw_axis_cg, nparts=block_cg_nparts)
    block_n_o, block_n_i = sch[depthwise_dfilter_res].split(dw_axis_n1, nparts=block_n_nparts)
    block_m_o, block_m_i = sch[depthwise_dfilter_res].split(dw_axis_m, nparts=block_m_nparts)

    # N
    dw_n1_l0o, dw_n1_l0i = sch[depthwise_dfilter_res].split(block_n_i, n1_l0_factor)
    dw_n1_ubo, dw_n1_ubi = sch[depthwise_dfilter_res].split(dw_n1_l0i, n1_ub_factor)

    pragma_overload_filter(block_n_nparts, block_m_nparts, sch[depthwise_dfilter_res], dw_n1_ubi)

    # M
    dw_m1, dw_m0 = sch[depthwise_dfilter_res].split(block_m_i, m0_l0_factor)
    dw_m1_l0o, dw_m1_l0i = sch[depthwise_dfilter_res].split(dw_m1, m1_l0_factor)
    dw_m1_ubo, dw_m1_ubi = sch[depthwise_dfilter_res].split(dw_m1_l0i, m1_ub_factor)
    sch[depthwise_dfilter_res].reorder(block_cg_o, block_m_o, block_n_o, block_cg_i, dw_n1_l0o, dw_m1_l0o, dw_n1_ubo,
                                       dw_m1_ubo, dw_n1_ubi, dw_m1_ubi, dw_axis_n0, dw_m0)
    sch[mad_cc].compute_at(sch[depthwise_dfilter_res], dw_m1_l0o)
    sch[mad_ubuf].compute_at(sch[depthwise_dfilter_res], dw_m1_ubo)
    sch[depthwise_dfilter_ubuf].compute_at(sch[depthwise_dfilter_res], dw_m1_ubo)

    sch[dout_cbuf].storage_align(sch[dout_cbuf].op.axis[2], BLOCK_SIZE, 0)
    sch[dout_ca].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE), (1, BLOCK_SIZE))
    sch[fmap_cb].buffer_align((1, 1), (1, 1), (1, 1), (1, 1), (1, BLOCK_SIZE), (1, BLOCK_SIZE))
    sch[fmap_cbuf_row_major].buffer_align((1, 1), (1, 1), (dout_w, dout_w), (1, 1), (filter_h, filter_h),
                                          (filter_w, filter_w), (1, BLOCK_SIZE))

    if tiling["manual_pingpong_buffer"]["AL1_pbuffer"] == DOUBLE_BUFFER:
        sch[dout_cbuf].double_buffer()
    if tiling["manual_pingpong_buffer"]["BL1_pbuffer"] == DOUBLE_BUFFER:
        sch[fmap_cbuf_nc1hwc0].double_buffer()
        sch[fmap_cbuf_row_major].double_buffer()
    if tiling["manual_pingpong_buffer"]["AL0_pbuffer"] == DOUBLE_BUFFER:
        sch[dout_ca].double_buffer()
    if tiling["manual_pingpong_buffer"]["BL0_pbuffer"] == DOUBLE_BUFFER:
        sch[fmap_cb].double_buffer()
    if tiling["manual_pingpong_buffer"]["CL0_pbuffer"] == DOUBLE_BUFFER:
        sch[mad_cc].double_buffer()
    if tiling["manual_pingpong_buffer"]["CUB_pbuffer"] == DOUBLE_BUFFER:
        sch[mad_ubuf].double_buffer()
        sch[depthwise_dfilter_ubuf].double_buffer()

    sch[mad_ubuf].reused_by(depthwise_dfilter_ubuf)

    # emit insn
    sch[fmap_cbuf_nc1hwc0].emit_insn(fmap_cbuf_nc1hwc0.op.axis[0], 'dma_copy')
    # emit convolution params.
    setfmatrix_dict = {
        "conv_kernel_h": depthwise_dfilter_res.op.attrs['kernel_h'],
        "conv_kernel_w": depthwise_dfilter_res.op.attrs['kernel_w'],
        "conv_padding_top": depthwise_dfilter_res.op.attrs['padding'][0],
        "conv_padding_bottom": depthwise_dfilter_res.op.attrs['padding'][1],
        "conv_padding_left": depthwise_dfilter_res.op.attrs['padding'][2],
        "conv_padding_right": depthwise_dfilter_res.op.attrs['padding'][3],
        "conv_stride_h": depthwise_dfilter_res.op.attrs['stride'][0],
        "conv_stride_w": depthwise_dfilter_res.op.attrs['stride'][1],
        "conv_fm_c": fmap.op.shape[2] * fmap.op.shape[5],
        "conv_fm_h": fmap.op.shape[3],
        "conv_fm_w": fmap.op.shape[4]
    }
    sch[fmap_cbuf_row_major].emit_insn(fmap_cbuf_row_major.op.axis[1], 'set_fmatrix', setfmatrix_dict)
    sch[fmap_cb].emit_insn(fmap_cb.op.axis[1], 'im2col')
    sch[dout_cbuf].emit_insn(dout_cbuf.op.axis[0], 'dma_copy')
    sch[dout_ca].emit_insn(dout_ca.op.axis[0], 'dma_copy')

    sch[mad_ubuf].emit_insn(mad_ubuf.op.axis[0], 'dma_copy')
    # mad_pattern value: 0 for gemm, 1 for gemv, 2 for convolution
    mad_dict = {
        "mad_pattern": cce_params.CONV_MODE,
        'k_outer': [block_batch_o, block_batch_i, block_h_o, mad_cc_ak1_l1o, mad_cc_bk1_l1o, mad_cc_k1_l0o]
    }
    sch[mad_cc].emit_insn(mad_cc_m1_l0i, 'mad', mad_dict)
    sch[depthwise_dfilter_ubuf].reorder(depthwise_dfilter_ubuf.op.axis[0], depthwise_dfilter_ubuf.op.axis[2],
                                        depthwise_dfilter_ubuf.op.axis[1], depthwise_dfilter_ubuf.op.axis[3])
    sch[depthwise_dfilter_ubuf].emit_insn(depthwise_dfilter_ubuf.op.axis[1], 'elewise_single_diagonal')
    sch[depthwise_dfilter_res].emit_insn(dw_n1_ubi, 'dma_copy')

    # for multi cores
    block = tvm.thread_axis("blockIdx.x")
    block_axis = sch[depthwise_dfilter_res].fuse(block_cg_o, block_m_o, block_n_o)
    sch[depthwise_dfilter_res].bind(block_axis, block)

    return sch


def _tiling_fetch(dx_res, stride, mad_cc, dout_shape, weight_shape):
    """"
    tiling_fetch
    """
    padding_top = int(dx_res.op.attrs['dilated_pad'][0])
    padding_bottom = int(dx_res.op.attrs['dilated_pad'][1])
    padding_left = int(dx_res.op.attrs['dilated_pad'][2])
    padding_right = int(dx_res.op.attrs['dilated_pad'][3])
    # after expand, full model sliding window, value must be 1
    stride_h = int(dx_res.op.attrs['dilated_strides'][0])
    stride_w = int(dx_res.op.attrs['dilated_strides'][1])
    kernel_h = int(dx_res.op.attrs['weight_height'])
    kernel_w = int(dx_res.op.attrs['weight_width'])
    kernel_name = dx_res.op.attrs['kernel_name']
    # expand stride equal ops interface parameter
    strideh_expand = stride
    stridew_expand = stride
    dilation_h = 1
    dilation_w = 1

    in_dtype = "float16"
    w_dtype = "float16"
    res_dtype = mad_cc.dtype
    mad_dtype = mad_cc.dtype
    group_num = 1

    dout_shape_batch, dout_shape_output_c1, _, dout_shape_output_height, dout_shape_output_width, dout_shape_block = \
        dout_shape
    dout_shape_tiling = \
        dout_shape_batch, dout_shape_output_c1, dout_shape_output_height, dout_shape_output_width, dout_shape_block
    weight_shape_c1, _, _, weight_shape_co, weight_shape_block = weight_shape
    weight_shape_tiling = weight_shape_co, weight_shape_c1, kernel_h, kernel_w, weight_shape_block
    padding_top_tiling = (padding_top + abs(padding_top)) // 2
    padding_bottom_tiling = (padding_bottom + abs(padding_bottom)) // 2
    padding_left_tiling = (padding_left + abs(padding_left)) // 2
    padding_right_tiling = (padding_right + abs(padding_right)) // 2

    wd_value = dout_shape_output_width * stride - (stride - 1)
    hd_value = dout_shape_output_height * stride - (stride - 1)
    wi_value = (wd_value + padding_left + padding_right - kernel_w) // stride_w + 1
    hi_value = (hd_value + padding_top + padding_bottom - kernel_h) // stride_h + 1
    # shape format must be 5HD
    dout_shape_tiling = list(map(int, dout_shape_tiling))
    weight_shape_tiling = list(map(int, weight_shape_tiling))
    tiling_new = tiling_query(a_shape=dout_shape_tiling,
                              b_shape=weight_shape_tiling,
                              c_shape=None,
                              a_dtype=in_dtype,
                              b_dtype=w_dtype,
                              c_dtype=res_dtype,
                              mad_dtype=mad_dtype,
                              padl=padding_left_tiling,
                              padr=padding_right_tiling,
                              padu=padding_top_tiling,
                              padd=padding_bottom_tiling,
                              strideh=stride_h,
                              stridew=stride_w,
                              strideh_expand=strideh_expand,
                              stridew_expand=stridew_expand,
                              dilationh=dilation_h,
                              dilationw=dilation_w,
                              group=group_num,
                              fused_double_operand_num=0,
                              bias_flag=0,
                              op_tag="depthwise_bp_input",
                              kernel_name=kernel_name)

    # get tiling params
    al1_tiling = tiling_new['AL1_shape']
    bl1_tiling = tiling_new['BL1_shape']
    al0_tiling = tiling_new['AL0_matrix']
    bl0_tiling = tiling_new['BL0_matrix']
    cl0_tiling = tiling_new['CL0_matrix']
    cub_tiling = tiling_new['CUB_matrix']
    aub_tiling = tiling_new['AUB_shape']
    double_buffer_tiling = tiling_new["manual_pingpong_buffer"]
    block_dim_tiling = tiling_new['block_dim']

    if al1_tiling == [] or al1_tiling is None:
        al1_tiling = [
            dout_shape_block * kernel_w * kernel_h,
            (hi_value * wi_value + (cl0_tiling[1] * 16) - 1) // (cl0_tiling[1] * 16), 1, 1
        ]
    if bl1_tiling == [] or bl1_tiling is None:
        bl1_tiling = [dout_shape_block * kernel_w * kernel_h, dout_shape_block // dout_shape_block, 1, 1]
    if bl0_tiling == []:
        bl0_tiling = [al0_tiling[1], 1, 16, 16, 1, al0_tiling[5]]
    return al1_tiling, bl1_tiling, al0_tiling, bl0_tiling, cl0_tiling, cub_tiling, aub_tiling, block_dim_tiling,\
        double_buffer_tiling


# pylint: disable=locally-disabled,too-many-locals,too-many-statements
# pylint: disable=too-many-branches
def depthwise_conv2d_backprop_input_d_schedule(dx_res):
    """
    the schedule of depthwise_conv2d_backprop_input
    """
    sch = create_schedule(dx_res.op)

    # get tensor info
    dx_cast = dx_res.op.input_tensors[0]
    mad_res = dx_cast.op.input_tensors[0]
    dout_col_pad = mad_res.op.input_tensors[0]
    weight_rotated = mad_res.op.input_tensors[1]
    weight = weight_rotated.op.input_tensors[0]
    dout_col = dout_col_pad.op.input_tensors[0]
    dout_dilated = dout_col.op.input_tensors[0]
    dout = dout_dilated.op.input_tensors[0]

    # set data flow
    dout_ubuf = sch.cache_read(dout, cce_params.scope_ubuf, [dout_dilated])
    dout_cbuf_nc1hwc0 = sch.cache_write(dout_dilated, cce_params.scope_cbuf)
    dout_dilated_ubuf = sch.cache_write(dout_cbuf_nc1hwc0, cce_params.scope_ubuf)
    dout_cbuf_row_major = sch.cache_write(dout_col, cce_params.scope_cbuf)
    sch[dout_dilated].compute_inline()
    sch[dout_col].compute_inline()
    dout_ca = sch.cache_write(dout_col_pad, cce_params.scope_ca)
    sch[dout_col_pad].compute_inline()

    weight_cbuf = sch.cache_read(weight, cce_params.scope_cbuf, [weight_rotated])
    weight_cb = sch.cache_write(weight_rotated, cce_params.scope_cb)
    sch[weight_rotated].compute_inline()

    mad_cc = sch.cache_write(mad_res, cce_params.scope_cc)
    mad_ubuf = sch.cache_write(dx_cast, cce_params.scope_ubuf)
    sch[mad_res].compute_inline()
    sch[dx_cast].compute_inline()

    # compute shape value, out input img2col_padding
    block_size = dout.op.shape[len(dout.op.shape) - 1].value
    _, _, _, _, dout_dilated_w, _ = dout_dilated.shape
    fmap_w = dout_dilated_w.value + dx_res.op.attrs['dilated_pad'][2].value + dx_res.op.attrs['dilated_pad'][
        3].value - dx_res.op.attrs['weight_width'].value + 1
    stride = int(dout_dilated.op.attrs["strides"][0].value)

    # get shape value
    weight_shape = [int(i.value) for i in weight.shape]
    dout_shape = [int(i.value) for i in dout.shape]

    def autoting():
        """"autoting"""
        loc_factor_na = al0_tiling[4]
        loc_factor_m = al0_tiling[0] * al0_tiling[2]
        loc_factor_nb = bl0_tiling[1]
        loc_factor_k = bl0_tiling[0]
        cub_factor_n4 = cub_tiling[4]
        cub_factor_m = cub_tiling[1] * cub_tiling[2]
        cub_factor_n0 = cub_tiling[0]
        res_loc_factor_m = al0_tiling[0] * al0_tiling[2]
        l1_factor_n = al1_tiling[2]
        l1_factor_m = al1_tiling[1] * cl0_tiling[1] * 16

        # double buffer
        double_buffer_flag = {
            'AUB_pbuffer': False,
            'AL1_pbuffer': False,
            'BL1_pbuffer': False,
            'AL0_pbuffer': False,
            'BL0_pbuffer': False,
            'CL0_pbuffer': False,
            'CUB_pbuffer': False,
            'UBG_pbuffer': False,
        }
        double_buffer_flag = double_buffer_tiling
        # muti core bind
        blocks = block_dim_tiling[0] * block_dim_tiling[3]
        mad_cc_axis_n, mad_cc_axis_cg, mad_cc_axis_co1, mad_cc_axis_howomad, mad_cc_axis_co0 = mad_cc.op.axis
        mad_cc_ncut_o_n, mad_cc_ncut_i_n = sch[mad_cc].split(mad_cc_axis_n, factor=loc_factor_na)
        mad_cc_mcut_o, mad_cc_mcut_i = sch[mad_cc].split(mad_cc_axis_howomad, factor=loc_factor_m)
        mad_cc_kcut_o, mad_cc_kcut_i = sch[mad_cc].split(mad_cc.op.reduce_axis[0], factor=loc_factor_k)
        mad_cc_ncut_o, mad_cc_ncut_i = sch[mad_cc].split(mad_cc_axis_co1, factor=loc_factor_nb)
        sch[mad_cc].reorder(mad_cc_ncut_o_n, mad_cc_axis_cg, mad_cc_ncut_o, mad_cc_mcut_o, mad_cc_kcut_o,
                            mad_cc_ncut_i_n, mad_cc_ncut_i, mad_cc_mcut_i, mad_cc_axis_co0, mad_cc_kcut_i,
                            mad_cc.op.reduce_axis[1])
        sch[dout_ca].compute_at(sch[mad_cc], mad_cc_kcut_o)
        sch[weight_cb].compute_at(sch[mad_cc], mad_cc_kcut_o)

        mad_ubuf_axis_n, mad_ubuf_axis_cg, mad_ubuf_axis_co1, mad_ubuf_axis_howomad, mad_ubuf_axis_co0 = \
            mad_ubuf.op.axis
        mad_ubuf_ncut_o_n, mad_ubuf_ncut_i_n = sch[mad_ubuf].split(mad_ubuf_axis_n, factor=cub_factor_n4)
        mad_ubuf_mcut_o, mad_ubuf_mcut_i = sch[mad_ubuf].split(mad_ubuf_axis_howomad, factor=cub_factor_m)
        mad_ubuf_ncut_o, mad_ubuf_ncut_i = sch[mad_ubuf].split(mad_ubuf_axis_co1, factor=cub_factor_n0)
        sch[mad_ubuf].reorder(mad_ubuf_ncut_o_n, mad_ubuf_axis_cg, mad_ubuf_ncut_o, mad_ubuf_mcut_o, mad_ubuf_ncut_i_n,
                              mad_ubuf_ncut_i, mad_ubuf_mcut_i, mad_ubuf_axis_co0)
        sch[mad_cc].compute_at(sch[mad_ubuf], mad_ubuf_mcut_o)

        conv_ncut_o, conv_ncut_i = sch[dx_res].split(dx_res.op.axis[0], factor=l1_factor_n)
        conv_hcut_o, conv_hcut_i = sch[dx_res].split(dx_res.op.axis[3], factor=l1_factor_m)
        conv_mcut_o, conv_mcut_i = sch[dx_res].split(conv_hcut_i, factor=res_loc_factor_m)
        sch[dx_res].reorder(conv_ncut_o, dx_res.op.axis[1], conv_hcut_o, conv_mcut_o, conv_ncut_i, dx_res.op.axis[2],
                            conv_mcut_i, dx_res.op.axis[4])
        sch[mad_ubuf].buffer_align((1, 1), (1, 1), (1, 1), (1, block_size), (1, block_size))
        sch[mad_ubuf].compute_at(sch[dx_res], conv_mcut_o)
        sch[dout_cbuf_row_major].buffer_align((1, 1), (1, 1), (fmap_w, fmap_w), (1, 1), (1, 1), (1, 1), (1, block_size))
        sch[dout_cbuf_row_major].compute_at(sch[dx_res], conv_hcut_o)
        sch[dout_cbuf_nc1hwc0].compute_at(sch[dx_res], conv_hcut_o)
        if tiling_out_invalid_flag is True:
            sch[weight_cbuf].compute_inline()
        else:
            sch[weight_cbuf].compute_at(sch[dx_res], conv_hcut_o)

        if stride > 1:
            ub_l1hcut_o, ub_l1hcut_i = sch[dout_cbuf_nc1hwc0].split(dout_cbuf_nc1hwc0.op.axis[3], factor=aub_tiling[1])
            dila_o_h, dila_i_h = sch[dout_dilated_ubuf].split(dout_dilated_ubuf.op.axis[3], factor=stride)
            dila_o_w, dila_i_w = sch[dout_dilated_ubuf].split(dout_dilated_ubuf.op.axis[4], factor=stride)
            sch[dout_dilated_ubuf].reorder(dila_i_h, dila_i_w, dila_o_h, dila_o_w)
            sch[dout_dilated_ubuf].unroll(dila_i_h)
            sch[dout_dilated_ubuf].unroll(dila_i_w)

            sch[dout_dilated_ubuf].compute_at(sch[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            sch[dout_ubuf].compute_at(sch[dout_cbuf_nc1hwc0], ub_l1hcut_o)
            sch[dout_cbuf_nc1hwc0].emit_insn(ub_l1hcut_i, 'dma_copy')
            sch[dout_dilated_ubuf].emit_insn(dila_o_h, 'dma_padding')
            sch[dout_ubuf].emit_insn(dout_ubuf.op.axis[0], 'dma_copy')

            # aub double buffer
            if double_buffer_flag["AUB_pbuffer"] == 2:
                sch[dout_dilated_ubuf].double_buffer()
                sch[dout_ubuf].double_buffer()
                sch[dout_dilated_ubuf].preload()
                sch[dout_ubuf].preload()
        else:
            sch[dout_dilated_ubuf].compute_inline()
            sch[dout_ubuf].compute_inline()
            sch[dout_cbuf_nc1hwc0].emit_insn(dout_cbuf_nc1hwc0.op.axis[0], 'dma_copy')

        # emit convolution params.
        setfmatrix_dict = {
            "conv_kernel_h": dx_res.op.attrs['weight_height'],
            "conv_kernel_w": dx_res.op.attrs['weight_width'],
            "conv_padding_top": dx_res.op.attrs['dilated_pad'][0],
            "conv_padding_bottom": dx_res.op.attrs['dilated_pad'][1],
            "conv_padding_left": dx_res.op.attrs['dilated_pad'][2],
            "conv_padding_right": dx_res.op.attrs['dilated_pad'][3],
            "conv_stride_h": dx_res.op.attrs['dilated_strides'][0],
            "conv_stride_w": dx_res.op.attrs['dilated_strides'][1],
            "conv_fm_c": dout_dilated.shape[2] * dout_dilated.shape[5],
            "conv_fm_h": dout_dilated.shape[3],
            "conv_fm_w": dout_dilated.shape[4]
        }
        is_overload = False
        stride_h = int(dx_res.op.attrs['dilated_strides'][0])
        stride_w = int(dx_res.op.attrs['dilated_strides'][1])
        kernel_h = int(dx_res.op.attrs['weight_height'])
        kernel_w = int(dx_res.op.attrs['weight_width'])
        if block_dim_tiling[1] > 1 or (block_dim_tiling[2] > 1 and (stride_h < kernel_h or stride_w < kernel_w)):
            is_overload = True
        set_pragma_for_cache_read_mode(is_overload, sch[dx_res], conv_ncut_i)

        sch[dout_cbuf_row_major].emit_insn(dout_cbuf_row_major.op.axis[1], 'set_fmatrix', setfmatrix_dict)
        sch[dout_ca].emit_insn(dout_ca.op.axis[1], 'im2col')
        if tiling_out_invalid_flag is False:
            sch[weight_cbuf].emit_insn(weight_cbuf.op.axis[0], 'dma_copy')
        sch[weight_cb].emit_insn(weight_cb.op.axis[3], 'dma_copy')
        sch[mad_ubuf].emit_insn(mad_ubuf_ncut_i_n, 'dma_copy')
        mad_dict = {"mad_pattern": cce_params.CONV_MODE, "k_outer": mad_cc_kcut_o}
        sch[mad_cc].emit_insn(mad_cc_ncut_i_n, 'mad', mad_dict)
        sch[dx_res].emit_insn(conv_ncut_i, 'dma_copy')

        # turn on dubole buffer
        # al1
        if double_buffer_flag["AL1_pbuffer"] == 2:
            sch[dout_cbuf_nc1hwc0].double_buffer()
        # bl1
        if double_buffer_flag["BL1_pbuffer"] == 2:
            sch[weight_cbuf].double_buffer()
        # l0a
        if double_buffer_flag["AL0_pbuffer"] == 2:
            sch[dout_ca].double_buffer()
        # l0b
        if double_buffer_flag["BL0_pbuffer"] == 2:
            sch[weight_cb].double_buffer()
        # L0C
        if double_buffer_flag["CL0_pbuffer"] == 2:
            sch[mad_cc].double_buffer()
        # CUB
        if double_buffer_flag["CUB_pbuffer"] == 2:
            sch[mad_ubuf].double_buffer()
        sch[dx_res].reorder(conv_ncut_o, dx_res.op.axis[1], conv_hcut_o, conv_mcut_o, conv_ncut_i, dx_res.op.axis[2],
                            conv_mcut_i, dx_res.op.axis[4])

        # bind muti core
        if blocks != 1:
            res_nncut_o, res_nncut_i = sch[dx_res].split(conv_ncut_o, nparts=block_dim_tiling[0])
            res_cccut_o, res_cccut_i = sch[dx_res].split(dx_res.op.axis[1], nparts=block_dim_tiling[3])
            sch[dx_res].reorder(res_nncut_o, res_cccut_o, res_nncut_i, res_cccut_i)
            out_fused = sch[dx_res].fuse(res_nncut_o, res_cccut_o)
            out_fused_out, _ = sch[dx_res].split(out_fused, nparts=blocks)
            bind_out, _ = sch[dx_res].split(out_fused_out, 1)
            blockidx = tvm.thread_axis("blockIdx.x")
            sch[dx_res].bind(bind_out, blockidx)

    al1_tiling, _, al0_tiling, bl0_tiling, cl0_tiling, cub_tiling, aub_tiling, block_dim_tiling, double_buffer_tiling \
        = _tiling_fetch(dx_res, stride, mad_cc, dout_shape, weight_shape)
    # when tiling value invalid, set mini value
    tiling_out_invalid_flag = False
    if al0_tiling[2] == 32:
        al1_tiling = [1, 1, 1, 1]
        al0_tiling = [1, 1, 32, 16, 1, 1]
        bl0_tiling = [1, 1, 16, 16, 1, 1]
        cl0_tiling = [1, 1, 16, 16, 1, 1]
        cub_tiling = [1, 1, 16, 16, 1, 1]
        aub_tiling = [1, 1, 1, 1]
        block_dim_tiling = [1, 1, 1, 1]
        double_buffer_tiling = {'AUB_pbuffer': 1, 'BUB_pbuffer': 1, 'AL1_pbuffer': 1,
                                'BL1_pbuffer': 1, 'AL0_pbuffer': 1, 'BL0_pbuffer': 1,
                                'CL0_pbuffer': 1, 'CUB_pbuffer': 1, 'UBG_pbuffer': 2}
        tiling_out_invalid_flag = True
    autoting()

    return sch
