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
util_conv3d
"""
from impl.util.platform_adapter import error_manager_util
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import tbe_platform

BIAS_LENGTH = 1
DYNAMIC_DIM_VAL = -1
_BLOCK_SIZE = 16

def transform_shape_with_exception(src_format, to_format, ori_shape,
                                   format_white_list, attr_name):

    res = transform_shape_with_format(src_format, to_format, ori_shape,
                                      format_white_list)
    if res is None:
        dict_args = {
            'errCode': 'E60008',
            'param_name': attr_name,
            'expected_format_list': ",".join(format_white_list),
            'format': src_format if src_format not in format_white_list else to_format
        }
        raise RuntimeError(dict_args,
                           error_manager_util.get_error_message(dict_args))
    return res

def transform_shape_with_format(src_format, to_format, ori_shape, format_white_list):
    # input format is not expected
    if ((src_format not in format_white_list) or
        (to_format not in format_white_list)):
        return None
    # need not to transform
    if src_format == to_format:
        return list(ori_shape)
    res_shape = [1 for _ in range(len(to_format))]
    for i in range(len(to_format)):
        for j in range(len(src_format)):
            if to_format[i] == src_format[j]:
                res_shape[i] = ori_shape[j]
                break
    return res_shape

def check_bias(bias, res_dtype):
    """
    algorithm: Check the input params of bias

    Parameters
    ----------

    bias: A dict with keys(shape and dtype) or None
        input bias tensor

    res_dtype: The dtype of output

    Returns
    -------
    None
    """
    bias_shape = bias.get("ori_shape")
    if len(bias_shape) != BIAS_LENGTH:
        dict_args = {
            'errCode': 'E60006',
            'param_name': 'bias',
            'expected_length': '1',
            'length': '{}'.format(len(bias_shape))
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
    bias_dtype = bias.get("dtype").lower()
    if bias_dtype != res_dtype:
        dict_args = {
            'errCode': 'E65002',
            'param_1': 'bias_dtype',
            'param_2': 'res_dtype'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))


def generalize_input_keep_rank(param_dict):
    ori_shape, ori_format = param_dict["ori_shape"], param_dict["ori_format"]
    idx_n = ori_format.find('N')
    idx_d = ori_format.find('D')
    idx_h = ori_format.find('H')
    idx_w = ori_format.find('W')
    idx_tup = (idx_n, idx_d, idx_h, idx_w)
    ori_shape = list(ori_shape)
    for idx in idx_tup:
        ori_shape[idx] = DYNAMIC_DIM_VAL
    param_dict["ori_shape"] = tuple(ori_shape)


def check_l1_limitation_dx(w_value, stride_d, filter_h_dilation, filter_d_dilation, block_size_k):
    if w_value > _BLOCK_SIZE:
        h_value_max = filter_h_dilation
    else:
        h_value_max = filter_h_dilation + _BLOCK_SIZE // w_value
    if w_value % _BLOCK_SIZE != 0 and _BLOCK_SIZE % w_value != 0:
        h_value_max += 1
    a_l1_size = h_value_max * w_value * ((filter_d_dilation - 2) // stride_d + 2) * block_size_k * 2
    b_l1_size = _BLOCK_SIZE * block_size_k * 2
    l1_size = tbe_platform.get_soc_spec("L1_SIZE")
    if (a_l1_size + b_l1_size) > l1_size:
        dict_args = {
            'errCode': 'E60026'
        }
        raise RuntimeError(dict_args,
                            error_manager_util.get_error_message(dict_args))
