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
util_gemm
"""
import warnings


RANGE_MAX = 2147483647 # 2**31-1
BATCH_GEAR = [0, 1, 3, 7, 15, 31, RANGE_MAX]
SHAPE_GEAR_MATMUL_ND = [0, 16*3, 16*7, 16*15, 16*31, 16*63, 16*127, 16*191, 16*255,
                        16*511, 16*767, 16*1023, RANGE_MAX] # for fp16
DYNAMIC_DIM_VAL = -1
DYNAMIC_UNKNOWN_RANK = [-2]


def _get_shape_gear(dim, shape_gear):
    pos = 1
    while(pos < len(shape_gear) and shape_gear[pos] < dim):
        pos += 1
    return (shape_gear[pos - 1] + 1, shape_gear[pos])


def cal_gemm_shape_range(shape, ori_format):
    """
    cal gemm shape range
    """
    shape_range = []
    shape_len = len(shape)
    if ori_format == "ND":
        # shape like (batch1, ..., batchn, m, k)
        # process batch dim
        for i in range(0, shape_len - 2):
            if shape[i] > RANGE_MAX:
                return "LOWER_LIMIT"
            shape_range.append(_get_shape_gear(shape[i], BATCH_GEAR))

        # process m/k/n dim and bias
        for i in range(-min(shape_len, 2), 0):
            if shape[i] > RANGE_MAX:
                return "LOWER_LIMIT"
            shape_range.append(_get_shape_gear(shape[i], SHAPE_GEAR_MATMUL_ND))
    else:
        return "LOWER_LIMIT"
    return tuple(shape_range)


def _generate_unknown_shape_gemm(shape):
    """
    generate unknown shape gemm
    """
    return [DYNAMIC_DIM_VAL for i in shape]


def generalize_input_keep_rank_gemm(input_dict):
    """
    generalize input keep rank gemm
    """
    if input_dict.get("ori_format") in ("ND"):
        input_dict["ori_shape"] = _generate_unknown_shape_gemm(input_dict["ori_shape"])


def is_graph_mode(tensor):
    """
    check whether is graph mode
    """
    # check graph mode or single mode in fuzzy compile
    if ((DYNAMIC_DIM_VAL in tensor.get("ori_shape") and "ori_range" in tensor.keys()) or
        list(tensor.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK):
        return True
    return False


def matmul_range_check(input_x1, input_x2, bias):
    """
    check matmul range
    """
    x1_ori_range = input_x1.get("range")
    x2_ori_range = input_x2.get("range")
    input_list = [x1_ori_range, x2_ori_range]
    param_index_info = []
    type_info = []

    op_type = "MatMul" if len(input_x1.get("ori_shape")) == 2 else "BatchMatMul"
    if bias is not None:
        bias_ori_range = bias.get("range")
        input_list.append(bias_ori_range)

    if (list(input_x1.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK or
        list(input_x2.get("ori_shape")) == DYNAMIC_UNKNOWN_RANK):
        # if x1 and x2 are -2, fe should excute static compile
        warnings.warn("{}, input x1 and input_x2 should not be -2".format(op_type))
        type_info = ["lower_limit"]

    for idx, item in enumerate(input_list):
        for range_val in item:
            # if upper range exceed int32 or -1, return upper_limit
            if range_val[1] is None or range_val[1] > RANGE_MAX:
                param_index_info.append(idx)
                type_info.append("upper_limit")
                warnings.warn("{}, if range is none or exceed int32, it's upper limit".format(op_type))
            # if lower range exceed int32, return lower_limit
            if range_val[1] is not None and (range_val[0] > RANGE_MAX or range_val[0] > range_val[1]):
                param_index_info.append(idx)
                type_info.append("lower_limit")
                warnings.warn("{}, if lower range exceed int32 or be larger than upper limit, "
                              "it's lower limit".format(op_type))
    if type_info:
        if "lower_limit" in type_info:
            # for lower_limit, fe should excute static compile
            param_index_info = list(range(len(input_list)))
            type_info = len(input_list) * ["lower_limit"]
        json_info = [{"result": "UNSUPPORTED", "reason": {"param_index": param_index_info, "type": type_info}}]
        return False, json_info
    return True, []
