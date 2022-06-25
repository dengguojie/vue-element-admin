#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
norm tilingcase
"""
import copy
import functools
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterable
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

from tbe import tvm
from tbe.common.platform import ASCEND_910B
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import decode
from tbe.common.utils import do_op_tiling
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.padding import padding
from tbe.tvm.expr import IntImm
from tbe.tvm.expr import Var
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor

from .norm_helper import classify_actions
from ... import util
from ...computation import Computation
from ...constants import BROADCAST_INSNS
from ...constants import CompileInfo
from ...constants import DST_SRC_NO_REUSE_SET
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import NormPattern
from ...constants import Pattern
from ...constants import REDUCE_INSNS
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import TERNARY_INSNS

COMMON_MODE = "common_mode"
PAD_MODE = "pad_mode"
REDUCE_TRANS_MODE = "reduce_trans_mode"
WORKSPACE_MODE = "workspace_mode"

BLOCK = 16
BLOCK_SIZE_BYTE = 32
FAKE_WORKSPACE_SIZE = 32

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4

# temp space for common reduce
COMMON_REDUCE_TEMP_SPACE = 256
# temp space for last axis fp16 broadcast use vnchwconv
VNCHWCONV_TEMP_SPACE_FP16 = 1024
# temp space for last axis fp32 broadcast use vnchwconv
VNCHWCONV_TEMP_SPACE_FP32 = 8192
# number extra nodes that enable reduce transpose
REDUCE_TRANSPOSE_EXTRA_NODES = 2
# number extra nodes that enable align and remove pad
ALIGN_AND_REMOVE_PAD_EXTRA_NODES = 2
# temp space for enable align and remove pad with avoiding bank conflict
ALIGN_AND_REMOVE_PAD_TEMP_SPACE = 1024
# number extra nodes that enable no overlap 3
NO_OVERLAP_THREE_EXTRA_NODES = 2
# upper limit of last axis that enable reduce transpose
REDUCE_TRANSPOSE_UPPER_THRESHOLD = 128
# number extra nodes that enable broadcast not align
NO_ALIGN_BROADCAST_EXTRA_NODES = 2
# bytes that one repeat can process
ONE_REPEAT_PROCESS_BYTES = 256


DTYPE_AND_BLOCK_SIZE_MAP = {
    "bool": 32,
    "int8": 32,
    "uint8": 32,
    "float16": 16,
    "float32": 8,
    "int32": 8,
    "int64": 4
}

DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP = {
    "bool": BLOCK * BLOCK * 4,
    "int8": BLOCK * BLOCK * 4,
    "uint8": BLOCK * BLOCK * 4,
    "float16": BLOCK * BLOCK,
    "float32": BLOCK * BLOCK // 2,
    "int32": BLOCK * BLOCK // 2,
    "int64": BLOCK * BLOCK // 4
}


class CalcNormTilingCase(Computation):
    """
    calculate norm tiling case
    """
    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

    def get_sub_pattern(self):
        return NormPattern.N_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.NORM]

    @classmethod
    def get_supported_soc(cls):
        return ["default"]

    def do_tiling_case(self):
        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]

        current_compute = get_context().get_current_compute()
        if _judge_current_compute_is_const():
            current_compute.add("_mode", "const")

        # construct information of graph and add them to ComputeContext
        norm_compute_graph_info = NormComputeGraphInfo(outs)
        norm_info = NormInfo(norm_compute_graph_info)
        current_compute.add("_compute_graph_info", norm_compute_graph_info)
        current_compute.add("_norm_info", norm_info)

        tiling_case_list = []
        tiling_case_list += _add_tiling_case(norm_compute_graph_info, norm_info)
        _apply_compile_info(norm_compute_graph_info, norm_info)
        # calc tiling key
        for tiling_case in tiling_case_list:
            _calc_tiling_key(tiling_case)
        # the const process is not performed when const and dynamic are mixed
        if not get_context().get("_const_and_dynamic_mixed") and current_compute.get("_mode") == "const":
            return _gen_const_tiling_case(norm_info, norm_compute_graph_info)

        return tiling_case_list


def _judge_current_compute_is_const():
    current_compute = get_context().get_current_compute()
    current_compute_shapes = current_compute.get("_input_shapes")
    for single_shape in current_compute_shapes:
        for single_dim in single_shape:
            if not isinstance(single_dim, (int, IntImm)):
                return False

    return True


def raise_error(message):
    """
    raise error
    """
    dict_args = {"errCode": "E90003", "detailed_cause": message}
    raise RuntimeError(dict_args, get_error_message(dict_args))


def get_block_size(dtype):
    """
    get block size according to dtype.
    """
    if dtype not in DTYPE_AND_BLOCK_SIZE_MAP:
        raise_error("[%s] is not support type in norm" % dtype)

    return DTYPE_AND_BLOCK_SIZE_MAP.get(dtype)


def get_transpose_entire_size(dtype):
    """
    get pad entire size according to dtype.
    """
    if dtype not in DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP:
        raise_error("[%s] is not support type in norm" % dtype)

    return DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP.get(dtype)


def get_broadcast_axis(broadcast_tensor):
    """
    get broadcast axis of broadcast tensor
    """
    dst_shape = util.shape_to_list(broadcast_tensor.shape)
    dst_shape_len = len(dst_shape)
    if not hasattr(broadcast_tensor.op, "input_tensors") or not broadcast_tensor.op.input_tensors:
        return list(range(dst_shape_len))

    src_shape = util.shape_to_list(broadcast_tensor.op.input_tensors[0].shape)
    src_shape_len = len(src_shape)
    if src_shape_len < dst_shape_len:
        src_shape = [1] * (dst_shape_len - src_shape_len) + src_shape

    broadcast_axis = []
    for idx in range(dst_shape_len):
        if not util.expr_equal(src_shape[idx], dst_shape[idx]):
            broadcast_axis.append(idx)

    return broadcast_axis


def judge_tvm_shape_equal(shape_a, shape_b):
    """
    compare two tvm shape
    """
    length_a = len(shape_a)
    length_b = len(shape_b)
    if length_a != length_b:
        return False
    for idx in range(length_a):
        if not util.expr_equal(shape_a[idx], shape_b[idx]):
            return False

    return True


def _traverse_tensor_set(root_tensor, visited_set, traverse_map, stop_tensor_set=None):
    visited_set.add(root_tensor)
    if stop_tensor_set is not None and root_tensor in stop_tensor_set:
        return
    if root_tensor in traverse_map:
        for input_tensor in traverse_map.get(root_tensor):
            _traverse_tensor_set(input_tensor, visited_set, traverse_map, stop_tensor_set)
    else:
        return


def _traverse_tensor_list(root_tensor, visited_list, traverse_map, stop_tensor_set=None):
    if root_tensor not in visited_list:
        visited_list.append(root_tensor)
    if stop_tensor_set is not None and root_tensor in stop_tensor_set:
        return
    if root_tensor in traverse_map:
        for input_tensor in traverse_map.get(root_tensor):
            _traverse_tensor_list(input_tensor, visited_list, traverse_map, stop_tensor_set)
    else:
        return


def _apply_compile_info(graph_info, norm_info):
    # common info
    core_num = get_soc_spec("CORE_NUM")
    min_block_size = get_block_size(graph_info.min_type)
    common_info = [core_num, min_block_size, norm_info.transpose_max_entire_size]
    add_compile_info_inner("_common_info", common_info)
    # whether there is output after reduce
    exist_output_after_reduce = False
    if graph_info.real_output_tensor_set & graph_info.after_reduce_tensor_set or \
            get_compile_info().get("_exist_output_after_reduce"):
        exist_output_after_reduce = True
    add_compile_info_inner("_exist_output_after_reduce", exist_output_after_reduce)
    # available size list of each compute, including:
    # common sch available size, workspace sch available size and aligned or removed in ub sch
    current_compute = get_context().get_current_compute()
    current_available_ub_size_list = current_compute.get("_available_ub_size_list")[:]
    current_norm_pattern = current_compute.get("_norm_pattern")
    pattern_and_available_size_map = get_compile_info().get("_available_ub_size")
    if pattern_and_available_size_map is None:
        pattern_and_available_size_map = {}
        add_compile_info_inner("_available_ub_size", pattern_and_available_size_map)
    pattern_and_available_size_map[current_norm_pattern] = current_available_ub_size_list

    pattern_and_block_size_map = get_compile_info().get("_block_size")
    if pattern_and_block_size_map is None:
        pattern_and_block_size_map = {}
        add_compile_info_inner("_block_size", pattern_and_block_size_map)
    pattern_and_block_size_map[current_norm_pattern] = get_block_size(graph_info.min_type)

    if graph_info.exist_vc_unsupported_type:
        add_compile_info_inner("_exist_vc_unsupported_type", graph_info.exist_vc_unsupported_type)


def _calc_reduce_product(before_reduce_shape, reduce_axis, block_size):
    """
    calculate reduce product
    """
    reduce_product = 1
    for reduce_idx in reduce_axis:
        reduce_dim = before_reduce_shape[reduce_idx]
        if isinstance(reduce_dim, int):
            if reduce_idx == len(before_reduce_shape) - 1:
                reduce_dim = (reduce_dim + block_size - 1) // block_size * block_size
            reduce_product *= reduce_dim

    return reduce_product


def _enable_align_and_remove_pad(is_continuous_pattern, pad_shape, reduce_axis,
                                 block_size, ub_size, entire_size, ub_factor=None):
    """
    determine whether to generate align and remove pad case
    """
    if not is_continuous_pattern:
        return False

    if len(pad_shape) < 2:
        return False

    # there are several common axis(current do not support AARR...)
    if len(pad_shape) - len(reduce_axis) > 1:
        return False

    reduce_product = _calc_reduce_product(pad_shape, reduce_axis, block_size)
    if reduce_product > ub_size:
        return False

    first_dim_is_one = util.expr_equal(pad_shape[0], 1)
    ub_factor_is_one = isinstance(ub_factor, int) and ub_factor == 1
    const_one_count = int(first_dim_is_one or ub_factor_is_one)
    for index in range(1, len(pad_shape)):
        if util.expr_equal(pad_shape[index], 1):
            const_one_count += 1
    # num of not_1 axes in ub must greater than or equal to 2
    if len(pad_shape) - const_one_count < 2:
        return False

    threshold_dim_size = ub_size // entire_size
    if isinstance(pad_shape[-1], int):
        is_last_dim_align = pad_shape[-1] % block_size == 0
        if is_last_dim_align:
            return False

        last_dim_align_size = (pad_shape[-1] + block_size - 1) // block_size * block_size
        if last_dim_align_size > threshold_dim_size:
            return False

    last_dim_lower_bound = util.get_bound(pad_shape[-1])[0]
    last_dim_align_lower_bound = (last_dim_lower_bound + block_size - 1) // block_size * block_size
    if last_dim_align_lower_bound > threshold_dim_size:
        return False

    return True


def _enable_partial_reorder(is_multi_reduce, shape_before_reduce, block_size):
    if not is_multi_reduce:
        return False
    if isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] >= block_size:
        return False

    return True


def _enable_last_reduce_align(is_last_reduce, shape_before_reduce, block_size):
    if not is_last_reduce:
        return False
    if isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] % block_size != 0:
        return False

    return True


def _enable_reduce_transpose(shape, reduce_axis, block_size, ub_size, entire_size, exist_vc_unsupported_type,
                             ub_factor=None):
    """
    determine whether to generate reduce transpose case
    """
    # not AR pattern
    is_ar = len(shape) == 2 and 0 not in reduce_axis and 1 in reduce_axis
    if not is_ar:
        return False

    first_dim_is_one = util.expr_equal(shape[0], 1)
    ub_factor_is_one = isinstance(ub_factor, int) and ub_factor == 1
    second_dim_is_one = util.expr_equal(shape[1], 1)

    # num of not_1 axes in ub must greater than or equal to 2
    if first_dim_is_one or ub_factor_is_one or second_dim_is_one:
        return False

    threshold_dim_size = ub_size // entire_size
    if isinstance(shape[-1], int):
        if shape[-1] > threshold_dim_size:
            return False
        if shape[-1] > REDUCE_TRANSPOSE_UPPER_THRESHOLD:
            return False
        if not exist_vc_unsupported_type:
            if shape[-1] % block_size == 0:
                return False

    last_dim_lower_bound = util.get_bound(shape[-1])[0]
    if last_dim_lower_bound > last_dim_lower_bound or last_dim_lower_bound > REDUCE_TRANSPOSE_UPPER_THRESHOLD:
        return False

    return True


def _add_tiling_case(norm_graph_info, norm_info):
    def __add_partial_reorder_tiling_case():
        for _block_split_axis in range(len(shape_before_reduce)):
            # block split on A axis
            if _block_split_axis in reduce_axis_index:
                continue
            for _ub_split_axis in reduce_axis_index:
                # last reduce and reduce dim is 1 don't need workspace sch that ub split last dim
                _illegal_workspace_case = _ub_split_axis in reduce_axis_index and \
                    _ub_split_axis == len(shape_before_reduce) - 1 and \
                    isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] == 1
                if _illegal_workspace_case:
                    continue
                _tiling_case = NormTilingCase()
                _tiling_case.block_split_axis_index = _block_split_axis
                _tiling_case.ub_split_axis_index = _ub_split_axis
                # tilingcase that block split 0 can enable multi_core
                _tiling_case.multi_core = _block_split_axis == 0
                _tiling_case.is_partial_reorder_case = True
                tiling_case_list.append(_tiling_case)

    def __add_all_reduce_tiling_case():
        # workspace sch
        for _ub_split_axis in range(len(shape_before_reduce)):
            _tiling_case = NormTilingCase()
            _tiling_case.ub_split_axis_index = _ub_split_axis
            _tiling_case.multi_core = False
            tiling_case_list.append(_tiling_case)
        # normal sch, dont split block and ub
        # product of known reduce dims is not greater than ub available size
        if reduce_product <= norm_info.available_ub_size:
            _tiling_case = NormTilingCase()
            _tiling_case.multi_core = False
            tiling_case_list.append(_tiling_case)

    def __add_aligned_in_ub_tiling_case():
        all_reduce_normal_case = norm_info.is_all_reduce
        _tiling_case = NormTilingCase()
        _tiling_case.block_split_axis_index = None if all_reduce_normal_case else 0
        _tiling_case.ub_split_axis_index = None if all_reduce_normal_case else 0
        _tiling_case.multi_core = not all_reduce_normal_case
        _tiling_case.is_aligned_in_ub_case = True
        tiling_case_list.append(_tiling_case)

    def __add_last_reduce_align_case():
        if reduce_product > norm_info.available_ub_size:
            return
        if norm_info.is_all_reduce:
            _tiling_case = NormTilingCase()
            _tiling_case.block_split_axis_index = None
            _tiling_case.ub_split_axis_index = None
            _tiling_case.multi_core = False
            _tiling_case.is_last_reduce_align_case = True
            tiling_case_list.append(_tiling_case)
        else:
            for _block_split_axis in range(len(shape_before_reduce)):
                if _block_split_axis in reduce_axis_index:
                    continue
                for _ub_split_axis in range(_block_split_axis, len(shape_before_reduce)):
                    if _ub_split_axis in reduce_axis_index:
                        continue
                    _tiling_case = NormTilingCase()
                    _tiling_case.block_split_axis_index = _block_split_axis
                    _tiling_case.ub_split_axis_index = _ub_split_axis
                    _tiling_case.multi_core = True
                    _tiling_case.is_last_reduce_align_case = True
                    tiling_case_list.append(_tiling_case)

    def __add_reduce_transpose_tiling_case(_is_align=False):
        _tiling_case = NormTilingCase()
        _tiling_case.block_split_axis_index = 0
        _tiling_case.ub_split_axis_index = 0
        _tiling_case.multi_core = True
        _tiling_case.is_reduce_transpose_case = True
        _tiling_case.is_last_reduce_align_case = _is_align
        tiling_case_list.append(_tiling_case)

    shape_before_reduce = util.shape_to_list(norm_info.shape_before_reduce)
    reduce_axis_index = norm_info.reduce_axis_indices
    block_size = get_block_size(norm_graph_info.min_type)
    reduce_product = _calc_reduce_product(shape_before_reduce, reduce_axis_index, block_size)
    tiling_case_list = []

    for block_split_axis in range(len(shape_before_reduce)):
        # block split on A axis
        if block_split_axis in reduce_axis_index:
            continue
        for ub_split_axis in range(len(shape_before_reduce)):
            # ub tiling axis is after block tiling axis if ub tiling split normal axis
            illegal_normal_case = ub_split_axis not in reduce_axis_index and \
                (ub_split_axis < block_split_axis or reduce_product > norm_info.available_ub_size)
            if illegal_normal_case:
                continue
            # last reduce and reduce dim is 1 don't need workspace sch that ub split last dim
            illegal_workspace_case = ub_split_axis in reduce_axis_index and \
                ub_split_axis == len(shape_before_reduce) - 1 and \
                isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] == 1
            if illegal_workspace_case:
                continue
            tiling_case = NormTilingCase()
            tiling_case.block_split_axis_index = block_split_axis
            tiling_case.ub_split_axis_index = ub_split_axis
            tiling_case.multi_core = True
            tiling_case_list.append(tiling_case)

    # some special cases need workspace sch with partial reorder:
    # 1. have multi discontinuous reduce axis
    # 2. last dim is less than block_size
    if _enable_partial_reorder(norm_info.is_discontinuous_reduce_axis, shape_before_reduce, block_size):
        __add_partial_reorder_tiling_case()

    # all reduce don't have A axis
    if norm_info.is_all_reduce:
        __add_all_reduce_tiling_case()

    # enable align and remove pad sch
    # 1. AR/ARR/... or RR/RRR...
    # 2. last axis is not align
    # 3. last axis * entire_size is less than or equal to ub tensor size
    # 4. num of not_1 axes in ub must greater than or equal to 2
    if _enable_align_and_remove_pad(norm_info.is_continuous_data_move, shape_before_reduce,
                                    reduce_axis_index, block_size,
                                    norm_info.pad_available_ub_size,
                                    norm_info.transpose_max_entire_size):
        __add_aligned_in_ub_tiling_case()

    # enable last reduce align sch
    # 1. last reduce
    # 2. last axis is align
    if _enable_last_reduce_align(norm_info.is_reduce_last_axis, shape_before_reduce, block_size):
        __add_last_reduce_align_case()

    # enable reduce transpose sch
    # 1. AR
    # 2. last axis * entire_size is less than or equal to ub tensor size
    # 3. num of not_1 axes in ub must greater than or equal to 2
    if _enable_reduce_transpose(shape_before_reduce, reduce_axis_index, block_size,
                                norm_info.reduce_transpose_available_ub_size,
                                norm_info.transpose_max_entire_size,
                                norm_graph_info.exist_vc_unsupported_type):
        __add_reduce_transpose_tiling_case()
        _enable_align_reduce_transpose = norm_graph_info.exist_vc_unsupported_type and \
            not (isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] % block_size != 0)
        if _enable_align_reduce_transpose:
            __add_reduce_transpose_tiling_case(_is_align=True)

    return tiling_case_list


def _gen_const_tiling_case(norm_info, graph_info):
    def __construct_inputs_and_outputs():
        # the order of ops inputs and sch inputs may not be consistent
        ops_input_shapes = current_compute.get("_input_shapes")
        for single_shape in ops_input_shapes:
            for single_tensor in graph_info.input_tensor_set:
                if judge_tvm_shape_equal(single_tensor.shape, single_shape):
                    inputs.append({"shape": util.shape_to_list(single_tensor.shape), "dtype": single_tensor.dtype})
                    break

        for single_tensor in graph_info.real_output_tensor_set:
            shape = util.shape_to_list(single_tensor.shape)
            outputs.append({"shape": shape, "dtype": single_tensor.dtype})

    def __select_tiling_format():
        if norm_info.is_all_reduce:
            # all reduce but reduce product can be put in ub, normal sch
            if before_reduce_product <= norm_info.available_ub_size:
                _tiling_format = {}
            # workspace sch
            else:
                _tiling_format = {"ub_axis": "int", "ub_factor": "int"}
        else:
            _tiling_format = {"block_axis": "int", "block_factor": "int", "ub_axis": "int", "ub_factor": "int"}

        return _tiling_format

    def __add_workspace_info_in_json():
        # used to calculate the size of workspace, 1 means before reduce and 0 means after reduce
        workspace_type_list = []
        # number of bytes of an element
        workspace_bytes_list = []
        for workspace_tensor in const_workspace_tensors:
            if workspace_tensor in graph_info.before_reduce_tensor_set:
                workspace_type_list.append(1)
            else:
                workspace_type_list.append(0)
            workspace_bytes_list.append(DTYPE_BYTE_MAPPING.get(workspace_tensor.dtype))

        after_reduce_align_shape = util.shape_to_list(norm_info.shape_after_reduce)[:]
        after_reduce_align_shape[-1] = (after_reduce_align_shape[-1] + block_size - 1) // block_size * block_size
        after_reduce_product = functools.reduce(lambda x1, x2: x1 * x2, after_reduce_align_shape)

        workspace_size = []
        workspace_type = []
        workspace_num = len(const_workspace_tensors)
        for i in range(workspace_num):
            if workspace_type_list[i] == 1:
                workspace_size.append(before_reduce_product * workspace_bytes_list[i])
            else:
                workspace_size.append(after_reduce_product * workspace_bytes_list[i])
            workspace_type.append(0)

        if workspace_num != 0:
            workspace_dict_in_json = {
                "num": workspace_num,
                "size": workspace_size,
                "type": workspace_type
            }
            get_op_context().add_build_json_result("workspace", workspace_dict_in_json)

    def __save_temp_disable_fuse_axes_info():
        _disable_fuse_axes = []
        _compile_info = get_compile_info()
        if "_disable_fuse_axes" in _compile_info:
            _disable_fuse_axes = _compile_info.get("_disable_fuse_axes")
            _compile_info["_disable_fuse_axes"] = []

        return _disable_fuse_axes

    def __rollback_disable_fuse_axes(_disable_fuse_axes):
        _compile_info = get_compile_info()
        if "_disable_fuse_axes" in _compile_info:
            _compile_info["_disable_fuse_axes"] = _disable_fuse_axes

    def __obtain_tiling_data_and_run_info():
        _run_info = do_op_tiling("AutoTiling", get_compile_info(), inputs, outputs)

        _tiling_format = __select_tiling_format()
        _tiling_data = decode(_run_info.get("tiling_data"), _tiling_format)

        const_tiling_case.block_split_axis_index = _tiling_data.get("block_axis")
        const_tiling_case.block_factor = _tiling_data.get("block_factor")
        const_tiling_case.ub_split_axis_index = _tiling_data.get("ub_axis")
        const_tiling_case.ub_factor = _tiling_data.get("ub_factor")
        const_tiling_case.multi_core = True if _run_info.get("block_dim") > 1 else False

        return _tiling_data, _run_info

    def __enable_double_buffer():
        # const enable db:
        # 1. don't have reduce fork or broadcast fork or after reduce mid_output
        # 2. isn't partial reorder case
        # 3. is multi reduce or need workpace or nlast reduce or align
        # 4. cannot use reduce transpose
        has_fork = graph_info.broadcast_fork_tensor_set or graph_info.special_after_reduce_tensor_set
        if has_fork:
            return False
        if const_tiling_case.is_partial_reorder_case:
            return False
        last_and_continuous_reduce = norm_info.is_reduce_last_axis and not norm_info.is_discontinuous_reduce_axis
        no_workspace_and_not_align = not need_workspace and before_reduce_shape[-1] % block_size != 0
        if last_and_continuous_reduce and no_workspace_and_not_align:
            return False
        # use reduce transpose
        if graph_info.exist_vc_unsupported_type and before_reduce_shape[-1] <= REDUCE_TRANSPOSE_UPPER_THRESHOLD:
            return False

        _compile_info = get_compile_info()
        _current_norm_pattern = get_context().get_current_compute().get("_norm_pattern")
        _available_ub_list = _compile_info.get("_available_ub_size").get(_current_norm_pattern)
        _available_ub_list[0] = norm_info.const_common_db_size
        _available_ub_list[1] = norm_info.const_workspace_db_size

        return True

    def __rollback_double_buffer(_ori_tiling_data, _ori_run_info):
        # in some cases, double buffer should not enable
        # 1. don't split block or ub
        # 2. ub split r axis but dont need workspace when disable db
        # 3. ub outer is 1
        _block_index = const_tiling_case.block_split_axis_index
        _ub_index = const_tiling_case.ub_split_axis_index
        _need_rollback_double_buffer = False

        if _block_index is None or _ub_index is None:
            _need_rollback_double_buffer = True
        else:
            if not need_workspace and _ub_index in reduce_axis_index:
                _need_rollback_double_buffer = True
            _ub_dim = before_reduce_shape[_ub_index] if _block_index != _ub_index else const_tiling_case.block_factor
            if _ub_dim // const_tiling_case.ub_factor == 1:
                _need_rollback_double_buffer = True

        if not _need_rollback_double_buffer:
            return _ori_tiling_data, _ori_run_info

        const_tiling_case.is_enable_db = False
        _compile_info = get_compile_info()
        _current_compute = get_context().get_current_compute()
        _cur_ub_list = _compile_info.get("_available_ub_size").get(_current_compute.get("_norm_pattern"))
        _ori_ub_size_list = get_context().get_current_compute().get("_available_ub_size_list")
        _cur_ub_list[0] = _ori_ub_size_list[0]
        _cur_ub_list[1] = _ori_ub_size_list[1]

        return __obtain_tiling_data_and_run_info()

    current_compute = get_context().get_current_compute()
    # flag of const
    add_compile_info_inner("_is_const", True)

    block_size = get_block_size(graph_info.min_type)
    before_reduce_shape = util.shape_to_list(norm_info.shape_before_reduce)[:]
    before_reduce_align_shape = before_reduce_shape[:]
    before_reduce_align_shape[-1] = (before_reduce_align_shape[-1] + block_size - 1) // block_size * block_size
    before_reduce_product = functools.reduce(lambda x1, x2: x1 * x2, before_reduce_align_shape)

    const_tiling_case = NormTilingCase()
    const_tiling_case.is_partial_reorder_case = _enable_partial_reorder(norm_info.is_discontinuous_reduce_axis,
                                                                        before_reduce_shape, block_size)
    const_workspace_tensors = graph_info.workspace_and_reduce_tensor_set \
        if const_tiling_case.is_partial_reorder_case else graph_info.workspace_tensor_set
    exist_workspace_after_reduce = bool(const_workspace_tensors & graph_info.after_reduce_tensor_set)
    add_compile_info_inner("_exist_workspace_after_reduce", exist_workspace_after_reduce)

    disable_fuse_axes = __save_temp_disable_fuse_axes_info()
    # before fuse axis
    ori_reduce_axis = get_compile_info().get("_ori_reduce_axis")
    # after fuse axis
    reduce_axis_index = norm_info.reduce_axis_indices
    add_compile_info_inner("_ori_reduce_axis", reduce_axis_index)
    ori_broadcast_axis = get_compile_info().get("_ori_broadcast_axis")
    if ori_broadcast_axis is not None:
        broadcast_axis = get_context().get_current_compute().get("_broadcast_axis")
        add_compile_info_inner("_ori_broadcast_axis", broadcast_axis)

    # the flag of invoking op_tiling interface during compilation
    add_compile_info_inner("_const_shape_post", False)
    inputs = []
    outputs = []
    __construct_inputs_and_outputs()

    need_workspace = \
        _calc_reduce_product(before_reduce_align_shape, reduce_axis_index, block_size) > norm_info.available_ub_size
    const_tiling_case.is_enable_db = __enable_double_buffer()
    # invoke op_tiling interface and decode
    tiling_data, run_info = __obtain_tiling_data_and_run_info()
    if const_tiling_case.is_enable_db:
        tiling_data, run_info = __rollback_double_buffer(tiling_data, run_info)
    # cases that const can enable reduce transpose sch
    # 1. AR
    # 2. last axis * entire_size is less than or equal to ub tensor size
    # 3. num of not_1 axes in ub must greater than or equal to 2
    const_tiling_case.is_reduce_transpose_case = _enable_reduce_transpose(
        before_reduce_shape, reduce_axis_index, block_size, norm_info.reduce_transpose_available_ub_size,
        norm_info.transpose_max_entire_size, graph_info.exist_vc_unsupported_type, const_tiling_case.ub_factor)
    # cases that const can enable align and remove pad sch:
    # 1. AR pattern
    # 2. last axis is not align
    # 3. last axis * entire_size is not larger than ub tensor size
    # 4. last two dims don't have 1
    # 5. ub_factor is not equal to 1
    const_tiling_case.is_aligned_in_ub_case = not const_tiling_case.is_reduce_transpose_case and \
        _enable_align_and_remove_pad(norm_info.is_continuous_data_move, before_reduce_shape,
                                     reduce_axis_index, block_size, norm_info.pad_available_ub_size,
                                     norm_info.transpose_max_entire_size, const_tiling_case.ub_factor)
    # cases that const can enable last reduce align sch
    # 1. last reduce
    # 2. last axis is align
    const_tiling_case.is_last_reduce_align_case = \
        _enable_last_reduce_align(norm_info.is_reduce_last_axis, before_reduce_shape, block_size)
    const_tiling_case.tiling_key = current_compute.get("_norm_pattern")

    # the flag of invoking op_tiling interface during running
    add_compile_info_inner("_const_shape_post", True)
    __rollback_disable_fuse_axes(disable_fuse_axes)
    if ori_reduce_axis is not None:
        add_compile_info_inner("_ori_reduce_axis", ori_reduce_axis)
    if ori_broadcast_axis is not None:
        add_compile_info_inner("_ori_broadcast_axis", ori_broadcast_axis)
    block_dims = get_compile_info().get("_const_block_dims")
    if block_dims is None:
        block_dims = {}
        add_compile_info_inner("_const_block_dims", block_dims)
    block_dims[const_tiling_case.tiling_key] = run_info.get("block_dim")

    # add workspace info in json
    if tiling_data.get("ub_axis") in reduce_axis_index and get_op_context() and \
            get_op_context().get_op_mode() != "dynamic":
        __add_workspace_info_in_json()

    return [const_tiling_case]


def _calc_tiling_key(tiling_case):
    block_split_axis = tiling_case.block_split_axis_index
    ub_split_axis = tiling_case.ub_split_axis_index

    current_compute = get_context().get_current_compute()
    all_broadcast_axis_known = 1 if current_compute.get("_broadcast_axis") is not None else 0
    norm_pattern = current_compute.get("_norm_pattern")

    db = 0
    sch_type = 0
    if tiling_case.is_last_reduce_align_case and tiling_case.is_reduce_transpose_case:
        sch_type = 5
    elif tiling_case.is_partial_reorder_case:
        sch_type = 1
    elif tiling_case.is_aligned_in_ub_case:
        sch_type = 2
    elif tiling_case.is_last_reduce_align_case:
        sch_type = 3
    elif tiling_case.is_reduce_transpose_case:
        sch_type = 4

    tiling_key = _get_tiling_key(db, all_broadcast_axis_known, sch_type, norm_pattern,
                                 block_split_axis, ub_split_axis)
    tiling_case.tiling_key = tiling_key


def _get_tiling_key(db, all_broadcast_axis_known, sch_type, norm_pattern, block_split_axis, ub_split_axis):
    """
    :param db: int number in [0, 1]. "0": enable db, "1": close db.
    :param all_broadcast_axis_known: int number in [0, 1]. "1": all broadcast axes of inputs are known and the same.
    :param sch_type: int number in [0, 9]. Diff numbers represent diff types of sch.
                     "0": normal sch, "1": partial sch, "2": align and remove pad in ub sch.
    :param norm_pattern: int number in [0, 999999].
    :param block_split_axis: int number in [0, 7] that represent index of split axis and 9 means None.
    :param ub_split_axis: int number in [0, 7] that represent index of split axis and 9 means None.
    :return: key(uint32)
    """
    key = db * (10 ** 9) * 2 + all_broadcast_axis_known * (10 ** 9) + sch_type * (10 ** 8) + norm_pattern * (10 ** 2)

    if block_split_axis is None:
        block_split_axis = 9
    key += block_split_axis * 10

    if ub_split_axis is None:
        ub_split_axis = 9
    key += ub_split_axis

    return key


def _norm_pre_build(*args):
    """
    to ensure that the numbers of input parameters are equal
    """
    def __add_fake_workspace(_tensors, _workspace_info, _actual_workspace_len):
        # _tensors: actual output + actual workspace
        _actual_workspace_start_index = -_actual_workspace_len
        if _actual_workspace_len == 0:
            _actual_workspace = []
        else:
            _actual_workspace = _tensors[_actual_workspace_start_index:]
            _tensors[_actual_workspace_start_index:] = []

        _fake_workspace_count = 0
        _actual_workspace_count = 0
        for _single_info in _workspace_info:
            # add fake workspace
            if _single_info == FAKE_WORKSPACE_SIZE:
                _fake_workspace = tvm.placeholder([], dtype="uint8",
                                                  name="fake_workspace_" + str(_fake_workspace_count))
                _tensors.append(_fake_workspace)
                _fake_workspace_count += 1
            else:
                _tensors.append(_actual_workspace[_actual_workspace_count])
                _actual_workspace_count += 1

    def __workspace_info_encode(_input_tuple):
        # _input_tuple: (workspace_type, workspace_byte)
        # fake workspace
        if len(_input_tuple) < 2:
            return FAKE_WORKSPACE_SIZE
        # after reduce workspace tensor is represented by a negative number
        if _input_tuple[0] == 0:
            return -1 * _input_tuple[1]

        return _input_tuple[1]

    schedule_list, _ = args
    ori_workspace_len_list = []
    total_common_workspace_list = []
    total_reduce_workspace_list = []
    total_broadcast_fork_workspace_list = []
    max_common_workspace_len = 0
    max_reduce_workspace_len = 0
    max_broadcast_fork_workspace_len = 0

    for same_compute_schs in schedule_list:
        for single_sch in same_compute_schs:
            sch_context = util.get_sch_additional_entry(single_sch, "context")
            # key is "common_workspace", "reduce_workspace", "broadcast_fork_workspace",
            # value is [(type, bytes), ...]
            workspace_info_dict = sch_context.get("_workspace_info")

            total_common_workspace_list.append(workspace_info_dict.get("common_workspace"))
            single_common_workspace_len = len(total_common_workspace_list[-1])
            if single_common_workspace_len > max_common_workspace_len:
                max_common_workspace_len = single_common_workspace_len

            total_reduce_workspace_list.append(workspace_info_dict.get("reduce_workspace"))
            single_reduce_workspace_len = len(total_reduce_workspace_list[-1])
            if single_reduce_workspace_len > max_reduce_workspace_len:
                max_reduce_workspace_len = single_reduce_workspace_len

            total_broadcast_fork_workspace_list.append(workspace_info_dict.get("broadcast_fork_workspace"))
            single_broadcast_fork_workspace_len = len(total_broadcast_fork_workspace_list[-1])
            if single_broadcast_fork_workspace_len > max_broadcast_fork_workspace_len:
                max_broadcast_fork_workspace_len = single_broadcast_fork_workspace_len

            ori_workspace_len_list.append(single_common_workspace_len + single_reduce_workspace_len +
                                          single_broadcast_fork_workspace_len)

    sch_count = 0
    workspace_compile_info = {}
    exist_workspace_after_reduce = False
    for same_compute_schs in schedule_list:
        for single_sch in same_compute_schs:
            # actual common workspace + fake common workspace
            # actual reduce workspace + fake reduce workspace
            # actual broadcast fork workspace + fake broadcast fork workspace
            single_out = []
            for workspace in total_common_workspace_list[sch_count]:
                single_out.append(__workspace_info_encode(workspace))
            single_out.extend([FAKE_WORKSPACE_SIZE] * (max_common_workspace_len -
                                                       len(total_common_workspace_list[sch_count])))

            for workspace in total_reduce_workspace_list[sch_count]:
                single_out.append(__workspace_info_encode(workspace))
            single_out.extend([FAKE_WORKSPACE_SIZE] * (max_reduce_workspace_len -
                                                       len(total_reduce_workspace_list[sch_count])))

            for workspace in total_broadcast_fork_workspace_list[sch_count]:
                single_out.append(__workspace_info_encode(workspace))
            single_out.extend([FAKE_WORKSPACE_SIZE] * (max_broadcast_fork_workspace_len -
                                                       len(total_broadcast_fork_workspace_list[sch_count])))

            for encode_workspace in single_out:
                if encode_workspace < 0:
                    exist_workspace_after_reduce = True

            workspace_compile_info[single_sch.tiling_key] = single_out
            real_outs = util.get_sch_additional_entry(single_sch, "real_outs")
            ori_workspace_len = ori_workspace_len_list[sch_count]
            __add_fake_workspace(real_outs, single_out, ori_workspace_len)
            util.add_sch_additional_entry(single_sch, "real_outs", real_outs)
            sch_count += 1

    add_compile_info_inner("_exist_workspace_after_reduce", exist_workspace_after_reduce)
    add_compile_info_inner("_workspace_info", workspace_compile_info)


def _norm_post_build():
    """
    encode normal vars in norm sch
    """
    def _encode_var_name(_var_names):
        after_encode_name = []
        for name in _var_names:
            names = name[1:].split('_')
            if names[0] == 'dim':
                if len(name) == 3:
                    after_encode_name.append(10000 + int(names[1]) * 100 + int(names[2]))
                else:
                    after_encode_name.append(20000 + int(names[1]))
            elif names[0] == 'block':
                after_encode_name.append(30000)
            elif names[0] == 'ub':
                after_encode_name.append(40000)
            elif names[0] == 'ori' and names[1] == 'dim':
                after_encode_name.append(50000 + int(names[2]) * 100 + int(names[3]))
            else:
                raise_error("unknown var name in norm schedule, please check")

        return after_encode_name

    normal_vars = get_compile_info().get(CompileInfo.NORMAL_VARS)
    norm_vars = {}
    for tiling_key, var_names in normal_vars.items():
        norm_vars[tiling_key] = _encode_var_name(var_names)
    add_compile_info_inner("_norm_vars", norm_vars)


@register_build_pointcut(pattern=Pattern.NORM)
def build_pointcut(func, *args, **kwargs):
    """
    norm build pointcut
    """
    _norm_pre_build(*args)
    func(*args, **kwargs)
    _norm_post_build()


class NormComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """
    def __init__(self, output_tensors: Iterable[Tensor]):
        """
        Initialize containers and try to collect info
        """
        self.output_tensor_set: Optional[Set[Tensor]] = None
        # real_output_tensor_set: output set doesn't contain fake_node
        self.real_output_tensor_set: Optional[Set[Tensor]] = None
        # real_pure_output_tensor_set: output set doesn't contain fake_node and middle output
        self.real_pure_output_tensor_set: Optional[Set[Tensor]] = None
        self.tensor_consumers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_producers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_list: Optional[List[Tensor]] = None
        # extra info initialized by hooks
        self.reduce_tensor_set: Set[Tensor] = set()
        self.broadcast_tensor_set: Set[Tensor] = set()
        self.elewise_tensor_set: Set[Tensor] = set()
        self.set_value_tensor_set: Set[Tensor] = set()
        self.input_tensor_set: Set[Tensor] = set()
        self.non_gm_input_tensor_set: Set[Tensor] = set()
        # extra info initialized after pre-initialization
        self.mid_output_tensor_set: Set[Tensor] = set()
        self.mid_tensor_set: Set[Tensor] = set()
        # res tensor
        self.endpoint_output_tensor: Tensor = None
        # workspace tensor and cache clone tensor
        self.workspace_tensor_set: Set[Tensor] = set()
        self.workspace_and_reduce_tensor_set: Set[Tensor] = set()
        self.cache_clone_tensor_list: List[Tensor] = []
        self.cache_clone_tensor_and_num_path_map: Dict[Tensor, int] = {}
        # extra workspace tensor and its info
        self.workspace_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # extra split tensor and its sub_graph
        self.split_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # special tensors
        self.special_after_reduce_tensor_set: Set[Tensor] = set()
        self.broadcast_fork_tensor_set: Set[Tensor] = set()
        self.reduce_fork_tensor_set: Set[Tensor] = set()
        self.after_mid_out_and_before_broadcast_tensor_set: Set[Tensor] = set()
        self.before_mid_out_and_after_reduce_tensor_set: Set[Tensor] = set()
        # before_reduce/after_reduce/other tensor set
        self.before_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.after_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.other_tensor_set: Optional[Set[Tensor]] = set()
        # special cast tensor that need reuse
        self.special_cast_tensor_and_ori_tensor_map: Dict[Tensor, Tensor] = {}
        # tensor and set value actions map
        self.tensor_and_set_value_actions_map = {}
        # max type and min type in graph
        self.max_type: Optional[str] = None
        self.min_type: Optional[str] = None
        # exist vc unsupported type flag
        self.exist_vc_unsupported_type = False
        # do info collection
        self.collect_info(output_tensors)
        self.fake_node()
        self.get_sch_set_value_tensors()
        self.get_tensors_before_and_after_reduce()
        self.get_tensors_in_fork()
        self.get_mid_out_special_tensors()
        self.find_workspace_tensor()
        self.is_exist_vc_unsupported_type()

    def collect_info(self, output_tensors: Iterable[Tensor]):
        """
        Collect necessary information
        """
        self.output_tensor_set = set(output_tensors)
        self.tensor_list, self.tensor_consumers_map, self.tensor_producers_map = \
            self.dfs_compute_graph(self.output_tensor_set,
                                   (   # self.input_tensor_set hook
                                       (lambda _tensor: isinstance(_tensor.op, PlaceholderOp),
                                        lambda _tensor: self.input_tensor_set.add(_tensor),
                                        lambda _tensor: self.non_gm_input_tensor_set.add(_tensor)
                                        if not _tensor.op.input_tensors else None),
                                       # self.reduce_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("reduce") != -1,
                                        lambda _tensor: self.reduce_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.broadcast_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("broadcast") != -1,
                                        lambda _tensor: self.broadcast_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.elewise_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("elewise") != -1,
                                        lambda _tensor: self.elewise_tensor_set.add(_tensor),
                                        lambda _tensor: None),
                                       # self.set_value_tensor_set hook
                                       (lambda _tensor: _tensor.op.tag.find("set_value") != -1,
                                        lambda _tensor: self.set_value_tensor_set.add(_tensor),
                                        lambda _tensor: None)
                                   ))
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor
        self.gen_endpoint_output_tensor()

    def gen_endpoint_output_tensor(self):
        """
        get endpoint tensor
        """
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map.get(output_tensor):
                self.endpoint_output_tensor = output_tensor
                break

    def gen_mid_tensor_sets(self):
        """
        get mid tensors
        """
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map.get(tensor):
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    def fake_node(self):
        """
        do fake node
        """
        # after collect_info, middle output tensors have been assured
        # fake_node does not need them as producers for it
        def _fake_node_compute(tensors):
            # fake_node must be the biggest node (type and shape)
            dtype = tensors[0].dtype
            dim_length = max(len(t.shape) for t in tensors)
            shape = [1] * dim_length

            # update fake_node's shape and dtype
            for tensor_i in tensors:
                if DTYPE_BYTE_MAPPING.get(tensor_i.dtype) > DTYPE_BYTE_MAPPING.get(dtype):
                    dtype = tensor_i.dtype
                shape_i = util.shape_to_list(tensor_i.shape)
                diff_length = dim_length - len(shape_i)
                shape_i = [1] * diff_length + shape_i
                for j in range(diff_length, dim_length):
                    if util.equals_one(shape[j]):
                        shape[j] = shape_i[j]
                    elif not util.expr_equal(shape[j], shape_i[j]) and not util.equals_one(shape_i[j]):
                        shape[j] = tvm.max(shape_i[j], shape[j])

            def __compute(*indexes):
                _res = tvm.const(1, dtype)
                for _tensor in tensors:
                    _cur_indexes = []
                    for _idx, _dim in enumerate(_tensor.shape):
                        if util.equals_one(_dim):
                            _cur_indexes.append(0)
                        else:
                            _cur_indexes.append(indexes[_idx])
                    _res *= tvm.expr.Cast(dtype, _tensor(*_cur_indexes))

                return _res

            with tvm.tag_scope(FAKE_NODE_TAG):
                res = tvm.compute(shape, __compute, name="fake_node")

            return res

        self.real_output_tensor_set = self.output_tensor_set
        self.real_pure_output_tensor_set = self.real_output_tensor_set - self.mid_output_tensor_set
        if len(self.real_pure_output_tensor_set) > 1:
            fake_out = _fake_node_compute(list(self.real_pure_output_tensor_set))
            # update info with fake_node as output
            self.collect_info([fake_out])

    @staticmethod
    def dfs_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor],
                          hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                             Callable[[Tensor], Any],
                                             Callable[[Tensor], Any]], ...],
                          stop_tensor_set=None):
        """
        compute graph using dfs algorithm
        """
        def recursive_func(_root_tensor: Tensor,
                           _visited_list: Set[Tensor],
                           _tensor_consumers_map: Dict[Tensor, Set[Tensor]],
                           _tensor_producers_map: Dict[Tensor, Set[Tensor]],
                           _hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                               Callable[[Tensor], Any],
                                               Callable[[Tensor], Any]], ...]):
            _visited_list.add(_root_tensor)
            _tensor_producers_map.setdefault(_root_tensor, set())
            _tensor_consumers_map.setdefault(_root_tensor, set())
            for hook in hooks:
                if hook[0](_root_tensor):
                    hook[1](_root_tensor)
                else:
                    hook[2](_root_tensor)
            for in_tensor in _root_tensor.op.input_tensors:
                _tensor_consumers_map.setdefault(in_tensor, set())
                _tensor_consumers_map.get(in_tensor).add(_root_tensor)
                _tensor_producers_map.get(_root_tensor).add(in_tensor)
                if stop_tensor_set is not None and in_tensor in stop_tensor_set:
                    _visited_list.add(in_tensor)
                    _tensor_producers_map[in_tensor] = set()
                    continue
                recursive_func(in_tensor,
                               _visited_list,
                               _tensor_consumers_map,
                               _tensor_producers_map,
                               _hooks)

        visited_list = set()
        tensor_consumers_map = {}
        tensor_producers_map = {}
        if isinstance(root_tensor, (list, tuple, set)):
            for tensor in root_tensor:
                recursive_func(tensor, visited_list,
                               tensor_consumers_map,
                               tensor_producers_map,
                               hooks)
        elif isinstance(root_tensor, Tensor):
            recursive_func(root_tensor, visited_list,
                           tensor_consumers_map, tensor_producers_map,
                           hooks)

        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def get_sch_set_value_tensors(self):
        """
        get set value tensors that schedule need to handle
        """
        current_compute = get_context().get_current_compute()
        if current_compute and current_compute.get("_in_5hd_process"):
            sch_set_value_actions = padding.calc_padding([self.endpoint_output_tensor])
            self.tensor_and_set_value_actions_map = classify_actions(sch_set_value_actions)

    def get_tensors_before_and_after_reduce(self):
        """
        get before reduce tensors and after reduce tensors
        """
        # Assume all reduce node have the same shape, axis and keepdims
        reduce_tensor = list(self.reduce_tensor_set)[0]
        shape_after_reduce = list(reduce_tensor.shape)
        shape_before_reduce = list(reduce_tensor.op.input_tensors[0].shape)

        self.max_type = self.tensor_list[0].dtype
        self.min_type = self.tensor_list[0].dtype

        for item in self.tensor_list:
            if DTYPE_BYTE_MAPPING.get(item.dtype) > DTYPE_BYTE_MAPPING.get(self.max_type):
                self.max_type = item.dtype
            elif DTYPE_BYTE_MAPPING.get(item.dtype) < DTYPE_BYTE_MAPPING.get(self.min_type):
                self.min_type = item.dtype
            if judge_tvm_shape_equal(list(item.shape), shape_before_reduce):
                self.before_reduce_tensor_set.add(item)
            elif judge_tvm_shape_equal(list(item.shape), shape_after_reduce):
                self.after_reduce_tensor_set.add(item)
            else:
                self.other_tensor_set.add(item)

    def get_tensors_in_fork(self):
        """
        get tensors in fork
        """
        # broadcast fork tensor
        for broadcast_tensor in self.broadcast_tensor_set:
            visited_set = set()
            _traverse_tensor_set(broadcast_tensor, visited_set, self.tensor_producers_map)
            # visited tensor cannot include reduce tensor
            if visited_set & self.reduce_tensor_set:
                continue
            self.broadcast_fork_tensor_set.update(visited_set)
        # output reduce fork tensor
        for out_tensor in self.real_pure_output_tensor_set & self.after_reduce_tensor_set:
            visited_set = set()
            _traverse_tensor_set(out_tensor, visited_set, self.tensor_producers_map, self.reduce_tensor_set)
            self.reduce_fork_tensor_set.update(visited_set)
        self.special_after_reduce_tensor_set.update(self.reduce_fork_tensor_set)

    def get_mid_out_special_tensors(self):
        """
        get some special tensors
        """
        for mid_output in self.mid_output_tensor_set:
            if mid_output in self.after_reduce_tensor_set:
                # after mid output tensor and before broadcast
                after_visited_set = set()
                _traverse_tensor_set(mid_output, after_visited_set, self.tensor_consumers_map,
                                     self.broadcast_tensor_set)
                self.after_mid_out_and_before_broadcast_tensor_set.update(
                    after_visited_set - self.broadcast_tensor_set)
                # before mid output tensor and after reduce
                before_visited_set = set()
                _traverse_tensor_set(mid_output, before_visited_set, self.tensor_producers_map,
                                     self.reduce_tensor_set)
                self.before_mid_out_and_after_reduce_tensor_set.update(before_visited_set)
            if "elewise_single_cast" in mid_output.op.tag:
                # mid_output tensor is fp32->fp16
                # its consumer tensor is fp16->fp32
                # so, its consumer tensor can reuse its producer tensor
                consumer_set = self.tensor_consumers_map.get(mid_output)
                producer_set = self.tensor_producers_map.get(mid_output)
                if len(consumer_set) != 1 or len(producer_set) != 1:
                    continue
                consumer_tensor = list(consumer_set)[0]
                producer_tensor = list(producer_set)[0]
                find_special_cast_tensor = "elewise_single_cast" in consumer_tensor.op.tag and \
                    mid_output.dtype == "float16" and consumer_tensor.dtype == "float32" and \
                    producer_tensor.dtype == "float32"
                if find_special_cast_tensor:
                    self.special_cast_tensor_and_ori_tensor_map[consumer_tensor] = producer_tensor

        self.special_after_reduce_tensor_set.update(
            self.after_mid_out_and_before_broadcast_tensor_set | self.before_mid_out_and_after_reduce_tensor_set
        )

    def find_workspace_tensor(self):
        """
        find workspace tensor
        """
        def _find_possible_cross_hierarchy_tensor(_idx, _current_tensor):
            # stop when current tensor is end tensor or cross hierarchy tensor
            if _current_tensor == self.endpoint_output_tensor or \
                    _current_tensor in cross_hierarchy_tensor_set:
                path_index_and_tensor_map.get(_idx).add(_current_tensor)
                return
            # _idx represents the index of path
            if _current_tensor in self.reduce_tensor_set:
                path_index_and_tensor_map.get(_idx).add(_current_tensor)
            for _consumer_tensor in self.tensor_consumers_map.get(_current_tensor):
                # Reduce nodes exist on this path
                if _consumer_tensor in self.reduce_tensor_set:
                    path_index_and_tensor_map.get(_idx).add(_consumer_tensor)
                _find_possible_cross_hierarchy_tensor(_idx, _consumer_tensor)

        def _judge_workspace_or_cache_clone(cross_hierarchy_tensor):
            # mte2 mte3 count 1.5
            # vector count 1
            # reduce and broadcast choose workspace
            visited_list = []
            # workspace need mte3 and at least 2 mte2
            workspace_count = 1.5 + 2 * 1.5
            cache_clone_count = 0
            _traverse_tensor_list(cross_hierarchy_tensor, visited_list, self.tensor_producers_map)
            for count_tensor in visited_list:
                if count_tensor in self.reduce_tensor_set | self.broadcast_tensor_set:
                    return "workspace", visited_list
                if count_tensor in self.input_tensor_set:
                    workspace_count += 1 * 1.5
                    cache_clone_count += 2 * 1.5
                else:
                    workspace_count += 1
                    cache_clone_count += 1 * 2
            if workspace_count > cache_clone_count:
                return "cache_clone", visited_list
            return "workspace", visited_list

        def _update_cache_clone_list(_visited_tensor):
            for tensor in _visited_tensor:
                if tensor not in self.cache_clone_tensor_list:
                    self.cache_clone_tensor_list.append(tensor)

        cross_hierarchy_tensor_set = set()
        tensors_and_num_path_map = {}
        # find the cross hierarchy tensors
        for single_tensor in self.tensor_producers_map:
            if len(self.tensor_consumers_map.get(single_tensor)) > 1:
                path_index_and_tensor_map = {}
                for idx, consumer_tensor in enumerate(self.tensor_consumers_map.get(single_tensor)):
                    path_index_and_tensor_map[idx] = set()
                    _find_possible_cross_hierarchy_tensor(idx, consumer_tensor)
                visited_special_tensor_set = path_index_and_tensor_map.get(0)
                count = 1
                for _, special_tensor in path_index_and_tensor_map.items():
                    if visited_special_tensor_set != special_tensor:
                        cross_hierarchy_tensor_set.add(single_tensor)
                        count += 1
                tensors_and_num_path_map[single_tensor] = count

        for single_tensor in cross_hierarchy_tensor_set - self.input_tensor_set:
            judge_result, visited_tensor = _judge_workspace_or_cache_clone(single_tensor)
            if judge_result == "workspace":
                self.workspace_tensor_set.add(single_tensor)
                self.workspace_and_reduce_tensor_set.add(single_tensor)
                continue
            _update_cache_clone_list(visited_tensor)

        for single_tensor in self.cache_clone_tensor_list:
            self.cache_clone_tensor_and_num_path_map[single_tensor] = \
                tensors_and_num_path_map.get(single_tensor) if single_tensor in tensors_and_num_path_map else 1

        for single_tensor in self.reduce_tensor_set:
            self.workspace_and_reduce_tensor_set.add(single_tensor)

        # get the sub_graph of split tensor
        for split_tensor in self.workspace_and_reduce_tensor_set:
            tensor_list, tensor_consumers_map, tensor_producers_map = \
                self.dfs_compute_graph(split_tensor, (), self.workspace_and_reduce_tensor_set)
            self.split_tensor_and_sub_graph_map[split_tensor] = {
                "sub_tensor_list": tensor_list,
                "sub_tensor_consumers_map": tensor_consumers_map,
                "sub_tensor_producers_map": tensor_producers_map
            }
        tensor_list, tensor_consumers_map, tensor_producers_map = \
            self.dfs_compute_graph(self.endpoint_output_tensor, (), self.workspace_and_reduce_tensor_set)
        self.split_tensor_and_sub_graph_map[self.endpoint_output_tensor] = {
            "sub_tensor_list": tensor_list,
            "sub_tensor_consumers_map": tensor_consumers_map,
            "sub_tensor_producers_map": tensor_producers_map
        }

        # get the sub_graph of workspace tensor
        for workspace_tensor in self.workspace_tensor_set:
            tensor_list, tensor_consumers_map, tensor_producers_map = \
                self.dfs_compute_graph(workspace_tensor, (), self.workspace_tensor_set)
            self.workspace_tensor_and_sub_graph_map[workspace_tensor] = {
                "sub_tensor_list": tensor_list,
                "sub_tensor_consumers_map": tensor_consumers_map,
                "sub_tensor_producers_map": tensor_producers_map
            }
        tensor_list, tensor_consumers_map, tensor_producers_map = \
            self.dfs_compute_graph(self.endpoint_output_tensor, (), self.workspace_tensor_set)
        self.workspace_tensor_and_sub_graph_map[self.endpoint_output_tensor] = {
            "sub_tensor_list": tensor_list,
            "sub_tensor_consumers_map": tensor_consumers_map,
            "sub_tensor_producers_map": tensor_producers_map
        }

        # check whether the outputs are legal:
        # 1. real pure outputs must in the last sub graph
        last_sub_tensor_set = \
            set(self.split_tensor_and_sub_graph_map.get(self.endpoint_output_tensor).get("sub_tensor_list"))
        if not self.real_pure_output_tensor_set & last_sub_tensor_set == self.real_pure_output_tensor_set:
            raise_error("norm schedule does not support output fork now")

    def is_exist_vc_unsupported_type(self):
        for tensor in self.reduce_tensor_set:
            if len(tensor.op.input_tensors[0].shape) - 1 in util.get_reduce_axis_indexes(tensor):
                if ("reduce_max" in tensor.op.tag or "reduce_min" in tensor.op.tag) and tensor.dtype == "float32":
                    self.exist_vc_unsupported_type = True


class NormInfo:
    """
    class for norm information
    """
    def __init__(self, compute_graph_info: NormComputeGraphInfo):
        self.graph_info: NormComputeGraphInfo = compute_graph_info
        # Assume all reduce node have the same shape, axis and keepdims
        self.reduce_tensor: Tensor = tuple(compute_graph_info.reduce_tensor_set)[0]
        self.all_axes: List[Var] = util.get_reduce_all_axes(self.reduce_tensor)
        self.reduce_axes: List[Var] = util.get_reduce_axes(self.reduce_tensor)

        self.shape_before_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.op.input_tensors[0].shape)
        self.shape_after_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.shape)
        self.reduce_axis_indices: List[int] = sorted(util.get_reduce_axis_indexes(self.reduce_tensor))
        self.is_reduce_last_axis: bool = len(self.shape_before_reduce) - 1 in self.reduce_axis_indices
        self.is_all_reduce: bool = len(self.shape_before_reduce) == len(self.reduce_axis_indices)
        self.is_none_reduce = self._judge_non_reduce()
        self.reduce_pattern = self._gen_reduce_pattern()
        # RAR/ARAR/...
        self.is_discontinuous_reduce_axis = self._judge_discontinuous_reduce_axis()
        # AR/ARR/ARRR... or RR/RRR...
        self.is_continuous_data_move = self.is_reduce_last_axis and not self.is_discontinuous_reduce_axis
        # bytes of max_type and min_type
        self.max_bytes = DTYPE_BYTE_MAPPING.get(compute_graph_info.max_type)
        self.min_bytes = DTYPE_BYTE_MAPPING.get(compute_graph_info.min_type)
        # transpose entire size
        self.transpose_max_entire_size = self._calc_transpose_max_entire_size()
        # ub size
        self.soc_ub_size = get_soc_spec("UB_SIZE")
        self.temp_ub_size = 0
        self.broadcast_temp_size = 0
        self.available_ub_size = 0
        self.workspace_available_ub_size = 0
        self.pad_available_ub_size = 0
        self.reduce_transpose_available_ub_size = 0
        self.const_common_db_size = 0
        self.const_workspace_db_size = 0
        # calculate available ub size
        self._calc_coexisting_quantities_and_temp_buffer_size()

    def _judge_non_reduce(self):
        for reduce_axis in self.reduce_axis_indices:
            if not util.expr_equal(self.shape_before_reduce[reduce_axis], 1):
                return False

        return True

    def _gen_reduce_pattern(self):
        shape_len = len(self.shape_before_reduce)
        pattern_list = ("R" if index in self.reduce_axis_indices else "A" for index in range(shape_len))

        return ''.join(pattern_list)

    def _judge_discontinuous_reduce_axis(self):
        if len(self.reduce_axis_indices) <= 1:
            return False
        discontinuous_axis_num = 1
        for i in range(1, len(self.reduce_axis_indices)):
            if self.reduce_axis_indices[i] != self.reduce_axis_indices[i - 1] + 1:
                discontinuous_axis_num += 1

        return discontinuous_axis_num >= 2

    def _calc_transpose_max_entire_size(self):
        """
        calculate transposs max entire size
        """
        return max(get_transpose_entire_size(self.graph_info.min_type) * self.min_bytes // self.max_bytes,
                   get_transpose_entire_size(self.graph_info.max_type))

    def _calc_coexisting_quantities_and_temp_buffer_size(self):
        """
        calculate the number of coexisting quantities and temp buffer size
        """
        def _correct_ub_size_by_cmp_sel(_tensor):
            if util.is_vcmp_insn(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE * (VCMP_INPUT_NUMBER - len(_tensor.op.input_tensors))
            if util.is_vsel_insn(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE * (VSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))
                if util.is_v200() and (VSEL_INPUT_NUMBER == len(_tensor.op.input_tensors)):
                    self.temp_ub_size += BLOCK_SIZE_BYTE
            if util.is_vcmpsel_insn(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE * (VCMPSEL_INPUT_NUMBER - len(_tensor.op.input_tensors))

        def _correct_ub_size_by_reduce(_tensor):
            tag = _tensor.op.tag
            dtype = _tensor.dtype

            def _reduce_sum_space():
                self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE

            def _reduce_max_space():
                if dtype in ["float32", "int32"]:
                    if not self.is_reduce_last_axis:
                        self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                    else:
                        self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE + BLOCK_SIZE_BYTE + BLOCK_SIZE_BYTE
                elif dtype in ["float16", ]:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                else:
                    raise RuntimeError("Not support dtype in reduce_max(min) is %s" % dtype)

            def _reduce_prod_space():
                if not self.is_reduce_last_axis:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                else:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE + BLOCK_SIZE_BYTE + BLOCK_SIZE_BYTE

            if tag.find("reduce") != -1:
                if tag in ["reduce_sum"]:
                    _reduce_sum_space()
                elif tag in ["reduce_max", "reduce_min"]:
                    _reduce_max_space()
                elif tag in ["reduce_prod"]:
                    _reduce_prod_space()
                else:
                    raise RuntimeError("Unknown reduce_insn is %s" % tag)

        def _calc_current_coexist_node(_tensor, _producers_map, _mode):
            def __handle_input_and_output(__current_coexist_node, __dst_is_no_reuse):
                _extra_node = 1 if not __dst_is_no_reuse else 0
                if _mode == REDUCE_TRANS_MODE:
                    pass
                elif _mode == PAD_MODE:
                    if _tensor in self.graph_info.input_tensor_set:
                        __current_coexist_node += ALIGN_AND_REMOVE_PAD_EXTRA_NODES + 1
                    if _tensor in self.graph_info.real_output_tensor_set:
                        # the space that remove pad buffer need
                        _pad_space = __current_coexist_node + _extra_node + 1 -\
                            len(self.graph_info.tensor_producers_map.get(_tensor)) + ALIGN_AND_REMOVE_PAD_EXTRA_NODES
                        __current_coexist_node = max(__current_coexist_node, _pad_space)
                else:
                    if _tensor in self.graph_info.real_output_tensor_set:
                        # the space that enable no_overlap 3 need
                        _no_overlap_three_space = __current_coexist_node + _extra_node - \
                             len(self.graph_info.tensor_producers_map.get(_tensor)) + NO_OVERLAP_THREE_EXTRA_NODES
                        __current_coexist_node = max(__current_coexist_node, _no_overlap_three_space)

                return __current_coexist_node

            def __handle_reduce(__current_coexist_node):
                if _mode == REDUCE_TRANS_MODE:
                    __current_coexist_node += REDUCE_TRANSPOSE_EXTRA_NODES
                    return __current_coexist_node

                if _mode == COMMON_MODE:
                    _is_multi_last_reduce = self.is_reduce_last_axis and self.is_discontinuous_reduce_axis
                    _src = _tensor.op.input_tensors[0]
                    _is_src_can_not_reuse = len(self.graph_info.tensor_consumers_map.get(_src)) > 1
                    if _is_multi_last_reduce and _is_src_can_not_reuse:
                        __current_coexist_node += 1

                # use vcmax or vcmin
                _reduce_last_axis_max_or_min = \
                    _tensor.op.tag in ("reduce_max", "reduce_min") and self.is_reduce_last_axis
                _last_dim = util.shape_to_list(self.shape_before_reduce)[-1]
                _last_dim_is_known = isinstance(_last_dim, int)
                if get_soc_spec("SHORT_SOC_VERSION") != ASCEND_910B:
                    # 910B use vcmax and vcmin all the time
                    _reduce_max_or_min_use_vc = _tensor.dtype == "float16"
                    if get_context().get_current_compute().get("_mode") == "const":
                        _reduce_max_or_min_use_vc = _reduce_max_or_min_use_vc and _last_dim > get_block_size("float16")
                else:
                    _reduce_max_or_min_use_vc = _tensor.dtype in ("float16", "float32")

                _reduce_max_or_min_need_extra_node = _reduce_last_axis_max_or_min and _reduce_max_or_min_use_vc
                if _mode != WORKSPACE_MODE and _reduce_max_or_min_need_extra_node:
                    __current_coexist_node += 1

                _eight_blocks_value = ONE_REPEAT_PROCESS_BYTES // DTYPE_BYTE_MAPPING.get(_tensor.dtype)
                _reduce_sum_use_vc = _tensor.op.tag == "reduce_sum" and self.reduce_pattern == "AR"
                _last_dim_is_not_larger_than_eight_blocks = _last_dim_is_known and _last_dim <= _eight_blocks_value
                if _mode != WORKSPACE_MODE and _reduce_sum_use_vc and not _last_dim_is_not_larger_than_eight_blocks:
                    __current_coexist_node += 1

                return __current_coexist_node

            def __handle_unified_broadcast(__current_coexist_node):
                _broadcast_axis = get_broadcast_axis(_tensor)
                _broadcast_num = len(_broadcast_axis)
                _tensor_shape = util.shape_to_list(_tensor.shape)
                _is_last_broadcast = len(_tensor_shape) - 1 in _broadcast_axis
                _last_dim_is_align = isinstance(_tensor_shape[-1], int) and \
                    _tensor_shape[-1] % get_block_size(self.graph_info.min_type) == 0

                if _mode == REDUCE_TRANS_MODE and not _last_dim_is_align:
                    if _broadcast_num == 1:
                        __current_coexist_node += NO_ALIGN_BROADCAST_EXTRA_NODES
                    elif _broadcast_num > 1:
                        __current_coexist_node += 1

                    return __current_coexist_node

                if _mode == WORKSPACE_MODE:
                    if _broadcast_num > 1:
                        __current_coexist_node += 1

                    return __current_coexist_node

                _is_enable_vnchwconv_fp16 = _tensor.dtype == "float16" and _is_last_broadcast
                _is_enable_vnchwconv_fp32 = \
                    _tensor.dtype == "float32" and _is_last_broadcast and _broadcast_num == 1
                if _is_enable_vnchwconv_fp16:
                    _is_enable_eight_dup = self.is_discontinuous_reduce_axis and \
                        get_context().get_current_compute().get("_mode") == "const"
                    if not _is_enable_eight_dup:
                        __current_coexist_node += 1
                        self.broadcast_temp_size = max(VNCHWCONV_TEMP_SPACE_FP16 + BLOCK_SIZE_BYTE,
                                                       self.broadcast_temp_size)
                elif _is_enable_vnchwconv_fp32:
                    self.broadcast_temp_size = max(VNCHWCONV_TEMP_SPACE_FP32 + BLOCK_SIZE_BYTE,
                                                   self.broadcast_temp_size)
                elif _broadcast_num > 1:
                    __current_coexist_node += 1

                return __current_coexist_node

            def __calc_dependent_coexist_node(_ori_dependent_map, _after_refresh_dependent_map):
                _tensor_insn = util.get_dsl_insn(_tensor)
                _insn_can_not_reuse = _tensor_insn in DST_SRC_NO_REUSE_SET or \
                    _tensor_insn in BROADCAST_INSNS or _tensor_insn in REDUCE_INSNS
                # if src are all in after refresh dependent map, dst can not reuse src
                _dst_and_src_independent = \
                    set(_tensor.op.input_tensors).issubset(set(_after_refresh_dependent_map.keys()))
                # if input tensor has nlast broadcast, dst can not reuse src
                _exist_nlast_broadcast_input = False
                for _input_tensor in _tensor.op.input_tensors:
                    if util.is_unified_broadcast(_input_tensor):
                        if len(_input_tensor.shape) - 1 not in get_broadcast_axis(_input_tensor):
                            _exist_nlast_broadcast_input = True
                _dst_can_not_reuse_src = util.is_placeholder(_tensor) or _insn_can_not_reuse or \
                    _dst_and_src_independent or _exist_nlast_broadcast_input
                # 1. one of the input of the ternary instruction must be reused with the output
                # 2. tensor in dependent_map
                # 3. dst can not reuse src
                if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in _ori_dependent_map or \
                        not _dst_can_not_reuse_src:
                    __current_coexist_node = len(_ori_dependent_map)
                else:
                    __current_coexist_node = len(_ori_dependent_map) + 1

                if _mode == WORKSPACE_MODE:
                    # if depended tensor is last reduce tensor and it does not aggregate,
                    # its tensor size is less than 32B
                    for _depended_tensor in _ori_dependent_map:
                        if util.get_dsl_insn(_depended_tensor) not in REDUCE_INSNS:
                            continue
                        _is_only_one_element_reduce = self.is_continuous_data_move and \
                            _depended_tensor not in self.graph_info.special_after_reduce_tensor_set
                        if _is_only_one_element_reduce:
                            __current_coexist_node -= 1
                            self.temp_ub_size += BLOCK_SIZE_BYTE

                return __current_coexist_node, _dst_can_not_reuse_src

            _ori_dependent_map = copy.deepcopy(dependent_map)
            _refresh_dependent(_tensor, _producers_map)
            _current_coexist_node, _dst_is_no_reuse = __calc_dependent_coexist_node(_ori_dependent_map, dependent_map)

            if util.need_extent_node(_tensor):
                _current_coexist_node += 1

            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE

            if _tensor in self.graph_info.before_reduce_tensor_set:
                _current_coexist_node = __handle_input_and_output(_current_coexist_node, _dst_is_no_reuse)

            if _tensor in self.graph_info.reduce_tensor_set:
                _current_coexist_node = __handle_reduce(_current_coexist_node)

            if util.is_unified_broadcast(_tensor):
                _current_coexist_node = __handle_unified_broadcast(_current_coexist_node)

            for _set_value_tensor, set_value_actions in self.graph_info.tensor_and_set_value_actions_map.items():
                if _set_value_tensor in _tensor.op.input_tensors and len(set_value_actions) > 1:
                    _current_coexist_node = _current_coexist_node + len(set_value_actions)

            return _current_coexist_node

        def _r_coexisting(_tensor, _producers_map, _consumers_map, _mode):
            if _tensor in dependent_map:
                return len(dependent_map)
            _coexist_node_list = []
            for _tensor_i in _producers_map.get(_tensor):
                _coexist_node_list.append(_r_coexisting(_tensor_i, _producers_map, _consumers_map, _mode))

            _current_coexist_node = _calc_current_coexist_node(_tensor, _producers_map, _mode)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_tensor)

            _coexist_node_list.append(_current_coexist_node)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = _consumers_map.get(_tensor).copy()

            return max(_coexist_node_list)

        def _refresh_dependent(_tensor, _producers_map):
            for _tensor_i in _producers_map.get(_tensor):
                if _tensor_i not in dependent_map:
                    continue
                dependent_map.get(_tensor_i).remove(_tensor)
                if not dependent_map.get(_tensor_i):
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
            _op_tag = util.get_dsl_insn(_tensor)
            _support_vector_scalar_insns = ("elewise_binary_add", "elewise_binary_mul")
            if _op_tag in set(SUPPORT_SCALAR_INSNS) - set(_support_vector_scalar_insns):
                return True

            if util.is_v100() and _op_tag in _support_vector_scalar_insns and _tensor.dtype == "int32":
                return True

            return False

        def _refine_workspace_coexisting_quantity(_workspace_tensor=None):
            _extra_node = 0
            _extra_size = 0
            if _workspace_tensor is None:
                _sub_tensor_list = self.graph_info.reduce_tensor_set
            else:
                _sub_tensor_list = \
                    self.graph_info.workspace_tensor_and_sub_graph_map.get(_workspace_tensor).get("sub_tensor_list")

            for _reduce_tensor in self.graph_info.reduce_tensor_set:
                if _reduce_tensor in _sub_tensor_list:
                    _is_only_one_element_reduce = self.is_continuous_data_move and \
                        _reduce_tensor not in self.graph_info.special_after_reduce_tensor_set
                    if _is_only_one_element_reduce:
                        _extra_size += BLOCK_SIZE_BYTE
                    else:
                        _extra_node += 1

            return _extra_node, _extra_size

        def _calc_available_ub_size(_graph_map_tuple, _res_tensor, _mode, _extra_size=0, _extra_node=0):
            _producers_map, _consumers_map = _graph_map_tuple
            for _tensor_i in _producers_map.get(_res_tensor):
                coexisting_quantities.append(_r_coexisting(_tensor_i, _producers_map, _consumers_map, _mode))
            if not _res_tensor.op.tag == FAKE_NODE_TAG:
                _local_current_coexist_node = _calc_current_coexist_node(_res_tensor, _producers_map, _mode)
                # correct ub size in vcmp or vsel or vcmpsel
                _correct_ub_size_by_cmp_sel(_res_tensor)
                # correct ub size in reduce
                _correct_ub_size_by_reduce(_res_tensor)
                coexisting_quantities.append(_local_current_coexist_node)

            _tensor_space = (self.soc_ub_size - self.temp_ub_size - self.broadcast_temp_size - _extra_size) //\
                (max(coexisting_quantities) + _extra_node)
            _ub_size = _tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // self.max_bytes
            _db_tensor_space = \
                (self.soc_ub_size // 2 - self.temp_ub_size - self.broadcast_temp_size - _extra_size) // \
                (max(coexisting_quantities) + _extra_node)
            _db_ub_size = _db_tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // self.max_bytes

            return _ub_size, _db_ub_size

        def _init_before_calculate():
            _coexisting_quantities = []
            _dependent_map = {}
            self.temp_ub_size = 0
            self.broadcast_temp_size = 0

            return _coexisting_quantities, _dependent_map

        _out = self.graph_info.endpoint_output_tensor

        # common sch
        coexisting_quantities, dependent_map = _init_before_calculate()
        self.available_ub_size, self.const_common_db_size = _calc_available_ub_size(
            (self.graph_info.tensor_producers_map, self.graph_info.tensor_consumers_map), _out, COMMON_MODE
        )

        # align and remove pad sch
        coexisting_quantities, dependent_map = _init_before_calculate()
        self.pad_available_ub_size, _ = _calc_available_ub_size(
            (self.graph_info.tensor_producers_map, self.graph_info.tensor_consumers_map), _out, PAD_MODE,
            _extra_size = ALIGN_AND_REMOVE_PAD_TEMP_SPACE
        )

        # reduce transpose sch
        coexisting_quantities, dependent_map = _init_before_calculate()
        self.reduce_transpose_available_ub_size, _ = _calc_available_ub_size(
            (self.graph_info.tensor_producers_map, self.graph_info.tensor_consumers_map), _out, REDUCE_TRANS_MODE
        )

        # workspace sch
        is_sub_graph_dependent = not self.graph_info.special_after_reduce_tensor_set and \
            not (self.graph_info.workspace_tensor_set & self.graph_info.after_reduce_tensor_set) and \
            self.is_reduce_last_axis and len(self.reduce_axis_indices) == 1

        if not is_sub_graph_dependent:
            sub_graph_available_ub_size_list, sub_graph_available_db_ub_size_list = [], []
            # calculate the number of coexisting quantities and available ub size of sub graph
            for workspace_tensor, sub_graph_map in self.graph_info.workspace_tensor_and_sub_graph_map.items():
                coexisting_quantities, dependent_map = _init_before_calculate()
                extra_node, extra_size = _refine_workspace_coexisting_quantity(_workspace_tensor=workspace_tensor)
                cur_available_ub_size, cur_available_db_ub_size = _calc_available_ub_size(
                    (sub_graph_map.get("sub_tensor_producers_map"), sub_graph_map.get("sub_tensor_consumers_map")),
                    workspace_tensor, WORKSPACE_MODE, _extra_node=extra_node, _extra_size=extra_size
                )
                sub_graph_available_ub_size_list.append(cur_available_ub_size)
                sub_graph_available_db_ub_size_list.append(cur_available_db_ub_size)
            self.workspace_available_ub_size = min(sub_graph_available_ub_size_list)
            self.const_workspace_db_size = min(sub_graph_available_db_ub_size_list)
        else:
            coexisting_quantities, dependent_map = _init_before_calculate()
            extra_node, extra_size = _refine_workspace_coexisting_quantity()
            self.workspace_available_ub_size, self.const_workspace_db_size = _calc_available_ub_size(
                (self.graph_info.tensor_producers_map, self.graph_info.tensor_consumers_map), _out, WORKSPACE_MODE,
                _extra_node=extra_node, _extra_size=extra_size
            )

        get_context().get_current_compute().add("_available_ub_size_list", [self.available_ub_size,
                                                                            self.workspace_available_ub_size,
                                                                            self.pad_available_ub_size,
                                                                            self.reduce_transpose_available_ub_size])


class NormTilingCase:
    """
    class for norm tiling case
    """
    def __init__(self):
        self.block_split_axis_index = None
        self.block_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core = None
        self.tiling_key = 2 ** 31 - 1
        self.is_partial_reorder_case = False
        self.is_aligned_in_ub_case = False
        self.is_last_reduce_align_case = False
        self.is_enable_db = False
        self.is_reduce_transpose_case = False
