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
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import decode
from tbe.common.utils import do_op_tiling
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.base.operation import register_build_pointcut
from tbe.tvm.expr import IntImm
from tbe.tvm.expr import Var
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor

from ... import util
from ...computation import Computation
from ...constants import CompileInfo
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import NormPattern
from ...constants import Pattern
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import TERNARY_INSNS

BLOCK = 16
BLOCK_SIZE_BYTE = 32
FAKE_WORKSPACE_SIZE = 32

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4

# temp space for last axis broadcast use vnchwconv
VNCHWCONV_TEMP_SPACE = 1024
# number extra nodes that enable align and remove pad
ALIGN_AND_REMOVE_PAD_EXTRA_NODES = 3

DTYPE_AND_BLOCK_SIZE_MAP = {
    "bool": 32,
    "int8": 32,
    "uint8": 32,
    "float16": 16,
    "float32": 8,
    "int32": 8,
    "int64": 4
}

DTYPE_AND_PAD_ENTIRE_SIZE_MAP = {
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

        # construct information of graph and add them to ComputeContext
        norm_compute_graph_info = NormComputeGraphInfo(outs)
        norm_info = NormInfo(norm_compute_graph_info)
        current_compute.add("_compute_graph_info", norm_compute_graph_info)
        current_compute.add("_norm_info", norm_info)
        if _judge_current_compute_is_const():
            current_compute.add("_mode", "const")

        tiling_case_list = []
        tiling_case_list += _add_tiling_case(norm_compute_graph_info, norm_info)
        _apply_compile_info(norm_compute_graph_info)
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
            if not isinstance(single_dim, int):
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


def get_align_and_remove_pad_entire_size(dtype):
    """
    get pad entire size according to dtype.
    """
    if dtype not in DTYPE_AND_PAD_ENTIRE_SIZE_MAP:
        raise_error("[%s] is not support type in norm" % dtype)

    return DTYPE_AND_PAD_ENTIRE_SIZE_MAP.get(dtype)


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
        if not util.expr_equal(src_shape[idx], dst_shape[idx]) or\
                (util.expr_equal(src_shape[idx], 1) and util.expr_equal(dst_shape[idx], 1)):
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


def _apply_compile_info(graph_info):
    # common info
    core_num = get_soc_spec("CORE_NUM")
    min_block_size = get_block_size(graph_info.min_type)
    common_info = [core_num, min_block_size, graph_info.pad_max_entire_size]
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
    current_available_ub_size_list = current_compute.get("_available_ub_size_list")
    current_norm_pattern = current_compute.get("_norm_pattern")
    pattern_and_available_size_map = get_compile_info().get("_available_ub_size")
    if pattern_and_available_size_map is None:
        pattern_and_available_size_map = {}
        add_compile_info_inner("_available_ub_size", pattern_and_available_size_map)
    pattern_and_available_size_map[current_norm_pattern] = current_available_ub_size_list


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

    first_dim_is_one = util.expr_equal(pad_shape[0], 1)
    ub_factor_is_one = isinstance(ub_factor, int) and ub_factor == 1
    const_one_count = int(first_dim_is_one or ub_factor_is_one)
    for index in range(1, len(pad_shape)):
        if util.expr_equal(pad_shape[index], 1):
            const_one_count += 1
    # num of not_1 axes in ub must greater than or equal to 2
    if len(pad_shape) - const_one_count < 2:
        return False

    if isinstance(pad_shape[-1], int):
        is_last_dim_align = pad_shape[-1] % block_size == 0
        if is_last_dim_align:
            return False

        last_dim_align_size = (pad_shape[-1] + block_size - 1) // block_size * block_size
        threshold_dim_size = ub_size // entire_size
        if last_dim_align_size > threshold_dim_size:
            return False

    return True


def _enable_partial_reorder(is_multi_reduce, shape_before_reduce, block_size):
    if not is_multi_reduce:
        return False
    if isinstance(shape_before_reduce[-1], int) and shape_before_reduce[-1] >= block_size:
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
                _tiling_case.multi_core = True if _block_split_axis == 0 else False
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
        if reduce_product <= norm_graph_info.available_ub_size:
            _tiling_case = NormTilingCase()
            _tiling_case.multi_core = False
            tiling_case_list.append(_tiling_case)

    def __add_aligned_in_ub_tiling_case():
        if reduce_product > norm_graph_info.pad_available_ub_size:
            return
        all_reduce_normal_case = norm_info.is_all_reduce
        _tiling_case = NormTilingCase()
        _tiling_case.block_split_axis_index = None if all_reduce_normal_case else 0
        _tiling_case.ub_split_axis_index = None if all_reduce_normal_case else 0
        _tiling_case.multi_core = False if all_reduce_normal_case else True
        _tiling_case.is_aligned_in_ub_case = True
        tiling_case_list.append(_tiling_case)

    shape_before_reduce = util.shape_to_list(norm_info.shape_before_reduce)
    reduce_axis_index = norm_info.reduce_axis_indices
    block_size = get_block_size(norm_graph_info.min_type)
    tiling_case_list = []

    reduce_product = 1
    for reduce_idx in reduce_axis_index:
        reduce_dim = shape_before_reduce[reduce_idx]
        if isinstance(reduce_dim, int):
            reduce_product *= reduce_dim

    for block_split_axis in range(len(shape_before_reduce)):
        # block split on A axis
        if block_split_axis in reduce_axis_index:
            continue
        for ub_split_axis in range(len(shape_before_reduce)):
            # ub tiling axis is after block tiling axis if ub tiling split normal axis
            illegal_normal_case = ub_split_axis not in reduce_axis_index and \
                (ub_split_axis < block_split_axis or reduce_product > norm_graph_info.available_ub_size)
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
                                    norm_graph_info.pad_available_ub_size,
                                    norm_graph_info.pad_max_entire_size):
        __add_aligned_in_ub_tiling_case()

    return tiling_case_list


def _gen_const_tiling_case(norm_info, graph_info):
    def __construct_inputs_and_outputs():
        # the order of ops inputs and sch inputs may not be consistent
        ops_input_shapes = current_compute.get("_input_shapes")
        for single_shape in ops_input_shapes:
            for single_tensor in graph_info.input_tensor_set:
                if util.shape_to_list(single_tensor.shape) == list(single_shape):
                    inputs.append({"shape": single_shape, "dtype": single_tensor.dtype})
                    break

        for single_tensor in graph_info.real_output_tensor_set:
            shape = util.shape_to_list(single_tensor.shape)
            outputs.append({"shape": shape, "dtype": single_tensor.dtype})

    def __select_tiling_format():
        if norm_info.is_all_reduce:
            # all reduce but reduce product can be put in ub, normal sch
            if before_reduce_product <= graph_info.available_ub_size:
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
            workspace_bytes_list.append(DTYPE_BYTE_MAPPING[workspace_tensor.dtype])

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
        return

    def __save_temp_disable_fuse_axes_info():
        disable_fuse_axes = []
        if "_disable_fuse_axes" in get_compile_info():
            disable_fuse_axes = get_compile_info()["_disable_fuse_axes"]
            get_compile_info()["_disable_fuse_axes"] = []

        return disable_fuse_axes

    def __rollback_disable_fuse_axes(disable_fuse_axes):
        if "_disable_fuse_axes" in get_compile_info():
            get_compile_info()["_disable_fuse_axes"] = disable_fuse_axes

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
    exist_workspace_after_reduce = True if const_workspace_tensors & graph_info.after_reduce_tensor_set else False
    add_compile_info_inner("_exist_workspace_after_reduce", exist_workspace_after_reduce)

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

    # invoke op_tiling interface and decode
    disable_fuse_axes = __save_temp_disable_fuse_axes_info()
    run_info = do_op_tiling("AutoTiling", get_compile_info(), inputs, outputs)
    __rollback_disable_fuse_axes(disable_fuse_axes)

    tiling_format = __select_tiling_format()
    tiling_data = decode(run_info["tiling_data"], tiling_format)

    const_tiling_case.block_split_axis_index = tiling_data.get("block_axis")
    const_tiling_case.block_factor = tiling_data.get("block_factor")
    const_tiling_case.ub_split_axis_index = tiling_data.get("ub_axis")
    const_tiling_case.ub_factor = tiling_data.get("ub_factor")
    const_tiling_case.multi_core = True if run_info.get("block_dim") > 1 else False
    # cases that const can enable align and remove pad sch:
    # 1. AR pattern
    # 1. last axis is not align
    # 2. last axis * entire_size is not larger than ub tensor size
    # 3. last two dims don't have 1
    # 4. ub_factor is not equal to 1
    const_tiling_case.is_aligned_in_ub_case = _enable_align_and_remove_pad(norm_info.is_continuous_data_move,
                                                                           before_reduce_shape,
                                                                           reduce_axis_index, block_size,
                                                                           graph_info.pad_available_ub_size,
                                                                           graph_info.pad_max_entire_size,
                                                                           const_tiling_case.ub_factor)
    const_tiling_case.tiling_key = current_compute.get("_norm_pattern")

    # the flag of invoking op_tiling interface during running
    add_compile_info_inner("_const_shape_post", True)
    if ori_reduce_axis is not None:
        add_compile_info_inner("_ori_reduce_axis", ori_reduce_axis)
    if ori_broadcast_axis is not None:
        add_compile_info_inner("_ori_broadcast_axis", ori_broadcast_axis)
    block_dims = get_compile_info().get("_const_block_dims")
    if block_dims is None:
        block_dims = {}
        add_compile_info_inner("_const_block_dims", block_dims)
    block_dims[const_tiling_case.tiling_key] = run_info["block_dim"]

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
    if tiling_case.is_partial_reorder_case:
        sch_type = 1
    elif tiling_case.is_aligned_in_ub_case:
        sch_type = 2

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
        self.soc_ub_size = get_soc_spec("UB_SIZE")
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
        self.workspace_info_map: Dict[Tensor, Dict] = {}
        self.workspace_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # extra split tensor and its sub_graph
        self.split_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # special tensors
        self.broadcast_fork_tensor_set: Set[Tensor] = set()
        self.reduce_fork_tensor_set: Set[Tensor] = set()
        self.after_mid_out_and_before_broadcast_tensor_set: Set[Tensor] = set()
        self.before_mid_out_and_after_reduce_tensor_set: Set[Tensor] = set()
        # before_reduce/after_reduce/other tensor set
        self.before_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.after_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.other_tensor_set: Optional[Set[Tensor]] = set()
        # max type and min type in graph
        self.max_type: Optional[str] = None
        self.min_type: Optional[str] = None
        # ub size
        self.temp_ub_size = 0
        self.available_ub_size = 0
        self.workspace_available_ub_size = 0
        self.pad_available_ub_size = 0
        self.pad_max_entire_size = 0
        # do info collection
        self.collect_info(output_tensors)
        self.fake_node()
        self.get_tensors_before_and_after_reduce()
        self.get_tensors_in_fork()
        self.get_mid_out_special_tensors()
        self.find_workspace_tensor()
        self.calc_coexisting_quantities_and_temp_buffer_size()

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
            if not self.tensor_consumers_map[output_tensor]:
                self.endpoint_output_tensor = output_tensor
                break

    def gen_mid_tensor_sets(self):
        """
        get mid tensors
        """
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map[tensor]:
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
                if DTYPE_BYTE_MAPPING[tensor_i.dtype] > DTYPE_BYTE_MAPPING[dtype]:
                    dtype = tensor_i.type
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
                _tensor_consumers_map[in_tensor].add(_root_tensor)
                _tensor_producers_map[_root_tensor].add(in_tensor)
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
            if DTYPE_BYTE_MAPPING[item.dtype] > DTYPE_BYTE_MAPPING[self.max_type]:
                self.max_type = item.dtype
            elif DTYPE_BYTE_MAPPING[item.dtype] < DTYPE_BYTE_MAPPING[self.min_type]:
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

            for single_tensor in visited_set:
                self.broadcast_fork_tensor_set.add(single_tensor)
        # output reduce fork tensor
        for out_tensor in self.real_pure_output_tensor_set & self.after_reduce_tensor_set:
            visited_set = set()
            _traverse_tensor_set(out_tensor, visited_set, self.tensor_producers_map, self.reduce_tensor_set)
            for single_tensor in visited_set:
                self.reduce_fork_tensor_set.add(single_tensor)

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
                for single_tensor in after_visited_set - self.broadcast_tensor_set:
                    self.after_mid_out_and_before_broadcast_tensor_set.add(single_tensor)
                # before mid output tensor and after reduce
                before_visited_set = set()
                _traverse_tensor_set(mid_output, before_visited_set, self.tensor_producers_map,
                                     self.reduce_tensor_set)
                for single_tensor in before_visited_set:
                    self.before_mid_out_and_after_reduce_tensor_set.add(single_tensor)

    def find_workspace_tensor(self):
        """
        find workspace tensor
        """
        def _find_possible_cross_hierarchy_tensor(_idx, _current_tensor):
            # stop when current tensor is end tensor or cross hierarchy tensor
            if _current_tensor == self.endpoint_output_tensor or \
                    _current_tensor in cross_hierarchy_tensor_set:
                path_index_and_tensor_map[_idx].add(_current_tensor)
                return
            # _idx represents the index of path
            if _current_tensor in self.reduce_tensor_set:
                path_index_and_tensor_map[_idx].add(_current_tensor)
            for _consumer_tensor in self.tensor_consumers_map[_current_tensor]:
                # Reduce nodes exist on this path
                if _consumer_tensor in self.reduce_tensor_set:
                    path_index_and_tensor_map[_idx].add(_consumer_tensor)
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

        def _update_cache_clone_list(visited_tensor):
            for tensor in visited_tensor:
                if tensor not in self.cache_clone_tensor_list:
                    self.cache_clone_tensor_list.append(tensor)

        cross_hierarchy_tensor_set = set()
        tensors_and_num_path_map = {}
        # find the cross hierarchy tensors
        for single_tensor in self.tensor_producers_map:
            if len(self.tensor_consumers_map[single_tensor]) > 1:
                path_index_and_tensor_map = {}
                for idx, consumer_tensor in enumerate(self.tensor_consumers_map[single_tensor]):
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
                tensors_and_num_path_map[single_tensor] if single_tensor in tensors_and_num_path_map else 1

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
        last_sub_tensor_set = set(self.split_tensor_and_sub_graph_map[self.endpoint_output_tensor]["sub_tensor_list"])
        if not (self.real_pure_output_tensor_set & last_sub_tensor_set == self.real_pure_output_tensor_set):
            raise_error("norm schedule does not support output fork now")

    def calc_coexisting_quantities_and_temp_buffer_size(self):
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
                if all_axes[-1] not in reduce_axes:
                    self.temp_ub_size += 64
                else:
                    if len(reduce_axes) == 1:
                        self.temp_ub_size += 4096
                    else:
                        self.temp_ub_size += 2048

            def _reduce_max_space():
                if dtype in ["float32", "int32"]:
                    if all_axes[-1] not in reduce_axes:
                        self.temp_ub_size += 64
                    else:
                        if len(reduce_axes) == 1:
                            self.temp_ub_size += 512
                        else:
                            self.temp_ub_size += 512
                elif dtype in ["float16", ]:
                    if all_axes[-1] not in reduce_axes:
                        self.temp_ub_size += 64
                    else:
                        self.temp_ub_size += 4096
                else:
                    raise RuntimeError("Not support dtype in reduce_max(min) is %s" % dtype)

            def _reduce_prod_space():
                if all_axes[-1] not in reduce_axes:
                    self.temp_ub_size += 64
                else:
                    if len(reduce_axes) == 1:
                        self.temp_ub_size += 512
                    else:
                        self.temp_ub_size += 512

            if tag.find("reduce") != -1:
                reduce_axes = util.get_reduce_axes(_tensor)
                all_axes = util.get_reduce_all_axes(_tensor)
                if all_axes[-1] not in reduce_axes:
                    self.temp_ub_size += 256
                if tag in ["reduce_sum", ]:
                    _reduce_sum_space()
                elif tag in ["reduce_max", "reduce_min"]:
                    _reduce_max_space()
                elif tag in ["reduce_prod", ]:
                    _reduce_prod_space()
                else:
                    raise RuntimeError("Unknown reduce_insn is %s" % tag)

        def _calc_current_space(_tensor, _is_workspace=False, _is_pad=False):
            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in dependent_map:
                _current_space = len(dependent_map)
            else:
                _current_space = len(dependent_map) + 1
            if util.need_extent_node(_tensor):
                _current_space += 1
            if _is_pad and _tensor in self.before_reduce_tensor_set:
                if _tensor in self.input_tensor_set:
                    _current_space += ALIGN_AND_REMOVE_PAD_EXTRA_NODES
                if _tensor in self.real_output_tensor_set:
                    # the space that remove pad buffer need
                    _pad_space = \
                        _current_space + ALIGN_AND_REMOVE_PAD_EXTRA_NODES - len(self.tensor_producers_map[_tensor])
                    _current_space = max(_current_space, _pad_space)
            # num of broadcast axis > 1 or float16 and last broadcast(normal sch or workspace sch but not AR)
            if util.is_unified_broadcast(_tensor):
                broadcast_axis = get_broadcast_axis(_tensor)
                broadcast_num = len(broadcast_axis)
                is_last_broadcast = len(_tensor.shape) - 1 in broadcast_axis
                is_enable_vnchwconv = _tensor.dtype == "float16" and is_last_broadcast and \
                    (not _is_workspace or (_is_workspace and not broadcast_num == 1))
                if is_enable_vnchwconv:
                    _current_space += 1
                    self.temp_ub_size += VNCHWCONV_TEMP_SPACE
                elif broadcast_num > 1:
                    _current_space += 1

            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE

            return _current_space

        def _r_coexisting(_tensor, _producer_map, _consumer_map, _is_workspace=False, _is_pad=False):
            if _tensor in dependent_map:
                return len(dependent_map)
            _need_space = []
            for _tensor_i in _producer_map[_tensor]:
                _need_space.append(_r_coexisting(_tensor_i, _producer_map, _consumer_map, _is_workspace, _is_pad))

            _current_space = _calc_current_space(_tensor, _is_workspace=_is_workspace, _is_pad=_is_pad)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_tensor)

            _need_space.append(_current_space)
            _refresh_dependent(_tensor, _producer_map)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = _consumer_map[_tensor].copy()

            return max(_need_space)

        def _refresh_dependent(_tensor, _producer_map):
            for _tensor_i in _producer_map[_tensor]:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
            op_tag = util.get_dsl_insn(_tensor)
            support_vector_scalar_insns = ("elewise_binary_add", "elewise_binary_mul")
            if op_tag in set(SUPPORT_SCALAR_INSNS) - set(support_vector_scalar_insns):
                return True

            if util.is_v100() and op_tag in support_vector_scalar_insns and _tensor.dtype == "int32":
                return True

        def _refine_coexisting_quantity(_coexisting_quantity, _workspace_tensor):
            sub_tensor_list = self.workspace_tensor_and_sub_graph_map.get(_workspace_tensor).get("sub_tensor_list")
            for reduce_tensor in self.reduce_tensor_set:
                if reduce_tensor in sub_tensor_list:
                    _coexisting_quantity += 1

            return _coexisting_quantity

        _out = self.endpoint_output_tensor
        # common sch
        coexisting_quantities = []
        dependent_map = {}
        for tensor_i in self.tensor_producers_map[_out]:
            coexisting_quantities.append(_r_coexisting(tensor_i, self.tensor_producers_map,
                                                       self.tensor_consumers_map))
        if not _out.op.tag == FAKE_NODE_TAG:
            _local_current_space = _calc_current_space(_out)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_space)
        tensor_space = (self.soc_ub_size - self.temp_ub_size) // max(coexisting_quantities)
        self.available_ub_size =\
            tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self.max_type]

        # align and remove pad sch
        coexisting_quantities = []
        dependent_map = {}
        self.temp_ub_size = 0
        for tensor_i in self.tensor_producers_map[_out]:
            coexisting_quantities.append(_r_coexisting(tensor_i, self.tensor_producers_map, self.tensor_consumers_map,
                                                       _is_pad=True))
        if not _out.op.tag == FAKE_NODE_TAG:
            _local_current_space = _calc_current_space(_out, _is_pad=True)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_space)
        tensor_space = (self.soc_ub_size - self.temp_ub_size) // max(coexisting_quantities)
        self.pad_available_ub_size =\
            tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self.max_type]
        self.pad_max_entire_size = max(get_align_and_remove_pad_entire_size(self.min_type) *
                                       DTYPE_BYTE_MAPPING[self.min_type] // DTYPE_BYTE_MAPPING[self.max_type],
                                       get_align_and_remove_pad_entire_size(self.max_type))

        # workspace sch
        # calculate the number of coexisting quantities and available ub size of sub graph
        for workspace_tensor in self.workspace_tensor_and_sub_graph_map:
            sub_tensor_producers_map = \
                self.workspace_tensor_and_sub_graph_map[workspace_tensor]["sub_tensor_producers_map"]
            sub_tensor_consumers_map = \
                self.workspace_tensor_and_sub_graph_map[workspace_tensor]["sub_tensor_consumers_map"]
            coexisting_quantities = []
            dependent_map = {}
            self.temp_ub_size = 0
            for tensor_i in sub_tensor_producers_map[workspace_tensor]:
                coexisting_quantities.append(_r_coexisting(tensor_i, sub_tensor_producers_map,
                                                           sub_tensor_consumers_map, _is_workspace=True))
            _local_current_space = _calc_current_space(workspace_tensor, _is_workspace=True)
            _correct_ub_size_by_cmp_sel(workspace_tensor)
            _correct_ub_size_by_reduce(workspace_tensor)
            coexisting_quantities.append(_local_current_space)
            coexisting_quantity = _refine_coexisting_quantity(max(coexisting_quantities), workspace_tensor)
            if coexisting_quantity == 1:
                self.temp_ub_size += BLOCK_SIZE_BYTE
            self.workspace_info_map[workspace_tensor] = {
                "temp_ub_size": self.temp_ub_size,
                "coexisting_quantity": coexisting_quantity
            }
        sub_graph_available_ub_size_list = []
        for tensor in self.workspace_info_map:
            local_ub_size = self.soc_ub_size - self.workspace_info_map[tensor]["temp_ub_size"]
            tensor_space = local_ub_size // self.workspace_info_map[tensor]["coexisting_quantity"]
            sub_graph_available_ub_size_list.append(tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE)
        self.workspace_available_ub_size = \
            min(sub_graph_available_ub_size_list) // DTYPE_BYTE_MAPPING[self.max_type]

        get_context().get_current_compute().add("_available_ub_size_list", [self.available_ub_size,
                                                                            self.workspace_available_ub_size,
                                                                            self.pad_available_ub_size])


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
        # RAR/ARAR/...
        self.is_discontinuous_reduce_axis = self._judge_discontinuous_reduce_axis()
        # AR/ARR/ARRR... or RR/RRR...
        self.is_continuous_data_move = self.is_reduce_last_axis and not self.is_discontinuous_reduce_axis

    def _judge_non_reduce(self):
        for reduce_axis in self.reduce_axis_indices:
            if not (util.expr_equal(self.shape_before_reduce[reduce_axis], 1)):
                return False

        return True

    def _judge_discontinuous_reduce_axis(self):
        if len(self.reduce_axis_indices) <= 1:
            return False
        discontinuous_axis_num = 1
        for i in range(1, len(self.reduce_axis_indices)):
            if self.reduce_axis_indices[i] == self.reduce_axis_indices[i - 1] + 1:
                continue
            else:
                discontinuous_axis_num += 1

        return discontinuous_axis_num >= 2


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
