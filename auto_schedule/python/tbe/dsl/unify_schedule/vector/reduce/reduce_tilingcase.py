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
Reduce Schedule Remake stage 1 - Tilingcase
"""

# Standard Package
from enum import Enum
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Union

from tbe import tvm
from tbe.common.platform import ASCEND_910
from tbe.common.platform import ASCEND_910B
from tbe.common.platform import SHORT_SOC_VERSION
from tbe.common.platform.platform_info import get_soc_spec
from tbe.common.utils import op_tiling
from tbe.dsl.base import operation
from tbe.dsl.base.operation import add_compile_info_inner
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import get_op_context
from tbe.dsl.base.operation import register_build_pointcut
from tbe.tvm.expr import IntImm
from tbe.tvm.expr import Var
from tbe.tvm.tensor import Tensor

from ...constants import ReduceSchType
from ...constants import CompileInfo
from ...constants import Pattern
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import ReducePattern
from ...constants import AtomicSupportMap910
from ...constants import AtomicSupportMap920A
from ...computation import Computation
from ...util import get_reduce_all_axes
from ...util import get_reduce_axes
from ...util import get_reduce_axis_indexes
from ...util import is_reduce_tensor
from ...util import shape_to_list
from .vector_info import ComputeGraphInfo
from .vector_tilingcase import TilingCaseBase
from .....common.utils.errormgr import get_error_message
from ... import util

CONST = "const"
ZERO = "zero"
DEFAULT = "default"
OP_TYPE_AUTO_TILING = "AutoTiling"
TILINGKEY_NONE_REDUCE_AXIS = 2 ** 31 - 2
REDUCE_CASE_LENGTH_TWO = 2

BLKOUTER = "block_outer"
BLKINNER = "block_inner"
UBOUTER = "ub_outer"
UBINNER = "ub_inner"
A = "A"
R = "R"
REDUCE_COMPUTE = {
    "reduce_min", "reduce_max", "reduce_sum",
    "reduce_prod", "tuple_reduce_sum",
}

REDUCE_EMITSN_SUPPORT_TRANSPOSE = {
    "reduce_sum", "reduce_min", "reduce_max"
}

DTYPE_AND_BLOCK_SIZE_MAP = {
    "bool": 32,
    "int8": 32,
    "uint8": 32,
    "float16": 16,
    "float32": 8,
    "int32": 8,
    "int64": 4
}

REDUCE_SUPPORT_TRANSPOSE = {
    # AR
    2: {"0_0"},
}

REDUCE_SUPPORT_PAD = {
    # AR
    2: {"0_0": False},

    # ARA
    3: {"0_0": True, "0_1": True, "1_1": False, "1_0": False}
}


class CalcReduceTilingCase(Computation):
    """
    Calculate Reduce TilingCase
    """

    def __init__(self, outs, option):
        self.outs = outs
        self.option = option

    def get_sub_pattern(self):
        return ReducePattern.R_0

    @classmethod
    def get_instance(cls, outs, option):
        return cls(outs, option)

    @classmethod
    def get_supported_pattern(cls):
        return [Pattern.REDUCE]

    @classmethod
    def get_supported_soc(cls):
        return [DEFAULT]

    def do_tiling_case(self):
        """
        do tiling case
        """
        # get result of tiling case
        return self.calc_tiling_case()

    def calc_tiling_case(self):
        """
        get tiling case of different situation
        """
        outs = list(self.outs) if isinstance(self.outs, (list, tuple)) else [self.outs]
        # to check reduce not support outputs with different shapes
        _current_support_check(outs)
        current_compute = get_context().get_current_compute()

        # construct information of graph
        compute_graph_info = ComputeGraphInfo(outs)
        single_reduce_info = SingleReduceInfo(compute_graph_info)
        apply_common_compile_info(compute_graph_info, single_reduce_info)
        current_compute.add("_compute_graph_info", compute_graph_info)
        current_compute.add("_single_reduce_info", single_reduce_info)
        if not compute_graph_info.reduce_tensor_set:
            raise RuntimeError("Couldn't find reduce node for ReduceSchedule")

        # Different Cases
        tiling_case_list: List[ReduceTilingCase] = []
        if current_compute.get("_mode") == CONST:
            def _calc_const_case(const_tiling_case_list):
                # Get all possible ub_info
                possible_case_list = []
                possible_case_list += _calculate_tiling_cases(single_reduce_info)
                possible_case_list += _calculate_atomic_tiling_cases(single_reduce_info)
                possible_case_list += _calculate_group_reduce_tiling_cases(single_reduce_info)
                for _case in possible_case_list:
                    if _case.is_reduce_transpose_case:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case,
                                                              ReduceSchType.TRANSPOSE)
                    elif _case.is_reduce_pad_case:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case,
                                                              ReduceSchType.PAD)
                    else:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case)
                apply_dyn_compile_info(single_reduce_info, possible_case_list, model=CONST)
                # Get Real TilingCase
                cst_case = ReduceTilingCase()
                _calc_const_tiling_case(single_reduce_info, compute_graph_info, cst_case)
                const_tiling_case_list.append(cst_case)

            _calc_const_case(tiling_case_list)
        elif current_compute.get("_mode") == ZERO:
            # SingleCase
            tiling_case_list += [_gen_zero_tiling_case(), ]
            ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, tiling_case_list[0])
            add_compile_info_inner("_zero_ub_factor", tiling_case_list[0].tensor_ub_size_after_reduce)
        else:
            def _calc_dynamic_case(dynamic_tiling_case_list):
                dynamic_tiling_case_list += _calculate_tiling_cases(single_reduce_info)
                dynamic_tiling_case_list += _calculate_atomic_tiling_cases(single_reduce_info)
                dynamic_tiling_case_list += _calculate_group_reduce_tiling_cases(single_reduce_info)
                for tiling_case in dynamic_tiling_case_list:
                    _calc_tiling_key(single_reduce_info, tiling_case)
                # Empty schedule will always be there in the tiling_case_list
                # noinspection PyProtectedMember
                if "_empty_schedule_flag" not in get_context()._addition:
                    dynamic_tiling_case_list.append(ReduceTilingCase())
                    dynamic_tiling_case_list[-1].type = ReduceTilingCase.Type.EMPTY
                    get_context().add("_empty_schedule_flag", True)

                for _case in dynamic_tiling_case_list:
                    if _case.type == ReduceTilingCase.Type.EMPTY:
                        continue
                    if _case.is_reduce_transpose_case:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case,
                                                              ReduceSchType.TRANSPOSE)
                    elif _case.is_reduce_pad_case:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case,
                                                              ReduceSchType.PAD)
                    else:
                        ComputeGraphInfo.get_maximum_subgraph(compute_graph_info, single_reduce_info, _case)
                apply_dyn_compile_info(single_reduce_info, dynamic_tiling_case_list)

            _calc_dynamic_case(tiling_case_list)

        return tiling_case_list


class SingleReduceInfo:
    """
    data struct for a single reduce info
    """

    def __init__(self, compute_graph_info: ComputeGraphInfo):
        if len(compute_graph_info.reduce_tensor_set) != 1:
            raise RuntimeError("ComputeGraph is not in Single Reduce Pattern: %s" %
                               compute_graph_info.reduce_tensor_set)
        # Assume only one reduce node
        self.reduce_tensor: Tensor = tuple(compute_graph_info.reduce_tensor_set)[0]
        self.all_axes: List[Var] = get_reduce_all_axes(self.reduce_tensor)
        self.reduce_axes: List[Var] = get_reduce_axes(self.reduce_tensor)

        compute = operation.get_context().get_current_compute()
        if compute.get("_mode") == "zero" and compute.get("_shape") == (1, -1, 0):
            self.shape_before_reduce = list(self.reduce_tensor.shape) + [tvm.expr.IntImm("int32", 0)]
        else:
            self.shape_before_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.op.input_tensors[0].shape)
        self.shape_after_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.shape)
        self.reduce_axis_indexes: List[int] = get_reduce_axis_indexes(self.reduce_tensor)
        self.keepdims: bool = len(self.shape_before_reduce) == len(self.shape_after_reduce)
        self.graph_info: ComputeGraphInfo = compute_graph_info

    def is_reduce_not_last_axis(self) -> bool:
        """
        check if reduce not last axis
        """
        compute = operation.get_context().get_current_compute()
        if compute.get("_mode") == "zero":
            return False

        is_not_last_axis = self.all_axes[-1] not in self.reduce_axes
        return is_not_last_axis

    def is_reduce_last_axis(self) -> bool:
        """
        check if reduce last axis
        """
        return self.all_axes[-1] in self.reduce_axes

    def is_reduce_all_axes(self) -> bool:
        """
        check if reduce all axis
        """
        return set(self.all_axes) == set(self.reduce_axes)

    @staticmethod
    def find_last_reduce_axis(shape, reduce_axis_indexes: Iterable[int]):
        """
        find last reduce axis
        """
        # shape_before_reduce:(ak+1,rk,...,r2,a2,r1,a1) or (ak,rk,...,r2,a1,r1)
        # find r1 position, r1 may contain continues axis
        r1_end_index = None
        for i in range(len(shape) - 1, -1, -1):
            if i in reduce_axis_indexes:
                r1_end_index = i
                break
        r1_start_index = r1_end_index
        if r1_end_index is None:
            return r1_start_index, r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_indexes:
                r1_start_index = i + 1
                break
            if i == 0:
                r1_start_index = i

        return r1_start_index, r1_end_index

    @staticmethod
    def find_last_none_reduce_axis(shape_before_reduce: list,
                                   reduce_axis_index: List[int]) -> Tuple[Optional[int], Optional[int]]:
        """
        :param shape_before_reduce
        :param reduce_axis_index
        :return the last axis or the last serials axises that are not in reduce_axis
        """
        # shape_before_reduce:(ak+1,rk,...,r2,a2,r1,a1) or (ak,rk,...,r2,a1,r1)
        # find a1 position, a1 may contain continues axis
        a1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i not in reduce_axis_index:
                a1_end_index = i
                break
        a1_start_index = a1_end_index
        if a1_end_index is None:
            return a1_start_index, a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break
            if i == 0:
                a1_start_index = i

        return a1_start_index, a1_end_index

    @staticmethod
    def find_none_reduce_axis_map(shape_before_reduce: list, reduce_axis_index: List[int]) -> Dict[int, int]:
        """
        find none reduce axis map
        """
        none_reduce_index_map = {}
        count = 0
        for i in range(0, len(shape_before_reduce)):
            if i not in reduce_axis_index:
                none_reduce_index_map[i] = count
                count += 1
        return none_reduce_index_map


class ReduceTilingCase(TilingCaseBase):
    """
    tiling case data struct for reduce
    """

    class Type(Enum):
        """
        Reduce type Enum
        """
        NORMAL_REDUCE = "NORMAL"
        ATOMIC_REDUCE = "ATOMIC"
        GROUP_REDUCE = "GROUP"
        EMPTY = "EMPTY"

    def __init__(self):
        self.type: Optional[ReduceTilingCase.Type] = ReduceTilingCase.Type.NORMAL_REDUCE
        self.block_split_axis_index = None
        self.block_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core: Optional[bool] = None
        self.db = False
        self.tiling_key = 2 ** 31 - 1
        self.tensor_ub_size_before_reduce: Optional[int] = None
        self.tensor_ub_size_after_reduce: Optional[int] = None
        self.is_reduce_pad_case = False
        self.need_remove_pad = False
        self.is_reduce_transpose_case = False

    def __repr__(self):
        segment0 = self.type.value
        segment1 = "ENABLED" if self.multi_core else "DISABLED"
        return "%s REDUCE: (%d, %d) with multicore %s" % (segment0,
                                                          self.block_split_axis_index, self.ub_split_axis_index,
                                                          segment1)

    def __hash__(self):
        return hash((self.type.value, self.block_split_axis_index, self.block_factor,
                     self.ub_split_axis_index, self.ub_factor, self.multi_core))

    def __eq__(self, other) -> bool:
        condition0 = other.type == self.type
        condition1 = other.block_split_axis_index == self.block_split_axis_index
        condition2 = other.block_factor == self.block_factor
        condition3 = other.ub_split_axis_index == self.ub_split_axis_index
        condition4 = other.ub_factor == self.ub_factor
        condition5 = other.multi_core == self.multi_core
        return (isinstance(other, ReduceTilingCase)
                and condition0 and condition1 and condition2 and condition3 and condition4 and condition5)

    def __ne__(self, other) -> bool:
        condition0 = other.type != self.type
        condition1 = other.block_split_axis_index != self.block_split_axis_index
        condition2 = other.block_factor != self.block_factor
        condition3 = other.ub_split_axis_index != self.ub_split_axis_index
        condition4 = other.ub_factor != self.ub_factor
        condition5 = other.multi_core != self.multi_core
        return (not isinstance(other, ReduceTilingCase)
                or condition0 or condition1 or condition2 or condition3 or condition4 or condition5)


class Dim:
    """
    do actions for dim
    """

    def __init__(self, axis_type, idx, var_type=None):
        self.axis_type = axis_type
        self.var_type = var_type
        self.idx = idx

    @staticmethod
    def split(in_shape, split_idx, model=None):
        """
        do dim split
        """
        if model == "UBSplit":
            outer, inner = UBOUTER, UBINNER
        else:
            outer, inner = BLKOUTER, BLKINNER

        # update index
        for item in in_shape[split_idx + 1:]:
            item.idx += 1

        # insert split
        in_shape[split_idx].var_type = outer
        in_shape.insert(split_idx + 1, Dim(in_shape[split_idx].axis_type, split_idx + 1, inner))

    @staticmethod
    def rfactor(in_shape, axis, factor_axis=0):
        """
        get rfactor shape
        """
        temp_shape, a_shape, r_shape = [], [], []
        for item in in_shape:
            if item not in axis:
                temp_shape.append(item)

        for item in reversed(axis):
            item.axis_type = A
            temp_shape.insert(factor_axis, item)

        for item in temp_shape:
            if item.axis_type == A:
                a_shape.append(item)
            else:
                r_shape.append(item)

        return a_shape, r_shape

    @staticmethod
    def group(nums):
        """
        group nums
        """
        nums = sorted(set(nums))
        gaps = [[s, e] for s, e in zip(nums, nums[1:]) if s + 1 < e]
        edges = iter(nums[:1] + sum(gaps, []) + nums[-1:])
        return list(zip(edges, edges))


@register_build_pointcut(pattern=Pattern.REDUCE)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """

    def _find_idx_in_tensor_list():
        if len(args) < 2:
            dict_args = {"errCode": "E90003", "detailed_cause": "Size of args should more than 2"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

        tensor_list = args[1].get("tensor_list")
        _before_reduce = operation.get_context().get("_placeholder_before_reduce")
        _after_reduce = operation.get_context().get("_placeholder_after_reduce")

        if not _before_reduce and not _after_reduce:
            return

        for tensors in tensor_list:
            for _idx, _tensor in enumerate(tensors):
                if _tensor in _before_reduce:
                    operation.add_compile_info_inner("_idx_before_reduce", _idx)
                    return
            for _idx, _tensor in enumerate(tensors):
                if _tensor in _after_reduce:
                    operation.add_compile_info_inner("_idx_before_reduce", _idx)
                    return

        dict_args = {"errCode": "E90003", "detailed_cause": "Can not find placeholder_op"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def _add_fake_workspace(_args):
        schedule_list, _ = _args
        parameter_length_set = set()
        for same_compute_schs in schedule_list:
            for single_sch in same_compute_schs:
                parameter_length_set.add(len(util.get_sch_additional_entry(single_sch, "real_outs")))

        parameter_max_length, parameter_min_length = max(parameter_length_set), min(parameter_length_set)
        if parameter_max_length == parameter_min_length:
            return

        for same_compute_schs in schedule_list:
            for single_sch in same_compute_schs:
                ori_real_outs = util.get_sch_additional_entry(single_sch, "real_outs")
                if len(ori_real_outs) == parameter_max_length:
                    continue

                for index in range(parameter_max_length - parameter_min_length + 1):
                    fake_workspace = tvm.placeholder([], dtype="uint8", name="fake_workspace_" + str(index))
                    ori_real_outs.append(fake_workspace)

                util.add_sch_additional_entry(single_sch, "real_outs", ori_real_outs)
        add_compile_info_inner("_workspace_size", parameter_max_length - parameter_min_length + 1)

        return

    _find_idx_in_tensor_list()
    _add_fake_workspace(args)
    func(*args, **kwargs)


def apply_common_compile_info(graph_info, reduce_info):
    """
    apply common compile info
    """
    # Common_Info: message from ori computation that only attach once
    pre_compile_info = get_compile_info()
    if pre_compile_info:
        reduce_tensor_dtype_byte = DTYPE_BYTE_MAPPING.get(reduce_info.reduce_tensor.dtype)
        if "_common_info" not in pre_compile_info.keys():
            atomic = 0
            if check_atomic_add_support(reduce_info):
                atomic = 1

            core_num = get_soc_spec("CORE_NUM")
            keep_dims = 1
            min_block_size = int(32 // DTYPE_BYTE_MAPPING[graph_info.min_type])
            support_transpose = 0
            if reduce_info.reduce_tensor.op.tag in REDUCE_EMITSN_SUPPORT_TRANSPOSE:
                support_transpose = 1
            support_group_reduce = get_soc_spec(SOC_VERSION) != ASCEND_910B
            common_info = [core_num, keep_dims, min_block_size, atomic, graph_info.coef,
                           graph_info.pad_max_entire_size, support_transpose, reduce_tensor_dtype_byte,
                           support_group_reduce]
            add_compile_info_inner("_common_info", common_info)
        else:
            # update reduce_tensor_dtype_byte
            cur_common_info = pre_compile_info.get("_common_info")
            reduce_tensor_dtype_byte_index = 7
            cur_common_info[reduce_tensor_dtype_byte_index] = max(cur_common_info[reduce_tensor_dtype_byte_index],
                                                                  reduce_tensor_dtype_byte)
    else:
        raise RuntimeError("pre_compile_info is Null")


def apply_dyn_compile_info(reduce_info, tiling_case_list, model="dynamic"):
    """
    apply dynamic compile info
    """

    # dynamic message from each case of tiling_case_list
    def _is_rfactor(_item):
        cond_0 = reduce_info.is_reduce_last_axis()
        cond_1 = reduce_info.all_axes[_case.ub_split_axis_index] in reduce_info.reduce_axes
        cond_2 = _item.type == ReduceTilingCase.Type.NORMAL_REDUCE
        if cond_0 and cond_1 and cond_2:
            return True
        return False

    if model == CONST:
        compile_axis = get_context().get_current_compute().get("_ori_axis")
        pattern_info = _gen_const_tiling_key(compile_axis)
    else:
        pattern_info = _get_pattern_key(reduce_info.shape_before_reduce,
                                        reduce_info.reduce_axis_indexes)

    pre_compile_info = get_compile_info()
    if pre_compile_info:
        # different patterns
        # pattern_info must not be repeated
        if "_pattern_info" not in pre_compile_info.keys():
            current_pattern = [pattern_info, ]
            ub_info = [None, ]
            ub_info_rf = [None, ]
            ub_info_pad = [0, ]
            ub_info_transpose = [0, ]
        else:
            current_pattern = pre_compile_info.get("_pattern_info")
            ub_info = pre_compile_info.get("_ub_info")
            ub_info_rf = pre_compile_info.get("_ub_info_rf")
            ub_info_pad = pre_compile_info.get("_ub_info_pad")
            ub_info_transpose = pre_compile_info.get("_ub_info_transpose")
            current_pattern.append(pattern_info)
            ub_info.append(None)
            ub_info_rf.append(None)
            ub_info_pad.append(0)
            ub_info_transpose.append(0)

        # different tilings in the pattern
        _idx = current_pattern.index(pattern_info)
        for _case in tiling_case_list:
            if _case.type == ReduceTilingCase.Type.EMPTY:
                continue

            max_ub_count = _case.tensor_ub_size_before_reduce
            if _case.is_reduce_pad_case:
                ub_info_pad[_idx] = max_ub_count
            elif _case.is_reduce_transpose_case:
                ub_info_transpose[_idx] = max_ub_count
            else:
                if _is_rfactor(_case):
                    ub_info_rf[_idx] = max_ub_count
                    if not ub_info[_idx]:
                        ub_info[_idx] = max_ub_count
                else:
                    ub_info[_idx] = max_ub_count
                    if not ub_info_rf[_idx]:
                        ub_info_rf[_idx] = max_ub_count

        add_compile_info_inner("_pattern_info", current_pattern)
        add_compile_info_inner("_ub_info_rf", ub_info_rf)
        add_compile_info_inner("_ub_info", ub_info)
        add_compile_info_inner("_ub_info_pad", ub_info_pad)
        add_compile_info_inner("_ub_info_transpose", ub_info_transpose)


def _calc_tiling_key(reduce_info, tiling):
    if get_context().get("_mode") == CONST:
        ori_axis = get_context().get_current_compute().get("_ori_axis")
        tiling_key = _gen_const_tiling_key(ori_axis)
    elif get_context().get("_mode") == ZERO:
        tiling_key = _gen_zero_tiling_key()
    else:
        # tiling: single_case
        shape = reduce_info.shape_before_reduce
        reduce_axis_idx = reduce_info.reduce_axis_indexes
        block_split_axis = tiling.block_split_axis_index
        ub_split_axis = tiling.ub_split_axis_index
        atomic = tiling.type == tiling.Type.ATOMIC_REDUCE
        db = 0
        shape_type = 0
        if tiling.is_reduce_pad_case:
            shape_type = 1
        elif tiling.is_reduce_transpose_case:
            shape_type = 2

        tiling_key = _get_tiling_key(atomic, db, shape_type,
                                     block_split_axis, ub_split_axis,
                                     shape, reduce_axis_idx)

    tiling.tiling_key = tiling_key


def _calc_const_tiling_case(single_reduce_info, compute_graph_info, const_tiling_case):
    add_compile_info_inner("_reduce_shape_known", True)
    shape_before_reduce = shape_to_list(single_reduce_info.shape_before_reduce)
    shape_after_reduce = shape_to_list(single_reduce_info.shape_after_reduce)
    reduce_axis_index = single_reduce_info.reduce_axis_indexes
    input_dtype = tuple(compute_graph_info.input_tensor_set)[0].dtype
    output_dtype = tuple(compute_graph_info.output_tensor_set)[0].dtype
    axes_dtype = get_context().get_current_compute().get("_axis_dtype")
    # staging axis info in ops
    # _ori_axis: original axes from func_enter of ReduceD, maybe None while op is ReduceSum
    # ori_axis: axes from classify, always existed
    ori_reduce_axis = get_compile_info().get("_ori_axis")
    compile_axis = get_context().get_current_compute().get("_ori_axis")
    if axes_dtype is None:
        # invoking op_tiling interface during compilation need axis info in sch
        add_compile_info_inner("_ori_axis", reduce_axis_index)
        inputs = [{"shape": shape_before_reduce, "dtype": input_dtype}]
    else:
        inputs = [{"shape": shape_before_reduce, "dtype": input_dtype},
                  {"shape": (len(reduce_axis_index),), "dtype": axes_dtype, "name": "axes",
                   "const_value": reduce_axis_index}]
    outputs = [{"shape": shape_after_reduce, "dtype": output_dtype}]
    # the flag of invoking op_tiling interface during compilation
    add_compile_info_inner("_const_shape_post", False)
    add_compile_info_inner("_compile_pattern", _gen_const_tiling_key(compile_axis))
    run_info = op_tiling.do_op_tiling(OP_TYPE_AUTO_TILING, get_compile_info(), inputs, outputs)

    _gen_const_tiling_case(single_reduce_info, const_tiling_case, run_info)
    # the flag of invoking op_tiling interface during running
    add_compile_info_inner("_const_shape_post", True)
    # invoking op_tiling interface during running need axis info in ops
    if ori_reduce_axis is not None:
        add_compile_info_inner("_ori_axis", ori_reduce_axis)

    block_dims = get_compile_info().get(CompileInfo.BLOCK_DIMS)
    if block_dims is None:
        block_dims = {}
        add_compile_info_inner(CompileInfo.BLOCK_DIMS, block_dims)
    block_dims[str(const_tiling_case.tiling_key)] = run_info["block_dim"]
    atomic_flags = get_compile_info().get(CompileInfo.ATOMIC_FLAGS)
    if atomic_flags is None:
        atomic_flags = {}
        add_compile_info_inner(CompileInfo.ATOMIC_FLAGS, atomic_flags)
    atomic_flags[str(const_tiling_case.tiling_key)] = run_info["clear_atomic"]

    if const_tiling_case.type == const_tiling_case.Type.GROUP_REDUCE:
        _add_workspace_info_in_json(single_reduce_info, compute_graph_info,
                                    const_tiling_case.block_split_axis_index, run_info.get("block_dim"))


def _gen_const_tiling_case(single_reduce_info, const_tiling_case, run_info):
    shape_before_reduce = shape_to_list(single_reduce_info.shape_before_reduce)
    tiling_format = {"block_axis": "int", "block_factor": "int", "ub_axis": "int", "ub_factor": "int",
                     "tensor_ub_size_before_reduce": "int", "tensor_ub_size_after_reduce": "int",
                     "sch_type": "int", "is_group_reduce": "int"}

    tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
    const_tiling_case.block_split_axis_index = tiling_data["block_axis"]
    const_tiling_case.block_factor = tiling_data["block_factor"]
    const_tiling_case.ub_split_axis_index = tiling_data["ub_axis"]
    const_tiling_case.ub_factor = tiling_data["ub_factor"]
    const_tiling_case.tensor_ub_size_before_reduce = tiling_data["tensor_ub_size_before_reduce"]
    const_tiling_case.tensor_ub_size_after_reduce = tiling_data["tensor_ub_size_after_reduce"]
    if run_info.get("clear_atomic"):
        if tiling_data.get("is_group_reduce"):
            const_tiling_case.type = const_tiling_case.Type.GROUP_REDUCE
        else:
            const_tiling_case.type = const_tiling_case.Type.ATOMIC_REDUCE
    else:
        const_tiling_case.type = const_tiling_case.Type.NORMAL_REDUCE
    const_tiling_case.multi_core = True if run_info["block_dim"] > 1 else False
    const_tiling_case.is_reduce_pad_case = True if tiling_data.get("sch_type") == 1 else False
    const_tiling_case.is_reduce_transpose_case = True if tiling_data.get("sch_type") == 2 else False

    def _calc_const_pad_case():
        if const_tiling_case.is_reduce_pad_case:
            split_axis_info = \
                str(const_tiling_case.block_split_axis_index) + "_" + str(const_tiling_case.ub_split_axis_index)
            support_pattern = REDUCE_SUPPORT_PAD.get(len(shape_before_reduce))
            if support_pattern is not None and split_axis_info in support_pattern:
                const_tiling_case.need_remove_pad = support_pattern.get(split_axis_info)
            else:
                const_tiling_case.is_reduce_pad_case = False
    _calc_const_pad_case()
    _calc_tiling_key(single_reduce_info, const_tiling_case)


def _add_workspace_info_in_json(single_reduce_info, compute_graph_info, block_tiling_index, block_dims):
    block_size = get_block_size(compute_graph_info.min_type)
    after_reduce_align_shape = util.shape_to_list(single_reduce_info.shape_after_reduce)[:]
    after_reduce_align_shape[-1] = (after_reduce_align_shape[-1] + block_size - 1) // block_size * block_size
    after_reduce_product = 1
    for index, value in enumerate(after_reduce_align_shape):
        if index >= block_tiling_index:
            after_reduce_product *= value
    reduce_tensor = single_reduce_info.reduce_tensor
    workspace_size = after_reduce_product * DTYPE_BYTE_MAPPING.get(reduce_tensor.dtype) * block_dims
    # sync tensor has been processed in pass
    workspace_dict_in_json = {
        "num": 1,
        "size": [workspace_size],
        "type": [0]
    }
    get_op_context().add_build_json_result("workspace", workspace_dict_in_json)


def _gen_zero_tiling_case():
    zero_tiling_case = ReduceTilingCase()

    zero_tiling_case.type = zero_tiling_case.Type.NORMAL_REDUCE
    zero_tiling_case.block_split_axis_index = 0
    zero_tiling_case.block_factor = 1
    zero_tiling_case.ub_split_axis_index = 1
    zero_tiling_case.ub_factor = operation.var_inner("_ub_factor")
    zero_tiling_case.multi_core = True
    zero_tiling_case.tiling_key = _gen_zero_tiling_key()

    return zero_tiling_case


def _calculate_tiling_cases(info: SingleReduceInfo) -> List[ReduceTilingCase]:
    tiling_case_list = []
    if info.is_reduce_not_last_axis():
        _gen_tiling_case_not_last_axis(info, tiling_case_list)
    else:
        _gen_tiling_case_last_axis(info, tiling_case_list)
        if not tiling_case_list:
            _gen_tiling_case_reduce_all(info, tiling_case_list)

    return tiling_case_list


def _calculate_atomic_tiling_cases(info: SingleReduceInfo) -> List[ReduceTilingCase]:
    tiling_case_list = []
    if check_atomic_add_support(info):
        shape_before_reduce = info.shape_before_reduce
        reduce_axis_index = info.reduce_axis_indexes
        if info.is_reduce_all_axes():
            tiling_case_list += _gen_atomic_tiling_case_reduce_all(shape_before_reduce)

        elif info.is_reduce_not_last_axis():
            tiling_case_list += _gen_atomic_tiling_case_not_last_axis(shape_before_reduce, reduce_axis_index)

        elif info.is_reduce_last_axis():
            tiling_case_list += _gen_atomic_tiling_case_last_axis(shape_before_reduce, reduce_axis_index)
    return tiling_case_list


def _calculate_group_reduce_tiling_cases(info: SingleReduceInfo) -> List[ReduceTilingCase]:
    tiling_case_list = []

    if check_atomic_add_support(info):
        return tiling_case_list
    if get_soc_spec(SOC_VERSION) == ASCEND_910B or get_soc_spec("CORE_NUM") == 1:
        return tiling_case_list
    # not ARA
    if len(info.reduce_axis_indexes) > 1 or info.is_reduce_last_axis():
        return tiling_case_list

    is_none_reduce = True
    for reduce_axis in info.reduce_axis_indexes:
        if not util.expr_equal(info.shape_before_reduce[reduce_axis], 1):
            is_none_reduce = False
            break
    if is_none_reduce:
        return tiling_case_list

    for block_split_axis in range(len(info.shape_before_reduce)):
        if block_split_axis not in info.reduce_axis_indexes:
            continue
        for ub_split_axis in range(len(info.shape_before_reduce)):
            if ub_split_axis < block_split_axis:
                continue
            tiling_case = ReduceTilingCase()
            tiling_case.type = ReduceTilingCase.Type.GROUP_REDUCE
            tiling_case.block_split_axis_index = block_split_axis
            tiling_case.ub_split_axis_index = ub_split_axis
            tiling_case.multi_core = True
            tiling_case_list.append(tiling_case)

            tiling_case_pad = _calc_pad_case(info.shape_before_reduce, block_split_axis, ub_split_axis,
                                             ReduceTilingCase.Type.GROUP_REDUCE)
            if tiling_case_pad is not None:
                tiling_case_list.append(tiling_case_pad)

    return tiling_case_list


def check_atomic_add_support(reduce_info: SingleReduceInfo):
    """
    check if current tiling case support atomic
    """
    # Common Regulation
    version = get_soc_spec(SHORT_SOC_VERSION)
    if version not in [ASCEND_910B, ASCEND_910]:
        return False

    reduce_tensor: Tensor = reduce_info.reduce_tensor
    if reduce_tensor is None:
        return False

    output_tensors = reduce_info.graph_info.output_tensor_set
    for output_tensor in output_tensors:
        if not is_reduce_tensor(output_tensor):
            return False

    # Special Regulation
    _map = AtomicSupportMap920A if version != ASCEND_910 else AtomicSupportMap910
    if reduce_tensor.dtype not in _map.get("support_dtype"):
        return False
    if reduce_tensor.op.tag not in _map.get("support_insn"):
        return False

    return True


def _gen_reduce_case(block_split_axis_index, ub_split_axis_index, case_type):
    tiling_case = ReduceTilingCase()
    tiling_case.block_split_axis_index = block_split_axis_index
    tiling_case.ub_split_axis_index = ub_split_axis_index
    tiling_case.multi_core = True
    tiling_case.type = case_type
    return tiling_case


def _gen_pad_case(block_split_axis_index, ub_split_axis_index, case_type):
    tiling_case = _gen_reduce_case(block_split_axis_index, ub_split_axis_index, case_type)
    tiling_case.is_reduce_pad_case = True
    return tiling_case


def _gen_reduce_transpose_case(block_split_axis_index, ub_split_axis_index, case_type):
    tiling_case = _gen_reduce_case(block_split_axis_index, ub_split_axis_index, case_type)
    tiling_case.is_reduce_transpose_case = True
    return tiling_case


def _gen_tiling_case_not_last_axis(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]) -> NoReturn:
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indexes

    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)

    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map.get(i)
        if orignal_axis not in reduce_axis_index:
            block_split_axis = orignal_axis

            for j in range(0, len(reordered_shape)):
                orignal_axis = reorder_to_orignal_axis_map.get(j)
                if orignal_axis not in reduce_axis_index and j < i:
                    continue
                ub_split_axis = reorder_to_orignal_axis_map.get(j)
                tiling_case = ReduceTilingCase()
                tiling_case.type = tiling_case.Type.NORMAL_REDUCE
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)

                tiling_case_pad = _calc_pad_case(shape_before_reduce, block_split_axis, ub_split_axis,
                                                 ReduceTilingCase.Type.NORMAL_REDUCE)
                if tiling_case_pad is not None:
                    tiling_case_list.append(tiling_case_pad)


def _calc_pad_case(shape_before_reduce, block_split_axis, ub_split_axis, case_type):
    # for pure move case don't need do pad
    first_dim_is_one = util.expr_equal(shape_before_reduce[0], 1)
    if first_dim_is_one and len(shape_before_reduce) == REDUCE_CASE_LENGTH_TWO:
        return None

    split_axis_info = str(block_split_axis) + "_" + str(ub_split_axis)
    support_pattern = REDUCE_SUPPORT_PAD.get(len(shape_before_reduce))
    if support_pattern is not None and split_axis_info in support_pattern:
        tiling_case_pad = _gen_pad_case(block_split_axis, ub_split_axis, case_type)
        tiling_case_pad.need_remove_pad = support_pattern.get(split_axis_info)
        return tiling_case_pad
    return None


def _calc_transpose_case(shape_before_reduce, block_split_axis, ub_split_axis, case_type):
    # for pure move case don't need do pad
    first_dim_is_one = util.expr_equal(shape_before_reduce[0], 1)
    if first_dim_is_one and len(shape_before_reduce) == REDUCE_CASE_LENGTH_TWO:
        return None

    split_axis_info = str(block_split_axis) + "_" + str(ub_split_axis)
    support_pattern = REDUCE_SUPPORT_TRANSPOSE.get(len(shape_before_reduce))
    if support_pattern is not None and split_axis_info in support_pattern:
        tiling_case_transpose = _gen_reduce_transpose_case(block_split_axis, ub_split_axis, case_type)
        return tiling_case_transpose
    return None


def _gen_tiling_case_last_axis(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indexes

    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)

    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map.get(i)
        if orignal_axis not in reduce_axis_index:
            block_split_axis = orignal_axis

            for j in range(i, len(reordered_shape)):
                ub_split_axis = reorder_to_orignal_axis_map.get(j)
                tiling_case = ReduceTilingCase()
                tiling_case.type = tiling_case.Type.NORMAL_REDUCE
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)

                tiling_case_pad = _calc_pad_case(shape_before_reduce, block_split_axis, ub_split_axis,
                                                 ReduceTilingCase.Type.NORMAL_REDUCE)
                if tiling_case_pad is not None:
                    tiling_case_list.append(tiling_case_pad)

                if info.reduce_tensor.op.tag in REDUCE_EMITSN_SUPPORT_TRANSPOSE:
                    tiling_case_transpose = _calc_transpose_case(shape_before_reduce, block_split_axis,
                                                                 ub_split_axis, ReduceTilingCase.Type.NORMAL_REDUCE)
                    if tiling_case_transpose is not None:
                        tiling_case_list.append(tiling_case_transpose)


def _gen_tiling_case_reduce_all(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indexes
    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)
    block_split_axis = 0
    for i in range(0, len(reordered_shape)):
        ub_split_axis = reorder_to_orignal_axis_map[i]
        tiling_case = ReduceTilingCase()
        tiling_case.type = tiling_case.Type.NORMAL_REDUCE
        tiling_case.block_split_axis_index = block_split_axis
        tiling_case.ub_split_axis_index = ub_split_axis
        tiling_case.multi_core = False
        tiling_case_list.append(tiling_case)


def _gen_atomic_tiling_case_not_last_axis(shape_before_reduce, reduce_axis_index) -> List[ReduceTilingCase]:
    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_nlast_shape(shape_before_reduce,
                                    reduce_axis_index)
    tiling_case_list = []
    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map[i]
        if orignal_axis in reduce_axis_index:
            block_split_axis = orignal_axis
            for j in range(0, len(reordered_shape)):
                orignal_axis = reorder_to_orignal_axis_map[j]
                if orignal_axis in reduce_axis_index and j < i:
                    continue
                ub_split_axis = reorder_to_orignal_axis_map[j]
                tiling_case = ReduceTilingCase()
                tiling_case.type = tiling_case.Type.ATOMIC_REDUCE
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)

                tiling_case_pad = _calc_pad_case(shape_before_reduce, block_split_axis, ub_split_axis,
                                                 ReduceTilingCase.Type.ATOMIC_REDUCE)
                if tiling_case_pad is not None:
                    tiling_case_list.append(tiling_case_pad)
    return tiling_case_list


def _gen_atomic_tiling_case_last_axis(shape_before_reduce, reduce_axis_index) -> List[ReduceTilingCase]:
    """
    :param shape_before_reduce:
    :param reduce_axis_index:
    :return:
    """
    reordered_shape, reorder_to_orignal_axis_map, _ = _reorder_reduce_last_shape(shape_before_reduce,
                                                                                 reduce_axis_index)
    tiling_case_list = []
    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map[i]
        if orignal_axis in reduce_axis_index:
            block_split_axis = orignal_axis
            for j in range(0, len(reordered_shape)):
                orignal_axis = reorder_to_orignal_axis_map[j]
                if orignal_axis in reduce_axis_index and j < i:
                    continue
                ub_split_axis = reorder_to_orignal_axis_map[j]
                tiling_case = ReduceTilingCase()
                tiling_case.type = tiling_case.Type.ATOMIC_REDUCE
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)
    return tiling_case_list


def _gen_atomic_tiling_case_reduce_all(shape_before_reduce) -> List[ReduceTilingCase]:
    tiling_case_list = []
    for i in range(0, len(shape_before_reduce)):
        block_split_axis = i
        for j in range(i, len(shape_before_reduce)):
            ub_split_axis = j
            tiling_case = ReduceTilingCase()
            tiling_case.type = tiling_case.Type.ATOMIC_REDUCE
            tiling_case.block_split_axis_index = block_split_axis
            tiling_case.ub_split_axis_index = ub_split_axis
            tiling_case.multi_core = True
            tiling_case_list.append(tiling_case)
    return tiling_case_list


def _reorder_reduce_nlast_shape(shape_before_reduce: list,
                                reduce_axis_index: List[int]):
    """
    reorder shape (r4,a4,r3,a3,r2,a2,r1,a1) to (a4,a3,a2, r4,r3,r2,r1,a1)
    :param shape_before_reduce: like (r4,a4,r3,a3,r2,a2,r1,a1)
    :param reduce_axis_index
    :return:
    """
    # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
    # find the last none-reduce axis a1

    a1_start_index, _ = SingleReduceInfo.find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
    last_none_reduce_axis = a1_start_index

    orignal_to_reorder_axis_map = {}
    reorder_to_orignal_axis_map = {}
    # (ak+1,ak,...,a2, rk,..,r2,r1,a1)
    reordered_shape = list(shape_before_reduce)
    temp_axis = last_none_reduce_axis - 1
    for i in range(len(reduce_axis_index) - 1, -1, -1):
        reordered_shape[temp_axis] = shape_before_reduce[
            reduce_axis_index[i]]
        reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
        orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
        temp_axis = temp_axis - 1
    for i in range(last_none_reduce_axis - 1, -1, -1):
        if i not in reduce_axis_index:
            reordered_shape[temp_axis] = shape_before_reduce[i]
            reorder_to_orignal_axis_map[temp_axis] = i
            orignal_to_reorder_axis_map[i] = temp_axis
            temp_axis = temp_axis - 1

    for i in range(last_none_reduce_axis, len(shape_before_reduce)):
        reorder_to_orignal_axis_map[i] = i
        orignal_to_reorder_axis_map[i] = i

    return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map


def _reorder_reduce_last_shape(shape_before_reduce: list,
                               reduce_axis_index: List[int]):
    """
    reorder shape (a4,r4,a3,r3,a2,r2,a1,r1) to (a4,a3,a2,a1,r4,r3,r2,r1)
    :param shape_before_reduce: like(a4,r4,a3,r3,a2,r2,a1,r1)
    :param reduce_axis_index
    :return:
    """
    # shape_before_reduce: (a4,r4,a3,r3,a2,r2,a1,r1)

    orignal_to_reorder_axis_map = {}
    reorder_to_orignal_axis_map = {}

    reordered_shape = []
    temp_axis = 0
    for i, ele in enumerate(shape_before_reduce):
        if i not in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_orignal_axis_map[temp_axis] = i
            orignal_to_reorder_axis_map[i] = temp_axis
            temp_axis = temp_axis + 1

    for i, ele in enumerate(shape_before_reduce):
        if i in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_orignal_axis_map[temp_axis] = i
            orignal_to_reorder_axis_map[i] = temp_axis
            temp_axis = temp_axis + 1

    return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map


def _get_tiling_key(atomic, db, shape_type, block_split_axis,
                    ub_split_axis, shape, reduce_idx_list):
    """
    :param atomic: "True": atomic_reduce, "False": normal_reduce.
    :param db: int number in [0,1]. "0": enable db, "1": close db.
    :param shape_type: int number in [0,99]. Diff numbers represent diff types of
           shapes. Example: "0": normal shape, "1": const shape, "2": special shape.
    :param block_split_axis: int number in [0,7] that represent index of split axis
    :param ub_split_axis: int number in [0,7] that represent index of split axis.
    :param shape: shape before reduce
    :param reduce_idx_list:

    :return: key(int32)
    """

    def _check(idx, _value):
        rule = [range(2), range(100), range(9), range(9),
                range(1000)]
        name = ["db", "shape_type", "block_split_axis", "ub_split_axis",
                "pattern"]
        if _value not in rule[idx]:
            dict_args = {"errCode": "E90003", "detailed_cause": "%s should in %s, but is %d" % (
                name[idx], str(rule[idx]), _value)}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    pattern = _get_pattern_key(shape, reduce_idx_list)
    pos = (db, shape_type, block_split_axis, ub_split_axis,
           pattern)
    val = (10 ** 9, 10 ** 7, 10 ** 6, 10 ** 5, 10 ** 2)
    key = 0
    for item, value in enumerate(pos):
        _check(item, value)
        key += value * val[item]
    if not atomic:
        key = (-1 * key) & 0xFFFFFFFF
    return key


def _get_pattern_key(_shape, _reduce_idx_list):
    pattern_key = 0
    length = len(_shape)
    for i in range(length):
        if i in _reduce_idx_list:
            pattern_key += 2 * 2 ** (length - i - 1)
        else:
            pattern_key += 2 ** (length - i - 1)

    return pattern_key


def _gen_const_tiling_key(reduce_axis):
    """
    generate dict key from reduce_axis
    :param reduce_axis:
    :return:
    """
    if not reduce_axis:
        return TILINGKEY_NONE_REDUCE_AXIS
    reduce_axis_local = list(reduce_axis)[:]
    reduce_axis_local = sorted(reduce_axis_local)
    dict_key = 0
    for i in reduce_axis_local:
        dict_key = 10 * dict_key + i + 1

    return dict_key


def _gen_zero_tiling_key():
    compute = get_context().get_current_compute()
    shape = compute.get("_shape")
    if shape == (1, -1, 0):
        return 10
    elif shape == (1, 0, -1):
        return 110
    else:
        _raise_error("Generate zero tiling key not support shape: {0}.".format(shape))


def _raise_error(message):
    dict_args = {"errCode": "E90003", "detailed_cause": message}
    raise RuntimeError(dict_args, get_error_message(dict_args))


def _current_support_check(outs):

    def _dfs_compute(tensor: tvm.tensor.Tensor):
        insn = util.get_dsl_insn(tensor)
        if insn in REDUCE_COMPUTE:
            return True

        for tensor_i in tensor.op.input_tensors:
            if _dfs_compute(tensor_i):
                return True

        return False

    for out in outs:
        ret = _dfs_compute(out)
        if not ret:
            _raise_error("Dynamic reduce schedule does "
                         "not support outputs before reduce tensor.")


def get_block_size(dtype):
    if dtype not in DTYPE_AND_BLOCK_SIZE_MAP:
        _raise_error("[%s] is not support type in norm" % dtype)
    return DTYPE_AND_BLOCK_SIZE_MAP.get(dtype)
