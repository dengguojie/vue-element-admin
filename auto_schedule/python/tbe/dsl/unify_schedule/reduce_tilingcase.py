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
from typing import Dict
from typing import Iterable
from typing import List
from typing import NoReturn
from typing import Optional
from typing import Tuple
from typing import Union

from tbe.dsl.base import operation
from tbe.dsl.base.operation import add_compile_info
from tbe.dsl.base.operation import get_compile_info
from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import register_build_pointcut
from tbe.dsl.base.operation import register_tiling_case
from te import tvm
from te.platform.cce_conf import CceProductParams as Version
from te.platform.cce_conf import get_soc_spec
from te.tvm.expr import IntImm
from te.tvm.expr import Var
# Third-party Packages
from te.tvm.tensor import Tensor

from .constants import CompileInfo
from .constants import Pattern
from .util import get_reduce_all_axes
from .util import get_reduce_axes
from .util import get_reduce_axis_indices
from .util import is_reduce_tensor
from .util import shape_to_list
from .vector_info import ComputeGraphInfo
from .vector_tilingcase import TilingCaseBase
from ..base import op_tiling
from ...common.utils.errormgr import get_error_message

CONST = "const"
ZERO = "zero"


def _get_block_size(dtype):
    if dtype in ["float32", "fp32", "int32"]:
        block_size = 8
    elif dtype in ["bool", "int8", "uint8"]:
        block_size = 32
    elif dtype in ["float16", "fp16"]:
        block_size = 16
    elif dtype in ["int64"]:
        block_size = 4
    else:
        raise RuntimeError("[%s] is not support type" % dtype)
    return block_size


def apply_compile_info(reduce_info, graph_info, tiling_list):
    # Common_Info
    atomic = 0
    for item in tiling_list:
        if item.is_atomic:
            atomic = 1
            break

    core_num = get_soc_spec("CORE_NUM")
    keep_dims = 1
    min_block_size = _get_block_size(graph_info.min_type)
    common_info = [core_num, keep_dims, min_block_size, atomic, graph_info.coef]

    # Diff_Info
    if get_context().get("mode") == CONST:
        # Compile Status: pattern must as same as tiling_key of const
        # that should be uniqueness.
        compile_axis = get_context().get_current_compute().get("ori_axis")
        pattern = _gen_const_tiling_key(compile_axis)
    else:
        pattern = _get_pattern_key(reduce_info.shape_before_reduce,
                                   reduce_info.reduce_axis_indices)
    max_ub_count = graph_info.tensor_ub_size_before_reduce
    pattern_info = [pattern]
    ub_info = [max_ub_count]

    pre_compile_info = get_compile_info()
    if pre_compile_info:
        info_map = {"common_info": common_info, "pattern_info": pattern_info,
                    "ub_info": ub_info}
        for key in info_map.keys():
            if key not in pre_compile_info.keys():
                add_compile_info(key, info_map.get(key))
            else:
                if key != "common_info":
                    key_info = pre_compile_info.get(key)
                    key_info += info_map.get(key)
                    add_compile_info(key, key_info)
    else:
        raise RuntimeError("pre_compile_info is Null")


def _calc_tiling_key(reduce_info, tiling):
    # tiling: single_case
    shape = reduce_info.shape_before_reduce
    reduce_axis_idx = reduce_info.reduce_axis_indices
    block_split_axis = tiling.block_split_axis_index
    ub_split_axis = tiling.ub_split_axis_index
    atomic = tiling.is_atomic
    shape_type, db = 0, 0

    if get_context().get("mode") == CONST:
        ori_axis = get_context().get_current_compute().get("ori_axis")
        tiling_key = _gen_const_tiling_key(ori_axis)
    elif get_context().get("mode") == ZERO:
        # TODO
        pass
    else:
        tiling_key = _get_tiling_key(atomic, db, shape_type,
                                     block_split_axis, ub_split_axis,
                                     shape, reduce_axis_idx)

    tiling.tiling_key = tiling_key


def _gen_const_tiling_case(single_reduce_info, compute_graph_info):
    add_compile_info("reduce_shape_known", True)
    const_tiling_case = ReduceTilingCase()
    shape_before_reduce = shape_to_list(single_reduce_info.shape_before_reduce)
    shape_after_reduce = shape_to_list(single_reduce_info.shape_after_reduce)
    reduce_axis_index = single_reduce_info.reduce_axis_indices
    input_dtype = tuple(compute_graph_info.input_tensor_set)[0].dtype
    output_dtype = tuple(compute_graph_info.output_tensor_set)[0].dtype
    axes_dtype = get_context().get_current_compute().get("axis_dtype")
    # staging axis info in ops
    # _ori_axis: original axes from func_enter of ReduceD, maybe None while op is ReduceSum
    # ori_axis: axes from classify, always existed
    ori_reduce_axis = get_compile_info().get("_ori_axis")
    compile_axis = get_context().get_current_compute().get("ori_axis")
    if axes_dtype is None:
        # invoking op_tiling interface during compilation need axis info in sch
        add_compile_info("_ori_axis", reduce_axis_index)
        inputs = [{"shape": shape_before_reduce, "dtype": input_dtype}]
    else:
        inputs = [{"shape": shape_before_reduce, "dtype": input_dtype},
                  {"shape": (len(reduce_axis_index),), "dtype": axes_dtype, "name": "axes",
                   "const_value": reduce_axis_index}]
    outputs = [{"shape": shape_after_reduce, "dtype": output_dtype}]
    # the flag of invoking op_tiling interface during compilation
    add_compile_info("const_shape_post", False)
    add_compile_info("compile_pattern", _gen_const_tiling_key(compile_axis))
    run_info = op_tiling.do_op_tiling(get_context().get_op_type(), get_compile_info(), inputs, outputs)
    tiling_format = {"block_axis": "int", "block_factor": "int", "ub_axis": "int", "ub_factor": "int"}
    tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
    const_tiling_case.block_split_axis_index = tiling_data["block_axis"]
    const_tiling_case.block_factor = tiling_data["block_factor"]
    const_tiling_case.ub_split_axis_index = tiling_data["ub_axis"]
    const_tiling_case.ub_factor = tiling_data["ub_factor"]
    const_tiling_case.is_atomic = run_info["clear_atomic"]
    const_tiling_case.multi_core = True if run_info["block_dim"] > 1 else False
    _calc_tiling_key(single_reduce_info, const_tiling_case)
    # the flag of invoking op_tiling interface during running
    add_compile_info("const_shape_post", True)
    # invoking op_tiling interface during running need axis info in ops
    if ori_reduce_axis is not None:
        add_compile_info("_ori_axis", ori_reduce_axis)

    block_dims = get_compile_info().get(CompileInfo.BLOCK_DIMS)
    if block_dims is None:
        block_dims = {}
        add_compile_info(CompileInfo.BLOCK_DIMS, block_dims)
    block_dims[str(const_tiling_case.tiling_key)] = run_info["block_dim"]
    atomic_flags = get_compile_info().get(CompileInfo.ATOMIC_FLAGS)
    if atomic_flags is None:
        atomic_flags = {}
        add_compile_info(CompileInfo.ATOMIC_FLAGS, atomic_flags)
    atomic_flags[str(const_tiling_case.tiling_key)] = run_info["clear_atomic"]

    return [const_tiling_case]


@register_tiling_case(pattern=Pattern.REDUCE)
def calc_tiling_case(outs, options=None):
    [options].clear()
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    current_compute = get_context().get_current_compute()

    # construct information of graph
    compute_graph_info = ComputeGraphInfo(outs)
    single_reduce_info = SingleReduceInfo(compute_graph_info)
    current_compute.add("compute_graph_info", compute_graph_info)
    current_compute.add("single_reduce_info", single_reduce_info)
    if not compute_graph_info.reduce_tensor_set:
        raise RuntimeError("Couldn't find reduce node for ReduceSchedule")

    if current_compute.get("mode") == ZERO:
        return [_gen_zero_tiling_case()]

    tiling_case_list = []
    # Normal reduce tiling cases
    tiling_case_list += _calculate_tiling_cases(single_reduce_info)
    # Atomic reduce tiling cases
    tiling_case_list += _calculate_atomic_tiling_cases(single_reduce_info)
    apply_compile_info(single_reduce_info, compute_graph_info, tiling_case_list)
    # calc_tiling_key
    for tiling_case in tiling_case_list:
        _calc_tiling_key(single_reduce_info, tiling_case)
    if get_context().get("mode") == CONST:
        return _gen_const_tiling_case(single_reduce_info, compute_graph_info)

    return tiling_case_list


def _gen_zero_tiling_case():
    zero_tiling_case = ReduceTilingCase()

    zero_tiling_case.is_atomic = False
    zero_tiling_case.block_split_axis_index = 0
    zero_tiling_case.block_factor = 1
    zero_tiling_case.ub_split_axis_index = 1
    zero_tiling_case.ub_factor = operation.var("ub_factor")
    zero_tiling_case.multi_core = True
    zero_tiling_case.tiling_key = _gen_zero_tiling_key()

    return zero_tiling_case


@register_build_pointcut(pattern=Pattern.REDUCE)
def build_pointcut(func, *args, **kwargs):
    """
    build pointcut
    :param func:
    :param args:
    :param kwargs:
    :return:
    """
    func(*args, **kwargs)


class SingleReduceInfo:
    def __init__(self, compute_graph_info: ComputeGraphInfo):
        if len(compute_graph_info.reduce_tensor_set) != 1:
            raise RuntimeError("ComputeGraph is not in Single Reduce Pattern: %s" %
                               compute_graph_info.reduce_tensor_set)
        # Assume only one reduce node
        self.reduce_tensor: Tensor = tuple(compute_graph_info.reduce_tensor_set)[0]
        self.all_axes: List[Var] = get_reduce_all_axes(self.reduce_tensor)
        self.reduce_axes: List[Var] = get_reduce_axes(self.reduce_tensor)

        compute = operation.get_context().get_current_compute()
        if compute.get("mode") == "zero" and compute.get("shape") == (1, -1, 0):
            self.shape_before_reduce = list(self.reduce_tensor.shape) + [tvm.expr.IntImm("int32", 0)]
        else:
            self.shape_before_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.op.input_tensors[0].shape)
        self.shape_after_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.shape)
        self.reduce_axis_indices: List[int] = get_reduce_axis_indices(self.reduce_tensor)
        self.keepdims: bool = len(self.shape_before_reduce) == len(self.shape_after_reduce)
        self.graph_info: ComputeGraphInfo = compute_graph_info

    def is_reduce_not_last_axis(self) -> bool:
        compute = operation.get_context().get_current_compute()
        if compute.get("mode") == "zero":
            return False

        is_not_last_axis = self.all_axes[-1] not in self.reduce_axes
        return is_not_last_axis

    def is_reduce_last_axis(self) -> bool:
        return self.all_axes[-1] in self.reduce_axes

    def is_reduce_all_axes(self) -> bool:
        return set(self.all_axes) == set(self.reduce_axes)

    @staticmethod
    def find_last_reduce_axis(shape, reduce_axis_indices: Iterable[int]):
        # shape_before_reduce:(ak+1,rk,...,r2,a2,r1,a1) or (ak,rk,...,r2,a1,r1)
        # find r1 position, r1 may contain continues axis
        r1_end_index = None
        for i in range(len(shape) - 1, -1, -1):
            if i in reduce_axis_indices:
                r1_end_index = i
                break
        r1_start_index = r1_end_index
        if r1_end_index is None:
            return r1_start_index, r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_indices:
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
        none_reduce_index_map = {}
        count = 0
        for i in range(0, len(shape_before_reduce)):
            if i not in reduce_axis_index:
                none_reduce_index_map[i] = count
                count += 1
        return none_reduce_index_map


class ReduceTilingCase(TilingCaseBase):
    def __init__(self):
        self.is_atomic = False
        self.block_split_axis_index = None
        self.block_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core: Optional[bool] = None
        self.tiling_key = None

    def __repr__(self):
        segment0 = "ATOMIC" if self.is_atomic else "NORMAL"
        segment1 = "ENABLED" if self.multi_core else "DISABLED"
        return "%s REDUCE: (%d, %d) with multicore %s" % (segment0,
                                                          self.block_split_axis_index, self.ub_split_axis_index,
                                                          segment1)

    def __hash__(self):
        return hash((self.is_atomic, self.block_split_axis_index, self.block_factor,
                     self.ub_split_axis_index, self.ub_factor, self.multi_core))

    def __eq__(self, other) -> bool:
        condition0 = other.self.is_atomic == self.is_atomic
        condition1 = other.block_split_axis == self.block_split_axis_index
        condition2 = other.block_factor == self.block_factor
        condition3 = other.ub_split_axis == self.ub_split_axis_index
        condition4 = other.ub_factor == self.ub_factor
        condition5 = other.multi_core == self.multi_core
        return (type(other) == type(self)
                and condition0 and condition1 and condition2 and condition3 and condition4 and condition5)

    def __ne__(self, other) -> bool:
        condition0 = other.self.is_atomic != self.is_atomic
        condition1 = other.block_split_axis != self.block_split_axis_index
        condition2 = other.block_factor != self.block_factor
        condition3 = other.ub_split_axis != self.ub_split_axis_index
        condition4 = other.ub_factor != self.ub_factor
        condition5 = other.multi_core != self.multi_core
        return (type(other) != type(self)
                or condition0 or condition1 or condition2 or condition3 or condition4 or condition5)


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
        reduce_axis_index = info.reduce_axis_indices
        if info.is_reduce_all_axes():
            tiling_case_list += _gen_atomic_tiling_case_reduce_all(shape_before_reduce)

        elif info.is_reduce_not_last_axis():
            tiling_case_list += _gen_atomic_tiling_case_not_last_axis(shape_before_reduce, reduce_axis_index)

        elif info.is_reduce_last_axis():
            tiling_case_list += _gen_atomic_tiling_case_last_axis(shape_before_reduce, reduce_axis_index)
    return tiling_case_list


def check_atomic_add_support(reduce_info: SingleReduceInfo):
    if not Version().is_cloud_version():
        return False
    reduce_tensor: Tensor = reduce_info.reduce_tensor
    if reduce_tensor is None:
        return False
    dtype = reduce_tensor.dtype
    if dtype != "float32":
        return False
    output_tensors: Tensor = reduce_info.graph_info.output_tensor_set
    for output_tensor in output_tensors:
        dtype = output_tensor.dtype
        if dtype != "float32":
            return False
        if not is_reduce_tensor(output_tensor):
            return False
    tag = reduce_tensor.op.tag
    if tag.find("sum") == -1:
        return False
    return True


def _gen_tiling_case_not_last_axis(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]) -> NoReturn:
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indices

    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)

    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map[i]
        if orignal_axis not in reduce_axis_index:
            block_split_axis = orignal_axis

            for j in range(0, len(reordered_shape)):
                orignal_axis = reorder_to_orignal_axis_map[j]
                if orignal_axis not in reduce_axis_index and j < i:
                    continue
                ub_split_axis = reorder_to_orignal_axis_map[j]
                tiling_case = ReduceTilingCase()
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)


def _gen_tiling_case_last_axis(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indices

    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)

    for i in range(0, len(reordered_shape)):
        orignal_axis = reorder_to_orignal_axis_map[i]
        if orignal_axis not in reduce_axis_index:
            block_split_axis = orignal_axis

            for j in range(i, len(reordered_shape)):
                ub_split_axis = reorder_to_orignal_axis_map[j]
                tiling_case = ReduceTilingCase()
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)


def _gen_tiling_case_reduce_all(info: SingleReduceInfo, tiling_case_list: List[ReduceTilingCase]):
    shape_before_reduce = info.shape_before_reduce
    reduce_axis_index = info.reduce_axis_indices
    reordered_shape, reorder_to_orignal_axis_map, _ = \
        _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)
    block_split_axis = 0
    for i in range(0, len(reordered_shape)):
        ub_split_axis = reorder_to_orignal_axis_map[i]
        tiling_case = ReduceTilingCase()
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
                tiling_case.is_atomic = True
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)
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
                tiling_case.is_atomic = True
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
            tiling_case.is_atomic = True
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

    def _check(idx, value):
        rule = [range(2), range(100), range(9), range(9),
                range(1000)]
        name = ["db", "shape_type", "block_split_axis", "ub_split_axis",
                "pattern"]
        if value not in rule[idx]:
            dict_args = dict()
            dict_args["errCode"] = "E90003"
            dict_args["detailed_cause"] = "%s should in %s, but is %d" % (
                name[idx], str(rule[idx]), value)
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
        key *= -1
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
        return -1
    reduce_axis_local = list(reduce_axis)[:]
    reduce_axis_local = sorted(reduce_axis_local)
    dict_key = 0
    for i in reduce_axis_local:
        dict_key = 10 * dict_key + i + 1

    return dict_key


def _gen_zero_tiling_key():
    compute = get_context().get_current_compute()
    shape = compute.get("shape")
    if shape == (1, -1, 0):
        return 10
    elif shape == (1, 0, -1):
        return 110
    else:
        _raise_error("Generate zero tiling key not support shape: {0}.".format(shape))


def _raise_error(message):
    dict_args = dict()
    dict_args["errCode"] = "E90003"
    dict_args["detailed_cause"] = message
    raise RuntimeError(dict_args, get_error_message(dict_args))
