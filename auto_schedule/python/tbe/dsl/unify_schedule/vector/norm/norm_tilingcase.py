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

BLOCK_SIZE_BYTE = 32

# vcmpsel constant
VCMP_INPUT_NUMBER = 2
VSEL_INPUT_NUMBER = 3
VCMPSEL_INPUT_NUMBER = 4

# temp space for last axis broadcast use vtranspose
VTRANSPOSE_TEMP_SPACE = 8192
# temp space for last axis broadcast use vnchwconv
VNCHWCONV_TEMP_SPACE = 1024


class CalcNormTilingCase(Computation):
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

        tiling_case_list = []
        tiling_case_list += _add_tiling_case(norm_info)
        _apply_compile_info(norm_compute_graph_info, norm_info)
        # calc tiling key
        for tiling_case in tiling_case_list:
            _calc_tiling_key(norm_info, tiling_case)

        if current_compute.get("_mode") == "const":
            return _gen_const_tiling_case(norm_info, norm_compute_graph_info)

        return tiling_case_list


def get_block_size(dtype):
    if dtype in ["float32", "fp32", "int32"]:
        block_size = 8
    elif dtype in ["bool", "int8", "uint8"]:
        block_size = 32
    elif dtype in ["float16", "fp16"]:
        block_size = 16
    elif dtype in ["int64"]:
        block_size = 4
    else:
        _raise_error("[%s] is not support type" % dtype)
    return block_size


def _map_append(input_map, key, value):
    if input_map.get(key):
        if isinstance(value, list):
            for tmp_value in value:
                if tmp_value not in input_map[key]:
                    input_map[key].append(tmp_value)
        else:
            if value not in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]


def _raise_error(message):
    dict_args = {}
    dict_args["errCode"] = "E90003"
    dict_args["detailed_cause"] = message
    raise RuntimeError(dict_args, get_error_message(dict_args))


def _apply_compile_info(graph_info, norm_info):
    core_num = get_soc_spec("CORE_NUM")
    keep_dims = 1
    min_block_size = get_block_size(graph_info.min_type)
    common_info = [core_num, min_block_size, keep_dims, graph_info.available_ub_size,
                   graph_info.workspace_available_min_ub_size]
    # used to calculate the size of workspace, 1 means before reduce, 0 means after reduce
    workspace_type_list = []
    # number of occupied bytes
    workspace_bytes_list = []
    local_set = graph_info.workspace_and_reduce_tensor_set \
        if norm_info.exist_partial_reorder else graph_info.workspace_tensor_set
    for workspace_tensor in local_set:
        if workspace_tensor in graph_info.tensors_before_reduce:
            workspace_type_list.append(1)
        else:
            workspace_type_list.append(0)
        workspace_bytes_list.append(DTYPE_BYTE_MAPPING[workspace_tensor.dtype])
    workspace_diff_count = len(local_set) - len(graph_info.workspace_tensor_set)
    workspace_info = {"_workspace_type": workspace_type_list,
                      "_workspace_bytes": workspace_bytes_list,
                      "_workspace_diff_count": workspace_diff_count}
    add_compile_info_inner("_common_info", common_info)
    add_compile_info_inner("_workspace_info", workspace_info)


def _add_tiling_case(norm_info):
    shape_before_reduce = norm_info.shape_before_reduce
    reduce_axis_index = norm_info.reduce_axis_indices
    tiling_case_list = []

    for block_split_axis in range(len(shape_before_reduce)):
        # block split on A axis
        if block_split_axis in reduce_axis_index:
            continue
        for ub_split_axis in range(len(shape_before_reduce)):
            # ub tiling axis is after block tiling axis if ub tiling split normal axis
            if ub_split_axis not in reduce_axis_index and ub_split_axis < block_split_axis:
                continue
            tiling_case = NormTilingCase()
            tiling_case.block_split_axis_index = block_split_axis
            tiling_case.ub_split_axis_index = ub_split_axis
            tiling_case.multi_core = True
            tiling_case_list.append(tiling_case)

    # some special cases need workspace sch with partial reorder
    if norm_info.exist_partial_reorder:
        for block_split_axis in range(len(shape_before_reduce)):
            # block split on A axis
            if block_split_axis in reduce_axis_index:
                continue
            for ub_split_axis in reduce_axis_index:
                tiling_case = NormTilingCase()
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = False
                tiling_case.is_partial_reorder_case = True
                tiling_case_list.append(tiling_case)

    # all reduce don't have A axis
    if norm_info.is_all_reduce:
        for ub_split_axis in range(len(shape_before_reduce)):
            tiling_case = NormTilingCase()
            tiling_case.ub_split_axis_index = ub_split_axis
            tiling_case.multi_core = False
            tiling_case_list.append(tiling_case)

    return tiling_case_list


def _gen_const_tiling_case(norm_info, compute_graph_info):
    def __add_workspace_info_in_json():
        workspace_size = []
        workspace_type = []

        workspace_info = get_compile_info().get("_workspace_info")

        before_reduce_align_shape = util.shape_to_list(norm_info.shape_before_reduce)[:]
        before_reduce_align_shape[-1] = (before_reduce_align_shape[-1] + block_size - 1) // block_size * block_size
        after_reduce_align_shape = util.shape_to_list(norm_info.shape_before_reduce)[:]
        after_reduce_align_shape[-1] = (after_reduce_align_shape[-1] + block_size - 1) // block_size * block_size
        before_reduce_product = functools.reduce(lambda x1, x2: x1 * x2, before_reduce_align_shape)
        after_reduce_product = functools.reduce(lambda x1, x2: x1 * x2, after_reduce_align_shape)

        workspace_num = len(workspace_info["_workspace_type"]) if const_tiling_case.is_partial_reorder_case else \
            len(workspace_info["_workspace_type"]) - workspace_info["_workspace_diff_count"]
        for i in range(workspace_num):
            if workspace_info["_workspace_type"][i] == 1:
                workspace_size.append(before_reduce_product * workspace_info["_workspace_bytes"][i])
            else:
                workspace_size.append(after_reduce_product * workspace_info["_workspace_bytes"][i])
            workspace_type.append(0)

        if workspace_num != 0:
            workspace_dict_in_json = {
                "num": workspace_num,
                "size": workspace_size,
                "type": workspace_type
            }
            get_op_context().add_build_json_result("workspace", workspace_dict_in_json)
        return workspace_size

    add_compile_info_inner("_reduce_shape_known", True)
    const_tiling_case = NormTilingCase()
    ori_reduce_axis = get_compile_info().get("_ori_axis")
    block_size = get_block_size(compute_graph_info.min_type)
    reduce_axis_index = norm_info.reduce_axis_indices

    inputs = []
    for single_tensor in compute_graph_info.input_tensor_set:
        shape = util.shape_to_list(single_tensor.shape)
        inputs.append({"shape": shape, "dtype": single_tensor.dtype})
    outputs = []
    for single_tensor in compute_graph_info.output_tensor_set:
        shape = util.shape_to_list(single_tensor.shape)
        outputs.append({"shape": shape, "dtype": single_tensor.dtype})
    add_compile_info_inner("_ori_axis", reduce_axis_index)

    # the flag of invoking op_tiling interface during compilation
    add_compile_info_inner("_const_shape_post", False)

    run_info = do_op_tiling(get_context().get_op_type(), get_compile_info(), inputs, outputs)
    tiling_format = {"block_axis": "int", "block_factor": "int", "ub_axis": "int", "ub_factor": "int"}\
        if not norm_info.is_all_reduce else {"ub_axis": "int", "ub_factor": "int"}
    tiling_data = decode(run_info["tiling_data"], tiling_format)
    const_tiling_case.block_split_axis_index = tiling_data.get("block_axis")
    const_tiling_case.block_factor = tiling_data.get("block_factor")
    const_tiling_case.ub_split_axis_index = tiling_data.get("ub_axis")
    const_tiling_case.ub_factor = tiling_data.get("ub_factor")
    const_tiling_case.multi_core = True if run_info.get("block_dim") > 1 else False
    if len(reduce_axis_index) > 1 and util.shape_to_list(norm_info.shape_before_reduce)[-1] < block_size and \
            not norm_info.is_all_reduce:
        const_tiling_case.is_partial_reorder_case = True

    _calc_tiling_key(norm_info, const_tiling_case)

    # the flag of invoking op_tiling interface during running
    add_compile_info_inner("_const_shape_post", True)
    if ori_reduce_axis is not None:
        add_compile_info_inner("_ori_axis", ori_reduce_axis)
    add_compile_info_inner(CompileInfo.BLOCK_DIMS, run_info["block_dim"])
    add_compile_info_inner("_const_tiling_key", const_tiling_case.tiling_key)

    workspace_size_list = []
    # add workspace info in json
    if tiling_data["ub_axis"] in reduce_axis_index and get_op_context():
        workspace_size_list = __add_workspace_info_in_json()
    add_compile_info_inner("_const_workspace_size", workspace_size_list)

    return [const_tiling_case]


def reorder_reduce_shape(shape_before_reduce, reduce_axis_index, is_reduce_last_axis):
    return _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index) if is_reduce_last_axis else \
        _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)


def _reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index):
    ori_to_reorder_axis_map = {}
    reorder_to_ori_axis_map = {}
    reordered_shape = []
    temp_axis = 0
    for i, ele in enumerate(shape_before_reduce):
        if i not in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis += 1

    for i, ele in enumerate(shape_before_reduce):
        if i in reduce_axis_index:
            reordered_shape.append(ele)
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis += 1

    return reordered_shape, reorder_to_ori_axis_map, ori_to_reorder_axis_map


def _reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index):
    last_none_reduce_axis = max(reduce_axis_index) + 1
    ori_to_reorder_axis_map = {}
    reorder_to_ori_axis_map = {}
    reordered_shape = list(shape_before_reduce)[:]
    temp_axis = last_none_reduce_axis - 1
    for i in range(len(reduce_axis_index) - 1, -1, -1):
        reordered_shape[temp_axis] = shape_before_reduce[reduce_axis_index[i]]
        reorder_to_ori_axis_map[temp_axis] = reduce_axis_index[i]
        ori_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
        temp_axis -= 1

    for i in range(last_none_reduce_axis - 1, -1, -1):
        if i not in reduce_axis_index:
            reordered_shape[temp_axis] = shape_before_reduce[i]
            reorder_to_ori_axis_map[temp_axis] = i
            ori_to_reorder_axis_map[i] = temp_axis
            temp_axis -= 1

    for i in range(last_none_reduce_axis, len(shape_before_reduce)):
        reorder_to_ori_axis_map[i] = i
        ori_to_reorder_axis_map[i] = i

    return reordered_shape, reorder_to_ori_axis_map, ori_to_reorder_axis_map


def _calc_tiling_key(norm_info, tiling_case):
    shape = norm_info.shape_before_reduce
    reduce_axis_idx = norm_info.reduce_axis_indices
    block_split_axis = tiling_case.block_split_axis_index
    ub_split_axis = tiling_case.ub_split_axis_index
    db = 0
    sch_type = 0
    is_const = get_context().get_current_compute().get("_mode") == "const"
    if is_const and not tiling_case.is_partial_reorder_case:
        sch_type = 1
    elif not is_const and tiling_case.is_partial_reorder_case:
        sch_type = 2
    elif is_const and tiling_case.is_partial_reorder_case:
        sch_type = 3

    tiling_key = _get_tiling_key(db, sch_type, block_split_axis, ub_split_axis, shape, reduce_axis_idx)
    tiling_case.tiling_key = tiling_key


def _get_tiling_key(db, sch_type, block_split_axis, ub_split_axis, shape, reduce_idx_list):
    """
    :param db: int number in [0,1]. "0": enable db, "1": close db.
    :param sch_type: int number in [0,99]. Diff numbers represent diff types of
           sch. Example: "0": normal shape, "1": const shape, "2": special shape.
    :param block_split_axis: int number in [0,7] that represent index of split axis
    :param ub_split_axis: int number in [0,7] that represent index of split axis.
    :param shape: shape before reduce
    :param reduce_idx_list:

    :return: key(int32)
    """

    pattern = _get_pattern_key(shape, reduce_idx_list)
    pos = (db, sch_type, block_split_axis, ub_split_axis, pattern) if block_split_axis is not None \
        else (db, sch_type, ub_split_axis, pattern)
    val = (10 ** 9, 10 ** 7, 10 ** 6, 10 ** 5, 10 ** 2) if block_split_axis is not None \
        else (10 ** 9, 10 ** 7, 10 ** 6, 10 ** 3)
    key = 0
    for item, value in enumerate(pos):
        key += value * val[item]

    return key


def _get_pattern_key(shape, reduce_idx_list):
    pattern = 0
    length = len(shape)
    for i in range(length):
        if i in reduce_idx_list:
            pattern += 2 * 2 ** (length - i - 1)
        else:
            pattern += 2 ** (length - i - 1)

    return pattern


def _after_build():
    def _encode_var_name(_var_names):
        after_encode_name = []
        for name in _var_names:
            names = name[1:].split('_')
            if names[0] == 'dim':
                after_encode_name.append(100 + int(names[1]))
            elif names[0] == 'block':
                after_encode_name.append(200)
            elif names[0] == 'ub':
                after_encode_name.append(300)
            else:
                _raise_error("unknown var name in norm schedule, please check")

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
    func(*args, **kwargs)
    _after_build()


class NormComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """
    def __init__(self, output_tensors: Iterable[Tensor]):
        """
        Initialize containers and try to collect info
        """
        # Basic Info
        self.soc_ub_size = get_soc_spec("UB_SIZE")
        # real_output_tensor_set: set doesn't contain fake_node
        self.output_tensor_set: Optional[Set[Tensor]] = None
        self.real_output_tensor_set: Optional[Set[Tensor]] = None
        self.tensor_consumers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_producers_map: Optional[Dict[Tensor, Set[Tensor]]] = None
        self.tensor_list: Optional[List[Tensor]] = None
        # Extra info initialized by hooks
        self.reduce_tensor_set: Set[Tensor] = set()
        self.broadcast_tensor_set: Set[Tensor] = set()
        self.elewise_tensor_set: Set[Tensor] = set()
        self.input_tensor_set: Set[Tensor] = set()
        self.non_gm_input_tensor_set: Set[Tensor] = set()
        # Extra info initialized after pre-initialization
        self.mid_output_tensor_set: Set[Tensor] = set()
        self.mid_tensor_set: Set[Tensor] = set()
        self.endpoint_output_tensor_set: Set[Tensor] = set()
        self.workspace_tensor_set: Set[Tensor] = set()
        self.workspace_and_reduce_tensor_set: Set[Tensor] = set()
        # Extra workspace tensor and its info
        self.workspace_info_map: Dict[Tensor, Dict] = {}
        self.workspace_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # Extra split tensor and its sub_graph
        self.split_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        self.cache_clone_tensor_set: Set[Tensor] = set()
        self.tensors_before_reduce: Optional[List[Tensor]] = []
        self.tensors_after_reduce: Optional[List[Tensor]] = []
        self.max_type: Optional[str] = None
        self.min_type: Optional[str] = None
        self.temp_ub_size = 0
        self.coexisting_quantity = 0
        self.available_ub_size = 0
        self.workspace_available_min_ub_size = 0
        self.reduce_axis_len = 0
        # Do info collection
        self.collect_info(output_tensors)
        self.get_tensors_before_and_after_reduce()
        self.find_workspace_tensor()
        self.calc_coexisting_quantities_and_temp_buffer_size()

    def collect_info(self, output_tensors: Iterable[Tensor]):
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
                                        lambda _tensor: None)
                                   ))
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor_set
        self.gen_endpoint_output_tensor_set()

    def gen_endpoint_output_tensor_set(self):
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map[output_tensor]:
                self.endpoint_output_tensor_set.add(output_tensor)

    def gen_mid_tensor_sets(self):
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map[tensor]:
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    @staticmethod
    def eq_tvm_shape(shape_a: List, shape_b: List):
        length_a = len(shape_a)
        length_b = len(shape_b)
        if length_a != length_b:
            return False
        for idx, _ in enumerate(range(length_a)):
            ret_value = hasattr(shape_a[idx], "value") and hasattr(shape_b[idx], "value")
            ret_name = hasattr(shape_a[idx], "name") and hasattr(shape_b[idx], "name")
            if ret_value:
                if shape_a[idx].value != shape_b[idx].value:
                    return False
            elif ret_name:
                if shape_a[idx].name != shape_b[idx].name:
                    return False
            else:
                if shape_a[idx] != shape_b[idx]:
                    return False
        return True

    @staticmethod
    def dfs_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor],
                          hooks: Tuple[Tuple[Callable[[Tensor], bool],
                                             Callable[[Tensor], Any],
                                             Callable[[Tensor], Any]], ...]):
        def recursive_func(_root_tensor: Tensor,
                           _visited_list: Set[Tensor],
                           _tensor_consumers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _tensor_producers_map: Dict[Tensor, Union[Set[Tensor]]],
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

    @staticmethod
    def dfs_sub_compute_graph(root_tensor: Union[Iterable[Tensor], Tensor], other_workspace_tensor_set: Set[Tensor]):
        def recursive_func(_root_tensor: Tensor,
                           _visited_list: Set[Tensor],
                           _tensor_consumers_map: Dict[Tensor, Union[Set[Tensor]]],
                           _tensor_producers_map: Dict[Tensor, Union[Set[Tensor]]]):
            _visited_list.add(_root_tensor)
            _tensor_producers_map.setdefault(_root_tensor, set())
            _tensor_consumers_map.setdefault(_root_tensor, set())

            for in_tensor in _root_tensor.op.input_tensors:
                _tensor_consumers_map.setdefault(in_tensor, set())
                _tensor_consumers_map[in_tensor].add(_root_tensor)
                _tensor_producers_map[_root_tensor].add(in_tensor)
                if in_tensor in other_workspace_tensor_set:
                    _visited_list.add(in_tensor)
                    _tensor_producers_map[in_tensor] = set()
                    continue
                recursive_func(in_tensor,
                               _visited_list,
                               _tensor_consumers_map,
                               _tensor_producers_map)
        visited_list = set()
        tensor_consumers_map = {}
        tensor_producers_map = {}
        if isinstance(root_tensor, (list, tuple, set)):
            for tensor in root_tensor:
                recursive_func(tensor, visited_list,
                               tensor_consumers_map,
                               tensor_producers_map)
        elif isinstance(root_tensor, Tensor):
            recursive_func(root_tensor, visited_list,
                           tensor_consumers_map, tensor_producers_map)

        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def get_tensors_before_and_after_reduce(self):
        # Assume all reduce node have the same shape, axis and keepdims
        reduce_tensor = list(self.reduce_tensor_set)[0]
        shape_after_reduce = list(reduce_tensor.shape)
        shape_before_reduce = list(reduce_tensor.op.input_tensors[0].shape)

        self.reduce_axis_len = len(util.get_reduce_axes(reduce_tensor))
        self.max_type = self.tensor_list[0].dtype
        self.min_type = self.tensor_list[0].dtype

        for item in self.tensor_list:
            if DTYPE_BYTE_MAPPING[item.dtype] > DTYPE_BYTE_MAPPING[self.max_type]:
                self.max_type = item.dtype
            elif DTYPE_BYTE_MAPPING[item.dtype] < DTYPE_BYTE_MAPPING[self.min_type]:
                self.min_type = item.dtype
            if self.eq_tvm_shape(list(item.shape), shape_before_reduce):
                self.tensors_before_reduce.append(item)
            elif self.eq_tvm_shape(list(item.shape), shape_after_reduce):
                self.tensors_after_reduce.append(item)
            else:
                _raise_error("unknown tensor")

        if not self.tensors_after_reduce:
            # pure data_move pattern
            self.tensors_after_reduce = self.tensors_before_reduce

    def find_workspace_tensor(self):
        """
        find workspace tensor
        """
        def _find_possible_cross_hierarchy_tensor(_idx, _current_tensor):
            if _current_tensor in self.endpoint_output_tensor_set:
                return
            # _idx represents the index of path
            if _current_tensor in self.reduce_tensor_set:
                path_index_and_tensor_map[_idx].add(_current_tensor)
            for _consumer_tensor in self.tensor_consumers_map[_current_tensor]:
                # Reduce nodes exist on this path
                if _consumer_tensor in self.reduce_tensor_set:
                    path_index_and_tensor_map[_idx].add(_consumer_tensor)
                _find_possible_cross_hierarchy_tensor(_idx, _consumer_tensor)

        def _traverse_tensor(_root_tensor, _visited_set):
            _visited_set.add(_root_tensor)
            for in_tensor in _root_tensor.op.input_tensors:
                _traverse_tensor(in_tensor, _visited_set)

        def _judge_workspace_or_cache_clone(cross_hierarchy_tensor):
            # mte2 mte3 count 1.5
            # vector count 1
            # reduce and broadcast choose workspace
            visited_set = set()
            # workspace need mte3 and at least 2 mte2
            workspace_count = 1.5 + 2 * 1.5
            cache_clone_count = 0
            _traverse_tensor(cross_hierarchy_tensor, visited_set)
            for count_tensor in visited_set:
                if count_tensor in self.reduce_tensor_set | self.broadcast_tensor_set:
                    return "workspace", visited_set
                if count_tensor in self.input_tensor_set:
                    workspace_count += 1 * 1.5
                    cache_clone_count += 2 * 1.5
                else:
                    workspace_count += 1
                    cache_clone_count += 1 * 2
            if workspace_count > cache_clone_count:
                return "cache_clone", visited_set
            return "workspace", visited_set

        cross_hierarchy_tensor_set = set()
        # find the cross hierarchy tensors
        for single_tensor in self.tensor_consumers_map:
            if len(self.tensor_consumers_map[single_tensor]) > 1:
                path_index_and_tensor_map = {}
                tensor_and_path_index_map = {}
                for idx, consumer_tensor in enumerate(self.tensor_consumers_map[single_tensor]):
                    path_index_and_tensor_map[idx] = set()
                    _find_possible_cross_hierarchy_tensor(idx, consumer_tensor)

                for path_index in path_index_and_tensor_map:
                    if not path_index_and_tensor_map[path_index]:
                        _map_append(tensor_and_path_index_map, tuple(self.endpoint_output_tensor_set)[0], path_index)
                    else:
                        _map_append(tensor_and_path_index_map, tuple(path_index_and_tensor_map[path_index])[0],
                                    path_index)
                # if multiple paths of a tensor have the same reduce node,
                # the tensor does not belong to the cross hierarchy tensor
                if len(tensor_and_path_index_map) > 1:
                    cross_hierarchy_tensor_set.add(single_tensor)

        for single_tensor in cross_hierarchy_tensor_set - self.input_tensor_set:
            judge_result, visited_tensor = _judge_workspace_or_cache_clone(single_tensor)
            if judge_result == "workspace":
                self.workspace_tensor_set.add(single_tensor)
                self.workspace_and_reduce_tensor_set.add(single_tensor)
            else:
                self.cache_clone_tensor_set = self.cache_clone_tensor_set | visited_tensor

        for single_tensor in self.reduce_tensor_set:
            self.workspace_and_reduce_tensor_set.add(single_tensor)

        # get the sub_graph of split tensor
        for split_tensor in self.workspace_and_reduce_tensor_set:
            other_split_tensor_set = self.workspace_and_reduce_tensor_set.copy()
            other_split_tensor_set.remove(split_tensor)
            tensor_list, tensor_consumers_map, tensor_producers_map = \
                self.dfs_sub_compute_graph(split_tensor, other_split_tensor_set)
            self.split_tensor_and_sub_graph_map[split_tensor] = {
                "sub_tensor_list": tensor_list,
                "sub_tensor_consumers_map": tensor_consumers_map,
                "sub_tensor_producers_map": tensor_producers_map
            }
        tensor_list, tensor_consumers_map, tensor_producers_map = \
            self.dfs_sub_compute_graph(tuple(self.endpoint_output_tensor_set)[0],
            self.workspace_and_reduce_tensor_set)
        self.split_tensor_and_sub_graph_map[tuple(self.endpoint_output_tensor_set)[0]] = {
            "sub_tensor_list": tensor_list,
            "sub_tensor_consumers_map": tensor_consumers_map,
            "sub_tensor_producers_map": tensor_producers_map
        }

        # get the sub_graph of workspace tensor
        for workspace_tensor in self.workspace_tensor_set:
            other_workspace_tensor_set = self.workspace_tensor_set.copy()
            other_workspace_tensor_set.remove(workspace_tensor)
            tensor_list, tensor_consumers_map, tensor_producers_map = \
                self.dfs_sub_compute_graph(workspace_tensor, other_workspace_tensor_set)
            self.workspace_tensor_and_sub_graph_map[workspace_tensor] = {
                "sub_tensor_list": tensor_list,
                "sub_tensor_consumers_map": tensor_consumers_map,
                "sub_tensor_producers_map": tensor_producers_map
            }
        tensor_list, tensor_consumers_map, tensor_producers_map = \
            self.dfs_sub_compute_graph(tuple(self.endpoint_output_tensor_set)[0], self.workspace_tensor_set)
        self.workspace_tensor_and_sub_graph_map[tuple(self.endpoint_output_tensor_set)[0]] = {
            "sub_tensor_list": tensor_list,
            "sub_tensor_consumers_map": tensor_consumers_map,
            "sub_tensor_producers_map": tensor_producers_map
        }

    def calc_coexisting_quantities_and_temp_buffer_size(self):
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

        def _calc_current_space(_tensor, _is_workspace=False):
            def __get_broadcast_axis_info(_broadcast_tensor):
                _dst_shape = util.shape_to_list(_broadcast_tensor.shape)
                if not hasattr(_broadcast_tensor.op, "input_tensors") or not _broadcast_tensor.op.input_tensors:
                    return len(_dst_shape), True
                _src_shape = util.shape_to_list(_broadcast_tensor.op.input_tensors[0].shape)
                if len(_src_shape) < len(_dst_shape):
                    _src_shape = [1] * (len(_dst_shape) - len(_src_shape)) + _src_shape
                _is_last_broadcast = False
                _broadcast_num = 0
                for _idx in range(len(_dst_shape)):
                    if not util.expr_equal(_src_shape[_idx], _dst_shape[_idx]):
                        _broadcast_num += 1
                        if _idx == len(_dst_shape) - 1:
                            _is_last_broadcast = True

                return _broadcast_num, _is_last_broadcast

            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in dependent_map:
                _current_space = len(dependent_map)
            else:
                _current_space = len(dependent_map) + 1
            if util.need_extent_node(_tensor):
                _current_space += 1
            # num of broadcast axis > 1 or float16 and last broadcast(normal sch or workspace sch but not AR)
            if util.is_unified_broadcast(_tensor):
                broadcast_num, is_last_broadcast = __get_broadcast_axis_info(_tensor)
                if _tensor.dtype == "float16" and is_last_broadcast and \
                        (not _is_workspace or (_is_workspace and not broadcast_num == 1)):
                    _current_space += 1
                    self.temp_ub_size += VNCHWCONV_TEMP_SPACE
                elif broadcast_num > 1:
                    _current_space += 1
            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE
            return _current_space

        def _r_coexisting(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            if util.is_vtranspose_broadcast(_tensor):
                self.temp_ub_size += VTRANSPOSE_TEMP_SPACE
            _need_space = []
            for _tensor_i in self.tensor_producers_map[_tensor]:
                _need_space.append(_r_coexisting(_tensor_i))

            _current_space = _calc_current_space(_tensor)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_tensor)

            _need_space.append(_current_space)
            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = self.tensor_consumers_map[_tensor].copy()

            return max(_need_space)

        def _r_coexisting_sub_graph(_tensor):
            if _tensor in dependent_map:
                return len(dependent_map)
            if util.is_vtranspose_broadcast(_tensor):
                self.temp_ub_size += VTRANSPOSE_TEMP_SPACE
            _need_space = []
            for _tensor_i in sub_tensor_producers_map[_tensor]:
                _need_space.append(_r_coexisting_sub_graph(_tensor_i))

            _current_space = _calc_current_space(_tensor, True)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_tensor)

            _need_space.append(_current_space)
            _refresh_dependent_sub_graph(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = sub_tensor_consumers_map[_tensor].copy()

            return max(_need_space)

        def _refresh_dependent(_tensor):
            for _tensor_i in self.tensor_producers_map[_tensor]:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _refresh_dependent_sub_graph(_tensor):
            for _tensor_i in sub_tensor_producers_map[_tensor]:
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

        coexisting_quantities = []
        dependent_map = {}
        _out = tuple(self.endpoint_output_tensor_set)[0]
        for tensor_i in self.tensor_producers_map[_out]:
            coexisting_quantities.append(_r_coexisting(tensor_i))
        if not _out.op.tag == FAKE_NODE_TAG:
            if util.is_vtranspose_broadcast(_out):
                self.temp_ub_size += VTRANSPOSE_TEMP_SPACE
            _local_current_space = _calc_current_space(_out)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_space)

        self.coexisting_quantity = max(coexisting_quantities)
        if self.coexisting_quantity == 1:
            self.temp_ub_size += BLOCK_SIZE_BYTE

        local_ub_size = self.soc_ub_size
        # delete tmp size
        local_ub_size -= self.temp_ub_size
        tensor_space = local_ub_size // self.coexisting_quantity
        self.available_ub_size = tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING[self.max_type]

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
                coexisting_quantities.append(_r_coexisting_sub_graph(tensor_i))
            if util.is_vtranspose_broadcast(workspace_tensor):
                self.temp_ub_size += VTRANSPOSE_TEMP_SPACE
            _local_current_space = _calc_current_space(workspace_tensor, True)
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
        self.workspace_available_min_ub_size = \
            min(sub_graph_available_ub_size_list) // DTYPE_BYTE_MAPPING[self.max_type]


class NormInfo:
    def __init__(self, compute_graph_info: NormComputeGraphInfo):
        self.graph_info: NormComputeGraphInfo = compute_graph_info
        # Assume all reduce node have the same shape, axis and keepdims
        self.reduce_tensor: Tensor = tuple(compute_graph_info.reduce_tensor_set)[0]
        self.all_axes: List[Var] = util.get_reduce_all_axes(self.reduce_tensor)
        self.reduce_axes: List[Var] = util.get_reduce_axes(self.reduce_tensor)

        self.shape_before_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.op.input_tensors[0].shape)
        self.shape_after_reduce: List[Union[Var, IntImm]] = list(self.reduce_tensor.shape)
        self.reduce_axis_indices: List[int] = sorted(util.get_reduce_axis_indexes(self.reduce_tensor))
        self.keepdims: bool = len(self.shape_before_reduce) == len(self.shape_after_reduce)
        self.is_reduce_last_axis: bool = len(self.shape_before_reduce) - 1 in self.reduce_axis_indices
        self.is_all_reduce: bool = len(self.shape_before_reduce) == len(self.reduce_axis_indices)
        self.is_none_reduce = True
        for reduce_axis in self.reduce_axis_indices:
            if util.shape_to_list(self.shape_before_reduce)[reduce_axis] != 1:
                self.is_none_reduce = False
        self.exist_partial_reorder = len(self.reduce_axis_indices) > 1 and not self.is_all_reduce


class NormTilingCase:
    def __init__(self):
        self.block_split_axis_index = None
        self.block_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core = None
        self.tiling_key = 2**31 - 1
        self.is_partial_reorder_case = False
