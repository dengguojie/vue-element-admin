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
Information containers for compute graph of vector operator schedule
"""

# Standard Packages
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
from tbe.dsl.base import operation
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor

from .constants import *
from .util import equals_one
from .util import expr_equal
from .util import get_reduce_all_axes
from .util import get_reduce_axes
from .util import is_placeholder
from .util import shape_to_list


class ComputeGraphInfo:
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
        # For ReduceSch(only)
        self.tensor_ub_size_before_reduce: Optional[int] = None
        self.tensor_ub_size_after_reduce: Optional[int] = None
        self.tensors_before_reduce: Optional[List[Tensor]] = []
        self.tensors_after_reduce: Optional[List[Tensor]] = []
        self.max_type: Optional[str] = None
        self.min_type: Optional[str] = None
        self.coef: Optional[int] = None
        # Do info collection
        self._collect_info(output_tensors)
        self._fake_node()
        self._current_support_check()
        self._init_max_ub_count()

    def _collect_info(self, output_tensors: Iterable[Tensor]):
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

    def _fake_node(self):
        # after _collect_info(outer_most), middle_output_tensors has been assured.
        # middle_output_tensors: connect to next node in graph,
        # fake_node doesn't need them as producers for it

        def _fake_node_compute(tensors):
            # fake_node must be the biggest node (type and shape)
            dtype = tensors[0].dtype
            dim_length = max([len(t.shape) for t in tensors])
            shape = [1] * dim_length

            # update fake_node's shape and dtype
            for tensor_i in tensors:
                if DTYPE_BYTE_MAPPING[tensor_i.dtype] > DTYPE_BYTE_MAPPING[dtype]:
                    dtype = tensor_i.type
                shape_i = shape_to_list(tensor_i.shape)
                diff = dim_length - len(shape_i)
                shape_i = [1] * diff + shape_i
                for j in range(diff, dim_length):
                    if equals_one(shape[j]):
                        shape[j] = shape_i[j]
                    elif not expr_equal(shape[j], shape_i[j]) and not equals_one(shape_i[j]):
                        # ReduceSch couldn't in the branch, while ReduceSch not support broadcast
                        shape[j] = tvm.max(shape_i[j], shape[j])

            def _compute(*indices):
                res_ = tvm.const(1, dtype)
                for tensor in tensors:
                    cur_indices = []
                    for idx, dim in enumerate(tensor.shape):
                        if equals_one(dim):
                            cur_indices.append(0)
                        else:
                            cur_indices.append(indices[idx])
                    res_ *= tvm.expr.Cast(dtype, tensor(*cur_indices))
                return res_

            with tvm.tag_scope(FAKE_NODE_TAG):
                res = tvm.compute(shape, _compute, name="fake_node")

            return res

        self.real_output_tensor_set = self.output_tensor_set
        pure_out_tensors = list(self.real_output_tensor_set - self.mid_output_tensor_set)
        if len(pure_out_tensors) > 1:
            _out = _fake_node_compute(pure_out_tensors)
            # update info while last_node is fake_node
            self._collect_info([_out, ])

    def _current_support_check(self):
        _shape = list(list(self.real_output_tensor_set)[0].shape)
        for _tensor in self.real_output_tensor_set:
            if list(_tensor.shape) != _shape:
                raise RuntimeError("Dynamic ReduceSchedule does "
                                   "not support outputs with different shapes")

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
    def _eq_tvm_shape(shapeA: List, shapeB: List):
        length_a = len(shapeA)
        length_b = len(shapeB)
        if length_a != length_b:
            return False
        for idx, _ in enumerate(range(length_a)):
            ret_value = hasattr(shapeA[idx], "value") and hasattr(shapeB[idx], "value")
            ret_name = hasattr(shapeA[idx], "name") and hasattr(shapeB[idx], "name")
            if ret_value:
                if shapeA[idx].value != shapeB[idx].value:
                    return False
            elif ret_name:
                if shapeA[idx].name != shapeB[idx].name:
                    return False
            else:
                if shapeA[idx] != shapeB[idx]:
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
        else:
            raise RuntimeError("dfs_compute_graph() supports list, tuple, Tensor only. Received %s"
                               % str(type(root_tensor)))
        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def get_all_tensors_before_reduce(self):
        """
        Strategy assumes:
        1. the graph not exist "broadcast".
        2. only support single reduce node.
        """
        reduce_tensor = list(self.reduce_tensor_set)[0]
        shape_after_reduce = list(reduce_tensor.shape)
        compute = operation.get_context().get_current_compute()
        if compute.get("_mode") == "zero" and compute.get("_shape") == (1, -1, 0):
            shape_before_reduce = list(reduce_tensor.shape) + [tvm.expr.IntImm("int32", 0)]
        else:
            shape_before_reduce = list(reduce_tensor.op.input_tensors[0].shape)

        self.max_type = self.tensor_list[0].dtype
        self.min_type = self.tensor_list[0].dtype

        for item in self.tensor_list:
            if DTYPE_BYTE_MAPPING[item.dtype] > DTYPE_BYTE_MAPPING[self.max_type]:
                self.max_type = item.dtype
            elif DTYPE_BYTE_MAPPING[item.dtype] < DTYPE_BYTE_MAPPING[self.min_type]:
                self.min_type = item.dtype

            if self._eq_tvm_shape(list(item.shape), shape_before_reduce):
                self.tensors_before_reduce.append(item)
            elif self._eq_tvm_shape(list(item.shape), shape_after_reduce):
                self.tensors_after_reduce.append(item)
            else:
                raise RuntimeError("unknown tensor")
        if not self.tensors_after_reduce:
            # pure data_move pattern
            self.tensors_after_reduce = self.tensors_before_reduce

    def _init_max_ub_count(self):
        """
        Func: Get the biggest value in node_exist_list that from different sub_graph.
              tensors_before_reduce belong to bNode, tensors_after_reduce belong to sNode.
              sizeof(bNode) = coef * sizeof(sNode)
        """
        def _analysis_dependent(dependent):
            _dict = {"_bNodeNum": [], "_sNodeNum": []}
            for item in dependent.keys():
                if item in self.tensors_before_reduce:
                    _dict["_bNodeNum"].append(item.dtype)
                elif item in self.tensors_after_reduce:
                    _dict["_sNodeNum"].append(item.dtype)
                else:
                    raise RuntimeError("tensor_i is not in discrimination.")
            return _dict

        def _current_compute_need_space(_tensor, _dict):
            tag = _tensor.op.tag
            dtype = _tensor.dtype

            def _reduce_sum_space(_reduce_tensor):
                if all_axes[-1] not in reduce_axes:
                    _dict["_sNodeNum"].append(dtype)
                    self.soc_ub_size -= 64
                else:
                    if len(reduce_axes) == 1:
                        _dict["_sNodeNum"].append(dtype)
                        self.soc_ub_size -= 4096
                    else:
                        _dict["_sNodeNum"].append(dtype)
                        _dict["_bNodeNum"].append(dtype)
                        self.soc_ub_size -= 2048

            def _reduce_max_space(_reduce_tensor):
                if dtype in ["float32", "int32"]:
                    if all_axes[-1] not in reduce_axes:
                        _dict["_sNodeNum"].append(dtype)
                        self.soc_ub_size -= 64
                    else:
                        if len(reduce_axes) == 1:
                            _dict["_sNodeNum"].append(dtype)
                            self.soc_ub_size -= 512
                        else:
                            self.soc_ub_size -= 512
                            _dict["_sNodeNum"].append(dtype)
                            _dict["_bNodeNum"].append(dtype)
                elif dtype in ["float16", ]:
                    if all_axes[-1] not in reduce_axes:
                        _dict["_sNodeNum"].append(dtype)
                        self.soc_ub_size -= 64
                    else:
                        _dict["_sNodeNum"].append(dtype)
                        _dict["_bNodeNum"].append(dtype)
                        self.soc_ub_size -= 4096
                else:
                    raise RuntimeError("Not support dtype in reduce_max(min) is %s" % dtype)

            def _reduce_prod_space(_reduce_tensor):
                if all_axes[-1] not in reduce_axes:
                    _dict["_sNodeNum"].append(dtype)
                    self.soc_ub_size -= 64
                else:
                    if len(reduce_axes) == 1:
                        _dict["_sNodeNum"].append(dtype)
                        self.soc_ub_size -= 512
                    else:
                        self.soc_ub_size -= 512
                        _dict["_sNodeNum"].append(dtype)
                        _dict["_bNodeNum"].append(dtype)

            if tag.find("reduce") != -1:
                reduce_axes = get_reduce_axes(_tensor)
                all_axes = get_reduce_all_axes(_tensor)
                if tag in ["reduce_sum", ]:
                    _reduce_sum_space(_tensor)
                elif tag in ["reduce_max", "reduce_min"]:
                    _reduce_max_space(_tensor)
                elif tag in ["reduce_prod", ]:
                    _reduce_prod_space(_tensor)
                else:
                    raise RuntimeError("Unknown reduce_insn is %s" % tag)
            else:
                if _tensor in self.tensors_before_reduce:
                    _dict["_bNodeNum"].append(dtype)
                else:
                    _dict["_sNodeNum"].append(dtype)

        def _refresh_dependent(_tensor):
            for _tensor_i in _tensor.op.input_tensors:
                if _tensor_i not in dependent_map:
                    continue
                dependent_map[_tensor_i].remove(_tensor)
                if not dependent_map[_tensor_i]:
                    dependent_map.pop(_tensor_i)

        def _r_coexisting(_tensor, _need_space):
            if _tensor in dependent_map:
                _need_space.append(_analysis_dependent(dependent_map))
                return

            for _tensor_i in _tensor.op.input_tensors:
                _r_coexisting(_tensor_i, _need_space)

            curr_dict = _analysis_dependent(dependent_map)
            _current_compute_need_space(_tensor, curr_dict)
            _need_space.append(curr_dict)

            _refresh_dependent(_tensor)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = self.tensor_consumers_map[_tensor].copy()

        def _get_max(_node_list, reduce_type, coefficients=1):
            # _node is [{},{},...,{}]
            # node belong to _sNodeNum that has space "1*DTYPE_BYTE_MAPPING[dtype]"
            # node belong to _bNodeNum that has space "coefficients*DTYPE_BYTE_MAPPING[dtype]"
            def _calc_value(_node):
                value = 0
                for item in _node.get("_sNodeNum"):
                    value += DTYPE_BYTE_MAPPING[item]
                for item in _node.get("_bNodeNum"):
                    value += coefficients * DTYPE_BYTE_MAPPING[item]
                return value

            max_item = _node_list[0]
            max_value = _calc_value(_node_list[0])
            for item in _node_list:
                item_value = _calc_value(item)
                if max_value < item_value:
                    max_value = item_value
                    max_item = item
                elif max_value == item_value and not item.get("_sNodeNum"):
                    # reduce maybe init in begin of graph
                    max_value = item_value
                    max_item = item
            # reduce maybe init in begin of graph
            if not max_item.get("_sNodeNum"):
                max_value += DTYPE_BYTE_MAPPING[reduce_type]
                max_item["_sNodeNum"].append(reduce_type)

            return max_item, max_value

        # [Common Reduce] Find maximum sub_graph
        self.get_all_tensors_before_reduce()
        if not operation.get_context().get("_placeholder_before_reduce") and \
                not operation.get_context().get("_placeholder_after_reduce"):
            placeholder_before_reduce = [x for x in self.tensors_before_reduce if is_placeholder(x)]
            operation.get_context().add("_placeholder_before_reduce", placeholder_before_reduce)
            placeholder_after_reduce = [x for x in self.tensors_after_reduce if is_placeholder(x)]
            operation.get_context().add("_placeholder_after_reduce", placeholder_after_reduce)
        coexisting_quantities, dependent_map = [], {}
        _out = list(self.output_tensor_set)[0]

        for tensor_i in _out.op.input_tensors:
            _r_coexisting(tensor_i, coexisting_quantities)

        # [Common Reduce] Multi outputs
        if not _out.op.tag == FAKE_NODE_TAG:
            curr_dict = _analysis_dependent(dependent_map)
            _current_compute_need_space(_out, curr_dict)
            coexisting_quantities.append(curr_dict)

        # [Common Reduce] coefficients between bNode and sNode
        self.coef = 1
        if self.tensors_before_reduce == self.tensors_after_reduce:
            # pure data_move
            self.coef = 1

        # [Single Reduce] ReduceNode may be initialized in the begin of graph
        reduce_tensors = list(self.reduce_tensor_set)
        reduce_type = reduce_tensors[0].dtype
        curr_dict, total_num = _get_max(coexisting_quantities, reduce_type, coefficients=self.coef)

        # [Single Reduce] Find roots of reduce
        def _r_parent(_child, _parent):
            if not self.tensor_producers_map.get(_child):
                # reduce_i isn't connect to input directly
                if reduce_i not in self.tensor_consumers_map.get(_child):
                    _parent.append(_child)
                return
            for _tensor in self.tensor_producers_map.get(_child):
                _r_parent(_tensor, _parent)

        reduce_parents = []
        for reduce_i in reduce_tensors:
            _r_parent(reduce_i, reduce_parents)
        reduce_parents = list(set(reduce_parents))

        for parent_i in reduce_parents:
            # Reserve space for inputs' "db"
            curr_dict["_bNodeNum"].append(parent_i.dtype)
            total_num += DTYPE_BYTE_MAPPING[parent_i.dtype] * self.coef

        # [Common Reduce] avoid bank conflict in malloc ub
        self.soc_ub_size -= 1024
        small_ub_size = self.soc_ub_size // total_num // 128 * 128
        self.tensor_ub_size_before_reduce = self.coef * small_ub_size
        self.tensor_ub_size_after_reduce = small_ub_size
        operation.add_compile_info_inner("_zero_ub_factor", self.tensor_ub_size_after_reduce)

    @staticmethod
    def set_map_deepcopy(_map: Dict[Tensor, Set[Tensor]]) -> Dict[Tensor, Set[Tensor]]:
        return dict((key, _map[key].copy()) for key in _map)
