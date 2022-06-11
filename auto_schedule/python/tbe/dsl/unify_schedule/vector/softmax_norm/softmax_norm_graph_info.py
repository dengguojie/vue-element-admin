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
from tbe.common.utils.errormgr import get_error_message
from tbe.tvm.tensor import PlaceholderOp
from tbe.tvm.tensor import Tensor

from ... import util
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import SUPPORT_SCALAR_INSNS
from ...constants import TERNARY_INSNS

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
ALIGN_AND_REMOVE_PAD_EXTRA_NODES = 3
# temp space for enable align and remove pad with avoiding bank conflict
ALIGN_AND_REMOVE_PAD_TEMP_SPACE = 1024
# number extra nodes that enable no overlap 3
NO_OVERLAP_THREE_EXTRA_NODES = 2
# number extra nodes that enable broadcast not align
NO_ALIGN_BROADCAST_EXTRA_NODES = 2

DTYPE_AND_BLOCK_SIZE_MAP = {
    "bool": 32,
    "int8": 32,
    "uint8": 32,
    "float16": 16,
    "float32": 8,
    "int32": 8,
    "int64": 4,
}

DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP = {
    "bool": BLOCK * BLOCK * 4,
    "int8": BLOCK * BLOCK * 4,
    "uint8": BLOCK * BLOCK * 4,
    "float16": BLOCK * BLOCK,
    "float32": BLOCK * BLOCK // 2,
    "int32": BLOCK * BLOCK // 2,
    "int64": BLOCK * BLOCK // 4,
}


class MultiReduceGraphInfo:
    """
    compute graph info collection and crawling
    """

    def __init__(self, output_tensors_set: Iterable[Tensor]):
        self.core_num = get_soc_spec("CORE_NUM")
        self.soc_ub_size = get_soc_spec("UB_SIZE")
        self.output_tensor_set: Optional[Set[Tensor]] = None
        # real_output_tensor_set: output set doesn't contain fake node
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
        self.cross_hierarchy_tensor_set: Set[Tensor] = set()
        self.workspace_tensor_set: Set[Tensor] = set()
        self.workspace_and_reduce_tensor_set: Set[Tensor] = set()
        # extra workspace tensor and its info
        self.workspace_info_map: Dict[Tensor, Dict] = {}
        self.workspace_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}
        # extra split tensor and its sub_graph
        self.split_tensor_and_sub_graph_map: Dict[Tensor, Dict] = {}

        # before_reduce/after_reduce/other tensor set
        self.before_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.after_reduce_tensor_set: Optional[Set[Tensor]] = set()
        self.other_tensor_set: Optional[Set[Tensor]] = set()

        # max type and min type in graph
        self.max_type: Optional[str] = None
        self.min_type: Optional[str] = None
        # ub size
        self.temp_ub_size = 0
        self.broadcast_temp_size = 0
        self.available_ub_size = 0
        self.workspace_availabe_ub_size = 0
        self.pad_available_ub_size = 0
        self.reduce_transpose_available_ub_size = 0
        self.transpose_max_entire_size = 0
        self.const_db_size = 0
        self.max_coexisting_quantity = 0
        # exist vc unsupported type flag
        self.exist_vc_unsupported_type = False
        # do info collection
        self.collect_info(output_tensors_set)
        self.fake_node()
        self.get_tensors_before_and_after_reduce()
        self.find_workspace_tensor()
        self.calc_subgraph_coexisting_quantities()

    def collect_info(self, output_tensors: Iterable[Tensor]):
        """
        Collect base compute graph info

        Args:
            output_tensors (Iterable[Tensor]): _description_
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

    def find_workspace_tensor(self):
        """
        find workspace tensor
        """

        def _judge_workspace_tensor(cross_hierarchy_tensor):
            stop_tensor_set = cross_hierarchy_tensor_set - set([cross_hierarchy_tensor])
            producers_visited_list = []
            _traverse_tensor_list(cross_hierarchy_tensor, producers_visited_list, self.tensor_producers_map,
                                  stop_tensor_set)
            consumers_visited_list = []
            _traverse_tensor_list(cross_hierarchy_tensor, consumers_visited_list, self.tensor_consumers_map,
                                  stop_tensor_set)
            for count_tensor in (producers_visited_list + consumers_visited_list):
                if count_tensor in self.reduce_tensor_set:
                    return "workspace"
            return "non_workspace"

        cross_hierarchy_tensor_set = set()
        close_cross_hierarchy_tensor_set = set()
        # find all cross hierarchy tensors in compute graph
        for single_tensor in self.tensor_producers_map:
            if len(self.tensor_consumers_map.get(single_tensor)) > 1:
                cross_hierarchy_tensor_set.add(single_tensor)
            if len(self.tensor_producers_map.get(single_tensor)) > 1:
                close_cross_hierarchy_tensor_set.add(single_tensor)

        self.cross_hierarchy_tensor_set = cross_hierarchy_tensor_set
        self.close_cross_hierarchy_tensor_set = close_cross_hierarchy_tensor_set

        for single_tensor in cross_hierarchy_tensor_set - self.input_tensor_set:
            judge_result = _judge_workspace_tensor(single_tensor)
            if judge_result == "workspace":
                self.workspace_tensor_set.add(single_tensor)
                self.workspace_and_reduce_tensor_set.add(single_tensor)
                continue

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
            raise_error("not support output fork now")

    def calc_subgraph_coexisting_quantities(self):
        """
        calculate the number of coexisting quantities in sub subgraph
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
                    if all_axes[-1] not in reduce_axes:
                        self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                    else:
                        self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE + BLOCK_SIZE_BYTE + BLOCK_SIZE_BYTE
                elif dtype in ["float16", ]:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                else:
                    raise RuntimeError("Not support dtype in reduce_max(min) is %s" % dtype)

            def _reduce_prod_space():
                if all_axes[-1] not in reduce_axes:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE
                else:
                    self.temp_ub_size += COMMON_REDUCE_TEMP_SPACE + BLOCK_SIZE_BYTE + BLOCK_SIZE_BYTE

            if tag.find("reduce") != -1:
                reduce_axes = util.get_reduce_axes(_tensor)
                all_axes = util.get_reduce_all_axes(_tensor)
                if tag in ["reduce_sum"]:
                    _reduce_sum_space()
                elif tag in ["reduce_max", "reduce_min"]:
                    _reduce_max_space()
                elif tag in ["reduce_prod"]:
                    _reduce_prod_space()
                else:
                    raise RuntimeError("Unknown reduce_insn is %s" % tag)

        def _calc_current_coexist_node(_tensor, _is_workspace=False, _is_pad=False, _is_reduce_transpose=False):
            def __handle_input_and_output(__current_coexist_node):
                if _is_reduce_transpose:
                    pass
                elif _is_pad:
                    if _tensor in self.input_tensor_set:
                        __current_coexist_node += ALIGN_AND_REMOVE_PAD_EXTRA_NODES
                    if _tensor in self.real_output_tensor_set:
                        # the space that remove pad buffer need
                        _pad_space = __current_coexist_node + ALIGN_AND_REMOVE_PAD_EXTRA_NODES -\
                            len(self.tensor_producers_map.get(_tensor))
                        __current_coexist_node = max(__current_coexist_node, _pad_space)
                else:
                    if _tensor in self.real_output_tensor_set:
                        # the space that enable no_overlap 3 need
                        _no_overlap_three_space = __current_coexist_node + NO_OVERLAP_THREE_EXTRA_NODES -\
                            len(self.tensor_producers_map.get(_tensor))
                        __current_coexist_node = max(__current_coexist_node, _no_overlap_three_space)

                return __current_coexist_node

            def __handle_reduce(__current_coexist_node):
                if _is_reduce_transpose:
                    __current_coexist_node += REDUCE_TRANSPOSE_EXTRA_NODES
                elif "reduce_max" in _tensor.op.tag and _tensor.dtype == "float16":
                    __current_coexist_node += 1

                return __current_coexist_node

            def __handle_unified_broadcast(__current_coexist_node):
                __broadcast_axis = get_broadcast_axis(_tensor)
                __broadcast_num = len(__broadcast_axis)
                __tensor_shape = util.shape_to_list(_tensor.shape)
                __is_last_broadcast = len(__tensor_shape) - 1 in __broadcast_axis
                __last_dim_is_align = isinstance(__tensor_shape[-1], int) and \
                    __tensor_shape[-1] % get_block_size(self.min_type) == 0

                if _is_reduce_transpose and not __last_dim_is_align:
                    if __broadcast_num == 1:
                        __current_coexist_node += NO_ALIGN_BROADCAST_EXTRA_NODES
                    elif __broadcast_num > 1:
                        __current_coexist_node += 1

                    return __current_coexist_node

                if _is_workspace and __broadcast_num > 1:
                    __current_coexist_node += 1

                    return __current_coexist_node

                __is_enable_vnchwconv_fp16 = _tensor.dtype == "float16" and __is_last_broadcast
                __is_enable_vnchwconv_fp32 = _tensor.dtype == "float32" and __is_last_broadcast and __broadcast_num == 1
                if __is_enable_vnchwconv_fp16:
                    __current_coexist_node += 1
                    self.broadcast_temp_size = max(VNCHWCONV_TEMP_SPACE_FP16 + BLOCK_SIZE_BYTE,
                                                   self.broadcast_temp_size)
                elif __is_enable_vnchwconv_fp32:
                    self.broadcast_temp_size = max(VNCHWCONV_TEMP_SPACE_FP32 + BLOCK_SIZE_BYTE,
                                                   self.broadcast_temp_size)
                elif __broadcast_num > 1:
                    __current_coexist_node += 1

                return __current_coexist_node

            # one of the input of the ternary instruction must be reused with the output
            if util.get_dsl_insn(_tensor) in TERNARY_INSNS or _tensor in dependent_map:
                _current_coexist_node = len(dependent_map)
            else:
                _current_coexist_node = len(dependent_map) + 1

            if util.need_extent_node(_tensor):
                _current_coexist_node += 1

            if util.need_temp_space(_tensor) or _need_external_space(_tensor):
                self.temp_ub_size += BLOCK_SIZE_BYTE

            if _tensor in self.before_reduce_tensor_set:
                _current_coexist_node = __handle_input_and_output(_current_coexist_node)

            if _tensor in self.reduce_tensor_set:
                _current_coexist_node = __handle_reduce(_current_coexist_node)

            if util.is_unified_broadcast(_tensor):
                _current_coexist_node = __handle_unified_broadcast(_current_coexist_node)

            return _current_coexist_node

        def _r_coexisting(_tensor, _producer_map, _consumer_map, _is_workspace=False, _is_pad=False,
                          _is_reduce_transpose=False):
            if _tensor in dependent_map:
                return len(dependent_map)
            _coexist_node_list = []
            for _tensor_i in _producer_map.get(_tensor):
                _coexist_node_list.append(_r_coexisting(_tensor_i, _producer_map, _consumer_map,
                                                        _is_workspace, _is_pad, _is_reduce_transpose))

            _current_coexist_node = _calc_current_coexist_node(_tensor, _is_workspace=_is_workspace, _is_pad=_is_pad,
                                                               _is_reduce_transpose=_is_reduce_transpose)

            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_tensor)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_tensor)

            _coexist_node_list.append(_current_coexist_node)
            _refresh_dependent(_tensor, _producer_map)
            if _tensor not in dependent_map:
                dependent_map[_tensor] = _consumer_map.get(_tensor).copy()

            return max(_coexist_node_list)

        def _refresh_dependent(_tensor, _producer_map):
            for _tensor_i in _producer_map.get(_tensor):
                if _tensor_i not in dependent_map:
                    continue
                dependent_map.get(_tensor_i).remove(_tensor)
                if not dependent_map.get(_tensor_i):
                    dependent_map.pop(_tensor_i)

        def _need_external_space(_tensor):
            op_tag = util.get_dsl_insn(_tensor)
            support_vector_scalar_insns = ("elewise_binary_add", "elewise_binary_mul")
            if op_tag in set(SUPPORT_SCALAR_INSNS) - set(support_vector_scalar_insns):
                return True

            if util.is_v100() and op_tag in support_vector_scalar_insns and _tensor.dtype == "int32":
                return True

            return False

        def _refine_coexisting_quantity(_coexisting_quantity, _workspace_tensor):
            sub_tensor_list = self.workspace_tensor_and_sub_graph_map.get(_workspace_tensor).get("sub_tensor_list")
            for reduce_tensor in self.reduce_tensor_set:
                if reduce_tensor in sub_tensor_list:
                    _sub_tensor_list = self.split_tensor_and_sub_graph_map.get(reduce_tensor).get("sub_tensor_list")
                    _sub_tensor_producers_map = self.split_tensor_and_sub_graph_map.get(reduce_tensor).get(
                        "sub_tensor_producers_map")
                    if not judge_before_reduce_tensor_cross_hierarchy(_sub_tensor_list, _sub_tensor_producers_map):
                        _coexisting_quantity += 1

            return _coexisting_quantity

        def judge_before_reduce_tensor_cross_hierarchy(_sub_tensor_list, _produce_map):
            for single_tensor in _sub_tensor_list:
                if len(_produce_map.get(single_tensor)) > 1:
                    return True
            return False

        _out = self.endpoint_output_tensor
        self.transpose_max_entire_size = max(
            get_transpose_entire_size(self.min_type) * DTYPE_BYTE_MAPPING.get(
                self.min_type) // DTYPE_BYTE_MAPPING.get(self.max_type), get_transpose_entire_size(self.max_type))
        # common sch
        coexisting_quantities = []
        dependent_map = {}
        for tensor_i in self.tensor_producers_map.get(_out):
            coexisting_quantities.append(_r_coexisting(tensor_i, self.tensor_producers_map,
                                                       self.tensor_consumers_map))
        if not _out.op.tag == FAKE_NODE_TAG:
            _local_current_coexist_node = _calc_current_coexist_node(_out)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_coexist_node)
        tensor_space = (self.soc_ub_size - self.temp_ub_size - self.broadcast_temp_size) // max(coexisting_quantities)
        self.available_ub_size =\
            tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self.max_type)
        const_db_tensor_space =\
            (self.soc_ub_size // 2 - self.temp_ub_size - self.broadcast_temp_size) // max(coexisting_quantities)
        self.const_db_size =\
            const_db_tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self.max_type)

        # align and remove pad sch
        coexisting_quantities = []
        dependent_map = {}
        self.temp_ub_size = 0
        self.broadcast_temp_size = 0
        for tensor_i in self.tensor_producers_map.get(_out):
            coexisting_quantities.append(_r_coexisting(tensor_i, self.tensor_producers_map, self.tensor_consumers_map,
                                                       _is_pad=True))
        if not _out.op.tag == FAKE_NODE_TAG:
            _local_current_coexist_node = _calc_current_coexist_node(_out, _is_pad=True)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_coexist_node)
        tensor_space = (self.soc_ub_size - self.temp_ub_size - self.broadcast_temp_size -
                        ALIGN_AND_REMOVE_PAD_TEMP_SPACE) // max(coexisting_quantities)
        self.pad_available_ub_size =\
            tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self.max_type)

        # reduce transpose sch
        coexisting_quantities = []
        dependent_map = {}
        self.temp_ub_size = 0
        self.broadcast_temp_size = 0
        for tensor_i in self.tensor_producers_map.get(_out):
            coexisting_quantities.append(_r_coexisting(tensor_i, self.tensor_producers_map, self.tensor_consumers_map,
                                                       _is_reduce_transpose=True))
        if not _out.op.tag == FAKE_NODE_TAG:
            _local_current_coexist_node = _calc_current_coexist_node(_out, _is_reduce_transpose=True)
            # correct ub size in vcmp or vsel or vcmpsel
            _correct_ub_size_by_cmp_sel(_out)
            # correct ub size in reduce
            _correct_ub_size_by_reduce(_out)
            coexisting_quantities.append(_local_current_coexist_node)
        tensor_space = (self.soc_ub_size - self.temp_ub_size - self.broadcast_temp_size) // max(coexisting_quantities)
        self.reduce_transpose_available_ub_size =\
            tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(self.max_type)

        # workspace sch
        # calculate the number of coexisting quantities and available ub size of sub graph
        coexisting_quantities_list = []
        for workspace_tensor in self.workspace_tensor_and_sub_graph_map:
            sub_tensor_producers_map = \
                self.workspace_tensor_and_sub_graph_map.get(workspace_tensor).get("sub_tensor_producers_map")
            sub_tensor_consumers_map = \
                self.workspace_tensor_and_sub_graph_map.get(workspace_tensor).get("sub_tensor_consumers_map")
            coexisting_quantities = []
            dependent_map = {}
            self.temp_ub_size = 0
            for tensor_i in sub_tensor_producers_map.get(workspace_tensor):
                coexisting_quantities.append(_r_coexisting(tensor_i, sub_tensor_producers_map,
                                                           sub_tensor_consumers_map, _is_workspace=True))
            _local_current_coexist_node = _calc_current_coexist_node(workspace_tensor, _is_workspace=True)
            _correct_ub_size_by_cmp_sel(workspace_tensor)
            _correct_ub_size_by_reduce(workspace_tensor)
            coexisting_quantities.append(_local_current_coexist_node)
            coexisting_quantity = _refine_coexisting_quantity(max(coexisting_quantities), workspace_tensor)
            if coexisting_quantity == 1:
                self.temp_ub_size += BLOCK_SIZE_BYTE
            self.workspace_info_map[workspace_tensor] = {
                "temp_ub_size": self.temp_ub_size,
                "coexisting_quantity": coexisting_quantity
            }
            coexisting_quantities_list.append(coexisting_quantity)
        self.max_coexisting_quantity = max(coexisting_quantities_list)
        sub_graph_available_ub_size_list = []
        for tensor in self.workspace_info_map:
            local_ub_size = self.soc_ub_size - self.workspace_info_map.get(tensor).get("temp_ub_size")
            tensor_space = local_ub_size // self.workspace_info_map.get(tensor).get("coexisting_quantity")
            sub_graph_available_ub_size_list.append(tensor_space // BLOCK_SIZE_BYTE * BLOCK_SIZE_BYTE)
        self.workspace_available_ub_size = \
            min(sub_graph_available_ub_size_list) // DTYPE_BYTE_MAPPING.get(self.max_type)


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


def get_transpose_entire_size(dtype):
    """
    get pad entire size according to dtype.
    """
    if dtype not in DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP:
        raise_error("[%s] is not support type in norm" % dtype)

    return DTYPE_AND_TRANSPOSE_ENTIRE_SIZE_MAP.get(dtype)


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
