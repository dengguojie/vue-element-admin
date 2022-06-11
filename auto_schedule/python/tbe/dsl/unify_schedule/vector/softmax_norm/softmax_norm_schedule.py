from tbe import tvm

from tbe.dsl.base.operation import get_context
from tbe.dsl.base.operation import var_inner
from tbe.dsl.base.operation import add_compile_info_inner

from ... import util
from ...constants import DTYPE_BYTE_MAPPING
from ...constants import FAKE_NODE_TAG
from ...constants import INSN_MAPPING
from .softmax_norm_graph_info import MultiReduceGraphInfo

LOCAL_UB = "local.UB"
BLOCK_IDX = "blockIdx.x"
DMA_COPY = "dma_copy"
PHONY_INSN = "phony_insn"
BLOCK_SIZE_BYTE = 32


def SoftmaxNormSchedule(outs, tiling_case):
    """_summary_
    schedule enterance

    Args:
        outs (_type_): _description_
        tiling_case (_type_): _description_
    """
    softmaxnorm_compute_graph_info = MultiReduceGraphInfo(outs)
    softmaxnorm_info = None
    # ub split on A axis or don't split ub
    sch_class = NormalSchedule(softmaxnorm_compute_graph_info, softmaxnorm_info, tiling_case, outs)

    sch = sch_class.do_schedule()
    # add tiling key to schedule object
    sch.tiling_key = tiling_case.get("key")
    return sch


class NormalSchedule:
    def __init__(self, graph_info, softmax_norm_info, tiling_case, outs):
        self._graph_info = graph_info
        self._softmax_norm_info = softmax_norm_info
        self._tiling_case = tiling_case
        self._is_align = tiling_case.get("is_align")
        self._outs = outs

        # cache read relation dictionary
        self._cache_read_tensor_map = dict()
        self._cache_read_buffer_to_tensor_map = dict()
        self._cache_read_tensor_to_buffer_map = dict()

        # cache write relation set and dictionary
        self._cache_write_tensor_set = set()
        self._cache_write_buffer_to_tensor_map = dict()
        self._cache_write_tensor_to_buffer_map = dict()

        # do tiling
        self._block_split_result_map = dict()
        self._ub_split_result_map = dict()
        self._multi_core_map = dict()

        # compute_at dictionary
        self._compute_at_map = dict()
        self._emit_insn_map = dict()

        # storage_align
        self._storage_align_map = dict()

        # init some func
        self._graph_info_parse()
        self._is_const = False

        # reused by part
        self._reused_by_list = []

        # common class paramter
        self._sch = None

    def _graph_info_parse(self):
        """
        parse compute graph information
        """
        self._input_tensor_set = self._graph_info.input_tensor_set
        self._mid_output_tensor_set = self._graph_info.mid_output_tensor_set
        self._output_tensor_set = self._graph_info.output_tensor_set
        self._real_output_tensor_set = self._graph_info.real_output_tensor_set
        self._real_pure_output_tensor_set = self._graph_info.real_pure_output_tensor_set
        self._tensor_consumers_map = self._graph_info.tensor_consumers_map
        self._tensor_producers_map = self._graph_info.tensor_producers_map
        self._reduce_tensor_set = self._graph_info.reduce_tensor_set
        self._broadcast_tensor_set = self._graph_info.broadcast_tensor_set
        self._mid_tensor_set = self._graph_info.mid_tensor_set
        self._endpoint_output_tensor = self._graph_info.endpoint_output_tensor
        self._available_size = self._graph_info.available_ub_size
        self._core_num = self._graph_info.core_num

    def do_schedule(self):
        """
        normal case schedule operate process
        """
        # parse graph info
        self._do_create_schedule()
        self._do_set_var_range()

        self._calc_cache_read()
        self._do_cache_read()
        self._calc_cache_write()
        self._do_cache_write()

        self._calc_reused_by()
        self._do_reused_by()

        self._do_set_scope()
        self._do_set_buffer_size()

        self._calc_compute_inline()
        self._do_compute_inline()

        self._do_tiling()

        self._calc_multi_core()
        self._do_multi_core()

        self._calc_compute_at()
        self._do_compute_at()

        self._calc_storage_align()
        self._do_storage_align()

        self._calc_emit_insn()
        self._do_emit_insn()
        self._add_compile_info()
        return self._sch

    def _do_create_schedule(self):
        # create schedule object
        self._sch = tvm.create_schedule([self._endpoint_output_tensor.op])

    def _do_set_var_range(self):
        pass

    def _calc_cache_read(self):
        """
        calc input tensor about cache read information
        """
        for input_tensor in self._input_tensor_set.union(self._mid_output_tensor_set):
            self._cache_read_tensor_map[input_tensor] = self._tensor_consumers_map.get(input_tensor)

    def _do_cache_read(self):
        """
        do cache read
        """
        for read_tensor, consumer_tensor_set in self._cache_read_tensor_map.items():
            read_buffer = self._sch.cache_read(read_tensor, LOCAL_UB, consumer_tensor_set)
            self._cache_read_buffer_to_tensor_map[read_buffer] = read_tensor
            self._cache_read_tensor_to_buffer_map.setdefault(read_tensor, set()).add(read_buffer)

    def _calc_cache_write(self):
        """
        calc output tensor about cache write information
        """
        self._cache_write_tensor_set.update(self._real_output_tensor_set)

    def _do_cache_write(self):
        """
        do cache write
        """
        for write_tensor in self._cache_write_tensor_set:
            write_buffer = self._sch.cache_write(write_tensor, LOCAL_UB)
            self._cache_write_buffer_to_tensor_map[write_buffer] = write_tensor
            self._cache_write_tensor_to_buffer_map.setdefault(write_tensor, set()).add(write_buffer)

    def _calc_reused_by(self):
        for mid_output_tensor in self._mid_output_tensor_set:
            self._reused_by_list.append([list(self._cache_write_tensor_to_buffer_map.get(mid_output_tensor))[0],
                                         list(self._cache_read_tensor_to_buffer_map.get(mid_output_tensor))[0]])

    def _do_reused_by(self):
        for reuse_param in self._reused_by_list:
            self._sch[reuse_param[0]].reused_by(reuse_param[1])
            self._sch[reuse_param[1]].reused_by(reuse_data=True)

    def _do_set_scope(self):
        """
        middle tensor set scope
        """
        for tensor in (self._mid_tensor_set - self._input_tensor_set - self._real_output_tensor_set):
            self._sch[tensor].set_scope(LOCAL_UB)

    def _do_set_buffer_size(self):
        """
        tensor set buffer size on local ub
        """
        for tensor in self._mid_tensor_set.union(self._cache_read_buffer_to_tensor_map.keys()) \
                .union(self._cache_write_buffer_to_tensor_map.keys()):
            storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING.get(self._graph_info.max_type) // \
                DTYPE_BYTE_MAPPING.get(tensor.dtype)
            self._sch[tensor].set_buffer_size(storage_bound_value)

        if self._endpoint_output_tensor.op.tag == FAKE_NODE_TAG:
            storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING.get(self._graph_info.max_type) // \
                DTYPE_BYTE_MAPPING.get(self._endpoint_output_tensor.dtype)
            self._sch[self._endpoint_output_tensor].set_buffer_size(storage_bound_value)

    def _calc_compute_inline(self):
        pass

    def _do_compute_inline(self):
        pass

    def _do_tiling(self):
        """
        do tiling strategy
        """
        block_split_axis_index = self._tiling_case.get("block_tiling_axis")
        ub_split_axis_index = self._tiling_case.get("ub_tiling_axis")
        block_nparts = self._tiling_case.get("block_nparts")
        ub_factor = self._tiling_case.get("ub_factor")

        block_split_nparts = block_nparts if self._is_const else var_inner("_block_nparts", (1, self._core_num))
        ub_split_factor = ub_factor if self._is_const else var_inner("_ub_factor", (1, self._available_size))

        ub_outer, ub_inner = self._sch[self._endpoint_output_tensor].split(
            self._endpoint_output_tensor.op.axis[ub_split_axis_index], factor=ub_split_factor)
        if block_split_axis_index == ub_split_axis_index:
            block_outer, block_inner = self._sch[self._endpoint_output_tensor].split(
                ub_outer, nparts=block_split_nparts)
            ub_outer = block_inner
        else:
            block_outer, block_inner = self._sch[self._endpoint_output_tensor].split(
                self._endpoint_output_tensor.op.axis[block_split_axis_index], nparts=block_split_nparts)

        self._block_split_result_map = {
            "split_axis": block_split_axis_index,
            "outer_itervar": block_outer,
            "inner_itervar": block_inner,
        }

        self._ub_split_result_map = {
            "split_axis": ub_split_axis_index,
            "outer_itervar": ub_outer,
            "inner_itervar": ub_inner,
        }

    def _calc_multi_core(self):
        multi_core_bind_itervar = self._block_split_result_map.get("outer_itervar")
        self._multi_core_map[self._endpoint_output_tensor] = multi_core_bind_itervar

    def _do_multi_core(self):
        """
        do multi core
        """
        block = tvm.thread_axis(BLOCK_IDX)
        for tensor, bind_itervar in self._multi_core_map.items():
            self._sch[tensor].bind(bind_itervar, block)

    def _calc_compute_at(self):
        """
        calc tensor compute at information
        """
        for tensor in self._tensor_consumers_map.keys():
            if tensor in (self._output_tensor_set - self._mid_output_tensor_set).union(self._input_tensor_set):
                continue
            self._compute_at_map[tensor] = {
                "root_tensor": self._endpoint_output_tensor,
                "loop_itervar": self._ub_split_result_map.get("outer_itervar"),
            }

        for read_buffer in self._cache_read_buffer_to_tensor_map.keys():
            self._compute_at_map[read_buffer] = {
                "root_tensor": self._endpoint_output_tensor,
                "loop_itervar": self._ub_split_result_map.get("outer_itervar"),
            }

        for write_buffer in self._cache_write_buffer_to_tensor_map.keys():
            self._compute_at_map[write_buffer] = {
                "root_tensor": self._endpoint_output_tensor,
                "loop_itervar": self._ub_split_result_map.get("outer_itervar"),
            }

    def _do_compute_at(self):
        """
        do compute at
        """
        for compute_tensor, value in self._compute_at_map.items():
            self._sch[compute_tensor].compute_at(self._sch[value["root_tensor"]], value["loop_itervar"])

    def _calc_storage_align(self):
        # judge whether do storage align
        if self._is_align:
            return

        for tensor in set(self._tensor_consumers_map.keys()).union(self._cache_read_buffer_to_tensor_map.keys()) \
                .union(self._cache_write_buffer_to_tensor_map.keys()):
            if tensor in self._input_tensor_set.union(self._real_output_tensor_set).union(self._reduce_tensor_set) \
                    or len(tensor.shape) < 2:
                continue
            # whether input last dim is 1, not do storage_align to optimize dma_copy
            condition1 = isinstance(tensor.shape[-1], tvm.expr.IntImm) and (int(tensor.shape[-1]) == 1)
            if condition1:
                continue
            per_block_size = BLOCK_SIZE_BYTE // DTYPE_BYTE_MAPPING.get(tensor.dtype)
            self._storage_align_map[tensor] = {"align_axis": -2, "align_num": per_block_size}

    def _do_storage_align(self):
        for tensor, value in self._storage_align_map.items():
            self._sch[tensor].storage_align(tensor.op.axis[value["align_axis"]], value["align_num"], 0)

    def _calc_emit_insn(self):
        """
        calc tensor emit insn
        """
        emit_insn_axis = 0
        for tensor in self._tensor_consumers_map.keys():
            # get emit itervar and
            emit_attr = None
            if tensor == self._endpoint_output_tensor:
                emit_itervar = self._ub_split_result_map.get("inner_itervar")
                if tensor.op.tag == FAKE_NODE_TAG:
                    emit_insn = PHONY_INSN
                else:
                    emit_insn = DMA_COPY
                    emit_attr = {"no_overlap": "process_unaliged_stride_with_malloc_buf",
                                 "no_overlap_malloc_buf_for_tail": 0}
            elif tensor in self._input_tensor_set:
                continue
            elif tensor in self._real_output_tensor_set:
                emit_itervar = tensor.op.axis[emit_insn_axis]
                emit_insn = DMA_COPY
                emit_attr = {"no_overlap": "process_unaliged_stride_with_malloc_buf",
                             "no_overlap_malloc_buf_for_tail": 0}
            else:
                emit_itervar = tensor.op.axis[emit_insn_axis]
                emit_insn = util.get_dsl_insn(tensor)
                storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING.get(self._graph_info.max_type) \
                    // DTYPE_BYTE_MAPPING.get(tensor.dtype)
                emit_attr = dict(storage_bound=storage_bound_value) if tensor in self._reduce_tensor_set else emit_attr

            self._emit_insn_map[tensor] = {
                "emit_itervar": emit_itervar,
                "emit_insn": INSN_MAPPING.get(emit_insn, emit_insn),
                "emit_attr": emit_attr,
            }

        for read_buffer in self._cache_read_buffer_to_tensor_map.keys():
            read_tensor = self._cache_read_buffer_to_tensor_map.get(read_buffer)
            if read_tensor in self._mid_output_tensor_set:
                emit_insn = PHONY_INSN
            else:
                emit_insn = DMA_COPY
            self._emit_insn_map[read_buffer] = {
                "emit_itervar": read_buffer.op.axis[emit_insn_axis],
                "emit_insn": INSN_MAPPING.get(emit_insn, emit_insn),
                "emit_attr": {"gm_to_ub_gap_opt": 1},
            }

        for write_buffer in self._cache_write_buffer_to_tensor_map.keys():
            gm_tensor = self._cache_write_buffer_to_tensor_map.get(write_buffer)
            emit_insn = util.get_dsl_insn(write_buffer)
            storage_bound_value = self._available_size * DTYPE_BYTE_MAPPING.get(self._graph_info.max_type) \
                // DTYPE_BYTE_MAPPING.get(tensor.dtype)
            emit_attr = dict(storage_bound=storage_bound_value) if gm_tensor in self._reduce_tensor_set else emit_attr
            self._emit_insn_map[write_buffer] = {
                "emit_itervar": write_buffer.op.axis[emit_insn_axis],
                "emit_insn": INSN_MAPPING.get(emit_insn, emit_insn),
                "emit_attr": None,
            }

    def _do_emit_insn(self):
        """
        do emit insn
        """
        for tensor, value in self._emit_insn_map.items():
            emit_attr = value.get("emit_attr")
            if emit_attr:
                self._sch[tensor].emit_insn(value["emit_itervar"], value["emit_insn"], emit_attr)
            else:
                self._sch[tensor].emit_insn(value["emit_itervar"], value["emit_insn"])

    def _add_compile_info(self):
        add_compile_info_inner("_max_type", self._graph_info.max_type)
        add_compile_info_inner("_min_type", self._graph_info.min_type)
        add_compile_info_inner("_core_num", self._core_num)
        add_compile_info_inner("_available_size", 6544)
        add_compile_info_inner("_align_base_key", 10000)
        add_compile_info_inner("_is_const", self._is_const)
        add_compile_info_inner("_is_template", True)
