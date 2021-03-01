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
elewise mutil out schedule
"""
import math
from tbe import tvm
from te import platform as cceconf
from te.platform import log
from . import util
from .elewise_schedule_new import ElewiseSchedule


# pylint: disable=too-many-return-statements,too-few-public-methods,too-many-arguments,too-many-statements,no-self-use,too-many-lines,too-many-instance-attributes,too-many-branches,
class ElewiseMultiSchedule(ElewiseSchedule):
    """
    class of cce elewise schedule

    Parameters
    ----------
    VectorSchedule: base class of elewise schedule

    Returns
    -------
    ElewiseSchedule_instance : instance of ElewiseSchedule
    """

    # pylint: disable=arguments-differ
    def do_schedule(self, out_tensors, sch, spec_node_list):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        out_tensors : the out tvm.tensor

        sch : schedule, the computation schedule for the op

        spec_node_list : special node list

        Returns
        -------
        Bool, now is true

        """
        if len(out_tensors) <= 1 or not util.MULTI_ELEMWISE:
            return False
        schedule = self.__pre_complement_tensors_map(out_tensors)
        self._out_tensors = out_tensors
        if schedule is None:
            return False

        log.debug("start elewise_multi_schedule")
        self._schedule = schedule
        self._construct_compute_graph(out_tensors, spec_node_list)

        # init for block num
        # pylint: disable=attribute-defined-outside-init
        self._block_num = util.INIT_SIZE
        if self.__calculate_align():
            return False

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self._calculate_tiling()
        self._do_tiling()

        self._do_buffer_tile()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._calculate_double_buffer()
        self._do_double_buffer()

        log.debug("end elewise_multi_schedule")
        sch[0] = self._schedule
        return True

    def __pre_complement_tensors_map(self, out_tensors):
        """
        pre handle syntax tree by replace compute node
        use fake node to make it into one out schedule

        Parameters
        ----------
        outTensors : the out tvm.tensor

        Returns
        -------
        Schedule, mock schedule

        """
        # pylint: disable=invalid-name, attribute-defined-outside-init
        temp_mid_output_tensors_dst_tensor_map = {}
        temp_mid_output_tensors_in_ub = []
        self._mid_output_tensors_in_gm = []
        self._temp_out_tensors = {}
        # travel syntax tree into map
        util.get_dst_tensor_map(out_tensors,
                                temp_mid_output_tensors_dst_tensor_map)

        # tell difference between pure out and mid out
        for out in out_tensors:
            if out in temp_mid_output_tensors_dst_tensor_map.keys():
                temp_mid_output_tensors_in_ub.append(out)
        self._buffer_tile_out = out_tensors

        # make mid output tensors copy itself to out
        # pylint: disable=unnecessary-lambda
        for out in temp_mid_output_tensors_in_ub:
            with tvm.tag_scope(util.SET_GM_SCOPE_TAG):
                out_gm = tvm.compute(out.shape, lambda *i: out(*i),
                                     name=out.name + "_gm")
            index = out_tensors.index(out)
            out_tensors[index] = out_gm
            self._temp_out_tensors[out] = out_gm
            self._mid_output_tensors_in_gm.append(out_gm)

        # use fake node to intercept schedule
        res = util.fake_node_fuse_fun(out_tensors)
        if res == util.FAKE_NODE_FUSE_FAILED:
            return None
        sch = tvm.create_schedule([res.op])
        self._last_output_tensor = res
        out_tensors.append(res)
        return sch

    def _get_block_num(self):
        return self._block_num

    def __calculate_align(self):
        min_type_bitsize = util.MAX_TYPE_SIZE_UNIT
        for tensor in self._mid_output_tensors:
            temp_dtype = tensor.dtype.lower()
            if temp_dtype not in util.DTYPE_WIDTH_MAP:
                return True
            if util.DTYPE_WIDTH_MAP[temp_dtype] < min_type_bitsize:
                min_type_bitsize = util.DTYPE_WIDTH_MAP[temp_dtype]
        shape = util.shape_to_list(self._last_output_tensor.shape)
        # for block align, it must more than 1 block out
        cur_core_num = cceconf.get_soc_spec("CORE_NUM")
        cur_shape_size = util.get_shape_size(shape)
        block_per_num = int(util.VECTOR_ONE_BLOCK_UNIT / min_type_bitsize)
        max_core_num = int(cur_shape_size // block_per_num)
        # pylint: disable=attribute-defined-outside-init
        self._block_num = min(cur_core_num, max_core_num)
        return False

    def _calculate_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._input_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i,
                            self._input_tensor_dst_tensor_map[i])

    def _calculate_cache_write(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        exclude_tensors = self._cache_write_exclude_tensors + self._mid_output_tensors_in_gm
        for i in self._mid_tensors:
            if i not in exclude_tensors:
                self._cache_write_tensors.append(i)

    def _calculate_emit_insn(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        ElewiseSchedule._calculate_emit_insn(self)
        res = self._last_output_tensor
        ub_tiling_result = self._tiling_result["ub_tiling"]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        # eliminate mid out tensor from gm to ub by fake node
        for tensor in self._mid_output_tensors_in_gm:
            para = {"scope": tensor.op.axis[ub_split_axis],
                    "instruction": 'dma_copy'}
            self._emit_insn_map[tensor] = para

        self._emit_insn_map[res] = {"scope": res_ub_inner,
                                    "instruction": util.FAKE_NODE_PRAGMA}
        self._schedule[res].set_scope("")

    def _do_buffer_tile(self):
        """
        Optimize end block by buffer_tile
        Now, back-end pass has already solved mutil-core trumple
        So, delete buffer_tile func.

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        return
