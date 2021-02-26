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
Pure broadcast schedule
"""
import math
import te

from te import tvm
from te.platform import log

from te.tvm.tensor import Tensor
from te.tvm.schedule import Schedule
from te.platform.cce_util import get_align_factor
from te.platform.cce_util import get_buffer
from te.platform.cce_util import apply_for_new_alloc
from te.platform.cce_params import scope_ubuf
from te.platform.cce_conf import CceProductParams as pver
from .pure_broadcast_intrin import last_axis_broadcast
from .pure_broadcast_intrin import mid_axis_broadcast
from .pure_broadcast_intrin import full_aligned_broadcast_selection
from .pure_broadcast_intrin import VECTOR_ENHANCED_MODE

DMA_COPY_FRIENDLY_FACTOR = 8
MULTI_BROADCAST_ENHANCEMENT_FACTOR = 2.5
MULTI_BROADCAST_ENHANCEMENT_FOR_FACTOR = 32


class PureBroadcastSchedule:  # pylint: disable=R0902
    """Pure Broadcast Schedule"""

    def __init__(self):
        # Tensor information containers
        self.schedule = None
        self.all_tensors = []
        self.in2out_tensor_dict = {}
        self.placeholders = []
        self.broadcasts = []
        self.dtype = None
        # Device info
        self.device_core_num = -1
        self.device_core_limitation = -1
        self.device_ub_size = -1
        # Broadcast info
        self.broadcast_original_shape = None
        self.broadcast_target_shape = None
        self.is_broadcast_list = []
        # Tiling info
        self.ub_tiling_axis = None
        self.ub_tiling_factor = None
        self.block_tiling_axis = None
        self.block_tiling_nparts = None
        # dtype related global info
        self.block_size = None
        self.dtype_byte_size = None
        self.block_byte_size = None
        # Utilities
        self.scope_ubuf = te.platform.cce_params.scope_ubuf
        # Stage after data_flow_control
        self.placeholder_ub = None
        self.broadcast_ub = None
        # Broadcast tensor axis after tiling
        self.ub_outer = None
        self.ub_inner = None
        self.block_outer = None
        self.block_inner = None
        # For non-last axis broadcast dma_copy opt
        self.no_broadcast = False
        self.axis_offset = None
        self.is_aligned_non_last = False
        # For reconstruction
        self.real_outs = []
        self.original_ub_tiling_axis = None
        self.reconstructed = False

    def get_real_outs(self):
        """Get real output tensors we are using"""
        return self.real_outs

    def do_schedule(self,  # pylint: disable=too-many-locals, too-many-statements
                    out: Tensor,
                    sch: Schedule) -> Schedule:
        """Execute schedule generating sequence"""
        # Get rid of invalid input
        if not isinstance(sch, Schedule):
            raise TypeError("Pure broadcast schedule can only process tvm.Schedule, not"
                            + str(type(sch)))
        if not isinstance(out, Tensor):
            raise TypeError("Pure broadcast schedule can only process tvm.Tensor, not"
                            + str(type(sch)))
        ################################
        # Currently support one broadcast only !!!!!
        ################################

        ################################
        # Schedule initialization
        ################################
        self.init(out)
        self.schedule = sch
        ################################
        # Summary
        ################################
        # Now we've got several lists and dictionaries, they are:
        # all_tensors                     ALL tensors in compute graph
        # in2out_tensor_dict              ALL target tensors of all tensors
        # placeholders                    ALL placeholders
        # broadcasts                      ALL broadcast tensors
        # device_core_num                 Number of core for current compiling target
        # device_ub_size                  Byte size of UB Buffer for current compiling target
        ################################
        # Tiling Calculation
        ################################
        # This schedule is designed for pure broadcast and supports only one broadcast operation
        # So, calculate tiling strategy directly on the broadcast tensor
        self.calculate_tiling_on_broadcast()
        ################################
        # Tiling Report
        ################################
        self.print_debug("UB Tiling axis", self.ub_tiling_axis)
        self.print_debug("UB Tiling factor", self.ub_tiling_factor)
        self.print_debug("Block Tiling axis", self.block_tiling_axis)
        self.print_debug("Block Tiling factor", self.block_tiling_nparts)

        self.apply_non_last_axis_optimization()
        self.print_debug("Now applying", "Non-last-axis optimization")
        self.print_debug("UB Tiling axis", self.ub_tiling_axis)
        self.print_debug("UB Tiling factor", self.ub_tiling_factor)
        self.print_debug("Block Tiling axis", self.block_tiling_axis)
        self.print_debug("Block Tiling factor", self.block_tiling_nparts)
        mid_broadcast_size = 1
        mid_broadcast_list = self.is_broadcast_list[:self.ub_tiling_axis]
        for idx, val in enumerate(mid_broadcast_list):
            if val:
                mid_broadcast_size *= self.broadcast_target_shape[idx]
        if mid_broadcast_size < MULTI_BROADCAST_ENHANCEMENT_FOR_FACTOR:
            self.apply_multi_broadcast_optimization()
            self.print_debug("Now applying", "Multi broadcast optimization")
            self.print_debug("UB Tiling axis", self.ub_tiling_axis)
            self.print_debug("UB Tiling factor", self.ub_tiling_factor)
            self.print_debug("Block Tiling axis", self.block_tiling_axis)
            self.print_debug("Block Tiling factor", self.block_tiling_nparts)
        self.data_flow_control()
        self.do_tiling()
        self.do_compute_at()
        self.do_emit_insn()
        self.schedule[self.placeholder_ub].double_buffer()
        return self.schedule

    def init(self, out):
        """Initialize some necessary information for schedule"""
        self.dfs_get_all_tensors(out, self.all_tensors, self.in2out_tensor_dict)
        for tensor in self.all_tensors:
            if not tensor.op.input_tensors and tensor.op.tag == "":
                self.placeholders.append(tensor)
            else:
                self.broadcasts.append(tensor)
        # Get device core num
        self.device_core_num = te.platform.get_soc_spec("CORE_NUM")
        # Get device ub size
        self.device_ub_size = te.platform.get_soc_spec("UB_SIZE")
        # Get tensor information
        self.dtype = str(self.placeholders[0].dtype)
        # Get block information
        self.block_size, self.dtype_byte_size = get_align_factor(self.dtype)
        self.block_byte_size = self.block_size * self.dtype_byte_size
        # Put original output tensor input real output tensors list
        self.real_outs.append(out)

    @staticmethod
    def dfs_get_all_tensors(_tensor: Tensor, _output=None, _in2out=None):
        """This function uses depth first algorithm to find all tensors and in->out relations"""
        if _output is None:
            _output = []
        if _in2out is None:
            _in2out = {}
        if _tensor not in _output:
            _output.append(_tensor)
        for sub_tensor in _tensor.op.input_tensors:
            if sub_tensor in _in2out:
                _in2out[sub_tensor].append(_tensor)
            else:
                _in2out[sub_tensor] = [_tensor]
            PureBroadcastSchedule.dfs_get_all_tensors(sub_tensor, _output, _in2out)
        return _output

    def print_debug(self, info_name, info):
        """Debug info print"""
        log.debug("[PureBroadcastSchedule] {} : {}".format(
            str(info_name), str(info)))

    def calculate_tiling_on_broadcast(self):
        """Get tiling strategy based on the only broadcast tensor"""
        broadcast = self.broadcasts[0]  # There is only one broadcast tensor
        # Get broadcast input
        self.broadcast_original_shape = list(map(int, broadcast.op.input_tensors[0].shape))
        self.broadcast_target_shape = list(map(int, broadcast.shape))
        # Support for different dimension broadcast
        if len(self.broadcast_target_shape) > len(self.broadcast_original_shape):
            difference = len(self.broadcast_target_shape) - len(self.broadcast_original_shape)
            self.broadcast_original_shape += [1] * difference + self.broadcast_original_shape
        if len(self.broadcast_target_shape) < len(self.broadcast_original_shape):
            raise RuntimeError("Invalid broadcast from " +
                               str(self.broadcast_original_shape) + " to " +
                               str(self.broadcast_target_shape))
        # Get broadcast axis information
        for orig, targ in zip(self.broadcast_original_shape, self.broadcast_target_shape):
            if orig != targ:
                if orig == 1:
                    self.is_broadcast_list.append(True)
                    continue
                raise RuntimeError("Invalid broadcast from " +
                                   str(self.broadcast_original_shape) + " to " +
                                   str(self.broadcast_target_shape))
            self.is_broadcast_list.append(False)
        ub_tiling_finished = False
        while not ub_tiling_finished:
            ub_tiling_finished = self.calculate_ub_tiling()
        if self.block_tiling_axis is None and (self.ub_tiling_axis > 0
                                               or self.ub_tiling_factor <
                                               self.broadcast_target_shape[self.ub_tiling_axis]):
            self.calculate_block_tiling()

    def calculate_ub_tiling(self):  # pylint: disable=R0911, R1710
        """Get calculation unit by calculating ub tiling strategy"""
        if self.ub_tiling_axis is None:
            self.ub_tiling_axis = len(self.broadcast_target_shape) - 1
            self.ub_tiling_factor = math.ceil(self.broadcast_target_shape[-1] / 2)
            return False
        current_calcunit_byte_size, \
            out_size, \
            tail_out_size = self.get_current_calculation_unit_size()
        # Rule 1:
        # Conditions:
        # 1. Current calculation unit is larger than half of device_ub_size
        # Actions:
        # 1. Decrease self.ub_tiling_factor
        # 2. Or, move ub_tiling_axis to lower dimension
        if current_calcunit_byte_size > self.device_ub_size // 2:
            self.ub_tiling_factor = math.ceil(self.ub_tiling_factor / 2)
            return False
        # Rule 2:
        # Conditions:
        # 1. Current calculation unit output is smaller than 1 block
        # 2. There are usable free axis on ub_tiling_axis's higher dimension
        # 3. Or, self.ub_tiling_factor is smaller than its axis
        # Actions:
        # 1. Increase self.ub_tiling_factor by 1
        # 2. Or, move ub_tiling_axis to higher dimension and completely consume the higher dim
        # Rule 1 extended:
        # Conditions:
        # 1. If the tail part: ub_tiling_axis_size % ub_tiling_factor
        #    is smaller than block_byte_size, execute Rule 1 action
        if out_size < self.block_byte_size or \
                (0 < tail_out_size < self.block_byte_size):
            if self.ub_tiling_axis != 0 and \
                    self.ub_tiling_factor == self.broadcast_target_shape[self.ub_tiling_axis]:
                self.ub_tiling_axis -= 1
                self.ub_tiling_factor = 1
                return False
            if self.ub_tiling_factor < self.broadcast_target_shape[self.ub_tiling_axis]:
                self.ub_tiling_factor += 1
                return False
            self.block_tiling_axis = -1
            self.block_tiling_nparts = 1
            return True
        # Rule 3:
        # Conditions:
        # 1. Current calculation unit is smaller than half of device_ub_size
        # 2. Usable free axis product is higher than core_num if action is taken
        # Actions:
        # 1. Increase self.ub_tiling_factor
        # 2. Or, move ub_tiling_axis to higher dimension
        available_free_axis, ideal_factor = self.rule_2_get_tiling_info(current_calcunit_byte_size)
        # Block tiling is already exhausted, abort
        if available_free_axis < self.device_core_num * 2:
            self.print_debug("UB Tiling info:", "Block tiling exhausted")
            return True
        available_free_axis = available_free_axis / self.device_core_num
        available_free_axis = round(available_free_axis)
        if ideal_factor > 1:
            need_rerun = self.apply_rule_3(available_free_axis, ideal_factor)
            if need_rerun is not None:
                return need_rerun

        return True

    def apply_rule_3(self, available_free_axis, ideal_factor):
        """Apply Rule 3 for ub tiling calculation here in order to avoid static check"""
        # Avoid produce 2.x available axis to ensure the data being equally divided
        if available_free_axis > 1:
            # Current tiling axis can fill half of ub
            if self.ub_tiling_factor * ideal_factor < \
                    self.broadcast_target_shape[self.ub_tiling_axis]:
                self.ub_tiling_factor *= min(ideal_factor, int(available_free_axis))
                return True
            # Current tiling axis cannot fill half of ub
            if self.ub_tiling_factor < self.broadcast_target_shape[self.ub_tiling_axis]:
                difference = self.broadcast_target_shape[self.ub_tiling_axis] - \
                    self.ub_tiling_factor
                self.ub_tiling_factor = min(self.ub_tiling_factor + difference,
                                            self.ub_tiling_factor * int(available_free_axis))
                return False
            # Current axis exhausted
            if self.ub_tiling_axis != 0 and \
                    self.broadcast_target_shape[self.ub_tiling_axis] == self.ub_tiling_factor:
                self.ub_tiling_axis -= 1
                self.ub_tiling_factor = 1
                return False
        else:
            # Current tiling factor can increase further
            if self.ub_tiling_factor < self.broadcast_target_shape[self.ub_tiling_axis]:
                self.ub_tiling_factor += 1
                return False
            # Current axis exhausted
            if self.ub_tiling_axis != 0 and \
                    self.broadcast_target_shape[self.ub_tiling_axis] == self.ub_tiling_factor:
                self.ub_tiling_axis -= 1
                self.ub_tiling_factor = 1
                return False
        return None

    def rule_2_get_tiling_info(self, current_calcunit_byte_size):
        """Extracted from get_ub_info() in order to avoid static checks"""
        # Ideal factor is reduce by half for double buffer
        ideal_factor = self.device_ub_size // 2 // current_calcunit_byte_size
        available_free_axis = 1
        for axis_size in self.broadcast_target_shape[:self.ub_tiling_axis]:
            available_free_axis *= axis_size
        # Get available axis for block tiling
        ub_available_axis = self.broadcast_target_shape[self.ub_tiling_axis] / \
            self.ub_tiling_factor
        available_free_axis *= ub_available_axis
        return available_free_axis, ideal_factor

    def is_multi_core_trample(self):
        """ to prevent multi-core trampling"""
        # for example :[1,2,17,12] ub _nparts = 5, 17 split to 4*4 and 1
        # blockdim is 10, the 5th core will trample the 6th core data
        ub_nparts = self.broadcast_target_shape[self.ub_tiling_axis] // self.ub_tiling_factor
        main_block = math.ceil(self.broadcast_target_shape[self.ub_tiling_axis] / ub_nparts)
        tile_block = self.broadcast_target_shape[self.ub_tiling_axis] % main_block

        if self.ub_tiling_axis < len(self.broadcast_target_shape) - 1:
            calculattion_unit_target_shape = self.broadcast_target_shape[self.ub_tiling_axis+1:]
            unit_size = 1
            for axis_size in calculattion_unit_target_shape:
                unit_size *= axis_size
            tile_block_data_size = tile_block * unit_size * self.dtype_byte_size
        else:
            tile_block_data_size = tile_block * self.dtype_byte_size

        calculattion_src_target_shape = self.broadcast_target_shape[0:self.ub_tiling_axis]
        src_size = 1
        for axis_size in calculattion_src_target_shape:
            src_size *= axis_size
        
        if (src_size > 1) and (tile_block_data_size < self.block_byte_size):
            return True
        
        return False

    def calculate_block_tiling(self):
        """"Get core num by calculating block tiling strategy"""
        available_axis = list(range(0, self.ub_tiling_axis))
        ub_available_size = math.ceil(self.broadcast_target_shape[self.ub_tiling_axis] /
                                      self.ub_tiling_factor)        
        core_num = 1
        block_split_nparts = None
        block_split_axis = None
        for axis in available_axis:
            axis_size = self.broadcast_target_shape[axis]
            if core_num * axis_size <= self.device_core_num:
                core_num *= axis_size
                block_split_nparts = axis_size
                block_split_axis = axis
            else:
                needed = self.device_core_num // core_num
                for i in range(min(axis_size, needed), 0, -1):
                    core_num *= i
                    block_split_nparts = i
                    block_split_axis = axis
                    break
                hdw_tsch = pver().is_cloud_version() or pver().is_ng1_version()
                if not hdw_tsch:
                    if core_num >= self.device_core_num:
                        break
        needed = self.device_core_num // core_num
        if needed > 1 and ub_available_size > 1:
            if ub_available_size < needed:
                block_split_nparts = ub_available_size
                block_split_axis = self.ub_tiling_axis
            else:
                block_split_nparts = needed
                block_split_axis = self.ub_tiling_axis

        self.block_tiling_axis = block_split_axis
        self.block_tiling_nparts = block_split_nparts

    def get_current_calculation_unit_size(self):
        """Get calculation unit size"""
        calculation_unit_original_shape = self.broadcast_original_shape[self.ub_tiling_axis:]
        calculation_unit_target_shape = self.broadcast_target_shape[self.ub_tiling_axis:]
        calculation_unit_target_tail_shape = self.broadcast_target_shape[self.ub_tiling_axis:]
        calculation_unit_factor = self.ub_tiling_factor
        # Avoid none calculation_unit_factor
        if calculation_unit_factor is None:
            calculation_unit_factor = self.broadcast_target_shape[self.ub_tiling_axis]
        calculation_unit_target_shape[0] = calculation_unit_factor
        calculation_unit_target_tail_shape[0] = self.broadcast_target_shape[self.ub_tiling_axis] % \
            calculation_unit_factor
        calculation_unit_original_shape[0] = \
            min(calculation_unit_factor, calculation_unit_original_shape[0])
        in_size = 1
        for axis_size in calculation_unit_original_shape:
            in_size *= axis_size
        out_size = 1
        for axis_size in calculation_unit_target_shape:
            out_size *= axis_size
        tail_out_size = 1
        for axis_size in calculation_unit_target_tail_shape:
            tail_out_size *= axis_size
        total_size = in_size + out_size
        total_byte_size = total_size * self.dtype_byte_size
        return total_byte_size,\
            out_size * self.dtype_byte_size,\
            tail_out_size * self.dtype_byte_size

    def apply_non_last_axis_optimization(self, do_rule_3=False):
        """
        Non-last axis broadcast optimization rules here
        Direct dma_copy is sometimes faster than any broadcast algorithm
        """
        if do_rule_3:
            current_calcunit_byte_size, \
                _, \
                _ = self.get_current_calculation_unit_size()
            available_free_axis, ideal_factor = self.rule_2_get_tiling_info(
                current_calcunit_byte_size)
            if available_free_axis <= self.device_core_num * 2:
                self.print_debug("UB Tiling info:", "Block tiling exhausted")
                return
            available_free_axis = available_free_axis / self.device_core_num
            available_free_axis = round(available_free_axis)
            if ideal_factor > 1:
                rule_3_finished = self.apply_rule_3(available_free_axis, ideal_factor)
                if not rule_3_finished:
                    self.apply_non_last_axis_optimization(do_rule_3=True)
            return
        # Get which axis after ub_tiling_axis is being broadcasted
        broadcast_list = self.is_broadcast_list[self.ub_tiling_axis:]
        # If no broadcast happened, enable no_broadcast enhancement
        # If broadcast happened, check if the broadcast unit is larger than dma_copy threshold
        if True not in broadcast_list or self.is_dma_copy_friendly_size(broadcast_list):
            self.no_broadcast = True
            if self.axis_offset is not None:
                self.ub_tiling_axis += self.axis_offset
                self.ub_tiling_factor = self.broadcast_target_shape[self.ub_tiling_axis]
                if self.is_aligned_non_last:
                    self.apply_non_last_axis_optimization(do_rule_3=True)

    def apply_multi_broadcast_optimization(self, do_reconstruct=False):
        """
        Multi broadcast optimization rules here
        Direct dma_copy is faster than any broadcast algorithm in this case
        """
        broadcast_list = self.is_broadcast_list[self.ub_tiling_axis:]
        before_broadcast_shape_list = self.broadcast_original_shape[:self.ub_tiling_axis]
        for idx, _ in enumerate(before_broadcast_shape_list):
            if before_broadcast_shape_list[idx] != 1:
                before_broadcast_shape_list[idx] = False
        dma_friendly_factor = math.ceil(self.block_size * MULTI_BROADCAST_ENHANCEMENT_FACTOR)
        if broadcast_list[0] and broadcast_list.count(True) > 1 and \
                False not in before_broadcast_shape_list:
            size = 1
            for axis_size in self.broadcast_target_shape[self.ub_tiling_axis + 1:]:
                size *= axis_size
            if size > dma_friendly_factor:
                if self.original_ub_tiling_axis is None:
                    self.original_ub_tiling_axis = self.ub_tiling_axis
                self.ub_tiling_axis += 1
                self.ub_tiling_factor = self.broadcast_target_shape[self.ub_tiling_axis]
                return self.apply_multi_broadcast_optimization(do_reconstruct=True)
        if do_reconstruct:
            self.no_broadcast = False
            self._do_reconstruct()

    def _do_reconstruct(self):
        """
        Multi broadcast optimization will remove original broadcast tensor
        Reinitialize the whole operator

        THIS IS THE WORST OPTIMIZATION I'VE EVER WRITTEN
        """
        # Turn on reconstructed switch
        self.reconstructed = True
        # Get reconstruct source
        input_tensor = self.placeholders[0]
        # Get middle broadcast shape
        mid_broadcast_shape = self.broadcast_target_shape[:]
        for i in range(self.ub_tiling_axis):
            if self.broadcast_original_shape[i] == 1:
                mid_broadcast_shape[i] = 1
        # Do reconstruct
        mid_broadcast = te.lang.cce.broadcast(input_tensor, mid_broadcast_shape)
        final_broadcast = te.lang.cce.broadcast(mid_broadcast, self.broadcast_target_shape)
        self.broadcast_ub = mid_broadcast
        self.broadcasts[0] = final_broadcast
        self.real_outs[0] = final_broadcast
        self.schedule = tvm.create_schedule(final_broadcast.op)
        # Reinitialization
        self.in2out_tensor_dict[input_tensor] = [mid_broadcast]
        self.in2out_tensor_dict[mid_broadcast] = [final_broadcast]
        self.schedule[self.broadcast_ub].set_scope(scope_ubuf)

    def data_flow_control(self):
        """Execute data flow control stage"""
        # Read placeholder into ub
        for placeholder in self.placeholders:
            consumers = self.in2out_tensor_dict[placeholder]
            self.placeholder_ub = self.schedule.cache_read(placeholder,
                                                           self.scope_ubuf,
                                                           consumers)
        if not self.no_broadcast and not self.reconstructed:
            self.broadcast_ub = self.schedule.cache_write(self.broadcasts[0], self.scope_ubuf)

    def is_dma_copy_friendly_size(self, broadcast_list):
        """Get non-last-axis broadcast size"""
        size = 1
        dma_friendly_factor = math.ceil(self.block_size * DMA_COPY_FRIENDLY_FACTOR)
        reversed_broadcast_list = list(reversed(broadcast_list))
        reversed_broadcast_source_shape = \
            list(reversed(self.broadcast_original_shape[self.ub_tiling_axis:]))
        axis_offset = len(reversed_broadcast_source_shape)
        changed = False
        for idx, axis in enumerate(reversed_broadcast_source_shape):
            if not reversed_broadcast_list[idx]:
                if changed:
                    break
                size *= axis
                axis_offset -= 1
            elif False in reversed_broadcast_list[idx:]:
                changed = True
                axis_offset -= 1
        if size < dma_friendly_factor:
            return False
        # For aligned non-last axis
        if size % self.block_size == 0 and axis_offset > 0:
            self.is_aligned_non_last = True
            block_tiling_axis = self.block_tiling_axis
            if block_tiling_axis is None:
                block_tiling_axis = -1
            reversed_broadcast_source_shape = \
                list(reversed(self.broadcast_original_shape[:self.ub_tiling_axis + axis_offset]))
            for idx, axis in enumerate(reversed_broadcast_source_shape):
                real_axis = self.ub_tiling_axis + axis_offset - 1 - idx
                if real_axis > block_tiling_axis and \
                        size * self.broadcast_original_shape[real_axis] < \
                        self.device_ub_size // 2 // self.dtype_byte_size:
                    axis_offset -= 1
                    size *= self.broadcast_original_shape[real_axis]
                else:
                    break
        self.axis_offset = axis_offset
        return True

    def do_tiling(self):
        """Execute ub tiling stage"""
        broadcast = self.broadcasts[0]
        sch_broadcast = self.schedule[broadcast]
        axis = self.ub_tiling_axis
        factor = self.ub_tiling_factor
        ub_nparts = self.broadcast_target_shape[self.ub_tiling_axis] // factor
        self.ub_outer, self.ub_inner = sch_broadcast.split(sch_broadcast.op.axis[axis],
                                                           nparts=ub_nparts)
        if self.block_tiling_axis is not None and self.block_tiling_axis >= 0:
            self.do_block_tiling()
        if self.block_inner is None:
            self.block_inner = self.ub_outer
        if self.block_outer is None:
            self.block_outer = self.block_inner

    def do_block_tiling(self):
        """Execute block tiling stage"""
        # Fuse all axis from 0 to block_split_axis
        fuse_needed_axis = list(range(0, self.block_tiling_axis))
        broadcast = self.broadcasts[0]
        sch_broadcast = self.schedule[broadcast]
        current_axis = None
        if fuse_needed_axis:
            for axis in fuse_needed_axis:
                if current_axis is None:
                    current_axis = sch_broadcast.op.axis[axis]
                    continue
                current_axis = sch_broadcast.fuse(current_axis, sch_broadcast.op.axis[axis])
        # Deal with last axis
        last_axis = self.block_tiling_axis
        if last_axis == self.ub_tiling_axis:
            if current_axis is None:
                outer, inner = sch_broadcast.split(self.ub_outer,
                                                   nparts=self.block_tiling_nparts)
                self.block_outer = outer
                self.block_inner = inner
                self.ub_outer = inner
            else:
                outer, inner = sch_broadcast.split(self.ub_outer,
                                                   nparts=self.block_tiling_nparts)
                self.block_outer = sch_broadcast.fuse(current_axis, outer)
                self.block_inner = inner
                self.ub_outer = inner
        else:
            if current_axis is None:
                outer, inner = sch_broadcast.split(sch_broadcast.op.axis[last_axis],
                                                   nparts=self.block_tiling_nparts)
                self.block_outer = outer
                self.block_inner = inner
            else:
                outer, inner = sch_broadcast.split(sch_broadcast.op.axis[last_axis],
                                                   nparts=self.block_tiling_nparts)
                self.block_outer = sch_broadcast.fuse(current_axis, outer)
                self.block_inner = inner
        sch_broadcast.bind(self.block_outer,
                           te.tvm.thread_axis("blockIdx.x"))

    def do_compute_at(self):
        """Excecute compute hierarchy setting stage"""
        placeholder = self.placeholder_ub
        placeholder_sch = self.schedule[placeholder]
        # Non-last axis broadcast enhancement here, direct dma_copy is sometimes faster
        compute_at_axis = self._get_placeholder_compute_at_axis()
        if self.no_broadcast:
            placeholder_sch.compute_at(self.schedule[self.broadcasts[0]],
                                       compute_at_axis)
        else:
            broadcast_sch = self.schedule[self.broadcast_ub]
            placeholder_sch.compute_at(self.schedule[self.broadcasts[0]],
                                       compute_at_axis)
            if self.reconstructed:
                broadcast_sch.compute_at(self.schedule[self.broadcasts[0]],
                                         compute_at_axis)
            else:
                broadcast_sch.compute_at(self.schedule[self.broadcasts[0]],
                                         self.ub_outer)

    def _get_placeholder_compute_at_axis(self):
        if self.broadcast_original_shape[self.ub_tiling_axis] != 1 and \
                self.ub_tiling_factor != self.broadcast_original_shape[self.ub_tiling_axis]:
            return self.ub_outer
        # Move dma_copy out of loop if possible
        placeholder_axes_pre_ub_tiling = self.broadcast_original_shape[0:self.ub_tiling_axis]
        compute_at_axis = self.block_outer
        for axis in placeholder_axes_pre_ub_tiling:
            if axis != 1:
                compute_at_axis = self.ub_outer
                break
        return compute_at_axis

    def do_emit_insn(self):
        """Execute setting emit instruction pragma stage"""
        placeholder = self.placeholder_ub
        placeholder_sch = self.schedule[placeholder]
        placeholder_sch.emit_insn(placeholder_sch.op.axis[0], "dma_copy")
        if self.no_broadcast:
            compute_at_axis = self._get_placeholder_compute_at_axis()
            if compute_at_axis == self.block_outer:
                compute_at_axis = self.block_inner
            if compute_at_axis == self.ub_outer:
                compute_at_axis = self.ub_inner
            broadcast_sch = self.schedule[self.broadcasts[0]]
            broadcast_sch.emit_insn(compute_at_axis, "dma_copy")
        else:
            broadcast_ub_sch = self.schedule[self.broadcast_ub]
            broadcast_ub_sch.emit_insn(broadcast_ub_sch.op.axis[self.ub_tiling_axis],
                                       "unified_broadcast")
            broadcast_sch = self.schedule[self.broadcasts[0]]
            if self.reconstructed and self.block_tiling_axis == self.ub_tiling_axis:
                broadcast_sch.emit_insn(self.ub_outer, "dma_copy")
            else:
                broadcast_sch.emit_insn(self.ub_inner, "dma_copy")


###############################################################################
#
#             /////////////////////////////////////////////
#             Intrinsic instructions (Broadcast algorithms)
#             /////////////////////////////////////////////
#
###############################################################################


@tvm.register_func("tvm.intrin.cce.unified_broadcast")
def unified_broadcast(stmt_op):  # pylint: disable=too-many-locals, too-many-statements
    """Universal broadcast operation"""

    # Get workspace scope if there is one, use local.UB as default value
    # This procedure is crucial if there exists workspace, failed to apply this procedure
    # may result in memory allocation failure
    if isinstance(stmt_op, tvm.stmt.AttrStmt):
        scope = str(stmt_op.value).replace("\"", "")
    else:
        scope = scope_ubuf
    # Get original shape and target shape
    original_shape = []
    original_layout = []
    target_shape = []
    target_layout = []
    loop_var_dict = {}

    def store_ori(_var):
        if not isinstance(_var, (tvm.expr.Var, tvm.expr.IntImm)):
            store_ori(_var.a)
            store_ori(_var.b)
            return
        if isinstance(_var, tvm.expr.Var):
            if str(_var.name) in loop_var_dict:
                original_shape.append(loop_var_dict[str(_var.name)])
                original_layout.append(_var.name)
            else:
                original_shape.append(1)
                original_layout.append(_var.name)
            return
        if isinstance(_var, tvm.expr.IntImm):
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(_var)))

    def store_tgt(_var):
        if not isinstance(_var, (tvm.expr.Var, tvm.expr.IntImm)):
            store_tgt(_var.a)
            store_tgt(_var.b)
            return
        if isinstance(_var, tvm.expr.Var):
            if str(_var.name) in loop_var_dict:
                target_shape.append(loop_var_dict[str(_var.name)])
                target_layout.append(_var.name)
            return
        if isinstance(_var, tvm.expr.IntImm):
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(_var)))

    def interpret_statement(stmt):
        if isinstance(stmt, tvm.stmt.Store):
            store_tgt(stmt.index)
            return
        if isinstance(stmt, tvm.expr.Load):
            store_ori(stmt.index)
            return
        if isinstance(stmt, tvm.stmt.For):
            loop_var_dict[str(stmt.loop_var)] = int(stmt.extent)
            return
        raise RuntimeError("Backend Error: Received unexpected statement: " + str(type(stmt)))

    def list_product(list_slice):
        result = 1
        for i in list_slice:
            result *= i
        return int(result)

    tvm.ir_pass.IRTransform(stmt_op, None, interpret_statement, ["For"])
    tvm.ir_pass.IRTransform(stmt_op, None, interpret_statement, ["Load", "Store"])
    if not original_shape and not original_layout:
        original_shape = [1]
        original_layout = [None]
    # Organize broadcast schedule
    broadcast_schedule = []
    current_index = len(original_layout)
    for _, var in enumerate(reversed(target_layout)):
        if var != original_layout[current_index - 1]:
            original_shape.insert(current_index, loop_var_dict[var])
            if not (broadcast_schedule
                    and broadcast_schedule[-1][0] == list_product(original_shape[0:current_index])):
                broadcast_schedule.append([list_product(original_shape[0:current_index]),
                                           list_product(original_shape[current_index + 1:]),
                                           loop_var_dict[var]])
            else:
                broadcast_schedule[-1][2] *= loop_var_dict[var]
        else:
            current_index -= 1
    # Initialization
    ir_builder = tvm.ir_builder.create()
    ins, outs = get_buffer(stmt_op, need_unique=True)
    input_buffer = ins[0]
    output_buffer = outs[0]
    align_factor = 32
    _, dtype_byte_size = get_align_factor(input_buffer.dtype)
    # Do Broadcast
    loop_broadcast(align_factor, broadcast_schedule, dtype_byte_size, input_buffer, ir_builder,
                   list_product, output_buffer, target_shape, scope)
    return ir_builder.get()


def loop_broadcast(*args):  # pylint: disable=too-many-locals, too-many-statements
    """Extracted from unified_broadcast for static check"""
    align_factor, broadcast_schedule, dtype_byte_size, input_buffer, ir_builder, \
        list_product, output_buffer, target_shape, scope = args
    last_buffer = input_buffer
    for index, broadcast in enumerate(broadcast_schedule[:-1]):
        original_output_buffer = None
        if index < len(broadcast_schedule) - 1:
            tmp = broadcast_schedule[index + 1]
            if tmp[1] != 1 and int(tmp[1] * dtype_byte_size % align_factor) == 0:
                if full_aligned_broadcast_selection(tmp[0], tmp[1], tmp[2],
                                                    input_buffer.dtype) == VECTOR_ENHANCED_MODE:
                    original_output_buffer = output_buffer
                    output_buffer = apply_for_new_alloc(ir_builder, input_buffer.dtype,
                                                        (tmp[0]
                                                         * tmp[1]
                                                         * tmp[2],),
                                                        scope)
        last_buffer = do_broadcast(ir_builder, index, last_buffer, output_buffer, *broadcast, scope)
        if original_output_buffer is not None:
            output_buffer = original_output_buffer
    if broadcast_schedule:
        do_broadcast(ir_builder, -1, last_buffer, output_buffer, *broadcast_schedule[-1], scope)
    else:
        do_broadcast(ir_builder, -1, last_buffer, output_buffer, *(1,
                                                                   list_product(target_shape),
                                                                   1),
                     scope)


def do_broadcast(ir_builder,  # pylint: disable=too-many-locals, too-many-arguments
                 index, input_buffer, output_buffer,
                 broadcast_src, broadcast_unit, broadcast_factor, scope):
    """Pick broadcast algorithm"""
    # Determine broadcast algorithm
    # There are currently two kinds of broadcast
    # Broadcasting x, 1 to x, y which is "Last Axis Broadcast"
    # Broadcasting x, 1, z to x, y, z which is "Mid axis Broadcast"
    if broadcast_unit == 1:
        output_buffer = last_axis_broadcast(ir_builder, index, input_buffer, output_buffer,
                                            broadcast_src, broadcast_factor, scope)
    else:
        # It is possible for mid_axis_broadcast to switch output_buffer
        output_buffer = mid_axis_broadcast(ir_builder, index, input_buffer, output_buffer,
                                           broadcast_src, broadcast_unit, broadcast_factor, scope)
    return output_buffer
