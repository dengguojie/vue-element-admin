# Copyright 2021 Huawei Technologies Co., Ltd
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
split_v_d
"""

import functools
from te import tik
from te import tvm
from te import platform as tbe_platform
from te.platform.cce_build import build_config
from te.utils import para_check
from te.utils import shape_util
from te.utils.error_manager import error_manager_vector
from tbe.dsl.compute.array import split_compute_com
from tbe.dsl.static_schedule.split_schedule import split_schedule_com
from impl.util import util_common
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

# vtranspose can deal 16*16
TRANSPOSE_SIZE = 256
UNIT = 16
RESERVED_UB_SIZE = 8 * 1024


def ceil(int_x, int_y):
    """
    get cel for int_x and int_y
    """
    if int_x == 0:
        return 1
    res = int_x // int_y
    if int_x % int_y == 0:
        return res

    return res + 1


# pylint: disable=locally-disabled,unused-argument,too-many-arguments,too-many-locals,too-many-return-statements
# pylint: disable=too-many-statements,too-many-branches,too-many-instance-attributes,too-many-boolean-expressions
class SplitEqual():
    """Function: use to finish SplitEqual main functions to reset data
    """

    def __init__(self, shape, dtype, split_dim, size_splits, kernel_name):
        """init SplitEqual parameters
        """
        self.shape = shape
        self.dtype = dtype
        self.split_dim = split_dim
        self.size_splits = size_splits
        self.kernel_name = kernel_name
        self.first_dim = self.shape[0]
        self.last_dim = self.shape[1]
        self.split_last_dim = size_splits[0]
        self.output_num = len(self.size_splits)
        self.x_unit = UNIT
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.block_ele = 32 // self.dtype_size
        self.ub_ele = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) // self.dtype_size - \
                      self.block_ele
        self.half_ub_ele = self.ub_ele // 2
        min_unit = self.get_min_unit()
        self.x_unit = self.half_ub_ele // (UNIT * self.last_dim) // min_unit * min_unit
        self.core_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        self.tik_instance = tik.Tik()
        self.input_tensor = self.tik_instance.Tensor(self.dtype, self.shape, name="gm_in", scope=tik.scope_gm)
        self.output_tensor_list = []
        for _, i in enumerate(range(len(self.size_splits))):
            output_tensor = self.tik_instance.Tensor(self.dtype, [self.first_dim, size_splits[i]],
                                                     name="gm_out" + str(i),
                                                     scope=tik.scope_gm)
            self.output_tensor_list.append(output_tensor)

    def check_support(self):
        """
        check if input shape can select this branch
        """
        if len(set(self.size_splits)) != 1:
            return False
        if self.dtype not in ("float16",):
            return False
        if self.first_dim <= 512:
            return False
        if self.size_splits[0] % self.block_ele == 0:
            return False
        if self.x_unit <= 0:
            return False
        if self.x_unit > self.first_dim:
            return False
        return True

    def mov_input_to_output(self, x_offset, ub_a, ub_b):
        """
        move data from input_gm to output_gm
        """
        self.tik_instance.data_move(ub_a, self.input_tensor[x_offset * self.last_dim], 0, 1,
                                    self.x_unit * self.last_dim // self.block_ele, 0, 0)
        ub_a_src_list = [ub_a[self.x_unit * self.last_dim * i] for i in range(UNIT)]
        ub_b_dst_list = [ub_b[UNIT * i] for i in range(UNIT)]
        self.tik_instance.vnchwconv(True, True, ub_b_dst_list, ub_a_src_list, self.x_unit * self.last_dim // UNIT,
                                    UNIT * UNIT // self.block_ele, UNIT // self.block_ele)
        for i in range(self.output_num):
            offset_ub_b = i * self.split_last_dim * UNIT
            offset_ub_a = i * self.split_last_dim * UNIT * self.x_unit
            self.tik_instance.data_move(ub_a[offset_ub_a], ub_b[offset_ub_b], 0, self.x_unit,
                                        self.split_last_dim * UNIT // self.block_ele,
                                        (self.output_num - 1) * self.split_last_dim, 0)
        ub_a_src_list_2 = [ub_a[UNIT * i] for i in range(UNIT)]
        ub_b_dst_list_2 = [ub_b[self.x_unit * self.last_dim * i] for i in range(UNIT)]
        self.tik_instance.vnchwconv(True, True, ub_b_dst_list_2, ub_a_src_list_2, self.x_unit * self.last_dim // UNIT,
                                    UNIT // self.block_ele, UNIT * UNIT // self.block_ele)
        for i in range(self.output_num):
            offset_ub_b = self.split_last_dim * self.x_unit * i
            offset_output_gm = x_offset * self.split_last_dim
            self.tik_instance.data_move(self.output_tensor_list[i][offset_output_gm], ub_b[offset_ub_b], 0, 1,
                                        self.split_last_dim * self.x_unit // self.block_ele, 0, 0)

    def process_each_core(self, core_ele, core_offset):
        loop_num = core_ele // self.x_unit
        self.ub_a = self.tik_instance.Tensor(self.dtype, (self.half_ub_ele,), name="ub_a", scope=tik.scope_ubuf)
        self.ub_b = self.tik_instance.Tensor(self.dtype, (self.half_ub_ele,), name="ub_b", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            loop_offset = core_offset + loop_idx * self.x_unit
            self.mov_input_to_output(loop_offset, self.ub_a, self.ub_b)

    def get_min_unit(self):
        list_sixteen = {1, 2, 4, 8, 16}
        for i in list_sixteen:
            if self.split_last_dim * i % 16 == 0:
                return i
        return 16

    def run(self):
        """run function
        """
        block_len = ceil(self.first_dim, self.x_unit)
        core_len = ceil(block_len, self.core_num)
        core_ele = core_len * self.x_unit
        core_used = self.first_dim // core_ele
        if self.first_dim % core_ele != 0:
            core_used = core_used + 1
        last_core_ele = self.first_dim - (core_used - 1) * core_ele
        with self.tik_instance.for_range(0, core_used, block_num=core_used) as core_index:
            core_offset = core_index * core_ele
            if last_core_ele != core_ele:
                with self.tik_instance.if_scope(core_index < (core_used - 1)):
                    self.process_each_core(core_ele, core_offset)

                with self.tik_instance.else_scope():
                    core_offset = (core_used - 2) * core_ele + last_core_ele
                    self.process_each_core(core_ele, core_offset)
            else:
                self.process_each_core(core_ele, core_offset)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=self.output_tensor_list)
        return self.tik_instance
