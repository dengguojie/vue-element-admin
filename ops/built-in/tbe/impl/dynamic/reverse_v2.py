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
reverse_v2.py
"""
import functools
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import constant_util as constant


# max int64
MAX_INT64 = 2 ** 63 - 1
# 1 byte = 8 bit
EIGHT_BIT = 8
# reserved ub size
RESERVED_UB = 1024 * 8


# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements,too-many-arguments,too-many-locals
class ReverseExt2:
    """
    Function: use to store reverse_ext2 schedule parameters
    Modify : 2021-04-09
    """
    def __init__(self, shape_x, dtype_x, axis, kernel_name):
        """
        init scatter_nd base parameters

        Parameters
        ----------
        shape_x: tuple or list
            the shape of input tensor
        dtype_x: string
            the dtype of input tensor
        axis: dict
            the axis list for reverse
        kernel_name: str
            kernel name, default value is "reverse_ext2"

        Returns
        -------
        None
        """
        self.tik_instance = tik.Tik()
        self.aicore_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB
        self.shape_x = list(shape_x)
        self.inner_dtype = "int16"
        self.inner_bytes_size = tbe_platform.get_bit_len(self.inner_dtype) // EIGHT_BIT
        self.input_bytes_size = tbe_platform.get_bit_len(dtype_x) // EIGHT_BIT
        self.ub_element_number = self.ub_size_bytes // self.inner_bytes_size
        self.avaliable_ub = self.ub_element_number // 2
        self.block_num = constant.BLOCK_SIZE // self.inner_bytes_size
        self.process_num = 0

        self.axis = axis
        self.input_data = self.tik_instance.Tensor(self.inner_dtype, (MAX_INT64,), name="input_data",
                                                   scope=tik.scope_gm)
        self.input_axis = self.tik_instance.Tensor(self.axis.get("dtype"), (MAX_INT64,),
                                                   name="input_axis",
                                                   scope=tik.scope_gm)
        self.output_data = self.tik_instance.Tensor(self.inner_dtype, (MAX_INT64,), name="output_data",
                                                    scope=tik.scope_gm)
        self.tilling_gm = self.tik_instance.Tensor("int64", (32,), name="tilling_gm", scope=tik.scope_gm)

        self.kernel_name = kernel_name
        self.inner_shape_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_0")
        self.inner_shape_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_1")
        self.inner_shape_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_2")
        self.inner_shape_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_3")
        self.inner_shape_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_4")
        self.inner_shape_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_5")
        self.inner_shape_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_shape_6")
        self.inner_dividends_0 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_0")
        self.inner_dividends_1 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_1")
        self.inner_dividends_2 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_2")
        self.inner_dividends_3 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_3")
        self.inner_dividends_4 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_4")
        self.inner_dividends_5 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_5")
        self.inner_dividends_6 = self.tik_instance.Scalar(dtype="int64", name="inner_dividends_6")
        self.inner_axis_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_0")
        self.inner_axis_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_1")
        self.inner_axis_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_2")
        self.inner_axis_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_3")
        self.inner_axis_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_4")
        self.inner_axis_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_5")
        self.inner_axis_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_inner_axis_6")
        self.outer_shape_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_0")
        self.outer_shape_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_1")
        self.outer_shape_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_2")
        self.outer_shape_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_3")
        self.outer_shape_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_4")
        self.outer_shape_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_5")
        self.outer_shape_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_shape_6")
        self.outer_dividends_0 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_0")
        self.outer_dividends_1 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_1")
        self.outer_dividends_2 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_2")
        self.outer_dividends_3 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_3")
        self.outer_dividends_4 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_4")
        self.outer_dividends_5 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_5")
        self.outer_dividends_6 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_6")
        self.outer_dividends_7 = self.tik_instance.Scalar(dtype="int64", name="outer_dividends_7")
        self.outer_axis_0 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_0")
        self.outer_axis_1 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_1")
        self.outer_axis_2 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_2")
        self.outer_axis_3 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_3")
        self.outer_axis_4 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_4")
        self.outer_axis_5 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_5")
        self.outer_axis_6 = self.tik_instance.Scalar(dtype="int64", name="tiling_outer_axis_6")
        self.is_split_axi_reverse = self.tik_instance.Scalar("int64", name="tiling_is_split_axi_reverse")
        self.split_part_num = self.tik_instance.Scalar("int64", name="tiling_split_part_num")
        self.split_dim = self.tik_instance.Scalar("int64", name="tiling_split_dim")
        self.tiling_key = self.tik_instance.Scalar("int64", name="tiling_tiling_key")
        self.outer_shape = [
            self.outer_shape_0, self.outer_shape_1, self.outer_shape_2, self.outer_shape_3, self.outer_shape_4,
            self.outer_shape_5, self.outer_shape_6
        ]

        self.outer_dividends = [
            self.outer_dividends_0, self.outer_dividends_1, self.outer_dividends_2, self.outer_dividends_3,
            self.outer_dividends_4, self.outer_dividends_5, self.outer_dividends_6
        ]
        self.inner_dividends = [
            self.inner_dividends_0, self.inner_dividends_1, self.inner_dividends_2, self.inner_dividends_3,
            self.inner_dividends_4, self.inner_dividends_5, self.inner_dividends_6
        ]
        self.inner_shape = [
            self.inner_shape_0, self.inner_shape_1, self.inner_shape_2, self.inner_shape_3, self.inner_shape_4,
            self.inner_shape_5, self.inner_shape_6
        ]
        self.inner_total_num_list = None
        self.inner_loop = None
        self.inner_axis = None
        self.outer_axis = None
        self.move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
        self.current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")

        self.core_outer_num = None
        self.core_outer_start = None
        self.input_data_ub = None
        self.output_data_ub = None
        # max_vnchw_block_num must be 256 align
        self.max_vnchw_block_num = (self.avaliable_ub // 16 // 256) * 256

    def execute_tilling(self):
        """
        execute_tilling, copy tiling and read
        """
        with self.tik_instance.new_stmt_scope():
            tilling_ub = self.tik_instance.Tensor("int64", (32,), name="tilling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tilling_ub, self.tilling_gm, 0, 1, 8, 0, 0)
            self.set_tilling_param(tilling_ub)
        self.inner_total_num_list = []
        for i, _ in enumerate(self.inner_shape):
            inner_shape_offset = self.tik_instance.Scalar(dtype="int64", name="inner_shape_offset_" + str(i))
            inner_shape_offset.set_as(functools.reduce(lambda x, y: x * y, self.inner_shape[0:i + 1]))
            self.inner_total_num_list.append(inner_shape_offset)

    def core_scudule_tilling(self, _core_idx):
        """
        core_scudule_tilling
        """
        self.core_outer_num = self.tik_instance.Scalar("int64", name="core_outer_num")
        self.core_outer_start = self.tik_instance.Scalar("int64", name="core_outer_start")
        outer_num = self.tik_instance.Scalar(dtype="int64", name="outer_num")
        outer_num.set_as(functools.reduce(lambda x, y: x * y, self.outer_shape))
        self.core_outer_num.set_as((outer_num + self.aicore_num - 1) // self.aicore_num)
        self.core_outer_start.set_as(_core_idx * self.core_outer_num)
        with self.tik_instance.if_scope(outer_num % self.aicore_num != 0):
            with self.tik_instance.if_scope(_core_idx >= (outer_num % self.aicore_num)):
                self.core_outer_num.set_as(self.core_outer_num - 1)
                self.core_outer_start.set_as(_core_idx * self.core_outer_num + outer_num % self.aicore_num)

    def set_tilling_param(self, tilling_ub):
        """
        set_tilling_param
        """
        self.inner_shape_0.set_as(tilling_ub[0])
        self.inner_shape_1.set_as(tilling_ub[1])
        self.inner_shape_2.set_as(tilling_ub[2])
        self.inner_shape_3.set_as(tilling_ub[3])
        self.inner_shape_4.set_as(tilling_ub[4])
        self.inner_shape_5.set_as(tilling_ub[5])
        self.inner_shape_6.set_as(tilling_ub[6])

        self.inner_axis_0.set_as(tilling_ub[7])
        self.inner_axis_1.set_as(tilling_ub[8])
        self.inner_axis_2.set_as(tilling_ub[9])
        self.inner_axis_3.set_as(tilling_ub[10])
        self.inner_axis_4.set_as(tilling_ub[11])
        self.inner_axis_5.set_as(tilling_ub[12])
        self.inner_axis_6.set_as(tilling_ub[13])

        self.outer_shape_0.set_as(tilling_ub[14])
        self.outer_shape_1.set_as(tilling_ub[15])
        self.outer_shape_2.set_as(tilling_ub[16])
        self.outer_shape_3.set_as(tilling_ub[17])
        self.outer_shape_4.set_as(tilling_ub[18])
        self.outer_shape_5.set_as(tilling_ub[19])
        self.outer_shape_6.set_as(tilling_ub[20])

        self.outer_axis_0.set_as(tilling_ub[21])
        self.outer_axis_1.set_as(tilling_ub[22])
        self.outer_axis_2.set_as(tilling_ub[23])
        self.outer_axis_3.set_as(tilling_ub[24])
        self.outer_axis_4.set_as(tilling_ub[25])
        self.outer_axis_5.set_as(tilling_ub[26])
        self.outer_axis_6.set_as(tilling_ub[27])

        self.is_split_axi_reverse.set_as(tilling_ub[28])
        self.split_part_num.set_as(tilling_ub[29])
        self.split_dim.set_as(tilling_ub[30])
        self.tiling_key.set_as(tilling_ub[31])

        self.outer_axis = [
            self.outer_axis_0, self.outer_axis_1, self.outer_axis_2, self.outer_axis_3, self.outer_axis_4,
            self.outer_axis_5, self.outer_axis_6
        ]
        self.outer_dividends_0.set_as(self.outer_shape_1 * self.outer_shape_2 * self.outer_shape_3 *
                                      self.outer_shape_4 * self.outer_shape_5 * self.outer_shape_6)
        self.outer_dividends_1.set_as(self.outer_shape_2 * self.outer_shape_3 * self.outer_shape_4 *
                                      self.outer_shape_5 * self.outer_shape_6)
        self.outer_dividends_2.set_as(self.outer_shape_3 * self.outer_shape_4 * self.outer_shape_5 *
                                      self.outer_shape_6)
        self.outer_dividends_3.set_as(self.outer_shape_4 * self.outer_shape_5 * self.outer_shape_6)
        self.outer_dividends_4.set_as(self.outer_shape_5 * self.outer_shape_6)
        self.outer_dividends_5.set_as(self.outer_shape_6)
        self.outer_dividends_6.set_as(1)

    def axis_compute_for_new(self, index, current_index, move_out_index, loop_shape, axies, dividends):
        """
        axis_compute_for_new
        """
        current_index.set_as(self.outer_axis_0)
        move_out_index.set_as(0)
        # first idx
        idx_tmp_not_axis = (index // dividends[0] * dividends[0]) * (1 - axies[0])
        for axis_id in range(6):
            idx_tmp_not_axis = idx_tmp_not_axis \
                               + ((index % dividends[axis_id]) // dividends[axis_id + 1] * dividends[axis_id + 1]) \
                               * (1 - axies[axis_id + 1])

        idx_tmp_axis = ((loop_shape[0] - 1 - index // dividends[0]) * dividends[0]) * axies[0]
        for axis_id in range(6):
            idx_tmp_axis = idx_tmp_axis \
                           + ((loop_shape[axis_id + 1] - 1 - (index % dividends[axis_id]) // dividends[axis_id + 1])
                              * dividends[axis_id + 1]) * axies[axis_id + 1]
        move_out_index.set_as(move_out_index + idx_tmp_not_axis + idx_tmp_axis)

    def axis_compute(self, index, current_index, move_out_index, loop_shape, axies, dividends):
        """
        axis_compute_for_new
        """
        current_index.set_as(index)
        move_out_index.set_as(0)
        for axis_id in range(7):
            with self.tik_instance.if_scope(axies[axis_id] == 0):
                move_out_index.set_as(move_out_index + current_index // dividends[axis_id] * dividends[axis_id])
            with self.tik_instance.if_scope(axies[axis_id] != 0):
                move_out_index.set_as(move_out_index +
                                      (loop_shape[axis_id] - 1 - current_index // dividends[axis_id]) *
                                      dividends[axis_id])
            current_index.set_as(current_index - current_index // dividends[axis_id] * dividends[axis_id])

    def tilling_rules(self):
        """
        tilling_rules
        tiling_0: do not reverse with last dim, and last dim < 16
        tiling_1: do not reverse with last dim, and last dim is 16 align and < 512
        tiling_2: do not reverse with last dim, and last dim is not 16 align and < 512
        tiling_3: do not reverse with last dim, and last dim > 512
        tiling_4: do reverse with last dim, and last dim < 128
        tiling_5: do reverse with last dim, and last dim > 128 and < 512
        tiling_6: do reverse with last dim, and last dim > 512
        """
        self.inner_loop = [
            1, self.inner_shape_0, self.inner_shape_1, self.inner_shape_2, self.inner_shape_3, self.inner_shape_4,
            self.inner_shape_5
        ]
        self.inner_axis = [
            0, self.inner_axis_0, self.inner_axis_1, self.inner_axis_2, self.inner_axis_3, self.inner_axis_4,
            self.inner_axis_5
        ]
        max_split_part_num = self.tik_instance.Scalar(dtype="int64", name="max_split_part_num")
        self.inner_dividends_0.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[1::]))
        self.inner_dividends_1.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[2::]))
        self.inner_dividends_2.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[3::]))
        self.inner_dividends_3.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[4::]))
        self.inner_dividends_4.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[5::]))
        self.inner_dividends_5.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[6::]))
        self.inner_dividends_6.set_as(1)
        with self.tik_instance.if_scope(self.tiling_key == 0):
            with self.tik_instance.new_stmt_scope():
                mid_split_num = self.inner_total_num_list[-2] // self.split_dim
                max_split_part_num.set_as(self.max_vnchw_block_num // mid_split_num)
                self.tik_instance.scalar_min(self.split_part_num, self.split_dim, max_split_part_num)
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_non_last_axis_small(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 1):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_non_last_axis_16_aligned(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 2):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_non_last_axis(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 3):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_non_last_axis_large(index, move_out_index)

        self.inner_loop = [
            1, 1, self.inner_shape_0, self.inner_shape_1, self.inner_shape_2, self.inner_shape_3, self.inner_shape_4
        ]
        self.inner_axis = [
            0, 0, self.inner_axis_0, self.inner_axis_1, self.inner_axis_2, self.inner_axis_3, self.inner_axis_4
        ]
        self.inner_dividends_0.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[1::]))
        self.inner_dividends_1.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[2::]))
        self.inner_dividends_2.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[3::]))
        self.inner_dividends_3.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[4::]))
        self.inner_dividends_4.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[5::]))
        self.inner_dividends_5.set_as(functools.reduce(lambda x, y: x * y, self.inner_loop[6::]))
        self.inner_dividends_6.set_as(1)
        with self.tik_instance.if_scope(self.tiling_key == 4):
            with self.tik_instance.new_stmt_scope():
                mid_split_num = self.inner_total_num_list[-2] // self.split_dim
                max_split_part_num.set_as(self.max_vnchw_block_num // mid_split_num)
                self.tik_instance.scalar_min(self.split_part_num, self.split_dim, max_split_part_num)
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_small_shape(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 5):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.large_reverse(index, move_out_index)
        with self.tik_instance.if_scope(self.tiling_key == 6):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.for_range(self.core_outer_start,
                                                 self.core_outer_num + self.core_outer_start) as index:
                    current_index = self.tik_instance.Scalar(dtype="int64", name="current_index")
                    move_out_index = self.tik_instance.Scalar(dtype="int64", name="move_out_index")
                    self.axis_compute(index, current_index, move_out_index, self.outer_shape, self.outer_axis,
                                      self.outer_dividends)
                    self.reverse_last_axis_large(index, move_out_index)

    def op_run(self):
        """
        op_run: core scedule and cut core
        """
        with self.tik_instance.for_range(0, self.aicore_num, block_num=self.aicore_num) as core_index:
            self.execute_tilling()
            self.core_scudule_tilling(core_index)
            self.tilling_rules()

        opt_config = {"out_of_bound_sync_check": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.input_data, self.input_axis),
                                   outputs=[self.output_data],
                                   flowtable=[self.tilling_gm],
                                   config=opt_config)
        # add compile info
        # dtype_rate mean input_dtype byte // inner_dtype(fp16)
        # input_dtype is fp16/int16 dtype_rate == 1
        # input_dtype is fp32/int32 dtype_rate == 2
        dtype_rate = self.input_bytes_size // self.inner_bytes_size
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.aicore_num,
            "max_elements": self.avaliable_ub,
            "max_elements_last_large_size": 512,
            "dtype_rate": dtype_rate
        })

        return self.tik_instance

    def reverse_non_last_axis(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis for tiling 2
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)
        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(self.inner_total_num_list[-2] // self.split_dim * (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32")
        loop_times.set_as((self.split_dim + self.split_part_num - 1) // self.split_part_num)

        block_num = self.tik_instance.Scalar("int32")
        block_num.set_as((self.inner_shape[-1] + 15) // 16)
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)
        burst_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num // 16
        move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
        mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num
        index_lenth = self.inner_total_num_list[-1]
        with self.tik_instance.for_range(0, loop_times) as split_id:
            bias = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.for_range(0, 16) as index:
                self.tik_instance.data_move(
                    self.input_data_ub[index * block_num * 16],
                    self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                    index * self.inner_shape[-1]], 0, burst_num + 1, block_num,
                    self.inner_shape[-1] - block_num, block_num * 15)

            current_index = self.tik_instance.Scalar(dtype="int32")
            move_out_index = self.tik_instance.Scalar(dtype="int32")
            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute(index, current_index, move_out_index, self.inner_loop, self.inner_axis,
                                  self.inner_dividends)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(self.output_data_ub[move_out_index * block_num * 16],
                                            self.input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            with self.tik_instance.if_scope(move_num % 16 != 0):
                with self.tik_instance.if_scope(move_num // 16 > 0):
                    with self.tik_instance.for_range(0, 16) as index:
                        self.tik_instance.data_move(
                            self.output_data[bias + outer_move_out * index_lenth + index * self.inner_shape[-1]],
                            self.output_data_ub[index * block_num * 16], 0, move_num // 16, block_num, block_num * 15,
                            self.inner_shape[-1] - block_num)
                    self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                                self.output_data_ub, 0, move_num // 16, 1, block_num * 15 + 1,
                                                self.inner_shape[-1] - 1)

                with self.tik_instance.for_range(0, move_num % 16 - 1) as index:
                    self.tik_instance.data_move(
                        self.output_data[bias + outer_move_out * index_lenth +
                                         move_num // 16 * 16 * self.inner_shape[-1] + index * self.inner_shape[-1]],
                        self.output_data_ub[(move_num // 16 * 16 + index) * block_num * 16], 0, 1, block_num, 0, 0)
                self.tik_instance.data_move(
                    self.output_data[bias + outer_move_out * index_lenth +
                                     (move_num // 16 * 16 + move_num % 16 - 1) * self.inner_shape[-1]],
                    self.output_data_ub[(move_num // 16 * 16 + move_num % 16 - 1) * block_num * 16], 0, 1, 1, 0, 0)

            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(move_num // 16 > 0):
                    with self.tik_instance.for_range(0, 15) as index:
                        self.tik_instance.data_move(
                            self.output_data[bias + outer_move_out * index_lenth + index * self.inner_shape[-1]],
                            self.output_data_ub[index * block_num * 16], 0, move_num // 16, block_num, block_num * 15,
                            self.inner_shape[-1] - block_num)
                    self.tik_instance.data_move(
                        self.output_data[bias + outer_move_out * index_lenth + 15 * self.inner_shape[-1]],
                        self.output_data_ub[15 * block_num * 16], 0, move_num // 16 - 1, block_num, block_num * 15,
                        self.inner_shape[-1] - block_num)
                    self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                                self.output_data_ub, 0, move_num // 16, 1, block_num * 15 + 1,
                                                self.inner_shape[-1] - 1)

                self.tik_instance.data_move(
                    self.output_data[bias + outer_move_out * index_lenth + self.inner_shape[-1] * (move_num - 1)],
                    self.output_data_ub[(move_num - 1) * block_num * 16], 0, 1, block_num - 1, 0, 0)

            with self.tik_instance.for_range(0, 16) as last_loop:
                self.input_data_ub[last_loop].set_as(self.output_data_ub[(move_num - 1) * block_num * 16 +
                                                                         self.inner_shape[-1] - 16 + last_loop])
            self.tik_instance.data_move(
                self.output_data[bias + outer_move_out * index_lenth + self.inner_shape[-1] * move_num - 16],
                self.input_data_ub, 0, 1, 1, 0, 0)

    def reverse_non_last_axis_16_aligned(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis_16_aligned for tiling 1
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-2] // self.split_dim *
                (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        block_num = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.inner_shape[-1] % 16 == 0):
            block_num.set_as(self.inner_shape[-1] // 16)
        with self.tik_instance.else_scope():
            block_num.set_as((self.inner_shape[-1] // 16 + 1))
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

            move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32")
            move_out_index = self.tik_instance.Scalar(dtype="int32")

            bias = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            self.tik_instance.data_move(self.input_data_ub,
                                        self.input_data[split_id * move_in_bias + outer_move_in * index_lenth], 0, 1,
                                        move_num * block_num, 0, 0)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute(index, current_index, move_out_index, self.inner_loop, self.inner_axis,
                                  self.inner_dividends)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(self.output_data_ub[move_out_index * block_num * 16],
                                            self.input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth], self.output_data_ub, 0,
                                        1, move_num * block_num, 0, 0)

    def reverse_non_last_axis_small(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis_small for tiling 0
        """
        input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                 name="input_data_ub",
                                                 scope=tik.scope_ubuf)
        output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                  name="output_data_ub",
                                                  scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-2] // self.split_dim * (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)

        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32")
        move_in_bias.set_as(self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        mat_num = self.tik_instance.Scalar("int32")

        loop_times = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        block_num = 1
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        # for copy_in_info
        stride_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        single_copy = self.tik_instance.Scalar("int32", name="stride_copy")
        stride_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) // 16)
        single_copy.set_as((self.inner_total_num_list[-2] // self.split_dim * split_part_num) % 16)
        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-2] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

            move_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32")
            move_out_index = self.tik_instance.Scalar(dtype="int32")

            bias = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(stride_copy > 0):
                with self.tik_instance.for_range(0, 16) as index:
                    self.tik_instance.data_move(input_data_ub[index * block_num * 16],
                                                self.input_data[split_id * move_in_bias + outer_move_in * index_lenth
                                                                + index * self.inner_shape[-1]],
                                                0, stride_copy, block_num,
                                                self.inner_shape[-1] - block_num, block_num * 15)
            with self.tik_instance.for_range(0, single_copy) as index:
                self.tik_instance.data_move(
                    input_data_ub[index * block_num * 16 + stride_copy * 16 * 16],
                    self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                    index * self.inner_shape[-1] + stride_copy * 16 * self.inner_shape[-1]],
                    0, 1, block_num, 0, 0)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute(index, current_index, move_out_index, self.inner_loop, self.inner_axis,
                                  self.inner_dividends)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(output_data_ub[move_out_index * block_num * 16],
                                            input_data_ub[index * block_num * 16], 0, 1, block_num, 0, 0)

            mat_num.set_as((mid_loop + 255) // 256)

            src_list = [output_data_ub[i * 16 * 16 * block_num * mat_num] for i in range(16)]
            dst_list = [input_data_ub[i * 16] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 16 * block_num * mat_num, 16, 1)

            self.tik_instance.data_move(output_data_ub, input_data_ub, 0, 16 * mat_num, self.inner_shape[-1],
                                        16 - self.inner_shape[-1], 0)

            src_list = [output_data_ub[i * 16] for i in range(16)]
            dst_list = [input_data_ub[i * 16 * self.inner_shape[-1] * mat_num] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.inner_shape[-1] * mat_num, 1, 16)

            move_out_num = mid_loop * self.inner_shape[-1]
            with self.tik_instance.if_scope(move_out_num % 16 == 0):
                self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias], input_data_ub, 0, 1,
                                            move_out_num // 16, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias], input_data_ub, 0, 1,
                                            move_out_num // 16, 0, 0)
                with self.tik_instance.for_range(0, 16) as refill_id:
                    input_data_ub[refill_id].set_as(input_data_ub[move_out_num - 16 + refill_id])

                self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias + move_out_num - 16],
                                            input_data_ub, 0, 1, 1, 0, 0)

    def reverse_non_last_axis_large(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_non_last_axis_large for tiling 3
        """
        self.process_num = 3200
        offset_after_first_copy = self.tik_instance.Scalar("int32",
                                                           name="offset_after_first_copy",
                                                           init_value=self.block_num)
        with self.tik_instance.if_scope(self.inner_shape[-1] % self.block_num != 0):
            offset_after_first_copy.set_as(self.inner_shape[-1] % self.block_num)

        data_ub_one_block = self.tik_instance.Tensor(self.inner_dtype, (self.block_num,),
                                                     name="data_ub_one_block",
                                                     scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub_one_block, self.input_data[outer_move_in * self.inner_shape[-1]], 0, 1, 1,
                                    0, 0)
        self.tik_instance.data_move(self.output_data[outer_move_out * self.inner_shape[-1]], data_ub_one_block, 0, 1,
                                    1, 0, 0)
        new_copy_num = ((self.inner_shape[-1] - 1) // self.block_num) * self.block_num
        copy_loop = new_copy_num // self.process_num
        copy_tail = new_copy_num % self.process_num

        src_origin_offset = outer_move_in * self.inner_shape[-1] + offset_after_first_copy
        dst_origin_offset = outer_move_out * self.inner_shape[-1] + offset_after_first_copy
        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as index:
            data_ub = self.tik_instance.Tensor("int16", (self.process_num,), name="data_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_ub, self.input_data[src_origin_offset + index * self.process_num], 0, 1,
                                        self.process_num // self.block_num, 0, 0)
            self.tik_instance.data_move(self.output_data[dst_origin_offset + index * self.process_num], data_ub, 0, 1,
                                        self.process_num // self.block_num, 0, 0)
        with self.tik_instance.if_scope(copy_tail != 0):
            data_ub_tail = self.tik_instance.Tensor("int16", (self.process_num,),
                                                    name="data_ub_tail",
                                                    scope=tik.scope_ubuf)
            self.tik_instance.data_move(data_ub_tail,
                                        self.input_data[src_origin_offset + copy_loop * self.process_num], 0, 1,
                                        copy_tail // self.block_num, 0, 0)
            self.tik_instance.data_move(self.output_data[dst_origin_offset + copy_loop * self.process_num],
                                        data_ub_tail, 0, 1, copy_tail // self.block_num, 0, 0)

    def reverse_last_axis_large(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_last_axis_large for tiling 6
        """
        self.process_num = 2560
        offset_after_first_copy = self.tik_instance.Scalar("int32",
                                                           name="offset_after_first_copy",
                                                           init_value=self.block_num)
        with self.tik_instance.if_scope(self.inner_shape[-1] % (self.block_num * 16) != 0):
            offset_after_first_copy.set_as(self.inner_shape[-1] % (self.block_num * 16))
        with self.tik_instance.else_scope():
            offset_after_first_copy.set_as(256)

        data_ub_16_block_a = self.tik_instance.Tensor(self.inner_dtype, (self.block_num * 16,),
                                                      name="data_ub_16_block_a",
                                                      scope=tik.scope_ubuf)

        data_ub_16_block_b = self.tik_instance.Tensor(self.inner_dtype, (self.block_num * 16,),
                                                      name="data_ub_16_block_b",
                                                      scope=tik.scope_ubuf)
        self.tik_instance.data_move(data_ub_16_block_a,
                                    self.input_data[(outer_move_in + 1) * self.inner_shape[-1] - 256], 0, 1, 16, 0, 0)

        src_list = [data_ub_16_block_a[i * 16] for i in range(16)]
        dst_list = [data_ub_16_block_b[i * 16] for i in range(15, -1, -1)]
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

        src_list = [data_ub_16_block_b[i * 16] for i in range(16)]
        dst_list = [data_ub_16_block_a[i * 16] for i in range(15, -1, -1)]
        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

        self.tik_instance.data_move(self.output_data[outer_move_out * self.inner_shape[-1]], data_ub_16_block_a, 0, 1,
                                    16, 0, 0)

        new_copy_num = ((self.inner_shape[-1] - 1) // (self.block_num * 16)) * self.block_num * 16

        copy_loop = new_copy_num // self.process_num
        copy_tail = self.tik_instance.Scalar("int32", name="copy_tail")
        with self.tik_instance.if_scope(self.inner_shape[-1] < self.process_num):
            copy_tail.set_as(self.inner_shape[-1])
        with self.tik_instance.else_scope():
            copy_tail.set_as(new_copy_num % self.process_num)

        with self.tik_instance.for_range(0, copy_loop, thread_num=2) as index:
            data_ub_a = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                 name="data_ub_a", scope=tik.scope_ubuf)
            data_ub_b = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                 name="data_ub_b", scope=tik.scope_ubuf)
            self.tik_instance.data_move(
                data_ub_a, self.input_data[(outer_move_in + 1) * self.inner_shape[-1] -
                                           (index + 1) * self.process_num - offset_after_first_copy], 0, 1,
                self.process_num // self.block_num, 0, 0)

            src_list = [data_ub_a[i * 16] for i in range(16)]
            dst_list = [data_ub_b[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.process_num // 256, 16, 16)

            src_list = [data_ub_b[i * 16] for i in range(16)]
            dst_list = [data_ub_a[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.process_num // 256, 16, 16)

            with self.tik_instance.for_range(0, self.process_num // 256) as reorder_index:
                self.tik_instance.data_move(data_ub_b[(self.process_num // 256 - reorder_index - 1) * 256],
                                            data_ub_a[reorder_index * 256], 0, 1, 16, 0, 0)

            self.tik_instance.data_move(
                self.output_data[outer_move_out * self.inner_shape[-1] + offset_after_first_copy +
                                 index * self.process_num], data_ub_b, 0, 1, self.process_num // self.block_num, 0, 0)

        with self.tik_instance.if_scope(copy_tail != 0):
            data_ub_tail_a = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                      name="data_ub_tail_a",
                                                      scope=tik.scope_ubuf)

            data_ub_tail_b = self.tik_instance.Tensor(self.inner_dtype, (self.process_num,),
                                                      name="data_ub_tail_b",
                                                      scope=tik.scope_ubuf)

            self.tik_instance.data_move(data_ub_tail_a, self.input_data[outer_move_in * self.inner_shape[-1]], 0, 1,
                                        copy_tail // self.block_num, 0, 0)

            src_list = [data_ub_tail_a[i * 16] for i in range(16)]
            dst_list = [data_ub_tail_b[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, copy_tail // 256, 16, 16)

            src_list = [data_ub_tail_b[i * 16] for i in range(16)]
            dst_list = [data_ub_tail_a[i * 16] for i in range(15, -1, -1)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, copy_tail // 256, 16, 16)

            with self.tik_instance.for_range(0, copy_tail // 256) as reorder_index:
                self.tik_instance.data_move(data_ub_tail_b[(copy_tail // 256 - 1 - reorder_index) * 256],
                                            data_ub_tail_a[reorder_index * 256], 0, 1, 16, 0, 0)

            self.tik_instance.data_move(self.output_data[(outer_move_out + 1) * self.inner_shape[-1] - copy_tail],
                                        data_ub_tail_b, 0, 1, copy_tail // self.block_num, 0, 0)

    def large_reverse(self, outer_move_in=0, outer_move_out=0):
        """
        large_reverse for tiling 5
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)

        phase = self.tik_instance.Scalar("int64", name="phase")

        phase.set_as(
            self.inner_total_num_list[-3] // self.split_dim *
            (self.split_dim - self.split_part_num))
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        block_num = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.inner_shape[-1] % 256 == 0):
            block_num.set_as(self.inner_shape[-1] // 256 * 16)
        with self.tik_instance.else_scope():
            block_num.set_as((self.inner_shape[-1] // 256 + 1) * 16)

        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)
        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                phase.set_as(
                    self.inner_total_num_list[-3] // self.split_dim *
                    (self.split_dim - split_part_num))

            burst_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num // 16
            move_num = self.inner_total_num_list[-3] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            repeat_times = mid_loop * block_num // 16

            index_lenth = self.inner_total_num_list[-1]

            reorder_pieces = block_num // 16

            current_index = self.tik_instance.Scalar(dtype="int32")
            move_out_index = self.tik_instance.Scalar(dtype="int32")

            with self.tik_instance.for_range(0, 16) as index:
                self.tik_instance.data_move(
                    self.input_data_ub[index * block_num * 16],
                    self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                    index * self.inner_shape[-1]], 0, burst_num + 1, block_num,
                    self.inner_shape[-1] - block_num, block_num * 15)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute(index, current_index, move_out_index, self.inner_loop, self.inner_axis,
                                  self.inner_dividends)
                move_out_index = move_out_index - phase
                with self.tik_instance.for_range(0, reorder_pieces) as block_index:
                    self.tik_instance.data_move(
                        self.output_data_ub[move_out_index * self.inner_shape[-2] * block_num * 16 +
                                            block_index * 256],
                        self.input_data_ub[index * self.inner_shape[-2] * block_num * 16 +
                                           (reorder_pieces - 1 - block_index) * 256], 0, self.inner_shape[-2], 16,
                        (reorder_pieces - 1) * 16, (reorder_pieces - 1) * 16)

            src_list = [self.output_data_ub[i * 16] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16] for i in range(15, -1, -1)]

            self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeat_times, 16, 16)

            src_list = [self.input_data_ub[i * 16] for i in range(16)]
            dst_list = [self.output_data_ub[i * 16] for i in range(15, -1, -1)]

            self.tik_instance.vnchwconv(False, False, dst_list, src_list, repeat_times, 16, 16)

            bias = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.if_scope(self.inner_shape[-1] % 256 == 0):
                self.tik_instance.data_move(self.output_data[bias + outer_move_out * index_lenth],
                                            self.output_data_ub,
                                            0, 1, mid_loop * self.inner_shape[-1] // 16, 0, 0)

            with self.tik_instance.else_scope():
                width = self.tik_instance.Scalar(dtype="int32")
                width.set_as((mid_loop + 15) // 16 * block_num)

                src_list = [self.output_data_ub[i * width * 16] for i in range(16)]
                dst_list = [self.input_data_ub[i * 16] for i in range(16)]
                self.tik_instance.vnchwconv(False, False, dst_list, src_list, width, 16, 1)

                redundant_part = block_num * 16 - self.inner_shape[-1]
                self.tik_instance.data_move(self.output_data_ub, self.input_data_ub[redundant_part * 16], 0,
                                            width // block_num, block_num * 16 - redundant_part, redundant_part, 0)

                adjusted_width = ((width // block_num) * self.inner_shape[-1] // 256 + 1) * 16
                src_list = [self.output_data_ub[i * 16] for i in range(16)]
                dst_list = [self.input_data_ub[i * adjusted_width * 16] for i in range(16)]

                self.tik_instance.vnchwconv(False, False, dst_list, src_list, adjusted_width, 1, 16)

                cols = self.tik_instance.Scalar("int32")
                with self.tik_instance.if_scope(mid_loop % 16 == 0):
                    cols.set_as(mid_loop // 16)
                with self.tik_instance.else_scope():
                    cols.set_as(mid_loop // 16 + 1)

                loop = self.tik_instance.Scalar("int32")
                with self.tik_instance.if_scope(mid_loop % cols == 0):
                    loop.set_as(mid_loop // cols)
                with self.tik_instance.else_scope():
                    loop.set_as(mid_loop // cols + 1)

                last_row_valid_data = mid_loop - (loop - 1) * cols

                with self.tik_instance.for_range(0, loop - 1) as index:
                    self.tik_instance.data_move(
                        self.output_data[bias + index * cols * self.inner_shape[-1] + outer_move_out * index_lenth],
                        self.input_data_ub[index * adjusted_width * 16], 0, 1, adjusted_width, 0, 0)
                self.tik_instance.data_move(
                    self.output_data[bias + (loop - 1) * cols * self.inner_shape[-1] + outer_move_out * index_lenth],
                    self.input_data_ub[(loop - 1) * adjusted_width * 16], 0, 1,
                    last_row_valid_data * self.inner_shape[-1] // 16, 0, 0)
                move_out_num = adjusted_width * 16 * (loop - 1) + last_row_valid_data * self.inner_shape[-1]
                with self.tik_instance.for_range(0, 16) as refill_id:
                    self.input_data_ub[refill_id].set_as(self.input_data_ub[move_out_num - 16 + refill_id])
                self.tik_instance.data_move(
                    self.output_data[outer_move_out * index_lenth + bias + mid_loop * self.inner_shape[-1] - 16],
                    self.input_data_ub, 0, 1, 1, 0, 0)

    def reverse_small_shape(self, outer_move_in=0, outer_move_out=0):
        """
        reverse_small_shape for tiling 4
        """
        self.input_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                      name="input_data_ub",
                                                      scope=tik.scope_ubuf)
        self.output_data_ub = self.tik_instance.Tensor(self.inner_dtype, (self.avaliable_ub,),
                                                       name="output_data_ub",
                                                       scope=tik.scope_ubuf)
        phase = self.tik_instance.Scalar("int64", name="phase")
        with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
            phase.set_as(
                self.inner_total_num_list[-3] // self.split_dim *
                (self.split_dim - self.split_part_num))
        with self.tik_instance.else_scope():
            phase.set_as(0)
        total_num = self.inner_total_num_list[-1]

        move_in_bias = self.tik_instance.Scalar("int32")
        move_in_bias.set_as(
            self.inner_total_num_list[-1] // self.split_dim * self.split_part_num)

        loop_times = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.split_dim % self.split_part_num == 0):
            loop_times.set_as(self.split_dim // self.split_part_num)
        with self.tik_instance.else_scope():
            loop_times.set_as(self.split_dim // self.split_part_num + 1)

        mat_num = self.tik_instance.Scalar("int64")

        block_num = self.tik_instance.Scalar("int32")
        with self.tik_instance.if_scope(self.inner_shape[-1] % 16 == 0):
            block_num.set_as(self.inner_shape[-1] // 16)
        with self.tik_instance.else_scope():
            block_num.set_as(self.inner_shape[-1] // 16 + 1)
        split_part_num = self.tik_instance.Scalar("int32", name="tmp_split_num")
        split_part_num.set_as(self.split_part_num)

        with self.tik_instance.for_range(0, loop_times) as split_id:
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                split_part_num.set_as(self.split_dim - self.split_part_num * split_id)
                with self.tik_instance.if_scope(self.is_split_axi_reverse != 0):
                    phase.set_as(
                        self.inner_total_num_list[-3] // self.split_dim *
                        (self.split_dim - split_part_num))
                with self.tik_instance.else_scope():
                    phase.set_as(0)

            burst_num = self.inner_total_num_list[-2] // self.split_dim * split_part_num // 16
            move_num = self.inner_total_num_list[-3] // self.split_dim * split_part_num
            mid_loop = self.inner_total_num_list[-2] // self.split_dim * split_part_num

            index_lenth = self.inner_total_num_list[-1]

            current_index = self.tik_instance.Scalar(dtype="int32")
            move_out_index = self.tik_instance.Scalar(dtype="int32")

            bias = self.tik_instance.Scalar("int32")
            with self.tik_instance.if_scope(split_id == loop_times - 1):
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(0)
                with self.tik_instance.else_scope():
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
            with self.tik_instance.else_scope():
                with self.tik_instance.if_scope(self.is_split_axi_reverse):
                    bias.set_as(total_num - (split_id + 1) * mid_loop * self.inner_shape[-1])
                with self.tik_instance.else_scope():
                    bias.set_as(split_id * mid_loop * self.inner_shape[-1])

            with self.tik_instance.for_range(0, 16) as index:
                self.tik_instance.data_move(
                    self.input_data_ub[index * block_num * 16],
                    self.input_data[split_id * move_in_bias + outer_move_in * index_lenth +
                                    index * self.inner_shape[-1]], 0, burst_num + 1, block_num,
                    self.inner_shape[-1] - block_num, block_num * 15)

            with self.tik_instance.for_range(0, move_num) as index:
                self.axis_compute(index, current_index, move_out_index, self.inner_loop, self.inner_axis,
                                  self.inner_dividends)
                move_out_index = move_out_index - phase
                self.tik_instance.data_move(
                    self.output_data_ub[move_out_index * self.inner_shape[-2] * block_num * 16],
                    self.input_data_ub[index * self.inner_shape[-2] * block_num * 16], 0,
                    self.inner_shape[-2] * block_num, 1, 0, 0)

            mat_num.set_as((mid_loop + 255) // 256)

            src_list = [self.output_data_ub[i * 16 * 16 * block_num * mat_num] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16] for i in range(15, -1, -1)]

            last_block_num = self.tik_instance.Scalar("int32")
            last_block_num.set_as(self.inner_shape[-1] - self.inner_shape[-1] // 16 * 16)
            with self.tik_instance.if_scope(self.inner_shape[-1] % 16 == 0):
                last_block_num.set_as(16)

            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 16 * block_num * mat_num, 16, 1)
            with self.tik_instance.for_range(0, block_num - 1) as data_move_index:
                self.tik_instance.data_move(
                    self.output_data_ub[(self.inner_shape[-1] - data_move_index * 16 - 16) * 16],
                    self.input_data_ub[data_move_index * 16 * 16], 0, 16 * mat_num, 16, 16 * (block_num - 1),
                    last_block_num + (block_num - 2) * 16)
            self.tik_instance.data_move(self.output_data_ub,
                                        self.input_data_ub[block_num * 16 * 16 - last_block_num * 16], 0, 16 * mat_num,
                                        last_block_num, 16 - last_block_num + (block_num - 1) * 16,
                                        16 * (block_num - 1))

            src_list = [self.output_data_ub[i * 16] for i in range(16)]
            dst_list = [self.input_data_ub[i * 16 * self.inner_shape[-1] * mat_num] for i in range(16)]
            self.tik_instance.vnchwconv(True, False, dst_list, src_list, self.inner_shape[-1] * mat_num, 1, 16)

            move_out_num = mid_loop * self.inner_shape[-1]
            self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias], self.input_data_ub, 0, 1,
                                        move_out_num // 16, 0, 0)
            with self.tik_instance.if_scope(move_out_num % 16 != 0):
                with self.tik_instance.for_range(0, 16) as refill_id:
                    self.input_data_ub[refill_id].set_as(self.input_data_ub[move_out_num - 16 + refill_id])

                self.tik_instance.data_move(self.output_data[outer_move_out * total_num + bias + move_out_num - 16],
                                            self.input_data_ub, 0, 1, 1, 0, 0)

        return self.tik_instance


# pylint: disable=unused-argument,invalid-name
@register_operator("ReverseV2")
def reverse_v2(x, axis, y, kernel_name):
    """ calculating reverse_v2 tensor by axis parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    axis: dict
        shape and dtype of axis
    y: dict
        shape and dtype of output

    kernel_name : str
        cce kernel name, default value is "pad"

    Returns
    -------
    None.
    """
    x_dtype = x.get("dtype")
    x_shape = x.get("shape")
    tik_instance = ReverseExt2(x_shape, x_dtype, axis, kernel_name)
    tik_instance = tik_instance.op_run()

    return tik_instance
