#!/usr/bin/env python
# coding: utf-8
# Copyright 2019 Huawei Technologies Co., Ltd
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
nhwc_2_fractal_z_c04
"""
# pylint: disable=too-many-lines
import functools

import te.platform as tbe_platform
from te.utils import para_check
from te import tik

# available ub size
TOTAL_UB_MEMORY = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
# available number of cores
MAX_CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
# bytes of type float16
SIZE_TWO_BYTES = 2
# size of the cube unit
CUBE_SIZE = 16
# minimum unit of data_move: 32Bytes
DATA_MOVE_MIN_UNIT = 32
# maximum repeat number
MAX_REPEATS = 255
# maximum blk stride
MAX_STRIDE_BLK = 65535
# maximum rep stride
MAX_STRIDE_REP = 255
# maximum mask
MAX_MASK = 128
# number of cubes processed by one vadds instruction
NUM_CUBE = MAX_MASK // CUBE_SIZE
# the value of C0
C0 = 4


# pylint: disable=too-many-instance-attributes,too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,unused-argument,too-many-public-methods
def _cal_core(tik_instance, total_core_loop_num, num_core, core_number):
    """
    calculate the loop number on each core
    """
    core_loop = tik_instance.Scalar("uint64")
    sum_core = tik_instance.Scalar("uint64")

    with tik_instance.if_scope(num_core < total_core_loop_num % MAX_CORE_NUM):
        core_loop.set_as((total_core_loop_num + core_number - 1) //
                         core_number)
        sum_core.set_as(core_loop * num_core)

    with tik_instance.else_scope():
        core_loop.set_as(total_core_loop_num // core_number)
        sum_core.set_as((core_loop + 1) *
                        (total_core_loop_num % MAX_CORE_NUM) +
                        core_loop *
                        (num_core - total_core_loop_num % MAX_CORE_NUM))

    return core_loop, sum_core


def _set_core_num(loop_number):
    """
    set the block_num
    """
    if loop_number < MAX_CORE_NUM:
        return loop_number

    return MAX_CORE_NUM


def _cal_core_loop(tik_instance, num_data_one_loop, core_loop, ub_ori):
    """
    calculate the number of loops and remainder on each core
    """
    align_loop = tik_instance.Scalar("uint64")
    align_loop.set_as(ub_ori // num_data_one_loop)
    with tik_instance.if_scope(align_loop * num_data_one_loop > ub_ori):
        align_loop.set_as(align_loop - 1)

    remainder = tik_instance.Scalar("uint64")
    remainder.set_as(core_loop % align_loop)
    with tik_instance.if_scope(remainder == 0):
        remainder.set_as(align_loop)

    return align_loop, remainder


def _cal_core_loop_python(num_data_one_loop, core_loop, ub_ori):
    """
    calculate the number of loops and remainder on each core and return python
    variable
    """
    align_loop = ub_ori // num_data_one_loop

    if align_loop * num_data_one_loop > ub_ori:
        align_loop = align_loop - 1

    remainder = core_loop % align_loop

    return align_loop, remainder


class NHWC2FRACTAL4Compute:
    """
    Rearranges data from NHWC format into FRACTAL_Z_C04 format

    Functions
    ----------
    __init__:
        initialize some properties
    set_tik_instance:
        set tik_instance
    set_src_dst_tensor:
        set input and output tensor
    cal_core_loop:
        calculate the loop number on each core
    set_format_transfer_case:
        divide the transfer case from nhwc to fractal_z_c04
    vector_dup_zero:
        vector_dup zeros when dup_number is Scalar
    vector_dup_zero_python:
        vector_dup zeros when dup_number is python variable
    data_rearrange_c_four:
        rearrange data when UB can put in H * W * C0 data and
        zero padding is not required for processed N axis and
        C == 4
    data_rearrange_case_zero:
        rearrange data when UB can put in H * W * C0 data and
        zero padding is not required for processed N axis
    data_rearrange_case_one:
        rearrange data when UB can put in H * W * C0 data and
        zero padding is required for processed N axis
    data_rearrange_case_two:
        rearrange data when UB can not put in H * W * C0 data and
        process hw_number * C data at a time and
        zero padding is not required for processed N axis
    data_rearrange_case_three:
        rearrange data when UB can not put in H * W * C0 data and
        process hw_number * C data at a time and
        zero padding is required for processed N axis
    data_move_ub2gm_case_zero:
        move data from gm to UB when UB can put in H * W * C0 data
    data_move_ub2gm_case_one:
        move data from gm to UB when UB can not put in H * W * C0 data
    data_move_case_zero:
        the data_move process when UB can put in H * W * C0 data and
        H * W * C0 % 16 == 0 and
        zero padding is not required for processed N axis
    data_move_case_one:
        the data_move process when UB can put in H * W * C0 data and
        zero padding is required for some processed N axis and
        not required for others
    data_move_case_two:
        the data_move process when UB can put in H * W * C0 data and
        zero padding is required for processed N axis
    data_move_case_three:
        the data_move process when UB can put in H * W * C0 data and
        H * W * C0 % 16 != 0 and
        zero padding is not required for some processed N axis
    data_move_case_four:
        the data_move process when UB can put not in H * W * C0 data and
        zero padding is not required for some processed N axis
    data_move_case_five:
        the data_move process when UB can put not in H * W * C0 data and
        zero padding is required for some processed N axis
    format_transfer_case_zero:
        the transfer process when UB can put in H * W * C0 data
    format_transfer_case_one:
        the transfer process when UB can not put in H * W * C0 data
    nhwc_2_fractal_z_c04_compute:
        the overall transfer process
    get_tik_instance:
        obtain tik instance

    Returns
    -------
    None
    """
    def __init__(self, src_shape, dtype, kernel_name):
        """
        initialize some properties
        """
        self.src_shape = src_shape
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.dst_shape = [(self.src_shape[1] * self.src_shape[2] * C0 +
                           CUBE_SIZE - 1) // CUBE_SIZE,
                          (self.src_shape[0] + CUBE_SIZE - 1) // CUBE_SIZE,
                          CUBE_SIZE, CUBE_SIZE]

        self.num_byte = SIZE_TWO_BYTES
        self.mask = MAX_MASK
        # the number of data that can be moved in each data_move
        self.num_data = DATA_MOVE_MIN_UNIT // self.num_byte
        para_check.check_shape_rule(self.dst_shape)
        para_check.check_tensor_shape_size(self.dst_shape)
        # the number of data that UB can put in
        self.ub_memory = min(TOTAL_UB_MEMORY, 252 * 1024) // self.num_byte // 2
        self.src_gm = None
        self.dst_gm = None

    def set_tik_instance(self):
        """
        set tik_instance
        """
        tik_instance = tik.Tik()
        self.set_src_dst_tensor(tik_instance)

        return tik_instance

    def set_src_dst_tensor(self, tik_instance):
        """
        set input and output tensor
        """
        src_element_number = functools.reduce(lambda x1, x2: x1 * x2,
                                              self.src_shape[:])
        dst_element_number = functools.reduce(lambda x1, x2: x1 * x2,
                                              self.dst_shape[:])
        self.src_gm = tik_instance.Tensor(self.dtype,
                                          (src_element_number,),
                                          name="src_gm",
                                          scope=tik.scope_gm)
        self.dst_gm = tik_instance.Tensor(self.dtype,
                                          (dst_element_number,),
                                          name="dst_gm",
                                          scope=tik.scope_gm)

    def set_format_transfer_case(self):
        """
        divide the transfer case from nhwc to fractal_z_c04
        """
        format_transfer_case = 0
        if self.dst_shape[0] * self.dst_shape[3] > self.ub_memory:
            format_transfer_case = 1

        return format_transfer_case

    def vector_dup_zero(self, tik_instance, ub_trans, dup_number, offset):
        """
        vector_dup zeros when dup_number is Scalar
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        repeat_number = dup_number // MAX_MASK
        tail = dup_number % MAX_MASK

        with tik_instance.for_range(0, repeat_number // MAX_REPEATS) as \
                num_repeat_loop:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[MAX_MASK * MAX_REPEATS *
                                             num_repeat_loop + offset],
                                    scalar_zero,
                                    MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        with tik_instance.if_scope(repeat_number % MAX_REPEATS != 0):
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[repeat_number // MAX_REPEATS *
                                             MAX_MASK * MAX_REPEATS + offset],
                                    scalar_zero,
                                    repeat_number % MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        with tik_instance.if_scope(tail != 0):
            tik_instance.vector_dup(tail,
                                    ub_trans[MAX_MASK * repeat_number +
                                             offset],
                                    scalar_zero,
                                    1,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)

    def vector_dup_zero_python(self, tik_instance, ub_trans, dup_number,
                               offset):
        """
        vector_dup zeros when dup_number is python variable
        """
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        repeat_number = dup_number // MAX_MASK
        tail = dup_number % MAX_MASK

        with tik_instance.for_range(0, repeat_number // MAX_REPEATS) as \
                num_repeat_loop:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[MAX_MASK * MAX_REPEATS *
                                             num_repeat_loop + offset],
                                    scalar_zero,
                                    MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        if repeat_number % MAX_REPEATS != 0:
            tik_instance.vector_dup(MAX_MASK,
                                    ub_trans[repeat_number // MAX_REPEATS *
                                             MAX_MASK * MAX_REPEATS + offset],
                                    scalar_zero,
                                    repeat_number % MAX_REPEATS,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)
        if tail != 0:
            tik_instance.vector_dup(tail,
                                    ub_trans[MAX_MASK * repeat_number +
                                             offset],
                                    scalar_zero,
                                    1,
                                    self.num_byte // 2,
                                    MAX_MASK // self.num_data)

    def data_rearrange_c_four(self, tik_instance, ub_ori, ub_trans,
                              loop_number):
        """
        rearrange data when UB can put in H * W * C0 data and
        zero padding is not required for processed N axis and
        C == 4
        """
        hwc = self.src_shape[1] * self.src_shape[2] * C0
        scalar_zero = tik_instance.Scalar(dtype=self.dtype, init_value=0.0)
        with tik_instance.if_scope(loop_number == 1):
            if hwc % CUBE_SIZE != 0:
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - hwc % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)
                offset = hwc - hwc % CUBE_SIZE
                tik_instance.vector_dup([0, mask],
                                        ub_ori[offset],
                                        scalar_zero, 1, 0, 0)
        with tik_instance.else_scope():
            if hwc % CUBE_SIZE != 0:
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - hwc % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)
                if self.dst_shape[0] * self.dst_shape[3] // self.num_data > \
                        MAX_STRIDE_REP:
                    with tik_instance.for_range(0, loop_number) as num_loop:
                        offset = (num_loop + 1) * self.dst_shape[0] * \
                                 self.dst_shape[3] - CUBE_SIZE
                        tik_instance.vector_dup([0, mask],
                                                ub_ori[offset],
                                                scalar_zero, 1, 0, 0)
                else:
                    with tik_instance.for_range(0, loop_number //
                                                   MAX_REPEATS) as num_repeat:
                        offset = (num_repeat * MAX_REPEATS + 1) * \
                                 self.dst_shape[0] * self.dst_shape[3] - \
                                 CUBE_SIZE
                        tik_instance.vector_dup([0, mask],
                                                ub_ori[offset],
                                                scalar_zero,
                                                MAX_REPEATS, 0,
                                                self.dst_shape[0] *
                                                self.dst_shape[3] //
                                                self.num_data)
                    with tik_instance.if_scope(loop_number % MAX_REPEATS != 0):
                        offset = (loop_number // MAX_REPEATS *
                                  MAX_REPEATS + 1) * self.dst_shape[0] * \
                                 self.dst_shape[3] - CUBE_SIZE
                        tik_instance.vector_dup([0, mask],
                                                ub_ori[offset],
                                                scalar_zero,
                                                loop_number % MAX_REPEATS, 0,
                                                self.dst_shape[0] *
                                                self.dst_shape[3] //
                                                self.num_data)
            with tik_instance.for_range(0, loop_number // NUM_CUBE) \
                    as num_vadds:
                with tik_instance.for_range(0, self.dst_shape[0] //
                                               MAX_REPEATS) as num_repeat:
                    tik_instance.vadds(MAX_MASK,
                                       ub_trans[MAX_REPEATS * num_repeat *
                                                loop_number *
                                                self.dst_shape[3] +
                                                num_vadds * NUM_CUBE *
                                                self.dst_shape[3]],
                                       ub_ori[num_vadds * NUM_CUBE *
                                              self.dst_shape[0] *
                                              self.dst_shape[3] + MAX_REPEATS *
                                              num_repeat * self.dst_shape[3]],
                                       scalar_zero, MAX_REPEATS,
                                       self.num_byte // 2,
                                       self.dst_shape[0] * self.dst_shape[3] //
                                       self.num_data,
                                       loop_number * self.dst_shape[3] //
                                       self.num_data,
                                       self.num_byte // 2)
                if self.dst_shape[0] % MAX_REPEATS != 0:
                    tik_instance.vadds(MAX_MASK,
                                       ub_trans[self.dst_shape[0] //
                                                MAX_REPEATS *
                                                MAX_REPEATS * loop_number *
                                                self.dst_shape[3] +
                                                num_vadds * NUM_CUBE *
                                                self.dst_shape[3]],
                                       ub_ori[num_vadds * NUM_CUBE *
                                              self.dst_shape[0] *
                                              self.dst_shape[3] +
                                              self.dst_shape[0] //
                                              MAX_REPEATS * MAX_REPEATS *
                                              self.dst_shape[3]],
                                       scalar_zero,
                                       self.dst_shape[0] % MAX_REPEATS,
                                       self.num_byte // 2,
                                       self.dst_shape[0] * self.dst_shape[3] //
                                       self.num_data,
                                       loop_number * self.dst_shape[3] //
                                       self.num_data,
                                       self.num_byte // 2)
            with tik_instance.if_scope(loop_number % NUM_CUBE != 0):
                with tik_instance.for_range(0, self.dst_shape[0] //
                                               MAX_REPEATS) as num_repeat:
                    tik_instance.vadds((loop_number % NUM_CUBE) * CUBE_SIZE,
                                       ub_trans[MAX_REPEATS * num_repeat *
                                                loop_number *
                                                self.dst_shape[3] +
                                                loop_number // NUM_CUBE *
                                                NUM_CUBE *
                                                self.dst_shape[3]],
                                       ub_ori[loop_number // NUM_CUBE *
                                              NUM_CUBE * self.dst_shape[0] *
                                              self.dst_shape[3] + MAX_REPEATS *
                                              num_repeat * self.dst_shape[3]],
                                       scalar_zero, MAX_REPEATS,
                                       self.num_byte // 2,
                                       self.dst_shape[0] * self.dst_shape[3] //
                                       self.num_data,
                                       loop_number * self.dst_shape[3] //
                                       self.num_data,
                                       self.num_byte // 2)
                if self.dst_shape[0] % MAX_REPEATS != 0:
                    tik_instance.vadds((loop_number % NUM_CUBE) * CUBE_SIZE,
                                       ub_trans[self.dst_shape[0] //
                                                MAX_REPEATS * MAX_REPEATS *
                                                loop_number *
                                                self.dst_shape[3] +
                                                loop_number // NUM_CUBE *
                                                NUM_CUBE * self.dst_shape[3]],
                                       ub_ori[loop_number // NUM_CUBE *
                                              NUM_CUBE * self.dst_shape[0] *
                                              self.dst_shape[3] +
                                              self.dst_shape[0] //
                                              MAX_REPEATS * MAX_REPEATS *
                                              self.dst_shape[3]],
                                       scalar_zero,
                                       self.dst_shape[0] % MAX_REPEATS,
                                       self.num_byte // 2,
                                       self.dst_shape[0] * self.dst_shape[3] //
                                       self.num_data,
                                       loop_number * self.dst_shape[3] //
                                       self.num_data,
                                       self.num_byte // 2)

    def data_rearrange_case_zero(self, tik_instance, ub_ori, ub_trans,
                                 loop_number):
        """
        rearrange data when UB can put in H * W * C0 data and
        zero padding is not required for processed N axis
        """
        hwc = self.src_shape[1] * self.src_shape[2] * C0
        # zero padding if C != 4
        if self.src_shape[3] != C0:
            self.vector_dup_zero(tik_instance, ub_trans,
                                 loop_number * self.dst_shape[0] *
                                 self.dst_shape[3], 0)

            with tik_instance.for_range(0, loop_number) as num_loop:
                with tik_instance.for_range(0, hwc // CUBE_SIZE) as num_row:
                    with tik_instance.for_range(0, CUBE_SIZE // 4) as \
                            row_index:
                        trans_offset = loop_number * self.dst_shape[3] * \
                                       num_row + num_loop * CUBE_SIZE + \
                                       row_index * 4
                        ori_offset = num_loop * self.src_shape[1] * \
                                     self.src_shape[2] * self.src_shape[3] + \
                                     num_row * self.src_shape[3] * C0 + \
                                     row_index * self.src_shape[3]
                        with tik_instance.for_range(0, self.src_shape[3]) \
                                as num_c:
                            ub_trans[trans_offset + num_c].set_as(
                                ub_ori[ori_offset + num_c])

                if hwc % CUBE_SIZE != 0:
                    with tik_instance.for_range(0, (hwc % CUBE_SIZE) // 4) \
                            as row_index:
                        trans_offset = (self.dst_shape[0] - 1) * \
                                       self.dst_shape[3] * loop_number + \
                                       num_loop * CUBE_SIZE + row_index * 4
                        ori_offset = num_loop * self.src_shape[1] * \
                                     self.src_shape[2] * self.src_shape[3] + \
                                     (self.dst_shape[0] - 1) * \
                                     self.src_shape[3] * C0 + \
                                     row_index * self.src_shape[3]
                        with tik_instance.for_range(0, self.src_shape[3]) \
                                as num_c:
                            ub_trans[trans_offset + num_c].set_as(
                                ub_ori[ori_offset + num_c])

        else:
            self.data_rearrange_c_four(tik_instance, ub_ori, ub_trans,
                                       loop_number)

    def data_rearrange_case_one(self, tik_instance, ub_trans, loop_number):
        """
        rearrange data when UB can put in H * W * C0 data and
        zero padding is required for processed N axis
        """
        # dst is all 0
        dup_number = self.dst_shape[0] * self.dst_shape[3] * loop_number
        self.vector_dup_zero(tik_instance, ub_trans, dup_number, 0)

    def data_rearrange_case_two(self, tik_instance, ub_ori, ub_trans,
                                hw_number, is_last):
        """
        rearrange data when UB can not put in H * W * C0 data and
        process hw_number * C data at a time and
        zero padding is not required for processed N axis
        """
        if self.src_shape[3] == C0:
            row_number = (hw_number * C0 + CUBE_SIZE - 1) // CUBE_SIZE
            if is_last == 1 and hw_number * self.src_shape[3] % CUBE_SIZE != 0:
                scalar_zero = tik_instance.Scalar(dtype=self.dtype,
                                                  init_value=0.0)
                mask = 0
                for i, _ in enumerate(range(CUBE_SIZE - hw_number *
                                            self.src_shape[3] % CUBE_SIZE)):
                    mask += 2 ** (CUBE_SIZE - 1 - i)
                offset = row_number * CUBE_SIZE - CUBE_SIZE
                tik_instance.vector_dup([0, mask],
                                        ub_ori[offset],
                                        scalar_zero, 1, 0, 0)
        else:
            dup_number = (hw_number * C0 + CUBE_SIZE - 1) // CUBE_SIZE * \
                         CUBE_SIZE
            self.vector_dup_zero_python(tik_instance, ub_trans, dup_number,
                                        0)
            row_number = hw_number * C0 // CUBE_SIZE
            with tik_instance.for_range(0, row_number) as num_row:
                with tik_instance.for_range(0, CUBE_SIZE // 4) as row_index:
                    trans_offset = self.dst_shape[3] * num_row + \
                                   row_index * 4
                    ori_offset = num_row * CUBE_SIZE // 4 * \
                                 self.src_shape[3] + row_index * \
                                 self.src_shape[3]
                    with tik_instance.for_range(0, self.src_shape[3]) as num_c:
                        ub_trans[trans_offset + num_c].set_as(ub_ori
                                                              [ori_offset +
                                                               num_c])
            if hw_number * C0 % CUBE_SIZE != 0:
                with tik_instance.for_range(
                        0, (hw_number * C0 % CUBE_SIZE) // 4) as row_index:
                    trans_offset = self.dst_shape[3] * row_number + \
                                   row_index * 4
                    ori_offset = row_number * CUBE_SIZE // 4 * \
                                 self.src_shape[3] + row_index * \
                                 self.src_shape[3]
                    with tik_instance.for_range(0, self.src_shape[3]) \
                            as num_c:
                        ub_trans[trans_offset + num_c].set_as(ub_ori
                                                              [ori_offset +
                                                               num_c])

    def data_rearrange_case_three(self, tik_instance, ub_trans, hw_number):
        """
        rearrange data when UB can not put in H * W * C0 data and
        process hw_number * C data at a time and
        zero padding is required for processed N axis
        """
        # dst is all 0
        dup_number = (C0 * hw_number + CUBE_SIZE - 1) // CUBE_SIZE * CUBE_SIZE
        self.vector_dup_zero_python(tik_instance, ub_trans, dup_number, 0)

    def data_move_ub2gm_case_zero(self, tik_instance, ub_ori, ub_trans,
                                  loop_number, num_n):
        '''
        move data from gm to UB when UB can put in H * W * C0 data
        '''
        with tik_instance.if_scope(
                (self.dst_shape[1] * self.dst_shape[2] -
                 loop_number) * self.dst_shape[3] //
                self.num_data > MAX_STRIDE_BLK):
            with tik_instance.for_range(0, self.dst_shape[0]) \
                    as num_x1:
                dst_gm_index = num_x1 * self.dst_shape[1] * \
                               self.dst_shape[2] * self.dst_shape[3] + \
                               (num_n - (loop_number - 1)) * \
                               self.dst_shape[3]
                if self.src_shape[3] == C0:
                    with tik_instance.if_scope(loop_number == 1):
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_ori[num_x1 * loop_number *
                                                      self.dst_shape[3]],
                                               0, 1,
                                               loop_number *
                                               self.dst_shape[3] //
                                               self.num_data, 0, 0)
                    with tik_instance.else_scope():
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[num_x1 * loop_number *
                                                        self.dst_shape[3]],
                                               0, 1,
                                               loop_number *
                                               self.dst_shape[3] //
                                               self.num_data, 0, 0)
                else:
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[num_x1 * loop_number *
                                                    self.dst_shape[3]],
                                           0, 1,
                                           loop_number *
                                           self.dst_shape[3] //
                                           self.num_data, 0, 0)
        with tik_instance.else_scope():
            dst_gm_index = (num_n - (loop_number - 1)) * self.dst_shape[3]
            if self.src_shape[3] == C0:
                with tik_instance.if_scope(loop_number == 1):
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_ori[0],
                                           0, self.dst_shape[0],
                                           loop_number * self.dst_shape[3] //
                                           self.num_data,
                                           0,
                                           (self.dst_shape[1] *
                                            self.dst_shape[2] -
                                            loop_number) *
                                           self.dst_shape[3] //
                                           self.num_data)
                with tik_instance.else_scope():
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0],
                                           0, self.dst_shape[0],
                                           loop_number * self.dst_shape[3] //
                                           self.num_data,
                                           0,
                                           (self.dst_shape[1] *
                                            self.dst_shape[2] -
                                            loop_number) *
                                           self.dst_shape[3] //
                                           self.num_data)
            else:
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, self.dst_shape[0],
                                       loop_number * self.dst_shape[3] //
                                       self.num_data,
                                       0,
                                       (self.dst_shape[1] *
                                        self.dst_shape[2] -
                                        loop_number) *
                                       self.dst_shape[3] //
                                       self.num_data)

    def data_move_ub2gm_case_one(self, tik_instance, ub_ori, ub_trans,
                                 loop_hw, loop_len, loop_time, num_n):
        '''
        move data from gm to UB when UB can not put in H * W * C0 data
        '''
        if (self.dst_shape[1] * self.dst_shape[2] - 1) * \
                self.dst_shape[3] // self.num_data > MAX_STRIDE_BLK:
            with tik_instance.for_range(0, (C0 * loop_len +
                                            CUBE_SIZE - 1) // CUBE_SIZE) \
                    as num_x1:
                dst_gm_index = (C0 * loop_hw // CUBE_SIZE *
                                loop_time + num_x1) * self.dst_shape[1] * \
                               self.dst_shape[2] * self.dst_shape[3] + \
                               num_n * self.dst_shape[3]
                if self.src_shape[3] == C0:
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_ori[num_x1 *
                                                  self.dst_shape[3]],
                                           0, 1, 1, 0, 0)
                else:
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[num_x1 *
                                                    self.dst_shape[3]],
                                           0, 1, 1, 0, 0)
        else:
            dst_gm_index = C0 * loop_hw // CUBE_SIZE *  loop_time * \
                           self.dst_shape[1] * self.dst_shape[2] * \
                           self.dst_shape[3] + num_n * self.dst_shape[3]
            if self.src_shape[3] == C0:
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_ori[0],
                                       0,
                                       (C0 * loop_len + CUBE_SIZE - 1) //
                                       CUBE_SIZE, 1, 0,
                                       (self.dst_shape[1] *
                                        self.dst_shape[2] - 1) *
                                       self.dst_shape[3] // self.num_data)
            else:
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0,
                                       (C0 * loop_len + CUBE_SIZE - 1) //
                                       CUBE_SIZE, 1, 0,
                                       (self.dst_shape[1] *
                                        self.dst_shape[2] - 1) *
                                       self.dst_shape[3] // self.num_data)

    def data_move_case_zero(self, tik_instance, ub_ori, ub_trans, core_loop,
                            sum_core, align_loop, remainder):
        """
        the data_move process when UB can put in H * W * C0 data and
        H * W * C0 % 16 == 0 and
        zero padding is not required for processed N axis
        """
        num_data_one_loop = self.src_shape[1] * self.src_shape[2] * \
                            self.src_shape[3]
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            total_core_loop = sum_core + num_core_loop
            num_n = total_core_loop

            with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                               align_loop == 0,
                                               num_core_loop !=
                                               core_loop - 1)):
                src_gm_index = (num_n - (align_loop - 1)) * num_data_one_loop
                src_ub_index = 0
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       (align_loop * num_data_one_loop +
                                        self.num_data - 1) // self.num_data,
                                       0, 0)
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              align_loop)
                self.data_move_ub2gm_case_zero(tik_instance, ub_ori, ub_trans,
                                               align_loop, num_n)

            with tik_instance.if_scope(num_core_loop == core_loop - 1):
                src_gm_index = num_n * num_data_one_loop - \
                               (remainder - 1) * num_data_one_loop
                src_ub_index = 0
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       (remainder * num_data_one_loop +
                                        self.num_data - 1) // self.num_data,
                                       0, 0)
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              remainder)
                self.data_move_ub2gm_case_zero(tik_instance, ub_ori, ub_trans,
                                               remainder, num_n)

    def data_move_case_one(self, tik_instance, ub_ori, ub_trans, core_loop,
                           sum_core):
        """
        the data_move process when UB can put in H * W * C0 data and
        zero padding is required for some processed N axis and
        not required for others
        """
        num_data_one_loop = self.src_shape[1] * self.src_shape[2] * \
                            self.src_shape[3]
        loop_number = tik_instance.Scalar(dtype="uint64", init_value=1)
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            total_core_loop = sum_core + num_core_loop
            num_n = total_core_loop
            with tik_instance.if_scope(num_n < self.src_shape[0]):
                src_gm_index = num_n * num_data_one_loop
                src_ub_index = 0
                tik_instance.data_move(ub_ori[src_ub_index],
                                       self.src_gm[src_gm_index],
                                       0, 1,
                                       (num_data_one_loop + self.num_data -
                                        1) // self.num_data,
                                       0, 0)
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              loop_number)
                self.data_move_ub2gm_case_zero(tik_instance, ub_ori, ub_trans,
                                               loop_number, num_n)
            with tik_instance.else_scope():
                self.data_rearrange_case_one(tik_instance, ub_trans,
                                             loop_number)
                with tik_instance.if_scope(
                        (self.dst_shape[1] * self.dst_shape[2] - 1) *
                        self.dst_shape[3] // self.num_data >
                        MAX_STRIDE_BLK):
                    with tik_instance.for_range(0, self.dst_shape[0]) \
                            as num_x1:
                        dst_gm_index = num_x1 * self.dst_shape[1] * \
                                       self.dst_shape[2] * \
                                       self.dst_shape[3] + num_n * \
                                       self.dst_shape[3]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[num_x1 *
                                                        self.dst_shape[3]],
                                               0, 1,
                                               self.dst_shape[3] //
                                               self.num_data, 0, 0)
                with tik_instance.else_scope():
                    dst_gm_index = num_n * self.dst_shape[3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0],
                                           0, self.dst_shape[0],
                                           self.dst_shape[3] // self.num_data,
                                           0,
                                           (self.dst_shape[1] *
                                            self.dst_shape[2] - 1) *
                                           self.dst_shape[3] // self.num_data)

    def data_move_case_two(self, tik_instance, ub_trans, core_loop,
                           sum_core, align_loop, remainder):
        """
        the data_move process when UB can put in H * W * C0 data and
        zero padding is required for processed N axis
        """
        with tik_instance.for_range(0, core_loop) as num_core_loop:
            total_core_loop = sum_core + num_core_loop
            num_n = total_core_loop

            with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                               align_loop == 0,
                                               num_core_loop !=
                                               core_loop - 1)):
                self.data_rearrange_case_one(tik_instance, ub_trans,
                                             align_loop)
                with tik_instance.if_scope(
                        (self.dst_shape[1] * self.dst_shape[2] -
                         align_loop) * self.dst_shape[3] //
                        self.num_data > MAX_STRIDE_BLK):
                    with tik_instance.for_range(0, self.dst_shape[0]) \
                            as num_x1:
                        dst_gm_index = num_x1 * self.dst_shape[1] * \
                                       self.dst_shape[2] * \
                                       self.dst_shape[3] + \
                                       (num_n - (align_loop - 1)) * \
                                       self.dst_shape[3]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[num_x1 * align_loop *
                                                        self.dst_shape[3]],
                                               0, 1,
                                               align_loop *
                                               self.dst_shape[3] //
                                               self.num_data, 0, 0)
                with tik_instance.else_scope():
                    dst_gm_index = (num_n - (align_loop - 1)) * \
                                   self.dst_shape[3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0],
                                           0, self.dst_shape[0],
                                           align_loop * self.dst_shape[3] //
                                           self.num_data,
                                           0,
                                           (self.dst_shape[1] *
                                            self.dst_shape[2] - align_loop) *
                                           self.dst_shape[3] // self.num_data)

            with tik_instance.if_scope(num_core_loop == core_loop - 1):
                self.data_rearrange_case_one(tik_instance, ub_trans,
                                             remainder)
                with tik_instance.if_scope(
                        (self.dst_shape[1] * self.dst_shape[2] -
                         remainder) * self.dst_shape[3] //
                        self.num_data > MAX_STRIDE_BLK):
                    with tik_instance.for_range(0, self.dst_shape[0]) \
                            as num_x1:
                        dst_gm_index = num_x1 * self.dst_shape[1] * \
                                       self.dst_shape[2] * \
                                       self.dst_shape[3] + \
                                       (num_n - (remainder - 1)) * \
                                       self.dst_shape[3]
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[num_x1 * remainder *
                                                        self.dst_shape[3]],
                                               0, 1,
                                               remainder * self.dst_shape[3] //
                                               self.num_data, 0, 0)
                with tik_instance.else_scope():
                    dst_gm_index = (num_n - (remainder - 1)) * \
                                   self.dst_shape[3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[0],
                                           0, self.dst_shape[0],
                                           remainder * self.dst_shape[3] //
                                           self.num_data,
                                           0,
                                           (self.dst_shape[1] *
                                            self.dst_shape[2] - remainder) *
                                           self.dst_shape[3] // self.num_data)

    def data_move_case_three(self, tik_instance, ub_ori, ub_trans, core_loop,
                             sum_core, align_loop, remainder):
        """
        the data_move process when UB can put in H * W * C0 data and
        H * W * C0 % 16 != 0 and
        zero padding is not required for some processed N axis
        """
        with tik_instance.for_range(0, core_loop) as num_core_loop:

            total_core_loop = sum_core + num_core_loop
            num_n = total_core_loop
            src_gm_index = num_n * self.src_shape[1] * self.src_shape[2] * \
                           self.src_shape[3]
            src_ub_index = (num_core_loop % align_loop) * \
                           self.dst_shape[0] * self.dst_shape[3]
            tik_instance.data_move(ub_ori[src_ub_index],
                                   self.src_gm[src_gm_index],
                                   0, 1,
                                   self.dst_shape[0] * self.dst_shape[3] //
                                   self.num_data, 0, 0)
            with tik_instance.if_scope(tik.all((num_core_loop + 1) %
                                               align_loop == 0,
                                               num_core_loop !=
                                               core_loop - 1)):
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              align_loop)
                with tik_instance.if_scope(
                        (self.dst_shape[1] * self.dst_shape[2] -
                         align_loop) * self.dst_shape[3] //
                        self.num_data > MAX_STRIDE_BLK):
                    with tik_instance.for_range(0, self.dst_shape[0]) \
                            as num_x1:
                        dst_gm_index = num_x1 * self.dst_shape[1] * \
                                       self.dst_shape[2] * \
                                       self.dst_shape[3] + \
                                       (num_n - (align_loop - 1)) * \
                                       self.dst_shape[3]
                        with tik_instance.if_scope(align_loop == 1):
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_ori[num_x1 * align_loop *
                                                          self.dst_shape[3]],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[3] //
                                                   self.num_data, 0, 0)
                        with tik_instance.else_scope():
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans[num_x1 *
                                                            align_loop *
                                                            self.dst_shape[3]],
                                                   0, 1,
                                                   align_loop *
                                                   self.dst_shape[3] //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    dst_gm_index = (num_n - (align_loop - 1)) * \
                                   self.dst_shape[3]
                    with tik_instance.if_scope(align_loop == 1):
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_ori[0],
                                               0, self.dst_shape[0],
                                               align_loop *
                                               self.dst_shape[3] //
                                               self.num_data,
                                               0,
                                               (self.dst_shape[1] *
                                                self.dst_shape[2] -
                                                align_loop) *
                                               self.dst_shape[3] //
                                               self.num_data)
                    with tik_instance.else_scope():
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0],
                                               0, self.dst_shape[0],
                                               align_loop *
                                               self.dst_shape[3] //
                                               self.num_data,
                                               0,
                                               (self.dst_shape[1] *
                                                self.dst_shape[2] -
                                                align_loop) *
                                               self.dst_shape[3] //
                                               self.num_data)

            with tik_instance.if_scope(num_core_loop == core_loop - 1):
                self.data_rearrange_case_zero(tik_instance, ub_ori, ub_trans,
                                              remainder)
                with tik_instance.if_scope(
                        (self.dst_shape[1] * self.dst_shape[2] -
                         remainder) * self.dst_shape[3] //
                        self.num_data > MAX_STRIDE_BLK):
                    with tik_instance.for_range(0, self.dst_shape[0]) \
                            as num_x1:
                        dst_gm_index = num_x1 * self.dst_shape[1] * \
                                       self.dst_shape[2] * \
                                       self.dst_shape[3] + \
                                       (num_n - (remainder - 1)) * \
                                       self.dst_shape[3]
                        with tik_instance.if_scope(remainder == 1):
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_ori[num_x1 * remainder *
                                                          self.dst_shape[3]],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[3] //
                                                   self.num_data, 0, 0)
                        with tik_instance.else_scope():
                            tik_instance.data_move(self.dst_gm[dst_gm_index],
                                                   ub_trans[num_x1 *
                                                            remainder *
                                                            self.dst_shape[3]],
                                                   0, 1,
                                                   remainder *
                                                   self.dst_shape[3] //
                                                   self.num_data, 0, 0)
                with tik_instance.else_scope():
                    dst_gm_index = (num_n - (remainder - 1)) * \
                                   self.dst_shape[3]
                    with tik_instance.if_scope(remainder == 1):
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_ori[0],
                                               0, self.dst_shape[0],
                                               remainder * self.dst_shape[3] //
                                               self.num_data,
                                               0,
                                               (self.dst_shape[1] *
                                                self.dst_shape[2] -
                                                remainder) *
                                               self.dst_shape[3] //
                                               self.num_data)
                    with tik_instance.else_scope():
                        tik_instance.data_move(self.dst_gm[dst_gm_index],
                                               ub_trans[0],
                                               0, self.dst_shape[0],
                                               remainder * self.dst_shape[3] //
                                               self.num_data,
                                               0,
                                               (self.dst_shape[1] *
                                                self.dst_shape[2] -
                                                remainder) *
                                               self.dst_shape[3] //
                                               self.num_data)

    def data_move_case_four(self, tik_instance, ub_ori, ub_trans,
                            num_n, loop_hw, loop_remainder):
        """
        the data_move process when UB can put not in H * W * C0 data and
        zero padding is not required for some processed N axis
        """
        ori_hw = self.src_shape[1] * self.src_shape[2]
        is_last = 0
        with tik_instance.for_range(0, ori_hw // loop_hw) as num_loop_hw:

            src_gm_index = num_n * self.src_shape[3] * ori_hw + num_loop_hw * \
                           loop_hw * self.src_shape[3]
            tik_instance.data_move(ub_ori[0],
                                   self.src_gm[src_gm_index],
                                   0, 1,
                                   loop_hw * self.src_shape[3] //
                                   self.num_data,
                                   0, 0)
            self.data_rearrange_case_two(tik_instance, ub_ori, ub_trans,
                                         loop_hw, is_last)
            self.data_move_ub2gm_case_one(tik_instance, ub_ori, ub_trans,
                                          loop_hw, loop_hw, num_loop_hw,
                                          num_n)

        if loop_remainder != 0:
            is_last = 1
            src_gm_index = num_n * self.src_shape[3] * ori_hw + ori_hw // \
                           loop_hw * loop_hw * self.src_shape[3]
            tik_instance.data_move(ub_ori[0],
                                   self.src_gm[src_gm_index],
                                   0, 1,
                                   (loop_remainder * self.src_shape[3] +
                                    self.num_data - 1) // self.num_data,
                                   0, 0)
            self.data_rearrange_case_two(tik_instance, ub_ori, ub_trans,
                                         loop_remainder, is_last)
            self.data_move_ub2gm_case_one(tik_instance, ub_ori, ub_trans,
                                          loop_hw, loop_remainder,
                                          ori_hw // loop_hw, num_n)

        return tik_instance

    def data_move_case_five(self, tik_instance, ub_trans, num_n,
                            loop_hw, loop_remainder):
        """
        the data_move process when UB can put not in H * W * C0 data and
        zero padding is required for some processed N axis
        """
        ori_hw = self.src_shape[1] * self.src_shape[2]

        with tik_instance.for_range(0, ori_hw // loop_hw) as num_loop_hw:
            self.data_rearrange_case_three(tik_instance, ub_trans, loop_hw)
            if (self.dst_shape[1] * self.dst_shape[2] - 1) * \
                    self.dst_shape[3] // self.num_data > MAX_STRIDE_BLK:
                with tik_instance.for_range(0, C0 * loop_hw // CUBE_SIZE) \
                        as num_x1:
                    dst_gm_index = (C0 * loop_hw // CUBE_SIZE * num_loop_hw +
                                    num_x1) * self.dst_shape[1] * \
                                   self.dst_shape[2] * self.dst_shape[3] + \
                                   num_n * self.dst_shape[3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[num_x1 * self.dst_shape[3]],
                                           0, 1, 1, 0, 0)
            else:
                dst_gm_index = C0 * loop_hw // CUBE_SIZE * num_loop_hw * \
                               self.dst_shape[1] * self.dst_shape[2] * \
                               self.dst_shape[3] + num_n * self.dst_shape[3]
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0, C0 * loop_hw // CUBE_SIZE,
                                       1, 0,
                                       (self.dst_shape[1] *
                                        self.dst_shape[2] - 1) *
                                       self.dst_shape[3] // self.num_data)
        if loop_remainder != 0:
            self.data_rearrange_case_three(tik_instance, ub_trans,
                                           loop_remainder)
            if (self.dst_shape[1] * self.dst_shape[2] - 1) * \
                    self.dst_shape[3] // self.num_data > MAX_STRIDE_BLK:
                with tik_instance.for_range(0, (C0 * loop_remainder +
                                                CUBE_SIZE - 1) // CUBE_SIZE) \
                        as num_x1:
                    dst_gm_index = (C0 * loop_hw // CUBE_SIZE *
                                    (ori_hw // loop_hw) +
                                    num_x1) * self.dst_shape[1] * \
                                   self.dst_shape[2] * self.dst_shape[3] + \
                                   num_n * self.dst_shape[3]
                    tik_instance.data_move(self.dst_gm[dst_gm_index],
                                           ub_trans[num_x1 *
                                                    self.dst_shape[3]],
                                           0, 1, 1, 0, 0)
            else:
                dst_gm_index = C0 * loop_hw // CUBE_SIZE * \
                               (ori_hw // loop_hw) * \
                               self.dst_shape[1] * self.dst_shape[2] * \
                               self.dst_shape[3] + num_n * self.dst_shape[3]
                tik_instance.data_move(self.dst_gm[dst_gm_index],
                                       ub_trans[0],
                                       0,
                                       (C0 * loop_remainder + CUBE_SIZE - 1) //
                                       CUBE_SIZE, 1, 0,
                                       (self.dst_shape[1] *
                                        self.dst_shape[2] - 1) *
                                       self.dst_shape[3] // self.num_data)

        return tik_instance

    def format_transfer_case_zero(self, tik_instance):
        """
        the transfer process when UB can put in H * W * C0 data
        """
        ub_ori_data = self.ub_memory
        ub_trans_data = ub_ori_data
        # divide the core according to N1 * N0
        total_core_loop_num = self.dst_shape[1] * self.dst_shape[2]
        core_number = _set_core_num(total_core_loop_num)
        num_data_one_loop = self.src_shape[1] * self.src_shape[2] * C0

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_trans_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            align_loop, remainder = _cal_core_loop(tik_instance,
                                                   self.dst_shape[0] *
                                                   self.dst_shape[3],
                                                   core_loop, ub_ori_data)

            if self.src_shape[0] % CUBE_SIZE != 0:
                with tik_instance.if_scope(sum_core + core_loop - 1 <
                                           self.src_shape[0]):
                    if num_data_one_loop % CUBE_SIZE != 0 and \
                            self.src_shape[3] == 4:
                        self.data_move_case_three(tik_instance, ub_ori,
                                                  ub_trans, core_loop,
                                                  sum_core, align_loop,
                                                  remainder)
                    else:
                        self.data_move_case_zero(tik_instance, ub_ori,
                                                 ub_trans, core_loop,
                                                 sum_core, align_loop,
                                                 remainder)

                with tik_instance.if_scope(tik.all(sum_core <
                                                   self.src_shape[0],
                                                   sum_core + core_loop - 1 >=
                                                   self.src_shape[0])):
                    self.data_move_case_one(tik_instance, ub_ori, ub_trans,
                                            core_loop, sum_core)

                with tik_instance.if_scope(sum_core >= self.src_shape[0]):
                    self.data_move_case_two(tik_instance, ub_trans,
                                            core_loop, sum_core,
                                            align_loop, remainder)
            else:
                if num_data_one_loop % CUBE_SIZE != 0 and \
                        self.src_shape[3] == 4:
                    self.data_move_case_three(tik_instance, ub_ori,
                                              ub_trans, core_loop,
                                              sum_core, align_loop,
                                              remainder)
                else:
                    self.data_move_case_zero(tik_instance, ub_ori,
                                             ub_trans, core_loop,
                                             sum_core, align_loop,
                                             remainder)

        return tik_instance

    def format_transfer_case_one(self, tik_instance):
        """
        the transfer process when UB can not put in H * W * C0 data
        """
        ub_ori_data = self.ub_memory - self.ub_memory % (CUBE_SIZE * C0)
        # loop_hw is a multiple of 16
        loop_hw, loop_remainder = _cal_core_loop_python(
            C0, self.src_shape[1] * self.src_shape[2], ub_ori_data)
        # divide the core according to N1 * N0
        total_core_loop_num = self.dst_shape[1] * self.dst_shape[2]
        core_number = _set_core_num(total_core_loop_num)

        with tik_instance.for_range(0, core_number, block_num=core_number) \
                as num_core:
            ub_ori = tik_instance.Tensor(self.dtype,
                                         (ub_ori_data,),
                                         name="ub_ori",
                                         scope=tik.scope_ubuf)
            ub_trans = tik_instance.Tensor(self.dtype,
                                           (ub_ori_data,),
                                           name="ub_trans",
                                           scope=tik.scope_ubuf)
            core_loop, sum_core = _cal_core(tik_instance, total_core_loop_num,
                                            num_core, core_number)
            with tik_instance.for_range(0, core_loop) as num_core_loop:
                total_core_loop = sum_core + num_core_loop
                num_n = total_core_loop

                with tik_instance.if_scope(num_n < self.src_shape[0]):
                    self.data_move_case_four(tik_instance, ub_ori, ub_trans,
                                             num_n, loop_hw, loop_remainder)

                with tik_instance.else_scope():
                    self.data_move_case_five(tik_instance, ub_ori, num_n,
                                             loop_hw, loop_remainder)

        return tik_instance

    def nhwc_2_fractal_z_c04_compute(self):
        """
        the overall data move process
        """
        tik_instance = self.set_tik_instance()
        format_transfer_case = self.set_format_transfer_case()
        if format_transfer_case == 0:
            tik_instance = self.format_transfer_case_zero(tik_instance)
        elif format_transfer_case == 1:
            tik_instance = self.format_transfer_case_one(tik_instance)

        return tik_instance

    def get_tik_instance(self):
        """
        obtain tik instance
        """
        tik_instance = self.nhwc_2_fractal_z_c04_compute()
        tik_instance.BuildCCE(kernel_name=self.kernel_name,
                              inputs=[self.src_gm],
                              outputs=[self.dst_gm])

        return tik_instance


@para_check.check_input_type(dict, dict, str, str, str)
def nhwc_2_fractal_z_c04(src, dst, src_format, dst_format,
                         kernel_name="nhwc_2_fractal_z_c04"):
    """
    algorithm: nhwc_2_fractal_z_c04

    Parameters
    ----------
    src: dict
        dict with keys(shape, dtype) of src
    dst: dict
        dict with keys(shape, dtype) of dst
    src_format: str
        data format of src
    dst_format: str
        data format of dst
    kernel_name: str
        kernel name, default value is "nhwc_2_fractal_z_c04"

    Returns
    -------
    tik_instance: tik_instance
    """
    src_shape = src.get("shape")
    src_dtype = src.get("dtype").lower()
    para_check.check_kernel_name(kernel_name)
    para_check.check_shape_rule(src_shape)
    para_check.check_tensor_shape_size(src_shape)
    check_list = ("float16")
    para_check.check_dtype_rule(src_dtype, check_list)
    if len(src_shape) != 4:
        raise RuntimeError("nhwc_2_fractal_z_c04 only support 4D "
                           "while src shape is %s" %
                           ", ".join(src_shape))

    if src_shape[3] > 4:
        raise RuntimeError("nhwc_2_fractal_z_c04 only support C <= 4 "
                           "while src shape is %s" %
                           ", ".join(src_shape))

    if src_format.upper() != "NHWC":
        raise RuntimeError("nhwc_2_fractal_z_c04 only support %s "
                           "while src format is %s" %
                           ("NHWC", src_format))

    if dst_format.upper() != "FRACTAL_Z_C04":
        raise RuntimeError("nhwc_2_fractal_z_c04 only support %s "
                           "while dst format is %s" %
                           ("FRACTAL_Z_C04", dst_format))

    src_shape = list(src_shape)

    nhwc_2_fractal_z_c04_template = NHWC2FRACTAL4Compute(src_shape, src_dtype,
                                                         kernel_name)
    return nhwc_2_fractal_z_c04_template.get_tik_instance()
