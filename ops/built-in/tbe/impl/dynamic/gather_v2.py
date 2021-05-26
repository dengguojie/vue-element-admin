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
gather_v2d
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

PARAMS_SIZE = 2 ** 31 - 1
INDICES_NUM = 2 ** 31 - 1
TILING_ARG_NUM = 32
# 100K, caches input params data
CACHE_UB_SIZE = 100 * 1024
# reserved ub size
RESERVED_UB_SIZE = 2 * 1024

# data type of int32
INT32 = "int32"
# data type of int64
INT64 = "int64"
# one block size takes up 32b
BLOCK_SIZE = 32

# A. block tiling: indices tiling
# paramsRowSize < 32
# params is not cache in UB
TILING_MODE_1 = 1
# params is cache in UB
TILING_MODE_4 = 4
# params is cache in L1
TILING_MODE_13 = 13

# paramsRow is not 32B aligned
TILING_MODE_2 = 2
# one paramsRow of data can not store in half UB
TILING_MODE_5 = 5

# paramsRow is 32B aligned
# params is not cache in UB or L1
TILING_MODE_3 = 3
# params is cache in UB
TILING_MODE_6 = 6
# params is not cache in UB or L1
TILING_MODE_7 = 7

# B. block tiling: params_pre tiling
# paramsRowSize < 32
# params is not cache in UB
TILING_MODE_8 = 8
# params is cache in UB
TILING_MODE_9 = 9

# paramsRow is 32B aligned
# params is not cache in UB or L1
TILING_MODE_10 = 10
# params is cache in UB
TILING_MODE_11 = 11
# params is not cache in UB or L1
TILING_MODE_12 = 12

TYPE_LEN_DICT = {"float16": 2, "float32": 4, "int8": 1, "uint8": 1,
                 "int16": 2, "uint16": 2, "int32": 4, "uint32": 4,
                 "int64": 8, "uint64": 8}


# pylint: disable=too-many-public-methods,invalid-name,too-many-arguments,too-many-locals
# pylint: disable=too-many-lines,too-many-instance-attributes,too-many-statements
def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """
    return (value + factor - 1) // factor


def align_value(value, factor):
    """
    Alignment based on factor.

    Parameters
    ----------
    value: input number
    factor: alignment base

    Returns
    -------
    res:
    """
    return (value + factor - 1) // factor * factor


def check_supported(x_dict, indices_dict, axis_dict, y_dict, kernel_name="GatherV2"):
    """
    Parameters
    ----------
    x_dict: input params shape, dtype and range, if shape contains -2, the process doesn't support
    """
    shape_x = x_dict.get("ori_shape")
    shape_indices = indices_dict.get("ori_shape")
    shape_axis = axis_dict.get("ori_shape")

    if -2 in shape_x:
        reason = "shape_x contains -2."
        return False, reason
    if -2 in shape_indices:
        reason = "shape_indices contains -2."
        return False, reason
    if -2 in shape_axis:
        reason = "shape_axis contains -2."
        return False, reason

    return True, ""


class GatherV2():
    """
        Function: use to store concat base parameters
    """
    def __init__(self, params_dict, indices_dict, axis_dict, y_dict, kernel_name):
        """
        constructor of GatherV2

        Parameters
        ----------
        params_dict: dict
            shape and dtype of input params
        indices_dict: dict
            shape and dtype of input indices
        axis_dict: dict
            shape and dtype of input axis
        y_dict: dict
            shape and dtype of output, should be same dtype as input
        kernel_name: str
            kernel name, default value is "GatherV2"

        Returns
        -------
        None
        """
        self.params_dtype = params_dict.get("dtype").lower()
        self.indices_dtype = indices_dict.get("dtype").lower()
        self.axis_dtype = axis_dict.get("dtype").lower()
        self.y_dtype = y_dict.get("dtype").lower()
        self.tiling_dtype = INT64
        dtype_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64", "float16", "float32")
        indices_support_dtype_list = ("int32", "int64")
        para_check.check_dtype(self.params_dtype, dtype_list, param_name="x")
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.axis_dtype, (INT32, INT64), param_name="axis")
        if self.y_dtype != self.params_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "y", "x",
                                                                  self.y_dtype, self.params_dtype)

        profile = tik.Dprofile()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.core_num = profile.get_aicore_num()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.kernel_name = kernel_name

        self.axis_shape = (1,)
        self.x_shape = (PARAMS_SIZE,)
        self.indices_shape = (INDICES_NUM,)
        self.y_shape = (PARAMS_SIZE,)

        self.params_dsize = TYPE_LEN_DICT.get(self.params_dtype)
        self.indices_dsize = TYPE_LEN_DICT.get(self.indices_dtype)
        self.block_elem = BLOCK_SIZE // self.params_dsize

        self.x = None
        self.indices = None
        self.axis = None
        self.tiling_gm = None
        self.y = None

        self.params_pre = None
        self.params_axis = None
        self.params_row = None
        self.indices_num = None

        self.cache_params = None
        self.need_core_num = None
        self.tail_process_core = None
        self.indices_num_each_core = None
        self.indices_num_remaining = None
        self.indices_loop_num = None
        self.indices_row_num_once = None
        self.indices_row_num_last = None

        self.row_num_once_ub = None
        self.row_num_once_tail_ub = None
        self.inner_loop_num = None
        self.row_num_last_ub = None
        self.row_num_last_tail_ub = None
        self.inner_loop_num_last = None

        self.params_total = None
        self.one_row_loop = None
        self.one_row_tail = None
        self.params_pre_each_core = None
        self.params_pre_remaining = None

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from gather_nd tiling

        Returns
        -------
        None
        """
        self.params_pre = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_pre")
        self.params_pre.set_as(tiling_ub[1])
        self.params_axis = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_axis")
        self.params_axis.set_as(tiling_ub[2])
        self.params_row = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_row")
        self.params_row.set_as(tiling_ub[3])
        self.indices_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num")
        self.indices_num.set_as(tiling_ub[4])

        self.cache_params = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="cache_params")
        self.cache_params.set_as(tiling_ub[5])
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.need_core_num.set_as(tiling_ub[6])
        self.tail_process_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tail_process_core")
        self.tail_process_core.set_as(tiling_ub[7])
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_each_core.set_as(tiling_ub[8])
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_num_remaining.set_as(tiling_ub[9])
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_loop_num.set_as(tiling_ub[10])
        self.indices_row_num_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_once")
        self.indices_row_num_once.set_as(tiling_ub[11])
        self.indices_row_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_last")
        self.indices_row_num_last.set_as(tiling_ub[12])

        self.row_num_once_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_ub")
        self.row_num_once_ub.set_as(tiling_ub[13])
        self.row_num_once_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_tail_ub")
        self.row_num_once_tail_ub.set_as(tiling_ub[14])
        self.inner_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num")
        self.inner_loop_num.set_as(tiling_ub[15])
        self.row_num_last_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_ub")
        self.row_num_last_ub.set_as(tiling_ub[16])
        self.row_num_last_tail_ub = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_tail_ub")
        self.row_num_last_tail_ub.set_as(tiling_ub[17])
        self.inner_loop_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="inner_loop_num_last")
        self.inner_loop_num_last.set_as(tiling_ub[18])

        self.params_total = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_total")
        self.params_total.set_as(tiling_ub[19])
        self.one_row_loop = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_loop")
        self.one_row_loop.set_as(tiling_ub[20])
        self.one_row_tail = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="one_row_tail")
        self.one_row_tail.set_as(tiling_ub[21])
        self.params_pre_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_pre_each_core")
        self.params_pre_each_core.set_as(tiling_ub[22])
        self.params_pre_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_pre_remaining")
        self.params_pre_remaining.set_as(tiling_ub[23])

    def gather_v2_compute_tiling(self):
        """
        Main process of gather_v2

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        half_ub_size = (self.ub_size - 2 * 1024) // 2
        quarter_ub_size = (self.ub_size - 2 * 1024) // 4
        remain_half_ub_size = (self.ub_size - CACHE_UB_SIZE - RESERVED_UB_SIZE) // 2
        tik_instance = self.tik_instance

        # get tiling data
        tiling_ub = tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,), name="tiling_ub",
                                        scope=tik.scope_ubuf)
        tik_instance.data_move(tiling_ub, self.tiling_gm, 0,
                               1, ceil_value(TILING_ARG_NUM * TYPE_LEN_DICT.get(self.tiling_dtype), BLOCK_SIZE),
                               0, 0)

        tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="tiling_mode")
        tiling_mode.set_as(tiling_ub[0])

        # get run info
        self.get_tiling_args(tiling_ub)

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            with tik_instance.if_scope(block_id < self.need_core_num):
                with tik_instance.if_scope(tiling_mode == TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_1(half_ub_size, quarter_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_2):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_2(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_3):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_3(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_4):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_4(remain_half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_5):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_5(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_6):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_6(remain_half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_7):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_7(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_8):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_8(remain_half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_9):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_9(half_ub_size, quarter_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_10):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_10(remain_half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_11):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_11(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_12):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_12(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_13):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_13(half_ub_size, block_id)

    def compute_mode_1(self, half_ub_size, quarter_ub_size, block_id):
        """
        compute for tiling mode 1

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + 256) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((quarter_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        block_ub = tik_instance.Tensor(self.params_dtype, ((quarter_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="block_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, self.params_pre) as pre_i:
            # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
            # a. process indices_row_num_once * indices_loop_num
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num > 0):
                    self.process_loop_mode_1(self.inner_loop_num, indices_num_offset, pre_i, quarter_ub_size,
                                             indices_ub, res_ub, block_ub, self.x)

                # a2. process row_num_once_tail_ub
                with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                    self.process_last_mode_1(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, self.x)

            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num_last > 0):
                    self.process_loop_mode_1(self.inner_loop_num_last, indices_num_offset, pre_i,
                                             quarter_ub_size, indices_ub, res_ub, block_ub, self.x)

                with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                    self.process_last_mode_1(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, self.x)

            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.tail_process_core)):
                self.process_remaining_tail_mode_1(pre_i, indices_ub, res_ub, self.x)

    def process_loop_mode_1(self, loop_num, indices_num_offset, pre_i, quarter_ub_size, indices_ub,
                            res_ub, block_ub, x_src):
        """
        previous loop_num times process for tiling mode 1

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        pre_i: params_pre
        quarter_ub_size: a quarter of ub size
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        block_ub: cache result data in UB
        x_src: source params

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, loop_num) as inner_loop_i:

            inner_indices_offset = inner_loop_i * self.row_num_once_ub

            # count num of every loop from result_ub to gm
            res_ub_to_gm_per_loop_num = quarter_ub_size / self.params_dsize
            # loop num from result_ub to gm
            res_ub_to_gm_loop_num = (self.row_num_once_ub + res_ub_to_gm_per_loop_num - 1) // res_ub_to_gm_per_loop_num
            # elements num in block_ub
            block_per_loop_num = quarter_ub_size / BLOCK_SIZE

            res_to_gm_loop_num = tik_instance.Scalar(dtype=self.tiling_dtype, name="res_to_gm_loop_num")
            row_num_once_remaining = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_once_remaining")
            res_to_gm_loop_num.set_as(res_ub_to_gm_loop_num)
            row_num_once_remaining.set_as(self.row_num_once_ub)

            with tik_instance.for_range(0, res_to_gm_loop_num) as row_g:
                with tik_instance.if_scope(res_to_gm_loop_num == 2):
                    with tik_instance.if_scope(row_g == 0):
                        row_num_once_remaining.set_as(res_ub_to_gm_per_loop_num)
                    with tik_instance.else_scope():
                        row_num_once_remaining.set_as(self.row_num_once_ub - res_ub_to_gm_per_loop_num)

                # loop num from x_src to block_ub
                x_to_block_loop_num = row_num_once_remaining // block_per_loop_num
                # compute output offset of every loop
                output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row \
                                + row_g * res_ub_to_gm_per_loop_num
                burst_len_res = ceil_value(row_num_once_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)

                with tik_instance.for_range(0, x_to_block_loop_num) as row_h:
                    with tik_instance.for_range(0, block_per_loop_num) as row_i:
                        # compute gm offset of x
                        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                            init_value=0)
                        indices_value.set_as(indices_ub[inner_indices_offset + row_g * res_ub_to_gm_per_loop_num
                                                        + (row_h * block_per_loop_num + row_i)])
                        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                        # copy params row from gm to block_ub
                        tik_instance.data_move(block_ub[self.block_elem * row_i], x_src[gm_offset], 0, 1, 1, 0, 0)

                    with tik_instance.for_range(0, block_per_loop_num) as row_i:
                        # set result to res_ub
                        res_ub_offset = (row_h * block_per_loop_num + row_i) * self.params_row
                        block_ub_offset = row_i * self.block_elem
                        with tik_instance.for_range(0, self.params_row) as i:
                            res_ub[res_ub_offset + i].set_as(block_ub[block_ub_offset + i])

                # move tail data from ub to gm
                with tik_instance.new_stmt_scope(disable_sync=True):
                    tail_indices = x_to_block_loop_num * block_per_loop_num

                    with tik_instance.for_range(tail_indices, row_num_once_remaining) as row_i:
                        # compute gm offset of x
                        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                            init_value=0)
                        indices_value.set_as(indices_ub[inner_indices_offset + row_g * res_ub_to_gm_per_loop_num
                                                        + row_i])
                        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                        # copy params row from gm to block_ub
                        tik_instance.data_move(block_ub[self.block_elem * (row_i - tail_indices)], x_src[gm_offset],
                                               0, 1, 1, 0, 0)

                with tik_instance.for_range(tail_indices, row_num_once_remaining) as row_i:
                    # set result to res_ub
                    res_ub_offset = row_i * self.params_row
                    block_ub_offset = (row_i - tail_indices) * self.block_elem
                    with tik_instance.for_range(0, self.params_row) as i:
                        res_ub[res_ub_offset + i].set_as(block_ub[block_ub_offset + i])

                # copy result data from ub to gm
                tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_last_mode_1(self, row_num_last, indices_num_offset, inner_indices_offset,
                            pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, x_src):
        """
        process row_num_last indices for tiling mode 1

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        quarter_ub_size: a quarter of ub size
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        block_ub: cache result data in UB
        x_src: source params

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # count num of every loop from result_ub to gm
        res_ub_to_gm_per_loop_num = quarter_ub_size / self.params_dsize
        # loop num from result_ub to gm
        res_ub_to_gm_loop_num = (row_num_last + res_ub_to_gm_per_loop_num - 1) // res_ub_to_gm_per_loop_num
        # elements num in block_ub
        block_per_loop_num = quarter_ub_size / BLOCK_SIZE

        res_to_gm_loop_num = tik_instance.Scalar(dtype=self.tiling_dtype, name="res_to_gm_loop_num")
        row_num_last_remaining = tik_instance.Scalar(dtype=self.tiling_dtype, name="row_num_last_remaining")
        res_to_gm_loop_num.set_as(res_ub_to_gm_loop_num)
        row_num_last_remaining.set_as(row_num_last)

        with tik_instance.for_range(0, res_to_gm_loop_num) as row_g:
            with tik_instance.if_scope(res_to_gm_loop_num == 2):
                with tik_instance.if_scope(row_g == 0):
                    row_num_last_remaining.set_as(res_ub_to_gm_per_loop_num)
                with tik_instance.else_scope():
                    row_num_last_remaining.set_as(row_num_last - res_ub_to_gm_per_loop_num)

            # loop num from x_src to block_ub
            x_to_block_loop_num = row_num_last_remaining // block_per_loop_num
            # compute output offset of every loop
            burst_len_res = ceil_value(row_num_last_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
            output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row \
                            + row_g * res_ub_to_gm_per_loop_num

            with tik_instance.for_range(0, x_to_block_loop_num) as row_h:
                with tik_instance.for_range(0, block_per_loop_num) as row_i:
                    # compute gm offset of x
                    indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                    indices_value.set_as(indices_ub[inner_indices_offset + row_g * res_ub_to_gm_per_loop_num
                                                    + (row_h * block_per_loop_num + row_i)])
                    gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                    # copy params row from gm to block_ub
                    tik_instance.data_move(block_ub[self.block_elem * row_i], x_src[gm_offset], 0, 1, 1, 0, 0)

                with tik_instance.for_range(0, block_per_loop_num) as row_i:
                    # set result to res_ub
                    res_ub_offset = (row_h * block_per_loop_num + row_i) * self.params_row
                    block_ub_offset = row_i * self.block_elem
                    with tik_instance.for_range(0, self.params_row) as i:
                        res_ub[res_ub_offset + i].set_as(block_ub[block_ub_offset + i])

            with tik_instance.new_stmt_scope(disable_sync=True):
                tail_indices = x_to_block_loop_num * block_per_loop_num

                with tik_instance.for_range(tail_indices, row_num_last_remaining) as row_i:
                    # compute gm offset of x
                    indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                    indices_value.set_as(indices_ub[inner_indices_offset + row_g * res_ub_to_gm_per_loop_num
                                                    + row_i])
                    gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                    # copy params row from gm to block_ub
                    tik_instance.data_move(block_ub[self.block_elem * (row_i - tail_indices)], x_src[gm_offset],
                                           0, 1, 1, 0, 0)

            with tik_instance.for_range(tail_indices, row_num_last_remaining) as row_i:
                # set result to res_ub
                res_ub_offset = row_i * self.params_row
                block_ub_offset = (row_i - tail_indices) * self.block_elem
                with tik_instance.for_range(0, self.params_row) as i:
                    res_ub[res_ub_offset + i].set_as(block_ub[block_ub_offset + i])

            # move result data from ub to gm
            tail_elem = (row_num_last_remaining * self.params_row) % self.block_elem
            with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                               scope=tik.scope_ubuf)
                with tik_instance.for_range(0, self.block_elem) as num_i:
                    block_ub[num_i].set_as(res_ub[row_num_last_remaining * self.params_row - self.block_elem + num_i])

                tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res - 1, 0, 0)
                tik_instance.data_move(self.y[output_offset + (row_num_last_remaining * self.params_row
                                                               - self.block_elem)],
                                       block_ub, 0, 1, 1, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_remaining_tail_mode_1(self, pre_i, indices_ub, res_ub, x_src):
        """
        process remaining tail indices in core 0 for tiling mode 1

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_num_offset = self.need_core_num * self.indices_num_each_core
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                               ceil_value(self.indices_num_remaining * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_res = ceil_value(self.indices_num_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset) * self.params_row

        with tik_instance.for_range(0, self.indices_num_remaining, thread_num=1) as row_i:
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            # copy params row from gm to block_ub
            tik_instance.data_move(block_ub, x_src[gm_offset], 0, 1, 1, 0, 0)

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[i])

        # copy result data from ub to gm
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def compute_mode_13(self, half_ub_size, block_id):
        """
        compute for tiling mode 13

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + 256) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        x_cbuf = tik_instance.Tensor(self.params_dtype, (self.l1_size // self.params_dsize,),
                                     name="x_l1", scope=tik.scope_cbuf)

        # cache params data from gm in L1
        tik_instance.data_move(x_cbuf, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        with tik_instance.for_range(0, self.params_pre) as pre_i:
            # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
            # a. process indices_row_num_once * indices_loop_num
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num > 0):
                    self.process_loop_mode_13(self.inner_loop_num, indices_num_offset, pre_i,
                                              indices_ub, res_ub, x_cbuf)

                # a2. process row_num_once_tail_ub
                with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                    self.process_last_mode_13(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, indices_ub, res_ub, x_cbuf)

            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num_last > 0):
                    self.process_loop_mode_13(self.inner_loop_num_last, indices_num_offset, pre_i,
                                             indices_ub, res_ub, x_cbuf)

                with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                    self.process_last_mode_13(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, indices_ub, res_ub, x_cbuf)

            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.tail_process_core)):
                self.process_remaining_tail_mode_13(pre_i, indices_ub, res_ub, x_cbuf)

    def process_loop_mode_13(self, loop_num, indices_num_offset, pre_i, indices_ub, res_ub, x_src):
        """
        previous loop_num times process for tiling mode 13

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source params

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                # compute gm offset of x
                indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                indices_value.set_as(indices_ub[inner_indices_offset + row_i])
                gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
                l1_offset = gm_offset // self.block_elem * self.block_elem
                ub_offset = gm_offset % self.block_elem

                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem * 2,), name="block_ub",
                                               scope=tik.scope_ubuf)
                # copy params row from gm to block_ub
                tik_instance.data_move(block_ub, x_src[l1_offset], 0, 1, 2, 0, 0)

                # set result to res_ub
                res_ub_offset = row_i * self.params_row
                with tik_instance.for_range(0, self.params_row) as i:
                    res_ub[res_ub_offset + i].set_as(block_ub[ub_offset + i])

            # copy result data from ub to gm
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_last_mode_13(self, row_num_last, indices_num_offset, inner_indices_offset,
                            pre_i, indices_ub, res_ub, x_src):
        """
        process row_num_last indices for tiling mode 13

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        pre_i: params pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source params

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
            # compute gm offset of x
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[inner_indices_offset + row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
            l1_offset = gm_offset // self.block_elem * self.block_elem
            ub_offset = gm_offset % self.block_elem

            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem * 2,), name="block_ub",
                                           scope=tik.scope_ubuf)
            # copy params row from gm to block_ub
            tik_instance.data_move(block_ub, x_src[l1_offset], 0, 1, 2, 0, 0)

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[ub_offset + i])

        # move result data from ub to gm
        tail_elem = (row_num_last * self.params_row) % self.block_elem
        with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            with tik_instance.for_range(0, self.block_elem) as num_i:
                block_ub[num_i].set_as(res_ub[row_num_last * self.params_row - self.block_elem + num_i])

            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res - 1, 0, 0)
            tik_instance.data_move(self.y[output_offset + (row_num_last * self.params_row - self.block_elem)],
                                   block_ub, 0, 1, 1, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_remaining_tail_mode_13(self, pre_i, indices_ub, res_ub, x_src):
        """
        process remaining tail indices in core 0 for tiling mode 13

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_num_offset = self.need_core_num * self.indices_num_each_core
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                               ceil_value(self.indices_num_remaining * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_res = ceil_value(self.indices_num_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset) * self.params_row

        with tik_instance.for_range(0, self.indices_num_remaining, thread_num=1) as row_i:
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
            l1_offset = gm_offset // self.block_elem * self.block_elem
            ub_offset = gm_offset % self.block_elem

            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem * 2,), name="block_ub",
                                           scope=tik.scope_ubuf)
            # copy params row from gm to block_ub
            tik_instance.data_move(block_ub, x_src[l1_offset], 0, 1, 2, 0, 0)

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(block_ub[ub_offset + i])

        # copy result data from ub to gm
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def compute_mode_4(self, half_ub_size, block_id):
        """
        compute for tiling mode 4, params row < 32b, and params data cached in UB

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + 256) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)

        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        with tik_instance.for_range(0, self.params_pre) as pre_i:
            # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
            # a. process indices_row_num_once * indices_loop_num
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num > 0):
                    self.process_loop_mode_4(self.inner_loop_num, indices_num_offset, pre_i, indices_ub, res_ub, x_ub)

                # a2. process row_num_once_tail_ub
                with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                    self.process_last_mode_4(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, indices_ub, res_ub, x_ub)

            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num_last > 0):
                    self.process_loop_mode_4(self.inner_loop_num_last, indices_num_offset, pre_i,
                                             indices_ub, res_ub, x_ub)

                with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                    self.process_last_mode_4(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                             pre_i, indices_ub, res_ub, x_ub)

            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.tail_process_core)):
                self.process_remaining_tail_mode_4(pre_i, indices_ub, res_ub, x_ub)

    def process_loop_mode_4(self, loop_num, indices_num_offset, pre_i, indices_ub, res_ub, x_ub):
        """
        previous loop_num times process for tiling mode 4

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                # compute gm offset of x
                indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                indices_value.set_as(indices_ub[inner_indices_offset + row_i])
                gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                # set result to res_ub
                res_ub_offset = row_i * self.params_row
                with tik_instance.for_range(0, self.params_row) as i:
                    res_ub[res_ub_offset + i].set_as(x_ub[gm_offset + i])

            # copy result data from ub to gm
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_last_mode_4(self, row_num_last, indices_num_offset, inner_indices_offset,
                            pre_i, indices_ub, res_ub, x_ub):
        """
        process row_num_last indices for tiling mode 4

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        pre_i: params pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
            # compute gm offset of x
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[inner_indices_offset + row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(x_ub[gm_offset + i])

        # move result data from ub to gm
        tail_elem = (row_num_last * self.params_row) % self.block_elem
        with tik_instance.if_scope(tik.all(tail_elem != 0, burst_len_res > 1)):
            block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                           scope=tik.scope_ubuf)
            with tik_instance.for_range(0, self.block_elem) as num_i:
                block_ub[num_i].set_as(res_ub[row_num_last * self.params_row - self.block_elem + num_i])

            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res - 1, 0, 0)
            tik_instance.data_move(self.y[output_offset + (row_num_last * self.params_row - self.block_elem)],
                                   block_ub, 0, 1, 1, 0, 0)
        with tik_instance.else_scope():
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_remaining_tail_mode_4(self, pre_i, indices_ub, res_ub, x_ub):
        """
        process remaining tail indices in core 0 for tiling mode 2

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_ub: source params that cache in UB

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_num_offset = self.need_core_num * self.indices_num_each_core
        # move indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                               ceil_value(self.indices_num_remaining * self.indices_dsize, BLOCK_SIZE), 0, 0)

        burst_len_res = ceil_value(self.indices_num_remaining * self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset) * self.params_row

        with tik_instance.for_range(0, self.indices_num_remaining, thread_num=1) as row_i:
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

            # set result to res_ub
            res_ub_offset = row_i * self.params_row
            with tik_instance.for_range(0, self.params_row) as i:
                res_ub[res_ub_offset + i].set_as(x_ub[gm_offset + i])

        # copy result data from ub to gm
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def compute_mode_2(self, half_ub_size, block_id):
        """
        compute for tiling mode 2

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_dsize = self.indices_dsize
        params_dsize = self.params_dsize

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.params_pre) as pre_i:
            # a. process indices_row_num_once
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * indices_dsize, BLOCK_SIZE), 0, 0)

                self.indices_inner_gather_2(self.indices_row_num_once, indices_offset, pre_i, indices_ub, res_ub)
                # process last one loop in indices_loop_num
                self.indices_inner_gather_tail_2(self.indices_row_num_once, indices_offset, pre_i, indices_ub, res_ub)

            # b. process indices_row_num_last
            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_offset = block_id * self.indices_num_each_core + \
                                 self.indices_loop_num * self.indices_row_num_once
                tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * indices_dsize, BLOCK_SIZE), 0, 0)

                # process first (indices_row_num_last - 1) loop
                self.indices_inner_gather_2(self.indices_row_num_last, indices_offset, pre_i,
                                            indices_ub, res_ub)
                # process last one loop
                self.indices_inner_gather_tail_2(self.indices_row_num_last, indices_offset, pre_i,
                                                 indices_ub, res_ub)

            # c. process indices remaining
            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
                indices_offset = self.indices_num_each_core * self.need_core_num + block_id
                self.indices_inner_gather_remain_2(indices_offset, pre_i, indices_ub, res_ub)

    def indices_inner_gather_2(self, loop_num, indices_offset, pre_i, indices_ub, res_ub):
        """
        process row_num_once_ub indices for tiling mode 2
        """
        tik_instance = self.tik_instance
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num - 1) as row_i:
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
            output_offset = (pre_i * self.indices_num + indices_offset + row_i) * self.params_row

            tik_instance.data_move(res_ub, self.x[gm_offset], 0, 1, burst_len_row, 0, 0)
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row, 0, 0)

    def indices_inner_gather_tail_2(self, loop_num, indices_offset, pre_i, indices_ub, res_ub):
        """
        process last loop num indices for tiling mode 2
        """
        tik_instance = self.tik_instance
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        indices_value.set_as(indices_ub[loop_num - 1])
        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
        output_offset = (pre_i * self.indices_num + indices_offset + loop_num - 1) * self.params_row

        tik_instance.data_move(res_ub, self.x[gm_offset], 0, 1, burst_len_row, 0, 0)

        # set tail 32B of result to block_ub
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.block_elem) as num_i:
            block_ub[num_i].set_as(res_ub[self.params_row - self.block_elem + num_i])

        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row - 1, 0, 0)
        tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0, 1, 1, 0, 0)

    def indices_inner_gather_remain_2(self, indices_offset, pre_i, indices_ub, res_ub):
        """
        process tail indices in previous indices_num_remaining core for tiling mode 2
        """
        tik_instance = self.tik_instance
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        # move one indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_offset], 0, 1, 1, 0, 0)

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        indices_value.set_as(indices_ub[0])
        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row
        output_offset = (pre_i * self.indices_num + indices_offset) * self.params_row

        # copy one params_row from gm to res_ub
        tik_instance.data_move(res_ub, self.x[gm_offset], 0, 1, burst_len_row, 0, 0)

        # set tail 32B of result to block_ub
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, self.block_elem) as num_i:
            block_ub[num_i].set_as(res_ub[self.params_row - self.block_elem + num_i])

        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row - 1, 0, 0)
        tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0, 1, 1, 0, 0)

    def compute_mode_5(self, half_ub_size, block_id):
        """
        compute for tiling mode 5

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        half_ub_params_elem = half_ub_size // self.params_dsize

        with tik_instance.for_range(0, self.params_pre) as pre_i:
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     indices_loop_i * self.indices_row_num_once
                # move indices data to ub from gm
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

                # process one indices_row_num_once
                self.process_loop_mode_5(self.indices_row_num_once, indices_num_offset, pre_i, indices_ub, res_ub,
                                         half_ub_params_elem)

            # b. process indices_row_num_last
            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data to ub from gm
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

                self.process_loop_mode_5(self.indices_row_num_last, indices_num_offset, pre_i, indices_ub, res_ub,
                                         half_ub_params_elem)

            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
                indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
                self.process_remaining_tail_mode_5(indices_num_offset, pre_i, indices_ub, res_ub, half_ub_params_elem)

    def process_loop_mode_5(self, loop_num, indices_num_offset, pre_i, indices_ub, res_ub, half_ub_params_elem):
        """
        previous loop_num times process for tiling mode 5

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        pre_i: params pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        half_ub_params_elem: number of params element that can be stored in half UB space

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub", scope=tik.scope_ubuf)
        burst_len_sub_row = ceil_value(half_ub_params_elem, self.block_elem)
        burst_len_sub_row_last = ceil_value(self.one_row_tail, self.block_elem)

        # indices_row_num_once
        with tik_instance.for_range(0, loop_num) as row_i:
            output_offset = (pre_i * self.indices_num + indices_num_offset + row_i) * self.params_row
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
            indices_value.set_as(indices_ub[row_i])
            gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

            # process the front part of one params_row
            with tik_instance.for_range(0, self.one_row_loop) as row_inner_i:
                # move half_ub_params_elem data of one row to res_ub from gm
                tik_instance.data_move(res_ub, self.x[gm_offset + row_inner_i * half_ub_params_elem], 0,
                                       1, burst_len_sub_row, 0, 0)
                # copy result data to gm from ub
                tik_instance.data_move(self.y[output_offset + row_inner_i * half_ub_params_elem], res_ub,
                                       0, 1, burst_len_sub_row, 0, 0)

            # process of one the tail part of params_row: one_row_tail
            with tik_instance.if_scope(self.one_row_tail > 0):
                # move one_row_tail data to res_ub from gm
                tik_instance.data_move(res_ub, self.x[gm_offset + (self.params_row - self.one_row_tail)], 0,
                                       1, burst_len_sub_row_last, 0, 0)

                # copy result data to gm from ub
                with tik_instance.if_scope(tik.all(self.one_row_tail % self.block_elem != 0, loop_num - 1 == row_i)):
                    with tik_instance.for_range(0, self.block_elem) as num_i:
                        block_ub[num_i].set_as(res_ub[self.one_row_tail - self.block_elem + num_i])

                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                           0, 1, burst_len_sub_row_last - 1, 0, 0)
                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                                           1, 1, 0, 0)
                with tik_instance.else_scope():
                    tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                           0, 1, burst_len_sub_row_last, 0, 0)

    def process_remaining_tail_mode_5(self, indices_num_offset, pre_i, indices_ub, res_ub, half_ub_params_elem):
        """
        process tail indices in previous indices_num_remaining core for tiling mode 5

        Parameters
        ----------
        indices_num_offset: indices num offset
        pre_i: params_row
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        half_ub_params_elem: number of params element that can be stored in half UB space

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # move one indices data to ub from gm
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

        burst_len_sub_row = ceil_value(half_ub_params_elem, self.block_elem)
        burst_len_sub_row_last = ceil_value(self.one_row_tail, self.block_elem)
        output_offset = (pre_i * self.indices_num + indices_num_offset) * self.params_row

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        indices_value.set_as(indices_ub[0])
        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

        # process the front part of one params_row: one_row_loop * half_ub_params_elem
        with tik_instance.for_range(0, self.one_row_loop) as row_inner_i:
            # move half_ub_params_elem data of one row to res_ub from gm
            tik_instance.data_move(res_ub, self.x[gm_offset + row_inner_i * half_ub_params_elem], 0,
                                   1, burst_len_sub_row, 0, 0)
            # copy result data to gm from ub
            tik_instance.data_move(self.y[output_offset + row_inner_i * half_ub_params_elem], res_ub,
                                   0, 1, burst_len_sub_row, 0, 0)

        # process of one the tail part of params_row: one_row_tail
        with tik_instance.if_scope(self.one_row_tail > 0):
            # move one_row_tail data to res_ub from gm
            tik_instance.data_move(res_ub, self.x[gm_offset + (self.params_row - self.one_row_tail)], 0,
                                   1, burst_len_sub_row_last, 0, 0)
            # copy result data to gm from ub
            with tik_instance.if_scope(self.one_row_tail % self.block_elem != 0):
                block_ub = tik_instance.Tensor(self.params_dtype, (self.block_elem,), name="block_ub",
                                               scope=tik.scope_ubuf)
                with tik_instance.for_range(0, self.block_elem) as num_i:
                    block_ub[num_i].set_as(res_ub[self.one_row_tail - self.block_elem + num_i])

                tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                       0, 1, burst_len_sub_row_last - 1, 0, 0)
                tik_instance.data_move(self.y[output_offset + (self.params_row - self.block_elem)], block_ub, 0,
                                       1, 1, 0, 0)
            with tik_instance.else_scope():
                tik_instance.data_move(self.y[output_offset + (self.params_row - self.one_row_tail)], res_ub,
                                       0, 1, burst_len_sub_row_last, 0, 0)

    def compute_mode_3(self, half_ub_size, block_id):
        """
        compute for tiling mode 3

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        self.compute_mode_32b_aligned(half_ub_size, block_id, self.x)

    def compute_mode_6(self, remain_half_ub_size, block_id):
        """
        compute for tiling mode 6

        Parameters
        ----------
        remain_half_ub_size: bytes of half remain UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)
        # cache params data from gm in UB
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        self.compute_mode_32b_aligned(remain_half_ub_size, block_id, x_ub)

    def compute_mode_32b_aligned(self, half_ub_size, block_id, x_src):
        """
        compute for tiling mode of 32B aligned

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, self.params_pre) as pre_i:
            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data to ub from gm
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num > 0):
                    self.process_loop_32b_aligned(self.inner_loop_num, indices_num_offset, pre_i,
                                                  indices_ub, res_ub, x_src)

                # a2. process row_num_once_tail_ub
                with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                    self.process_last_32b_aligned(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                                  pre_i, indices_ub, res_ub, x_src)
            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data to ub from gm
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

                with tik_instance.if_scope(self.inner_loop_num_last > 0):
                    self.process_loop_32b_aligned(self.inner_loop_num_last, indices_num_offset, pre_i,
                                                  indices_ub, res_ub, x_src)

                # b2. process row_num_last_tail_ub
                with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                    inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                    self.process_last_32b_aligned(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                                  pre_i, indices_ub, res_ub, x_src)

            with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.indices_num_remaining)):
                indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
                self.process_remaining_tail_32b_aligned(indices_num_offset, pre_i, indices_ub, res_ub, x_src)

    def process_loop_32b_aligned(self, loop_num, indices_num_offset, pre_i, indices_ub, res_ub, x_src):
        """
        previous loop_num times process for tiling mode of 32B aligned

        Parameters
        ----------
        loop_num: loop times
        indices_num_offset: indices num offset
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(self.row_num_once_ub * self.params_row * self.params_dsize, BLOCK_SIZE)
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)

        with tik_instance.for_range(0, loop_num) as inner_loop_i:
            inner_indices_offset = inner_loop_i * self.row_num_once_ub
            output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

            with tik_instance.new_stmt_scope(disable_sync=True):
                with tik_instance.for_range(0, self.row_num_once_ub, thread_num=2) as row_i:
                    indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                    indices_value.set_as(indices_ub[inner_indices_offset + row_i])
                    gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                    # move params_row from gm or UB or L1 to res_ub
                    tik_instance.data_move(res_ub[row_i * self.params_row], x_src[gm_offset], 0,
                                           1, burst_len_row, 0, 0)

            # copy result data from ub to gm
            tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_last_32b_aligned(self, row_num_last, indices_num_offset, inner_indices_offset, pre_i,
                                 indices_ub, res_ub, x_src):
        """
        process last row_num_last indices for tiling mode of 32B aligned

        Parameters
        ----------
        row_num_last: the last indices num
        indices_num_offset: indices num offset
        inner_indices_offset: inner indices num offset
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        burst_len_res = ceil_value(row_num_last * self.params_row * self.params_dsize, BLOCK_SIZE)
        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset + inner_indices_offset) * self.params_row

        with tik_instance.new_stmt_scope(disable_sync=True):
            with tik_instance.for_range(0, row_num_last, thread_num=2) as row_i:
                indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
                indices_value.set_as(indices_ub[inner_indices_offset + row_i])
                gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

                # move params_row data from gm or UB or L1 to res_ub
                tik_instance.data_move(res_ub[row_i * self.params_row], x_src[gm_offset], 0, 1, burst_len_row, 0, 0)

        # move result data from ub to gm
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_res, 0, 0)

    def process_remaining_tail_32b_aligned(self, indices_num_offset, pre_i, indices_ub, res_ub, x_src):
        """
        process tail indices in previous indices_num_remaining core for tiling mode of 32B aligned

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        # move one indices data from gm to ub
        tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

        burst_len_row = ceil_value(self.params_row * self.params_dsize, BLOCK_SIZE)
        output_offset = (pre_i * self.indices_num + indices_num_offset) * self.params_row

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value", init_value=0)
        indices_value.set_as(indices_ub[0])
        gm_offset = (pre_i * self.params_axis + indices_value) * self.params_row

        # copy one params_row from gm or UB or L1 to res_ub
        tik_instance.data_move(res_ub, x_src[gm_offset], 0, 1, burst_len_row, 0, 0)

        # copy result data from ub to gm
        tik_instance.data_move(self.y[output_offset], res_ub, 0, 1, burst_len_row, 0, 0)

    def compute_mode_7(self, half_ub_size, block_id):
        """
        compute for tiling mode 7

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        x_cbuf = tik_instance.Tensor(self.params_dtype, (self.l1_size // self.params_dsize,),
                                     name="x_l1", scope=tik.scope_cbuf)
        # cache params data from gm in UB
        tik_instance.data_move(x_cbuf, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        self.compute_mode_32b_aligned(half_ub_size, block_id, x_cbuf)

    def compute_mode_8(self, remain_half_ub_size, block_id):
        """
        compute for tiling mode 8

        Parameters
        ----------
        remain_half_ub_size: bytes of half remain UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((remain_half_ub_size + 256) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((remain_half_ub_size + BLOCK_SIZE) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)

        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        range_left = block_id * self.params_pre_each_core
        range_right = (block_id + 1) * self.params_pre_each_core
        with tik_instance.for_range(range_left, range_right) as pre_i:
            self.compute_mode_less_than_32b_tiling_params_ub(pre_i, indices_ub, res_ub, x_ub)
        with tik_instance.if_scope(tik.all(self.params_pre_remaining > 0, block_id < self.params_pre_remaining)):
            pre_i = self.need_core_num * self.params_pre_each_core + block_id
            self.compute_mode_less_than_32b_tiling_params_ub(pre_i, indices_ub, res_ub, x_ub)

    def compute_mode_less_than_32b_tiling_params_ub(self, pre_i, indices_ub, res_ub, x_src):
        """
        compute for tiling mode of less than 32B when tiling params_pre

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = indices_loop_i * self.indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_mode_4(self.inner_loop_num, indices_num_offset, pre_i, indices_ub, res_ub, x_src)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_mode_4(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                         pre_i, indices_ub, res_ub, x_src)

        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = self.indices_loop_num * self.indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                self.process_loop_mode_4(self.inner_loop_num_last, indices_num_offset, pre_i,
                                         indices_ub, res_ub, x_src)

            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                self.process_last_mode_4(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                         pre_i, indices_ub, res_ub, x_src)

    def compute_mode_9(self, half_ub_size, quarter_ub_size, block_id):
        """
        compute for tiling mode 9

        Parameters
        ----------
        half_ub_size: bytes of half remain UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        indices_ub = tik_instance.Tensor(self.indices_dtype, ((half_ub_size + 256) // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, ((quarter_ub_size + 256) // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)
        block_ub = tik_instance.Tensor(self.params_dtype, ((quarter_ub_size + 256) // self.params_dsize,),
                                     name="block_ub", scope=tik.scope_ubuf)

        range_left = block_id * self.params_pre_each_core
        range_right = (block_id + 1) * self.params_pre_each_core
        with tik_instance.for_range(range_left, range_right) as pre_i:
            self.compute_mode_less_than_32b_tiling_params(pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, self.x)
        with tik_instance.if_scope(tik.all(self.params_pre_remaining > 0, block_id < self.params_pre_remaining)):
            pre_i = self.need_core_num * self.params_pre_each_core + block_id
            self.compute_mode_less_than_32b_tiling_params(pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, self.x)

    def compute_mode_less_than_32b_tiling_params(self, pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, x_src):
        """
        compute for tiling mode of less than 32B when tiling params_pre

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # 1. indices_num_each_core: indices_row_num_once * indices_loop_num + indices_row_num_last
        # a. process indices_row_num_once * indices_loop_num
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = indices_loop_i * self.indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_mode_1(self.inner_loop_num, indices_num_offset, pre_i, quarter_ub_size,
                                         indices_ub, res_ub, block_ub, x_src)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_mode_1(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                         pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, x_src)

        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = self.indices_loop_num * self.indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                self.process_loop_mode_1(self.inner_loop_num_last, indices_num_offset, pre_i, quarter_ub_size,
                                         indices_ub, res_ub, block_ub, x_src)

            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                self.process_last_mode_1(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                         pre_i, quarter_ub_size, indices_ub, res_ub, block_ub, x_src)

    def compute_mode_10(self, remain_half_ub_size, block_id):
        """
        compute for tiling mode 10

        Parameters
        ----------
        remain_half_ub_size: bytes of half remain UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)
        # cache params data from gm in UB
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        indices_ub = tik_instance.Tensor(self.indices_dtype, (remain_half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (remain_half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        range_left = block_id * self.params_pre_each_core
        range_right = (block_id + 1) * self.params_pre_each_core
        with tik_instance.for_range(range_left, range_right) as pre_i:
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, x_ub)

        with tik_instance.if_scope(tik.all(self.params_pre_remaining > 0, block_id < self.params_pre_remaining)):
            pre_i = self.need_core_num * self.params_pre_each_core + block_id
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, x_ub)

    def compute_mode_32b_aligned_tiling_params(self, pre_i, indices_ub, res_ub, x_src):
        """
        compute for tiling mode of 32B aligned when tiling params_pre

        Parameters
        ----------
        pre_i: params_pre
        indices_ub: cache indices data in UB
        res_ub: cache result data in UB
        x_src: source x tensor

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = indices_loop_i * self.indices_row_num_once
            # move indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num > 0):
                self.process_loop_32b_aligned(self.inner_loop_num, indices_num_offset, pre_i,
                                              indices_ub, res_ub, x_src)

            # a2. process row_num_once_tail_ub
            with tik_instance.if_scope(self.row_num_once_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num * self.row_num_once_ub
                self.process_last_32b_aligned(self.row_num_once_tail_ub, indices_num_offset, inner_indices_offset,
                                              pre_i, indices_ub, res_ub, x_src)
        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = self.indices_loop_num * self.indices_row_num_once
            # copy indices data to ub from gm
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                   ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.inner_loop_num_last > 0):
                self.process_loop_32b_aligned(self.inner_loop_num_last, indices_num_offset, pre_i,
                                              indices_ub, res_ub, x_src)

            # b2. process row_num_last_tail_ub
            with tik_instance.if_scope(self.row_num_last_tail_ub > 0):
                inner_indices_offset = self.inner_loop_num_last * self.row_num_last_ub
                self.process_last_32b_aligned(self.row_num_last_tail_ub, indices_num_offset, inner_indices_offset,
                                              pre_i, indices_ub, res_ub, x_src)

    def compute_mode_11(self, half_ub_size, block_id):
        """
        compute for tiling mode 11

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        x_cbuf = tik_instance.Tensor(self.params_dtype, (self.l1_size // self.params_dsize,),
                                     name="x_l1", scope=tik.scope_cbuf)
        # cache params data from gm in L1
        tik_instance.data_move(x_cbuf, self.x, 0, 1, ceil_value(self.params_total, self.block_elem), 0, 0)

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        range_left = block_id * self.params_pre_each_core
        range_right = (block_id + 1) * self.params_pre_each_core
        with tik_instance.for_range(range_left, range_right) as pre_i:
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, x_cbuf)

        with tik_instance.if_scope(tik.all(self.params_pre_remaining > 0, block_id < self.params_pre_remaining)):
            pre_i = self.need_core_num * self.params_pre_each_core + block_id
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, x_cbuf)

    def compute_mode_12(self, half_ub_size, block_id):
        """
        compute for tiling mode 12

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                         name="indices_ub", scope=tik.scope_ubuf)
        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

        range_left = block_id * self.params_pre_each_core
        range_right = (block_id + 1) * self.params_pre_each_core
        with tik_instance.for_range(range_left, range_right) as pre_i:
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, self.x)

        with tik_instance.if_scope(tik.all(self.params_pre_remaining > 0, block_id < self.params_pre_remaining)):
            pre_i = self.need_core_num * self.params_pre_each_core + block_id
            self.compute_mode_32b_aligned_tiling_params(pre_i, indices_ub, res_ub, self.x)

    def gather_v2_compute(self):
        """
        compute of gather_v2

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.x = self.tik_instance.Tensor(self.params_dtype, self.x_shape,
                                          name="x", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape,
                                                name="indices", scope=tik.scope_gm)
        self.axis = self.tik_instance.Tensor(self.axis_dtype, self.axis_shape,
                                             name="axis", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                                  name="ddr_arg", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(self.y_dtype, shape=self.y_shape,
                                          name="y", scope=tik.scope_gm)

        self.gather_v2_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }

        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "l1_size": self.l1_size,
                                        "params_dsize": self.params_dsize,
                                        "indices_dsize": self.indices_dsize
                                        })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices, self.axis),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True, config=opt_config)

    def gather_compute(self):
        """
        compute of gather

        Parameters
        ----------
        None

        Returns
        -------
        compile info
        """
        self.x = self.tik_instance.Tensor(self.params_dtype, self.x_shape,
                                          name="x", scope=tik.scope_gm)
        self.indices = self.tik_instance.Tensor(self.indices_dtype, self.indices_shape,
                                                name="indices", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, (TILING_ARG_NUM,),
                                                  name="ddr_arg", scope=tik.scope_gm)
        self.y = self.tik_instance.Tensor(self.y_dtype, shape=self.y_shape,
                                          name="y", scope=tik.scope_gm)

        self.gather_v2_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }

        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "l1_size": self.l1_size,
                                        "params_dsize": self.params_dsize,
                                        "indices_dsize": self.indices_dsize
                                        })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True, config=opt_config)


@register_operator("GatherV2")
def gather_v2(x_dict, indices_dict, axis_dict, y_dict, kernel_name="GatherV2"):
    """
    gather_v2 interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    indices_dict: input indices shape, dtype and range
    axis_dict: input axis shape, dtype and range
    y_dict: output shape, dtype and range
    kernel_name: kernel name of gather_v2 op

    Returns
    -------
    compile info
    """
    obj = GatherV2(x_dict, indices_dict, axis_dict, y_dict, kernel_name)
    return obj.gather_v2_compute()
