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
gather_elements
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context

PARAMS_SIZE = 2 ** 31 - 1
INDICES_NUM = 2 ** 31 - 1
TILING_ARG_NUM = 14
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

# A: params larger than cache_ub
# B: indices larger than the number contained in one block for each core
# C: remaining indices larger than one block

# A
TILING_MODE_1 = 1
# B
TILING_MODE_4 = 4
# 
TILING_MODE_2 = 2
# B C
TILING_MODE_5 = 5
# A B
TILING_MODE_3 = 3
# A B C
TILING_MODE_6 = 6


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



class GatherElements():
    """
        Function: use to store concat base parameters
    """
    def output_matrix(self, indices_index):
        """
        compute final output matrix index

        Parameters
        ----------
        indices_index: index of indices

        Returns
        -------
        tail_row, loop_pre
        """
        tail_row = indices_index % self.params_row
        loop_pre = indices_index // (self.params_row * self.params_axis)

        return tail_row, loop_pre

    def __init__(self, params_dict, indices_dict, y_dict, axis, kernel_name):
        """
        constructor of GatherElements

        Parameters
        ----------
        params_dict: dict
            shape and dtype of input params
        indices_dict: dict
            shape and dtype of input indices
        axis: int
            which axis to gather on
        y_dict: dict
            shape and dtype of output, should be same dtype as input
        kernel_name: str
            kernel name, default value is "GatherElements"

        Returns
        -------
        None
        """
        self.params_dtype = params_dict.get("dtype").lower()
        self.indices_dtype = indices_dict.get("dtype").lower()
        self.y_dtype = y_dict.get("dtype").lower()
        self.tiling_dtype = INT64

        self.check_params(kernel_name)

        profile = tik.Dprofile()
        self.ub_size = profile.get_unified_buffer_size()
        self.l1_size = profile.get_l1_buffer_size()
        self.core_num = profile.get_aicore_num()
        self.tik_instance = tik.Tik(profile, disable_debug=True)
        self.kernel_name = kernel_name

        self.x_shape = (PARAMS_SIZE,)
        self.indices_shape = (INDICES_NUM,)
        self.y_shape = (INDICES_NUM,)

        self.params_dsize = TYPE_LEN_DICT.get(self.params_dtype)
        self.indices_dsize = TYPE_LEN_DICT.get(self.indices_dtype)

        self.x = None
        self.indices = None
        self.axis = axis
        self.tiling_gm = None
        self.y = None

        self.params_pre = None
        self.params_axis = None
        self.params_row = None
        self.indices_num = None

        self.need_core_num = None
        self.indices_num_each_core = None
        self.indices_num_remaining = None
        self.indices_loop_num = None
        self.indices_row_num_once = None
        self.indices_row_num_last = None

        self.params_total = None
        self.remaining_block_remain = None
        self.remaining_block_num = None

    def check_params(self, kernel_name):
        """
        check params

        Parameters
        ----------
        kernel_name

        Returns
        -------
        None
        """       
        dtype_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                      "uint32", "uint64", "float16", "float32")

        indices_support_dtype_list = ("int32", "int64")

        para_check.check_dtype(self.params_dtype, dtype_list, param_name="x")
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")

        if self.y_dtype != self.params_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(kernel_name, "y", "x",
                                                                  self.y_dtype, self.params_dtype)

    def get_tiling_args(self, tiling_ub):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from gather_elements tiling

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

        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="need_core_num")
        self.need_core_num.set_as(tiling_ub[5])
        self.indices_num_each_core = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_each_core")
        self.indices_num_each_core.set_as(tiling_ub[6])
        self.indices_num_remaining = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_num_remaining")
        self.indices_num_remaining.set_as(tiling_ub[7])
        self.indices_loop_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_loop_num")
        self.indices_loop_num.set_as(tiling_ub[8])
        self.indices_row_num_once = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_once")
        self.indices_row_num_once.set_as(tiling_ub[9])
        self.indices_row_num_last = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="indices_row_num_last")
        self.indices_row_num_last.set_as(tiling_ub[10])

        self.params_total = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="params_total")
        self.params_total.set_as(tiling_ub[11])
        self.remaining_block_remain = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="remaining_block_remain")
        self.remaining_block_remain.set_as(tiling_ub[12])
        self.remaining_block_num = self.tik_instance.Scalar(dtype=self.tiling_dtype, name="remaining_block_num")
        self.remaining_block_num.set_as(tiling_ub[13])                

    def gather_elements_compute_tiling(self):
        """
        Main process of gather_elements

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance
        half_ub_size = (self.ub_size - 2 * 1024) // 2
        remain_half_ub_size = (self.ub_size - CACHE_UB_SIZE - RESERVED_UB_SIZE) // 2

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

        y_32b_unaligned = tik_instance.Tensor(self.y_dtype, self.y_shape, name="y_32b_unaligned", 
                                            scope=tik.scope_gm, is_workspace=True)

        with tik_instance.for_range(0, self.core_num, block_num=self.core_num) as block_id:
            with tik_instance.if_scope(block_id < self.need_core_num):
                with tik_instance.if_scope(tiling_mode == TILING_MODE_1):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_1(half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_2):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_2(remain_half_ub_size, block_id)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_3):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_3(half_ub_size, block_id, y_32b_unaligned)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_4):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_4(remain_half_ub_size, block_id, y_32b_unaligned)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_5):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_5(remain_half_ub_size, block_id, y_32b_unaligned)
                with tik_instance.if_scope(tiling_mode == TILING_MODE_6):
                    with tik_instance.new_stmt_scope():
                        self.compute_mode_6(half_ub_size, block_id, y_32b_unaligned)

        # only do aligned for unaligned data
        with tik_instance.if_scope(tiling_mode == TILING_MODE_3):
            self.process_aligned(half_ub_size, y_32b_unaligned, self.remaining_block_num + 1)
        with tik_instance.if_scope(tiling_mode == TILING_MODE_4):
            self.process_aligned(half_ub_size, y_32b_unaligned, self.remaining_block_num + 1)
        with tik_instance.if_scope(tiling_mode == TILING_MODE_5):
            self.process_aligned(half_ub_size, y_32b_unaligned, self.remaining_block_num + 1)
        with tik_instance.if_scope(tiling_mode == TILING_MODE_6):
            self.process_aligned(half_ub_size, y_32b_unaligned, self.remaining_block_num + 1)
    
    def process_aligned(self, half_ub_size, y_32b_unaligned, remain_block):
        """
        align unligned ub data

        Parameters
        ----------
        half_ub_size: bytes of half UB
        y_32b_unaligned: unaligned y in gm 
        remain_block: remaining indices blocks amount

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance       

        # amount of block to contain indices number on one core
        block_num_each_core = ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE)
        # amount of indices number on one core
        unaligned_num_each_core = block_num_each_core * (BLOCK_SIZE // self.indices_dsize)

        unaligned_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.params_dsize,), name="unaligned_ub",
                                        scope=tik.scope_ubuf)

        tik_instance.data_move(unaligned_ub, y_32b_unaligned, 0, 1, block_num_each_core * self.need_core_num + remain_block, 0, 0)

        with tik_instance.for_range(0, self.need_core_num) as block_id:
            # move unaligned_num_each_core to indices_num_each_core
            res_offset = block_id * unaligned_num_each_core
            tik_instance.data_move(self.y[block_id * self.indices_num_each_core], unaligned_ub[res_offset], 0, 1, block_num_each_core, 0, 0)

        # move remaining
        tik_instance.data_move(self.y[self.need_core_num * self.indices_num_each_core], 
                                unaligned_ub[self.need_core_num * unaligned_num_each_core], 0, 1, remain_block, 0, 0)

    def compute_mode_1(self, half_ub_size, block_id):
        """
        compute for tiling mode 1
        params larger than cache_ub
        indices lessthan the number contained in one block for each core

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        with tik_instance.if_scope(block_id < 1):

            indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                            name="indices_ub", scope=tik.scope_ubuf)

            res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

            self.compute_x_larger_cache(block_id, indices_ub, res_ub, self.x, self.y)

    def compute_mode_2(self, half_ub_size, block_id):
        """
        compute for tiling mode 2
        params less than cache_ub
        indices less than the number contained in one block for each core

        Parameters
        ----------
        half_ub_size: bytes of remain half UB
        block_id: id of ai core

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)
        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total,  BLOCK_SIZE // self.params_dsize), 0, 0)

        with tik_instance.if_scope(block_id < 1):

            indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                            name="indices_ub", scope=tik.scope_ubuf)

            res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                     name="res_ub", scope=tik.scope_ubuf)

            with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
                indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
                # move indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

            with tik_instance.if_scope(self.indices_row_num_last > 0):
                indices_num_offset = block_id * self.indices_num_each_core + \
                                     self.indices_loop_num * self.indices_row_num_once
                # copy indices data from gm to ub
                tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                       ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)
                                       
            self.compute_x_less_cache(block_id, indices_ub, res_ub, x_ub, self.y)

    def compute_mode_3(self, half_ub_size, block_id, y_dst):
        """
        compute for tiling mode 3
        params larger than cache_ub
        indices larger than the number contained in one block for each core
        remaining indices less than one block

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        y_dst: y workspace

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # amount of block to contain indices number on one core
        block_num_each_core = ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE)
        res_offset = block_id * block_num_each_core * (BLOCK_SIZE // self.indices_dsize)

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                        name="indices_ub", scope=tik.scope_ubuf)

        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                    name="res_ub", scope=tik.scope_ubuf)

        # move loop indices data on one core to ub from gm
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                    self.indices_loop_num * self.indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

        self.compute_x_larger_cache(block_id, indices_ub, res_ub, self.x, y_dst[res_offset])

        # process indices remaining
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == 0)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id
            y_offset = self.need_core_num * block_num_each_core * (BLOCK_SIZE // self.indices_dsize)
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)
            self.process_remaining_less_block_x_block(indices_num_offset, indices_ub, res_ub, self.x)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)

    def compute_mode_4(self, half_ub_size, block_id, y_dst):
        """
        compute for tiling mode 4
        params less than cache_ub
        indices larger than the number contained in one block for each core
        remaining indices less than one block

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        y_dst: y workspace

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        indices_loop_num = self.indices_num_each_core / half_ub_size
        indices_row_num_once = half_ub_size
        indices_row_num_last = self.indices_num_each_core % indices_row_num_once

        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                    name="x_ub", scope=tik.scope_ubuf)
        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total, BLOCK_SIZE // self.params_dsize), 0, 0)

        # amount of block to contain indices number on one core
        block_num_each_core = ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE)
        res_offset = block_id * block_num_each_core * (BLOCK_SIZE // self.indices_dsize)

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                        name="indices_ub", scope=tik.scope_ubuf)

        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                    name="res_ub", scope=tik.scope_ubuf)

        # move loop indices data on one core to ub from gm
        with tik_instance.for_range(0, indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

        with tik_instance.if_scope(indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_num * indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

        self.compute_x_less_cache(block_id, indices_ub, res_ub, x_ub, y_dst[res_offset])

        # process indices remaining
        # remain_num <= indices_num per core, using one block to move
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == 0)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num
            y_offset = self.need_core_num * block_num_each_core * (BLOCK_SIZE // self.indices_dsize)

            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

            self.process_remaining_less_block(indices_num_offset, indices_ub, res_ub, x_ub)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)

    def compute_mode_5(self, half_ub_size, block_id, y_dst):
        """
        compute for tiling mode 5
        params less than cache_ub
        indices larger than the number contained in one block for each core
        remaining indices larger than one block

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        y_dst: y workspace

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        indices_loop_num = self.indices_num_each_core / half_ub_size
        indices_row_num_once = half_ub_size
        indices_row_num_last = self.indices_num_each_core % indices_row_num_once

        x_ub = tik_instance.Tensor(self.params_dtype, (CACHE_UB_SIZE // self.params_dsize,),
                                   name="x_ub", scope=tik.scope_ubuf)
        # cache params data in UB from gm
        tik_instance.data_move(x_ub, self.x, 0, 1, ceil_value(self.params_total,  BLOCK_SIZE // self.params_dsize), 0, 0)

        # amount of block to contain indices number on one core
        block_num_each_core = ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE)
        indices_num_each_block  = BLOCK_SIZE // self.indices_dsize
        res_offset = block_id * block_num_each_core * indices_num_each_block

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                        name="indices_ub", scope=tik.scope_ubuf)

        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                    name="res_ub", scope=tik.scope_ubuf)

        # move loop indices data on one core to ub from gm
        with tik_instance.for_range(0, indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

        with tik_instance.if_scope(indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                    self.indices_loop_num * indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

        self.compute_x_less_cache(block_id, indices_ub, res_ub, x_ub, y_dst[res_offset])

        # process indices remaining
        # move blocks data first
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.remaining_block_num)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id * indices_num_each_block
            y_offset = (self.need_core_num * block_num_each_core + block_id) * indices_num_each_block

            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

            self.process_remaining_num_block(indices_num_each_block, indices_num_offset, indices_ub, res_ub, x_ub)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)

        # move remain of remaining
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.remaining_block_num)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id * indices_num_each_block
            y_offset = (self.need_core_num * block_num_each_core + self.remaining_block_num) * indices_num_each_block

            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

            self.process_remaining_num_remain(indices_num_offset, indices_ub, res_ub, x_ub)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)        
    

    def compute_mode_6(self, half_ub_size, block_id, y_dst):
        """
        compute for tiling mode 6
        params larger than cache_ub
        indices larger than the number contained in one block for each core
        remaining indices larger than one block

        Parameters
        ----------
        half_ub_size: bytes of half UB
        block_id: id of ai core
        y_dst: y workspace

        Returns
        -------
        None
        """
        tik_instance = self.tik_instance

        # amount of block to contain indices number on one core
        block_num_each_core = ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE)
        indices_num_each_block = BLOCK_SIZE //self.indices_dsize
        res_offset = block_id * block_num_each_core * indices_num_each_block

        indices_ub = tik_instance.Tensor(self.indices_dtype, (half_ub_size // self.indices_dsize,),
                                        name="indices_ub", scope=tik.scope_ubuf)

        res_ub = tik_instance.Tensor(self.params_dtype, (half_ub_size // self.indices_dsize,),
                                    name="res_ub", scope=tik.scope_ubuf)

        # move loop indices data on one core to ub from gm
        with tik_instance.for_range(0, self.indices_loop_num) as indices_loop_i:
            indices_num_offset = block_id * self.indices_num_each_core + indices_loop_i * self.indices_row_num_once
            # move indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(self.indices_row_num_once * self.indices_dsize, BLOCK_SIZE), 0, 0)

        with tik_instance.if_scope(self.indices_row_num_last > 0):
            indices_num_offset = block_id * self.indices_num_each_core + \
                                    self.indices_loop_num * self.indices_row_num_once
            # copy indices data from gm to ub
            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1,
                                    ceil_value(self.indices_row_num_last * self.indices_dsize, BLOCK_SIZE), 0, 0)

        self.compute_x_larger_cache(block_id, indices_ub, res_ub, self.x, y_dst[res_offset])

        # process indices remaining
        # move blocks data first
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id < self.remaining_block_num)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id * indices_num_each_block
            y_offset = (self.need_core_num * block_num_each_core + block_id) * indices_num_each_block

            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)
            
            self.process_remaining_num_block_x_block(indices_num_each_block, indices_num_offset, indices_ub, res_ub, self.x)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)

        # move remain of remaining
        with tik_instance.if_scope(tik.all(self.indices_num_remaining > 0, block_id == self.remaining_block_num)):
            indices_num_offset = self.indices_num_each_core * self.need_core_num + block_id * indices_num_each_block
            y_offset = (self.need_core_num * block_num_each_core + self.remaining_block_num) * indices_num_each_block

            tik_instance.data_move(indices_ub, self.indices[indices_num_offset], 0, 1, 1, 0, 0)

            self.process_remaining_num_remain_x_block(indices_num_offset, indices_ub, res_ub, self.x)       
            # copy result data from ub to gm
            tik_instance.data_move(y_dst[y_offset], res_ub, 0, 1, 1, 0, 0)        
    


    def compute_x_larger_cache(self, block_id, indices_ub, res_ub, x_src, y_dst):
        """
        move data x larger than cache

        Parameters
        ----------
        block_id: id of ai core
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params
        y_dst: output 
        
        Returns
        -------
        None
        """        
        tik_instance = self.tik_instance
        
        # move data by indices on one core
        with tik_instance.for_range(0, self.indices_num_each_core) as index_i:
            self.process_block(block_id, index_i, indices_ub, res_ub, x_src)

        # copy one core result data from ub to gm
        tik_instance.data_move(y_dst, res_ub, 0, 1, ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE), 0, 0)

    def compute_x_less_cache(self, block_id, indices_ub, res_ub, x_src, y_dst):
        """
        move data x less than cache

        Parameters
        ----------
        block_id: id of ai core
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params
        y_dst: output destination

        Returns
        -------
        None
        """        
        tik_instance = self.tik_instance
        
        # move data by indices on one core
        with tik_instance.for_range(0, self.indices_num_each_core) as index_i:
            self.process_loop(block_id, index_i, indices_ub, res_ub, x_src)

        # copy one core result data from ub to gm
        tik_instance.data_move(y_dst, res_ub, 0, 1, ceil_value(self.indices_num_each_core * self.indices_dsize, BLOCK_SIZE), 0, 0)

    def process_loop(self, block_id, index_i, indices_ub, res_ub, x_src):
        """
        indices num each core times process for moving data x to res

        Parameters
        ----------
        block_id: id of ai core
        index_i: current number in indices on one core
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        indices_offset = block_id * self.indices_num_each_core
        tail_row, loop_pre = self.output_matrix(indices_offset + index_i)

        with tik_instance.new_stmt_scope(disable_sync=True):
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                init_value=0)
            indices_value.set_as(indices_ub[index_i])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            res_ub[index_i].set_as(x_src[gm_offset])
    
    def process_remaining_num_block(self, indices_num_each_block, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices' divided part which cannot be divided by num per block

        Parameters
        ----------
        indices_num_each_block: amount of indices in one block
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                            init_value=0)
        with tik_instance.for_range(0, indices_num_each_block) as indices_num_i:
            tail_row, loop_pre = self.output_matrix(indices_num_offset + indices_num_i)
            indices_value.set_as(indices_ub[indices_num_i])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            res_ub[indices_num_i].set_as(x_src[gm_offset])

    def process_remaining_num_remain(self, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices' remain part which cannot be divided by num per block

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance

        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                            init_value=0)
        with tik_instance.for_range(0, self.remaining_block_remain) as indices_num_i:
            tail_row, loop_pre = self.output_matrix(indices_num_offset + indices_num_i)
            indices_value.set_as(indices_ub[indices_num_i])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            res_ub[indices_num_i].set_as(x_src[gm_offset])

    def process_remaining_less_block(self, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices in one block

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        
        indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                            init_value=0)
        with tik_instance.for_range(0, self.indices_num_remaining) as indices_num_i:
            tail_row, loop_pre = self.output_matrix(indices_num_offset + indices_num_i)
            indices_value.set_as(indices_ub[indices_num_i])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            res_ub[indices_num_i].set_as(x_src[gm_offset])


    def process_remaining_less_block_x_block(self, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices in one block
        move x data by block

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        
        with tik_instance.new_stmt_scope(disable_sync=True):
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                init_value=0)            
            tail_row, loop_pre = self.output_matrix(indices_num_offset)
            indices_value.set_as(indices_ub[0])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            block_ub = tik_instance.Tensor(self.params_dtype, (BLOCK_SIZE // self.params_dsize,), name="block_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(block_ub, x_src[gm_offset], 0, 1, 1, 0, 0)
            with tik_instance.for_range(0, self.indices_num_remaining) as indices_num_i:
                res_ub[indices_num_i].set_as(block_ub[indices_num_i])

    def process_remaining_num_remain_x_block(self, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices' remain part which cannot be divided by num per block
        move x data by block

        Parameters
        ----------
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        
        with tik_instance.new_stmt_scope(disable_sync=True):
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                init_value=0)     
            with tik_instance.for_range(0, self.remaining_block_remain) as indices_num_i:
                tail_row, loop_pre = self.output_matrix(indices_num_offset + indices_num_i)
                indices_value.set_as(indices_ub[indices_num_i])
                gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

                block_ub = tik_instance.Tensor(self.params_dtype, (BLOCK_SIZE // self.params_dsize,), name="block_ub", scope=tik.scope_ubuf)

                tik_instance.data_move(block_ub, x_src[gm_offset], 0, 1, 1, 0, 0)
                res_ub[indices_num_i].set_as(block_ub)

    def process_remaining_num_block_x_block(self, indices_num_each_block, indices_num_offset, indices_ub, res_ub, x_src):
        """
        dealing with remaining indices' block part which cannot be divided by num per block
        move x data by block

        Parameters
        ----------
        indices_num_each_block: amount of indices in one block
        indices_num_offset: indices num offset
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        
        with tik_instance.new_stmt_scope(disable_sync=True):
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                init_value=0)     
            with tik_instance.for_range(0, indices_num_each_block) as indices_num_i:
                tail_row, loop_pre = self.output_matrix(indices_num_offset + indices_num_i)
                indices_value.set_as(indices_ub[indices_num_i])
                gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

                block_ub = tik_instance.Tensor(self.params_dtype, (BLOCK_SIZE // self.params_dsize,), name="block_ub", scope=tik.scope_ubuf)

                tik_instance.data_move(block_ub, x_src[gm_offset], 0, 1, 1, 0, 0)
                res_ub[indices_num_i].set_as(block_ub)

    def process_block(self, block_id, index_i, indices_ub, res_ub, x_src):
        """
        indices num each core times process for movine data x to res

        Parameters
        ----------
        block_id: id of ai core
        index_i: current number in indices on one core
        indices_ub: indices ub
        res_ub: result ub
        x_src: source params

        Returns
        -------
        None
        """   
        tik_instance = self.tik_instance
        indices_offset = block_id * self.indices_num_each_core
        tail_row, loop_pre = self.output_matrix(indices_offset + index_i)

        with tik_instance.new_stmt_scope(disable_sync=True):
            indices_value = tik_instance.Scalar(dtype=self.indices_dtype, name="indices_value",
                                                init_value=0)
            indices_value.set_as(indices_ub[index_i])
            gm_offset = (loop_pre * self.params_axis + indices_value) * self.params_row + tail_row

            block_ub = tik_instance.Tensor(self.params_dtype, (BLOCK_SIZE // self.params_dsize,), name="block_ub", scope=tik.scope_ubuf)
            
            tik_instance.data_move(block_ub, x_src[gm_offset], 0, 1, 1, 0, 0)
            res_ub[index_i].set_as(block_ub)

    def gather_elements_compute(self):
        """
        compute of gather_elements

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

        self.gather_elements_compute_tiling()

        opt_config = {
            "out_of_bound_sync_check": True,
            "enable_const_fold": True
        }

        # add compile info
        tbe_context.get_context().add_compile_info("vars", {"core_num": self.core_num,
                                        "ub_size": self.ub_size,
                                        "l1_size": self.l1_size,
                                        "params_dsize": self.params_dsize,
                                        "indices_dsize": self.indices_dsize,
                                        "attr": self.axis
                                        })

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.x, self.indices),
                                   outputs=(self.y,),
                                   flowtable=(self.tiling_gm,), enable_l2=True, config=opt_config)


@register_operator("GatherElements")
def gather_elements(x_dict, indices_dict, y_dict, axis=0, kernel_name="GatherElements"):
    """
    gather_elements interface

    Parameters
    ----------
    x_dict: input params shape, dtype and range
    indices_dict: input indices shape, dtype and range
    y_dict: output shape, dtype and range
    axis: which axis to gather on, attr
    kernel_name: kernel name of gather_elements op

    Returns
    -------
    compile info
    """
    obj = GatherElements(x_dict, indices_dict, y_dict, axis, kernel_name)
    return obj.gather_elements_compute()
