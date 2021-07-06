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
inplace_index_add.py
"""
import math


from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector

# max int64 value
MAX_INT64_VALUE = 2**64 -1
# tiling param num
TILING_ARG_NUM = 8
# reserved ub size
RESERVED_UB_SIZE = 8 * 1024
# 8 bit
EIGHT_BIT = 8
# mask size
ONE_VECTOR_CALC_SIZE = 256
# mask of type
TYPE_BYTES_MAP = {"float16": 2, "float32": 4, "int8": 2, "uint8": 2, "int16": 2, "int32": 4}
# bytes of one block
ONE_BLOCK_SIZE = 32
# max repeat times
MAX_REPEAT_TIMES = 255
# use rate of max ubsize use
MAX_UBSIZE_USE_RATE = 0.9


# pylint: disable=too-many-arguments,too-many-instance-attributes
# pylint: disable=invalid-name,bad-continuation,attribute-defined-outside-init,unused-argument
class InplaceIndexAdd():
    """
       Function: use to store scatter base parameters
       Modify : 2021-5-18
    """

    def __init__(self, var, indices, updates, var_out, axis, kernel_name):
        """
        Init scatter base parameters
        :param var: dict
            data of input
            datatype suports float32,float16,int32,int8,uint8
        :param indices: dict
            data of indices
            datatype supports int32
        :param updates: dict
            data of updates
            datatype supports float32,float16,int32,int8,uint8
        :param var_out: dict
            data of input
        :param axis: int
            which axis to compute index add
        :param kernel_name:
        """
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.var_dtype = var.get("dtype").lower()
        self.indices_dtype = indices.get("dtype").lower()
        self.updates_dtype = updates.get("dtype").lower()
        self.var_out_dtype = var_out.get("dtype").lower()
        self.axis = axis
        self.kernel_name = kernel_name

        self.vconv_dst_dtype = "float16"
        self.vconv_dtype_bytes_size = tbe_platform.get_bit_len(
            self.vconv_dst_dtype)
        self.vconv_data_each_block = ONE_BLOCK_SIZE // self.vconv_dtype_bytes_size

        # check input attr params
        self.check_param()

        self.ai_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.ub_size_bytes = (tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB_SIZE) * MAX_UBSIZE_USE_RATE
        self.var_dtype_bytes_size = tbe_platform.get_bit_len(self.var_dtype) // EIGHT_BIT
        self.indices_dtype_bytes_size = tbe_platform.get_bit_len(self.indices_dtype) // EIGHT_BIT
        self.var_data_each_block = ONE_BLOCK_SIZE // self.var_dtype_bytes_size
        self.indices_data_each_block = ONE_BLOCK_SIZE // self.indices_dtype_bytes_size

        self.tiling_gm = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)
        self.var_gm = self.tik_instance.Tensor(self.var_dtype, (MAX_INT64_VALUE,), name="var_gm", scope=tik.scope_gm)
        self.indices_gm = self.tik_instance.Tensor(self.indices_dtype, (MAX_INT64_VALUE,), name="indices_gm", scope=tik.scope_gm)
        self.updates_gm = self.tik_instance.Tensor(self.updates_dtype, (MAX_INT64_VALUE,), name="updates_gm", scope=tik.scope_gm)
        self.var_out_gm = self.tik_instance.Tensor(self.var_out_dtype, (MAX_INT64_VALUE,), name="var_out_gm", scope=tik.scope_gm)

        # decide the mask of computation
        self.max_num_one_repeat = ONE_VECTOR_CALC_SIZE // TYPE_BYTES_MAP[self.var_dtype]

        # init some variable
        self.init_variable()
    
    def check_param(self):
        """
        Check whether the input parameters is valid or not
        """
        indices_support_dtype_list = ("int32",)
        var_support_dtype_list = ("float16", "float32", "int32", "int8", "uint8")
        if self.var_out_dtype == "bool":
            self.var_out_dtype = "int8"
        para_check.check_dtype(self.indices_dtype, indices_support_dtype_list, param_name="indices")
        para_check.check_dtype(self.var_dtype, var_support_dtype_list, param_name="var")
        if self.var_dtype != self.updates_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "update", "var", self.updates_dtype, self.var_dtype)
        if self.var_dtype != self.var_out_dtype:
            error_manager_vector.raise_err_inputs_dtype_not_equal(self.kernel_name, "var_out", "var", self.var_out_dtype, self.var_dtype)

    def init_variable(self):
        """
        init Variable
        :return:
        """
        self.indices_ub_number = 0
        self.updates_ub_number = 0
        self.index_dims = 1
        self.vconv_ub_number = None

        # state some ub tensor
        self.core_loop_index = None
        self.core_loop_index = self.tik_instance.Scalar("int32", name="core_loop_index")

        self.var_vconv_ub = None
        self.updates_vconv_ub = None
        self.var_tail_vconv_ub = None
        self.updates_tail_vconv_ub = None

        self.var_ub = None
        self.updates_ub = None
        self.indices_ub = None
        self.var_tail_ub = None
        self.updates_tail_ub = None

        self.var_read_index = None
        self.updates_read_index = None
        self.indices_loop_index = None
        self.indices_tmp = None

        self.outer_loop_start_index_every_block = None
        self.outer_loops_ub_per_block = None
        self.outer_loop_start_index_of_var = None
        self.outer_loop_start_index_of_updates = None
    
    def tiling_args(self):
        """
        get runtime params from tiling

        Parameters
        ----------
        tiling_ub: tensor, runtime params from inpalce_index_add tiling

        Returns
        -------
        None
        """
        self.tiling_mode = self.tik_instance.Scalar("int32", name="tiling_mode")
        self.core_num = self.tik_instance.Scalar("int32", name="core_num")
        self.indices_num = self.tik_instance.Scalar("int32", name="indices_num")
        self.outer_loop = self.tik_instance.Scalar("int32", name="outer_loop")
        self.outer_loops_per_block = self.tik_instance.Scalar("int32", name="outer_loops_per_block")
        self.axis_and_after_data_num_of_updates = self.tik_instance.Scalar("int32", name="axis_and_after_data_num_of_updates")
        self.axis_and_after_data_num_of_var = self.tik_instance.Scalar("int32", name="axis_and_after_data_num_of_var")
        self.update_data_num = self.tik_instance.Scalar("int32", name="update_data_num")

        self.tiling_mode.set_as(self.tiling_ub[0])
        self.core_num.set_as(self.tiling_ub[1])
        self.indices_num.set_as(self.tiling_ub[2])
        self.outer_loop.set_as(self.tiling_ub[3])
        self.outer_loops_per_block.set_as(self.tiling_ub[4])
        self.axis_and_after_data_num_of_updates.set_as(self.tiling_ub[5])
        self.axis_and_after_data_num_of_var.set_as(self.tiling_ub[6])
        self.update_data_num.set_as(self.tiling_ub[7])

    def init_ub_tensor_para(self):
        """
        calc the tiling parameters
        :return:
        """
        with self.tik_instance.if_scope(self.tiling_mode == 1):
            # if updates size * 2 is smaller than 0.9 ub size
            self.updates_ub_number = self.ub_size_bytes // 96 * 48 // self.var_dtype_bytes_size
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block
            self.indices_ub_number = self.ub_size_bytes // 10 // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            last_num = self.update_data_num % self.updates_ub_number
            if (last_num < self.var_data_each_block and self.update_data_num > self.updates_ub_number):
                self.updates_ub_number -= self.var_data_each_block
        with self.tik_instance.if_scope(self.tiling_mode == 2):
            # if indices size is smaller than 0.9 ub size
            self.indices_ub_number = self.ub_size_bytes // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            self.updates_ub_number = self.ub_size_bytes // 10 // 2 // self.var_dtype_bytes_size
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block
            last_num = self.update_data_num % self.updates_ub_number
            if (last_num < self.var_data_each_block and self.update_data_num > self.updates_ub_number):
                self.updates_ub_number -= self.var_data_each_block
        with self.tik_instance.if_scope(self.tiling_mode == 3):
            self.updates_ub_number = self.ub_size_bytes // 96 * 24 // self.var_dtype_bytes_size
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block
            self.indices_ub_number = self.ub_size_bytes // 96 * 48 // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            last_num = self.update_data_num % self.updates_ub_number
            if (last_num < self.var_data_each_block and self.update_data_num > self.updates_ub_number):
                self.updates_ub_number -= self.var_data_each_block
        with self.tik_instance.if_scope(self.tiling_mode == 4):
            self.updates_ub_number = self.ub_size_bytes // 96 * 48 // self.var_dtype_bytes_size // 2
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block

            self.vconv_ub_number = self.ub_size_bytes // 96 * 48 // self.vconv_data_each_block // 2
            self.vconv_ub_number = math.ceil(
                self.vconv_ub_number /
                self.vconv_data_each_block) * self.vconv_data_each_block

            self.indices_ub_number = self.ub_size_bytes // 10 // self.indices_dtype_bytes_size // 2
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
        with self.tik_instance.if_scope(self.tiling_mode == 5):
            self.indices_ub_number = self.ub_size_bytes // self.indices_dtype_bytes_size
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            
            self.updates_ub_number = self.ub_size_bytes // 10 // 2 // self.var_dtype_bytes_size // 2
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block

            self.vconv_ub_number = self.ub_size_bytes // 10 // 2 // self.vconv_data_each_block // 2
            self.vconv_ub_number = math.ceil(
                self.vconv_ub_number /
                self.vconv_data_each_block) * self.vconv_data_each_block
        with self.tik_instance.if_scope(self.tiling_mode == 6):
            self.updates_ub_number = self.ub_size_bytes // 96 * 24 // self.var_dtype_bytes_size // 2
            self.updates_ub_number = math.ceil(
                self.updates_ub_number /
                self.var_data_each_block) * self.var_data_each_block
            self.indices_ub_number = self.ub_size_bytes // 96 * 48 // self.indices_dtype_bytes_size // 2
            self.indices_ub_number = math.ceil(
                self.indices_ub_number /
                self.indices_data_each_block) * self.indices_data_each_block
            self.vconv_ub_number = self.updates_ub_number
        with self.tik_instance.if_scope(self.tiling_mode == 7):
            self.updates_ub_number = self.var_data_each_block
            self.indices_ub_number = (self.ub_size_bytes - 64) // self.indices_dtype_bytes_size // \
                                     self.indices_data_each_block * self.indices_data_each_block
            self.vconv_ub_number = self.updates_ub_number

    def init_ub_tensor(self):
        """
        Init the ub tensor
        :return:
        """
        # calc the tiling parameters
        self.init_ub_tensor_para()

        # init int8 or uint8 ub tensors
        self.init_ub_tensor_of_int8_uint8()

        # var/indices/updates ub create
        self.init_ub_tensor_of_non_int8_uint8()

        # init ub tensor of read index
        self.init_ub_tensor_of_read_index()

    def init_ub_tensor_of_int8_uint8(self):
        """
        init ub tensor if var dtype is in (int8, uint8)
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            self.var_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="var_vconv_ub",
                scope=tik.scope_ubuf)
            self.updates_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.vconv_ub_number,),
                name="updates_vconv_ub",
                scope=tik.scope_ubuf)

            self.var_tail_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="var_tail_vconv_ub",
                scope=tik.scope_ubuf)
            self.updates_tail_vconv_ub = self.tik_instance.Tensor(
                self.vconv_dst_dtype, (self.var_data_each_block,),
                name="updates_tail_vconv_ub",
                scope=tik.scope_ubuf)

    def init_ub_tensor_of_non_int8_uint8(self):
        """
        init ub tensor of non int8 or uint8
        :return:
        """
        self.var_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.updates_ub_number,),
            name="var_ub",
            scope=tik.scope_ubuf)
        self.updates_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.updates_ub_number,),
            name="updates_ub",
            scope=tik.scope_ubuf)
        self.indices_ub = self.tik_instance.Tensor(
            self.indices_dtype, (self.indices_ub_number,),
            name="indices_ub",
            scope=tik.scope_ubuf)

        self.var_tail_ub = self.tik_instance.Tensor(
            self.var_dtype, (self.var_data_each_block,),
            name="var_tail_ub",
            scope=tik.scope_ubuf)
        self.updates_tail_ub = self.tik_instance.Tensor(
            self.updates_dtype, (self.var_data_each_block,),
            name="updates_tail_ub",
            scope=tik.scope_ubuf)

    def init_ub_tensor_of_read_index(self):
        """
        Init ub tensor of loop/index
        :return:
        """
        self.var_read_index = self.tik_instance.Scalar("int32")
        self.var_read_index.set_as(0)

        self.updates_read_index = self.tik_instance.Scalar("int32")
        self.updates_read_index.set_as(0)

        self.indices_loop_index = self.tik_instance.Scalar("int32")
        self.indices_loop_index.set_as(0)

        self.indices_tmp = self.tik_instance.Scalar("int32")
        self.indices_tmp.set_as(0)

        self.outer_loop_start_index_every_block = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_every_block.set_as(0)

        self.outer_loops_ub_per_block = self.tik_instance.Scalar("int32")
        self.outer_loops_ub_per_block.set_as(self.outer_loop)

        self.outer_loop_start_index_of_var = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_of_var.set_as(0)

        self.outer_loop_start_index_of_updates = self.tik_instance.Scalar("int32")
        self.outer_loop_start_index_of_updates.set_as(0)

        self.index_offset = self.tik_instance.Scalar("int32", name="index_offset")

    def get_var_read_index(self, indices_ub_index):
        """
        Get var absolute index according to indices index
        :param indices_ub_index:
        :return:
        """
        self.var_read_index.set_as(self.indices_ub[indices_ub_index])

    def get_updates_read_index(self, indices_ub_index):
        """
        Get absolute updates index according to indices index
        :param indices_ub_index:
        :return:
        """
        read_index_updates = self.tik_instance.Scalar("int32", name="read_index_updates")
        read_index_updates.set_as(self.outer_loop_start_index_of_updates + indices_ub_index * self.update_data_num)
        self.updates_read_index.set_as(read_index_updates)

    def updates_the_var(self, indices_in_index, indice_num):
        """
        Update the update fragment corresponding to the index
        :param indices_in_index: start indices index
        :param indice_num: indices_num this time to update
        :return:
        """
        indices_burst_len = self.tik_instance.Scalar("int32", name="indices_burst_len")
        indices_burst_len.set_as((indice_num - 1) // self.indices_data_each_block + 1)
        with self.tik_instance.if_scope(self.indices_num == 1):
            self.tik_instance.data_move(self.indices_ub, self.indices_gm, 0, 1,
                                        indices_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(self.indices_ub,
                                        self.indices_gm[indices_in_index], 0, 1,
                                        indices_burst_len, 0, 0)

        indice_loop_num = self.tik_instance.Scalar("int32", name="indice_loop_num")
        indice_loop_num.set_as(indice_num)

        # for loop start
        with self.tik_instance.for_range(0,
                                         indice_loop_num) as indices_ub_index:
            self.get_var_read_index(indices_ub_index)
            self.get_updates_read_index(indices_ub_index + indices_in_index)
            self.var_read_index.set_as(self.var_read_index *
                                       self.update_data_num + self.outer_loop_start_index_of_var)
            self.calc_updates()

    def calc_updates(self):
        """
        Calculate updates fragment
        :return:
        """
        # calc the loop times of update data once to ub
        updates_loop = self.tik_instance.Scalar("int32", name="updates_loop")
        updates_loop.set_as(self.update_data_num // self.updates_ub_number)
        with self.tik_instance.if_scope(updates_loop > 0):
            with self.tik_instance.for_range(0, updates_loop) as loop_index:
                self.calc_updates_small(loop_index * self.updates_ub_number,
                                        self.updates_ub_number)

        # deal with tail num
        last_num_updates = self.tik_instance.Scalar("int32", name="last_num_updates")
        last_num_updates.set_as(self.update_data_num % self.updates_ub_number)
        with self.tik_instance.if_scope(last_num_updates > 0):
            self.calc_updates_small(updates_loop * self.updates_ub_number,
                                    last_num_updates)

    def calc_updates_small(self, read_index_offset, element_num):
        """
        Move corresponding updates/var to UB and calculate
        :param read_index_offset: offset index of inner ub for loop
        :param element_num: element number once ub
        :return:
        """
        updates_burst_len = self.tik_instance.Scalar("int32", name="updates_burst_len")
        updates_burst_len.set_as((element_num - 1) // self.var_data_each_block + 1)
        self.tik_instance.data_move(
            self.var_ub, self.var_gm[self.var_read_index + read_index_offset],
            0, 1, updates_burst_len, 0, 0)

        self.tik_instance.data_move(
            self.updates_ub,
            self.updates_gm[self.updates_read_index + read_index_offset], 0, 1,
            updates_burst_len, 0, 0)

        tail_ele_num = self.tik_instance.Scalar("int32", name="tail_ele_num")
        tail_ele_num.set_as(element_num % self.var_data_each_block)
        align_offset = self.tik_instance.Scalar("int32", name="align_offset")
        align_offset.set_as(0)
        align_ele_num = self.tik_instance.Scalar("int32", name="align_ele_num")

        with self.tik_instance.if_scope(tik.all(tail_ele_num != 0, self.update_data_num > self.var_data_each_block)):
            align_ele_num.set_as(
                    element_num // self.var_data_each_block *
                    self.var_data_each_block)
            align_offset.set_as(
                    read_index_offset + align_ele_num -
                    (self.var_data_each_block - tail_ele_num))
            self.tik_instance.data_move(
                self.var_tail_ub,
                self.var_gm[self.var_read_index + align_offset], 0, 1, 1, 0, 0)

            self.tik_instance.data_move(
                self.updates_tail_ub,
                self.updates_gm[self.updates_read_index + align_offset], 0, 1,
                1, 0, 0)

        # start calc repeat loop
        self.start_calc_repeat_loop(element_num)

        # compute the mask
        self.compute_mask(read_index_offset, element_num, tail_ele_num, updates_burst_len, align_offset)

    def start_calc_repeat_loop(self, element_num):
        """
        start calc repeat loop
        :param element_num:
        :return:
        """
        compute_loop = self.tik_instance.Scalar("int32", name="compute_loop")
        compute_loop.set_as(element_num // self.max_num_one_repeat // MAX_REPEAT_TIMES)

        with self.tik_instance.if_scope(compute_loop > 0):
            with self.tik_instance.for_range(0, compute_loop) as index:
                self.index_offset.set_as(index * self.max_num_one_repeat * MAX_REPEAT_TIMES)
                self.calc_process(self.max_num_one_repeat, self.index_offset,
                                  self.index_offset, self.index_offset, MAX_REPEAT_TIMES, False)
        
        last_loop = self.tik_instance.Scalar("int32", name="last_loop")
        last_loop.set_as(element_num % (self.max_num_one_repeat *
                                   MAX_REPEAT_TIMES) // self.max_num_one_repeat)

        with self.tik_instance.if_scope(last_loop > 0):
            self.index_offset.set_as(compute_loop * self.max_num_one_repeat * MAX_REPEAT_TIMES)
            self.calc_process(self.max_num_one_repeat, self.index_offset,
                              self.index_offset, self.index_offset, last_loop, False)

    def compute_mask(self, read_index_offset, element_num, tail_ele_num, updates_burst_len, align_offset):
        """
        compute the var and update data according every repeat
        :param read_index_offset:
        :param element_num:
        :param tail_ele_num:
        :param updates_burst_len:
        :param align_offset:
        :return:
        """
        compute_mask = self.tik_instance.Scalar("int32", name="compute_mask")
        compute_mask.set_as(element_num % self.max_num_one_repeat)
        with self.tik_instance.if_scope(compute_mask > 0):
            self.index_offset.set_as(
                    element_num // self.max_num_one_repeat *
                    self.max_num_one_repeat)
            with self.tik_instance.if_scope(tik.any(tail_ele_num == 0, self.update_data_num < self.var_data_each_block)):
                self.calc_process(compute_mask, self.index_offset, self.index_offset,
                                  self.index_offset, 1, False)

                self.tik_instance.data_move(
                    self.var_out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len, 0, 0)
            with self.tik_instance.else_scope():
                self.calc_process(self.var_data_each_block, 0, 0, 0, 1, True)
                self.calc_process(compute_mask, self.index_offset, self.index_offset,
                                  self.index_offset, 1, False)
                self.tik_instance.data_move(
                    self.var_out_gm[self.var_read_index + read_index_offset],
                    self.var_ub, 0, 1, updates_burst_len, 0, 0)
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(
                self.var_out_gm[self.var_read_index + read_index_offset],
                self.var_ub, 0, 1, updates_burst_len, 0, 0)

    def calc_process(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                     is_tail):
        """
        Execute the corresponding calculation instruction
        :param mask: calc mask
        :param dest_addr: dst
        :param src_addr1: src1
        :param src_addr2: src2
        :param repeat_times: repeat times
        :param is_tail: bool, is tail
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride = self.compute_paras(
            mask, dest_addr, src_addr1, src_addr2, repeat_times, is_tail)

        self.tik_instance.vadd(mask, dst_ub, src1_ub, src2_ub, repeat_times,
                                1, 1, 1, compute_repeat_stride,
                                compute_repeat_stride, compute_repeat_stride)

        if self.var_dtype in need_vconv_dtype:
            if is_tail:
                self.tik_instance.vconv(mask, "", self.var_tail_ub,
                                        self.var_tail_vconv_ub, repeat_times, 1,
                                        1, 4, 8)
            else:
                self.tik_instance.vconv(mask, "", self.var_ub[src_addr1],
                                        self.var_vconv_ub[dest_addr],
                                        repeat_times, 1, 1, 4, 8)

    def compute_paras(self, mask, dest_addr, src_addr1, src_addr2, repeat_times,
                      is_tail):
        """
        compute the computation paras
        :param mask: calc mask
        :param dest_addr: dest_addr
        :param src_addr1: src_addr1
        :param src_addr2: src_addr2
        :param repeat_times: repeat_times
        :param is_tail: is tail or not
        :return:
        """
        need_vconv_dtype = ("int8", "uint8")
        if self.var_dtype in need_vconv_dtype:
            if is_tail:
                self.tik_instance.vconv(mask, "",
                                        self.var_tail_vconv_ub[dest_addr],
                                        self.var_tail_ub[src_addr1],
                                        repeat_times, 1, 1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.updates_tail_vconv_ub[dest_addr],
                                        self.updates_tail_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_stride = 8
                src1_ub = self.var_tail_vconv_ub
                src2_ub = self.updates_tail_vconv_ub
                dst_ub = self.var_tail_vconv_ub
                mask = self.var_data_each_block
            else:
                self.tik_instance.vconv(mask, "", self.var_vconv_ub[dest_addr],
                                        self.var_ub[src_addr1], repeat_times, 1,
                                        1, 8, 4)
                self.tik_instance.vconv(mask, "",
                                        self.updates_vconv_ub[dest_addr],
                                        self.updates_ub[src_addr2],
                                        repeat_times, 1, 1, 8, 4)
                compute_repeat_stride = 8
                src1_ub = self.var_vconv_ub[src_addr1]
                src2_ub = self.updates_vconv_ub[src_addr2]
                dst_ub = self.var_vconv_ub[dest_addr]

        else:
            if is_tail:
                compute_repeat_stride = (
                        self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_tail_ub
                src2_ub = self.updates_tail_ub
                dst_ub = self.var_tail_ub
                mask = self.var_data_each_block
            else:
                compute_repeat_stride = (
                        self.max_num_one_repeat // self.var_data_each_block)
                src1_ub = self.var_ub[src_addr1]
                src2_ub = self.updates_ub[src_addr2]
                dst_ub = self.var_ub[dest_addr]

        return mask, dst_ub, src1_ub, src2_ub, compute_repeat_stride

    def traversing_indices(self):
        """
        Traversing the indices and update the var
        :return:
        """
        max_ub_idx_num = (
                self.indices_ub_number // self.index_dims * self.index_dims)
        indices_loop_num = self.tik_instance.Scalar("int32", name="indices_loop_num")
        indices_loop_num.set_as(self.indices_num // max_ub_idx_num)

        with self.tik_instance.if_scope(indices_loop_num > 0):
            with self.tik_instance.for_range(
                    0, indices_loop_num) as indices_loop_index:
                self.updates_the_var(indices_loop_index * max_ub_idx_num,
                                     max_ub_idx_num)

        indices_last_num = self.tik_instance.Scalar("int32", name="indices_last_num")
        indices_last_num.set_as(self.indices_num % max_ub_idx_num)
        with self.tik_instance.if_scope(indices_last_num > 0):
            self.updates_the_var(indices_loop_num * max_ub_idx_num,
                                 indices_last_num)

    def get_outer_loop_index_of_updates(self, outer_loop_num):
        """
        Get absolute outer loop start index of update
        :param outer_loop_num: which outer loop it belongs to
        :return: None
        """
        real_index_updates = self.tik_instance.Scalar("int32", name="real_index_updates")
        real_index_updates.set_as(outer_loop_num * self.axis_and_after_data_num_of_updates)
        self.outer_loop_start_index_of_updates.set_as(real_index_updates)

    def get_outer_loop_index_of_var(self, outer_loop_num):
        """
        get absolute outer loop start index of var
        :param outer_loop_num: which outer loop it belongs to
        :return: None
        """
        real_index_var = self.tik_instance.Scalar("int32", name="real_index_var")
        real_index_var.set_as(outer_loop_num * self.axis_and_after_data_num_of_var)
        self.outer_loop_start_index_of_var.set_as(real_index_var)

    def traversing_outer_loop_per_block(self):
        """
        traversing outer loop per block
        :return: None
        """
        with self.tik_instance.for_range(0, self.outer_loops_ub_per_block) as outer_i:
            self.get_outer_loop_index_of_var((self.outer_loop_start_index_every_block + outer_i))
            self.get_outer_loop_index_of_updates((self.outer_loop_start_index_every_block + outer_i))
            self.traversing_indices()

    def inplace_index_add_computer_tiling(self):
        """
        Main process of inplace_index_add
        """
        with self.tik_instance.for_range(0, self.ai_core_num, block_num=self.ai_core_num) as core_index:
            self.tiling_ub = self.tik_instance.Tensor("int32", (TILING_ARG_NUM,), name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1, 1, 0, 0)
            self.tiling_args()

            with self.tik_instance.if_scope(core_index < self.core_num):
                self.init_ub_tensor()
                self.outer_loop_start_index_every_block.set_as(core_index * self.outer_loops_per_block)
                self.outer_loops_ub_per_block.set_as(self.outer_loops_per_block)
                with self.tik_instance.if_scope(core_index == self.core_num - 1):
                    self.outer_loops_ub_per_block.set_as(self.outer_loop - self.outer_loop_start_index_every_block)
                self.traversing_outer_loop_per_block()

    
    def inplace_index_add_operator(self):
        """
        inplace_index_add operation
        """
        self.inplace_index_add_computer_tiling()
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=(self.var_gm, self.indices_gm, self.updates_gm),
                                   outputs=(self.var_out_gm),
                                   flowtable=[self.tiling_gm],
                                   config=opt_config)
        ub_size_bytes = self.ub_size_bytes
        tbe_context.get_context().add_compile_info(
            "vars", {
                "ub_size": ub_size_bytes,
                "core_num": self.ai_core_num,
                "var_size": self.var_dtype_bytes_size,
                "indices_size": self.indices_dtype_bytes_size,
                "vconv_size": self.vconv_dtype_bytes_size,
                "axis": self.axis
        })

        return self.tik_instance


#pylint: disable=unused-argument
@register_operator("InplaceIndexAdd")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def inplace_index_add(var, indices, updates, var_out, axis, kernel_name="index_add"):
    """
    inplace_index_add interface
    :param var: input var data
    :param indices: input indices
    :param updates: update data
    :param var_out: output
    :param axis: axis to update
    :param kernel_name:
    :return: inplace index add result will return
    """
    obj = InplaceIndexAdd(var, indices, updates, var_out, axis, kernel_name)
    return obj.inplace_index_add_operator()
