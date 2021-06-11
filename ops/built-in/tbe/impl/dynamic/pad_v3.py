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
pad_v3.py
"""
import functools
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator
from impl import constant_util as constant


# max int64
MAX_INT64 = 2 ** 64 - 1
# tiling param nums
TILING_NUMS = 20
# 1 byte = 8 bit
EIGHT_BIT = 8
# bytes of one block
BLOCK_BYTES = 32
# reserved ub size
RESERVED_UB = 1024
# vnchw the minest block
TRANS_MIN_BLKS = 16
# compute only zero axis, cut last dim
MODE0 = 0
# compute only one axis
MODE1 = 1
# compute merge two axis
MODE2 = 2
# compute big last axis
MODE3 = 3
# judge how much ub size to fill the tensor
THRESHHOLD_VALUE = 8192
BLOCK = 32


# pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# pylint: disable=too-many-arguments,invalid-name
class PadV3Init(object):
    """
    Function: class that execute pad_v3
    """

    def __init__(self, x, paddings, constant_values, y, mode='constant', padding_contiguous=True,
                 kernel_name='pad_v3'):
        """
        init the op
        :param
        x: the input tensor
        :param
        paddings: the list of paddings
        :param
        constant_values: the value to fill the tensor
        :param
        y: the output of op
        :param
        mode: the fill mode
        :param
        padding_contiguous: is the memory is contiguous
        :param
        kernel_name: the kernel name of op
        :return
        None
        """

        self.tik_instance = tik.Tik()
        self.unknown_max_shape = (MAX_INT64,)
        self.tiling_dtype = "int64"
        self.tiling_shape = (TILING_NUMS,)

        self.x_dtype = x.get("dtype")
        self.inner_dtype = "float16"
        self.paddings_dtype = paddings.get('dtype')
        self.constant_values_dtype = constant_values.get('dtype')
        self.y_dtype = y.get('dtype')
        self.kernel_name = kernel_name
        self.input_gm = None
        self.output_gm = None
        self.tiling_gm = None
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_bytes_size = 0

        self.inner_bytes_size = tbe_platform.get_bit_len(self.inner_dtype) // EIGHT_BIT
        self.block_num = constant.BLOCK_SIZE // self.inner_bytes_size
        self.dump_mask_max_x = 8 * self.block_num
        self.max_repeat_time = 255

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - RESERVED_UB
        self.ub_number = self.ub_size_bytes // self.inner_bytes_size
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        # default copy data number in one time
        self.copy_num = 3200
        self.max_numel_vec_dup_one_loop = None

        self.sync_workspace = self.tik_instance.Tensor(
            'int64', (4 * self.core_nums,), tik.scope_gm, 'sync_workspace', is_workspace=True,
            is_atomic_add=True)
        self.pad_scalar = self.tik_instance.Scalar(dtype=self.constant_values_dtype, name='pad_scalar')
        self.constant_values_gm = self.tik_instance.Tensor(self.constant_values_dtype, (self.block_num,),
                                                           name='constant_values_gm', scope=tik.scope_gm)
        self.mode = mode
        self.padding_contiguous = padding_contiguous
        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.tiling_input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_0", init_value=0)
        self.tiling_input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_1", init_value=0)
        self.tiling_input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_2", init_value=0)
        self.tiling_input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_3", init_value=0)
        self.tiling_input_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_4", init_value=0)
        self.tiling_input_dim_5 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_5", init_value=0)
        self.tiling_pading_00 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_00", init_value=0)
        self.tiling_pading_01 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_01", init_value=0)
        self.tiling_pading_10 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_10", init_value=0)
        self.tiling_pading_11 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_11", init_value=0)
        self.tiling_pading_20 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_20", init_value=0)
        self.tiling_pading_21 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_21", init_value=0)
        self.tiling_pading_30 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_30", init_value=0)
        self.tiling_pading_31 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_31", init_value=0)
        self.tiling_pading_40 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_40", init_value=0)
        self.tiling_pading_41 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_41", init_value=0)
        self.tiling_pading_50 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_50", init_value=0)
        self.tiling_pading_51 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_pading_51", init_value=0)
        self.tiling_input_dim_cut_axis = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_cut_axis",
                                                                  init_value=0)
        self.tiling_output_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_0", init_value=0)
        self.tiling_output_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_1", init_value=0)
        self.tiling_output_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_2", init_value=0)
        self.tiling_output_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_3", init_value=0)
        self.tiling_output_dim_4 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_4", init_value=0)
        self.tiling_output_dim_5 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_5", init_value=0)

        self.tiling_input_shape = [self.tiling_input_dim_0, self.tiling_input_dim_1, self.tiling_input_dim_2,
                                   self.tiling_input_dim_3, self.tiling_input_dim_4, self.tiling_input_dim_5]
        self.tiling_output_shape = [self.tiling_output_dim_0, self.tiling_output_dim_1, self.tiling_output_dim_2,
                                    self.tiling_output_dim_3, self.tiling_output_dim_4, self.tiling_output_dim_5]
        self.tiling_pading_value = [[self.tiling_pading_00, self.tiling_pading_01],
                                    [self.tiling_pading_10, self.tiling_pading_11],
                                    [self.tiling_pading_20, self.tiling_pading_21],
                                    [self.tiling_pading_30, self.tiling_pading_31],
                                    [self.tiling_pading_40, self.tiling_pading_41],
                                    [self.tiling_pading_50, self.tiling_pading_51]]
        self.input_offset = []
        self.output_offset = []

        # core scaler init
        self.core_outer_num = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_num", init_value=0)
        self.core_outer_start = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_start", init_value=0)
        self.core_inner_num = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_num", init_value=0)
        self.core_inner_start = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_start", init_value=0)

    def get_pad_scalar(self):
        constant_values_ub = self.tik_instance.Tensor(self.constant_values_dtype, (self.block_num,),
                                                      name='constant_values_ub', scope=tik.scope_ubuf)
        self.tik_instance.data_move(constant_values_ub, self.constant_values_gm, 0, 1, 1, 0, 0)
        self.pad_scalar.set_as(constant_values_ub[0])

    def core_schedule_args(self, core_index):
        """
        core_schedule_args
        """
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 0):
            core_outer_all = self.tiling_input_shape[-1]
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
            self.core_outer_num.set_as(self.core_outer_num * self.block_num)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(self.core_outer_start + self.core_outer_num > core_outer_all):
                self.core_outer_num.set_as(core_outer_all - self.core_outer_start)
                self.tik_instance.scalar_max(self.core_outer_num, self.core_outer_num, 0)
                with self.tik_instance.if_scope(self.core_outer_num % self.block_num != 0):
                    self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
                    self.core_outer_num.set_as(self.core_outer_num * self.block_num)
                self.core_outer_start.set_as(core_outer_all - self.core_outer_num)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 1):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:5])
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + core_outer_all % self.core_nums)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 2):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:4])
            with self.tik_instance.if_scope(self.tiling_output_dim_5 * self.tiling_input_dim_4 < self.block_num):
                # the last two is less one block, only can process use one core
                self.core_outer_num.set_as(0)
                self.core_outer_start.set_as(0)
                with self.tik_instance.if_scope(core_index == 0):
                    self.core_outer_num.set_as(core_outer_all)
            with self.tik_instance.else_scope():
                self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
                self.core_outer_start.set_as(core_index * self.core_outer_num)
                with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                    with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                        self.core_outer_num.set_as(self.core_outer_num - 1)
                        self.core_outer_start.set_as(core_index * self.core_outer_num
                                                     + core_outer_all % self.core_nums)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 3):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:3])
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + core_outer_all % self.core_nums)
        for i, _ in enumerate(self.tiling_input_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "input_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_input_shape[i:]))
            self.input_offset.append(scalar)
        for i, _ in enumerate(self.tiling_output_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "output_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_output_shape[i:]))
            self.output_offset.append(scalar)

    def tiling_args(self):
        """
        when input shape is less 6, will. expand to 6
        tiling info:
            tiling_key:
            tiling_input_dim_0
            tiling_input_dim_1
            tiling_input_dim_2
            tiling_input_dim_3
            tiling_input_dim_4
            tiling_input_dim_5
            tiling_pading_00
            tiling_pading_01
            tiling_pading_10
            tiling_pading_11
            tiling_pading_20
            tiling_pading_21
            tiling_pading_30
            tiling_pading_31
            tiling_pading_40
            tiling_pading_41
            tiling_pading_50
            tiling_pading_51
            tiling_input_dim_cut_axis: which dim will be cut
        """
        with self.tik_instance.new_stmt_scope():
            tiling_ub = self.tik_instance.Tensor("int64", (TILING_NUMS,),
                                                 name="tiling_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, TILING_NUMS // 4, 0, 0)
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_input_dim_0.set_as(tiling_ub[1])
            self.tiling_input_dim_1.set_as(tiling_ub[2])
            self.tiling_input_dim_2.set_as(tiling_ub[3])
            self.tiling_input_dim_3.set_as(tiling_ub[4])
            self.tiling_input_dim_4.set_as(tiling_ub[5])
            self.tiling_input_dim_5.set_as(tiling_ub[6])
            self.tiling_pading_00.set_as(tiling_ub[7])
            self.tiling_pading_01.set_as(tiling_ub[8])
            self.tiling_pading_10.set_as(tiling_ub[9])
            self.tiling_pading_11.set_as(tiling_ub[10])
            self.tiling_pading_20.set_as(tiling_ub[11])
            self.tiling_pading_21.set_as(tiling_ub[12])
            self.tiling_pading_30.set_as(tiling_ub[13])
            self.tiling_pading_31.set_as(tiling_ub[14])
            self.tiling_pading_40.set_as(tiling_ub[15])
            self.tiling_pading_41.set_as(tiling_ub[16])
            self.tiling_pading_50.set_as(tiling_ub[17])
            self.tiling_pading_51.set_as(tiling_ub[18])
            self.tiling_input_dim_cut_axis.set_as(tiling_ub[19])

            # calcu output_dim
            for i, _ in enumerate(self.tiling_input_shape):
                input_dims = self.tiling_input_shape[i]
                pad_left = self.tiling_pading_value[i][0]
                pad_right = self.tiling_pading_value[i][1]
                output_dims = self.tiling_output_shape[i]
                output_dims.set_as(input_dims + pad_left + pad_right)

    def init_src_dst_gm(self, input_dict_list, output_dict_list, pad_input_idx=0, pad_outnput_idx=0):
        """
        init gm tensor set tiling, input, paddings output tensor(gm)
        :param
        input_dict_list: the dict of input_dict
        :param
        output_dict_list: output_dict_list
        :param
        pad_input_idx: pad_input_idx
        :param
        pad_outnput_idx: pad_outnput_idx
        :return:
        None
        """
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                  name="tiling_gm", scope=tik.scope_gm)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype, self.tiling_shape,
                                                  name="tiling_gm", scope=tik.scope_gm)
        for i, input_dict in enumerate(input_dict_list):
            input_dtype = input_dict.get("dtype") if pad_input_idx != i else self.inner_dtype
            input_gm = self.tik_instance.Tensor(input_dtype, self.unknown_max_shape,
                                                name="input_gm_" + str(i), scope=tik.scope_gm)
            self.input_gm_list.append(input_gm)
        self.input_gm_list.append(self.constant_values_gm)

        for i, output_dict in enumerate(output_dict_list):
            output_dtype = output_dict.get("dtype")
            if pad_outnput_idx == i:
                # the pad output must will atomic_add to clear the output_gm to all zero
                output_gm = self.tik_instance.Tensor(self.inner_dtype, self.unknown_max_shape,
                                                     name="output_gm_" + str(i), scope=tik.scope_gm)
                self.input_bytes_size = tbe_platform.get_bit_len(output_dtype) // EIGHT_BIT
            else:
                output_gm = self.tik_instance.Tensor(output_dtype, self.unknown_max_shape,
                                                     name="output_gm_" + str(i), scope=tik.scope_gm)
            self.output_gm_list.append(output_gm)

        self.input_gm = self.input_gm_list[pad_input_idx]
        self.output_gm = self.output_gm_list[pad_outnput_idx]

    def fill_gm_output_tensor(self, core_index):
        '''
        fill gm output tensor
        :param
        core_index: the index of each core
        :return:
        None
        '''
        self.max_numel_vec_dup_one_loop = self.max_repeat_time * self.dump_mask_max_x
        total_output_tensor = self.tik_instance.Scalar(dtype='int32', name='total_output_tensor', init_value=1)
        total_output_tensor_each_core = self.tik_instance.Scalar(dtype='int32', name='total_output_tensor_each_core')
        total_output_tensor_last_core = self.tik_instance.Scalar(dtype='int32', name='total_output_tensor_last_core')
        offset_gm = self.tik_instance.Scalar(dtype='int32', name='offset_gm')
        for ele in self.tiling_output_shape:
            total_output_tensor.set_as(total_output_tensor * ele)
        block = self.tik_instance.Scalar(dtype='int32')
        if self.x_dtype == 'float32':
            block.set_as(BLOCK // 4)
        elif self.x_dtype == 'float16':
            block.set_as(BLOCK // 2)
        core_nums = self.tik_instance.Scalar(dtype='int32')
        core_nums.set_as(self.core_nums)
        total_output_tensor_each_core.set_as((total_output_tensor - 1) // core_nums + 1)
        with self.tik_instance.for_range(0, BLOCK) as i:
            with self.tik_instance.if_scope(total_output_tensor_each_core < block):
                core_nums.set_as(core_nums - 1)
                total_output_tensor_each_core.set_as((total_output_tensor - 1) // core_nums + 1)
        total_output_tensor_each_core.set_as(((total_output_tensor_each_core - 1) // block + 1) * block)
        core_nums.set_as((total_output_tensor - 1) // total_output_tensor_each_core + 1)
        total_output_tensor_last_core.set_as(((total_output_tensor % total_output_tensor_each_core - 1) // block + 1)
                                             * block)
        with self.tik_instance.if_scope(core_index < core_nums):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.if_scope(tik.all(core_index == core_nums - 1,
                                                        (total_output_tensor % total_output_tensor_each_core) > 0)):
                    with self.tik_instance.if_scope(total_output_tensor_last_core >= THRESHHOLD_VALUE):
                        fill_tensor_ub = self.tik_instance.Tensor(self.x_dtype, (THRESHHOLD_VALUE,),
                                                                  name='fill_tensor_ub', scope=tik.scope_ubuf)
                        repeat_time = THRESHHOLD_VALUE // self.dump_mask_max_x
                        self.tik_instance.vec_dup(self.dump_mask_max_x,
                                                  fill_tensor_ub,
                                                  self.pad_scalar, repeat_time, 8)
                        times = self.tik_instance.Scalar(dtype='int32')
                        tail_burst_len = self.tik_instance.Scalar(dtype='int32')
                        times.set_as(total_output_tensor_last_core // THRESHHOLD_VALUE)
                        tail_burst_len.set_as(total_output_tensor_last_core % THRESHHOLD_VALUE // block)
                        if self.x_dtype == 'float32':
                            fill_tensor_ub = fill_tensor_ub.reinterpret_cast_to('float16')
                        with self.tik_instance.for_range(0, times) as i:
                            if self.x_dtype == 'float32':
                                offset_gm.set_as((core_index * total_output_tensor_each_core +
                                                  i * THRESHHOLD_VALUE) * 2)
                            elif self.x_dtype == 'float16':
                                offset_gm.set_as(core_index * total_output_tensor_each_core +
                                                 i * THRESHHOLD_VALUE)
                            self.tik_instance.data_move(self.output_gm[offset_gm], fill_tensor_ub, 0, 1,
                                                        THRESHHOLD_VALUE // block, 0, 0)
                        with self.tik_instance.if_scope(tail_burst_len > 0):
                            if self.x_dtype == 'float32':
                                offset_gm.set_as((core_index * total_output_tensor_each_core +
                                                  times * THRESHHOLD_VALUE) * 2)
                            elif self.x_dtype == 'float16':
                                offset_gm.set_as(core_index * total_output_tensor_each_core +
                                                 times * THRESHHOLD_VALUE)
                            self.tik_instance.data_move(self.output_gm[offset_gm], fill_tensor_ub, 0, 1,
                                                        tail_burst_len, 0, 0)

                    with self.tik_instance.else_scope():
                        fill_tensor_ub = self.tik_instance.Tensor(self.x_dtype, (THRESHHOLD_VALUE,),
                                                                  name='fill_tensor_ub', scope=tik.scope_ubuf)
                        repeat_time = self.tik_instance.Scalar(dtype='int32')
                        tail_mask = self.tik_instance.Scalar(dtype='int32')
                        repeat_time.set_as(total_output_tensor_last_core // self.dump_mask_max_x)
                        with self.tik_instance.if_scope(repeat_time > 0):
                            self.tik_instance.vec_dup(self.dump_mask_max_x,
                                                      fill_tensor_ub,
                                                      self.pad_scalar, repeat_time, 8)
                        tail_mask.set_as(total_output_tensor_last_core % self.dump_mask_max_x)
                        with self.tik_instance.if_scope(tail_mask > 0):
                            self.tik_instance.vec_dup(tail_mask,
                                                      fill_tensor_ub[repeat_time * self.dump_mask_max_x],
                                                      self.pad_scalar, 1, 8)

                        if self.x_dtype == 'float32':
                            fill_tensor_ub = fill_tensor_ub.reinterpret_cast_to('float16')
                            offset_gm.set_as(core_index * total_output_tensor_each_core * 2)
                        elif self.x_dtype == 'float16':
                            offset_gm.set_as(core_index * total_output_tensor_each_core)
                        self.tik_instance.data_move(self.output_gm[offset_gm],
                                                    fill_tensor_ub, 0, 1,
                                                    total_output_tensor_last_core // block, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.if_scope(total_output_tensor_each_core >= THRESHHOLD_VALUE):
                        fill_tensor_ub = self.tik_instance.Tensor(self.x_dtype, (THRESHHOLD_VALUE,),
                                                                  name='fill_tensor_ub', scope=tik.scope_ubuf)
                        repeat_time = THRESHHOLD_VALUE // self.dump_mask_max_x
                        self.tik_instance.vec_dup(self.dump_mask_max_x,
                                                  fill_tensor_ub,
                                                  self.pad_scalar, repeat_time, 8)
                        times = self.tik_instance.Scalar(dtype='int32')
                        tail_burst_len = self.tik_instance.Scalar(dtype='int32')
                        times.set_as(total_output_tensor_each_core // THRESHHOLD_VALUE)
                        tail_burst_len.set_as(total_output_tensor_each_core % THRESHHOLD_VALUE // block)

                        if self.x_dtype == 'float32':
                            fill_tensor_ub = fill_tensor_ub.reinterpret_cast_to('float16')
                        with self.tik_instance.for_range(0, times) as i:
                            if self.x_dtype == 'float32':
                                offset_gm.set_as((core_index * total_output_tensor_each_core +
                                                  i * THRESHHOLD_VALUE) * 2)
                            elif self.x_dtype == 'float16':
                                offset_gm.set_as(core_index * total_output_tensor_each_core +
                                                 i * THRESHHOLD_VALUE)
                            self.tik_instance.data_move(self.output_gm[offset_gm], fill_tensor_ub, 0, 1,
                                                        THRESHHOLD_VALUE // block, 0, 0)
                        with self.tik_instance.if_scope(tail_burst_len > 0):
                            if self.x_dtype == 'float32':
                                offset_gm.set_as((core_index * total_output_tensor_each_core +
                                                  times * THRESHHOLD_VALUE) * 2)
                            elif self.x_dtype == 'float16':
                                offset_gm.set_as(core_index * total_output_tensor_each_core +
                                                 times * THRESHHOLD_VALUE)
                            self.tik_instance.data_move(self.output_gm[offset_gm], fill_tensor_ub, 0, 1,
                                                        tail_burst_len, 0, 0)

                    with self.tik_instance.else_scope():
                        fill_tensor_ub = self.tik_instance.Tensor(self.x_dtype, (THRESHHOLD_VALUE,),
                                                                  name='fill_tensor_ub', scope=tik.scope_ubuf)
                        repeat_time = self.tik_instance.Scalar(dtype='int32')
                        tail_mask = self.tik_instance.Scalar(dtype='int32')
                        repeat_time.set_as(total_output_tensor_each_core // self.dump_mask_max_x)
                        with self.tik_instance.if_scope(repeat_time > 0):
                            self.tik_instance.vec_dup(self.dump_mask_max_x,
                                                      fill_tensor_ub,
                                                      self.pad_scalar, repeat_time, 8)
                        tail_mask.set_as(total_output_tensor_each_core % self.dump_mask_max_x)
                        with self.tik_instance.if_scope(tail_mask > 0):
                            self.tik_instance.vec_dup(tail_mask,
                                                      fill_tensor_ub[repeat_time * self.dump_mask_max_x],
                                                      self.pad_scalar, 1, 8)

                        if self.x_dtype == 'float32':
                            fill_tensor_ub = fill_tensor_ub.reinterpret_cast_to('float16')
                            offset_gm.set_as(core_index * total_output_tensor_each_core * 2)
                        elif self.x_dtype == 'float16':
                            offset_gm.set_as(core_index * total_output_tensor_each_core)
                        self.tik_instance.data_move(self.output_gm[offset_gm],
                                                    fill_tensor_ub, 0, 1,
                                                    total_output_tensor_each_core // block, 0, 0)

    def pad_v3_d_compute_tiling(self):
        """
        pad_v3 operation
        """

        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_index:
            self.get_pad_scalar()
            self.tiling_args()
            self.core_schedule_args(core_index)
            self.fill_gm_output_tensor(core_index)
            self.tik_instance.block_barrier(self.sync_workspace)
            self.do_pad()

    def get_output_outer_idx(self, in_idx, outer_num=5):
        """
        get_output_outer_idx use in_idx
        """
        input_dim_0 = in_idx // self.input_offset[1]
        input_dim_1 = (in_idx % self.input_offset[1]) // self.input_offset[2]
        input_dim_2 = (in_idx % self.input_offset[2]) // self.input_offset[3]
        input_dim_3 = (in_idx % self.input_offset[3]) // self.input_offset[4]
        input_dim_4 = (in_idx % self.input_offset[4]) // self.input_offset[5]
        input_dim_5 = in_idx % self.input_offset[5]

        input_list = [input_dim_0, input_dim_1, input_dim_2, input_dim_3, input_dim_4, input_dim_5]
        output_list = []
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = input_list[i]
            pad_left = self.tiling_pading_value[i][0]
            output_dims = input_dims + pad_left
            output_list.append(output_dims)

        output_idx = 0
        for i in range(outer_num):
            output_idx = output_idx + output_list[i] * self.output_offset[i + 1]
        return output_idx

    def data_move(self, gm_src_info, gm_dst_info, copy_len, used_ub):
        """
        func for data_move
        :param
        gm_src_info:gm_src_info
        :param
        gm_dst_info:gm_dst_info
        :param
        copy_len:copy_len
        :param
        used_ub:used_ub
        :return:
        None
        """
        input_gm, input_offset = gm_src_info
        output_gm, output_offset = gm_dst_info
        bursn_len = (copy_len + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(used_ub,
                                    input_gm[input_offset],
                                    0, 1, bursn_len, 0, 0)
        self.tik_instance.data_move(output_gm[output_offset],
                                    used_ub,
                                    0, 1, bursn_len, 0, 0)

    def data_move_with_mask_less_block(self, gm_src_info, gm_dst_info, copy_len, used_ub, ub_one_block):
        """
        func for data_move_with_mask
        """
        input_gm, input_offset = gm_src_info
        output_gm, output_offset = gm_dst_info
        bursn_len = (copy_len + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(ub_one_block,
                                    input_gm[input_offset],
                                    0, 1, bursn_len, 0, 0)
        vnchw_src_list = [ub_one_block] * TRANS_MIN_BLKS
        vnchw_dst_list = [used_ub[i * TRANS_MIN_BLKS] for i in range(TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False,
                                    vnchw_dst_list,
                                    vnchw_src_list,
                                    1, 0, 0)
        burst_num = 1
        burst_len = self.block_num - copy_len + 1
        self.tik_instance.data_move(used_ub[copy_len * self.block_num:],
                                    ub_one_block[self.block_num:],
                                    0, burst_num, burst_len, 0, 0)
        vnchw_src_list = [used_ub[i * TRANS_MIN_BLKS] for i in range(TRANS_MIN_BLKS)]
        vnchw_dst_list = \
            [used_ub[i * TRANS_MIN_BLKS + TRANS_MIN_BLKS * TRANS_MIN_BLKS] for i in range(TRANS_MIN_BLKS)]
        self.tik_instance.vnchwconv(False, False,
                                    vnchw_dst_list,
                                    vnchw_src_list,
                                    1, 0, 0)
        self.tik_instance.data_move(output_gm[output_offset],
                                    used_ub[TRANS_MIN_BLKS * TRANS_MIN_BLKS],
                                    0, 1, bursn_len, 0, 0)

    def do_tiling_key_mode_0(self):
        """
        do_tiling_key_mode_0: in this tiling self.core_outer_num mean the last dim num for each core
        """
        outer_all_dim_num = self.tik_instance.Scalar(dtype="int64", name="outer_all_dim_num")
        outer_all_dim_num.set_as(functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:5]))
        copy_num = self.copy_num
        scalar_copy_num = self.tik_instance.Scalar(dtype="int32", name="scalar_copy_num")
        scalar_copy_num.set_as(copy_num)
        with self.tik_instance.if_scope(scalar_copy_num > self.core_outer_num):
            scalar_copy_num.set_as(self.core_outer_num)

        copy_loop_ceil = self.tik_instance.Scalar(dtype="int64", name="copy_loop_ceil")
        copy_loop_floor = self.tik_instance.Scalar(dtype="int64", name="copy_loop_floor")
        copy_loop_ceil.set_as((self.core_outer_num + scalar_copy_num - 1) // scalar_copy_num)
        copy_loop_floor.set_as(self.core_outer_num // scalar_copy_num)
        copy_tail = self.core_outer_num % scalar_copy_num
        process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                  name="process_num_ub", scope=tik.scope_ubuf)
        process_num_ub[0].set_as(scalar_copy_num)
        process_num_ub[1].set_as(copy_tail)

        def _run_one_dim(input_outer_idx, input_ub_list):
            '''
            run one dim
            :param
            input_outer_idx: the input index of outer data
            :param
            input_ub_list: the list of input ub
            :return:
            None
            '''
            data_ub_ping, data_ub_pang, _ = input_ub_list
            output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
            input_gm_offset = input_outer_idx * self.tiling_input_shape[-1]
            output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset) + self.tiling_pading_value[-1][0])

            with self.tik_instance.for_range(0, copy_loop_ceil // 2) as copy_idx:
                ping_idx = copy_idx * 2
                idx_scalar = self.tik_instance.Scalar(dtype="int32", name="idx_scalar")
                idx_scalar.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                               [self.output_gm, output_outer_offset
                                + ping_idx * scalar_copy_num + self.core_outer_start],
                               idx_scalar, data_ub_ping)
                pang_idx = copy_idx * 2 + 1
                idx_scalar1 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar1")
                idx_scalar1.set_as(process_num_ub[pang_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + pang_idx * scalar_copy_num + self.core_outer_start],
                               [self.output_gm, output_outer_offset
                                + pang_idx * scalar_copy_num + self.core_outer_start],
                               idx_scalar1, data_ub_pang)
            with self.tik_instance.if_scope(copy_loop_ceil % 2 != 0):
                ping_idx = copy_loop_ceil - 1
                idx_scalar2 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar2")
                idx_scalar2.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + ping_idx * scalar_copy_num + self.core_outer_start],
                               [self.output_gm, output_outer_offset
                                + ping_idx * scalar_copy_num + self.core_outer_start],
                               idx_scalar2, data_ub_ping)

        ping_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_ping", scope=tik.scope_ubuf)
        ping_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_pang", scope=tik.scope_ubuf)
        ping_data_ub_tail = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_tail", scope=tik.scope_ubuf)

        pang_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_ping", scope=tik.scope_ubuf)
        pang_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_pang", scope=tik.scope_ubuf)
        pang_data_ub_tail = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_tail", scope=tik.scope_ubuf)

        ping_ub_list = [ping_data_ub_ping, ping_data_ub_pang, ping_data_ub_tail]
        pang_ub_list = [pang_data_ub_ping, pang_data_ub_pang, pang_data_ub_tail]

        with self.tik_instance.for_range(0, outer_all_dim_num // 2) as _outer_num_idx:
            _outer_idx = _outer_num_idx * 2
            _run_one_dim(_outer_idx, ping_ub_list)
            _outer_idx = _outer_num_idx * 2 + 1
            _run_one_dim(_outer_idx, pang_ub_list)
        with self.tik_instance.if_scope(outer_all_dim_num % 2 != 0):
            _outer_idx = outer_all_dim_num - 1
            _run_one_dim(_outer_idx, ping_ub_list)

    def do_tiling_key_mode_1(self):
        """
        do_tiling_key_mode_1  when tiling key = 1
        """
        copy_num = self.copy_num
        scalar_copy_num = self.tik_instance.Scalar(dtype="int32", name="scalar_copy_num")
        scalar_copy_num.set_as(copy_num)
        with self.tik_instance.if_scope(scalar_copy_num > self.tiling_input_shape[-1]):
            scalar_copy_num.set_as(self.tiling_input_shape[-1])

        block_copy_num = self.tik_instance.Scalar(dtype="int32", name="block_copy_num", init_value=self.block_num)
        with self.tik_instance.if_scope(scalar_copy_num < self.block_num):
            block_copy_num.set_as(self.tiling_input_shape[-1])

        copy_tail = self.tiling_input_shape[-1] % self.block_num
        tail_copy_offset = self.tik_instance.Scalar(dtype="int64", name="tail_copy_offset")
        tail_copy_offset.set_as(copy_tail)
        with self.tik_instance.if_scope(copy_tail == 0):
            tail_copy_offset.set_as(self.block_num)

        copy_new_num = self.tiling_input_shape[-1] - tail_copy_offset
        with self.tik_instance.if_scope(scalar_copy_num > copy_new_num):
            scalar_copy_num.set_as(copy_new_num)
        copy_loop_ceil = self.tik_instance.Scalar(dtype="int64", name="copy_loop_ceil")
        copy_loop_floor = self.tik_instance.Scalar(dtype="int64", name="copy_loop_floor")
        copy_loop_ceil.set_as((copy_new_num + scalar_copy_num - 1) // scalar_copy_num)
        copy_loop_floor.set_as(copy_new_num // scalar_copy_num)
        copy_tail = copy_new_num % scalar_copy_num
        process_num_ub = self.tik_instance.Tensor("int32", (self.block_num,),
                                                  name="process_num_ub", scope=tik.scope_ubuf)
        process_num_ub[0].set_as(scalar_copy_num)
        process_num_ub[1].set_as(copy_tail)
        process_num_ub[2].set_as(copy_new_num - scalar_copy_num * (copy_loop_ceil - 1))

        def _run_one_dim(input_outer_idx, input_ub_list):
            '''
            run one dim
            :param
            input_outer_idx: the input index of outer data
            :param
            input_ub_list: the list of input ub
            :return:
            None
            '''
            ub_one_block, data_ub_ping, data_ub_pang, _ = input_ub_list
            output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
            input_gm_offset = input_outer_idx * self.tiling_input_shape[-1]
            output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset) + self.tiling_pading_value[-1][0])
            # copy one block first
            self.data_move_with_mask_less_block([self.input_gm, input_gm_offset],
                                                [self.output_gm, output_outer_offset],
                                                block_copy_num, data_ub_pang, ub_one_block)

            with self.tik_instance.for_range(0, copy_loop_ceil // 2) as copy_idx:
                ping_idx = copy_idx * 2
                idx_scalar = self.tik_instance.Scalar(dtype="int32", name="idx_scalar")
                idx_scalar.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               idx_scalar, data_ub_ping)
                pang_idx = copy_idx * 2 + 1
                idx_scalar1 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar1")
                idx_scalar1.set_as(process_num_ub[pang_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + pang_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + pang_idx * scalar_copy_num],
                               idx_scalar1, data_ub_pang)
            with self.tik_instance.if_scope(copy_loop_ceil % 2 != 0):
                ping_idx = copy_loop_ceil - 1
                idx_scalar2 = self.tik_instance.Scalar(dtype="int32", name="idx_scalar2")
                idx_scalar2.set_as(process_num_ub[ping_idx // copy_loop_floor])
                self.data_move([self.input_gm, input_gm_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               [self.output_gm, output_outer_offset + tail_copy_offset + ping_idx * scalar_copy_num],
                               idx_scalar2, data_ub_ping)

        ping_data_ub_one_block = self.tik_instance.Tensor(self.inner_dtype,
                                                          (self.block_num * self.block_num + self.block_num,),
                                                          name="ping_data_ub_one_block", scope=tik.scope_ubuf)
        ping_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_ping", scope=tik.scope_ubuf)
        ping_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_pang", scope=tik.scope_ubuf)
        ping_data_ub_tail = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="ping_data_ub_tail", scope=tik.scope_ubuf)

        pang_data_ub_one_block = self.tik_instance.Tensor(self.inner_dtype,
                                                          (self.block_num * self.block_num + self.block_num,),
                                                          name="pang_data_ub_one_block", scope=tik.scope_ubuf)
        pang_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_ping", scope=tik.scope_ubuf)
        pang_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_pang", scope=tik.scope_ubuf)
        pang_data_ub_tail = self.tik_instance.Tensor(self.inner_dtype, (copy_num,),
                                                     name="pang_data_ub_tail", scope=tik.scope_ubuf)

        if self.x_dtype == "float32":
            ping_data_ub_one_block = ping_data_ub_one_block.reinterpret_cast_to('float32')
            pang_data_ub_one_block = pang_data_ub_one_block.reinterpret_cast_to('float32')
            self.tik_instance.vector_dup(self.block_num * 4, ping_data_ub_one_block[16:], self.pad_scalar, 4, 1, 8)
            self.tik_instance.vector_dup(self.block_num * 4, pang_data_ub_one_block[16:], self.pad_scalar, 4, 1, 8)
            ping_data_ub_one_block = ping_data_ub_one_block.reinterpret_cast_to('float16')
            pang_data_ub_one_block = pang_data_ub_one_block.reinterpret_cast_to('float16')
        elif self.x_dtype == "float16":
            self.tik_instance.vector_dup(self.block_num * 8, ping_data_ub_one_block[16:], self.pad_scalar, 2, 1, 8)
            self.tik_instance.vector_dup(self.block_num * 8, pang_data_ub_one_block[16:], self.pad_scalar, 2, 1, 8)

        ping_ub_list = [ping_data_ub_one_block, ping_data_ub_ping, ping_data_ub_pang, ping_data_ub_tail]
        pang_ub_list = [pang_data_ub_one_block, pang_data_ub_ping, pang_data_ub_pang, pang_data_ub_tail]

        with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_num_idx:
            _outer_idx = _outer_num_idx * 2 + self.core_outer_start
            _run_one_dim(_outer_idx, ping_ub_list)
            _outer_idx = _outer_num_idx * 2 + 1 + self.core_outer_start
            _run_one_dim(_outer_idx, pang_ub_list)
        with self.tik_instance.if_scope(self.core_outer_num % 2 != 0):
            _outer_idx = self.core_outer_num - 1 + self.core_outer_start
            _run_one_dim(_outer_idx, ping_ub_list)

    def do_tiling_key_mode_2(self):
        """
        do_tiling_key_mode_2 when tiling key = 2
        """
        max_line_in_ub = 16
        max_output_size = 480 * 2
        second_dim_input_num = self.tiling_input_shape[-2]
        third_dim_input_num = self.tiling_input_shape[-1]
        third_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")

        second_dim_cut_num.set_as(max_output_size // third_dim_output_num)
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        first_dim_cut_num.set_as(max_line_in_ub * second_dim_cut_num)

        # cut inner first dim and second dim info
        second_dim_total_loop_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_num")
        second_dim_total_loop_tail = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_tail")
        second_dim_total_loop_num.set_as(second_dim_input_num // second_dim_cut_num)
        second_dim_total_loop_tail.set_as(second_dim_input_num % second_dim_cut_num)

        second_dim_outer_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_cut_num")
        second_dim_outer_cut_num.set_as(max_line_in_ub)
        with self.tik_instance.if_scope(second_dim_total_loop_num < max_line_in_ub):
            second_dim_outer_cut_num.set_as(second_dim_total_loop_num)

        second_dim_outer_loop_num_ceil = \
            (second_dim_total_loop_num + second_dim_outer_cut_num - 1) // second_dim_outer_cut_num
        second_dim_outer_loop_num_floor = second_dim_total_loop_num // second_dim_outer_cut_num

        second_dim_outer_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                               name="second_dim_outer_sigment_ub",
                                                               scope=tik.scope_ubuf)
        second_dim_outer_sigment_ub[0].set_as(second_dim_outer_cut_num)
        second_dim_outer_sigment_ub[1].set_as(second_dim_total_loop_num % second_dim_outer_cut_num)

        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_input_num % second_dim_cut_num)

        loop_align_tail = self.tik_instance.Scalar(dtype="int64", name="loop_align_tail")
        tail_align_tail = self.tik_instance.Scalar(dtype="int64", name="tail_align_tail")
        one_core_flag = self.tik_instance.Scalar(dtype="int64", name="one_core_flag", init_value=0)
        loop_align_tail.set_as((second_dim_cut_num * third_dim_output_num) % self.block_num)
        tail_align_tail.set_as((second_dim_total_loop_tail * third_dim_output_num) % self.block_num)
        with self.tik_instance.if_scope(self.tiling_output_shape[-1] * self.tiling_input_shape[-2] <= self.block_num):
            loop_align_tail.set_as(0)
            tail_align_tail.set_as(0)
            one_core_flag.set_as(self.block_num - 1)

        vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
        vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
        vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
        vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
        vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
        vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        vnchw_repeat0.set_as(((second_dim_cut_num * third_dim_input_num) + self.block_num - 1) // self.block_num)
        vnchw_repeat1.set_as(((second_dim_cut_num * third_dim_output_num) + self.block_num - 1) // self.block_num)
        with self.tik_instance.if_scope(vnchw_repeat0 == 1):
            vnchw_src_stride0.set_as(0)
            vnchw_dst_stride0.set_as(0)
        with self.tik_instance.if_scope(vnchw_repeat1 == 1):
            vnchw_src_stride1.set_as(0)
            vnchw_dst_stride1.set_as(0)

        def run_outer_by_outer(second_dim_start, do_inner_num, do_outer_num, align_tail, disable_sync_mte3=False):
            '''
            run_outer_by_outer
            :param
            second_dim_start:the second dim start of input
            :param
            do_inner_num: the number which do inner
            :param
            do_outer_num: the number which do outer
            :param
            align_tail: the tail of align
            :param
            disable_sync_mte3: disable_sync_mte3
            :return:
            None
            '''

            def _run_one_outer(_outer_num_idx, ub_list):
                '''
                _run_one_outer
                :param
                _outer_num_idx:
                :param
                ub_list:
                :return:
                none
                '''
                origin_data_ub, vnchw_data_ub, vnchw_output_data_ub, _, _ = ub_list
                _, _, _, origin_output_data_ub, origin_output_tail_data_ub = ub_list
                input_outer_idx = _outer_num_idx + self.core_outer_start
                input_gm_offset = input_outer_idx * self.input_offset[4]
                output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 4))

                # step1. copy 16 dims in origin_data_ub
                with self.tik_instance.new_stmt_scope(disable_sync=True):
                    with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                        burst_len = ((do_inner_num * third_dim_input_num) + self.block_num - 1) // self.block_num
                        src_offset = (second_dim_start + _copy_idx * do_inner_num) * third_dim_input_num
                        self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                    self.input_gm[input_gm_offset + src_offset],
                                                    0, 1, burst_len, 0, 0)
                # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                origin_data_ub_list = [origin_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                vnchw_data_ub_list = [vnchw_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False,
                                            vnchw_data_ub_list,
                                            origin_data_ub_list,
                                            vnchw_repeat0,
                                            vnchw_dst_stride0, vnchw_src_stride0)

                pad_left = self.tiling_pading_value[-1][0]
                pad_right = self.tiling_pading_value[-1][1]
                # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                # step3.0 copy input data to vnchw_output_data_ub with datamove
                burst_num = do_inner_num
                burst_len = third_dim_input_num
                src_offset = 0
                dst_offset = pad_left * self.block_num
                src_stride = 0
                dst_stride = pad_left + pad_right
                self.tik_instance.data_move(vnchw_output_data_ub[dst_offset],
                                            vnchw_data_ub[src_offset],
                                            0, burst_num, burst_len, src_stride, dst_stride)

                # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                origin_output_data_ub_list = \
                    [origin_output_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                vnchw_output_data_ub_list = \
                    [vnchw_output_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False,
                                            origin_output_data_ub_list,
                                            vnchw_output_data_ub_list,
                                            vnchw_repeat1,
                                            vnchw_dst_stride1, vnchw_src_stride1)

                # step5. copy 16 dims to output
                # step5.1 copy do_outer_num - 1 lines to output use ceil_div block
                with self.tik_instance.if_scope(do_inner_num * third_dim_output_num % self.block_num != 0):
                    with self.tik_instance.new_stmt_scope(disable_sync=disable_sync_mte3):
                        with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                            burst_len = (do_inner_num * third_dim_output_num + self.block_num - 1) // self.block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[4][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * self.output_offset[5]
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_data_ub[_copy_idx * max_output_size],
                                                        0, 1, burst_len, 0, 0)
                        # step5.1 copy the last do_outer_num lines to output use floor_div block
                        burst_len = (do_inner_num * third_dim_output_num + one_core_flag) // self.block_num
                        dst_offset = \
                            output_outer_offset + \
                            (self.tiling_pading_value[4][0] + second_dim_start + (do_outer_num - 1) * do_inner_num) \
                            * self.output_offset[5]
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    origin_output_data_ub[(do_outer_num - 1) * max_output_size],
                                                    0, 1, burst_len, 0, 0)

                    # step6. process tail for the last line
                    with self.tik_instance.if_scope(align_tail != 0):
                        origin_output_data_ub_list = \
                            [origin_output_tail_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                        vnchw_output_data_ub_list = \
                            [vnchw_output_data_ub[i * 16 + (do_inner_num * third_dim_output_num - 16) * 16]
                             for i in range(0, TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(False, False,
                                                    origin_output_data_ub_list,
                                                    vnchw_output_data_ub_list,
                                                    1, 0, 0)
                        burst_len = 1
                        dst_offset = \
                            output_outer_offset \
                            + (self.tiling_pading_value[4][0] + second_dim_start + do_outer_num * do_inner_num) \
                              * self.output_offset[5] \
                            - self.block_num
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    origin_output_tail_data_ub[(do_outer_num - 1) * 16],
                                                    0, 1, burst_len, 0, 0)
                with self.tik_instance.else_scope():
                    with self.tik_instance.new_stmt_scope(disable_sync=True):
                        with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                            burst_len = (do_inner_num * third_dim_output_num + self.block_num - 1) // self.block_num
                            dst_offset = \
                                output_outer_offset + \
                                (self.tiling_pading_value[4][0] + second_dim_start + _copy_idx * do_inner_num) \
                                * self.output_offset[5]
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_data_ub[_copy_idx * max_output_size],
                                                        0, 1, burst_len, 0, 0)

            origin_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_ping", scope=tik.scope_ubuf)
            vnchw_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

            vnchw_output_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (max_line_in_ub * max_output_size,),
                                                                 name="vnchw_output_data_ub_ping",
                                                                 scope=tik.scope_ubuf)
            origin_output_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype,
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)

            if self.x_dtype == "float32":
                vnchw_output_data_ub_ping = vnchw_output_data_ub_ping.reinterpret_cast_to('float32')
                self.tik_instance.vector_dup(self.block_num * 4, vnchw_output_data_ub_ping, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.block_num // 4, 1, 8)
                vnchw_output_data_ub_ping = vnchw_output_data_ub_ping.reinterpret_cast_to('float16')

            elif self.x_dtype == "float16":
                self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_ping, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)

            origin_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_pang", scope=tik.scope_ubuf)
            vnchw_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

            vnchw_output_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (max_line_in_ub * max_output_size,),
                                                                 name="vnchw_output_data_ub_ping",
                                                                 scope=tik.scope_ubuf)
            origin_output_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype,
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)

            if self.x_dtype == "float32":
                vnchw_output_data_ub_pang = vnchw_output_data_ub_pang.reinterpret_cast_to('float32')
                self.tik_instance.vector_dup(self.block_num * 4, vnchw_output_data_ub_pang, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.block_num // 4, 1, 8)
                vnchw_output_data_ub_pang = vnchw_output_data_ub_pang.reinterpret_cast_to('float16')
            elif self.x_dtype == "float16":
                self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_pang, self.pad_scalar,
                                             max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)

            ping_ub_list = [origin_data_ub_ping, vnchw_data_ub_ping, vnchw_output_data_ub_ping,
                            origin_output_data_ub_ping, origin_output_tail_data_ub_ping]
            pang_ub_list = [origin_data_ub_pang, vnchw_data_ub_pang, vnchw_output_data_ub_pang,
                            origin_output_data_ub_pang, origin_output_tail_data_ub_pang]
            with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                _run_one_outer(_outer_idx * 2, ping_ub_list)
                _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
            with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                _run_one_outer(self.core_outer_num - 1, ping_ub_list)

        with self.tik_instance.for_range(0, second_dim_outer_loop_num_ceil) as second_dim_outer_idx:
            second_dim_outer_start = second_dim_outer_idx * second_dim_outer_cut_num * second_dim_cut_num
            second_dim_outer_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_process_num")
            second_dim_outer_process_num.set_as(
                second_dim_outer_sigment_ub[second_dim_outer_idx // second_dim_outer_loop_num_floor])
            run_outer_by_outer(second_dim_outer_start, second_dim_cut_num,
                               second_dim_outer_process_num, loop_align_tail)

        with self.tik_instance.if_scope(second_dim_total_loop_tail != 0):
            second_dim_outer_tail_start = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_tail_start")
            second_dim_outer_tail_start.set_as((second_dim_input_num // second_dim_cut_num) * second_dim_cut_num)
            with self.tik_instance.if_scope(second_dim_total_loop_tail * third_dim_output_num < self.block_num):
                new_tail_num = (self.block_num + third_dim_output_num - 1) // third_dim_output_num
                second_dim_outer_tail_start.set_as(
                    second_dim_outer_tail_start - new_tail_num + second_dim_total_loop_tail)
                second_dim_total_loop_tail.set_as(new_tail_num)

            run_outer_by_outer(second_dim_outer_tail_start, second_dim_total_loop_tail, 1, tail_align_tail)

    def do_tiling_key_mode_3(self, is_last_output_algin=False):
        """
        do_tiling_key_mode_3 when tiling key = 3
        """
        max_line_in_ub = 16
        max_output_size = 480 * 2
        first_dim_input_num = self.tiling_input_shape[-3]
        second_dim_input_num = self.tiling_input_shape[-2]
        third_dim_input_num = self.tiling_input_shape[-1]
        third_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        first_dim_cut_num.set_as(max_line_in_ub)
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")
        second_dim_cut_num.set_as(max_output_size // third_dim_output_num)
        with self.tik_instance.if_scope(first_dim_cut_num > first_dim_input_num):
            first_dim_cut_num.set_as(first_dim_input_num)
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        # cut inner first dim and second dim info
        first_dim_loop_num_ceil = (first_dim_input_num + first_dim_cut_num - 1) // first_dim_cut_num
        first_dim_loop_num_floor = first_dim_input_num // first_dim_cut_num
        first_dim_tail_num = first_dim_input_num % first_dim_cut_num
        second_dim_loop_num_ceil = (second_dim_input_num + second_dim_cut_num - 1) // second_dim_cut_num
        second_dim_loop_num_floor = second_dim_input_num // second_dim_cut_num
        second_dim_tail_num = second_dim_input_num % second_dim_cut_num
        first_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                        name="first_dim_sigment_ub", scope=tik.scope_ubuf)
        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub", scope=tik.scope_ubuf)
        first_dim_sigment_ub[0].set_as(first_dim_cut_num)
        first_dim_sigment_ub[1].set_as(first_dim_tail_num)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_tail_num)

        with self.tik_instance.for_range(0, first_dim_loop_num_ceil) as first_dim_idx:
            first_dim_start = first_dim_idx * first_dim_cut_num
            first_dim_process_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_process_num")
            first_dim_process_num.set_as(first_dim_sigment_ub[first_dim_idx // first_dim_loop_num_floor])
            with self.tik_instance.for_range(0, second_dim_loop_num_ceil) as second_dim_idx:
                second_dim_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_process_num")
                second_dim_process_num.set_as(second_dim_sigment_ub[second_dim_idx // second_dim_loop_num_floor])
                second_dim_start = second_dim_idx * second_dim_cut_num

                vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
                vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
                vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
                vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
                vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
                vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
                vnchw_repeat0.set_as(
                    ((second_dim_process_num * third_dim_input_num) + self.block_num - 1) // self.block_num)
                vnchw_repeat1.set_as(
                    ((second_dim_process_num * third_dim_output_num) + self.block_num - 1) // self.block_num)
                with self.tik_instance.if_scope(vnchw_repeat0 == 1):
                    vnchw_src_stride0.set_as(0)
                    vnchw_dst_stride0.set_as(0)
                with self.tik_instance.if_scope(vnchw_repeat1 == 1):
                    vnchw_src_stride1.set_as(0)
                    vnchw_dst_stride1.set_as(0)

                def _run_one_outer(_outer_num_idx, ub_list):
                    '''
                    run_one_outer
                    :param _outer_num_idx:
                    :param ub_list:
                    :return:
                    None
                    '''
                    origin_data_ub, vnchw_data_ub, vnchw_output_data_ub, _, _ = ub_list
                    _, _, _, origin_output_data_ub, origin_output_tail_data_ub = ub_list
                    input_outer_idx = _outer_num_idx + self.core_outer_start
                    input_gm_offset = input_outer_idx * self.input_offset[3]
                    output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                    output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 3)
                                               + self.tiling_pading_value[2][0] * self.output_offset[3])

                    # step1. copy 16 dims in origin_data_ub
                    with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                        burst_len = \
                            ((second_dim_process_num * third_dim_input_num) + self.block_num - 1) // self.block_num
                        src_offset = \
                            (first_dim_start + _copy_idx) * self.input_offset[4] \
                            + second_dim_start * self.input_offset[5]
                        self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                    self.input_gm[input_gm_offset + src_offset],
                                                    0, 1, burst_len, 0, 0)

                    # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                    origin_data_ub_list = [origin_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                    vnchw_data_ub_list = [vnchw_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False,
                                                vnchw_data_ub_list,
                                                origin_data_ub_list,
                                                vnchw_repeat0,
                                                vnchw_dst_stride0, vnchw_src_stride0)

                    pad_left = self.tiling_pading_value[-1][0]
                    pad_right = self.tiling_pading_value[-1][1]
                    # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                    # step3.0 copy input data to vnchw_output_data_ub with datamove

                    burst_num = second_dim_process_num
                    burst_len = third_dim_input_num
                    src_offset = 0
                    dst_offset = pad_left * self.block_num
                    src_stride = 0
                    dst_stride = pad_left + pad_right
                    self.tik_instance.data_move(vnchw_output_data_ub[dst_offset],
                                                vnchw_data_ub[src_offset],
                                                0, burst_num, burst_len, src_stride, dst_stride)

                    # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                    origin_output_data_ub_list = \
                        [origin_output_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                    vnchw_output_data_ub_list = \
                        [vnchw_output_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False,
                                                origin_output_data_ub_list,
                                                vnchw_output_data_ub_list,
                                                vnchw_repeat1,
                                                vnchw_dst_stride1, vnchw_src_stride1)

                    # step5. copy 16 dims to output
                    with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                        burst_len = (second_dim_process_num * third_dim_output_num) // self.block_num
                        dst_offset = \
                            output_outer_offset \
                            + (self.tiling_pading_value[3][0] + (first_dim_start + _copy_idx)) \
                              * self.output_offset[4] \
                            + (self.tiling_pading_value[4][0] + second_dim_start) * self.output_offset[5]
                        self.tik_instance.data_move(self.output_gm[dst_offset],
                                                    origin_output_data_ub[_copy_idx * max_output_size],
                                                    0, 1, burst_len, 0, 0)
                    # is_last_output_algin = True
                    if not is_last_output_algin:
                        copy_tail_offset = self.tik_instance.Scalar(dtype="int64", name="copy_tail_offset")
                        copy_tail_offset.set_as(third_dim_output_num % 16)
                        with self.tik_instance.if_scope(copy_tail_offset == 0):
                            copy_tail_offset.set_as(16)
                        with self.tik_instance.else_scope():
                            copy_tail_offset.set_as(16 - copy_tail_offset)
                        vnchw_repeat = 1
                        origin_output_tail_data_ub_list = \
                            [origin_output_tail_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                        vnchw_output_data_ub_list = \
                            [vnchw_output_data_ub[((third_dim_output_num * second_dim_process_num - 16) + i) * 16]
                             for i in range(0, TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(False, False,
                                                    origin_output_tail_data_ub_list,
                                                    vnchw_output_data_ub_list,
                                                    vnchw_repeat,
                                                    0, 0)

                        with self.tik_instance.for_range(0, first_dim_process_num) as _copy_idx:
                            dst_offset = \
                                output_outer_offset \
                                + (self.tiling_pading_value[3][0] + (first_dim_start + _copy_idx)) \
                                  * self.output_offset[4] \
                                + (self.tiling_pading_value[4][0] + second_dim_start) * self.output_offset[5] \
                                + second_dim_process_num * third_dim_output_num - 16
                            self.tik_instance.data_move(self.output_gm[dst_offset],
                                                        origin_output_tail_data_ub[_copy_idx * 16],
                                                        0, 1, 16 // self.block_num, 0, 0)

                origin_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                               name="origin_data_ub_ping", scope=tik.scope_ubuf)
                vnchw_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                              name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

                vnchw_output_data_ub_ping = \
                    self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                             name="vnchw_output_data_ub_ping", scope=tik.scope_ubuf)
                origin_output_data_ub_ping = \
                    self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                             name="origin_output_data_ub_ping", scope=tik.scope_ubuf)
                origin_output_tail_data_ub_ping = \
                    self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                             name="origin_output_tail_data_ub_ping", scope=tik.scope_ubuf)

                if self.x_dtype == "float32":
                    vnchw_output_data_ub_ping = vnchw_output_data_ub_ping.reinterpret_cast_to('float32')
                    self.tik_instance.vector_dup(self.block_num * 4, vnchw_output_data_ub_ping, self.pad_scalar,
                                                 max_line_in_ub * max_output_size // self.block_num // 4, 1, 8)
                    vnchw_output_data_ub_ping = vnchw_output_data_ub_ping.reinterpret_cast_to('float16')
                elif self.x_dtype == "float16":
                    self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_ping, self.pad_scalar,
                                                 max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)

                origin_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                               name="origin_data_ub_pang", scope=tik.scope_ubuf)
                vnchw_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                              name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

                vnchw_output_data_ub_pang = \
                    self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                             name="vnchw_output_data_ub_ping", scope=tik.scope_ubuf)
                origin_output_data_ub_pang = \
                    self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                             name="origin_output_data_ub_ping", scope=tik.scope_ubuf)
                origin_output_tail_data_ub_pang = \
                    self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                             name="origin_output_tail_data_ub_ping", scope=tik.scope_ubuf)

                if self.x_dtype == "float32":
                    vnchw_output_data_ub_pang = vnchw_output_data_ub_pang.reinterpret_cast_to('float32')
                    self.tik_instance.vector_dup(self.block_num * 4, vnchw_output_data_ub_pang, self.pad_scalar,
                                                 max_line_in_ub * max_output_size // self.block_num // 4, 1, 8)
                    vnchw_output_data_ub_pang = vnchw_output_data_ub_pang.reinterpret_cast_to('float16')
                elif self.x_dtype == "float16":
                    self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_pang, self.pad_scalar,
                                                 max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)

                ping_ub_list = [origin_data_ub_ping, vnchw_data_ub_ping,
                                vnchw_output_data_ub_ping, origin_output_data_ub_ping,
                                origin_output_tail_data_ub_ping]
                pang_ub_list = [origin_data_ub_pang, vnchw_data_ub_pang,
                                vnchw_output_data_ub_pang, origin_output_data_ub_pang,
                                origin_output_tail_data_ub_pang]
                with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                    _run_one_outer(_outer_idx * 2, ping_ub_list)
                    _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
                with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                    _run_one_outer(self.core_outer_num - 1, ping_ub_list)

    def do_pad(self):
        """
        do_pad with different tiling key
        """
        with self.tik_instance.if_scope(self.tiling_key == MODE0):
            # use data move to pad_v3, cut by last dim
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_0()
        with self.tik_instance.if_scope(self.tiling_key == MODE1):
            # use data move to pad_v3, cut by 0-4 dims
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_1()
        with self.tik_instance.if_scope(self.tiling_key == MODE2):
            # use vnchw to pad_v3, cut by 0-3 dims
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_2()
        with self.tik_instance.if_scope(self.tiling_key == MODE3):
            # use vnchw to pad_v3, cut by 0-2 dims
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_3()

    def pad_compute(self, outer_compile_info=None):
        """
        pad_compute
        """
        self.pad_v3_d_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}

        # add compile info
        # dtype_rate mean input_dtype byte // inner_dtype(fp16)
        # input_dtype is fp16/int16 dtype_rate == 1
        # input_dtype is fp32/int32 dtype_rate == 2
        dtype_rate = self.input_bytes_size // self.inner_bytes_size
        wr_compile_info = dict()
        wr_compile_info["ub_size"] = self.ub_number
        wr_compile_info["core_num"] = self.core_nums
        wr_compile_info["dtype_rate"] = dtype_rate
        # for StridedSliceGrad add attr to compile info
        if outer_compile_info is not None:
            for key in outer_compile_info.keys():
                wr_compile_info[key] = outer_compile_info[key]
        tbe_context.get_context().add_compile_info("vars", wr_compile_info)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=self.input_gm_list,
                                   flowtable=[self.tiling_gm],
                                   outputs=self.output_gm_list,
                                   config=opt_config)

        return self.tik_instance


@register_operator("PadV3")
def pad_v3(x, paddings, constant_values, y, mode='constant', padding_contiguous=True, kernel_name="pad_v3"):
    """ calculating pad_v3 tensor by paddings parameters

    Parameters
    ----------
    x : dict
        shape and dtype of input
    paddings: dict
        shape and dtype of output
        For each dimension D of input, paddings[D, 0] indicates how many
        values to add
        before the contents of tensor in that dimension, and paddings[D, 1]
        indicates
        how many values to add after the contents of tensor in that dimension.
    constant_values: dict
        the value to fill the tensor
    y: dict
        shape and dtype of output
    mode:str
        the cal mode of op
    padding_contiguous: bool
        judge whether the memory is contiguous
    kernel_name : str
        cce kernel name, default value is "pad_v3"

    Returns
    -------
    None.
    """
    src_dtype = x.get("dtype").lower()
    paddings_dtype = paddings.get("dtype").lower()

    supported_dtype = ("float16",)
    para_check.check_dtype(src_dtype, supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, ("int32", "int64"), param_name="paddings")

    obj = PadV3Init(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)
    obj.init_src_dst_gm((x, paddings), (y,), pad_input_idx=0, pad_outnput_idx=0)

    return obj.pad_compute()

