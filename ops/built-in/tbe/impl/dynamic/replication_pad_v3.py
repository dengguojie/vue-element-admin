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
replication_pad_v3.py
"""
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import register_operator


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # max int64
    MAX_INT64 = 2**64 - 1
    # tiling param nums
    TILING_NUMS = 16
    # 1 byte = 8 bit
    EIGHT_BIT = 8
    # reserved ub size
    RESERVED_UB = 1024
    # vnchw the minest block
    TRANS_MIN_BLKS = 16
    # vnchw the minest block for float
    TRANS_MIN_BLKS_FP32 = 8
    MODE0 = 0
    MODE1 = 1
    MODE2 = 2
    # the block size
    BLOCK_SIZE = 32


# 'pylint: disable=too-many-instance-attributes,too-many-statements,too-many-locals,too-many-lines
# 'pylint: disable=too-many-arguments,invalid-name
class ReplicationPadV3Init:
    """
    Function: class that execute replication_pad_v3
    """

    def __init__(self,
                 x,
                 paddings,
                 constant_values,
                 y,
                 mode,
                 padding_contiguous=True,
                 kernel_name='replication_pad_v3'):
        """
        init the op
        :param
        x: the input tensor
        :param
        paddings: the list of paddings
        :param
        constant_values: pad value for constant mode
        :param
        y: the output of op
        :param
        mode: pad mode
        :param
        padding_contiguous: contiguous pad or not
        :param
        kernel_name: the kernel name of op
        :return
        None
        """
        self.tik_instance = tik.Tik()
        self.unknown_max_shape = (Constant.MAX_INT64,)
        self.tiling_dtype = "int64"
        self.tiling_shape = (Constant.TILING_NUMS,)

        self.x_dtype = x.get("dtype")
        self.inner_dtype = "float16"
        self.paddings_dtype = paddings.get('dtype')
        self.constant_values = constant_values
        self.y_dtype = y.get('dtype')
        self.addtional_dtype = ["float32", "int32"]
        self.mode = mode
        self.padding_contiguous = padding_contiguous
        self.kernel_name = kernel_name
        self.input_gm = None
        self.output_gm = None
        self.tiling_gm = None
        self.block_less_16 = None
        self.input_gm_list = []
        self.output_gm_list = []
        self.input_bytes_size = 0
        self.rate = 1
        if self.x_dtype in self.addtional_dtype:
            self.rate = 2

        self.inner_bytes_size = tbe_platform.get_bit_len(self.inner_dtype) // Constant.EIGHT_BIT
        self.block_num = Constant.BLOCK_SIZE // self.inner_bytes_size
        self.dump_mask_max_x = Constant.EIGHT_BIT * self.block_num
        self.max_repeat_time = 255

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE) - Constant.RESERVED_UB
        self.ub_number = self.ub_size_bytes // self.inner_bytes_size
        self.core_nums = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)

        # tiling scaler init
        self.tiling_key = self.tik_instance.Scalar(self.tiling_dtype, "tiling_key", init_value=0)
        self.tiling_input_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_0", init_value=0)
        self.tiling_input_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_1", init_value=0)
        self.tiling_input_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_2", init_value=0)
        self.tiling_input_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_input_dim_3", init_value=0)
        self.tiling_output_dim_0 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_0", init_value=0)
        self.tiling_output_dim_1 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_1", init_value=0)
        self.tiling_output_dim_2 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_2", init_value=0)
        self.tiling_output_dim_3 = self.tik_instance.Scalar(self.tiling_dtype, "tiling_output_dim_3", init_value=0)
        self.tiling_input_shape = None
        self.tiling_output_shape = None
        self.per_core_num = self.tik_instance.Scalar(self.tiling_dtype, "per_core_num", init_value=0)
        self.padding_index_0 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_0", init_value=0)
        self.padding_index_1 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_1", init_value=0)
        self.padding_index_2 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_2", init_value=0)
        self.padding_index_3 = self.tik_instance.Scalar(self.tiling_dtype, "padding_index_3", init_value=0)

        # core scaler init
        self.core_used_num = self.tik_instance.Scalar(self.tiling_dtype, "core_used_num", init_value=0)
        self.not_last_core_num = self.tik_instance.Scalar(self.tiling_dtype, "not_last_core_num", init_value=0)
        self.last_core_num = self.tik_instance.Scalar(self.tiling_dtype, "last_core_num", init_value=0)

    def tiling_args(self, tiling_ub):
        """
        when input shape is less 6, will. expand to 6
        tiling_input_dim_cut_axis: which dim will be cut
        """
        with self.tik_instance.new_stmt_scope():
            self.tiling_key.set_as(tiling_ub[0])
            self.tiling_input_dim_0.set_as(tiling_ub[1])
            self.tiling_input_dim_1.set_as(tiling_ub[2])
            self.tiling_input_dim_2.set_as(tiling_ub[3])
            self.tiling_input_dim_3.set_as(tiling_ub[4])
            self.tiling_input_dim_3.set_as(self.tiling_input_dim_3 * self.rate)
            self.tiling_output_dim_0.set_as(tiling_ub[5])
            self.tiling_output_dim_1.set_as(tiling_ub[6])
            self.tiling_output_dim_2.set_as(tiling_ub[7])
            self.tiling_output_dim_3.set_as(tiling_ub[8])
            self.tiling_output_dim_3.set_as(self.tiling_output_dim_3 * self.rate)
            self.padding_index_0.set_as(tiling_ub[10])
            self.padding_index_0.set_as(self.padding_index_0 * self.rate)
            self.padding_index_1.set_as(tiling_ub[11])
            self.padding_index_1.set_as(self.padding_index_1 * self.rate)
            self.padding_index_2.set_as(tiling_ub[12])
            self.padding_index_3.set_as(tiling_ub[13])
            self.not_last_core_num.set_as(tiling_ub[14])
            self.last_core_num.set_as(tiling_ub[15])

    # 'pylint: disable=unused-argument
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
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
                                                  self.tiling_shape,
                                                  name="tiling_gm",
                                                  scope=tik.scope_gm)
        x_dtype = input_dict_list[0].get("dtype")
        paddings_dtype = input_dict_list[1].get("dtype")
        x_gm = self.tik_instance.Tensor(self.inner_dtype, self.unknown_max_shape, name="x", scope=tik.scope_gm)
        paddings_gm = self.tik_instance.Tensor(paddings_dtype,
                                               self.unknown_max_shape,
                                               name="paddings",
                                               scope=tik.scope_gm)

        self.input_gm_list.append(x_gm)
        self.input_gm_list.append(paddings_gm)
        if self.constant_values:
            constant_values_gm = self.tik_instance.Tensor(x_dtype,
                                                          self.unknown_max_shape,
                                                          name="constant_values",
                                                          scope=tik.scope_gm)
            self.input_gm_list.append(constant_values_gm)
        y_gm = self.tik_instance.Tensor(self.inner_dtype, self.unknown_max_shape, name="y", scope=tik.scope_gm)
        self.input_bytes_size = tbe_platform.get_bit_len(x_dtype) // Constant.EIGHT_BIT
        self.output_gm_list.append(y_gm)

        self.input_gm = self.input_gm_list[pad_input_idx]
        self.output_gm = self.output_gm_list[pad_outnput_idx]

    def replication_pad_v3_compute_tiling(self):
        """
        replication_pad_v3_compute_tiling
        """
        tiling_ub = self.tik_instance.Tensor("int64", (Constant.TILING_NUMS,), name="tiling_ub", scope=tik.scope_ubuf)
        self.tik_instance.data_move(tiling_ub, self.tiling_gm, 0, 1, Constant.TILING_NUMS // 4, 0, 0)
        self.core_used_num.set_as(tiling_ub[9])
        with self.tik_instance.for_range(0, self.core_nums, block_num=self.core_nums) as core_index:
            with self.tik_instance.if_scope(core_index < self.core_used_num):
                self.tiling_args(tiling_ub)
                self.do_pad(core_index)

    def do_tiling_key_mode_0(self, core_index):
        """
        do_tiling_key_mode_0 when tiling key = 0
        """
        ub_size_second_mode = 4
        per_ub_size = self.ub_size_bytes // self.inner_bytes_size // ub_size_second_mode
        input_ele_per_core = self.tik_instance.Scalar('int64', name='input_ele_per_core')
        output_ele_per_core = self.tik_instance.Scalar('int64', name='output_ele_per_core')
        ranges = self.tik_instance.Scalar('int64', name='ranges')
        flag = self.tik_instance.Scalar('int64', name='flag')
        units = self.tik_instance.Scalar('int64', name='units')
        first_ub_not_complete_offset = self.tik_instance.Scalar('int64', name='first_ub_not_complete_offset')
        first_ub_need_first_move_lines = self.tik_instance.Scalar('int64', name='first_ub_need_first_move_lines')
        gm_offset = self.tik_instance.Scalar('int64', name='gm_offset')
        first_ub_first_offset = self.tik_instance.Scalar('int64', name='first_ub_first_offset')
        first_ub_need_last_move_lines = self.tik_instance.Scalar('int64', name='first_ub_need_last_move_lines')
        align_output_dim_2 = self.tik_instance.Scalar('int64', name='align_output_dim_2')
        align_output_dim_3 = self.tik_instance.Scalar('int64', name='align_output_dim_2')
        time_2 = self.tik_instance.Scalar('int64', name='time')
        time_3 = self.tik_instance.Scalar('int64', name='time')

        align_output_dim_2.set_as(((self.tiling_output_dim_2 - 1) // self.block_num + 1) * self.block_num)
        align_output_dim_3.set_as(((self.tiling_output_dim_3 - 1) // self.block_num + 1) * self.block_num)
        time_2.set_as(align_output_dim_2 // Constant.TRANS_MIN_BLKS)
        time_3.set_as(align_output_dim_3 // Constant.TRANS_MIN_BLKS)
        units.set_as((Constant.TRANS_MIN_BLKS - 1) // (self.tiling_output_dim_2 * self.tiling_output_dim_3) + 1)
        first_ub_need_first_move_lines.set_as(units * self.tiling_output_dim_2 -
                                              Constant.TRANS_MIN_BLKS // self.tiling_output_dim_3)
        first_ub_not_complete_offset.set_as(self.tiling_output_dim_3 - (
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) % self.tiling_output_dim_3))
        first_ub_need_last_move_lines.set_as(
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) // self.tiling_output_dim_3)
        first_ub_first_offset.set_as((first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                     first_ub_not_complete_offset)

        input_ele_per_core.set_as(self.not_last_core_num * self.tiling_input_dim_2 * self.tiling_input_dim_3)
        output_ele_per_core.set_as(self.not_last_core_num * self.tiling_output_dim_2 * self.tiling_output_dim_3)
        with self.tik_instance.if_scope(core_index == self.core_used_num - 1):
            ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            ranges.set_as(self.not_last_core_num)
            flag.set_as((output_ele_per_core - Constant.TRANS_MIN_BLKS) //
                        (self.tiling_output_dim_3 * self.tiling_output_dim_2))
            gm_offset.set_as((core_index + 1) * output_ele_per_core - Constant.TRANS_MIN_BLKS)

        with self.tik_instance.new_stmt_scope():
            ping_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_1',
                                                 scope=tik.scope_ubuf)
            ping_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_2',
                                                 scope=tik.scope_ubuf)
            pang_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_1',
                                                 scope=tik.scope_ubuf)
            pang_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_2',
                                                 scope=tik.scope_ubuf)
            self.block_less_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                          name='block_less_16',
                                                          scope=tik.scope_ubuf)
            block_over_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                     name='block_over_16',
                                                     scope=tik.scope_ubuf)

            with self.tik_instance.for_range(0, ranges) as index:
                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[i * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[(self.padding_index_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      i * self.tiling_input_dim_3], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[(self.padding_index_2 + self.tiling_input_dim_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (self.tiling_input_dim_2 - 1) * self.tiling_input_dim_3], 0, 1, 1, 0, 0)

                with self.tik_instance.if_scope(time_2 > 1):
                    with self.tik_instance.for_range(0, time_2) as i:
                        src_list = []
                        dst_list = []
                        for j in range(Constant.TRANS_MIN_BLKS):
                            src_list.append(ping_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                      Constant.TRANS_MIN_BLKS * j])
                            dst_list.append(pang_ub_1[time_2 * Constant.TRANS_MIN_BLKS * j +
                                                      Constant.TRANS_MIN_BLKS * i])
                        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                with self.tik_instance.else_scope():
                    src_list = [ping_ub_1[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                    dst_list = [pang_ub_1[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.if_scope(time_2 > 1):
                    with self.tik_instance.for_range(0, self.padding_index_0) as i:
                        top_offset = i % self.rate * align_output_dim_2
                        self.tik_instance.data_move(ping_ub_2[i * align_output_dim_2], pang_ub_1[top_offset], 0, 1, 2,
                                                    0, 0)
                    with self.tik_instance.for_range(0, self.tiling_input_dim_3) as i:
                        self.tik_instance.data_move(ping_ub_2[(self.padding_index_0 + i) * align_output_dim_2],
                                                    pang_ub_1[i * align_output_dim_2], 0, 1, 2, 0, 0)
                    with self.tik_instance.for_range(0, self.padding_index_1) as i:
                        bottom_offset = self.rate - i % self.rate
                        self.tik_instance.data_move(
                            ping_ub_2[(self.padding_index_0 + self.tiling_input_dim_3 + i) * align_output_dim_2],
                            pang_ub_1[(self.tiling_input_dim_3 - bottom_offset) * align_output_dim_2], 0, 1, 2, 0, 0)

                with self.tik_instance.if_scope(time_2 == 1):
                    with self.tik_instance.for_range(0, self.padding_index_0) as i:
                        top_offset = i % self.rate * align_output_dim_2
                        self.tik_instance.data_move(ping_ub_2[i * Constant.TRANS_MIN_BLKS], pang_ub_1[top_offset], 0, 1,
                                                    1, 0, 0)
                    with self.tik_instance.for_range(0, self.tiling_input_dim_3) as i:
                        self.tik_instance.data_move(ping_ub_2[(self.padding_index_0 + i) * Constant.TRANS_MIN_BLKS],
                                                    pang_ub_1[i * align_output_dim_2], 0, 1, 1, 0, 0)
                    with self.tik_instance.for_range(0, self.padding_index_1) as i:
                        bottom_offset = self.rate - i % self.rate
                        self.tik_instance.data_move(
                            ping_ub_2[(self.padding_index_0 + self.tiling_input_dim_3 + i) * Constant.TRANS_MIN_BLKS],
                            pang_ub_1[(self.tiling_input_dim_3 - bottom_offset) * align_output_dim_2], 0, 1, 1, 0, 0)

                with self.tik_instance.if_scope(core_index == self.core_used_num - 1):
                    with self.tik_instance.if_scope(tik.all(time_2 == 1, time_3 == 1)):
                        src_list = [ping_ub_2[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                        dst_list = [pang_ub_2[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3], pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0,
                                1, 1, 0, 0)

                    with self.tik_instance.if_scope(tik.all(time_2 > 1, time_3 == 1)):
                        with self.tik_instance.for_range(0, time_2) as i:
                            src_list = []
                            dst_list = []
                            for j in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(ping_ub_2[time_2 * Constant.TRANS_MIN_BLKS * j +
                                                          Constant.TRANS_MIN_BLKS * i])
                                dst_list.append(pang_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                          Constant.TRANS_MIN_BLKS * j])
                            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3], pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0,
                                1, 1, 0, 0)

                    with self.tik_instance.if_scope(tik.all(time_2 == 1, time_3 > 1)):
                        with self.tik_instance.for_range(0, time_3) as i:
                            src_list = []
                            dst_list = []
                            for j in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                          Constant.TRANS_MIN_BLKS * j])
                                dst_list.append(pang_ub_2[time_3 * Constant.TRANS_MIN_BLKS * j +
                                                          Constant.TRANS_MIN_BLKS * i])
                            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3],
                                pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3], 0, 1, 2, 0, 0)

                    with self.tik_instance.if_scope(tik.all(time_2 > 1, time_3 > 1)):
                        with self.tik_instance.for_range(0, time_2) as i:
                            with self.tik_instance.for_range(0, time_3) as j:
                                src_list = []
                                dst_list = []
                                for k in range(Constant.TRANS_MIN_BLKS):
                                    src_list.append(
                                        ping_ub_2[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS * k])
                                    dst_list.append(
                                        pang_ub_2[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                                  Constant.TRANS_MIN_BLKS * i + time_3 * Constant.TRANS_MIN_BLKS * k])
                                self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3],
                                pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3], 0, 1, 2, 0, 0)

                with self.tik_instance.if_scope(tik.all(self.core_used_num > 1, core_index < self.core_used_num - 1)):
                    with self.tik_instance.if_scope(tik.all(time_2 == 1, time_3 == 1)):
                        src_list = [ping_ub_2[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                        dst_list = [pang_ub_2[Constant.TRANS_MIN_BLKS * i] for i in range(Constant.TRANS_MIN_BLKS)]
                        self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.if_scope(index < flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3],
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                        with self.tik_instance.if_scope(index == flag):
                            with self.tik_instance.for_range(0, first_ub_need_first_move_lines) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3],
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                            with self.tik_instance.for_range(
                                    first_ub_first_offset,
                                (first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                    self.tiling_output_dim_3) as i:
                                self.block_less_16[i - first_ub_first_offset].set_as(pang_ub_2[i])

                            with self.tik_instance.for_range(first_ub_need_first_move_lines,
                                                             self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       (i - first_ub_need_first_move_lines) * self.tiling_output_dim_3 +
                                                       j].set_as(pang_ub_2[i * 16 + j])
                            with self.tik_instance.if_scope(flag == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1, 1, 0,
                                                            0)

                        with self.tik_instance.if_scope(index > flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(index - flag - 1) * self.tiling_output_dim_3 *
                                                       self.tiling_output_dim_2 +
                                                       (first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       first_ub_need_last_move_lines * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3 + j].set_as(pang_ub_2[i * 16 + j])
                            with self.tik_instance.if_scope(index == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1, 1, 0,
                                                            0)

                    with self.tik_instance.if_scope(tik.all(time_2 > 1, time_3 == 1)):
                        with self.tik_instance.for_range(0, time_2) as i:
                            src_list = []
                            dst_list = []
                            for j in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(ping_ub_2[time_2 * Constant.TRANS_MIN_BLKS * j +
                                                          Constant.TRANS_MIN_BLKS * i])
                                dst_list.append(pang_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                          Constant.TRANS_MIN_BLKS * j])
                            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                        with self.tik_instance.if_scope(index < flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3],
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                        with self.tik_instance.if_scope(index == flag):
                            with self.tik_instance.for_range(0, first_ub_need_first_move_lines) as i:
                                self.tik_instance.data_move(
                                    self.output_gm[core_index * output_ele_per_core +
                                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                                   i * self.tiling_output_dim_3],
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                            with self.tik_instance.for_range(
                                    first_ub_first_offset,
                                (first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                    self.tiling_output_dim_3) as i:
                                self.block_less_16[i - first_ub_first_offset].set_as(pang_ub_2[i])

                            with self.tik_instance.for_range(first_ub_need_first_move_lines,
                                                             self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       (i - first_ub_need_first_move_lines) * self.tiling_output_dim_3 +
                                                       j].set_as(pang_ub_2[i * 16 + j])
                            with self.tik_instance.if_scope(flag == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1, 1, 0,
                                                            0)

                        with self.tik_instance.if_scope(index > flag):
                            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                                    self.block_less_16[(index - flag - 1) * self.tiling_output_dim_3 *
                                                       self.tiling_output_dim_2 +
                                                       (first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                                       self.tiling_output_dim_3 - first_ub_first_offset +
                                                       first_ub_need_last_move_lines * self.tiling_output_dim_3 +
                                                       i * self.tiling_output_dim_3 + j].set_as(pang_ub_2[i * 16 + j])
                            with self.tik_instance.if_scope(index == ranges - 1):
                                self.tik_instance.data_move(self.output_gm[gm_offset], self.block_less_16, 0, 1, 1, 0,
                                                            0)

                    with self.tik_instance.if_scope(tik.all(time_2 == 1, time_3 > 1)):
                        with self.tik_instance.for_range(0, time_3) as i:
                            src_list = []
                            dst_list = []
                            for j in range(Constant.TRANS_MIN_BLKS):
                                src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                          Constant.TRANS_MIN_BLKS * j])
                                dst_list.append(pang_ub_2[time_3 * Constant.TRANS_MIN_BLKS * j +
                                                          Constant.TRANS_MIN_BLKS * i])
                            self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3],
                                pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3], 0, 1, 1, 0, 0)
                            self.tik_instance.data_move(
                                block_over_16,
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3 +
                                               self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                            with self.tik_instance.for_range(Constant.TRANS_MIN_BLKS, self.tiling_output_dim_3) as j:
                                block_over_16[j - self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS].set_as(
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3 + j])
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3 +
                                               self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS], block_over_16, 0, 1,
                                1, 0, 0)

                    with self.tik_instance.if_scope(tik.all(time_2 > 1, time_3 > 1)):
                        with self.tik_instance.for_range(0, time_2) as i:
                            with self.tik_instance.for_range(0, time_3) as j:
                                src_list = []
                                dst_list = []
                                for k in range(Constant.TRANS_MIN_BLKS):
                                    src_list.append(
                                        ping_ub_2[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j + time_3 * Constant.TRANS_MIN_BLKS * k])
                                    dst_list.append(
                                        pang_ub_2[time_3 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                                  Constant.TRANS_MIN_BLKS * i + time_3 * Constant.TRANS_MIN_BLKS * k])
                                self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)
                        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3],
                                pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3], 0, 1, 1, 0, 0)

                            self.tik_instance.data_move(
                                block_over_16,
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3 +
                                               self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                            with self.tik_instance.for_range(
                                    self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS,
                                    self.tiling_output_dim_3) as j:
                                block_over_16[j - self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS].set_as(
                                    pang_ub_2[i * Constant.TRANS_MIN_BLKS * time_3 + j])
                            self.tik_instance.data_move(
                                self.output_gm[core_index * output_ele_per_core +
                                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                               i * self.tiling_output_dim_3 +
                                               self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS], block_over_16, 0, 1,
                                1, 0, 0)

    def do_tiling_key_mode_1(self, core_index):
        """
        do_tiling_key_mode_1 when tiling key = 1
        """
        ub_size_second_mode = 4
        per_ub_size = self.ub_size_bytes // self.inner_bytes_size // ub_size_second_mode
        input_ele_per_core = self.tik_instance.Scalar('int64', name='input_ele_per_core')
        output_ele_per_core = self.tik_instance.Scalar('int64', name='output_ele_per_core')
        ranges = self.tik_instance.Scalar('int64', name='ranges')
        input_ele_per_core.set_as(self.not_last_core_num * self.tiling_input_dim_2 * self.tiling_input_dim_3)
        output_ele_per_core.set_as(self.not_last_core_num * self.tiling_output_dim_2 * self.tiling_output_dim_3)
        with self.tik_instance.if_scope(core_index == self.core_used_num - 1):
            ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            ranges.set_as(self.not_last_core_num)
        with self.tik_instance.new_stmt_scope():
            ping_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_1',
                                                 scope=tik.scope_ubuf)
            ping_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='ping_ub_2',
                                                 scope=tik.scope_ubuf)
            pang_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_1',
                                                 scope=tik.scope_ubuf)
            pang_ub_2 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,),
                                                 name='pang_ub_2',
                                                 scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, ranges) as index:

                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[i * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[(self.padding_index_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      i * self.tiling_input_dim_3], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        ping_ub_1[(self.padding_index_2 + self.tiling_input_dim_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (self.tiling_input_dim_2 - 1) * self.tiling_input_dim_3], 0, 1, 1, 0, 0)
                align_out_dim_2 = self.tik_instance.Scalar('int64', 'align_out_dim_2')
                align_out_dim_2.set_as(
                    ((self.tiling_output_dim_2 - 1) // Constant.TRANS_MIN_BLKS + 1) * Constant.TRANS_MIN_BLKS)
                time = self.tik_instance.Scalar('int64', name='time')
                time.set_as(align_out_dim_2 // Constant.TRANS_MIN_BLKS)
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(Constant.TRANS_MIN_BLKS):
                        src_list.append(ping_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j])
                        dst_list.append(pang_ub_1[self.padding_index_0 * time * Constant.TRANS_MIN_BLKS +
                                                  time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                    self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_0) as i:
                    top_offset = i % self.rate
                    self.tik_instance.data_move(
                        pang_ub_1[i * Constant.TRANS_MIN_BLKS * time],
                        pang_ub_1[(self.padding_index_0 + top_offset) * time * Constant.TRANS_MIN_BLKS], 0, 1, time, 0,
                        0)
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(Constant.TRANS_MIN_BLKS):
                        src_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                        dst_list.append(ping_ub_1[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j])
                    self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_2[i * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                    self.tik_instance.data_move(
                        ping_ub_2[(self.padding_index_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      (i + 1) * self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        ping_ub_2[(self.padding_index_2 + self.tiling_input_dim_2 + i) * Constant.TRANS_MIN_BLKS],
                        self.input_gm[core_index * input_ele_per_core +
                                      index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                      self.tiling_input_dim_2 * self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0,
                        1, 1, 0, 0)
                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(Constant.TRANS_MIN_BLKS):
                        src_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j])
                        dst_list.append(pang_ub_2[time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                    self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_1) as i:
                    bottom_offset = self.rate - i % self.rate
                    self.tik_instance.data_move(
                        pang_ub_2[time * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS +
                                  i * Constant.TRANS_MIN_BLKS * time],
                        pang_ub_2[(Constant.TRANS_MIN_BLKS - bottom_offset) * time * Constant.TRANS_MIN_BLKS], 0, 1,
                        time, 0, 0)

                with self.tik_instance.for_range(0, time) as i:
                    src_list = []
                    dst_list = []
                    for j in range(Constant.TRANS_MIN_BLKS):
                        src_list.append(pang_ub_2[self.padding_index_1 * time * Constant.TRANS_MIN_BLKS +
                                                  time * Constant.TRANS_MIN_BLKS * j + Constant.TRANS_MIN_BLKS * i])
                        dst_list.append(ping_ub_2[Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                                  Constant.TRANS_MIN_BLKS * j])
                    self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_2) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_1[i * Constant.TRANS_MIN_BLKS], 0, 1, 1,
                        0, 0)
                    self.tik_instance.data_move(
                        pang_ub_1, self.input_gm[core_index * input_ele_per_core +
                                                 index * self.tiling_input_dim_2 * self.tiling_input_dim_3], 0, 1,
                        self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3 + self.padding_index_0], pang_ub_1, 0, 1,
                        self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    with self.tik_instance.if_scope(self.tiling_input_dim_3 % Constant.TRANS_MIN_BLKS > 0):
                        self.tik_instance.data_move(
                            pang_ub_1, self.input_gm[core_index * input_ele_per_core +
                                                     index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                                     self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           i * self.tiling_output_dim_3 + self.padding_index_0 +
                                           self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], pang_ub_1, 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3 + self.padding_index_0 + self.tiling_input_dim_3 +
                                       self.padding_index_1 - Constant.TRANS_MIN_BLKS],
                        ping_ub_2[i * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (self.padding_index_2 + i) * self.tiling_output_dim_3],
                        ping_ub_1[(self.padding_index_2 + i) * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        pang_ub_1, self.input_gm[core_index * input_ele_per_core +
                                                 index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                                 i * self.tiling_input_dim_3], 0, 1,
                        self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (self.padding_index_2 + i) * self.tiling_output_dim_3 + self.padding_index_0],
                        pang_ub_1, 0, 1, self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    with self.tik_instance.if_scope(self.tiling_input_dim_3 % Constant.TRANS_MIN_BLKS > 0):
                        self.tik_instance.data_move(
                            pang_ub_1, self.input_gm[core_index * input_ele_per_core +
                                                     index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                                     (i + 1) * self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS], 0, 1,
                            1, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           (self.padding_index_2 + i) * self.tiling_output_dim_3 +
                                           self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS + self.padding_index_0],
                            pang_ub_1, 0, 1, 1, 0, 0)

                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (i + self.padding_index_2) * self.tiling_output_dim_3 + self.padding_index_0 +
                                       self.tiling_input_dim_3 + self.padding_index_1 - Constant.TRANS_MIN_BLKS],
                        ping_ub_2[(self.padding_index_2 + i) * Constant.TRANS_MIN_BLKS], 0, 1, 1, 0, 0)

                with self.tik_instance.for_range(0, self.padding_index_3) as i:
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (self.padding_index_2 + self.tiling_input_dim_2 + i) * self.tiling_output_dim_3],
                        ping_ub_1[(self.padding_index_2 + self.tiling_input_dim_2 + i) * Constant.TRANS_MIN_BLKS], 0, 1,
                        1, 0, 0)
                    self.tik_instance.data_move(
                        pang_ub_1, self.input_gm[core_index * input_ele_per_core +
                                                 index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                                 (self.tiling_input_dim_2 - 1) * self.tiling_input_dim_3], 0, 1,
                        self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (self.padding_index_2 + self.tiling_input_dim_2 + i) * self.tiling_output_dim_3 +
                                       self.padding_index_0], pang_ub_1, 0, 1,
                        self.tiling_input_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    with self.tik_instance.if_scope(self.tiling_input_dim_3 % Constant.TRANS_MIN_BLKS > 0):
                        self.tik_instance.data_move(
                            pang_ub_1,
                            self.input_gm[core_index * input_ele_per_core +
                                          index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                                          self.tiling_input_dim_2 * self.tiling_input_dim_3 - Constant.TRANS_MIN_BLKS],
                            0, 1, 1, 0, 0)
                        self.tik_instance.data_move(
                            self.output_gm[core_index * output_ele_per_core +
                                           index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                           (self.padding_index_2 + self.tiling_input_dim_2 + i) *
                                           self.tiling_output_dim_3 + self.tiling_input_dim_3 -
                                           Constant.TRANS_MIN_BLKS + self.padding_index_0], pang_ub_1, 0, 1, 1, 0, 0)

                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       (i + self.padding_index_2 + self.tiling_input_dim_2) * self.tiling_output_dim_3 +
                                       self.padding_index_0 + self.tiling_input_dim_3 + self.padding_index_1 -
                                       Constant.TRANS_MIN_BLKS],
                        ping_ub_2[(self.padding_index_2 + i + self.tiling_input_dim_2) * Constant.TRANS_MIN_BLKS], 0, 1,
                        1, 0, 0)

    def pad_line(self, align_out_dim_3, core_index, index, ping_ub_1, input_ele_per_core):
        """pad_line
        """
        with self.tik_instance.for_range(0, self.padding_index_2) as i:
            self.tik_instance.data_move(
                ping_ub_1[i * align_out_dim_3],
                self.input_gm[core_index * input_ele_per_core +
                              index * self.tiling_input_dim_2 * self.tiling_input_dim_3], 0, 1,
                align_out_dim_3 // self.block_num, 0, 0)
        with self.tik_instance.for_range(0, self.tiling_input_dim_2) as i:
            self.tik_instance.data_move(
                ping_ub_1[(self.padding_index_2 + i) * align_out_dim_3],
                self.input_gm[core_index * input_ele_per_core +
                              index * self.tiling_input_dim_2 * self.tiling_input_dim_3 + i * self.tiling_input_dim_3],
                0, 1, align_out_dim_3 // self.block_num, 0, 0)
        with self.tik_instance.for_range(0, self.padding_index_3) as i:
            self.tik_instance.data_move(
                ping_ub_1[(self.padding_index_2 + self.tiling_input_dim_2 + i) * align_out_dim_3],
                self.input_gm[core_index * input_ele_per_core +
                              index * self.tiling_input_dim_2 * self.tiling_input_dim_3 +
                              (self.tiling_input_dim_2 - 1) * self.tiling_input_dim_3], 0, 1,
                align_out_dim_3 // self.block_num, 0, 0)

    def transpose_line_to_col(self, time, time2, ping_ub_1, pang_ub_1):
        """transpose_line_to_col
        """
        with self.tik_instance.for_range(0, time) as i:
            with self.tik_instance.for_range(0, time2) as j:
                src_list = []
                dst_list = []
                for k in range(Constant.TRANS_MIN_BLKS):
                    src_list.append(ping_ub_1[time2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time2 * Constant.TRANS_MIN_BLKS * k])
                    dst_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time * Constant.TRANS_MIN_BLKS *
                                              (k + self.padding_index_0)])
                self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

    def pad_col(self, time, pang_ub_1):
        """pad_col
        """
        with self.tik_instance.for_range(0, self.padding_index_0) as i:
            top_offset = i % self.rate
            self.tik_instance.data_move(pang_ub_1[i * Constant.TRANS_MIN_BLKS * time],
                                        pang_ub_1[(self.padding_index_0 + top_offset) * time * Constant.TRANS_MIN_BLKS],
                                        0, 1, time, 0, 0)
        with self.tik_instance.for_range(0, self.padding_index_1) as i:
            bottom_offset = self.rate - i % self.rate
            self.tik_instance.data_move(
                pang_ub_1[(self.padding_index_0 + i + self.tiling_input_dim_3) * Constant.TRANS_MIN_BLKS * time],
                pang_ub_1[(self.padding_index_0 + self.tiling_input_dim_3 - bottom_offset) * time *
                          Constant.TRANS_MIN_BLKS], 0, 1, time, 0, 0)

    def transpose_col_to_line(self, time, time2, pang_ub_1, ping_ub_1):
        """transpose_col_to_line
        """
        with self.tik_instance.for_range(0, time) as i:
            with self.tik_instance.for_range(0, time2) as j:
                src_list = []
                dst_list = []
                for k in range(Constant.TRANS_MIN_BLKS):
                    src_list.append(pang_ub_1[time * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * j +
                                              Constant.TRANS_MIN_BLKS * i + time * Constant.TRANS_MIN_BLKS * k])
                    dst_list.append(ping_ub_1[time2 * Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS * i +
                                              Constant.TRANS_MIN_BLKS * j + time2 * Constant.TRANS_MIN_BLKS * k])
                self.tik_instance.vnchwconv(True, False, dst_list, src_list, 1, 0, 0)

    def data_move_to_output_gm_last_core(self, core_index, index, output_ele_per_core, align_out_dim_3, ping_ub_1):
        """data_move_to_output_gm_last_core
        """
        with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
            self.tik_instance.data_move(
                self.output_gm[core_index * output_ele_per_core +
                               index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                               i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                align_out_dim_3 // self.block_num, 0, 0)

    def not_last_core_less_flag(self, index, flag, core_index, output_ele_per_core, align_out_dim_3, ping_ub_1):
        """not_last_core_less_flag
        """
        with self.tik_instance.if_scope(index < flag):
            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                self.tik_instance.data_move(
                    self.output_gm[core_index * output_ele_per_core +
                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                    align_out_dim_3 // self.block_num, 0, 0)

    def not_last_core_equal_flag(self, index, flag, ranges, first_ub_need_first_move_lines, core_index,
                                 output_ele_per_core, ping_ub_1, align_out_dim_3, first_ub_first_offset, block_less_16,
                                 gm_offset):
        """not_last_core_equal_flag
        """
        with self.tik_instance.if_scope(index == flag):
            with self.tik_instance.for_range(0, first_ub_need_first_move_lines) as i:
                self.tik_instance.data_move(
                    self.output_gm[core_index * output_ele_per_core +
                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                    align_out_dim_3 // self.block_num, 0, 0)

            with self.tik_instance.for_range(first_ub_first_offset,
                                             (first_ub_need_first_move_lines - 1) * align_out_dim_3 +
                                             self.tiling_output_dim_3) as i:
                block_less_16[i - first_ub_first_offset].set_as(ping_ub_1[i])

            with self.tik_instance.for_range(first_ub_need_first_move_lines, self.tiling_output_dim_2) as i:
                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                    block_less_16[(first_ub_need_first_move_lines - 1)
                                  * align_out_dim_3 + self.tiling_output_dim_3
                                  - first_ub_first_offset +
                                  (i - first_ub_need_first_move_lines) *
                                  self.tiling_output_dim_3 + j].set_as \
                        (ping_ub_1[i * align_out_dim_3 + j])
            with self.tik_instance.if_scope(flag == ranges - 1):
                self.tik_instance.data_move(self.output_gm[gm_offset], block_less_16, 0, 1, 1, 0, 0)

    def not_last_core_over_flag(self, index, flag, first_ub_need_first_move_lines, align_out_dim_3,
                                first_ub_first_offset, first_ub_need_last_move_lines, ranges, gm_offset, block_less_16,
                                ping_ub_1):
        """not_last_core_over_flag
        """
        with self.tik_instance.if_scope(index > flag):
            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                with self.tik_instance.for_range(0, self.tiling_output_dim_3) as j:
                    block_less_16[(index - flag - 1) * self.tiling_output_dim_3 *
                                  self.tiling_output_dim_2 + (
                                      first_ub_need_first_move_lines - 1)
                                  * align_out_dim_3 + self.tiling_output_dim_3
                                  - first_ub_first_offset +
                                  first_ub_need_last_move_lines *
                                  self.tiling_output_dim_3 + i *
                                  self.tiling_output_dim_3 + j].set_as \
                        (ping_ub_1[i * align_out_dim_3 + j])
            with self.tik_instance.if_scope(index == ranges - 1):
                self.tik_instance.data_move(self.output_gm[gm_offset], block_less_16, 0, 1, 1, 0, 0)

    def output_dim_3_over_16(self, core_index, output_ele_per_core, index, ping_ub_1, align_out_dim_3, block_over_16,
                             ranges):
        """output_dim_3_over_16
        """
        with self.tik_instance.if_scope(index == ranges - 1):
            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                with self.tik_instance.if_scope(i == self.tiling_output_dim_2 - 1):
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                        self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS, 0, 0)
                    with self.tik_instance.for_range(
                        (self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS -
                         (Constant.TRANS_MIN_BLKS - self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS)),
                            self.tiling_output_dim_3) as j:
                        block_over_16[j - (
                            self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS -
                            (Constant.TRANS_MIN_BLKS - self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS))].set_as(
                                ping_ub_1[i * align_out_dim_3 + j])
                    self.tik_instance.data_move(
                        self.output_gm[
                            core_index * output_ele_per_core +
                            index * self.tiling_output_dim_2 * self.tiling_output_dim_3 + i * self.tiling_output_dim_3 +
                            (self.tiling_output_dim_3 // Constant.TRANS_MIN_BLKS * Constant.TRANS_MIN_BLKS -
                             (Constant.TRANS_MIN_BLKS - self.tiling_output_dim_3 % Constant.TRANS_MIN_BLKS))],
                        block_over_16, 0, 1, 1, 0, 0)

                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(
                        self.output_gm[core_index * output_ele_per_core +
                                       index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                       i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                        align_out_dim_3 // self.block_num, 0, 0)

        with self.tik_instance.else_scope():
            with self.tik_instance.for_range(0, self.tiling_output_dim_2) as i:
                self.tik_instance.data_move(
                    self.output_gm[core_index * output_ele_per_core +
                                   index * self.tiling_output_dim_2 * self.tiling_output_dim_3 +
                                   i * self.tiling_output_dim_3], ping_ub_1[i * align_out_dim_3], 0, 1,
                    align_out_dim_3 // self.block_num, 0, 0)

    def data_move_to_output_gm_not_last_core(self, core_index, output_ele_per_core, index, ping_ub_1, align_out_dim_3,
                                             block_over_16, ranges, flag, first_ub_need_first_move_lines,
                                             first_ub_first_offset, block_less_16, gm_offset,
                                             first_ub_need_last_move_lines):
        """data_move_to_output_gm_not_last_core
        """
        with self.tik_instance.if_scope(self.tiling_output_dim_3 <= Constant.TRANS_MIN_BLKS):
            self.not_last_core_less_flag(index, flag, core_index, output_ele_per_core, align_out_dim_3, ping_ub_1)
            self.not_last_core_equal_flag(index, flag, ranges, first_ub_need_first_move_lines, core_index,
                                          output_ele_per_core, ping_ub_1, align_out_dim_3, first_ub_first_offset,
                                          block_less_16, gm_offset)
            self.not_last_core_over_flag(index, flag, first_ub_need_first_move_lines, align_out_dim_3,
                                         first_ub_first_offset, first_ub_need_last_move_lines, ranges, gm_offset,
                                         block_less_16, ping_ub_1)

        with self.tik_instance.if_scope(self.tiling_output_dim_3 > Constant.TRANS_MIN_BLKS):
            self.output_dim_3_over_16(core_index, output_ele_per_core, index, ping_ub_1, align_out_dim_3, block_over_16,
                                      ranges)

    def do_tiling_key_mode_2(self, core_index):
        """
        do_tiling_key_mode_1 when tiling key = 2
        """
        ub_size_second_mode = 2
        per_ub_size = self.ub_size_bytes // self.inner_bytes_size // ub_size_second_mode
        input_ele_per_core = self.tik_instance.Scalar('int64', name='input_ele_per_core')
        output_ele_per_core = self.tik_instance.Scalar('int64', name='output_ele_per_core')
        ranges = self.tik_instance.Scalar('int64', name='ranges')
        flag = self.tik_instance.Scalar('int64', name='flag')
        units = self.tik_instance.Scalar('int64', name='units')
        first_ub_not_complete_offset = self.tik_instance.Scalar('int64', name='first_ub_not_complete_offset')
        first_ub_need_first_move_lines = self.tik_instance.Scalar('int64', name='first_ub_need_first_move_lines')
        gm_offset = self.tik_instance.Scalar('int64', name='gm_offset')
        first_ub_first_offset = self.tik_instance.Scalar('int64', name='first_ub_first_offset')
        first_ub_need_last_move_lines = self.tik_instance.Scalar('int64', name='first_ub_need_last_move_lines')
        units.set_as((Constant.TRANS_MIN_BLKS - 1) // (self.tiling_output_dim_2 * self.tiling_output_dim_3) + 1)
        first_ub_need_first_move_lines.set_as(units * self.tiling_output_dim_2 -
                                              Constant.TRANS_MIN_BLKS // self.tiling_output_dim_3)
        first_ub_not_complete_offset.set_as(self.tiling_output_dim_3 - (
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) % self.tiling_output_dim_3))
        first_ub_need_last_move_lines.set_as(
            (Constant.TRANS_MIN_BLKS -
             (units - 1) * self.tiling_output_dim_3 * self.tiling_output_dim_2) // self.tiling_output_dim_3)
        first_ub_first_offset.set_as((first_ub_need_first_move_lines - 1) * Constant.TRANS_MIN_BLKS +
                                     first_ub_not_complete_offset)

        input_ele_per_core.set_as(self.not_last_core_num * self.tiling_input_dim_2 * self.tiling_input_dim_3)
        output_ele_per_core.set_as(self.not_last_core_num * self.tiling_output_dim_2 * self.tiling_output_dim_3)
        with self.tik_instance.if_scope(core_index == self.core_used_num - 1):
            ranges.set_as(self.last_core_num)
        with self.tik_instance.else_scope():
            ranges.set_as(self.not_last_core_num)
            flag.set_as((output_ele_per_core - Constant.TRANS_MIN_BLKS) //
                        (self.tiling_output_dim_3 * self.tiling_output_dim_2))
            gm_offset.set_as((core_index + 1) * output_ele_per_core - Constant.TRANS_MIN_BLKS)
        align_out_dim_2 = self.tik_instance.Scalar('int64', 'align_out_dim_2')
        align_out_dim_2.set_as(
            ((self.tiling_output_dim_2 - 1) // Constant.TRANS_MIN_BLKS + 1) * Constant.TRANS_MIN_BLKS)
        time = self.tik_instance.Scalar('int64', name='time')
        time.set_as(align_out_dim_2 // Constant.TRANS_MIN_BLKS)
        align_out_dim_3 = self.tik_instance.Scalar('int64', 'align_out_dim_3')
        align_out_dim_3.set_as(
            ((self.tiling_output_dim_3 - 1) // Constant.TRANS_MIN_BLKS + 1) * Constant.TRANS_MIN_BLKS)
        time2 = self.tik_instance.Scalar('int64', name='time')
        time2.set_as(align_out_dim_3 // Constant.TRANS_MIN_BLKS)
        ping_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,), name='ping_ub_1', scope=tik.scope_ubuf)
        pang_ub_1 = self.tik_instance.Tensor(self.inner_dtype, (per_ub_size,), name='pang_ub_1', scope=tik.scope_ubuf)
        block_over_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                 name='block_over_16',
                                                 scope=tik.scope_ubuf)
        block_less_16 = self.tik_instance.Tensor(self.inner_dtype, (Constant.TRANS_MIN_BLKS,),
                                                 name='block_less_16',
                                                 scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, ranges) as index:
            self.pad_line(align_out_dim_3, core_index, index, ping_ub_1, input_ele_per_core)
            self.transpose_line_to_col(time, time2, ping_ub_1, pang_ub_1)
            self.pad_col(time, pang_ub_1)
            self.transpose_col_to_line(time, time2, pang_ub_1, ping_ub_1)
            with self.tik_instance.if_scope(core_index == self.core_used_num - 1):
                self.data_move_to_output_gm_last_core(core_index, index, output_ele_per_core, align_out_dim_3,
                                                      ping_ub_1)

            with self.tik_instance.if_scope(tik.all(self.core_used_num > 1, core_index < self.core_used_num - 1)):
                self.data_move_to_output_gm_not_last_core(core_index, output_ele_per_core, index, ping_ub_1,
                                                          align_out_dim_3, block_over_16, ranges, flag,
                                                          first_ub_need_first_move_lines, first_ub_first_offset,
                                                          block_less_16, gm_offset, first_ub_need_last_move_lines)

    def do_pad(self, core_index):
        """
        do_pad with different tiling key
        """
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE0):
            with self.tik_instance.new_stmt_scope():
                with self.tik_instance.if_scope(self.tiling_input_dim_3 > Constant.TRANS_MIN_BLKS_FP32):
                    self.do_tiling_key_mode_2(core_index)
                with self.tik_instance.else_scope():
                    self.do_tiling_key_mode_0(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE1):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_1(core_index)
        with self.tik_instance.if_scope(self.tiling_key == Constant.MODE2):
            with self.tik_instance.new_stmt_scope():
                self.do_tiling_key_mode_2(core_index)

    def pad_compute(self, outer_compile_info=None):
        """
        pad_compute
        """
        self.replication_pad_v3_compute_tiling()
        opt_config = {"out_of_bound_sync_check": True}

        # add compile info
        wr_compile_info = {}
        wr_compile_info["core_num"] = self.core_nums
        wr_compile_info["mode"] = self.mode
        wr_compile_info["padding_contiguous"] = self.padding_contiguous
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


# use the same tiling logic as reflection_pad_v3
@register_operator("ReflectionPadV3")
def replication_pad_v3(x,
                       paddings,
                       constant_values,
                       y,
                       mode,
                       padding_contiguous=True,
                       kernel_name="replication_pad_v3"):
    """ calculating replication_pad_v3 tensor by paddings parameters

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
    mode: str
        the mode of calculate
    padding_contiguous: bool
        judge whether the memory is contiguous
    kernel_name : str
        cce kernel name, default value is "replication_pad_v3"

    Returns
    -------
    None.
    """
    x_dtype = x.get("dtype").lower()
    paddings_dtype = paddings.get("dtype").lower()
    x_supported_dtype = ("float16", "float32")
    paddings_supported_dtype = ("int32", "int64")
    para_check.check_dtype(x_dtype, x_supported_dtype, param_name="x")
    para_check.check_dtype(paddings_dtype, paddings_supported_dtype, param_name="paddings")
    obj = ReplicationPadV3Init(x, paddings, constant_values, y, mode, padding_contiguous, kernel_name)
    obj.init_src_dst_gm((x, paddings, constant_values), (y,), pad_input_idx=0, pad_outnput_idx=0)
    return obj.pad_compute()
