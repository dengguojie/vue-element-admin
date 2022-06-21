"""
Copyright (C) 2021-2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

swin_attention_score
"""

from te.utils import para_check
import te.platform as tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import error_manager_vector


# 'pylint: disable=too-many-arguments,too-many-locals
# 'pylint: disable=too-few-public-methods
# 'pylint: disable=too-many-statements, too-many-arguments, too-many-lines
class Constant:
    """
    The class for constant
    """
    MININUM_NUM_FLOAT = -(3.4028235 ** 38)
    DTYPE_BYTES = {"float32": 4, "float16": 2}
    TRAINGING = 0
    TUILI = 1


# 'pylint: disable=too-many-public-methods
class MatMulSoftmax:
    """
    MatMulSoftmax class
    """

    # 'pylint: disable=unused-argument
    def __init__(self, x1, x2, add_x1, add_x2, drop_mask, x3, softmax_output, y, kernel_name):
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.tik = tik
        self.cur_op_core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.matmul_dtype = self.vector_dtype = "float16"
        self.x1_shape, self.x2_shape, self.x3_shape = x1["shape"], x2["shape"], x3["shape"]
        self.model_type = Constant.TRAINGING
        self.model_type = Constant.TUILI

        self.ele_shape1 = add_x1["shape"]
        if add_x2 is not None:
            self.structure_swin = True
            self.ele_shape2 = add_x2["shape"]
            self.drop_shape = drop_mask["shape"]
        else:
            self.structure_swin = False
        self.y_shape = y["shape"]

        self.batch_dim = self.x1_shape[0] * self.x1_shape[1] // self.cur_op_core_num
        self.first_m_dim = self.x1_shape[3]
        self.second_m_dim = self.x1_shape[3]

        self.first_k_dim = self.x1_shape[2]
        self.second_k_dim = self.x2_shape[3]

        self.first_n_dim = self.x2_shape[3]
        self.second_n_dim = self.x3_shape[2]

        self.block_stride = 1
        self.repeat_stride = 8
        self.repeat_once_size = 128

        self.batch_outer_num = self.batch_dim
        self.m_num = self.first_m_dim

        self.block_num = 16
        self.mul_x_shape = [self.block_num]
        self.double_factor = 2
        self.kernel_name = kernel_name
        self.init_gm()

    def init_gm(self):
        self.x1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x1_shape, name="x1_gm", scope=self.tik.scope_gm)
        self.x2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x2_shape, name="x2_gm", scope=self.tik.scope_gm)
        self.mul_gm = self.tik_instance.Tensor(self.matmul_dtype, self.mul_x_shape,
                                               name="mul_gm", scope=self.tik.scope_gm)
        self.add1_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape1,
                                                name="add1_gm", scope=self.tik.scope_gm)
        if self.structure_swin:
            self.add2_gm = self.tik_instance.Tensor(self.matmul_dtype, self.ele_shape2,
                                                    name="add2_gm", scope=self.tik.scope_gm)
            self.drop_mask_gm = self.tik_instance.Tensor(self.matmul_dtype, self.drop_shape,
                                                         name="drop_mask_gm", scope=self.tik.scope_gm)
        self.x3_gm = self.tik_instance.Tensor(self.matmul_dtype, self.x3_shape, name="x3_gm", scope=self.tik.scope_gm)
        if self.model_type == Constant.TRAINGING:
            self.softmax_output_gm = self.tik_instance.Tensor(self.matmul_dtype, self.softmax_output_shape,
                                                              name="softmax_output_gm", scope=self.tik.scope_gm)

        self.y_gm = self.tik_instance.Tensor(self.matmul_dtype, self.y_shape, name="y_gm", scope=self.tik.scope_gm)

    def mid_data_to_ub(self, tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size,
                       ub_mask, block_idx, cur_b_idx, cur_om_idx):
        tensor_a_src_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                              m * single_m_size * self.block_num * self.block_num
        tensor_a_dst_offset = 0
        tensor_a_repeat_times = self.first_n_dim
        tesnor_a_data_size = single_m_size * self.block_num
        tensor_a_src_stride = (om_size - single_m_size) * self.block_num
        tensor_a_dst_stride = 0
        self.tik_instance.data_move(tensor_c_ub[tensor_a_dst_offset], tensor_c[tensor_a_src_offset],
                                    sid=0, nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                    src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

        mask_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                      self.block_num * self.block_num + \
                      cur_b_idx * self.first_m_dim * self.first_n_dim * self.block_num * self.block_num + \
                      cur_om_idx * om_size * self.first_n_dim * self.block_num * self.block_num + \
                      cur_m_idx * single_m_size * self.first_n_dim * self.block_num * self.block_num
        mask_length = single_m_size * self.first_n_dim * self.block_num
        if self.model_type == Constant.TRAINGING and self.structure_swin:
            self.tik_instance.data_move(ub_mask[0], self.drop_mask_gm[mask_offset],
                                        0, 1, mask_length, 0, 0)

    def mat_mul_second_compute_front(self, second_bmm_compute_buffers, second_bmm_compute_idxs,
                                     second_bmm_compute_each_layer_size):
        tensor_a, tensor_c, tensor_c_ub, ub_mask, ub_cast, reduce_ub = second_bmm_compute_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                self.mid_data_to_ub(tensor_c, tensor_c_ub, om_size, cur_m_idx, cur_m_size, m, single_m_size,
                                    ub_mask, block_idx, cur_b_idx, cur_om_idx)
                tensor_c_ub_back = self.softmax_compute_last_part(tensor_c_ub, om_size, cur_m_size,
                                                                  single_m_size, single_k_size,
                                                                  block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m,
                                                                  ub_cast, reduce_ub, ub_mask)
                self.tik_instance.data_move(tensor_a[0], tensor_c_ub_back[0], sid=0,
                                            nburst = 1, burst = single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

    def mat_mul_second_compute_mid(self, second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                   second_bmm_compute_idxs, second_bmm_compute_each_layer_size):
        tensor_a, tensor_c, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, ub_mask = second_bmm_compute_buffers
        tensor_a_l1a_s_ub, tensor_c_ub2, ub_mask2, ub_cast2, reduce_ub2 = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                # do the next time
                self.mid_data_to_ub(tensor_c, tensor_c_ub2, om_size, cur_m_idx + 1, cur_m_size, m, single_m_size,
                                    ub_mask2, block_idx, cur_b_idx, cur_om_idx)
                l1a_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                # do the last time
                if self.model_type == Constant.TRAINGING:
                    self.tik_instance.load2dv1(tensor_a_l0a, tensor_a[l1a_offset], 0, l1a_repeat_times, 1, 0, False)
                else:
                    self.tik_instance.load2dv2(tensor_a_l0a, tensor_a[l1a_offset], 0, l1a_repeat_times, 0, 1, 0, False)
                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num, 0)
                # do the next time
                tensor_c_ub_back_2 = self.softmax_compute_last_part(tensor_c_ub2, om_size, cur_m_size,
                                                                    single_m_size, single_k_size,
                                                                    block_idx, cur_b_idx, cur_om_idx, cur_m_idx + 1, m,
                                                                    ub_cast2, reduce_ub2, ub_mask2)
                self.tik_instance.data_move(tensor_a_l1a_s_ub[0], tensor_c_ub_back_2[0], sid=0,
                                            nburst=1, burst=single_k_size * single_m_size * self.block_num,
                                            src_stride=0, dst_stride=0)

                # do the last time
                cc_to_ub_dst_stride = cc_to_ub_src_stride = 0
                ub_mask16 = ub_mask.reinterpret_cast_to("float16")
                self.tik_instance.tensor_mov(ub_mask16, tensor_c_l0c, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)
                inner_blk = (block_idx * self.batch_outer_num + cur_b_idx) % self.x1_shape[1]
                outer_blk = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[1]
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.x1_shape[0] * self.second_m_dim - single_m_size) * self.block_num

                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                with self.tik_instance.for_range(0, repeat_times) as repeat_time:
                    self.tik_instance.data_move(self.y_gm[output_dst_offset + repeat_time * single_data_size],
                                                ub_mask16[repeat_time * (single_data_size + output_dst_stride)],
                                                sid=0, nburst=1, burst=single_data_size,
                                                src_stride=0, dst_stride=0)

    def mat_mul_second_compute_last(self, second_bmm_tensor_b_l0b, second_bmm_compute_db_buffers,
                                   second_bmm_compute_idxs, second_bmm_compute_each_layer_size):
        tensor_b_l0b = second_bmm_tensor_b_l0b
        tensor_a_l1a_s_ub, tensor_a_l0a_ub, tensor_c_l0c_ub, tensor_c_ub2 = second_bmm_compute_db_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = second_bmm_compute_idxs
        om_size, cur_m_size, cur_k_size, cur_n_size = second_bmm_compute_each_layer_size

        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range = cur_m_size // single_m_size
        n_range = cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                l1a_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                # do the last time
                if self.model_type == Constant.TRAINGING:
                    self.tik_instance.load2dv1(tensor_a_l0a_ub, tensor_a_l1a_s_ub[l1a_offset],
                                               0, l1a_repeat_times, 1, 0, False)
                else:
                    self.tik_instance.load2dv2(tensor_a_l0a_ub, tensor_a_l1a_s_ub[l1a_offset],
                                               0, l1a_repeat_times, 0, 1, 0, False)
                self.tik_instance.mmad(tensor_c_l0c_ub, tensor_a_l0a_ub, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num, 0)

                # do the last time
                cc_to_ub_dst_stride = cc_to_ub_src_stride = 0

                self.tik_instance.tensor_mov(tensor_c_ub2, tensor_c_l0c_ub, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)
                outer_blk = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[1]
                inner_blk = (block_idx * self.batch_outer_num + cur_b_idx) % self.x1_shape[1]
                single_data_size = single_m_size * self.block_num
                repeat_times = single_n_size
                output_dst_stride = (self.x1_shape[0] * self.second_m_dim - single_m_size) * self.block_num
                # inner depend which row
                # outer depends which 2[32, 16, 16]
                output_dst_offset = inner_blk * single_n_size * self.x1_shape[0] * \
                                    self.second_m_dim * self.block_num * self.block_num + \
                                    outer_blk * self.second_m_dim * self.block_num * self.block_num + \
                                    cur_om_idx * om_size * self.block_num * self.block_num + \
                                    cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                    m * single_m_size * self.block_num * self.block_num

                with self.tik_instance.for_range(0, repeat_times) as repeat_time:
                    self.tik_instance.data_move(self.y_gm[output_dst_offset + repeat_time * single_data_size],
                                                tensor_c_ub2[repeat_time * (single_data_size + output_dst_stride)],
                                                sid=0, nburst=1, burst=single_data_size,
                                                src_stride=0, dst_stride=0)

    def mat_mul_compute(self, first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size):
        """
        first bmm compute.
        """
        tensor_a, tensor_y, tensor_a_l0a, tensor_b_l0b, tensor_c_l0c, tensor_c_ub, mul_value, \
            elewise_data1_ub2, elewise_data2_ub2 = first_bmm_compute_buffers
        block_idx, cur_b_idx, cur_om_idx, cur_m_idx = first_bmm_compute_idx
        om_size, cur_m_size, cur_k_size, cur_n_size = first_bmm_compute_each_layer_size
        single_m_size, single_k_size, single_n_size = cur_m_size, cur_k_size, cur_n_size
        m_range, n_range = cur_m_size // single_m_size, cur_n_size // single_n_size

        with self.tik_instance.for_range(0, m_range) as m:
            with self.tik_instance.for_range(0, n_range) as n:
                for ck in range(cur_k_size):
                    tensor_a_src_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_k_dim * \
                                          self.block_num * self.block_num + cur_b_idx * self.first_m_dim * \
                                          self.first_k_dim * self.block_num * self.block_num + \
                                          cur_om_idx * om_size * self.block_num * self.block_num + \
                                          cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                          m * single_m_size * self.block_num * self.block_num + \
                                          ck * self.first_m_dim * self.block_num * self.block_num
                    tensor_a_dst_offset = ck * self.block_num * self.block_num
                    tesnor_a_data_size = self.block_num
                    tensor_a_repeat_times = single_m_size
                    tensor_a_src_stride = 0
                    tensor_a_dst_stride = (cur_k_size - 1) * self.block_num
                    self.tik_instance.data_move(tensor_a[tensor_a_dst_offset], self.x1_gm[tensor_a_src_offset],
                                                sid=0, nburst=tensor_a_repeat_times, burst=tesnor_a_data_size,
                                                src_stride=tensor_a_src_stride, dst_stride=tensor_a_dst_stride)

                first_mov1 = (block_idx * self.batch_outer_num + cur_b_idx) % self.x1_shape[1]
                ele_move_offset1 = first_mov1 * self.first_m_dim * self.first_n_dim * \
                                   self.block_num * self.block_num + \
                                   cur_om_idx * om_size * self.block_num * self.block_num + \
                                   cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                   m * single_m_size * self.block_num * self.block_num + \
                                   n * self.first_m_dim * self.block_num * self.block_num
                
                first_mov2 = (block_idx * self.batch_outer_num + cur_b_idx) // self.x1_shape[0]
                ele_move_offset2 = first_mov2 * self.first_m_dim * self.first_n_dim * \
                                   self.block_num * self.block_num + \
                                   cur_om_idx * om_size * self.block_num * self.block_num + \
                                   cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                                   m * single_m_size * self.block_num * self.block_num + \
                                   n * self.first_m_dim * self.block_num * self.block_num
                
                ele_move_repeat_times = single_n_size
                ele_move_data_size = single_m_size * self.block_num
                ele_move_src_stride = (self.first_m_dim - single_m_size) * self.block_num
                ele_move_dst_stride = 0
                self.tik_instance.data_move(elewise_data1_ub2, self.add1_gm[ele_move_offset1],
                                            sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                            src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)
                if self.structure_swin:
                    self.tik_instance.data_move(elewise_data2_ub2, self.add2_gm[ele_move_offset2],
                                                sid=0, nburst=ele_move_repeat_times, burst=ele_move_data_size,
                                                src_stride=ele_move_src_stride, dst_stride=ele_move_dst_stride)

                l1a_offset = 0
                l1a_repeat_times = single_m_size * cur_k_size
                if self.model_type == Constant.TRAINGING:
                    self.tik_instance.load2dv1(tensor_a_l0a, tensor_a[l1a_offset],
                                               0, l1a_repeat_times, 1, 0, False)
                else:
                    self.tik_instance.load2dv2(tensor_a_l0a, tensor_a[l1a_offset],
                                               0, l1a_repeat_times, 0, 1, 0, False)

                self.tik_instance.mmad(tensor_c_l0c, tensor_a_l0a, tensor_b_l0b,
                                       single_m_size * self.block_num,
                                       single_k_size * self.block_num,
                                       single_n_size * self.block_num, 0)

                cc_to_ub_dst_stride = cc_to_ub_src_stride = 0

                self.tik_instance.tensor_mov(tensor_c_ub, tensor_c_l0c, 'm', 1, single_m_size * single_n_size,
                                             cc_to_ub_dst_stride, cc_to_ub_src_stride)

                ele_compute_repeat_times = single_m_size * single_n_size * 16 * 16 // self.repeat_once_size
                tail = max(ele_compute_repeat_times // 255, 0) * (ele_compute_repeat_times % 255)
                ele_compute_repeat_times = min(255, ele_compute_repeat_times)
                self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub, tensor_c_ub, mul_value,
                                        ele_compute_repeat_times, self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride)
                if tail != 0:
                    self.tik_instance.vmuls(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                            tensor_c_ub[255 * self.repeat_once_size],
                                            mul_value, tail, self.block_stride, self.block_stride,
                                            self.repeat_stride, self.repeat_stride)

                self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                       elewise_data1_ub2, ele_compute_repeat_times,
                                       self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                if self.structure_swin:
                    self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub, tensor_c_ub,
                                        elewise_data2_ub2, ele_compute_repeat_times,
                                        self.block_stride, self.block_stride, self.block_stride,
                                        self.repeat_stride, self.repeat_stride, self.repeat_stride)

                if tail != 0:
                    self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                           tensor_c_ub[255 * self.repeat_once_size],
                                           elewise_data1_ub2[255 * self.repeat_once_size], tail,
                                           self.block_stride, self.block_stride, self.block_stride,
                                           self.repeat_stride, self.repeat_stride, self.repeat_stride)
                    if self.structure_swin:
                        self.tik_instance.vadd(self.repeat_once_size, tensor_c_ub[255 * self.repeat_once_size],
                                            tensor_c_ub[255 * self.repeat_once_size],
                                            elewise_data2_ub2[255 * self.repeat_once_size], tail,
                                            self.block_stride, self.block_stride, self.block_stride,
                                            self.repeat_stride, self.repeat_stride, self.repeat_stride)
                self.softmax_compute(tensor_c_ub, single_m_size, single_n_size)

                # copy_ub_to_l1
                mid_offset = cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                             m * single_m_size * self.block_num * self.block_num
                mid_data_lengh = single_m_size * self.block_num
                mid_src_stride = 0
                mid_dst_stride = (om_size - single_m_size) * self.block_num
                self.tik_instance.data_move(tensor_y[mid_offset], tensor_c_ub, sid=0,
                                            nburst=single_n_size, burst=mid_data_lengh,
                                            src_stride=mid_src_stride, dst_stride=mid_dst_stride)

    def softmax_compute_last_part(self, tensor_c_ub, om_size, cur_m_size,
                                  cur_m, cur_n, block_idx, cur_b_idx, cur_om_idx, cur_m_idx, m,
                                  ub_cast, reduce_ub, ub_mask):
        fp32_repeat_once_nums = 64
        max_repeat_times = 255
        repeat_times = cur_m * cur_n * self.block_num * self.block_num // fp32_repeat_once_nums
        repeat_times = min(repeat_times, max_repeat_times)
        insn_tail = max(repeat_times // 255, 0) * (repeat_times % 255)

        self.tik_instance.vconv(fp32_repeat_once_nums, "", ub_cast[0], tensor_c_ub[0], repeat_times, 1, 1, 8, 4)
        self.tik_instance.vexp(fp32_repeat_once_nums, ub_cast[0], ub_cast[0], repeat_times, 1, 1, 8, 8)
        self.tik_instance.vconv(fp32_repeat_once_nums, "", tensor_c_ub[0], ub_cast[0], repeat_times, 1, 1, 4, 8)
        if insn_tail > 0:
            self.tik_instance.vconv(fp32_repeat_once_nums, "", ub_cast[repeat_times * fp32_repeat_once_nums],
                                    tensor_c_ub[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 8, 4)
            self.tik_instance.vexp(fp32_repeat_once_nums, ub_cast[repeat_times * fp32_repeat_once_nums],
                                   ub_cast[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 8, 8)
            self.tik_instance.vconv(fp32_repeat_once_nums, "", tensor_c_ub[repeat_times * fp32_repeat_once_nums],
                                    ub_cast[repeat_times * fp32_repeat_once_nums], insn_tail, 1, 1, 4, 8)
        vmax_range = cur_n
        src_tensor = ub_cast
        while (vmax_range > 1):
            if vmax_range % 2 == 0:
                repeat_time = cur_m * vmax_range * self.block_num * self.block_num // fp32_repeat_once_nums // 2
                src_offset = cur_m * vmax_range * self.block_num * self.block_num // 2
                self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range // 2
            else:
                repeat_time = cur_m * self.block_num * self.block_num // fp32_repeat_once_nums
                src_offset = (vmax_range - 1) * cur_m * self.block_num * self.block_num
                self.tik_instance.vadd(fp32_repeat_once_nums, src_tensor[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range - 1

        repeat_time = cur_m * self.block_num
        self.tik_instance.vcadd(self.block_num, reduce_ub[0], src_tensor[0], repeat_time,
                                self.block_stride, self.block_stride, 2)
        vrec_mask = repeat_time
        self.tik_instance.vrec(vrec_mask, reduce_ub[0], reduce_ub[0], 1, 1, 1, 0, 0)
        ub_reduceadd_fp16 = reduce_ub.reinterpret_cast_to("float16")
        self.tik_instance.vconv(vrec_mask, "", ub_reduceadd_fp16[0], reduce_ub[0], 1, 1, 1, 0, 0)
        # broadcast
        ub_broadcast = ub_cast.reinterpret_cast_to("uint16")

        self.tik_instance.vector_dup(self.repeat_once_size,
                                     ub_broadcast[cur_m * cur_n * self.block_num * self.block_num],
                                     self.tik_instance.Scalar(init_value=0, dtype="uint16"), 1, 1, 8)

        ub_reduceadd_int16 = reduce_ub.reinterpret_cast_to("uint16")

        for cur_fz in range(cur_m):
            dst_offset = cur_fz * self.block_num * self.block_num
            src_offset = cur_fz * self.block_num
            self.tik_instance.vor(self.block_num, ub_broadcast[dst_offset], ub_reduceadd_int16[src_offset],
                                  ub_broadcast[cur_m * cur_n * self.block_num * self.block_num],
                                  self.block_num, 1, 1, 0, 1, 0, 0)
            self.tik_instance.vtranspose(ub_broadcast[dst_offset], ub_broadcast[dst_offset])

        ub_broadcast_fp16 = ub_cast.reinterpret_cast_to("float16")
        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vmul(self.repeat_once_size, tensor_c_ub[idx * self.repeat_once_size],
                                   tensor_c_ub[idx * self.repeat_once_size],
                                   ub_broadcast_fp16[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, cur_m * self.block_num, cur_m * self.block_num, 0)
        if self.model_type == Constant.TRAINGING:
            ub_broadcast_fp16_mid_offset = cur_n * cur_m * self.block_num * self.block_num
            vconv_repeat_times = cur_n * cur_m * self.block_num * self.block_num // self.repeat_once_size
            self.tik_instance.vconv(self.repeat_once_size, "", ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                    ub_mask[0], vconv_repeat_times, 1, 1, 8, 4)
            self.tik_instance.vmul(self.repeat_once_size, ub_broadcast_fp16[ub_broadcast_fp16_mid_offset],
                                   ub_broadcast_fp16[ub_broadcast_fp16_mid_offset], tensor_c_ub[0],
                                   vconv_repeat_times, 1, 1, 1, 8, 8, 8)
            trans_nz_zz_dst_repeat_stride = self.block_num
            trans_nz_zz_src_repeat_stride = cur_m * self.block_num
            for i in range(cur_m):
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num],
                                        ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                          i * self.block_num * self.block_num], 0, cur_n, 1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride, 0)
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num +
                                                          self.repeat_once_size],
                                        ub_broadcast_fp16[ub_broadcast_fp16_mid_offset +
                                                          i * self.block_num * self.block_num + self.repeat_once_size],
                                        0, cur_n, 1, 1, trans_nz_zz_dst_repeat_stride,
                                        trans_nz_zz_src_repeat_stride, 0)
        else:
            trans_nz_zz_dst_repeat_stride = self.block_num
            trans_nz_zz_src_repeat_stride = cur_m * self.block_num
            for i in range(cur_m):
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num],
                                        tensor_c_ub[i * self.block_num * self.block_num], 0, cur_n, 1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride, 0)
                self.tik_instance.vadds(self.repeat_once_size,
                                        ub_broadcast_fp16[i * cur_n * self.block_num * self.block_num +
                                                          self.repeat_once_size],
                                        tensor_c_ub[i * self.block_num * self.block_num +
                                                    self.repeat_once_size], 0, cur_n, 1, 1,
                                        trans_nz_zz_dst_repeat_stride, trans_nz_zz_src_repeat_stride, 0)

        # dma copy_ub_to_gm
        mid_gm_offset = block_idx * self.batch_outer_num * self.first_m_dim * self.first_n_dim * \
                        self.block_num * self.block_num + cur_b_idx * self.first_n_dim * self.first_m_dim * \
                        self.block_num * self.block_num + cur_om_idx * om_size * self.block_num * self.block_num + \
                        cur_m_idx * cur_m_size * self.block_num * self.block_num + \
                        m * cur_m * self.block_num * self.block_num

        mid_data_lengh = cur_m * self.block_num
        mid_src_stride = 0
        mid_dst_stride = (self.first_m_dim - cur_m) * self.block_num
        if self.model_type == Constant.TRAINGING:
            self.tik_instance.data_move(self.softmax_output_gm[mid_gm_offset], tensor_c_ub, sid=0,
                                        nburst=cur_n, burst=mid_data_lengh,
                                        src_stride=mid_src_stride, dst_stride=mid_dst_stride)

        return ub_broadcast_fp16

    def softmax_compute(self, tensor_input, cur_m, cur_n):
        # do softmax compute ori 32 32 16 16] cur 32 2 16 16
        softmax_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_n, cur_m, self.block_num, self.block_num],
                                              name="softmax_ub", scope=self.tik.scope_ubuf)
        reduce_ub = self.tik_instance.Tensor(self.matmul_dtype, [cur_m * self.block_num],
                                             name="reduce_ub", scope=self.tik.scope_ubuf)
        ub_broadcast = self.tik_instance.Tensor("uint16", (self.first_n_dim * self.block_num,),
                                                name="ub_broadcast", scope=self.tik.scope_ubuf)
        vmax_range = cur_n
        src_tensor = tensor_input
        while (vmax_range > 1):
            if vmax_range != cur_n:
                src_tensor = softmax_ub
            if vmax_range % 2 == 0:
                repeat_time = cur_m * vmax_range * self.block_num * self.block_num // self.repeat_once_size // 2
                src_offset = cur_m * vmax_range * self.block_num * self.block_num // 2
                self.tik_instance.vmax(self.repeat_once_size, softmax_ub[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range // 2
            else:
                repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size
                src_offset = (vmax_range - 1) * cur_m * self.block_num * self.block_num
                self.tik_instance.vmax(self.repeat_once_size, softmax_ub[0], src_tensor[0], src_tensor[src_offset],
                                       repeat_time, self.block_stride, self.block_stride, self.block_stride,
                                       self.repeat_stride, self.repeat_stride, self.repeat_stride)
                vmax_range = vmax_range - 1

        repeat_time = cur_m * self.block_num * self.block_num // self.repeat_once_size

        self.tik_instance.vcgmax(self.repeat_once_size, reduce_ub[0], softmax_ub[0], repeat_time,
                                 self.block_stride, self.block_stride, self.repeat_stride)

        ub_dup = softmax_ub.reinterpret_cast_to("uint16")
        self.tik_instance.vector_dup(self.repeat_once_size, ub_dup[cur_m * self.block_num * self.block_num],
                                     self.tik_instance.Scalar(init_value=0, dtype="uint16"),
                                     self.block_stride, self.block_stride, self.repeat_stride)
        ub_reducemax_int16 = reduce_ub.reinterpret_cast_to("uint16")
        for cur_fz in range(cur_m):
            dst_offset = cur_fz * self.block_num * self.block_num
            src_offset = cur_fz * self.block_num
            self.tik_instance.vor(self.block_num, ub_dup[dst_offset],
                                  ub_reducemax_int16[src_offset], ub_dup[cur_m * self.block_num * self.block_num],
                                  self.block_num, 1, 1, 0, 1, 0, 0)
            self.tik_instance.vtranspose(ub_dup[dst_offset], ub_dup[dst_offset])

        sub_range = cur_m * self.block_num * self.block_num // self.repeat_once_size
        sub_dst_stride = cur_m * self.block_num
        sub_src_stride = cur_m * self.block_num
        with self.tik_instance.for_range(0, sub_range) as idx:
            self.tik_instance.vsub(self.repeat_once_size, tensor_input[idx * self.repeat_once_size],
                                   tensor_input[idx * self.repeat_once_size], softmax_ub[idx * self.repeat_once_size],
                                   cur_n, 1, 1, 1, sub_dst_stride, sub_src_stride, 0)

        return tensor_input

    def tiling_batch_m_axis(self, m_size):
        batch_range_value = self.x1_shape[0] * self.x1_shape[1] // self.cur_op_core_num
        if m_size >= 32:
            outer_m_range_value = 2
            inner_m_range_value = self.x1_shape[3] // 4
        else:
            outer_m_range_value = 1
            if m_size > 16:
                inner_m_range_value = self.x1_shape[3] // 2
            elif m_size == 16:
                inner_m_range_value = 4
            else:
                inner_m_range_value = 2

        return batch_range_value, outer_m_range_value, inner_m_range_value

    def first_bmm_move_tensor_b_from_gm_to_l1(self, block_idx, batch_range_value, cur_b_idx):
        first_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.first_k_dim, self.first_n_dim,
                                                                          self.block_num, self.block_num],
                                                      name="first_bmm_tensor_b", scope=self.tik.scope_cbuf)
        first_bmm_tensor_b_offset = block_idx * batch_range_value * self.first_n_dim * self.first_k_dim * \
                                    self.block_num * self.block_num + \
                                    cur_b_idx * self.first_n_dim * self.first_k_dim * self.block_num * self.block_num
        first_bmm_tensor_b_burst = self.first_n_dim * self.block_num
        first_bmm_tensor_b_repeat_times = self.first_k_dim
        first_bmm_tensor_b_src_stride = first_bmm_tensor_b_dst_stride = 0
        self.tik_instance.data_move(first_bmm_tensor_b, self.x2_gm[first_bmm_tensor_b_offset], sid=0,
                                    nburst=first_bmm_tensor_b_repeat_times, burst=first_bmm_tensor_b_burst,
                                    src_stride=first_bmm_tensor_b_src_stride, dst_stride=first_bmm_tensor_b_dst_stride)
        return first_bmm_tensor_b

    def second_bmm_move_tensor_b_from_gm_to_l1(self, block_idx, cur_b_idx):
        second_bmm_tensor_b = self.tik_instance.Tensor(self.matmul_dtype, [self.second_n_dim, self.second_k_dim,
                                                                           self.block_num, self.block_num],
                                                      name="second_bmm_tensor_b", scope=self.tik.scope_cbuf)
        for dma_idx in range(self.second_n_dim):
            second_bmm_tensor_b_src_offset = block_idx * self.batch_outer_num * self.second_n_dim * \
                                             self.second_k_dim * self.block_num * self.block_num + \
                                             cur_b_idx * self.second_n_dim * self.second_k_dim * \
                                             self.block_num * self.block_num + \
                                             dma_idx * self.second_k_dim * self.block_num * self.block_num
            second_bmm_tensor_b_dst_offset = dma_idx * self.block_num * self.block_num
            second_bmm_tensor_b_burst = self.block_num
            second_bmm_tensor_b_repeat_times = self.second_k_dim
            second_bmm_tensor_b_src_stride = 0
            second_bmm_tensor_b_dst_stride = (self.second_n_dim - 1) * self.block_num
            self.tik_instance.data_move(second_bmm_tensor_b[second_bmm_tensor_b_dst_offset],
                                        self.x3_gm[second_bmm_tensor_b_src_offset],
                                        sid=0, nburst=second_bmm_tensor_b_repeat_times,
                                        burst=second_bmm_tensor_b_burst,
                                        src_stride=second_bmm_tensor_b_src_stride,
                                        dst_stride=second_bmm_tensor_b_dst_stride)
        return second_bmm_tensor_b

    def apply_buffer_for_tensor_b_l0_and_move_data_in(self, first_bmm_tensor_b):
        first_bmm_tensor_b_l0b = self.tik_instance.Tensor(self.matmul_dtype,
                                                          [self.first_k_dim, self.first_n_dim,
                                                           self.block_num, self.block_num],
                                                          name="first_bmm_tensor_b_l0b",
                                                          scope=self.tik.scope_cb)
        if self.model_type == Constant.TRAINGING:
            self.tik_instance.load2dv1(first_bmm_tensor_b_l0b, first_bmm_tensor_b[0],
                                       0, self.first_k_dim * self.first_n_dim, 1, 0, False)
        else:
            self.tik_instance.load2dv2(first_bmm_tensor_b_l0b, first_bmm_tensor_b[0],
                                       0, self.first_k_dim * self.first_n_dim, 0, 1, 0, False)
        return first_bmm_tensor_b_l0b

    def first_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                           inner_m_range_value, mul_value):
        """
        compute first bmm once for outer m range.
        """
        first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        with self.tik_instance.for_range(0, inner_m_range_value // self.double_factor) as inner_m_idx:
            inner_m_size = outer_m_range_once_m_size // inner_m_range_value
            inner_k_size, inner_n_size = self.first_k_dim, self.first_n_dim
            l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]

            first_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="first_bmm_tensor_a",
                                                          scope=self.tik.scope_cbuf)
            first_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                  self.block_num * self.block_num],
                                                              name="first_bmm_tensor_a_l0a", scope=self.tik.scope_ca)

            first_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="first_bmm_tensor_c_l0c", scope=self.tik.scope_cc)

            first_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                 self.block_num * self.block_num],
                                                             name="first_bmm_tensor_c_ub", scope=self.tik.scope_ubuf)

            elewise_add1_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                                self.block_num * self.block_num],
                                                             name="elewise_add1_data_ub", scope=self.tik.scope_ubuf)
            elewise_add2_data_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * self.first_n_dim *
                                                                                self.block_num * self.block_num],
                                                             name="elewise_add2_data_ub", scope=self.tik.scope_ubuf)
            first_bmm_compute_buffers = [first_bmm_tensor_a, first_bmm_tensor_c,
                                         first_bmm_tensor_a_l0a, first_bmm_tensor_b_l0b, first_bmm_tensor_c_l0c,
                                         first_bmm_tensor_c_ub, mul_value, elewise_add1_data_ub, elewise_add2_data_ub]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx]
            first_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
            self.mat_mul_compute(first_bmm_compute_buffers, first_bmm_compute_idx, first_bmm_compute_each_layer_size)

            first_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="first_bmm_tensor_a_db",
                                                             scope=self.tik.scope_cbuf)

            first_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                     self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_a_l0a_db",
                                                                 scope=self.tik.scope_ca, start_addr=32768)

            first_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                                 name="first_bmm_tensor_c_l0c_db",
                                                                 scope=self.tik.scope_cc, start_addr=32768)

            first_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                    self.block_num * self.block_num],
                                                                name="first_bmm_tensor_c_ub_db",
                                                                scope=self.tik.scope_ubuf)

            elewise_add1_data_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                   self.block_num * self.block_num],
                                                              name="elewise_add1_data_ub_db", scope=self.tik.scope_ubuf)
            elewise_add2_data_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_n_size *
                                                                                   self.block_num * self.block_num],
                                                              name="elewise_add2_data_ub_db", scope=self.tik.scope_ubuf)
            first_bmm_compute_db_buffers = [first_bmm_tensor_a_db, first_bmm_tensor_c,
                                            first_bmm_tensor_a_l0a_db, first_bmm_tensor_b_l0b,
                                            first_bmm_tensor_c_l0c_db, first_bmm_tensor_c_ub_db, mul_value,
                                            elewise_add1_data_ub_db, elewise_add2_data_ub_db]
            first_bmm_compute_idx = [block_idx, cur_b_idx, cur_om_idx, 2 * inner_m_idx + 1]
            self.mat_mul_compute(first_bmm_compute_db_buffers, first_bmm_compute_idx,
                                 first_bmm_compute_each_layer_size)

    def second_bmm_compute_for_outer_m_once(self, preload_buffers, range_idxs, outer_m_range_once_m_size,
                                            inner_m_range_value):
        """
        second bmm compute for outer m once.
        """
        second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c = preload_buffers
        block_idx, cur_b_idx, cur_om_idx = range_idxs
        if self.model_type == Constant.TRAINGING:
            self.tik_instance.load2dv1(second_bmm_tensor_b_l0b, second_bmm_tensor_b[0],
                                       0, self.first_n_dim * self.second_n_dim, 1, 0, True)
        else:
            self.tik_instance.load2dv2(second_bmm_tensor_b_l0b, second_bmm_tensor_b[0],
                                       0, self.first_n_dim * self.second_n_dim, 0, 1, 0, True)
                                       
        inner_m_size = outer_m_range_once_m_size // inner_m_range_value
        inner_k_size, inner_n_size = self.second_k_dim, self.second_n_dim
        l1a_shape = [inner_k_size, inner_m_size, self.block_num, self.block_num]

        ub_start_addr = 0
        second_bmm_tensor_a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="second_bmm_tensor_a",
                                                       scope=self.tik.scope_cbuf)
        second_bmm_tensor_a_l0a = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                           name="second_bmm_tensor_a_l0a", scope=self.tik.scope_ca)

        second_bmm_tensor_c_l0c = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_tensor_c_l0c", scope=self.tik.scope_cc)

        second_bmm_tensor_c_ub = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                              self.block_num * self.block_num],
                                                          name="second_bmm_tensor_c_ub",
                                                          scope=self.tik.scope_ubuf, start_addr=0)
        ub_start_addr = ub_start_addr + 32768
        second_bmm_mask_ub = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                self.block_num * self.block_num],
                                                      name="second_bmm_mask_ub",
                                                      scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                       self.block_num * self.block_num],
                                                           name="second_bmm_mask_cast_ub",
                                                           scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                name="second_bmm_softmax_reduce_ub",
                                                                scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        # db part tensor
        second_bmm_tensor_a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape, name="second_bmm_tensor_a_db",
                                                          scope=self.tik.scope_cbuf, start_addr=524288)
        second_bmm_tensor_a_l0a_db = self.tik_instance.Tensor(self.matmul_dtype, l1a_shape,
                                                              name="second_bmm_tensor_a_l0a_db",
                                                              scope=self.tik.scope_ca, start_addr=32768)

        second_bmm_tensor_c_l0c_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_n_size *
                                                                          self.block_num * self.block_num],
                                                              name="second_bmm_tensor_c_l0c_db",
                                                              scope=self.tik.scope_cc, start_addr=32768)
        ub_start_addr = 131072

        second_bmm_tensor_c_ub_db = self.tik_instance.Tensor(self.matmul_dtype, [inner_m_size * inner_k_size *
                                                                                 self.block_num * self.block_num],
                                                             name="second_bmm_tensor_c_ub_db",
                                                             scope=self.tik.scope_ubuf, start_addr=ub_start_addr)
        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 2
        second_bmm_mask_ub_db = self.tik_instance.Tensor("uint8", [inner_m_size * inner_k_size *
                                                                   self.block_num * self.block_num],
                                                         name="second_bmm_mask_ub_db",
                                                         scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num
        second_bmm_mask_cast_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * inner_k_size *
                                                                          self.block_num * self.block_num],
                                                              name="second_bmm_mask_cast_ub_db",
                                                              scope=self.tik.scope_ubuf, start_addr=ub_start_addr)

        ub_start_addr = ub_start_addr + inner_m_size * inner_k_size * self.block_num * self.block_num * 4
        second_bmm_softmax_reduce_ub_db = self.tik_instance.Tensor("float32", [inner_m_size * self.block_num],
                                                                   name="second_bmm_softmax_reduce_ub_db",
                                                                   scope=self.tik.scope_ubuf, start_addr=ub_start_addr)
        second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_c,
                                      second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                      second_bmm_softmax_reduce_ub]

        second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_b_l0b,
                                         second_bmm_tensor_a_l0a_db, second_bmm_tensor_c_l0c_db,
                                         second_bmm_tensor_c_ub_db, second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                         second_bmm_softmax_reduce_ub_db]
        second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, 0]
        second_bmm_compute_each_layer_size = [outer_m_range_once_m_size, inner_m_size, inner_k_size, inner_n_size]
        self.mat_mul_second_compute_front(second_bmm_compute_buffers, second_bmm_compute_idxs,
                                          second_bmm_compute_each_layer_size)
        unroll_range = inner_m_range_value - 1
        for cur_m_idx in range(unroll_range):
            second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, cur_m_idx]
            if cur_m_idx % 2 == 0:
                second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_c, second_bmm_tensor_a_l0a,
                                              second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c, second_bmm_mask_ub]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_c_ub_db,
                                                 second_bmm_mask_ub_db, second_bmm_mask_cast_ub_db,
                                                 second_bmm_softmax_reduce_ub_db]
            else:
                second_bmm_compute_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_c,
                                              second_bmm_tensor_a_l0a_db, second_bmm_tensor_b_l0b,
                                              second_bmm_tensor_c_l0c_db, second_bmm_mask_ub_db]

                second_bmm_compute_db_buffers = [second_bmm_tensor_a, second_bmm_tensor_c_ub, second_bmm_mask_ub,
                                                 second_bmm_mask_cast_ub, second_bmm_softmax_reduce_ub]
            self.mat_mul_second_compute_mid(second_bmm_compute_buffers, second_bmm_compute_db_buffers,
                                            second_bmm_compute_idxs, second_bmm_compute_each_layer_size)

        second_bmm_compute_buffers = [second_bmm_tensor_a, second_bmm_tensor_b, second_bmm_tensor_c,
                                      second_bmm_tensor_a_l0a, second_bmm_tensor_b_l0b, second_bmm_tensor_c_l0c,
                                      second_bmm_tensor_c_ub, second_bmm_mask_ub, second_bmm_mask_cast_ub,
                                      second_bmm_softmax_reduce_ub]

        second_bmm_compute_db_buffers = [second_bmm_tensor_a_db, second_bmm_tensor_a_l0a_db,
                                         second_bmm_tensor_c_l0c_db, second_bmm_tensor_c_ub_db]
        second_bmm_compute_idxs = [block_idx, cur_b_idx, cur_om_idx, unroll_range]
        self.mat_mul_second_compute_last(second_bmm_tensor_b_l0b, second_bmm_compute_db_buffers,
                                         second_bmm_compute_idxs, second_bmm_compute_each_layer_size)

    def compute_process(self):
        with self.tik_instance.for_range(0, self.cur_op_core_num, block_num=self.cur_op_core_num) as block_idx:
            batch_range_value, outer_m_range_value, inner_m_range_value = self.tiling_batch_m_axis(self.first_m_dim)
            self.batch_outer_num = batch_range_value
            outer_m_range_once_m_size = self.first_m_dim // outer_m_range_value
            mul_value = self.tik_instance.Scalar("float16", "mul_value", init_value=-1)
            mul_x_ub = self.tik_instance.Tensor(self.matmul_dtype, [self.block_num],
                                                name="mul_x_ub", scope=self.tik.scope_ubuf)
            self.tik_instance.data_move(mul_x_ub, self.mul_gm[0], sid=0, nburst=1, burst=1, src_stride=0, dst_stride=0)
            mul_value.set_as(mul_x_ub[0])
            with self.tik_instance.for_range(0, batch_range_value) as cur_b_idx:
                first_bmm_tensor_b = self.first_bmm_move_tensor_b_from_gm_to_l1(block_idx, batch_range_value, cur_b_idx)
                first_bmm_tensor_c = self.tik_instance.Tensor(self.vector_dtype,
                                                              [self.first_n_dim,
                                                               self.first_m_dim // outer_m_range_value,
                                                               self.block_num, self.block_num],
                                                              name="first_bmm_tensor_c", scope=self.tik.scope_cbuf)
                first_bmm_tensor_b_l0b = self.apply_buffer_for_tensor_b_l0_and_move_data_in(first_bmm_tensor_b)
                with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                    first_preload_buffers = [first_bmm_tensor_b, first_bmm_tensor_c, first_bmm_tensor_b_l0b]
                    range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                    self.first_bmm_compute_for_outer_m_once(first_preload_buffers, range_idxs,
                                                            outer_m_range_once_m_size, inner_m_range_value, mul_value)

                second_bmm_tensor_b = self.second_bmm_move_tensor_b_from_gm_to_l1(block_idx, cur_b_idx)
                with self.tik_instance.for_range(0, outer_m_range_value) as cur_om_idx:
                    second_bmm_tensor_b_l0b = first_bmm_tensor_b_l0b
                    second_bmm_tensor_c = first_bmm_tensor_c
                    range_idxs = [block_idx, cur_b_idx, cur_om_idx]
                    second_bmm_preload_buffers = [second_bmm_tensor_b, second_bmm_tensor_b_l0b, second_bmm_tensor_c]
                    self.second_bmm_compute_for_outer_m_once(second_bmm_preload_buffers, range_idxs,
                                                             outer_m_range_once_m_size, inner_m_range_value)
        if self.structure_swin:
            input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.add1_gm, self.add2_gm,
                             self.mul_gm, self.drop_mask_gm]
        else:
            input_gm_list = [self.x1_gm, self.x2_gm, self.x3_gm, self.add1_gm, self.mul_gm]
        output_gm_list = [self.y_gm]
        if self.model_type == Constant.TRAINGING:
            output_gm_list = [self.y_gm, self.softmax_output_gm]
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=input_gm_list,
                                   outputs=output_gm_list, config={})


# 'pylint: disable=redefined-builtin
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.OPTION_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            (para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_LIST_INT),
                            para_check.KERNEL_NAME)
def swin_attention_score(query, key, value, padding_mask_1, padding_mask_2, scale, drop_mask,
                         swin_attention_score_output, softmax_output,
                         keep_prob=1.0, query_transpose=False, key_transpose=False,
                         bmm_score_transpose_a=False, bmm_score_transpose_b=False,
                         softmax_axes=-1, kernel_name="swin_attention_score"):
    op_init = MatMulSoftmax(query, key, padding_mask_1, padding_mask_2, drop_mask, value, softmax_output,
                            swin_attention_score_output, kernel_name)
    op_init.compute_process()
