# -*- coding: utf-8 -*-
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
from te import tik


def gate_compute(tik_instance, input_x_ub, input_h_ub, input_weight_gm, result_ub, batch_size, input_dims, num_units):
    x_data_ub = tik_instance.Tensor("float32", (batch_size, input_dims), name="data_x_ub",
                                    scope=tik.scope_ubuf)
    h_data_ub = tik_instance.Tensor("float32", (batch_size, num_units), name="data_h_ub",
                                    scope=tik.scope_ubuf)
    x_bias_ub = tik_instance.Tensor("float32", (batch_size, num_units), name="data_b_ub",
                                    scope=tik.scope_ubuf)
    x_bias_ub_fp16 = tik_instance.Tensor("float16", (batch_size, num_units), name="x_bias_ub_fp16",
                                         scope=tik.scope_ubuf)

    tik_instance.data_move(x_data_ub[0], input_x_ub[0], 0, 1, (batch_size * input_dims) // 8, 0, 0)
    tik_instance.data_move(h_data_ub[0], input_h_ub[0], 0, 1, (batch_size * num_units) // 8, 0, 0)

    tik_instance.data_move(x_bias_ub_fp16[0], input_weight_gm[(input_dims + num_units) * num_units], 0, 1,
                           (batch_size * num_units) // 16, 0, 0)
    tik_instance.vconv(64, '', x_bias_ub, x_bias_ub_fp16, 2, 1, 1, 8, 4)

    with tik_instance.new_stmt_scope():
        matmul(tik_instance, x_data_ub, input_weight_gm[0:input_dims * num_units], x_bias_ub, x_bias_ub)

    with tik_instance.new_stmt_scope():
        matmul(tik_instance, h_data_ub,
               input_weight_gm[input_dims * num_units:input_dims * num_units + num_units * num_units],
               x_bias_ub, x_bias_ub)

    tik_instance.data_move(result_ub[0], x_bias_ub[0], 0, 1, (batch_size * num_units) // 8, 0, 0)


def matmul(tik_instance, input_x, input_weight, bias, result):
    """
    general matrix multiply
    :param tik_instance:
    :param input_x:   [1, 256]
    :param input_weight:  [256, 128]
    :param bias: [1, 128]
    :param result:  [1, 128]
    :return:
    """
    mat_m, mat_k = input_x.shape
    _, mat_n = bias.shape

    weight_l1 = tik_instance.Tensor("float16", (mat_k, mat_n), name="weight_l1", scope=tik.scope_cbuf)
    tik_instance.data_move(weight_l1[0], input_weight[0], 0, 1, mat_k * mat_n // 16, 0, 0)

    res_ub = tik_instance.Tensor("float32", (mat_n,), name="res_ub", scope=tik.scope_ubuf)
    bias_ub = tik_instance.Tensor("float32", (mat_n,), name="bias_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(bias_ub[0], bias[0], 0, 1, mat_n // 8, 0, 0)

    src0_ub = tik_instance.Tensor("float32", (mat_k,), name="src0_ub", scope=tik.scope_ubuf)
    tik_instance.data_move(src0_ub[0], input_x[0], 0, 1, mat_k // 8, 0, 0)

    with tik_instance.for_range(0, mat_n) as i:
        src1_ub_fp16 = tik_instance.Tensor("float16", (mat_k,), name="src1_ub_fp16", scope=tik.scope_ubuf)
        src1_ub = tik_instance.Tensor("float32", (mat_k,), name="src1_ub", scope=tik.scope_ubuf)
        dst_ub = tik_instance.Tensor("float32", (mat_k,), name="dst_ub", scope=tik.scope_ubuf)

        tik_instance.data_move(src1_ub_fp16[0], weight_l1[i * mat_k], 0, 1, mat_k // 16, 0, 0)
        tik_instance.vconv(64, '', src1_ub, src1_ub_fp16, mat_k // 64, 1, 1, 8, 4)

        tik_instance.vector_dup(64, dst_ub, 0, mat_k // 64, 1, 8)

        tik_instance.vmla(64, dst_ub, src0_ub, src1_ub, mat_k // 64, 1, 1, 1, 8, 8, 8)
        tik_instance.vcadd(64, dst_ub, dst_ub, mat_k // 64, 1, 1, 8)
        tik_instance.vcadd(mat_k // 64, dst_ub, dst_ub, 1, 1, 1, 8)

        res_ub[i].set_as(dst_ub[0])

    tik_instance.vadd(64, res_ub, res_ub, bias_ub, mat_n // 64, 1, 1, 1, 8, 8, 8)
    tik_instance.data_move(result[0], res_ub[0], 0, 1, mat_n // 8, 0, 0)


def _mmad(tik_instance, loop_k, mmad_m, mmad_k, mat_a_l1, input_x, loop_m, mat_k, mat_a_l0, mat_b_l0, mmad_n, mat_b_l1,
          mat_n, loop_n, mat_c_l0):
    with tik_instance.for_range(0, loop_k) as loop_k:
        # load to l0a
        for ind_m in range(0, mmad_m // 16):
            for ind_k in range(0, mmad_k // 16):
                tik_instance.data_move(mat_a_l1[ind_m * mmad_k * 16 + ind_k * 256],
                                       input_x[loop_m * mmad_m * mat_k + loop_k * mmad_k +
                                               ind_m * mat_k * 16 + ind_k * 16], 0,
                                       nburst=16, burst=1, src_stride=loop_k * mmad_k // 16 - 1, dst_stride=0)
        tik_instance.load2dv1(mat_a_l0[0], mat_a_l1[0], 0, mmad_m * mmad_k // 256, 1, 0)
        # load to l0b
        for ind_k in range(0, mmad_k // 16):
            tik_instance.load2dv1(mat_b_l0[ind_k * mmad_n * 16], mat_b_l1[0],
                                  (loop_k * mmad_k // 16 + ind_k) * mat_n // 16 + loop_n * mmad_n // 16,
                                  mmad_n // 16, 1, 0)
        tik_instance.mmad(mat_c_l0[0], mat_a_l0[0], mat_b_l0[0], mmad_m, mmad_k, mmad_n, True)


def gemm(tik_instance, input_x, input_weight, bias, result):
    """
    general matrix multiply
    :param tik_instance:
    :param input_x:
    :param input_weight:
    :param bias:
    :param result:
    :return:
    """
    mat_m, mat_k = input_x.shape
    _, mat_n = bias.shape

    mmad_m = 16
    mmad_k = 128
    mmad_n = 128

    loop_m = mat_m // mmad_m
    loop_k = mat_k // mmad_k
    loop_n = mat_n // mmad_n
    # FRACTAL Z all matrix data
    mat_a_l1 = tik_instance.Tensor("float16", [mmad_m, mmad_k], name="mat_a_l1", scope=tik.scope_cbuf)  # L1 BUFFER
    # once dma weight
    mat_b_l1 = tik_instance.Tensor("float16", [mat_k, mat_n], name="mat_b_l1", scope=tik.scope_cbuf)  # L1 BUFFER
    tik_instance.data_move(mat_b_l1[0], input_weight[0], 0, 1, mat_k * mat_n // 16, 0, 0)
    # once dma bias
    mat_c_ub = tik_instance.Tensor("float16", [mmad_m, mmad_n], name="data_c_ub", scope=tik.scope_ubuf)
    mat_a_l0 = tik_instance.Tensor("float16", (mmad_m, mmad_k), name="mat_a_l0", scope=tik.scope_ca)  # L0A BUFFER
    mat_b_l0 = tik_instance.Tensor("float16", (mmad_k, mmad_n), name="mat_b_l0", scope=tik.scope_cb)  # L0B BUFFER
    mat_c_l0 = tik_instance.Tensor("float16", (mmad_m, mmad_n), name="mat_c_l0", scope=tik.scope_cc)  # L0C BUFFER
    # split to 4 M*mat_k*mat_n block
    with tik_instance.for_range(0, loop_n) as loop_n:
        with tik_instance.for_range(0, loop_m) as loop_m:
            for ind_n in range(0, mmad_n // 16):
                tik_instance.data_move(mat_c_ub[ind_n * mmad_m // 16 * 256],
                                       bias[loop_m * mmad_m * mat_n + loop_n * mmad_n + ind_n * 16], 0,
                                       nburst=mmad_m // 16 * 16, burst=1, src_stride=loop_n * mmad_n // 16 - 1,
                                       dst_stride=0)
            tik_instance.data_move(mat_c_l0[0], mat_c_ub[0], 0, nburst=1, burst=mmad_m * mmad_n // 256, src_stride=0,
                                   dst_stride=0)
            # mmad
            _mmad(tik_instance, loop_k, mmad_m, mmad_k, mat_a_l1, input_x, loop_m, mat_k, mat_a_l0, mat_b_l0, mmad_n,
                  mat_b_l1, mat_n, loop_n, mat_c_l0)

            tik_instance.data_move(mat_c_ub[0], mat_c_l0[0], 0, nburst=1, burst=mmad_m * mmad_n // 256, src_stride=0,
                                   dst_stride=0)
            with tik_instance.for_range(0, mmad_n // 16) as i:
                tik_instance.data_move(result[loop_m * mmad_m * mat_n + loop_n * mmad_n + i * 16],
                                       mat_c_ub[i * mmad_m * 16], 0, nburst=mmad_m, burst=1, src_stride=0,
                                       dst_stride=loop_n * mmad_n // 16 - 1)
