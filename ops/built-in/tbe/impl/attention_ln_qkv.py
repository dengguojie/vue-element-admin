# Copyright 2020 Huawei Technologies Co., Ltd
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
attention_ln_qkv
"""
from __future__ import absolute_import
import math
from functools import reduce
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from tbe.common.utils import shape_util


class Constant:
    '''
    Constant of attention_ln_qkv big kernels
    '''
    DTYPE_SIZE = {
        'float16': 2,
        'float32': 4
    }
    C0 = 16
    M0 = 16
    N0 = 16
    BLOCK_NUM_32 = 32
    CANDIDATE_TILING_M1 = 12
    CANDIDATE_TILING_M2 = 8
    CANDIDATE_TILING_N = 16
    M_INNER_INDEX = 3
    SQUARE_ROOT = -0.5
    DOUBLE_BUFFER = 2
    FRAC_SIZE = M0 * N0
    MASK_FP16 = 128
    MASK_FP32 = 64
    KERNEL_NUM = 3
    BLOCK_PER_REPEAT = 8
    BLOCK_BYTES = 32
    REPEAT_SIZE_MAX = 255
    FP32_DTYPE = "float32"
    FP16_DTYPE = "float16"
    NUM_FP32_PER_BLOCK = BLOCK_BYTES // DTYPE_SIZE.get(FP32_DTYPE)
    NUM_FP16_PER_BLOCK = BLOCK_BYTES // DTYPE_SIZE.get(FP16_DTYPE)
    FP32_REPEAT_SIZE = BLOCK_PER_REPEAT * NUM_FP32_PER_BLOCK
    FP16_REPEAT_SIZE = BLOCK_PER_REPEAT * NUM_FP16_PER_BLOCK
    FP32_REPEAT_STRIDE = FRAC_SIZE // NUM_FP32_PER_BLOCK
    FP32_BLOCK_STRIDE = C0 // NUM_FP32_PER_BLOCK
    FRAC_REPEAT_NUM = FRAC_SIZE // FP32_REPEAT_SIZE


class AttentionLnQKV:
    '''
    AttentionLnQKV
    '''
    def __init__(self, params):
        self.dtype = params.get("dtype")
        self.x_shape = params.get("input_x_shape")
        self.kernel_shape = params.get("kernel_shape")
        self.gamma_shape = params.get("gamma_shape")
        self.beta_shape = params.get("beta_shape")
        self.bias_flag = params.get("bias_flag")
        self.bias_shape = params.get("bias_shape")
        self.out_shape = params.get("out_shape")
        self.norm_shape = params.get("norm_shape")
        self.k1_shape = self.x_shape[0]
        self.m1_shape = self.x_shape[1]
        self.n1_shape = self.kernel_shape[0]
        self.mean_coeff = (self.k1_shape * Constant.C0) ** (-1)
        self.mv_out_shape = (self.m1_shape * Constant.M0,)
        self.epsilon = params.get("epsilon")
        self.kernel_name = params.get("kernel_name")
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self._tiling_args_compute()
        self._init_gm_tensor(self.tik_instance)
        self._init_ln_tensors(self.tik_instance)


    def attention_ln_qkv_compute(self):
        '''
        attention_ln_qkv_compute
        '''
        tik_instance = self.tik_instance
        one_l0b_ub = tik_instance.Tensor(self.dtype, (self.ln_k_al1, 1, Constant.M0, Constant.C0), name="one_l0b_ub",
            scope=tik.scope_ubuf)
        tik_instance.vector_dup(Constant.MASK_FP16, one_l0b_ub, 1, self.ln_k_al1 * Constant.FRAC_SIZE // \
            Constant.FP16_REPEAT_SIZE, 1, 8)
        tik_instance.data_move(self.one_l0b_bl1, one_l0b_ub, 0, 1, self.ln_k_al1 * Constant.M0, 0, 0)
        with tik_instance.for_range(0, self.block_m * self.block_n, block_num=self.block_m * self.block_n) as blk_idx:
            # do not split reduce_axis when load data to l1
            blk_m_idx = blk_idx % self.block_m
            blk_n_idx = blk_idx // self.block_m
            gamma_ub = tik_instance.Tensor(self.dtype, self.gamma_shape, name="gamma_ub", scope=tik.scope_ubuf)
            beta_ub = tik_instance.Tensor(self.dtype, self.beta_shape, name="beta_ub", scope=tik.scope_ubuf)
            tik_instance.data_move(gamma_ub, self.gamma_gm, 0, 1, self.k1_shape, 0, 0)
            self._vconv(gamma_ub, self.gamma_cast, self.k1_shape * Constant.C0 // Constant.FP32_REPEAT_SIZE, True)
            tik_instance.data_move(beta_ub, self.beta_gm, 0, 1, self.k1_shape, 0, 0)
            self._vconv(beta_ub, self.beta_cast, self.k1_shape * Constant.C0 // Constant.FP32_REPEAT_SIZE, True)
            with tik_instance.for_range(0, self.m_single_core) as m_single_core_idx:
                with tik_instance.for_range(0, self.ln_mal1_times) as mal1_times_idx:
                    ln_m_idx = (blk_m_idx * self.m_single_core + m_single_core_idx) * self.ln_mal1_times + \
                        mal1_times_idx
                    self._ln_compute(tik_instance, ln_m_idx, mal1_times_idx)
                matmul_m_idx = (blk_m_idx * self.m_single_core + m_single_core_idx) * self.matmul_m_al1
                l0c = tik_instance.Tensor(Constant.FP32_DTYPE, (self.matmul_n_l0, self.matmul_m_l0, Constant.M0,
                    Constant.N0), name="l0c", scope=tik.scope_cbuf_out)
                for i in range(Constant.KERNEL_NUM):
                    self._matmul_compute(tik_instance, blk_n_idx, matmul_m_idx, [l0c, self.inputs[1 + i],
                                         self.outputs[1 + i], self.inputs[len(self.inputs) + i - Constant.KERNEL_NUM]])
        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=self.inputs, outputs=self.outputs)


    def _tiling_args_compute(self):
        '''
        tiling args setting
        '''
        # in layer_norm cube, ln_m_al0 is same as ln_n_bl0, which must be set as 1
        self.ln_m_al1 = 1
        self.ln_m_al0 = 1
        self.ln_k_al1 = self.k1_shape
        self.ln_k_al0 = self.k1_shape

        self.block_m = self.core_num
        self.block_n = 1

        # matmul_m_l0 can only be 12 or 8
        m_inner = self.out_shape[Constant.M_INNER_INDEX]
        # when m_inner is factor of 12 or the opposite, choose 12 to be matmul_m_l0
        if m_inner % Constant.CANDIDATE_TILING_M1 == 0 or Constant.CANDIDATE_TILING_M1 % m_inner == 0:
            self.matmul_m_l0 = Constant.CANDIDATE_TILING_M1
        elif (self.m1_shape // self.block_m) % Constant.CANDIDATE_TILING_M2 == 0:
            # 8 is the left choice, make sure matmul_m_l0 should be factor of single core m data
            self.matmul_m_l0 = Constant.CANDIDATE_TILING_M2
        else:
            self.matmul_m_l0 = math.gcd(Constant.CANDIDATE_TILING_M1, Constant.CANDIDATE_TILING_M2)
        self.matmul_m_al1 = self.matmul_m_l0
        self.matmul_n_l0 = Constant.CANDIDATE_TILING_N
        self.matmul_n_l1 = self.matmul_n_l0
        # restrict matmul_k_l0 by L0A_SIZE && matmul_m_l0 / L0B_SIZE && matmul_n_l0
        self.matmul_k_l0 = min(
            tbe_platform.get_soc_spec("L0A_SIZE") // self.matmul_m_l0 // Constant.FRAC_SIZE // \
                Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER,
            tbe_platform.get_soc_spec("L0B_SIZE") // self.matmul_n_l0 // Constant.FRAC_SIZE // \
                Constant.DTYPE_SIZE.get(self.dtype) // Constant.DOUBLE_BUFFER)
        self.matmul_k_al1 = self.matmul_k_l0
        self.matmul_k_bl1 = self.matmul_k_l0
        self.ln_mal1_times = self.matmul_m_al1 // self.ln_m_al1
        self.m_single_core = self.m1_shape // self.block_m // self.matmul_m_al1
        self.n_single_core = self.n1_shape // self.block_n // self.matmul_n_l1


    def _init_gm_tensor(self, tik_instance):
        '''
        init gm tensors
        '''
        # init input_gm tensor
        self.x_gm = tik_instance.Tensor(self.dtype, self.x_shape, name="x_gm",
            scope=tik.scope_gm)
        self.kernel_query_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_query_gm",
            scope=tik.scope_gm)
        self.kernel_key_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_key_gm",
            scope=tik.scope_gm)
        self.kernel_value_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="kernel_value_gm",
            scope=tik.scope_gm)
        self.gamma_gm = tik_instance.Tensor(self.dtype, self.gamma_shape, name="gamma_gm",
            scope=tik.scope_gm)
        self.beta_gm = tik_instance.Tensor(self.dtype, self.beta_shape, name="beta_gm",
            scope=tik.scope_gm)
        self.inputs = [self.x_gm, self.kernel_query_gm, self.kernel_key_gm,
                       self.kernel_value_gm, self.gamma_gm, self.beta_gm]
        self.bias_query_gm = self.bias_key_gm = self.bias_value_gm = None
        if self.bias_flag:
            self.bias_query_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_query_gm",
                scope=tik.scope_gm)
            self.bias_key_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_key_gm",
                scope=tik.scope_gm)
            self.bias_value_gm = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_value_gm",
                scope=tik.scope_gm)
        self.inputs.extend([self.bias_query_gm, self.bias_key_gm, self.bias_value_gm])
        # init output_gm tensor
        self.norm_gm = tik_instance.Tensor(self.dtype, self.norm_shape, name="norm_gm",
            scope=tik.scope_gm)
        self.query_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="query_output_gm",
            scope=tik.scope_gm)
        self.key_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="key_output_gm",
            scope=tik.scope_gm)
        self.value_output_gm = tik_instance.Tensor(self.dtype, self.out_shape, name="value_output_gm",
            scope=tik.scope_gm)
        self.mean_out_gm = tik_instance.Tensor(self.dtype, self.mv_out_shape, name="mean_out_gm",
            scope=tik.scope_gm)
        self.var_out_gm = tik_instance.Tensor(self.dtype, self.mv_out_shape, name="var_out_gm",
            scope=tik.scope_gm)
        self.outputs = [self.norm_gm, self.query_output_gm, self.key_output_gm,
                        self.value_output_gm, self.mean_out_gm, self.var_out_gm]


    def _init_ln_tensors(self, tik_instance):
        '''
        init layer_norm tensors
        '''
        self.mad_shape = (self.ln_m_al0, self.ln_m_al0, Constant.M0, Constant.C0)
        one_l0b_bl1_shape = (self.ln_k_al1, 1, Constant.M0, Constant.C0)
        self.one_l0b_bl1 = tik_instance.Tensor(self.dtype, one_l0b_bl1_shape, name="one_l0b_bl1",
            scope=tik.scope_cbuf)
        self.gamma_cast = tik_instance.Tensor(Constant.FP32_DTYPE, self.gamma_shape, name="gamma_cast",
            scope=tik.scope_ubuf)
        self.beta_cast = tik_instance.Tensor(Constant.FP32_DTYPE, self.beta_shape, name="beta_cast",
            scope=tik.scope_ubuf)
        res_l1_shape = (self.ln_k_al1, self.matmul_m_al1, Constant.M0, Constant.C0)
        self.ln_res_l1 = tik_instance.Tensor(self.dtype, res_l1_shape, name="ln_res_l1", scope=tik.scope_cbuf)
        self.x_l1_shape = (self.ln_k_al1, self.ln_m_al1, Constant.M0, Constant.C0)
        self.x_l1 = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="x_l1", scope=tik.scope_cbuf)
        x_l0a_shape = (self.ln_m_al0, self.ln_k_al0, Constant.M0, Constant.C0)
        self.x_l0a = tik_instance.Tensor(self.dtype, x_l0a_shape, name="x_l0a", scope=tik.scope_ca)


    def _ln_compute(self, tik_instance, ln_m_idx, mal1_times_idx):
        '''
        ln_compute
        '''
        # Tik do not support set_2d, the ones tensor should be dumped from UB
        tik_instance.data_move(self.x_l1, self.x_gm[ln_m_idx * self.ln_m_al1 * Constant.M0 * Constant.C0:],
            0, self.ln_k_al1, Constant.M0, (self.m1_shape - 1) * Constant.M0, 0)
        x_ub = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="x_ub", scope=tik.scope_ubuf)
        cast_x_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.x_l1_shape, name="cast_x_ub", scope=tik.scope_ubuf)
        tik_instance.data_move(x_ub, self.x_l1, 0, 1, self.ln_k_al1 * Constant.M0, 0, 0)
        self._vconv(x_ub, cast_x_ub, self.ln_k_al1 * Constant.C0 * Constant.M0 // Constant.FP32_REPEAT_SIZE, True)
        # process mean
        x_sum_ub = self._mad_compute(tik_instance, 1, is_mean_mad=True)
        # process variance
        xx_sum_ub = self._mad_compute(tik_instance, self.ln_m_al0)
        squared_mean_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="squared_mean_ub",
            scope=tik.scope_ubuf)
        var_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="var_ub", scope=tik.scope_ubuf)
        # mean^2
        tik_instance.vmul(Constant.MASK_FP32, squared_mean_ub, x_sum_ub, x_sum_ub, Constant.FRAC_REPEAT_NUM,
            1, 1, 1, 8, 8, 8)
        mean_cast_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="mean_cast_ub", scope=tik.scope_ubuf)
        mean_trans_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="mean_trans_ub", scope=tik.scope_ubuf)
        # move x_sum_ub to mean_gm
        self._nz_to_nd_out(tik_instance, [x_sum_ub, mean_cast_ub, mean_trans_ub], self.mean_out_gm, ln_m_idx)
        # variance is x^2 - mean^2
        tik_instance.vsub(Constant.MASK_FP32, var_ub, xx_sum_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM,
            1, 1, 1, 8, 8, 8)
        # variance + epsilon
        tik_instance.vadds(Constant.MASK_FP32, squared_mean_ub, var_ub, self.epsilon, Constant.FRAC_REPEAT_NUM,
            1, 1, 8, 8)
        var_cast_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="var_cast_ub", scope=tik.scope_ubuf)
        var_trans_ub = tik_instance.Tensor(self.dtype, self.mad_shape, name="var_trans_ub", scope=tik.scope_ubuf)
        # move xx_sum_ub to variance_gm
        self._nz_to_nd_out(tik_instance, [var_ub, var_cast_ub, var_trans_ub], self.var_out_gm, ln_m_idx)
        # rsqrt of variance + epsilon
        tik_instance.vln(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        tik_instance.vmuls(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.SQUARE_ROOT,
            Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        tik_instance.vexp(Constant.MASK_FP32, squared_mean_ub, squared_mean_ub, Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        self._ln_scale(tik_instance, [x_sum_ub, squared_mean_ub, cast_x_ub], [ln_m_idx, mal1_times_idx])


    def _ln_scale(self, tik_instance, ub_tensor_list, idx_list):
        '''
        substract mean && variance division
        '''
        x_sum_ub, squared_mean_ub, cast_x_ub = ub_tensor_list
        ln_m_idx, mal1_times_idx = idx_list
        with tik_instance.for_range(0, Constant.M0 * Constant.C0 // Constant.FP32_REPEAT_SIZE) as sub_mul_idx:
            vsub_offset = Constant.FP32_REPEAT_SIZE * sub_mul_idx
            # x - mean
            tik_instance.vsub(Constant.MASK_FP32, cast_x_ub[vsub_offset:], cast_x_ub[vsub_offset:],
                x_sum_ub[vsub_offset:], self.ln_k_al1, 1, 1, 1, Constant.FP32_REPEAT_STRIDE,
                Constant.FP32_REPEAT_STRIDE, 0)
        with tik_instance.for_range(0, Constant.M0 * Constant.C0 // Constant.FP32_REPEAT_SIZE) as sub_mul_idx:
            vsub_offset = Constant.FP32_REPEAT_SIZE * sub_mul_idx
            # norm is x - mean divides sqrt of variance + epsilon
            tik_instance.vmul(Constant.MASK_FP32, cast_x_ub[vsub_offset:], cast_x_ub[vsub_offset:],
                squared_mean_ub[vsub_offset:], self.ln_k_al1, 1, 1, 1, Constant.FP32_REPEAT_STRIDE,
                Constant.FP32_REPEAT_STRIDE, 0)
        with tik_instance.for_range(0, Constant.FP32_BLOCK_STRIDE) as outer_idx:
            with tik_instance.for_range(0, Constant.FRAC_SIZE // Constant.C0 // 8) as inner_idx:
                # gamma muls norm
                tik_instance.vmul(Constant.MASK_FP32, cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                    cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:], self.gamma_cast[8 * outer_idx:],
                    self.ln_k_al1, Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                    Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, Constant.FP32_BLOCK_STRIDE)
        with tik_instance.for_range(0, Constant.FP32_BLOCK_STRIDE) as outer_idx:
            with tik_instance.for_range(0, Constant.FRAC_SIZE // Constant.C0 // 8) as inner_idx:
                # gamma muls norm add beta
                tik_instance.vadd(Constant.MASK_FP32, cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:],
                    cast_x_ub[8 * Constant.C0 * inner_idx + 8 * outer_idx:], self.beta_cast[8 * outer_idx:],
                    self.ln_k_al1, Constant.FP32_BLOCK_STRIDE, Constant.FP32_BLOCK_STRIDE, 0,
                    Constant.FP32_REPEAT_STRIDE, Constant.FP32_REPEAT_STRIDE, Constant.FP32_BLOCK_STRIDE)
        cast_ln_res = tik_instance.Tensor(self.dtype, self.x_l1_shape, name="cast_ln_res", scope=tik.scope_ubuf)
        self._vconv(cast_x_ub, cast_ln_res, self.ln_k_al1 * Constant.C0 * Constant.M0 // Constant.FP32_REPEAT_SIZE,
            False)
        # use cast_ln_res as x_input of matmul_qkv
        tik_instance.data_move(self.ln_res_l1[mal1_times_idx * Constant.M0 * Constant.C0:], cast_ln_res, 0,
            self.ln_k_al1, Constant.M0, 0, (self.matmul_m_al1 - self.ln_m_al1) * Constant.M0)
        tik_instance.data_move(self.norm_gm[ln_m_idx * Constant.M0 * Constant.C0:], cast_ln_res, 0, self.ln_k_al1,
            Constant.M0, 0, (self.m1_shape - self.ln_m_al1) * Constant.M0)


    def _mad_compute(self, tik_instance, mad_n, is_mean_mad=False):
        '''
        ln_mad_compute
        '''
        l0b_tensor_name = "x_l0b"
        ub_tensor_name = "xx_sum_ub"
        if is_mean_mad:
            l0b_tensor_name = "one_l0b"
            ub_tensor_name = "x_sum_ub"
        x_sum_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name=ub_tensor_name, scope=tik.scope_ubuf)
        dst_l0c = tik_instance.Tensor(Constant.FP32_DTYPE, self.mad_shape, name="dst_l0c", scope=tik.scope_cc)
        with tik_instance.for_range(0, self.ln_k_al1 // self.ln_k_al0) as kl1_factor_idx:
            l0b_shape = (self.ln_k_al0, self.ln_m_al0, Constant.M0, Constant.C0)
            one_l0b = tik_instance.Tensor(self.dtype, l0b_shape, name=l0b_tensor_name, scope=tik.scope_cb)
            if is_mean_mad:
                # in mean process, the calculation is sum(x)
                # al0 process
                with tik_instance.for_range(0, self.ln_m_al0) as mal0_idx:
                    self._load_2d(self.x_l0a, self.x_l1[mal0_idx * Constant.M0 * Constant.C0:], [0, self.ln_k_al0,
                        self.ln_m_al0, 0, False])
                # bl0 process
                self._load_2d(one_l0b, self.one_l0b_bl1, [0, self.ln_k_al0 * mad_n, 1, 0, False])
            else:
                # in variance process, the al0 can be reused; the calculation is (1,x) * (x,1) = x^2
                # bl0 process
                self._load_2d(one_l0b, self.x_l1, [0, self.ln_k_al0 * mad_n, 1, 0, False])
            # l0c process
            with tik_instance.if_scope(kl1_factor_idx == 0):
                tik_instance.mmad(dst_l0c, self.x_l0a, one_l0b, self.ln_m_al0 * Constant.M0,
                    self.ln_k_al0 * Constant.C0, mad_n * Constant.C0, 0)
            with tik_instance.else_scope():
                tik_instance.mmad(dst_l0c, self.x_l0a, one_l0b, self.ln_m_al0 * Constant.M0,
                    self.ln_k_al0 * Constant.C0, mad_n * Constant.C0, 1)
        tik_instance.data_move(x_sum_ub, dst_l0c, 0, 1, self.ln_m_al0, 0, 0)
        if not is_mean_mad:
            # use diagonal element fill the row to remove invalid entry in fractal_matrix
            with tik_instance.for_range(0, Constant.M0) as brc_idx:
                var_scalar = tik_instance.Scalar(Constant.FP32_DTYPE)
                var_scalar.set_as(x_sum_ub[0, 0, (brc_idx * (Constant.C0 + 1)) // Constant.C0,
                    (brc_idx * (Constant.C0 + 1)) % Constant.C0])
                # set vector mask as Constant.C0 to avoid vector_dup erase next value
                tik_instance.vector_dup(Constant.C0, x_sum_ub[brc_idx * Constant.C0:], var_scalar, 1, 1, 0)
        tik_instance.vmuls(Constant.MASK_FP32, x_sum_ub, x_sum_ub, self.mean_coeff, self.ln_m_al0 * \
            Constant.FRAC_REPEAT_NUM, 1, 1, 8, 8)
        return x_sum_ub


    def _nz_to_nd_out(self, tik_instance, ub_tensor_list, out_tensor, idx):
        '''
        data move n1mn0 to mn
        '''
        if self.core_num == Constant.BLOCK_NUM_32:
            src_ub, cast_ub, trans_ub = ub_tensor_list
            self._vconv(src_ub, cast_ub, Constant.FRAC_REPEAT_NUM, False)
            tik_instance.vtranspose(trans_ub, cast_ub)
            # after transpose, output is the first row
            tik_instance.data_move(out_tensor[idx * Constant.M0:], trans_ub, 0, self.ln_m_al0, 1, 0, 0)


    def _matmul_compute(self, tik_instance, blk_n_idx, matmul_m_idx, matmul_tensor_list):
        '''
        matmul qkv compute
        '''
        l0c, kernel_gm, out_gm, bias_gm = matmul_tensor_list
        with tik_instance.for_range(0, self.n_single_core) as n_single_core_idx:
            matmul_n_idx = (blk_n_idx * self.n_single_core + n_single_core_idx) * self.matmul_n_l1
            if self.bias_flag:
                bias_ub = tik_instance.Tensor(self.dtype, self.bias_shape, name="bias_ub",
                    scope=tik.scope_ubuf)
                cast_bias_ub = tik_instance.Tensor(Constant.FP32_DTYPE, self.bias_shape, name="cast_bias_ub",
                    scope=tik.scope_ubuf)
                tik_instance.data_move(bias_ub, bias_gm[matmul_n_idx * Constant.N0:], 0, 1, self.matmul_n_l0, 0, 0)
                self._vconv(bias_ub, cast_bias_ub, self.matmul_n_l0 * Constant.N0 // Constant.FP32_REPEAT_SIZE, True)
                with tik_instance.for_range(0, self.matmul_m_l0) as brc_idx:
                    tik_instance.broadcast_ub_to_l0c(l0c[brc_idx * Constant.FRAC_SIZE:], cast_bias_ub,
                        self.matmul_n_l0, 1, 0, self.matmul_m_l0 - 1)
            with tik_instance.for_range(0, self.k1_shape // self.matmul_k_l0 // Constant.DOUBLE_BUFFER) as kl1_idx:
                bl1_src_offset = (blk_n_idx * self.n_single_core + n_single_core_idx) * self.matmul_n_l1 * \
                    self.k1_shape * Constant.C0 * Constant.N0 + Constant.DOUBLE_BUFFER * kl1_idx * \
                    self.matmul_k_bl1 * Constant.C0 * Constant.N0
                # ping
                self._matmul_l0c_compute(tik_instance, kernel_gm, l0c, [bl1_src_offset, kl1_idx, 0])
                # pong
                bl1_src_offset += self.matmul_k_bl1 * Constant.C0 * Constant.N0
                self._matmul_l0c_compute(tik_instance, kernel_gm, l0c, [bl1_src_offset, kl1_idx, 1])
            # tensor_mov
            c_ub = tik_instance.Tensor(self.dtype, (self.matmul_n_l0, self.matmul_m_l0, Constant.M0, Constant.N0),
                name="c_ub", scope=tik.scope_ubuf)
            tik_instance.tensor_mov(c_ub, l0c, 'm', 1, self.matmul_n_l0 * self.matmul_m_l0, 0, 0)
            m_inner = self.out_shape[Constant.M_INNER_INDEX]
            out_offset = (matmul_m_idx % m_inner) * Constant.FRAC_SIZE + matmul_n_idx * m_inner * Constant.FRAC_SIZE + \
                matmul_m_idx // m_inner * self.n1_shape * m_inner * Constant.FRAC_SIZE
            if m_inner >= self.matmul_m_l0:
                tik_instance.data_move(out_gm[out_offset:], c_ub, 0, self.matmul_n_l0, self.matmul_m_l0 * Constant.M0,
                    0, (m_inner - self.matmul_m_l0) * Constant.M0)
            else:
                with tik_instance.for_range(0, self.matmul_m_l0 // m_inner) as m_inner_idx:
                    out_offset += m_inner_idx * self.n1_shape * m_inner * Constant.FRAC_SIZE
                    tik_instance.data_move(out_gm[out_offset:], c_ub[m_inner_idx * m_inner * Constant.FRAC_SIZE], 0,
                        self.matmul_n_l0, m_inner * Constant.M0, (self.matmul_m_l0 - m_inner) * Constant.M0, 0)


    def _matmul_l0c_compute(self, tik_instance, kernel_gm, l0c, ping_pong_params):
        '''
        matmul_l0c_compute
        '''
        bl1_src_offset, kl1_factor_idx, ping_pong = ping_pong_params
        ping_pong_suffix = "ping" if ping_pong == 0 else "pong"
        # bl1 process
        bl1 = tik_instance.Tensor(self.dtype, (self.matmul_n_l1, self.matmul_k_bl1, Constant.C0, Constant.N0),
            name="bl1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        tik_instance.data_move(bl1, kernel_gm[bl1_src_offset:], 0, self.matmul_n_l0, self.matmul_k_bl1 * Constant.C0,
            (self.k1_shape - self.matmul_k_bl1) * Constant.C0, 0)
        with tik_instance.for_range(0, self.matmul_m_al1 // self.matmul_m_l0):
            # al0 process
            al0 = tik_instance.Tensor(self.dtype, (self.matmul_m_l0, self.matmul_k_l0, Constant.M0, Constant.C0),
                name="al0_" + ping_pong_suffix, scope=tik.scope_ca)
            with tik_instance.for_range(0, self.matmul_m_l0) as mal0_idx:
                al1_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_al1 * \
                    self.matmul_m_al1 * Constant.M0 * Constant.C0 + mal0_idx * Constant.M0 * Constant.C0
                self._load_2d(al0[mal0_idx * self.matmul_k_l0 * Constant.M0 * Constant.C0:],
                    self.ln_res_l1[al1_offset:], [0, self.matmul_k_l0, self.matmul_m_l0, 0, False])
            with tik_instance.for_range(0, self.matmul_n_l1 // self.matmul_n_l0):
                # bl0 process
                bl0 = tik_instance.Tensor(self.dtype, (self.matmul_k_l0, self.matmul_n_l0, Constant.N0, Constant.C0),
                    name="bl0_" + ping_pong_suffix, scope=tik.scope_cb)
                with tik_instance.for_range(0, self.matmul_k_l0) as kl0_idx:
                    self._load_2d(bl0[kl0_idx * self.matmul_n_l0 * Constant.N0 * Constant.C0:], bl1[kl0_idx * \
                        Constant.C0 * Constant.N0:], [0, self.matmul_n_l0, self.matmul_k_l0, 0, True])
                # l0c process
                mad_tensors = [al0, bl0, l0c]
                self._matmul_l0c_process(tik_instance, ping_pong, kl1_factor_idx, mad_tensors)


    def _matmul_l0c_process(self, tik_instance, ping_pong, kl1_factor_idx, mad_tensors):
        '''
        matmul l0c_process
        '''
        al0, bl0, l0c = mad_tensors
        if ping_pong == 0:
            if self.bias_flag:
                tik_instance.mmad(l0c, al0, bl0, self.matmul_m_l0 * Constant.M0, self.matmul_k_l0 * Constant.C0,
                    self.matmul_n_l0 * Constant.N0, 1)
            else:
                with tik_instance.if_scope(kl1_factor_idx == 0):
                    tik_instance.mmad(l0c, al0, bl0, self.matmul_m_l0 * Constant.M0, self.matmul_k_l0 * Constant.C0,
                        self.matmul_n_l0 * Constant.N0, 0)
                with tik_instance.else_scope():
                    tik_instance.mmad(l0c, al0, bl0, self.matmul_m_l0 * Constant.M0, self.matmul_k_l0 * Constant.C0,
                        self.matmul_n_l0 * Constant.N0, 1)
        else:
            tik_instance.mmad(l0c, al0, bl0, self.matmul_m_l0 * Constant.M0, self.matmul_k_l0 * Constant.C0,
                self.matmul_n_l0 * Constant.N0, 1)


    def _load_2d(self, src, dst, instr_params):
        '''
        load_2d instr is different in Ascend910 and Ascend710
        '''
        tik_instance = self.tik_instance
        start_index, repeat, repeat_stride, sid, is_transpose = instr_params
        if tbe_platform.api_check_support("tik.load2dv2"):
            tik_instance.load2dv2(src, dst, start_index, repeat, 0, repeat_stride, sid, is_transpose)
        elif tbe_platform.api_check_support("tik.load2dv1"):
            tik_instance.load2dv1(src, dst, start_index, repeat, repeat_stride, sid, is_transpose)
        else:
            error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                       "load2d instr unsupported.")


    def _vconv(self, src_tensor, dst_tensor, vconv_repeat_size, fp16_to_fp32):
        '''
        vconv repeat size may exceeds 255, multi vconv instrs may needed
        '''
        tik_instance = self.tik_instance
        stride_params = [1, 1, 8, 4]
        if not fp16_to_fp32:
            stride_params = [1, 1, 4, 8]
        if vconv_repeat_size <= Constant.REPEAT_SIZE_MAX:
            tik_instance.vconv(Constant.MASK_FP32, "", dst_tensor, src_tensor, vconv_repeat_size, *stride_params)
        else:
            num_loops = vconv_repeat_size // Constant.REPEAT_SIZE_MAX
            for i in range(num_loops):
                offset = i * Constant.FP32_REPEAT_SIZE * Constant.REPEAT_SIZE_MAX
                tik_instance.vconv(Constant.MASK_FP32, "", dst_tensor[offset:], src_tensor[offset:],
                    Constant.REPEAT_SIZE_MAX, *stride_params)
            offset = num_loops * Constant.FP32_REPEAT_SIZE * Constant.REPEAT_SIZE_MAX
            repeat_size = vconv_repeat_size - num_loops * Constant.REPEAT_SIZE_MAX
            tik_instance.vconv(Constant.MASK_FP32, "", dst_tensor[offset:], src_tensor[offset:], repeat_size,
                *stride_params)


def _check_shape_and_dtype(x, kernels, outputs):
    '''
    shape and dtype check of attention_ln_qkv
    '''
    kernel_query, kernel_key, kernel_value = kernels
    query_output, key_output, value_output, mean, variance = outputs
    input_x_shape = shape_util.shape_to_list(x.get("shape"))
    input_x_ori_shape = shape_util.shape_to_list(x.get("ori_shape"))
    kernel_query_shape = shape_util.shape_to_list(kernel_query.get("shape"))
    kernel_key_shape = shape_util.shape_to_list(kernel_key.get("shape"))
    kernel_value_shape = shape_util.shape_to_list(kernel_value.get("shape"))
    query_out_shape = shape_util.shape_to_list(query_output.get("shape"))
    key_out_shape = shape_util.shape_to_list(key_output.get("shape"))
    value_out_shape = shape_util.shape_to_list(value_output.get("shape"))
    mean_shape = shape_util.shape_to_list(mean.get("shape"))
    var_shape = shape_util.shape_to_list(variance.get("shape"))
    k1_shape = input_x_shape[0]
    data_type = x.get("dtype")
    if data_type != Constant.FP16_DTYPE:
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "the only supported dtype is fp16.")
    # restrict k_shape with L0A_SIZE since layer_norm cube only support load k once
    if k1_shape > tbe_platform.get_soc_spec("L0A_SIZE") // (Constant.C0 * Constant.M0 *
        Constant.DTYPE_SIZE.get(data_type)):
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "k1_shape is too large to load once in layer_norm calculation.")
    if not (kernel_query_shape == kernel_key_shape and kernel_key_shape == kernel_value_shape):
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "kernel_shape is inconsistant for matmul_qkv.")
    if not (query_out_shape == key_out_shape and key_out_shape == value_out_shape):
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "output_shape is inconsistant for matmul_qkv.")
    if tbe_platform.get_soc_spec(tbe_platform.CORE_NUM) == Constant.BLOCK_NUM_32:
        if mean_shape[0] != input_x_ori_shape[0] or var_shape[0] != input_x_ori_shape[0]:
            error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                       "invalid mean_out_shape/variance_out_shape.")
    input_x_format = x.get("format").upper()
    kernel_format = kernel_query.get("format").upper()
    if input_x_format != "FRACTAL_NZ" or kernel_format != "FRACTAL_NZ":
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                   "only support FRACTAL_NZ format for input_x and kernel.")


@register_operator("attention_ln_qkv")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.OPTION_INPUT, para_check.OPTION_INPUT, para_check.OPTION_INPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_FLOAT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def attention_ln_qkv(x, kernel_query, kernel_key, kernel_value, gamma, beta, bias_query, bias_key, bias_value,
                     norm, query_output, key_output, value_output, mean, variance, epsilon=1e-7,
                     trans_a=False, trans_b=False, kernel_name="attention_ln_qkv"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16
    kernel_query: dict
        shape and dtype of input kernel_query, only support float16
    kernel_key: dict
        shape and dtype of input kernel_key, only support float16
    kernel_value: dict
        shape and dtype of input kernel_value, only support float16
    gamma: dict
        shape and dtype of input gamma, only support float16
    beta: dict
        shape and dtype of input beta, only support float16
    bias_query: dict
        shape and dtype of input bias_query, only support float16
    bias_key: dict
        shape and dtype of input bias_key, only support float16
    bias_value: dict
        shape and dtype of input bias_value, only support float16
    norm: dict
        shape and dtype of output, only support float16
    query_output: dict
        shape and dtype of output, only support float16
    key_output: dict
        shape and dtype of output, only support float16
    value_output: dict
        shape and dtype of output, only support float16
    mean: dict
        shape and dtype of output, only support float16
    variance: dict
        shape and dtype of output, only support float16
    epsilon: float
        Minimum positive number greater than 0
    trans_a: bool
        If True, shape_a == transposed before multiplication
    trans_b: bool
        If True, the shape in input_x2 must be transposed before multiplication
    kernel_name: str
        cce kernel name, default value is "attention_ln_qkv"

    Returns
    -------
    None
    """
    input_x_shape = shape_util.shape_to_list(x.get("shape"))
    kernel_shape = shape_util.shape_to_list(kernel_query.get("shape"))
    gamma_shape = shape_util.shape_to_list(gamma.get("shape"))
    beta_shape = shape_util.shape_to_list(beta.get("shape"))
    norm_shape = shape_util.shape_to_list(norm.get("shape"))
    out_shape = shape_util.shape_to_list(query_output.get("shape"))
    # check bias
    if bias_query and bias_key and bias_value:
        bias_flag = True
        bias_shape = shape_util.shape_to_list(bias_query.get("shape"))
        bias_shape_real = (reduce(lambda x, y: x * y, list(bias_shape)),)
    elif not bias_query and not bias_key and not bias_value:
        bias_shape_real = ()
        bias_flag = False
    else:
        error_manager_cube.raise_err_specific_user("attention_ln_qkv",
                                                    "bias_flag is inconsistant for matmul_qkv.")

    kernels = [kernel_query, kernel_key, kernel_value]
    outputs = [query_output, key_output, value_output, mean, variance]
    _check_shape_and_dtype(x, kernels, outputs)
    params = {
        "dtype": x.get("dtype"),
        "input_x_shape": input_x_shape,
        "kernel_shape": kernel_shape,
        "gamma_shape": gamma_shape,
        "beta_shape": beta_shape,
        "out_shape": out_shape,
        "bias_flag": bias_flag,
        "bias_shape": bias_shape_real,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "norm_shape": norm_shape,
        "epsilon": epsilon,
        "kernel_name": kernel_name
    }
    obj = AttentionLnQKV(params)
    obj.attention_ln_qkv_compute()
