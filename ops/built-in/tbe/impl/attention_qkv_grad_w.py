# Copyright 2022 Huawei Technologies Co., Ltd
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
attention_qkv_grad_w
"""
from impl.util.attention_qkv_util import Constant
from impl.util.attention_qkv_util import vconv
from impl.util.attention_qkv_util import matmul_l0c_process
from impl.util.attention_qkv_util import check_equal_shape
from impl.util.attention_qkv_util import check_dtype
from impl.util.attention_qkv_util import check_format
from impl.util.attention_qkv_util import check_trans_flag
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import error_manager_cube
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from tbe.common.utils import shape_util


class AttentionQKVGradW:
    '''
    AttentionQKVGradW
    '''
    def __init__(self, params):
        self.dtype = params.get("dtype")
        self.ln_input_shape = params.get("ln_input_shape")
        self.kernel_shape = params.get("kernel_shape")
        self.dw_out_shape = params.get("dw_out_shape")
        self.bias_out_shape = params.get("bias_out_shape")
        self.m1_shape = self.ln_input_shape[0]
        self.k1_shape = self.ln_input_shape[1]
        self.n1_shape = self.kernel_shape[0]
        self.kernel_name = params.get("kernel_name")
        self.tik_instance = tik.Tik()
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self._init_gm_tensor(self.tik_instance)
        self._tiling_args_compute()


    def attention_qkv_gradw_compute(self):
        '''
        attention_qkv_gradw_compute
        '''
        tik_instance = self.tik_instance
        m_single_core = self.m1_shape // self.block_m // self.matmul_m_l1
        with tik_instance.for_range(0, self.block_m * self.block_n, block_num=self.block_m * self.block_n) as blk_idx:
            blk_m_idx = blk_idx % self.block_m
            blk_n_idx = blk_idx // self.block_m
            with tik_instance.for_range(0, m_single_core) as m_single_core_idx:
                matmul_m_idx = (blk_m_idx * m_single_core + m_single_core_idx) * self.matmul_m_l1
                cl0_shape = (Constant.KERNEL_NUM * self.matmul_n_l0, self.matmul_m_l0, Constant.M0, Constant.N0)
                cl0_bias_shape = (Constant.KERNEL_NUM * self.matmul_n_l0, 1, Constant.M0, Constant.N0)
                l0c = tik_instance.Tensor(Constant.FP32_DTYPE, cl0_shape, name="l0c", scope=tik.scope_cbuf_out)
                l0c_bias = tik_instance.Tensor(Constant.FP32_DTYPE, cl0_bias_shape, name="l0c_bias",
                    scope=tik.scope_cbuf_out)
                tensor_list = [self.inputs[0], self.inputs[1:], self.outputs[:Constant.KERNEL_NUM],
                                  self.outputs[Constant.KERNEL_NUM:], l0c, l0c_bias]
                self._matmul_compute(tik_instance, blk_n_idx, matmul_m_idx, tensor_list)
        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=self.inputs, outputs=self.outputs)


    def _tiling_args_compute(self):
        '''
        tiling args setting
        '''
        self.matmul_m_l0 = Constant.CANDIDATE_TILING_N
        self.matmul_m_l1 = self.matmul_m_l0

        self.matmul_n_l0 = Constant.CANDIDATE_TILING_K
        self.matmul_n_l1 = self.matmul_n_l0
        self.matmul_k_l0 = Constant.CANDIDATE_TILING_K
        self.matmul_k_al1 = self.matmul_k_l0
        self.matmul_k_bl1 = self.matmul_k_l0

        self.block_m = Constant.CANDIDATE_BLOCK_M
        self.block_n = self.core_num // Constant.CANDIDATE_BLOCK_M


    def _init_gm_tensor(self, tik_instance):
        '''
        init gm tensors
        '''
        self.ln_input_gm = tik_instance.Tensor(self.dtype, self.ln_input_shape, name="ln_input_gm",
            scope=tik.scope_gm)
        self.query_dx_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="query_dx_gm",
            scope=tik.scope_gm)
        self.key_dw_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="key_dw_gm",
            scope=tik.scope_gm)
        self.value_dw_gm = tik_instance.Tensor(self.dtype, self.kernel_shape, name="value_dw_gm",
            scope=tik.scope_gm)
        self.inputs = [self.ln_input_gm, self.query_dx_gm, self.key_dw_gm, self.value_dw_gm]

        self.dw_query_gm = tik_instance.Tensor(self.dtype, self.dw_out_shape, name="dw_query_gm",
            scope=tik.scope_gm)
        self.dw_key_gm = tik_instance.Tensor(self.dtype, self.dw_out_shape, name="dw_key_gm",
            scope=tik.scope_gm)
        self.dw_value_gm = tik_instance.Tensor(self.dtype, self.dw_out_shape, name="dw_value_gm",
            scope=tik.scope_gm)
        self.dbias_query_gm = tik_instance.Tensor(self.dtype, self.bias_out_shape, name="dbias_query_gm",
            scope=tik.scope_gm)
        self.dbias_key_gm = tik_instance.Tensor(self.dtype, self.bias_out_shape, name="dbias_key_gm",
            scope=tik.scope_gm)
        self.dbias_value_gm = tik_instance.Tensor(self.dtype, self.bias_out_shape, name="dbias_value_gm",
            scope=tik.scope_gm)
        self.outputs = [self.dw_query_gm, self.dw_key_gm, self.dw_value_gm,
                        self.dbias_query_gm, self.dbias_key_gm, self.dbias_value_gm]


    def _matmul_compute(self, tik_instance, blk_n_idx, matmul_m_idx, tensor_list):
        '''
        matmul_dw_qkv compute
        '''
        x_gm, kernel_gms, out_gms, bias_out_gms, l0c, l0c_bias = tensor_list
        n_single_core = self.n1_shape // self.block_n // self.matmul_n_l1
        with tik_instance.for_range(0, n_single_core) as n_single_core_idx:
            matmul_n_idx = (blk_n_idx * n_single_core + n_single_core_idx) * self.matmul_n_l1
            # aub process
            aub_shape = (Constant.DOUBLE_BUFFER, self.matmul_k_al1, Constant.C0, Constant.M0)
            a_ub = tik_instance.Tensor(self.dtype, aub_shape, name="a_ub", scope=tik.scope_ubuf)
            tensor_list.append(a_ub)
            tik_instance.vector_dup(Constant.MASK_FP16, a_ub, 1, Constant.DOUBLE_BUFFER * self.matmul_k_al1 * \
                Constant.FRAC_SIZE // Constant.FP16_REPEAT_SIZE, 1, Constant.BLOCK_PER_REPEAT)
            with tik_instance.for_range(0, self.k1_shape // self.matmul_k_al1 // \
                Constant.DOUBLE_BUFFER) as kl1_factor_idx:
                idx_list = [matmul_m_idx, matmul_n_idx, kl1_factor_idx]
                # ping-pong
                for i in range(Constant.DOUBLE_BUFFER):
                    self._matmul_l0c_compute(tik_instance, tensor_list, idx_list, ping_pong=i)
            # dw_out process
            cub_shape = (Constant.KERNEL_NUM * self.matmul_n_l0, self.matmul_m_l0, Constant.M0, Constant.N0)
            c_ub = tik_instance.Tensor(self.dtype, cub_shape, name="c_ub", scope=tik.scope_ubuf)
            tik_instance.tensor_mov(c_ub, l0c, 'm', 1, Constant.KERNEL_NUM * self.matmul_n_l0 * self.matmul_m_l0, 0, 0)
            cub_burst_len = self.matmul_m_l0 * Constant.M0
            dst_stride = (self.m1_shape - self.matmul_m_l0) * Constant.M0
            out_offset = matmul_m_idx * Constant.FRAC_SIZE + matmul_n_idx * self.m1_shape * Constant.FRAC_SIZE
            for i in range(Constant.KERNEL_NUM):
                c_ub_offset = i * self.matmul_n_l0 * self.matmul_m_l0 * Constant.FRAC_SIZE
                tik_instance.data_move(out_gms[i][out_offset:], c_ub[c_ub_offset:], 0, self.matmul_n_l0, cub_burst_len,
                    0, dst_stride)
            # bias_out process
            cub_bias_shape = (Constant.KERNEL_NUM * self.matmul_n_l0, 1, Constant.M0, Constant.N0)
            c_ub_bias = tik_instance.Tensor(self.dtype, cub_bias_shape, name="c_ub_bias", scope=tik.scope_ubuf)
            tik_instance.tensor_mov(c_ub_bias, l0c_bias, 'm', 1, Constant.KERNEL_NUM * self.matmul_n_l0, 0, 0)
            out_offset = matmul_n_idx * Constant.N0
            for i in range(Constant.KERNEL_NUM):
                c_ub_offset = i * self.matmul_n_l0 * Constant.FRAC_SIZE
                tik_instance.data_move(bias_out_gms[i][out_offset:], c_ub_bias[c_ub_offset:], 0, self.matmul_n_l0, 1,
                    Constant.C0 - 1, 0)


    def _matmul_l0c_compute(self, tik_instance, tensor_list, idx_list, ping_pong):
        '''
        matmul_l0c_compute
        '''
        al1_shape = (self.matmul_m_l1, self.matmul_k_al1, Constant.C0, Constant.M0)
        bl1_shape = (Constant.KERNEL_NUM * self.matmul_n_l1, self.matmul_k_bl1, Constant.C0, Constant.N0)
        al0_shape = (self.matmul_m_l0, self.matmul_k_l0, Constant.M0, Constant.C0)
        bl0_shape = (self.matmul_k_l0, Constant.KERNEL_NUM * self.matmul_n_l0, Constant.N0, Constant.C0)
        matmul_m_idx, matmul_n_idx, kl1_factor_idx = idx_list
        x_gm, kernel_gms, _, _, l0c, l0c_bias, a_ub = tensor_list
        ping_pong_suffix = "ping" if ping_pong == 0 else "pong"
        # al1_process
        al1 = tik_instance.Tensor(self.dtype, al1_shape, name="al1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        a_src_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_al1 * \
            Constant.FRAC_SIZE + matmul_m_idx * self.k1_shape * Constant.FRAC_SIZE
        tik_instance.data_move(al1, x_gm[a_src_offset:], 0, self.matmul_m_l1, self.matmul_k_al1 * Constant.C0,
            (self.k1_shape - self.matmul_k_al1) * Constant.C0, 0)
        # bl1_process
        b_src_offset = (Constant.DOUBLE_BUFFER * kl1_factor_idx + ping_pong) * self.matmul_k_bl1 * \
            Constant.FRAC_SIZE + matmul_n_idx * self.k1_shape * Constant.FRAC_SIZE
        bl1 = tik_instance.Tensor(self.dtype, bl1_shape, name="bl1_" + ping_pong_suffix, scope=tik.scope_cbuf)
        # concat three kernels
        for i in range(Constant.KERNEL_NUM):
            bl1_offset = i * self.matmul_n_l1 * self.matmul_k_bl1 * Constant.FRAC_SIZE
            tik_instance.data_move(bl1[bl1_offset], kernel_gms[i][b_src_offset:], 0, self.matmul_n_l0,
                self.matmul_k_bl1 * Constant.C0, (self.k1_shape - self.matmul_k_bl1) * Constant.C0, 0)
        with tik_instance.for_range(0, self.matmul_m_l1 // self.matmul_m_l0):
            al0 = tik_instance.Tensor(self.dtype, al0_shape, name="al0_" + ping_pong_suffix, scope=tik.scope_ca)
            # al0 process
            tik_instance.load2dv1(al0, al1, 0, self.matmul_m_l0 * self.matmul_k_l0, 1, 0, True)
            # bl0 process
            bl0 = tik_instance.Tensor(self.dtype, bl0_shape, name="bl0_" + ping_pong_suffix, scope=tik.scope_cb)
            cond_params = [ping_pong, kl1_factor_idx, False]
            mad_tensors = [al0, bl0, l0c]
            mad_size = [self.matmul_m_l0, self.matmul_k_l0, Constant.KERNEL_NUM * self.matmul_n_l0]
            with tik_instance.for_range(0, self.matmul_n_l1 // self.matmul_n_l0):
                with tik_instance.for_range(0, self.matmul_k_l0) as kl0_idx:
                    bl1_offset = kl0_idx * Constant.C0 * Constant.N0
                    bl0_offset = Constant.KERNEL_NUM * kl0_idx * self.matmul_n_l0 * Constant.N0 * Constant.C0
                    tik_instance.load2dv1(bl0[bl0_offset:], bl1[bl1_offset:], 0, Constant.KERNEL_NUM * \
                        self.matmul_n_l0, self.matmul_k_l0, 0, True)
                # l0c process
                matmul_l0c_process(tik_instance, cond_params, mad_tensors, mad_size)
            tik_instance.data_move(al1, a_ub[ping_pong * self.matmul_k_al1 * Constant.FRAC_SIZE:], 0, 1,
                self.matmul_k_al1 * Constant.C0, 0, 0)
            tik_instance.load2dv1(al0, al1, 0, self.matmul_k_al1, 1, 0, False)
            # l0c_bias process
            mad_tensors[-1] = l0c_bias
            mad_size[0] = 1
            matmul_l0c_process(tik_instance, cond_params, mad_tensors, mad_size)


@register_operator("AttentionQKVGradW")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT,
                            para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_BOOL, para_check.OPTION_ATTR_BOOL,
                            para_check.KERNEL_NAME)
def attention_qkv_grad_w(x, query_dx, key_dw, value_dw, dw_query, dw_key, dw_value, dbias_query, dbias_key,
                         dbias_value, trans_a=True, trans_b=False, kernel_name="attention_qkv_grad_w"):
    """
    Parameters
    ----------
    x: dict
        shape and dtype of input x, only support float16
    query_dx: dict
        shape and dtype of input query_dx, only support float16
    key_dw: dict
        shape and dtype of input key_dw, only support float16
    value_dw: dict
        shape and dtype of input value_dw, only support float16
    dw_query: dict
        shape and dtype of output dw_query, only support float16
    dw_key: dict
        shape and dtype of output dw_key, only support float16
    dw_value: dict
        shape and dtype of output dw_value, only support float16
    dbias_query: dict
        shape and dtype of output dbias_query, only support float16
    dbias_key: dict
        shape and dtype of output dbias_key, only support float16
    dbias_value: dict
        shape and dtype of output dbias_value, only support float16
    trans_a: bool
        If True, shape_a == transposed before multiplication, default to be true
    trans_b: bool
        If True, the shape in input_x2 must be transposed before multiplication, default to be false
    kernel_name: str
        cce kernel name, default value is "attention_qkv_grad_w"

    Returns
    -------
    None
    """
    ln_input_shape = shape_util.shape_to_list(x.get("shape"))
    query_dx_shape = shape_util.shape_to_list(query_dx.get("shape"))
    key_dw_shape = shape_util.shape_to_list(key_dw.get("shape"))
    value_dw_shape = shape_util.shape_to_list(value_dw.get("shape"))
    dw_query_shape = shape_util.shape_to_list(dw_query.get("shape"))
    dw_key_shape = shape_util.shape_to_list(dw_key.get("shape"))
    dw_value_shape = shape_util.shape_to_list(dw_value.get("shape"))
    dbias_query_shape = shape_util.shape_to_list(dbias_query.get("shape"))
    dbias_key_shape = shape_util.shape_to_list(dbias_key.get("shape"))
    dbias_value_shape = shape_util.shape_to_list(dbias_value.get("shape"))
    data_type = x.get("dtype")
    check_dtype("attention_qkv_grad_w", data_type)
    input_x_format = x.get("format").upper()
    kernel_format = query_dx.get("format").upper()
    check_format("attention_qkv_grad_w", input_x_format, kernel_format)
    check_equal_shape("attention_qkv_grad_w", [query_dx_shape, key_dw_shape, value_dw_shape],
                      "kernel_shape in matmul_dw_qkv should be equal.")
    check_equal_shape("attention_qkv_grad_w", [dw_query_shape, dw_key_shape, dw_value_shape],
                      "matmul_dw_qkv out_shape should be equal.")
    check_equal_shape("attention_qkv_grad_w", [dbias_query_shape, dbias_key_shape, dbias_value_shape],
                      "matmul_dw_qkv bias_out_shape should be equal.")
    check_trans_flag("attention_qkv_grad_w", not trans_a, trans_b)
    params = {
        "dtype": x.get("dtype"),
        "ln_input_shape": ln_input_shape,
        "kernel_shape": query_dx_shape,
        "dw_out_shape": dw_query_shape,
        "bias_out_shape": dbias_query_shape,
        "trans_a": trans_a,
        "trans_b": trans_b,
        "kernel_name": kernel_name
    }
    obj = AttentionQKVGradW(params)
    obj.attention_qkv_gradw_compute()
