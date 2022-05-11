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
public function for attention_qkv big kernels
"""
from impl.util.platform_adapter import error_manager_cube


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
    CANDIDATE_TILING_K = 4
    CANDIDATE_BLOCK_M = 4
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


def check_equal_shape(op_type, shape_list, err_msg):
    '''
    check shape equal
    '''
    if shape_list[0] != shape_list[1] or shape_list[1] != shape_list[-1]:
        error_manager_cube.raise_err_specific_user(op_type, err_msg)


def check_dtype(op_type, data_type):
    '''
    check dtype is fp16
    '''
    if data_type != Constant.FP16_DTYPE:
        error_manager_cube.raise_err_specific_user(op_type, "the only supported dtype is fp16.")


def check_format(op_type, input_x_format, kernel_format):
    '''
    check format is FRACTAL_NZ
    '''
    if input_x_format != "FRACTAL_NZ" or kernel_format != "FRACTAL_NZ":
        error_manager_cube.raise_err_specific_user(op_type, "only support NZ format for matmul.")


def check_trans_flag(op_type, trans_a, trans_b):
    '''
    check trans_flag
    '''
    if trans_a or trans_b:
        error_manager_cube.raise_err_specific_user(op_type, "unsupported transpose flag for matmul.")


def matmul_l0c_process(tik_instance, cond_params, mad_tensors, mad_size):
    '''
    matmul_l0c_process
    '''
    ping_pong, kl1_factor_idx, bias_flag = cond_params
    al0, bl0, l0c = mad_tensors
    m_l0, k_l0, n_l0 = mad_size
    if ping_pong == 0:
        if bias_flag:
            tik_instance.mmad(l0c, al0, bl0, m_l0 * Constant.M0, k_l0 * Constant.C0, n_l0 * Constant.N0, 1)
        else:
            with tik_instance.if_scope(kl1_factor_idx == 0):
                tik_instance.mmad(l0c, al0, bl0, m_l0 * Constant.M0, k_l0 * Constant.C0, n_l0 * Constant.N0, 0)
            with tik_instance.else_scope():
                tik_instance.mmad(l0c, al0, bl0, m_l0 * Constant.M0, k_l0 * Constant.C0, n_l0 * Constant.N0, 1)
    else:
        tik_instance.mmad(l0c, al0, bl0, m_l0 * Constant.M0, k_l0 * Constant.C0, n_l0 * Constant.N0, 1)


def vconv(tik_instance, src_tensor, dst_tensor, vconv_repeat_size, fp16_to_fp32):
    '''
    vconv repeat size may exceeds 255, multi vconv instrs may needed
    '''
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