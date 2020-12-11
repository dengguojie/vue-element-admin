# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
mmad schedule
"""
from __future__ import absolute_import
from math import ceil

import te.platform.cce_params as cce
import te.platform.cce_conf as conf
from te.platform.cce_conf import CceProductParams as pver
from te.platform.fusion_manager import fusion_manager
from te.platform import get_soc_spec
from te.platform import cce_emitinsn_params
import tvm
from . import util

DTYPE_WIDTH_MAP = {"uint64": 4,
                   "float16": 1,
                   "float32": 2,
                   "int32": 2,
                   "int16": 1,
                   "uint16": 1,
                   "int8": 0.5,
                   "uint8": 0.5,
                   "bool": 0.5}

# need to process when cal width
DEQ_SCALE_CHILD_LIST = [
    "dequant",
    "dequant_scale",
    "dequant_sqrt",
]

DOUBLE_VALUE = 2
CORE_NUM_THRITY = 30

def gemm_para_check(gm_shape, l1_tiling_shape, l0_tiling_shape):
    """
    algorithm: gemm_para_check

    Parameters
    ----------
    gm_shape : the M,N,K shape

    l1_tiling_shape : L1 Tiling shape

    l0_tiling_shape : L0 Tiling shape

    Returns :
    -------
    None
    """
    gm_m = gm_shape[0]
    gm_k = gm_shape[1]
    gm_n = gm_shape[2]

    block_m = l1_tiling_shape[0]
    block_k = l1_tiling_shape[1]
    block_n = l1_tiling_shape[2]

    block_ml = l0_tiling_shape[0]
    block_kl = l0_tiling_shape[1]
    block_nl = l0_tiling_shape[2]

    # check the block
    if block_m <= 0 or block_n <= 0 or block_k <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual (block_m, "
            "block_n, block_k) = (%d, %d, %d)" % (block_m, block_n, block_k))
    if ((block_m % 16 != 0) or (block_n % 16 != 0) or (
            block_k % 16 != 0)) and block_m != 1 and block_n != 1:
        raise RuntimeError(
            "input shape block_m or block_k or block_n should be multiple of "
            "16: actual (block_m, block_k, block_n) = (%d, %d, %d)" %
            (block_m, block_k, block_n))

    # check the block L0
    if block_ml <= 0 or block_nl <= 0 or block_kl <= 0:
        raise RuntimeError(
            "input block param should not be less than 0: actual "
            "(block_ml, block_nl, block_kl) = (%d, %d, %d)" %
            (block_ml, block_nl, block_kl))
    if block_ml > block_m or block_nl > block_n or block_kl > block_k:
        raise RuntimeError(
            "input block param should not be less than blockL1 value: actual "
            "(block_ml, block_nl, block_kl) = (%d, %d, %d)" %
            (block_ml, block_nl, block_kl))
    if ((block_ml % 16 != 0) or (block_nl % 16 != 0) or (
            block_kl % 16 != 0)) and block_ml != 1 and block_nl != 1:
        raise RuntimeError(
            "input shape block_ml or block_kl or block_nl should be multiple "
            "of 16: actual (block_ml, block_kl, block_nl) = (%d, %d, %d)" %
            (block_ml, block_kl, block_nl))


def _add_res_ub(dequant_activation_tensor, res, sch):
    """
    add res_ub tensor to dequant_activation_tensor.
    """
    for tensor in dequant_activation_tensor:
        if tensor == res:
            res_ub = sch.cache_write(res, cce.scope_ubuf)
            dequant_activation_tensor.remove(tensor)
            dequant_activation_tensor.append(res_ub)


def _get_header_tensor_in_dequant_ew_fusion(dequant_activation_tensor,
                                            placeholder_tensors, sch):
    """
    add header_ub tensor to dequant_activation_tensor.
    """
    header_set = set(placeholder_tensors)
    header_ub_tensors = list()
    comm_2_elwt = dict()
    for ten_i in dequant_activation_tensor:
        common_tensors = header_set & set(ten_i.op.input_tensors)
        for common_tensor in common_tensors:
            if common_tensor in comm_2_elwt:
                comm_2_elwt[common_tensor].append(ten_i)
            else:
                comm_2_elwt[common_tensor] = [ten_i]
    for common_tensor, ten_in_list in comm_2_elwt.items():
        common_tensor_ub = sch.cache_read(
            common_tensor, cce.scope_ubuf, ten_in_list)
        header_ub_tensors.append(common_tensor_ub)
    dequant_activation_tensor += header_ub_tensors
    return header_ub_tensors


def _shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        tmp.append(i.value)
    return tmp


def get_special_l0_factor(src_shape, m_l0_shape, k_l0_shape, n_l0_shape):
    """
    get temp factors
    """
    m_shape = src_shape[0]
    k_shape = src_shape[1]
    n_shape = src_shape[2]
    if m_shape * n_shape * k_shape == m_l0_shape * n_l0_shape * k_l0_shape and \
            m_l0_shape != 1:
        m_l0_shape = int((m_l0_shape // 2))
        if int((m_l0_shape % 16)) != 0:
            m_l0_shape = int((m_l0_shape + 15) // 16 * 16)

    src_shape = [m_shape, k_shape, n_shape]
    if src_shape == [256, 64, 256]:
        m_l0_shape = 256
        k_l0_shape = 64
        n_l0_shape = 128
    elif src_shape == [256, 256, 64]:
        m_l0_shape = 64
        k_l0_shape = 256
        n_l0_shape = 64
    return m_l0_shape, k_l0_shape, n_l0_shape


def get_batch_factors(tensor_a_shape,  # pylint: disable=too-many-arguments
                      tensor_a_l0a, tensor_b_l0b, m_var, n_var, is_gemv,
                      n_nparts_mode):
    """
    get batch vars
    """
    m_shape = m_var[0]
    m_factors = m_var[1]
    n_shape = n_var[0]
    n_factors = n_var[1]
    core_inner_m = m_shape
    core_inner_n = n_shape
    if len(tensor_a_shape) == 3 or len(tensor_a_shape) == 5:
        batch = tensor_a_l0a.shape[0].value
        if is_gemv:
            batch = tensor_b_l0b.shape[0].value
    else:
        batch = 0
    if batch in (0, 1):
        block_in = cce.BLOCK_IN
        block_out = cce.BLOCK_OUT
        if m_shape != 1:
            core_inner_m = (((m_shape + block_in - 1) // block_in +
                             (m_factors - 1)) // m_factors) * block_in
        if n_nparts_mode:
            core_inner_n = (((n_shape + block_out - 1) // block_out +
                             (n_factors - 1)) // n_factors) * block_out
        else:
            # in quant fsuion factor mode, n_factors means each block
            # process fract number
            core_inner_n = n_factors * block_out

    return batch, core_inner_m, core_inner_n


def get_shape_map():
    """
    the knowledge of matmul schedule tiling
    """
    shape_map = {(1664, 4096, 1024, -1, 2): "176_320_176_176_80_176_2_2",
                 (1664, 4096, 1024, -1, 4): "176_320_176_176_80_176_2_2",
                 (1664, 1024, 4096, -1, 2): "240_512_128_240_64_128_2_2",
                 (1664, 1024, 4096, -1, 8): "240_512_64_240_64_64_2_2",
                 (1664, 16, 1024, -1, 2): "832_16_128_832_16_32_1_2",
                 (1664, 1024, 1024, -1, 2): "240_512_128_240_64_128_2_2",
                 (1664, 1024, 1024, -1, 4): "240_512_128_240_64_128_2_2",
                 (832, 4096, 1024, -1, 2): "176_320_176_176_80_176_2_2",
                 (832, 4096, 1024, -1, 4): "176_320_176_176_80_176_2_2",
                 (832, 1024, 4096, -1, 2): "240_512_128_240_64_128_2_2",
                 (832, 1024, 4096, -1, 8): "240_512_64_240_64_64_2_2",
                 (832, 1024, 1024, -1, 2): "240_512_128_240_64_128_2_2",
                 (832, 1024, 1024, -1, 4): "240_512_128_240_64_128_2_2",
                 (832, 16, 1024, -1, 2): "832_16_128_832_16_32_1_2",
                 (1280, 16, 768, -1, 2): "640_16_192_640_16_48_1_2",
                 (1280, 768, 768, -1, 2): "336_384_96_336_48_96_2_2",
                 (320, 64, 320, -1, 2): "320_64_192_320_48_96_1_2",
                 (1280, 768, 3072, -1, 2): "336_384_96_336_48_96_2_2",
                 (1280, 16, 768, -1, 2): "640_16_192_640_16_48_1_2",
                 (320, 64, 320, -1, 2): "320_64_192_320_48_96_1_2",
                 (1280, 768, 768, -1, 4): "320_384_96_320_48_96_2_2",
                 (16, -1, 4096, 0, 2): "16_16_1024_16_16_1024_2_2"
                 }

    return shape_map


def get_mini_frac_shape_map():
    """
    the knowledge of matmul schedule tiling
    """
    shape_map = {(304, -1, 4096, -1, 2): "304_80_192_304_80_192_2_2",
                 (304, -1, 4096, -1, 4): "304_80_192_304_80_192_2_2",
                 (304, -1, 4096, -1, 6): "304_80_192_304_80_192_2_2"
                 }

    return shape_map


def get_cloud_shape_map():
    """
    the knowledge of matmul schedule tiling
    """
    shape_map = {(1024, 20480, 1024, -1, 4): "256_512_160_256_64_160_2_2"
                 }

    return shape_map


def get_core_map():
    """
    the knowledge of matmul schedule core tiling
    """
    shape_map = {(1024, 20480, 1024): (4, 7),
                 (4096, 20480, 1024): (7, 4),
                 (20480, 4096, 1024): (15, 2),
                 (1024, 20480, 4096): (4, 7)
                 }
    return shape_map


def get_l1fusion_device_core_num(is_l1fusion):
    """
    get the device core num
    :param is_l1fusion: is l1 fusion
    :return: device core num
    """
    if is_l1fusion:
        device_core_num = 1
    else:
        device_core_num = conf.getValue("Device_core_num")
    return device_core_num


def _get_ub_res_byte(_get_out_tensors_width, dequant_activation_tensor,
                     fusion_ele, res, ub_res_byte):
    """
    calculate res ub byte by width
    """
    if fusion_ele or dequant_activation_tensor:
        width = _get_out_tensors_width(res)
        if ub_res_byte < width * 2:
            ub_res_byte = width * 2
    return ub_res_byte


def get_perfect_core_num(m_shape,  # pylint: disable=too-many-locals
                         n_shape, k_shape, l1_fusion_type):
    """
    :param input_shape_1:the tensor_a shape
    :param input_shape_2:the tensor_b shape
    :return:core_num
    """
    frac_size = 16
    is_l1fusion = l1_fusion_type in (0, 1)
    core_num = get_l1fusion_device_core_num(is_l1fusion)
    m_axis_outer = (m_shape + frac_size - 1) // frac_size
    if m_shape == 1:
        m_axis_outer = 1
        n_axis_outer = (n_shape + frac_size - 1) // frac_size
        if n_axis_outer > core_num:
            return 1, core_num
        return 1, 1

    m_axis_outer = m_shape // frac_size
    n_axis_outer = n_shape // frac_size
    if (m_axis_outer * n_axis_outer) <= core_num:
        return m_axis_outer, n_axis_outer
    tensor_a_size = m_shape * k_shape
    tensor_b_size = n_shape * k_shape
    min_copy_size = core_num * (tensor_a_size + tensor_b_size)
    m_factor = 1
    n_factor = 1

    for i in range(1, core_num + 1):
        # judge cur_factor
        if core_num % i != 0:
            continue

        cur_m_factor = i
        cur_n_factor = core_num // i
        if cur_m_factor > m_axis_outer or (m_axis_outer // cur_m_factor) == 0:
            continue
        if cur_n_factor > n_axis_outer or (n_axis_outer // cur_n_factor) == 0:
            continue

        cur_copy_size = cur_n_factor * tensor_a_size + cur_m_factor * \
            tensor_b_size
        temp_m_shape = m_shape
        temp_n_shape = n_shape
        if m_axis_outer % m_factor != 0:
            temp_m_shape = (((m_axis_outer // cur_m_factor) + 1) *
                            cur_m_factor) * frac_size

        if n_shape % n_factor != 0:
            temp_n_shape = (((n_axis_outer // cur_n_factor) + 1) *
                            cur_n_factor) * frac_size

        cur_copy_size = cur_n_factor * (temp_m_shape * k_shape) + \
            cur_m_factor * (temp_n_shape * k_shape)
        if cur_copy_size < min_copy_size:
            min_copy_size = cur_copy_size
            m_factor = cur_m_factor
            n_factor = cur_n_factor

    return m_factor, n_factor


def is_lhisi_cs_version():
    """
    check if 3796CS version
    -------

    Returns
    -------
    True: 3796CS version
    False: Other version
    """
    soc_version = get_soc_spec("SOC_VERSION")
    if soc_version in ["Hi3796CV300CS"]:
        return True
    return False


def get_knowledge_tiling(shape_tiling_args, is_b_nz, tiling_shape):
    """
    get knowledge tiling for matmul schedule
    """
    m_shape, k_shape, n_shape, b_trans, ub_res_byte = shape_tiling_args
    b_trans_val = -1
    if b_trans is not None:
        b_trans_val = 1 if b_trans else 0
    shape_args = (m_shape, k_shape, n_shape, b_trans_val, ub_res_byte)

    shape_map = {}
    core_num = get_soc_spec("CORE_NUM")
    if core_num == DOUBLE_VALUE:
        if is_b_nz:
            shape_map = get_shape_map()
        else:
            shape_map = get_mini_frac_shape_map()
    elif core_num == CORE_NUM_THRITY:
        shape_map = get_cloud_shape_map()
    if shape_map.get(shape_args) is not None:
        tiling_shape = shape_map[shape_args]
    else:
        shape_args = (m_shape, k_shape, n_shape, -1, ub_res_byte)
        if shape_map.get(shape_args) is not None:
            tiling_shape = shape_map[shape_args]
        else:
            shape_args = (m_shape, -1, n_shape, b_trans_val, ub_res_byte)
            if shape_map.get(shape_args) is not None:
                tiling_shape = shape_map[shape_args]
            else:
                shape_args = (m_shape, -1, n_shape, -1, ub_res_byte)
                if shape_map.get(shape_args) is not None:
                    tiling_shape = shape_map[shape_args]

    return tiling_shape


def get_knowledge_core(shape_mkn_args, m_factors, n_factors):
    """
    get knowledge of core set

    Parameters
    ----------
    shape_mkn_args : list, shape info

    m_factors: core split in m_factor

    n_factors: core split in n_factors

    Returns
    -------
    m_factors, n_factors, value of m, n core split
    """
    shape_map = {}
    core_num = get_soc_spec("CORE_NUM")
    if core_num == CORE_NUM_THRITY:
        shape_map = get_core_map()

    if shape_map.get(shape_mkn_args) is not None:
        m_factors, n_factors = shape_map[shape_mkn_args]

    return m_factors, n_factors


def update_op_pattern(fractal_a, fractal_b):
    """
    only support frac+frac for elementwise fusion
    """
    if not fractal_a or not fractal_b:
        fusion_manager.set_current_op_pattern("Opaque")


def set_overload_flag(overload_flag, current_op, pragma_axis):
    """
    set overload flag
    """
    if current_op is not None and pragma_axis is not None:
        if overload_flag:
            current_op.pragma(pragma_axis, "json_info_cache_read_mode", 0)
        else:
            current_op.pragma(pragma_axis, "json_info_cache_read_mode", 1)


def get_refresh_core_factors(m_factors, n_factors, batch):
    """
    get refresh
    """
    if batch > 1:
        m_factors = 1
        n_factors = 1

    return m_factors, n_factors


def get_tensor_c_axis(is_fractal_a, is_fractal_b, tensor_a_reuse_local,
                      tensor_b_reuse_local,
                      l1_n_outer, l1_m_outer):  # pylint: disable=too-many-arguments
    """
    get tensor c axis for allocate_at
    """
    tensor_c_l1_reuse_axis_outter = l1_n_outer
    tensor_c_l1_reuse_axis_inner = l1_m_outer
    if is_fractal_a and is_fractal_b:
        if tensor_a_reuse_local == 1 and tensor_b_reuse_local != 1:
            tensor_c_l1_reuse_axis_outter = l1_m_outer
            tensor_c_l1_reuse_axis_inner = l1_n_outer
        elif tensor_b_reuse_local != 1 and tensor_b_reuse_local == 1:
            tensor_c_l1_reuse_axis_outter = l1_n_outer
            tensor_c_l1_reuse_axis_inner = l1_m_outer
    else:
        tensor_c_l1_reuse_axis_outter = l1_m_outer
        tensor_c_l1_reuse_axis_inner = l1_n_outer

    return tensor_c_l1_reuse_axis_outter, tensor_c_l1_reuse_axis_inner


def get_res_axis(tensor_a_reuse_local, tensor_b_reuse_local, m_outer, n_outer):
    """
    get res axis for allocate
    """
    l1_reuse_axis_outter = m_outer
    l1_reuse_axis_inner = n_outer
    if tensor_a_reuse_local == 1 and tensor_b_reuse_local != 1:
        l1_reuse_axis_outter = m_outer
        l1_reuse_axis_inner = n_outer
    elif tensor_a_reuse_local != 1 and tensor_b_reuse_local == 1:
        l1_reuse_axis_outter = n_outer
        l1_reuse_axis_inner = m_outer
    elif tensor_a_reuse_local == 1 and tensor_b_reuse_local == 1:
        l1_reuse_axis_outter = m_outer
        l1_reuse_axis_inner = n_outer

    return l1_reuse_axis_outter, l1_reuse_axis_inner


def allocate_axis(sch, batch_double, double_once, tensor_a_reuse_local,
                  tensor_b_reuse_local, tensor_a_l1, tensor_b_l1,
                  tensor_c, res, n_outer_axis, m_outer_axis, l1_k_outer,
                  in_addr_type, input_l1_flag, l1_fusion_and_l1_size_0,
                  gemv_flag):  # pylint: disable=too-many-arguments
    """
    allocate_axis_for tensor_a and tensor_b
    """
    m_outer = m_outer_axis
    n_outer = n_outer_axis
    if gemv_flag:
        m_outer = n_outer_axis
        n_outer = m_outer_axis

    if batch_double:
        if double_once != 0 and tensor_a_reuse_local != 0:
            if in_addr_type == 0 and not l1_fusion_and_l1_size_0 and input_l1_flag != 1:
                sch[tensor_a_l1].allocate_at(sch[res], n_outer, run_once_axes=[n_outer])
                sch[tensor_a_l1].mem_unique()

        def _tensor_b_l1_allocate1():
            if double_once != 0 and tensor_b_reuse_local != 0:
                sch[tensor_b_l1].allocate_at(sch[res], m_outer, run_once_axes=[m_outer])
                sch[tensor_b_l1].mem_unique()
            if double_once == 0:
                sch[tensor_b_l1].compute_at(sch[res], n_outer)
            else:
                sch[tensor_b_l1].compute_at(sch[tensor_c], l1_k_outer)

        if not l1_fusion_and_l1_size_0:
            _tensor_b_l1_allocate1()
    else:
        if tensor_a_reuse_local != 0:
            if in_addr_type == 0 and not l1_fusion_and_l1_size_0 and input_l1_flag != 1:
                sch[tensor_a_l1].allocate_at(sch[res], n_outer, run_once_axes=[n_outer])
                sch[tensor_a_l1].mem_unique()

        def _tensor_b_l1_allocate2():
            if tensor_b_reuse_local != 0:
                sch[tensor_b_l1].allocate_at(sch[res], m_outer, run_once_axes=[m_outer])
                sch[tensor_b_l1].mem_unique()
            sch[tensor_b_l1].compute_at(sch[tensor_c], l1_k_outer)

        if not l1_fusion_and_l1_size_0:
            _tensor_b_l1_allocate2()

    return sch


def get_tensor_reuse(batch, core_inner_m, k_shape, core_inner_n,
                     m_l1_shape, k_l1_shape, n_l1_shape,
                     dtype_byte):  # pylint: disable=too-many-locals, unused-argument, too-many-arguments
    """
    get the result of resue axis
    """
    tensor_a = 0
    tensor_b = 0

    size = get_soc_spec("L1_SIZE") // dtype_byte // 2
    n_max_num = ((core_inner_n + n_l1_shape - 1) // n_l1_shape) * n_l1_shape

    m_max_num = ((core_inner_m + m_l1_shape - 1) // m_l1_shape) * m_l1_shape
    if k_shape >= size:
        tensor_a = 0
        tensor_b = 0
    tensor_a_num = m_max_num * k_shape
    tensor_b_num = n_max_num * k_shape

    if m_max_num != m_l1_shape:
        if m_max_num * k_shape <= size:
            tensor_a = 2
        elif m_l1_shape * k_shape <= size:
            tensor_a = 1

    if n_max_num != n_l1_shape:
        if n_max_num * k_shape <= size:
            tensor_b = 2
        elif n_l1_shape * k_shape <= size:
            tensor_b = 1

    if tensor_a == 1 and tensor_b == 1:
        aprts_a = core_inner_m // m_l1_shape
        aprts_b = core_inner_n // n_l1_shape
        if tensor_a_num * aprts_b < tensor_b_num * aprts_a:
            tensor_a = 0
        else:
            tensor_b = 0

    batch_double = False
    if batch > 1:
        if tensor_a_num <= size and tensor_b_num <= size:
            batch_double = True

    double_once = 0
    if core_inner_m != m_l1_shape and core_inner_n != n_l1_shape:
        double_once = 1

    return tensor_a, tensor_b, batch_double, double_once


def _map_apend(input_map, key, value):
    """
    map apend
    """
    if input_map.get(key):
        if isinstance(value, list):
            for sub_v in value:
                if sub_v not in input_map[key]:
                    input_map[key].append(sub_v)
        else:
            if value not in input_map[key]:
                input_map[key].append(value)
    else:
        if isinstance(value, list):
            input_map[key] = value
        else:
            input_map[key] = [value]


def _gen_in_out_tensor_map(out_tensor, in_out_tensor_map):
    """
    traverse tensors by Depth-First-Search

    Parameters
    ----------
    out_tensor : tensor
        traverse tensors from this tensor,
        traversing its input tensors recursively.

    tensor_list : list
        record tensors in the order of Depth-First-Search.

    """
    if out_tensor is None:
        return
    stack = [out_tensor]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
            _map_apend(in_out_tensor_map, in_tensor, cur_tensor)


def check_placeholders_shared(fusion_ele, tensor_a, tensor_b,
                              res, matmul_tensors):
    """check placeholders shared"""
    if not fusion_ele:
        return None

    in_out_tensor_map = {}
    _gen_in_out_tensor_map(res, in_out_tensor_map)
    if tensor_a in in_out_tensor_map:
        for ten_i in in_out_tensor_map[tensor_a]:
            if ten_i not in matmul_tensors:
                raise RuntimeError("matmul placeholders can't be shared "
                                   "with elementwise op")
    if tensor_b in in_out_tensor_map:
        for ten_i in in_out_tensor_map[tensor_b]:
            if ten_i not in matmul_tensors:
                raise RuntimeError("matmul placeholders can't be shared "
                                   "with elementwise op")


def get_output_format(tensor):
    """
    get output format

    Parameters
    ----------
    tensor : res tensor

    Return :
    output format
    """
    format_out = "FRACTAL_NZ"
    if tensor is None:
        return format_out
    if tensor.op.attrs is None:
        return format_out
    if 'format' not in tensor.op.attrs:
        return format_out
    format_out = tensor.op.attrs['format']
    return format_out


def match_and_get_tensor(tensors, tensor_name):
    """
    match and get tensor
    """
    for i in tensors:
        if tensor_name == i.op.name:
            return i
    return None


def get_weigths_and_compress_index(tensors):
    """
    get tensor_b_l1 and compress_index info
    """
    b_l1 = match_and_get_tensor(tensors, 'tensor_b_l1')

    comp_index = None
    if "tile_L1_n" in b_l1.op.attrs:
        comp_index = b_l1.op.input_tensors[0]

    return b_l1, comp_index


def get_compress_block_info(tight_mode,  # pylint: disable=too-many-locals
                            tensor_w, tile_k, tile_n):
    """
    get weigths compress info, like, block size, index size
    """
    block_size_max = 32 * 1024
    block_unit = 512

    data_size = tile_k * tile_n * block_unit

    size_max = block_size_max
    if 0 < data_size < size_max:
        size_max = data_size

    block_size = block_unit
    for block_idx in range(size_max, 0, block_unit*(-1)):
        if data_size % block_idx == 0:
            block_size = block_idx
            break

    return int(block_size)


def emit_insn_func(sche, insn_tensor, insn_axis, insn_tag):
    """
    emit instion function
    """
    if insn_tensor is not None:
        if insn_axis is None:
            tensor_len = len(insn_tensor.shape)
            if tensor_len in (2, 4):
                insn_idx = 0
            else:
                insn_idx = 1
            insn_axis = insn_tensor.op.axis[insn_idx]
        sche[insn_tensor].emit_insn(insn_axis, insn_tag)


def set_compress_info(sch,  # pylint: disable=R0913, R0914
                      compress_tensor, compress_index,
                      tile_k, tile_n, out_axis):
    """
    set weigths compress info
    """
    if out_axis is None:
        raise RuntimeError("compress index axis is None, it's error.")

    engine, ratios, channel, mode = get_soc_spec("UNZIP")
    frac_size = 512

    index_shape = compress_index.shape
    dim_k = compress_tensor.shape[0].value
    dim_n = compress_tensor.shape[1].value

    tile_k_value = compress_tensor.op.attrs["tile_L1_k"]
    tile_n_value = compress_tensor.op.attrs["tile_L1_n"]

    block_size = get_compress_block_info(mode,
                                         compress_tensor, tile_k, tile_n)

    k_value = block_size // (tile_n * frac_size)

    sch.set_var_range(tile_k_value, k_value, k_value)
    sch.set_var_range(tile_n_value, tile_n, tile_n)

    k_block_num = (dim_k + k_value - 1) // k_value
    n_block_num = (dim_n + tile_n - 1) // tile_n
    index_size = k_block_num * n_block_num

    tight_len = 2
    if mode == 1:
        tight_len = 8
    index_size = index_size * tight_len

    sch.set_var_range(index_shape[0], int(index_size), int(index_size))

    conflict = tvm.make.Call("int32", "tvm_tuple",
                             (block_size, index_size, mode, engine,
                              channel, ratios, k_value, tile_n),
                             tvm.expr.Call.PureIntrinsic, None, 0)
    sch[compress_tensor].pragma(compress_tensor.op.axis[0],
                                "json_info_compress_parameters", conflict)
    tensor_len = len(compress_tensor.shape)
    # transform data to continue by block_size
    sch[compress_tensor].emit_insn(compress_tensor.op.axis[tensor_len - 4],
                                   "unzip",
                                   {"compress_mode": mode,
                                    "block_size": block_size,
                                    "hoist_axis": out_axis})


def mmad_schedule(res, sch_list):
    """
    algorithm: mmad_schedule

    Parameters
    ----------
    res : the output tensor

    sch_list : schedule list

    Returns : if it is true,then we can get the valid schedule

    """
    # pylint: disable=too-many-locals, too-many-branches, too-many-statements

    emit_fusion_insn_map = {"dequant_NZ": "phony_insn",
                            "cast_f16_ub": "vector_conv",
                            "input_ub": "phony_insn",
                            "reform_by_vmuls": "vector_muls",
                            "scale_sqrt_ub": "vector_muls",
                            "offset_ub": "vector_adds",
                            "cast_i8_ub": "vector_conv",
                            "reform_by_vadds": "vector_adds"
                            }
    sch = sch_list[0]
    sqrt_flag = False
    placeholder_tensors = []  # to list placeholder type tensor
    compute_tensors = []  # to list compute type tensor
    compute_tensors_local = []
    batch_double = False
    double_once = 0
    in_addr_type = 0  # 0:DDR;1:L1
    out_addr_type = 0  # 0:DDR;1:L1
    l1_fusion_and_l1_size_0 = False

    def get_placeholder_tensor(tensor):
        """
        scan all the transient tensor during calculation
        tersor: target tensor which needs to find placeholder tensor
        """
        if tensor not in compute_tensors_local:
            compute_tensors_local.append(tensor)
        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        tensor_list = tensor.op.input_tensors
        for one_tensor in tensor_list:
            # check which tensor has not been checked
            if one_tensor not in compute_tensors_local:
                if isinstance((one_tensor.op), tvm.tensor.PlaceholderOp):
                    placeholder_tensors.append(one_tensor)
                else:
                    compute_tensors_local.append(one_tensor)
                    get_placeholder_tensor(one_tensor)
        return compute_tensors_local

    def _get_addr_type(tensor):
        addr_type = 0
        if 'addr_type' in tensor.op.attrs and \
                tensor.op.attrs['addr_type'].value == 1:
            addr_type = 1
        return addr_type

    def _get_l1_fusion_type(tensor):
        l1_fusion_type = -1
        if 'L1_fusion_type' in tensor.op.attrs:
            l1_fusion_type = tensor.op.attrs['L1_fusion_type'].value
        return l1_fusion_type

    def _get_l1_fusion_and_l1_size_0_flag(tensor_b, l1_fusion_type):
        trans_b = False
        if 'trans_b' in tensor_b.op.attrs:
            trans_b = tensor_b.op.attrs['trans_b'].value
        is_l1fusion = l1_fusion_type in (0, 1)
        size = get_soc_spec("L1_SIZE")
        if size == 0 and is_l1fusion:
            if trans_b:
                raise RuntimeError(
                    "If the size of L1 is zero, trans_b is not unexpected.")
            return True
        return False

    def _get_input_l1_paras(tensor):
        input_l1_flag = -1
        input_l1_size = None
        if 'L1_addr_flag' in tensor.op.attrs:
            input_l1_flag = tensor.op.attrs['L1_addr_flag'].value

        if input_l1_flag == 0:
            input_l1_size = -1
        elif input_l1_flag == 1:
            if 'L1_valid_size' in tensor.op.attrs:
                input_l1_size = tensor.op.attrs['L1_valid_size'].value
            else:
                input_l1_flag = -1
        else:
            pass

        return input_l1_flag, input_l1_size

    def _set_l1_fusion_workspace_tensor(input_l1_flag, tensor_a, tensor_a_l1_workspace):
        if input_l1_flag == 0:
            util.L1CommonParam.l1_fusion_tensors_map = {}
            util.L1CommonParam.l1_fusion_tensors_map[tensor_a] = tvm.var("dummy")
        elif input_l1_flag == 1:
            util.L1CommonParam.l1_fusion_tensors_map = {}
            util.L1CommonParam.l1_fusion_tensors_map[tensor_a] = tensor_a_l1_workspace
        else:
            pass

    def _set_l1_fusion_workspace_size(input_l1_flag, input_l1_size, tensor_a_l1_workspace):
        if input_l1_flag == 1 and input_l1_size > 0:
            sch[tensor_a_l1_workspace].set_storage_bound(input_l1_size)

    res_ori = res
    res = res[-1]

    out_addr_type = _get_addr_type(res)

    compute_tensors = get_placeholder_tensor(res)

    tensor_a_ub = match_and_get_tensor(compute_tensors, 'tensor_a_ub')
    tensor_a_l1 = match_and_get_tensor(compute_tensors, 'tensor_a_l1')
    tensor_b_ub = match_and_get_tensor(compute_tensors, 'tensor_b_ub')
    tensor_bias_ub = match_and_get_tensor(compute_tensors, 'tensor_bias_ub')
    tensor_a_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_a_ub_fract')
    tensor_b_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_b_ub_fract')
    tensor_c_ub_fract = match_and_get_tensor(compute_tensors,
                                             'tensor_c_ub_fract')
    tensor_c = match_and_get_tensor(compute_tensors, 'tensor_c')

    tensor_c_gm = match_and_get_tensor(compute_tensors, 'tensor_c_gm')
    with_transpose = hasattr(res, "matmul_with_transpose")
    format_out = get_output_format(tensor_c_gm)

    tensor_b_l1, compress_index = \
        get_weigths_and_compress_index(compute_tensors)

    tensor_c_ub = None
    tensor_sqrt = None
    dequant_nz = None
    dequant_nd = None
    dequant_nd_fract = False
    quant = None
    tensor_input_ub = None
    tensor_reform = None
    tensor_reform_by_vadds = None
    tensor_reform_by_vmuls = None
    dequant_fusion = False
    dequant_tensor = None
    quant_fusion = False
    requant_fusion = False
    requant_data_transfer = None
    requant_scale = None
    round_mode = "vector_conv"
    is_b_nz = False

    def __get_b_nz_flag(tensor):
        is_b_nz = False
        if tensor.op.name == "tensor_c_ub":
            is_b_nz = tensor.op.attrs['nz_b'].value
        return is_b_nz

    for i in compute_tensors:
        if 'tensor_c_ub' in i.op.name:
            tensor_c_ub = i
            is_b_nz = __get_b_nz_flag(i)
        if i.op.name == 'dequant':
            dequant_tensor = i
            dequant_fusion = True
        if 'dequant_sqrt' in i.op.name:
            tensor_sqrt = i
            sqrt_flag = True
        if 'dequant_NZ' in i.op.name:
            dequant_nz = i
        if 'dequant_ND' in i.op.name:
            dequant_nd = i
            dequant_nd_fract = True
        if i.op.tag == 'quant':
            quant = i
            quant_fusion = True
            round_mode = i.op.attrs['round_mode']
        if 'input_ub' in i.op.name:
            tensor_input_ub = i
        if i.op.tag == 'requant_scale' or i.op.tag == 'requant_vector':
            requant_fusion = True
            requant_scale = i
        if i.op.tag == 'requant_data_transfer':
            requant_data_transfer = i

    def get_gemv_gemv_flag(tensors):
        """
        get gemv and gevm info
        """
        gemv_flag = False
        gevm_flag = False
        for i in tensors:
            if i.op.tag == 'matmul_gemv':
                gemv_flag = True
            if i.op.tag == 'matmul_gevm':
                gevm_flag = True

        return gemv_flag, gevm_flag


    gemv_flag, gevm_flag = get_gemv_gemv_flag(compute_tensors)
    tensor_reform_by_vadds = match_and_get_tensor(compute_tensors, 'reform_by_vadds')
    tensor_reform_by_vmuls = match_and_get_tensor(compute_tensors, 'reform_by_vmuls')

    do_cache_write_flag = dequant_nd_fract and not quant_fusion
    if do_cache_write_flag:
        tensor_c_ub_fract = sch.cache_write(res, cce.scope_ubuf)

    matmul_end_tensor = tensor_c_gm
    if tensor_c_gm is None:
        matmul_end_tensor = tensor_c_ub

    matmul_tensors = []
    compute_tensors_local = []
    matmul_tensors = get_placeholder_tensor(matmul_end_tensor)

    matmul_dequant_tensor = []
    compute_tensors_local = []

    def _get_matmul_dequant_tensor():
        if quant is not None:
            matmul_dequant_tensor = get_placeholder_tensor(dequant_nz)
            return matmul_dequant_tensor
        return None

    matmul_dequant_tensor = _get_matmul_dequant_tensor()

    quantify_fusion = requant_fusion or dequant_fusion

    tensor_ele_map = []
    elemwise_tensors = []
    fusion_ele = False

    def _get_elewise_fusion_tensor():
        if quantify_fusion:
            return False

        if tensor_c_gm != res and tensor_c_gm is not None:
            for ten_in in compute_tensors:
                if ten_in == res:
                    continue
                if ten_in not in matmul_tensors and ten_in not in elemwise_tensors:
                    elemwise_tensors.append(ten_in)

            return True
        return False

    fusion_ele = _get_elewise_fusion_tensor()

    tensor_fusion_list = []

    def _get_quant_fusion_tensor():
        if not quant_fusion:
            return
        for ten_in in compute_tensors:
            if ten_in == res:
                continue
            if ten_in not in matmul_dequant_tensor and ten_in.op.name in emit_fusion_insn_map:
                tensor_fusion_list.append(ten_in)
    _get_quant_fusion_tensor()

    dequant_activation_tensor = []
    compute_tensors_local = []

    def _get_matmul_dequant_activation_tensor():
        if dequant_tensor is None:
            return None
        tensor_front_dequant = get_placeholder_tensor(dequant_tensor)
        for ten_in in compute_tensors:
            if ten_in not in tensor_front_dequant:
                dequant_activation_tensor.append(ten_in)
        if tensor_sqrt is not None:
            dequant_activation_tensor.remove(tensor_sqrt)
        for ten_quant in tensor_fusion_list:
            if ten_quant in dequant_activation_tensor:
                dequant_activation_tensor.remove(ten_quant)
        dequant_type_tensor = dequant_nz if dequant_nz is not None else dequant_nd
        if dequant_type_tensor in dequant_activation_tensor:
            dequant_activation_tensor.remove(dequant_type_tensor)
        if res in dequant_activation_tensor and quant is not None:
            dequant_activation_tensor.remove(res)
        return None
    _get_matmul_dequant_activation_tensor()
    _add_res_ub(dequant_activation_tensor, res, sch)
    header_ub_tensors = _get_header_tensor_in_dequant_ew_fusion(
        dequant_activation_tensor, placeholder_tensors, sch)

    def _get_reform_tensor(tensor_reform_by_vadds, tensor_reform_by_vmuls, requant_data_transfer):
        if tensor_reform_by_vadds is not None:
            return tensor_reform_by_vadds
        if tensor_reform_by_vmuls is not None:
            return tensor_reform_by_vmuls
        if requant_data_transfer is not None:
            return requant_data_transfer
        return None

    tensor_reform = _get_reform_tensor(tensor_reform_by_vadds,
                                       tensor_reform_by_vmuls,
                                       requant_data_transfer)

    reform_tensor_tag_list = ["reform_by_vadds",
                              "reform_by_vmuls",
                              "data_transfer"]

    fusion_list = []

    def _get_fusion_tensor():
        if tensor_c_gm != res and tensor_c_gm is not None:
            for ten_in in compute_tensors:
                if ten_in == res:
                    continue
                if ten_in not in matmul_tensors:
                    fusion_list.append(ten_in)

    _get_fusion_tensor()

    if tensor_a_ub is not None:
        tensor_a = tensor_a_ub.op.input_tensors[0]
    elif tensor_a_l1 is not None:
        tensor_a = tensor_a_l1.op.input_tensors[0]
    else:
        raise RuntimeError(
            "Lack of tensor_a_ub or tensor_a_l1.")

    in_addr_type = _get_addr_type(tensor_a)

    l1_fusion_type = _get_l1_fusion_type(tensor_a)

    input_l1_flag, input_l1_size = _get_input_l1_paras(tensor_a)

    tensor_a_shape = tensor_a.shape
    if tensor_b_ub is not None:
        tensor_b = tensor_b_ub.op.input_tensors[0]
    elif tensor_b_l1 is not None:
        if compress_index is not None:
            tensor_b = tensor_b_l1.op.input_tensors[1]
        else:
            tensor_b = tensor_b_l1.op.input_tensors[0]
    else:
        raise RuntimeError(
            "Lack of tensor_b_ub or tensor_b_l1.")
    tensor_b_shape = tensor_b.shape

    check_placeholders_shared(fusion_ele, tensor_a, tensor_b, res, matmul_tensors)

    is_with_bias = tensor_bias_ub is not None

    is_fractal_a = len(tensor_a_shape) == 4 or len(tensor_a_shape) == 5
    is_fractal_b = len(tensor_b_shape) == 4 or len(tensor_b_shape) == 5

    l1_fusion_and_l1_size_0 = \
        _get_l1_fusion_and_l1_size_0_flag(tensor_b, l1_fusion_type)

    if is_with_bias:
        tensor_c_add_bias = tensor_c_ub.op.input_tensors[0]
        tensor_bias_l0c = tensor_c_add_bias.op.input_tensors[0]

    tensor_a_l0a = tensor_c.op.input_tensors[0]
    tensor_b_l0b = tensor_c.op.input_tensors[1]

    def _get_tensor_a_l1_workspace(l1_fusion_and_l1_size_0):
        tensor_a_l1_workspace = None
        if input_l1_flag == 1:
            if tensor_a_ub is not None:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, cce.scope_cbuf_fusion, tensor_a_ub)
            elif tensor_a_l1 is not None and not l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, cce.scope_cbuf_fusion, tensor_a_l1)
            elif tensor_a_l0a is not None and l1_fusion_and_l1_size_0:
                tensor_a_l1_workspace = sch.cache_read(tensor_a, cce.scope_cbuf_fusion, tensor_a_l1)
        return tensor_a_l1_workspace

    tensor_a_l1_workspace = _get_tensor_a_l1_workspace(l1_fusion_and_l1_size_0)
    _set_l1_fusion_workspace_tensor(input_l1_flag, tensor_a, tensor_a_l1_workspace)
    _set_l1_fusion_workspace_size(input_l1_flag, input_l1_size, tensor_a_l1_workspace)

    if gemv_flag:
        tensor_a_l1 = tensor_b_l0b.op.input_tensors[0]
        tensor_b_l1 = tensor_a_l0a.op.input_tensors[0]
    else:
        tensor_a_l1 = tensor_a_l0a.op.input_tensors[0]
        tensor_b_l1 = tensor_b_l0b.op.input_tensors[0]

    def _fc_tensor_a_l1_inline():
        inline_flag = False
        if ((in_addr_type == 1 or input_l1_flag == 1) and is_fractal_a) or \
                l1_fusion_and_l1_size_0:
            sch[tensor_a_l1].compute_inline()
            inline_flag = True
        return inline_flag

    a_l1_inline_flag = _fc_tensor_a_l1_inline()

    def _fc_tensor_b_l1_inline():
        inline_flag = False
        if l1_fusion_and_l1_size_0 and is_fractal_b:
            sch[tensor_b_l1].compute_inline()
            inline_flag = True
        return inline_flag

    b_l1_inline_flag = _fc_tensor_b_l1_inline()

    if gevm_flag or gemv_flag:
        block_in = cce.BLOCK_VECTOR
        mad_pattern = cce.GEVM_MODE
    else:
        mad_pattern = cce.GEMM_MODE
        block_in = cce.BLOCK_IN

    l0_tensor_len_a = len(tensor_a_l0a.shape)
    l0_tensor_len_b = len(tensor_b_l0b.shape)
    tensor_len_a = len(tensor_a_l1.shape)
    tensor_len_b = len(tensor_b_l1.shape)
    tensor_len_c = len(tensor_c.shape)

    block_out = cce.BLOCK_OUT

    out_dtype = tensor_c.dtype

    def _get_block_reduce(out_dtype):
        if out_dtype in ("float16", "float32"):
            return cce.BLOCK_REDUCE
        return cce.BLOCK_REDUCE_INT8

    block_reduce = _get_block_reduce(out_dtype)

    # get matrix axis shapes
    m_shape = tensor_a_l0a.shape[l0_tensor_len_a - 4].value * block_in
    k_shape = tensor_a_l0a.shape[l0_tensor_len_a - 3].value * block_reduce
    n_shape = tensor_b_l0b.shape[l0_tensor_len_b - 3].value * block_out
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m", m_shape)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n", n_shape)

    core_inner_m = m_shape
    core_inner_n = n_shape
    n_nparts_mode = True
    m_factors, n_factors = get_perfect_core_num(m_shape, n_shape, k_shape, l1_fusion_type)

    shape_mkn_args = (m_shape, k_shape, n_shape)
    m_factors, n_factors = get_knowledge_core(shape_mkn_args, m_factors, n_factors)

    date_transfer_fusion = quant_fusion or requant_fusion
    # matmul + quant ub fusion, it need to ensure that the number of
    # fractal blocks processed by each core is even

    def _is_used_nparts_mode(date_transfer_fusion, n_factors):
        if not date_transfer_fusion:
            return True, n_factors

        if n_factors == 1:
            return True, n_factors

        fractal_n_number = (n_shape + block_out - 1) // block_out
        core_inner_n_num = (fractal_n_number + n_factors - 1) // n_factors

        if core_inner_n_num % 2 == 0:
            return True, n_factors

        factor = ceil(fractal_n_number // n_factors) + 1
        return False, factor

    # if the fract block processed in each core is not even,
    # the meaning of n_factors changes, this factor represents the number of
    # fractals processed by each core, not nparts split factor
    n_nparts_mode, n_factors = _is_used_nparts_mode(date_transfer_fusion,
                                                    n_factors)

    m_var = [m_shape, m_factors]
    n_var = [n_shape, n_factors]
    batch, core_inner_m, core_inner_n = get_batch_factors(
        tensor_a_shape, tensor_a_l0a, tensor_b_l0b, m_var, n_var, gemv_flag,
        n_nparts_mode)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("batch", batch)
    m_factors, n_factors = get_refresh_core_factors(m_factors, n_factors, batch)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_blk",
                                                      m_factors)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_blk",
                                                      n_factors)

    def _get_out_tensors_width(out_tensor):
        """
        get max width for tensors

        Parameters
        ----------
        out_tensor : tensor
            need to count all its input tensorss

        Return
        ------
            max width for tensors
        """
        # pylint: cell-var-from-loop
        in_out_tensor_map = {}
        _gen_in_out_tensor_map(out_tensor, in_out_tensor_map)
        stack = [out_tensor]
        width = len(stack)
        visited_list = []
        tmp_stack = stack
        matmul_end_tensor = tensor_c_gm.op.input_tensors[0]
        while tmp_stack:
            for tens in tmp_stack:
                if tens in in_out_tensor_map:
                    def calc_width_mid(width):
                        """
                        get mid tesnor width
                        """
                        all_out = True
                        for out_ten in in_out_tensor_map[tens]:  # pylint: disable=W0640
                            if out_ten not in visited_list:
                                all_out = False
                        if all_out and (tens not in visited_list):  # pylint: disable=W0640
                            visited_list.append(tens)  # pylint: disable=W0640
                            stack.remove(tens)  # pylint: disable=W0640
                            # the shape of deq_scale is very small
                            if tens.op.tag not in DEQ_SCALE_CHILD_LIST:
                                for in_ten in tens.op.input_tensors:  # pylint: disable=W0640
                                    if in_ten not in stack and \
                                            in_ten != matmul_end_tensor:
                                        stack.append(in_ten)
                            else:
                                stack.append(tens.op.input_tensors[0])
                            width_local = 0
                            cast_flag = False
                            for ele in stack:
                                width_local = width_local + DTYPE_WIDTH_MAP[ele.dtype]
                                if DTYPE_WIDTH_MAP[ele.dtype] == 2:
                                    cast_flag = True
                            if width_local == 2 and cast_flag:
                                width_local = 3
                            if width_local > width:
                                width = width_local
                        return width

                    width = calc_width_mid(width)

                else:
                    def calc_width_tail(width):
                        # pylint: cell-var-from-loop
                        visited_list.append(tens)  # pylint: disable=W0640
                        stack.remove(tens)  # pylint: disable=W0640
                        for in_ten in tens.op.input_tensors:  # pylint: disable=W0640
                            if in_ten not in stack and in_ten != matmul_end_tensor:
                                stack.append(in_ten)
                        width_local = 0
                        cast_flag = False
                        for ele in stack:
                            width_local = width_local + DTYPE_WIDTH_MAP[ele.dtype]
                            if DTYPE_WIDTH_MAP[ele.dtype] == 2:
                                cast_flag = True
                        if width_local == 2 and cast_flag:
                            width_local = 3
                        if width_local > width:
                            width = width_local
                        return width

                    width = calc_width_tail(width)

            tmp_stack = []
            for ele in stack:
                tmp_stack.append(ele)
        return width

    l0a_byte = int(DTYPE_WIDTH_MAP[tensor_a_l0a.dtype] * 2)
    l0b_byte = int(DTYPE_WIDTH_MAP[tensor_b_l0b.dtype] * 2)
    l0c_byte = int(DTYPE_WIDTH_MAP[tensor_c.dtype] * 2)
    l1a_byte = int(DTYPE_WIDTH_MAP[tensor_a_l1.dtype] * 2)
    l1b_byte = int(DTYPE_WIDTH_MAP[tensor_b_l1.dtype] * 2)

    def get_scope_byte_size(tensor_ub, tensor_ub_fract):
        """
        get unit byte size for buffer scope
        """
        # Calculating tiling para need a_ub info
        ub_byte = 0
        if tensor_ub is not None:
            ub_byte = int(DTYPE_WIDTH_MAP[tensor_ub.dtype] * 2)
            if tensor_ub_fract is not None:
                ub_byte = ub_byte * 2
        return ub_byte

    a_ub_byte = get_scope_byte_size(tensor_a_ub, tensor_a_ub_fract)
    b_ub_byte = get_scope_byte_size(tensor_b_ub, tensor_b_ub_fract)
    ub_res_byte = get_scope_byte_size(tensor_c_ub, tensor_c_ub_fract)
    ub_res_byte = _get_ub_res_byte(_get_out_tensors_width,
                                   dequant_activation_tensor, fusion_ele, res,
                                   ub_res_byte)

    if gemv_flag:
        tmp = a_ub_byte  # pylint: disable=R1712
        a_ub_byte = b_ub_byte
        b_ub_byte = tmp

    ub_reserve_buff = 0
    if dequant_fusion or tensor_c_ub.op.attrs['scale_drq'].value == "ENABLE":
        # quant parameter is fixed float16, it's 2 bytes
        # just support scalar now, not support vector yet
        ub_reserve_buff = cce.BLOCK_OUT * 2

    def _is_need_n_cut_even(date_transfer_fusion, core_inner_n):
        if not date_transfer_fusion:
            return False
        if core_inner_n == 16:
            return False
        return True

    n_cut_even = _is_need_n_cut_even(date_transfer_fusion, core_inner_n)

    get_tiling_shape = tvm.get_global_func("cce.matmul_tiling_gen")
    tiling_shape = get_tiling_shape(core_inner_m, k_shape, core_inner_n,
                                    a_ub_byte,
                                    b_ub_byte, l1a_byte,
                                    l1b_byte, l0a_byte, l0b_byte, l0c_byte,
                                    ub_res_byte, ub_reserve_buff,
                                    n_cut_even, is_b_nz)

    def get_tensor_trans_info(tensor_in):
        """
        get tensor transform info
        """
        b_trans = None
        if 'trans_b' in tensor_in.op.attrs:
            b_trans = tensor_in.op.attrs['trans_b'].value
        return b_trans

    b_trans = get_tensor_trans_info(tensor_b)
    shape_tiling_args = (m_shape, k_shape, n_shape, b_trans, ub_res_byte)
    tiling_shape = get_knowledge_tiling(shape_tiling_args, is_b_nz, tiling_shape)

    if tiling_shape.find('_') == -1:
        raise RuntimeError(tiling_shape)

    tiled_shape = tiling_shape.split('_')
    m_l1_shape = int(tiled_shape[0])
    k_l1_shape = int(tiled_shape[1])
    n_l1_shape = int(tiled_shape[2])

    m_l0_shape = int(tiled_shape[3])
    k_l0_shape = int(tiled_shape[4])
    n_l0_shape = int(tiled_shape[5])

    src_shape = [m_shape, k_shape, n_shape]
    m_l0_shape, k_l0_shape, n_l0_shape = get_special_l0_factor(
        src_shape, m_l0_shape, k_l0_shape, n_l0_shape)

    l0c_enable_db = True
    l0c_size = get_soc_spec("L0C_SIZE")
    if m_l0_shape * n_l0_shape * l0c_byte * DOUBLE_VALUE > l0c_size:
        l0c_enable_db = False

    m_l1_shape = m_l0_shape
    k_l1_shape = k_l0_shape
    n_l1_shape = n_l0_shape

    tensor_a_reuse, \
        tensor_b_reuse, \
        batch_double, \
        double_once = get_tensor_reuse(batch, core_inner_m, k_shape, core_inner_n,
                                       m_l1_shape, k_l1_shape, n_l1_shape,
                                       l1a_byte)

    def _get_m_l1_double_buffer(m_l1_shape, m_shape, k_l1_shape, k_shape):
        if m_l1_shape == m_shape and k_l1_shape == k_shape:
            return 1
        return 2

    m_l1_double_buffer = _get_m_l1_double_buffer(m_l1_shape, m_shape,
                                                 k_l1_shape, k_shape)

    def _get_n_l1_double_buffer(n_l1_shape, n_shape, k_l1_shape, k_shape):
        if n_l1_shape == n_shape and k_l1_shape == k_shape:
            return 1
        return 2

    n_l1_double_buffer = _get_n_l1_double_buffer(n_l1_shape, n_shape,
                                                 k_l1_shape, k_shape)

    l1_tiling_shape = (m_l1_shape, k_l1_shape, n_l1_shape)
    l0_tiling_shape = (m_l0_shape, k_l0_shape, n_l0_shape)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_m_split",
                                                      m_l0_shape)
    cce_emitinsn_params.cceEmitParamsIns.insert_param("matmul_n_split",
                                                      n_l0_shape)

    # tiling param check
    gemm_para_check([m_shape, k_shape, n_shape], l1_tiling_shape,
                    l0_tiling_shape)

    # compute L1->L0 tiling params
    m_l0_tile = m_l0_shape // block_in
    k_l0_tile = k_l0_shape // block_reduce
    n_l0_tile = n_l0_shape // block_out

    # compute GM to L1 tiling params
    m_l1_tile = (m_l1_shape // block_in)
    k_l1_tile = (k_l1_shape // block_reduce)
    n_l1_tile = (n_l1_shape // block_out)

    def _date_transfer_tiling_check(date_transfer_fusion, n_l1_tile, n_l0_tile):
        if not date_transfer_fusion:
            return
        if not n_cut_even:
            return
        if n_l1_tile % 2 != 0:
            raise RuntimeError("L1 n tiling factor should be even number, "
                               "actual factor equal %d " % (n_l1_tile,))
        if n_l0_tile % 2 != 0:
            raise RuntimeError("L0 n tiling factor should be even number, "
                               "actual factor equal %d " % (n_l0_tile,))

    _date_transfer_tiling_check(date_transfer_fusion, n_l1_tile, n_l0_tile)

    fusion_n_l1_tile = n_l1_tile
    fusion_n_l0_tile = n_l0_tile
    if date_transfer_fusion and n_cut_even:
        fusion_n_l0_tile = n_l0_tile // 2
        fusion_n_l1_tile = n_l1_tile // 2

    def _update_n_factor(n_nparts_mode, n_factors):
        if n_nparts_mode:
            return n_factors

        return n_factors // 2

    n_factors = _update_n_factor(n_nparts_mode, n_factors)

    repeat_a = False
    repeat_b = False

    def set_tensor_scope(one_tensor, scope_type):
        """
        set one tensors buffer scope
        """
        if one_tensor is not None:
            sch[one_tensor].set_scope(scope_type)

    def set_scope_buffer_type(header_tensors):
        """
        set all tensors buffer scope
        """
        if in_addr_type == 1:
            set_tensor_scope(tensor_a, cce.scope_cbuf_fusion)
        set_tensor_scope(tensor_a_l1, cce.scope_cbuf)
        set_tensor_scope(tensor_b_l1, cce.scope_cbuf)

        def __res_set_scope():
            if out_addr_type == 1:
                set_tensor_scope(res, cce.scope_cbuf_fusion)

        __res_set_scope()

        set_tensor_scope(tensor_a_ub, cce.scope_ubuf)
        set_tensor_scope(tensor_b_ub, cce.scope_ubuf)
        set_tensor_scope(tensor_a_ub_fract, cce.scope_ubuf)
        set_tensor_scope(tensor_b_ub_fract, cce.scope_ubuf)
        set_tensor_scope(tensor_c_ub_fract, cce.scope_ubuf)

        set_tensor_scope(tensor_a_l0a, cce.scope_ca)
        set_tensor_scope(tensor_b_l0b, cce.scope_cb)
        set_tensor_scope(tensor_c, cce.scope_cc)
        set_tensor_scope(tensor_c_ub, cce.scope_ubuf)

        if dequant_fusion:
            if sqrt_flag:
                sch[tensor_sqrt].set_scope(cce.scope_ubuf)
            for tensor in dequant_activation_tensor:
                sch[tensor].set_scope(cce.scope_ubuf)
            for tensor_list in tensor_fusion_list:
                sch[tensor_list].set_scope(cce.scope_ubuf)
        for tensor in fusion_list:
            sch[tensor].set_scope(cce.scope_ubuf)
        if is_with_bias:
            sch[tensor_bias_ub].set_scope(cce.scope_ubuf)
            sch[tensor_bias_l0c].set_scope(cce.scope_cc)
            sch[tensor_c_add_bias].set_scope(cce.scope_cc)

        gm_ub = None
        ele_header_ub_tensors = []
        if not fusion_ele:
            return gm_ub, ele_header_ub_tensors, dict()
        in_out_tensor_map = {}
        _gen_in_out_tensor_map(res, in_out_tensor_map)
        # multi output fusion with elementwise
        if fusion_ele and tensor_c_gm in res_ori:
            gm_ub = sch.cache_read(tensor_c_gm, cce.scope_ubuf,
                                   in_out_tensor_map[tensor_c_gm])

        tensor_ele_ub = []
        header_tensors = list(set(header_tensors))
        for ten_i in header_tensors:
            if in_out_tensor_map[ten_i][0] not in matmul_tensors:
                ele_ub = sch.cache_read(ten_i, cce.scope_ubuf,
                                        in_out_tensor_map[ten_i])
                tensor_ele_ub.append(ele_ub)
                ele_header_ub_tensors.append(ele_ub)

        axpy_2_parent = _get_elewise_ub_tensors(tensor_ele_ub)


        elemwise_tensors.clear()
        for ten_i in tensor_ele_ub:
            elemwise_tensors.append(ten_i)

        return gm_ub, ele_header_ub_tensors, axpy_2_parent

    def _get_elewise_ub_tensors(tensor_ele_ub):
        """
        get axpy_ub to axpy_parents[1]_ub dict, in order to set reused_by.
        """
        axpy_and_parent = list()
        for ten_i in elemwise_tensors:
            if "elewise_binary_scalar_axpy" in ten_i.op.tag:
                axpy_and_parent.append([ten_i, ten_i.op.input_tensors[1]])

        for ten_i in elemwise_tensors:
            ele_ub = sch.cache_write(ten_i, cce.scope_ubuf)
            for index, (axpy, parent) in enumerate(axpy_and_parent):
                if ten_i == axpy:
                    axpy_and_parent[index][0] = ele_ub
                if ten_i == parent:
                    axpy_and_parent[index][1] = ele_ub
            tensor_ele_ub.append(ele_ub)
            sch[ten_i].compute_inline()
        if axpy_and_parent:
            return dict(axpy_and_parent)
        return dict()

    gm_ub, ele_header_ub_tensors, axpy_2_parent = \
        set_scope_buffer_type(placeholder_tensors)

    def set_tensor_buffer_align(gm_pattern):
        """
        set tensor_c and tensor_c_add_bias buffer align
        """
        unchanged = 1
        # gevm single batch
        if len(tensor_c.shape) == 4:
            sch[tensor_c].buffer_align((unchanged, unchanged),
                                       (unchanged, unchanged),
                                       (unchanged, block_out),
                                       (unchanged, block_out),
                                       (unchanged, block_reduce),
                                       (unchanged, block_reduce))
            if is_with_bias:
                sch[tensor_c_add_bias].buffer_align((unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, block_out),
                                                    (unchanged, block_out))
        else:
            # gevm multi batch
            sch[tensor_c].buffer_align((unchanged, unchanged),
                                       (unchanged, unchanged),
                                       (unchanged, unchanged),
                                       (unchanged, block_out),
                                       (unchanged, block_out),
                                       (unchanged, block_reduce),
                                       (unchanged, block_reduce))
            if is_with_bias:
                sch[tensor_c_add_bias].buffer_align((unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, unchanged),
                                                    (unchanged, block_out),
                                                    (unchanged, block_out))
        # single batch
        if block_in != cce.BLOCK_VECTOR:
            if len(tensor_c.shape) == 4:
                if tensor_a_ub is not None:
                    sch[tensor_a_ub].buffer_align((unchanged, block_in),
                                                  (unchanged, block_reduce))
                sch[tensor_a_l0a].buffer_align((unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, block_in),
                                               (unchanged, block_reduce))
                sch[tensor_c_ub].buffer_align((unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, block_in),
                                              (unchanged, block_out))
            else:
                # multi batch
                if tensor_a_ub is not None:
                    sch[tensor_a_ub].buffer_align((unchanged, unchanged),
                                                  (unchanged, block_in),
                                                  (unchanged, block_reduce))
                sch[tensor_a_l0a].buffer_align((unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, unchanged),
                                               (unchanged, block_in),
                                               (unchanged, block_reduce))
                sch[tensor_c_ub].buffer_align((unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, unchanged),
                                              (unchanged, block_in),
                                              (unchanged, block_out))

    # set tensor buffer align
    set_tensor_buffer_align(mad_pattern)

    def _set_requant_transfer_buffer_align(requant_fusion,
                                           requant_data_transfer):
        if not requant_fusion:
            return

        unchanged = 1

        sch[requant_data_transfer].buffer_align((unchanged, unchanged),
                                                (unchanged, unchanged),
                                                (unchanged, 16),
                                                (unchanged, 16))
        return

    _set_requant_transfer_buffer_align(requant_fusion,
                                       requant_data_transfer)

    is_l1fusion = l1_fusion_type in (0, 1)
    core_num = get_l1fusion_device_core_num(is_l1fusion)
    core_thres = core_num // 2

    batch_outer = None
    # for multi batch use multi block
    if batch > 1:
        batch_factor = batch
        if batch > core_num:
            batch_factor = core_num

        batch_outer, batch_inner = sch[res].split(res.op.axis[0],
                                                  nparts=batch_factor)
        thread_block = tvm.thread_axis("blockIdx.x")
        sch[res].bind(batch_outer, thread_block)

    emit_insn_map = {"elewise_single_cast": "vector_conv",
                     "elewise_single_round_d": "vector_conv_round",
                     "elewise_single_VS_max": "vector_maxs",
                     "elewise_single_VS_min": "vector_mins",
                     "elewise_single_ceil": "elewise_single_ceil",
                     "elewise_single_log": "vector_ln",
                     "elewise_single_exp": "vector_exp",
                     "elewise_single_relu": "vector_relu",
                     "elewise_single_abs": "vector_abs",
                     "elewise_single_not": "vector_not",
                     "elewise_single_sqrt": "vector_sqrt",
                     "elewise_single_rsqrt": "vector_rsqrt",
                     "elewise_binary_mul": "vector_mul",
                     "elewise_single_rec": "vector_rec",
                     "elewise_single_VS_mul": "vector_muls",
                     "elewise_binary_div": "vector_div",
                     "elewise_binary_sub": "vector_sub",
                     "elewise_binary_add": "vector_add",
                     "elewise_single_VS_add": "vector_adds",
                     "elewise_binary_min": "vector_min",
                     "elewise_binary_max": "vector_max",
                     "elewise_binary_vcmpv_gt": "vector_gt",
                     "elewise_binary_vcmpv_ge": "vector_ge",
                     "elewise_binary_vcmpv_lt": "vector_lt",
                     "elewise_binary_vcmpv_le": "vector_le",
                     "elewise_binary_vcmpv_eq": "vector_eq",
                     "elewise_binary_vcmpv_ne": "vector_ne",
                     "elewise_binary_cmpsel_gt": "vector_select_gt",
                     "elewise_binary_cmpsel_ge": "vector_select_ge",
                     "elewise_binary_cmpsel_lt": "vector_select_lt",
                     "elewise_binary_cmpsel_le": "vector_select_le",
                     "elewise_binary_cmpsel_eq": "vector_select_eq",
                     "elewise_binary_cmpsel_ne": "vector_select_ne",
                     "elewise_binary_or": "vector_or",
                     "elewise_binary_and": "vector_and",
                     "elewise_multiple_mla": "vector_multiple",
                     "elewise_multiple_madd": "vector_multiple",
                     "elewise_multiple_maddrelu": "vector_multiple",
                     "elewise_multiple_sel": "vector_select_bool",
                     "elewise_binary_scalar_axpy": "vector_axpy",
                     "elewise_binary_cmpsel": "vector_cmpsel",
                     "broadcast": "vector_dup",
                     "emit_insn_elewise_multiple_sel": "elewise_multiple_sel",
                     "emit_insn_elewise_binary_cmp": "elewise_binary_cmp"
                     }

    requant_fusion_insn_map = {"tensor_c_gm": "phony_insn",
                               "tensor_c_ub": "phony_insn",
                               "s32_to_s8": "dma_copy",
                               "data_transfer": "dma_copy"}

    def _emit_requant_fusion_insn(tensor_reform):
        if tensor_reform is None:
            return
        insn = requant_fusion_insn_map.get(tensor_reform.op.name)
        sch[tensor_reform].emit_insn(tensor_reform.op.axis[2], insn)
        return

    def _update_compute_at_tensor_c_ub(tensor_c_ub):
        quant_fusion = requant_fusion or dequant_fusion
        if not quant_fusion:
            return tensor_c_ub
        return None

    def _is_multicore_bind_naxis(res, format_out, tensor_len_c,
                                 n_l1_tile, block_out):
        if format_out != "FRACTAL_NZ":
            res_len = len(res.shape)
            if res.shape[res_len - 1].value > n_l1_tile * block_out:
                return True
        else:
            n_block_cnt = (res.shape[tensor_len_c - 4].value +
                           n_l1_tile - 1) // n_l1_tile
            if n_block_cnt > 1:
                return True
        return False

    def _is_multicore_bind_maxis(res, format_out, tensor_len_c,
                                 m_l1_tile, block_in):
        if format_out != "FRACTAL_NZ":
            res_len = len(res.shape)
            if res.shape[res_len - 2].value > m_l1_tile * block_in:
                return True
        else:
            m_block_cnt = (res.shape[tensor_len_c - 3].value +
                           m_l1_tile - 1) // m_l1_tile
            if m_block_cnt > 1:
                return True
        return False

    def _quantify_fusion_entry(axis_block):
        if not quantify_fusion:
            return

        if requant_fusion:
            _requant_fusion_proc()

        if dequant_fusion:
            _dequant_fusion_proc(axis_block)

        if quant_fusion:
            _quant_fusion_proc()

        reform_fusion = quant_fusion or requant_fusion
        if reform_fusion:
            reform_c_outer, reform_c_inner = sch[tensor_reform].split(
                tensor_reform.op.axis[tensor_len_c - 1], factor=16)
            sch[tensor_reform].reorder(
                tensor_reform.op.axis[tensor_len_c - 4],
                tensor_reform.op.axis[tensor_len_c - 3],
                reform_c_outer,
                tensor_reform.op.axis[tensor_len_c - 2],
                reform_c_inner)
        return

    def _requant_fusion_proc():
        tensor_drq = requant_scale.op.input_tensors[1]
        tensor_drq_ub = sch.cache_read(tensor_drq, cce.scope_ubuf,
                                       [requant_scale])
        sch[tensor_drq_ub].emit_insn(tensor_drq_ub.op.axis[0],
                                     'dma_copy')

        sch[tensor_c_ub].compute_inline()
        sch[tensor_c_gm].compute_inline()
        sch[requant_scale].compute_inline()

    def _dequant_fusion_proc(axis_block):
        tensor_drq = dequant_tensor.op.input_tensors[1]
        if sqrt_flag:
            c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf,
                                  [dequant_tensor, tensor_sqrt])
        else:
            c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf,
                                  [dequant_tensor])
        sch[c_ub].emit_insn(c_ub.op.axis[0], 'dma_copy')
        if axis_block is not None:
            sch[c_ub].compute_at(sch[res], axis_block)

        dequant_emit_axis, deq_scale_mode = (1, "vector") \
            if "vector" in dequant_tensor.op.tag else (0, "scalar")
        if pver().is_ng1_version() or is_lhisi_cs_version():
            sch[dequant_tensor].emit_insn(
                dequant_tensor.op.axis[dequant_emit_axis], 'dma_copy')
        else:
            sch[dequant_tensor].pragma(
                dequant_tensor.op.axis[dequant_emit_axis], 'deq_scale',
                deq_scale_mode)
            if sqrt_flag:
                sch[tensor_sqrt].emit_insn(tensor_sqrt.op.axis[0],
                                           'vector_auto')
        sch[tensor_c_ub].compute_inline()
        sch[tensor_c_gm].compute_inline()
        _compute_inline_dequant_output()

    def _compute_inline_dequant_output():
        """
        compute inline dequant output tensor when dequant is not the last op.
        """
        if dequant_nz is not None and res != dequant_nz:
            sch[dequant_nz].compute_inline()
        if dequant_nd is not None and res != dequant_nd:
            sch[dequant_nd].compute_inline()


    def set_ub_inline(sche, tensor_ub, inline_flag):
        """
        need to compute_inline ub tensor while gevm and gemv for performance
        """
        if tensor_ub is None:
            return False

        length = len(tensor_ub.shape)
        if length in (0, 1):
            raise RuntimeError("tensor ub shape length should be larger than 1.", tensor_ub)
        if (tensor_ub.shape[length - 1].value == 1 or \
                tensor_ub.shape[length - 2].value == 1) and inline_flag:
            sche[tensor_ub].compute_inline()
            return False

        return True


    def _quant_fusion_proc():
        sch[tensor_input_ub].compute_inline()

    def _round_emit_insn(round_mode):
        """
        Obtains the conv instruction by the round mode attr

        Parameters
        ----------
        round_mode: the attr of round mode

        Returns
        -------
        instruction
        """
        if pver().is_mini_version():
            emit_insn_str = "vector_conv"
        else:
            if round_mode == "Round":
                emit_insn_str = "vector_conv_round"
            elif round_mode == "Ceil":
                emit_insn_str = "vector_conv_ceil"
            elif round_mode == "Floor":
                emit_insn_str = "vector_conv_floor"
            elif round_mode == "Trunc":
                emit_insn_str = "vector_conv_trunc"
            else:
                emit_insn_str = "vector_conv"
        return emit_insn_str

    def split_c_ub_tensor(c_ub_tensor, block_value, factor_value, tile_value):
        """
        split c ub tensor and return emit insn axis
        """
        if c_ub_tensor is None:
            return None

        block_in, block_out = block_value
        m_factor, n_factor = factor_value
        m_tile, n_tile = tile_value

        ten_len = len(c_ub_tensor.shape)

        m_outer, m_inner = sch[c_ub_tensor].split(
            c_ub_tensor.op.axis[ten_len - 2], factor=block_in)
        n_outer, n_inner = sch[c_ub_tensor].split(
            c_ub_tensor.op.axis[ten_len - 1], factor=block_out)

        m_outer_group = sch[c_ub_tensor].split(m_outer, factor=m_factor)
        n_outer_group = sch[c_ub_tensor].split(n_outer, factor=n_factor)

        l0_m_outer, l0_m_inner = sch[c_ub_tensor].split(
            m_outer_group[1], factor=m_tile)
        l0_n_outer, l0_n_inner = sch[c_ub_tensor].split(
            n_outer_group[1], factor=n_tile)
        sch[c_ub_tensor].reorder(m_outer_group[0], n_outer_group[0],
                                 l0_m_outer, l0_n_outer,
                                 l0_n_inner, l0_m_inner,
                                 m_inner, n_inner)
        return l0_n_inner

    def get_block_split_factor(tensor_out, n_nparts_mode, m_factors, n_factors):
        if len(tensor_out.shape) < 4:
            raise RuntimeError(
                "res shape error, should be >= 4, curr is ", tensor_out.shape)
        m_factor = (tensor_out.shape[-3].value + m_factors - 1) // m_factors
        m_cnt = m_factors
        n_cnt = 0
        if n_nparts_mode:
            n_factor = (tensor_out.shape[-4].value + n_factors - 1) // n_factors
            n_cnt = n_factors
        else:
            n_factor = n_factors
            n_cnt = (tensor_out.shape[-4].value + n_factor - 1) // n_factor

        return m_factor, n_factor, m_cnt, n_cnt

    def schedule_l1_mkn_l0_k_tiling():
        """
        CUT73 schedule method
        """

        # pylint: disable=too-many-locals, too-many-branches, too-many-statements
        # tiling L1 and L0 on tensor_c
        l1_n_outer, l1_n_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 4], factor=n_l1_tile)
        l1_m_outer, l1_m_inner = sch[tensor_c].split(
            tensor_c.op.axis[tensor_len_c - 3], factor=m_l1_tile)
        l1_k_outer, l1_k_inner = sch[tensor_c].split(tensor_c.op.reduce_axis[0],
                                                     factor=k_l1_tile)

        l0_n_outer, l0_n_inner = sch[tensor_c].split(l1_n_inner,
                                                     factor=n_l0_tile)
        l0_m_outer, l0_m_inner = sch[tensor_c].split(l1_m_inner,
                                                     factor=m_l0_tile)
        l0_k_outer, l0_k_inner = sch[tensor_c].split(l1_k_inner,
                                                     factor=k_l0_tile)

        # |---gm_n---|     |--------gm_n-------|
        # nGM, mGM, kGM, n_l1_tile, m_l1_tile, k_l1_tile, n_l0_tile,
        # |------------------Nz--------------------|
        # m_l0_tile, mBurstSize, nBurstSize, k_l0_tile, kBurstSize
        # 0, 1, 2
        # 0 pre the k is too large  A and B both cant reuse
        # 1 pre the tensor could reuse by just K axis
        # 2 pre the tensor could reuse by any th
        tensor_a_reuse_local = tensor_a_reuse
        tensor_b_reuse_local = tensor_b_reuse
        tensor_c_l1_reuse_axis_outter = l1_n_outer
        tensor_c_l1_reuse_axis_inner = l1_m_outer
        tensor_c_l1_reuse_axis_outter, \
            tensor_c_l1_reuse_axis_inner = get_tensor_c_axis(is_fractal_a, is_fractal_b,
                                                             tensor_a_reuse_local,
                                                             tensor_b_reuse_local,
                                                             l1_n_outer, l1_m_outer)

        sch[tensor_c].reorder(tensor_c_l1_reuse_axis_outter,
                              tensor_c_l1_reuse_axis_inner,
                              l1_k_outer,
                              l0_m_outer, l0_n_outer, l0_k_outer,
                              l0_n_inner, l0_m_inner,
                              tensor_c.op.axis[tensor_len_c - 2],
                              tensor_c.op.axis[tensor_len_c - 1],
                              l0_k_inner, tensor_c.op.reduce_axis[1])

        gevm_flag = tensor_a_l1.shape[-1] == 1 or tensor_a_l1.shape[-2] == 1
        a_ub_exist = set_ub_inline(sch, tensor_a_ub, gevm_flag)
        a_ub_fract_exist = set_ub_inline(sch, tensor_a_ub_fract, gevm_flag)
        b_ub_exist = set_ub_inline(sch, tensor_b_ub, False)
        b_ub_fract_exist = set_ub_inline(sch, tensor_b_ub_fract, False)

        def _do_tensor_ub():

            if a_ub_exist:
                sch[tensor_a_ub].compute_at(sch[tensor_c], l1_k_outer)
            if b_ub_exist:
                sch[tensor_b_ub].compute_at(sch[tensor_c], l1_k_outer)
            if a_ub_fract_exist:
                sch[tensor_a_ub_fract].compute_at(sch[tensor_c], l1_k_outer)
            if b_ub_fract_exist:
                sch[tensor_b_ub_fract].compute_at(sch[tensor_c], l1_k_outer)

        _do_tensor_ub()

        sch[tensor_a_l0a].compute_at(sch[tensor_c], l0_k_outer)
        sch[tensor_b_l0b].compute_at(sch[tensor_c], l0_k_outer)

        # tiling C
        axis_block = batch_outer
        res_ub = None
        gm_instn = None
        index_at_axis = None

        if is_fractal_a and is_fractal_b:
            def sche_l1_mkn_l0_k_frac_frac():
                """
                fractail schedule config
                """
                axis_block = batch_outer
                res_ub = None
                if fusion_ele:
                    if gm_ub is None:
                        sch[tensor_c_gm].compute_inline()
                    res_ub = sch.cache_write(res, cce.scope_ubuf)
                    elemwise_tensors.append(res_ub)

                m_factor, n_factor, m_part_cnt, n_part_cnt = get_block_split_factor(
                    res, n_nparts_mode, m_factors, n_factors)

                m_outer_group = sch[res].split(
                    res.op.axis[tensor_len_c - 3], factor=m_factor)
                n_outer_group = sch[res].split(
                    res.op.axis[tensor_len_c - 4], factor=n_factor)

                n_outer, n_inner = sch[res].split(n_outer_group[1],
                                                  factor=fusion_n_l1_tile)
                m_outer, m_inner = sch[res].split(m_outer_group[1],
                                                  factor=m_l1_tile)
                index_at_axis = m_outer if m_l1_tile * 2 < m_factor else -1

                tensor_a_reuse_local = tensor_a_reuse
                tensor_b_reuse_local = tensor_b_reuse
                l1_reuse_axis_outter = n_outer
                l1_reuse_axis_inner = m_outer

                l1_reuse_axis_outter, \
                    l1_reuse_axis_inner = get_res_axis(tensor_a_reuse_local,
                                                       tensor_b_reuse_local,
                                                       m_outer, n_outer)

                fuse_list = []
                if m_part_cnt > n_part_cnt:
                    fuse_list.append(n_outer_group[0])
                    fuse_list.append(m_outer_group[0])
                else:
                    fuse_list.append(m_outer_group[0])
                    fuse_list.append(n_outer_group[0])

                sch[res].reorder(*fuse_list,
                                 l1_reuse_axis_outter, l1_reuse_axis_inner,
                                 n_inner, m_inner,
                                 res.op.axis[tensor_len_c - 2],
                                 res.op.axis[tensor_len_c - 1])

                if axis_block is None:
                    axis_block = sch[res].fuse(*fuse_list)
                    thread_block = tvm.thread_axis("blockIdx.x")
                    sch[res].bind(axis_block, thread_block)

                overload_flag = False
                if n_part_cnt > 1:
                    overload_flag = True

                gm_n_inner = None
                if gm_ub is not None:
                    gm_n_outer, gm_n_inner = sch[tensor_c_gm].split(
                        tensor_c_gm.op.axis[tensor_len_c - 4], factor=n_l0_tile)
                    gm_m_outer, gm_m_inner = sch[tensor_c_gm].split(
                        tensor_c_gm.op.axis[tensor_len_c - 3], factor=m_l0_tile)

                    sch[tensor_c_gm].reorder(gm_n_outer, gm_m_outer,
                                             gm_n_inner, gm_m_inner,
                                             tensor_c_gm.op.axis[tensor_len_c - 2],
                                             tensor_c_gm.op.axis[tensor_len_c - 1])

                c_at_axis = l1_reuse_axis_inner
                c_ub_at_axis = l1_reuse_axis_inner
                res_insn_axis = n_inner

                sch[tensor_c].compute_at(sch[res], c_at_axis)

                if not a_l1_inline_flag:
                    sch[tensor_a_l1].compute_at(sch[tensor_c], l1_k_outer)
                allocate_axis(sch, batch_double, double_once,
                              tensor_a_reuse_local,
                              tensor_b_reuse_local,
                              tensor_a_l1, tensor_b_l1,
                              tensor_c, res,
                              n_outer, m_outer,
                              l1_k_outer, in_addr_type,
                              input_l1_flag, l1_fusion_and_l1_size_0, gemv_flag)

                def _do_frac_bias_compurt_at():
                    if is_with_bias:
                        sch[tensor_bias_ub].compute_at(sch[res], c_at_axis)
                        sch[tensor_bias_l0c].compute_at(sch[res], c_at_axis)
                        sch[tensor_c_add_bias].compute_at(sch[res], c_at_axis)
                        if not dequant_fusion:
                            if l0c_enable_db:
                                sch[tensor_bias_l0c].preload()
                                sch[tensor_bias_l0c].double_buffer()
                            sch[tensor_bias_ub].preload()
                            sch[tensor_bias_ub].double_buffer()
                _do_frac_bias_compurt_at()

                def _do_allocate_at(axis_nmk):
                    a_overload = axis_nmk
                    if repeat_a and not a_l1_inline_flag:
                        a_overload = False
                        sch[tensor_a_l1].allocate_at(
                            sch[tensor_c], l1_m_outer, run_once_axes=[n_outer])
                    if repeat_b and not b_l1_inline_flag:
                        sch[tensor_b_l1].allocate_at(
                            sch[tensor_c], l1_n_outer, run_once_axes=[m_outer])
                    return a_overload

                axis_nmk = l1_reuse_axis_outter == n_outer
                a_overload = _do_allocate_at(axis_nmk)
                overload_flag = overload_flag or a_overload

                def _do_fusion_compute_at(tensor_c_ub):
                    if tensor_c_ub is not None:
                        sch[tensor_c_ub].compute_at(sch[res], c_ub_at_axis)
                    for ten_in in elemwise_tensors:
                        sch[ten_in].compute_at(sch[res], c_ub_at_axis)
                    if dequant_tensor is not None:
                        sch[dequant_tensor].compute_at(sch[res], c_ub_at_axis)
                    if dequant_fusion and sqrt_flag:
                        sch[tensor_sqrt].compute_at(sch[res], c_ub_at_axis)
                    if dequant_fusion:
                        for tensor in dequant_activation_tensor:
                            sch[tensor].compute_at(sch[res], c_ub_at_axis)
                    for tensor_list in tensor_fusion_list:
                        sch[tensor_list].compute_at(sch[res], c_ub_at_axis)
                    if requant_data_transfer is not None:
                        sch[requant_data_transfer].compute_at(sch[res],
                                                              c_ub_at_axis)

                update_tensor_c_ub = \
                    _update_compute_at_tensor_c_ub(tensor_c_ub)
                _do_fusion_compute_at(update_tensor_c_ub)

                if gm_ub is not None:
                    sch[gm_ub].compute_at(sch[res], c_ub_at_axis)
                    sch[tensor_c_gm].compute_at(sch[res], c_ub_at_axis)
                    sch[tensor_c_ub].reused_by(gm_ub)

                set_overload_flag(overload_flag, sch[res], res_insn_axis)

                return axis_block, res_ub, res_insn_axis, gm_n_inner, index_at_axis

            axis_block, res_ub, res_insn_axis, gm_instn, index_at_axis = \
                sche_l1_mkn_l0_k_frac_frac()

        else:
            if not a_l1_inline_flag:
                sch[tensor_a_l1].compute_at(sch[tensor_c], l1_k_outer)

                def _set_tensor_buffer_tile(tensor_a_l1):
                    """
                    set tensor buffer tile for m in gevm
                    """
                    a_l1_shape = tensor_a_l1.shape
                    if tensor_a_l1.op.input_tensors:
                        tensor_in = tensor_a_l1.op.input_tensors[0]
                        in_shape = tensor_in.shape
                        if a_l1_shape[-2].value != 1 and in_shape[-2].value == 1 \
                                and k_l0_shape == block_reduce:
                            # k_l0_shape == block_reduce for resolve L1 align 256B
                            # only single loop for i1 can be optimized
                            tile = [(None, None,) for i, _ in enumerate(a_l1_shape)]
                            tile[-2] = (None, 1)
                            sch[tensor_a_l1].buffer_tile(*tile)

                _set_tensor_buffer_tile(tensor_a_l1)

            if not b_l1_inline_flag:
                sch[tensor_b_l1].compute_at(sch[tensor_c], l1_k_outer)

            if fusion_ele:
                sch[tensor_c_gm].compute_inline()
                res_ub = sch.cache_write(res, cce.scope_ubuf)
                elemwise_tensors.append(res_ub)

            res_len = len(res.shape)
            m_factor, n_factor, m_part_cnt, n_part_cnt = get_block_split_factor(
                tensor_c, n_nparts_mode, m_factors, n_factors)
            if format_out == "FRACTAL_NZ":

                m_outer_group = sch[res].split(
                    res.op.axis[tensor_len_c - 3], factor=m_factor)
                n_outer_group = sch[res].split(
                    res.op.axis[tensor_len_c - 4], factor=n_factor)

                m_outer, m_inner = sch[res].split(m_outer_group[1],
                                                  factor=m_l1_tile)
                n_outer, n_inner = sch[res].split(n_outer_group[1],
                                                  factor=fusion_n_l1_tile)

            else:
                c_ub_m_outer, c_ub_m_inner = sch[res].split(
                    res.op.axis[res_len - 2], factor=block_in)
                c_ub_n_outer, c_ub_n_inner = sch[res].split(
                    res.op.axis[res_len - 1], factor=block_out)

                m_outer_group = sch[res].split(c_ub_m_outer, factor=m_factor)
                n_outer_group = sch[res].split(c_ub_n_outer, factor=n_factor)

                m_outer, m_inner = sch[res].split(m_outer_group[1],
                                                  factor=m_l1_tile)
                n_outer, n_inner = sch[res].split(n_outer_group[1],
                                                  factor=n_l1_tile)
            index_at_axis = m_outer if m_l1_tile * 2 < m_factor else -1

            fuse_list = []
            if m_part_cnt > n_part_cnt:
                fuse_list.append(n_outer_group[0])
                fuse_list.append(m_outer_group[0])
            else:
                fuse_list.append(m_outer_group[0])
                fuse_list.append(n_outer_group[0])

            if format_out == "FRACTAL_NZ":
                sch[res].reorder(*fuse_list,
                                 m_outer, n_outer,
                                 n_inner, m_inner,
                                 res.op.axis[tensor_len_c - 2],
                                 res.op.axis[tensor_len_c - 1])
            else:
                sch[res].reorder(*fuse_list,
                                 m_outer, n_outer,
                                 n_inner, m_inner,
                                 c_ub_m_inner, c_ub_n_inner)

            overload_flag = False
            if n_part_cnt > 1:
                overload_flag = True

            # multi kernel axis must be > 1
            if axis_block is None:
                axis_block = sch[res].fuse(*fuse_list)
                thread_block = tvm.thread_axis("blockIdx.x")
                sch[res].bind(axis_block, thread_block)

            c_at_axis = n_outer
            c_ub_at_axis = n_outer
            res_insn_axis = n_inner

            sch[tensor_c].compute_at(sch[res], c_at_axis)

            def _do_nd_bias_compurt_at():
                if is_with_bias:
                    sch[tensor_bias_ub].compute_at(sch[res], c_at_axis)
                    sch[tensor_bias_l0c].compute_at(sch[res], c_at_axis)
                    sch[tensor_c_add_bias].compute_at(sch[res], c_at_axis)
                    if not dequant_fusion:
                        if l0c_enable_db:
                            sch[tensor_bias_l0c].preload()
                            sch[tensor_bias_l0c].double_buffer()
                        sch[tensor_bias_ub].preload()
                        sch[tensor_bias_ub].double_buffer()
            _do_nd_bias_compurt_at()

            if tensor_c_ub_fract is not None:
                sch[tensor_c_ub_fract].compute_at(sch[res], c_ub_at_axis)

            def _do_fusion_compute_at(quant_flag, tensor_c_ub, c_ub_at_axis):
                if not quant_flag:
                    sch[tensor_c_ub].compute_at(sch[res], c_ub_at_axis)
                for ten_in in elemwise_tensors:
                    sch[ten_in].compute_at(sch[res], c_ub_at_axis)
                if dequant_tensor is not None:
                    sch[dequant_tensor].compute_at(sch[res], c_ub_at_axis)
                if dequant_fusion and sqrt_flag:
                    sch[tensor_sqrt].compute_at(sch[res], c_ub_at_axis)
                if dequant_fusion:
                    for tensor in dequant_activation_tensor:
                        sch[tensor].compute_at(sch[res], c_ub_at_axis)
                for tensor_list in tensor_fusion_list:
                    sch[tensor_list].compute_at(sch[res], c_ub_at_axis)
                if requant_data_transfer is not None:
                    sch[requant_data_transfer].compute_at(sch[res],
                                                          c_ub_at_axis)

            quant_fusion = requant_fusion or dequant_fusion
            _do_fusion_compute_at(quant_fusion, tensor_c_ub, c_ub_at_axis)

            set_overload_flag(overload_flag, sch[res], res_insn_axis)

        emit_insn_func(sch, tensor_a_l0a, None, 'dma_copy')

        in_tensor = tensor_a_l1.op.input_tensors[0]
        scope_val = sch[in_tensor].scope
        l1_scope_val = sch[tensor_a_l1].scope

        if scope_val == l1_scope_val:
            emit_insn_func(sch, tensor_a_l1, None, 'phony_insn')
        else:
            emit_insn_func(sch, tensor_a_l1, None, 'dma_copy')

        def __tensor_weights_emit_insn(host_axis):
            """
            tensor_b_l1 emit insn operation for compress or not
            """
            if compress_index is not None:
                if not b_l1_inline_flag:
                    set_compress_info(sch, tensor_b_l1, compress_index,
                                      k_l1_tile, n_l1_tile, host_axis)
                    emit_insn_func(sch, tensor_b_l0b, None, 'dma_copy')
                else:
                    tensor_b_l0b.op.attrs["tile_L1_k"] = tensor_b_l1.op.attrs["tile_L1_k"]
                    tensor_b_l0b.op.attrs["tile_L1_n"] = tensor_b_l1.op.attrs["tile_L1_n"]
                    set_compress_info(sch, tensor_b_l0b, compress_index,
                                      k_l0_tile, n_l0_tile, host_axis)
            else:
                if not b_l1_inline_flag:
                    emit_insn_func(sch, tensor_b_l1, None, 'dma_copy')
                emit_insn_func(sch, tensor_b_l0b, None, 'dma_copy')

        __tensor_weights_emit_insn(index_at_axis)

        def _tensor_a_l1_workspace_emit():
            if input_l1_flag == 1:
                emit_insn_func(sch, tensor_a_l1_workspace, None, 'dma_copy')

        _tensor_a_l1_workspace_emit()

        emit_insn_func(sch, tensor_a_ub, None, 'dma_copy')
        emit_insn_func(sch, tensor_b_ub, None, 'dma_copy')

        def _emit_insn_fract():
            if a_ub_fract_exist:
                sch[tensor_a_ub_fract].emit_insn(
                    tensor_a_ub_fract.op.axis[tensor_len_a - 4], 'vector_auto')
            if b_ub_fract_exist:
                sch[tensor_b_ub_fract].emit_insn(
                    tensor_b_ub_fract.op.axis[tensor_len_b - 4], 'vector_auto')

        _emit_insn_fract()

        mad_dict = {"mad_pattern": mad_pattern,
                    "k_outer": [l1_k_outer, l0_k_outer]}
        if is_with_bias:
            sch[tensor_bias_ub].emit_insn(tensor_bias_ub.op.axis[0],
                                          'dma_copy')
            sch[tensor_bias_l0c].emit_insn(tensor_bias_l0c.op.axis[0],
                                           'dma_copy')
            sch[tensor_c_add_bias].emit_insn(tensor_c_add_bias.op.axis[0],
                                             'phony_insn')

            # reuse bias_l0c
            sch[tensor_bias_l0c].pragma(tensor_bias_l0c.op.axis[0],
                                        'reuse_output', 0)
            sch[tensor_c_add_bias].pragma(tensor_c_add_bias.op.axis[0],
                                          'replace_output', 0)
            sch[tensor_c].pragma(l0_n_inner, 'replace_output', 0)

            mad_dict["init_bias"] = 1
        # emit_insn

        # mad_pattern value: 0 for gemm, 1 for gemv
        # set pragma for k_outer
        sch[tensor_c].emit_insn(l0_n_inner, 'mad', mad_dict)
        # quantization config
        if not quantify_fusion:
            if tensor_c_ub.op.attrs['scale_drq'].value == "ENABLE":
                # tensor_drq is second input for tensor_c_ub
                tensor_drq = tensor_c_ub.op.input_tensors[1]
                c_ub = sch.cache_read(tensor_drq, cce.scope_ubuf, [tensor_c_ub])
                emit_insn_func(sch, c_ub, c_ub.op.axis[0], 'dma_copy')
                if axis_block is not None:
                    sch[c_ub].compute_at(sch[res], axis_block)

                if tensor_c_ub.op.attrs['sqrt_out'].value == "SQRT":
                    # Sqrt Mode
                    sch[tensor_c_ub].pragma(tensor_c_ub.op.axis[0],
                                            'deq_scale', 'scalar_sqrt')
                else:
                    # No Sqrt Mode
                    sch[tensor_c_ub].pragma(tensor_c_ub.op.axis[0],
                                            'deq_scale', 'scalar')
            else:
                sch[tensor_c_ub].emit_insn(tensor_c_ub.op.axis[0], 'dma_copy')

        _quantify_fusion_entry(axis_block)

        def _choose_dma_copy(sch, res, with_transpose, res_insn_axis):
            "choose dma copy pattern"
            if with_transpose:
                sch[res].emit_insn(res_insn_axis, 'dma_copy_matmul_transpose')
            else:
                sch[res].emit_insn(res_insn_axis, 'dma_copy')

        _choose_dma_copy(sch, res, with_transpose, res_insn_axis)

        m_factor, n_factor, _, _ = get_block_split_factor(
            tensor_c, n_nparts_mode, m_factors, n_factors)

        block_value = [block_in, block_out]
        factor_value = [m_factor, n_factor]
        tile_value = [m_l0_tile, n_l0_tile]
        emit_axis = split_c_ub_tensor(
            tensor_c_ub_fract, block_value, factor_value, tile_value)

        emit_insn_func(sch, tensor_c_ub_fract, emit_axis, 'vector_auto')

        def dequant_activation_emit_insn_simple():
            if dequant_fusion:
                for ten_in in dequant_activation_tensor:
                    if ten_in.op.tag.find("|") != -1:
                        str_list = ten_in.op.tag.split("|")
                        insn = emit_insn_map.get(str_list[0])
                    else:
                        insn = emit_insn_map.get(ten_in.op.tag)
                    if ten_in in header_ub_tensors:
                        insn = "dma_copy"
                    if insn is None:
                        insn = 'vector_auto'
                    if "elewise_binary_scalar_axpy" in ten_in.op.tag:
                        sch[ten_in].reused_by(ten_in.op.input_tensors[1])
                    sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

        def emit_insn_simple():
            """
            emit insion base on simple axis
            """
            dequant_activation_emit_insn_simple()
            for ten_in in tensor_fusion_list:
                if ten_in.op.name == "cast_i8_ub":
                    insn = _round_emit_insn(round_mode)
                else:
                    insn = emit_fusion_insn_map.get(ten_in.op.name)
                if ten_in.op.name in reform_tensor_tag_list:
                    sch[ten_in].emit_insn(ten_in.op.axis[2], insn)
                else:
                    sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

            for axpy, parent in axpy_2_parent.items():
                sch[parent].reused_by(axpy)

            for ten_in in elemwise_tensors:
                if ten_in.op.tag.find("|") != -1:
                    str_list = ten_in.op.tag.split("|")
                    insn = emit_insn_map.get(str_list[0])
                else:
                    insn = emit_insn_map.get(ten_in.op.tag)
                if ten_in in ele_header_ub_tensors:
                    insn = 'dma_copy'
                if insn is None:
                    insn = 'vector_auto'
                sch[ten_in].emit_insn(ten_in.op.axis[0], insn)

            if gm_ub is not None:
                sch[tensor_c_gm].emit_insn(gm_instn, 'dma_copy')
                sch[gm_ub].emit_insn(gm_ub.op.axis[0], 'phony_insn')

        emit_insn_simple()
        _emit_requant_fusion_insn(requant_data_transfer)

        def open_double_buffer():
            def _open_double_buffer_for_batch_double():
                if double_once == 0:
                    if not a_l1_inline_flag:
                        sch[tensor_a_l1].double_buffer()
                    if not b_l1_inline_flag:
                        sch[tensor_b_l1].double_buffer()

            def _open_double_buffer_for_not_batch_double():
                if m_l1_double_buffer == 2 and tensor_a_reuse == 0:
                    if not a_l1_inline_flag:
                        sch[tensor_a_l1].double_buffer()
                if n_l1_double_buffer == 2 and tensor_b_reuse == 0:
                    if not b_l1_inline_flag:
                        sch[tensor_b_l1].double_buffer()

            # double buffer
            def open_batch_double_buffer():
                if batch_double:
                    _open_double_buffer_for_batch_double()
                else:
                    _open_double_buffer_for_not_batch_double()

            open_batch_double_buffer()

            sch[tensor_a_l0a].double_buffer()
            sch[tensor_b_l0b].double_buffer()
            if tensor_b_reuse == 0 and tensor_a_reuse == 0:
                if l0c_enable_db:
                    sch[tensor_c].double_buffer()

                if not quantify_fusion:
                    if tensor_c_ub.op.tag != 'matmul' and \
                            tensor_c_ub.op.tag != 'matmul_gemv':
                        sch[tensor_c_ub].double_buffer()

                if gm_ub is not None:
                    sch[gm_ub].double_buffer()
                if a_ub_exist:
                    sch[tensor_a_ub].double_buffer()
                if b_ub_exist:
                    sch[tensor_b_ub].double_buffer()
                if a_ub_fract_exist:
                    sch[tensor_a_ub_fract].double_buffer()
                if b_ub_fract_exist:
                    sch[tensor_b_ub_fract].double_buffer()
                if tensor_c_ub_fract is not None:
                    sch[tensor_c_ub_fract].double_buffer()
                for ten_i in elemwise_tensors:
                    sch[ten_i].double_buffer()
                if requant_data_transfer is not None:
                    sch[requant_data_transfer].double_buffer()

        open_double_buffer()

    schedule_l1_mkn_l0_k_tiling()

    return True
