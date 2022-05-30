#!/usr/bin/env python
# -*- coding:utf-8 -*-
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
gemm_compute
"""
import math
from functools import reduce as functools_reduce

from tbe import tvm
import tbe.common.platform as tbe_platform
from tbe.common.platform import platform_info as tbe_platform_info
import tbe.common.utils as tbe_utils
from tbe.common.utils import broadcast_shapes
from tbe.dsl.compute.util import int_ceil_div
from tbe.dsl.compute import cube_util
from tbe.dsl.base.operation import in_dynamic
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import error_manager_cube
from tbe.tvm.tensor import Tensor
from .gemm_compute_util import FormatCompute


@para_check.check_input_type(Tensor, Tensor, dict)
def gemm(tensor_a, tensor_b, para_dict):
    """
    algorithm: gemm
    calculate matrix multiplication C = alpha*(AB + bias) + beta*C

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b: list or tuple
            Shape of the second tensor b with the same type with a,
            and shape_a, shape_b must be 2 dims
    shape_bias: list or tuple
            Shape of bias, only support the input data format with ND
    trans_a: bool
            If True, shape_a is transposed before multiplication
    trans_b: bool
            If True, shape_b is transposed before multiplication
    is_fractal: bool
            If True, the input data format of a and b must be fractal format

    Returns None
    """
    # ----temp
    trans_a = para_dict.get("trans_a")
    trans_b = para_dict.get("trans_b")
    format_a = para_dict.get("format_a")
    format_b = para_dict.get("format_b")
    if format_a == "FRACTAL_NZ":
        trans_a = not trans_a
    if format_b == "FRACTAL_NZ":
        trans_b = not trans_b
    para_dict["trans_a"] = trans_a
    para_dict["trans_b"] = trans_b

    gemm_compute = GEMMCompute(tensor_a, tensor_b, para_dict)
    gemm_compute.calculate()
    result = gemm_compute.res

    return result


class GEMMComputeParam:
    """
    be used by gemm_tilingcase
    """
    tiling_info_dict = {}
    dynamic_mode = None
    batch_a = False
    batch_b = False
    format_a = "Fractal_NZ"
    format_b = "Fractal_NZ"
    format_out = "Fractal_NZ"
    m_var_name = None
    k_var_name = None
    n_var_name = None
    block_in = tbe_platform.BLOCK_IN
    block_out = tbe_platform.BLOCK_OUT
    block_reduce = tbe_platform.BLOCK_REDUCE
    split_k_flag = False

    def __init__(self) -> None:
        pass

    @staticmethod
    def get_op_type_flag(format_a, format_b, mmad_mode):
        """
        0: a and b both ND input
        1: a and b both fractal input
        2: a is fractal b is ND
        3: a is ND b is fractal
        """
        if (format_a == "ND" and format_b == "ND"):
            op_type_flag = 0
        elif (format_a != "ND" and format_b == "ND"):
            op_type_flag = 2
        elif (format_a == "ND" and format_b != "ND"):
            op_type_flag = 3
        else:
            op_type_flag = 1
        if mmad_mode == "gemv" and op_type_flag in (2, 3):
            change_value_dict = {
                2: 3,
                3: 2
            }
            op_type_flag = change_value_dict.get(op_type_flag)

        return op_type_flag

    @staticmethod
    def check_tail_block(n_shape, ops_data_flow_mode, format_out, is_dynamic):
        """
        This function is used to calculate Extra block needed under specific data
        flow mode and n_shape
        """
        tail_block_flag = 0
        if format_out == "FRACTAL_NZ":
            return 1
        if is_dynamic and format_out == "ND":
            # Do not have n_shape runtime info therefore considering n_dynamic contains tail block
            return tail_block_flag
        if ops_data_flow_mode in ("int82int32", "int82fp32"):
            divide_factor = 32
            data_num = 8
        elif ops_data_flow_mode == "fp162fp16":
            divide_factor = 16
            data_num = 16
        else:
            divide_factor = 16
            data_num = 8

        if 1 <= int(n_shape) <= divide_factor or int(n_shape) % divide_factor == 0:
            tail_block_flag = 1
        elif int(n_shape) % divide_factor < data_num:
            tail_block_flag = 0
        return tail_block_flag

    @staticmethod
    def get_trans_flag(transpose_a, transpose_b):
        """
        get trans flag inorder to get tiling
        """
        trans_flag = 1
        if transpose_a:
            if transpose_b:
                trans_flag = 4
            else:
                trans_flag = 2
        elif transpose_b:
            trans_flag = 3
        return trans_flag

    @staticmethod
    def get_stride_w_value(tail_block_flag, split_k):
        """
        result_value's bit0 means have tail block,
        bit1 means need bind k axis in core
        """
        result_value = tail_block_flag
        split_k = int(split_k)
        result_value |= split_k << 1
        return result_value


class GetPerfCoreNum:
    """
    get perf core num by mte2/mte3, use in atomic add k.
    """
    BYTES_DTYPE = {
        "uint64": 8,
        "float16": 2,
        "float32": 4,
        "int32": 4,
        "int16": 2,
        "uint16": 2,
        "int8": 1,
        "uint8": 1,
        "int4": 0.5
    }
    soc_hbm_bandwidth_info = {8: 250, 32: 1100}
    soc_l2_bandwidth_info = {8: 1300, 32: 3300}
    atomic_addr_clean_cost_multi = 2

    def __init__(self):
        pass

    def get_best_perf_factor(self, shapes, blocks):
        """
        get best perf block dim factor by mte2 and mte3
        """
        m_factor = 1
        k_factor = 1
        n_factor = 1
        float16_size = 2
        m_shape, k_shape, n_shape = shapes
        block_in, block_reduce, block_out = blocks
        core_num = tbe_platform_info.get_soc_spec("CORE_NUM")
        l2_size = tbe_platform_info.get_soc_spec("L2_SIZE")
        if core_num < 8:
            return 1, 1, 1
        use_out_buffer_size = (m_shape * k_shape + k_shape * n_shape + m_shape * n_shape) * float16_size
        hbm_bandwidth, l2_bandwidth = self._get_bandwidth(core_num)
        cur_bandwidth = hbm_bandwidth
        if use_out_buffer_size < l2_size:
            cur_bandwidth = l2_bandwidth
        min_cost = core_num * (m_shape * n_shape + m_shape * k_shape + n_shape * k_shape) * float16_size / hbm_bandwidth

        m_axis_outer = int_ceil_div(m_shape, block_in)
        n_axis_outer = int_ceil_div(n_shape, block_out)
        k_axis_outer = int_ceil_div(k_shape, block_reduce)

        m_max_dim = core_num if (m_axis_outer > core_num) else m_axis_outer
        n_max_dim = core_num if (n_axis_outer > core_num) else n_axis_outer
        k_max_dim = core_num if (k_axis_outer > core_num) else k_axis_outer

        total_max_dim = m_max_dim * k_max_dim * n_max_dim
        for i in range(0, total_max_dim):
            k_dim = int(i / (m_max_dim * n_max_dim)) + 1
            n_dim = int(i / m_max_dim) % n_max_dim + 1
            m_dim = i % m_max_dim + 1
            if m_dim * k_dim * n_dim > core_num:
                continue
            if (m_dim > m_axis_outer) or (k_dim > k_axis_outer) or (n_dim > n_axis_outer):
                continue
            block_dims = (m_dim, k_dim, n_dim)
            cur_cost = self._compute_perf(shapes, block_dims, cur_bandwidth)
            if cur_cost < min_cost:
                min_cost = cur_cost
                m_factor, k_factor, n_factor = block_dims
        return m_factor, k_factor, n_factor

    def _get_bandwidth(self, core_num):
        hbm_bandwidth = self.soc_hbm_bandwidth_info.get(core_num, 0)
        l2_bandwidth = self.soc_l2_bandwidth_info.get(core_num, 0)
        if hbm_bandwidth == 0 or l2_bandwidth == 0:
            distant = abs(core_num - 8)
            core_num_best = 8
            all_corenum_value = self.soc_hbm_bandwidth_info.keys()
            for inner_core_num in all_corenum_value:
                if abs(core_num - inner_core_num) < distant:
                    distant = abs(core_num - inner_core_num)
                    core_num_best = inner_core_num
            hbm_bandwidth = self.soc_hbm_bandwidth_info.get(core_num_best, 0)
            l2_bandwidth = self.soc_l2_bandwidth_info.get(core_num_best, 0)
        return hbm_bandwidth, l2_bandwidth

    def _compute_perf(self, shapes, block_dims, cur_bandwidth):
        m_shape, k_shape, n_shape = shapes
        m_dim, k_dim, n_dim = block_dims
        m_shape_inner = int_ceil_div(m_shape, m_dim)
        k_shape_inner = int_ceil_div(k_shape, k_dim)
        n_shape_inner = int_ceil_div(n_shape, n_dim)
        out_data_size_fp32 = self.BYTES_DTYPE.get("float32", 0)
        in_data_size = self.BYTES_DTYPE.get("float16", 0)
        cast_node_cost = 0
        transdata_node_cost = 0
        atomic_add_bw_lose_radio = 1
        atomic_addr_clean_cost = 0
        if k_dim != 1:
            atomic_add_bw_lose_radio = 0.5
            atomic_addr_clean_cost = (m_shape * n_shape * out_data_size_fp32 /
                                      cur_bandwidth) * self.atomic_addr_clean_cost_multi

        mte3_cost = k_dim * (m_shape_inner * n_shape_inner * out_data_size_fp32) / (atomic_add_bw_lose_radio *
                                                                                    cur_bandwidth)
        base_load_cost = (m_shape_inner * k_shape_inner + k_shape_inner * n_shape_inner) * in_data_size / cur_bandwidth
        b_repeat_load_cost = (m_dim - 1) * k_shape_inner * n_shape_inner * in_data_size / cur_bandwidth
        a_repeat_load_cost = (n_dim - 1) * k_shape_inner * m_shape_inner * in_data_size / cur_bandwidth
        total_cost = (base_load_cost + mte3_cost + a_repeat_load_cost + b_repeat_load_cost + cast_node_cost +
                      transdata_node_cost + atomic_addr_clean_cost)

        return total_cost


class GEMMCompute(FormatCompute):
    """
    algorithm: mmad
    calculate matrix multiplication C = alpha*(AB + bias) + beta*C

    Parameters:
    tensor_a: the first tensor a

    tensor_b: the seconed tensor b with the same dtype with a

        If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
        must be 32*16.
        If A is transpose , then AShape classification matrix must be
        32*16 in gm/L1,then it is 16*32 in L0A.
        If B is transpose , then BShape classification matrix must be
        16*32 in gm/L1,then it is 32*16 in L0B.

    trans_a: if True, tensor_a needs to be transposed

    trans_b: if True, tensor_b needs to be transposed

    format_a: the format of tensor_a, support FRACTAL_NZ, FRACTAL_Z, ND
              default is "ND"

    format_b: the format of tensor_b, support FRACTAL_NZ, FRACTAL_Z, ND
              default is "ND"

    dst_dtype: output data type, support float16 float32 int32
               default is float32

    tensor_bias: the bias with used to add

    tensor_c: the c matrix with used to add

    format_out: output format, now support ND, FRACTAL_NZ

    kernel_name: kernel name, default is "gemm"

    Returns None
    """

    type_map = {
        "float16": "fp16",
        "float32": "fp32",
        "int8": "int8",
        "int32": "int32",
        "uint8": "uint8",
        "int4": "int4"
    }

    m_shape_dict = {"ND": -2, "FRACTAL_NZ": -3, "FRACTAL_Z": -4}
    m0_shape_dict = {"FRACTAL_NZ": -2, "FRACTAL_Z": -2}
    n_shape_dict = {"ND": -1, "FRACTAL_NZ": -4, "FRACTAL_Z": -3, "fractal": -3}
    n0_shape_dict = {"FRACTAL_NZ": -1, "FRACTAL_Z": -2, "fractal": -2}
    trans_dict = {-1: -2, -2: -1, -3: -4, -4: -3}

    # if K_DIM is  equal or larger than GEVM_MODE_K_DIM_LIMIT in gevm/gemv mode, use gemm mode.
    # K_DIM is k * k0
    GEVM_MODE_K_DIM_LIMIT = 9216
    # if (K_DIM, N_DIM) in GEVM_MODE_LIMIT_LIST, use gemm mode. N_DIM is n*n0
    GEVM_MODE_LIMIT_LIST = [(4096, 4096), (4096, 1008)]

    def __init__(self, tensor_a, tensor_b, para_dict):
        super(GEMMCompute, self).__init__()
        self.para_dict = para_dict
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.shape_a = [self._get_value(i) for i in tensor_a.shape]
        self.shape_b = [self._get_value(i) for i in tensor_b.shape]
        self.alpha = para_dict.get("alpha")
        self.beta = para_dict.get("beta")
        self.trans_a = para_dict.get("trans_a", False)
        self.trans_b = para_dict.get("trans_b", False)
        self.format_a = para_dict.get("format_a", "ND")
        self.format_b = para_dict.get("format_b", "ND")
        self.cache_tiling_flag = para_dict.get("cache_tiling_flag", False)
        self.offset_x, self.offset_w = self._get_offset_info(para_dict, tensor_a.dtype, tensor_b.dtype)
        self.tensor_bias = None
        self.tensor_c = para_dict.get("tensor_c")
        self.src_dtype = tensor_a.dtype
        self.dst_dtype = para_dict.get("dst_dtype", "float16")
        self.kernel_name = para_dict.get("kernel_name", "gemm")
        self.block_in = tbe_platform.BLOCK_IN
        self.block_out = tbe_platform.BLOCK_OUT
        self.block_reduce = tbe_platform.BLOCK_REDUCE
        self.matrix_type = "float32"
        self.mmad_mode = "gemm"
        self.quantize_params = para_dict.get("quantize_params")
        self.format_out = para_dict.get("format_out")
        self.fc_flag = para_dict.get("fc_flag", False)
        self.need_reformat_to_nd = False
        self.compress_index = para_dict.get("compress_index")
        self.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.attrs = para_dict.get("attrs", {})
        self.batch_shape = para_dict.get("batch_shape")
        self.op_type = para_dict.get("op_type", None)
        self.input_range = para_dict.get("input_range")
        self.beta_mode, self.alpha_mode = "none", "none"
        self.have_bias, self.have_c = False, False
        self.res = None
        self.placeholder_name = {}
        self.align_a, self.align_b = True, True
        self.ops_format = "ND"
        self.ori_km_shape, self.ori_kn_shape = 1, 1
        self.compress_flag = False
        self.ori_km_shape = 1
        self.l0c_support_fp32 = True
        self.ops_data_flow_mode = "fp162fp32"
        self.need_cast_to_fp16 = False
        # Consider not using only_use_gevm_gemv_flow
        self.only_use_gevm_gemv_flow = False
        self.int8_not_double_m = False
        self.split_k = False
        self.is_fusion = para_dict.get("is_fusion", False)
        self.best_split_k_block_dim = []

    @staticmethod
    def _get_value(shape_object):
        """
        get the value of shape_object when having attr "value"
        """
        return shape_object.value if hasattr(shape_object, "value") else shape_object

    @staticmethod
    def _elecnt_of_shape(shape):
        """
        calculate reduce shape
        """
        return functools_reduce(lambda x, y: x * y, shape)

    @staticmethod
    def _get_batch_dims(attrs):
        """
        get batch matmul ori batch dims
        Parameters:
            attrs: the dict of params

        Returns:
            batch_dims: dims of ori_batch_a, ori_batch_b, ori_batch_out, ori_batch_broadcast
        """
        batch_a = attrs.get("batch_shape_a", [])
        batch_b = attrs.get("batch_shape_b", [])
        batch_out = attrs.get("batch_shape_out", [])
        batch_a = shape_util.shape_to_list(batch_a)
        batch_b = shape_util.shape_to_list(batch_b)
        batch_out = shape_util.shape_to_list(batch_out)

        if batch_a or batch_b:
            batch_a, batch_b, batch_l0c = broadcast_shapes(batch_a, batch_b)
        else:
            batch_l0c = []

        return batch_a, batch_b, batch_out, batch_l0c

    @staticmethod
    def _get_offset_info(attrs_dict, dtype_a, dtype_b):
        """
        get offset info, like offset_x and offset_w
        """
        offset_x = 0
        offset_w = None
        if cube_util.is_v200_version_new() and dtype_a in ("uint8", "int8", "int4"):
            if attrs_dict.get("offset_a") is not None:
                offset_x = attrs_dict.get("offset_a")
            if dtype_b in ("int8", "int4"):
                offset_w = attrs_dict.get("offset_b")

        return offset_x, offset_w

    @staticmethod
    def _get_shape_a_in_nc1hwc0(tensor_a_zz):
        """
        get a shape's format nc1hwc0 inorder to get tiling
        """
        if GEMMComputeParam.batch_a:
            return [tensor_a_zz.shape[0], tensor_a_zz.shape[2], tensor_a_zz.shape[1],
                    GEMMComputeParam.block_in, GEMMComputeParam.block_reduce]
        return [1, tensor_a_zz.shape[1], tensor_a_zz.shape[0],
                GEMMComputeParam.block_in, GEMMComputeParam.block_reduce]

    @staticmethod
    def _get_shape_b_in_nc1hwc0(tensor_b_zn):
        """
        get b shape's format nc1hwc0 inorder to get tiling
        """
        if GEMMComputeParam.batch_b:
            return [tensor_b_zn.shape[1] * GEMMComputeParam.block_reduce, tensor_b_zn.shape[2],
                    1, 1, GEMMComputeParam.block_out]
        return [tensor_b_zn.shape[0] * GEMMComputeParam.block_reduce, tensor_b_zn.shape[1],
                1, 1, GEMMComputeParam.block_out]

    @staticmethod
    def _get_compress_tensor_compute(tensor_src, comp_index, op_name):
        """
        get compress tensor compute
        """
        _, _, _, compress_mode = tbe_platform_info.get_soc_spec("UNZIP")
        comp_size = 8 if compress_mode == 1 else 2

        tile_k_value = tvm.var("tile_L1_k", dtype="int32")
        tile_n_value = tvm.var("tile_L1_n", dtype="int32")
        n_dim = tvm.var("block_dim_n", dtype="int32")

        shape_src = tensor_src.shape
        block_n_num = int_ceil_div(shape_src[-3], tile_n_value)
        block_k_num = int_ceil_div(shape_src[-4], tile_k_value)
        n_dim_num = int_ceil_div(block_n_num, n_dim)
        n_dim_value = n_dim_num * tile_n_value

        # tile_mode is 1 when tile_n < dim_n, or tile_mode is 0
        if len(shape_src) == 4:
            tensor = tvm.compute(shape_src,
                                 lambda i, j, k, l: tvm.unzip(
                                     comp_index((j // n_dim_value * n_dim_num * block_k_num
                                                 + (j % n_dim_value) // tile_n_value * block_k_num
                                                 + i // tile_k_value) * comp_size),
                                     tensor_src(i, j, k, l)),
                                 name=op_name,
                                 attrs={"tile_L1_k": tile_k_value,
                                        "tile_L1_n": tile_n_value,
                                        "block_dim_n": n_dim}
                                 )
        else:
            reason = "The compress feature only support the tensor with 4 dims, "\
                     "but the length of input tensor is {}.".format(len(shape_src))
            error_manager_cube.raise_err_specific("GEMM", reason)
        return tensor

    @staticmethod
    def _get_quantize_params(quantize_params=None, dtype_dst="float16"):
        """
        algorithm: check matmul quantize parameters

        Parameters:
        quantize_params: quantization parameters,
                not None means enable quantization, it is dictionary structure
        dtype_dst: the dst tensor's dtype

        Returns:
            scale_drq: DISABLE: dequant, ENABLE: requant
            scale_drq_tensor: scale drq tensor
            sqrt_out: NON_SQRT: none sqrt quantize, SQRT: sqrt quantize
        """
        # quantization parameter default value
        sqrt_out = "NON_SQRT"
        scale_drq_tensor = None
        scale_drq = "DISABLE"

        if quantize_params is not None:
            if dtype_dst == "float16":
                scale_drq = "ENABLE"
                scale_drq_tensor = quantize_params.get("scale_drq")
                sqrt_out = quantize_params.get("sqrt_mode_out")
            else:
                reason = "Dst tensor's dtype should be float16, while it is {}.".format(dtype_dst)
                error_manager_cube.raise_err_specific("GEMM", reason)

        return scale_drq, scale_drq_tensor, sqrt_out

    @staticmethod
    def _check_quantize_params(quantize_params=None):
        """
        Parameters:
        quantize_params: not None means enable quantization, it is dictionary structure
            probable param keys as following:
            quantize_alg: quantize mode, support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'
            scale_mode_a: tensor_a inbound quantization mode, support 'SCALAR' and 'VECTOR'
            scale_mode_b: tensor_b inbound quantization mode, support 'SCALAR' and 'VECTOR'
            scale_mode_out: out tensor quantization mode, support 'SCALAR' and 'VECTOR'
            sqrt_mode_a: tensor_a inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
            sqrt_mode_b: tensor_b inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
            sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'
            scale_q_a: scale placeholder for tensor_a inbound quantization
            offset_q_a: offset placeholder for tensor_a inbound quantization
            scale_q_b: scale placeholder for tensor_b inbound quantization
            offset_q_b: offset placeholder for tensor_b inbound quantization
            scale_drq: scale placeholder for requantization or dequantization
            offset_drq: scale placeholder for requantization or dequantization
        """
        # check input quantization parameter info
        if quantize_params is None:
            return
        # quantization parameter default value
        quantize_mode = "NON_OFFSET"
        scale_out = "SCALAR"

        params_not_support = ["scale_mode_a", "scale_mode_b",
                              "scale_q_a", "scale_q_b",
                              "offset_q_a", "offset_q_b",
                              "sqrt_mode_a", "sqrt_mode_b"
                              "offset_drq"]
        params_necessary = ["quantize_alg", "scale_mode_out", "sqrt_mode_out", "scale_drq"]

        dict_args = {'errCode': 'E61001', 'reason': "TEMP REASON."}
        for param_str in params_not_support:
            if param_str in quantize_params:
                reason = "Parameter {} is not supported in the current version.".format(param_str)
                error_manager_cube.raise_err_specific("GEMM", reason)
        for param_str in params_necessary:
            if param_str not in quantize_params:
                reason = "Lack of parameter {}, need to supply it.".format(param_str)
                error_manager_cube.raise_err_specific("GEMM", reason)

        quantize_mode = quantize_params["quantize_alg"]
        if quantize_mode not in ("NON_OFFSET", "HALF_OFFSET_A"):
            reason = "Parameter quantize_alg should be 'NON_OFFSET' or "\
                     "'HALF_OFFSET_A', while it is {}.".format(quantize_mode)
            error_manager_cube.raise_err_specific("GEMM", reason)

        scale_out = quantize_params["scale_mode_out"]
        if scale_out not in ("SCALAR", "VECTOR"):
            reason = "Parameter scale_mode_out should be 'SCALAR' or "\
                     "'VECTOR', while it is {}.".format(scale_out)
            error_manager_cube.raise_err_specific("GEMM", reason)

        sqrt_out = quantize_params["sqrt_mode_out"]
        if sqrt_out not in ("NON_SQRT", "SQRT"):
            reason = "Parameter sqrt_mode_out should be 'NON_SQRT' or "\
                     "'SQRT', while it is {}.".format(sqrt_out)
            error_manager_cube.raise_err_specific("GEMM", reason)

        scale_drq_tensor = quantize_params.get("scale_drq")
        if scale_drq_tensor is None:
            reason = "scale_drq_tensor is None, need to supply it."
            error_manager_cube.raise_err_specific("GEMM", reason)

    def calculate(self):
        """
        the main func of gemm
        """
        self._tensor_b_swap_c1_hw()
        self._init_func()
        self.ops_format = self.format_a

        scale_drq, scale_drq_tensor, sqrt_out = self._get_quantize_params(self.quantize_params, self.dst_dtype)
        tensor_a_zz = self._get_tensor_a_zz()
        tensor_b_zn = self._get_tensor_b_zn()
        tensor_mmad = self._get_tensor_mmad(tensor_a_zz, tensor_b_zn, self.tensor_bias)
        tensor_mmad_with_scale = self._get_tensor_mmad_with_scale(tensor_mmad, scale_drq, scale_drq_tensor, sqrt_out)

        tensor_alpha_mmad = self._get_tensor_alpha_mmad(tensor_mmad_with_scale)
        tensor_beta_bias = self._get_tensor_beta_bias()
        tensor_gemm = self._get_tensor_gemm(tensor_alpha_mmad, tensor_beta_bias)
        self._compute_res(tensor_gemm)

        if in_dynamic():
            self._init_dynamic_base_params()
            self._init_dynamic_tiling_info_dict(tensor_a_zz, tensor_b_zn)

    def _init_dynamic_base_params(self):
        """
        init dynamic base params
        """
        GEMMComputeParam.batch_a = (len(self.tensor_a.shape) in (3, 5))
        GEMMComputeParam.batch_b = (len(self.tensor_b.shape) in (3, 5))
        if GEMMComputeParam.batch_a or GEMMComputeParam.batch_b:
            GEMMComputeParam.dynamic_mode = "dynamic_mknb"
        else:
            GEMMComputeParam.dynamic_mode = "dynamic_mkn"
        GEMMComputeParam.block_in = self.block_in
        GEMMComputeParam.block_out = self.block_out
        GEMMComputeParam.block_reduce = self.block_reduce
        GEMMComputeParam.format_a = self.format_a
        GEMMComputeParam.format_b = self.format_b
        GEMMComputeParam.format_out = self.format_out
        GEMMComputeParam.m_var_name = self._get_var_name(self.format_a, "m", self.cache_tiling_flag)
        GEMMComputeParam.k_var_name = self._get_var_name(self.format_a, "k", self.cache_tiling_flag)
        GEMMComputeParam.n_var_name = self._get_var_name(self.format_b, "n", self.cache_tiling_flag)
        GEMMComputeParam.split_k_flag = self.split_k

    def _init_dynamic_tiling_info_dict(self, tensor_a_zz, tensor_b_zn):
        """
        init dynamic tiling_info_dict
        Parameters:
            tensor_a_zz: the matrix a before mmad
            tensor_b_zn: the matrix b before mmad
        """
        aligned_coeff = 1 if len(self.tensor_b.shape) == 3 else self.block_out
        batch_idx_offset = 1 if len(self.tensor_b.shape) in (3, 5) else 0
        n_dim = (batch_idx_offset + 1) if self.trans_b else batch_idx_offset
        n1_shape = self.tensor_b.shape[n_dim]
        n_shape = aligned_coeff * n1_shape
        # This self.ops_format means format_a in this scenario.
        n_shape_dynamic_flag = isinstance(n1_shape, (tvm.expr.Var, tvm.expr.Expr))
        tail_block = GEMMComputeParam.check_tail_block(
            n_shape, self.ops_data_flow_mode, self.format_out,
            n_shape_dynamic_flag) if self.ops_format == "ND" else 1

        # padl means a_fused_num, and padr means b_fused_num, and padu means is_gevm/is_gemv
        GEMMComputeParam.tiling_info_dict = {
            "A_shape": self._get_shape_a_in_nc1hwc0(tensor_a_zz),
            "B_shape": self._get_shape_b_in_nc1hwc0(tensor_b_zn),
            "C_shape": None,
            "A_dtype": self.tensor_a.dtype,
            "B_dtype": self.tensor_b.dtype,
            "C_dtype": self.res.dtype,
            "mad_dtype": self.matrix_type,
            "padl": int(self.format_a == "ND") * 10,
            "padr": int(self.format_b == "ND") * 10,
            "padu": int(self.mmad_mode in ("gemv", "gevm")),
            "padd": 0,
            "strideH": GEMMComputeParam.get_op_type_flag(self.format_a, self.format_b, self.mmad_mode),
            "strideW": GEMMComputeParam.get_stride_w_value(tail_block, self.split_k),
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": GEMMComputeParam.get_trans_flag(self.trans_a, self.trans_b),
            "dilationW": 1,
            "group": 1,
            "fused_double_operand_num": int(self.format_out == "ND"),
            "bias_flag": (self.tensor_bias is not None),
            "op_tag": "matmul",
            "op_type": "matmul",
            "kernel_name": self.kernel_name,
            "dynamic_shape_flag": True,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b
        }

    def _set_bias_or_c(self):
        if (self.alpha is None) or (self.beta is None):
            self.tensor_bias, self.tensor_c = self.tensor_c, self.tensor_bias

    def _set_mmad_mode(self):
        """
        when m or n's length is 1 and k is align to 512Byte, can
        use gemv or gevm mode, else use gemm mode
        """
        shape_a = [self._get_value(i) for i in self.tensor_a.shape]
        shape_b = [self._get_value(i) for i in self.tensor_b.shape]
        # The op GEMM not use gevm/gemv now
        if self._get_not_gevm_gemv_flag() or (self.alpha is not None) or in_dynamic():
            self.mmad_mode = "gemm"
            return

        m_index = self.m_shape_dict.get(self.format_a)
        n_index = self.n_shape_dict.get(self.format_b)
        m_index = self.trans_dict.get(m_index) if self.trans_a else m_index
        n_index = self.trans_dict.get(n_index) if self.trans_b else n_index
        n_shape = shape_b[n_index]
        gevm_mode_flag, self.only_use_gevm_gemv_flow = self._get_gevm_flag(self.m0_shape_dict,
                                                                           self.trans_dict, shape_a, m_index)
        gevm_mode_flag = self._cancel_gevm_mode(gevm_mode_flag, shape_a[self.trans_dict.get(m_index)], n_shape)
        gemv_mode_flag = self._get_gemv_flag(self.n0_shape_dict, self.trans_dict, shape_b, n_index)
        both_fractal = self.format_a != "ND" and self.format_b != "ND"
        self._check_and_renew_block(gevm_mode_flag, gemv_mode_flag, both_fractal, self.only_use_gevm_gemv_flow)

    def _check_and_renew_block(self, gevm_mode_flag, gemv_mode_flag, both_fractal, only_use_gevm_gemv_flow):
        is_vector_mul_vector = gevm_mode_flag and gemv_mode_flag and both_fractal
        if is_vector_mul_vector:
            reason = "Not support vector mul vector when A and B both fractal."
            error_manager_cube.raise_err_specific("GEMM", reason)
        mmad_mode = "gevm" if gevm_mode_flag else ("gemv" if gemv_mode_flag else "gemm")
        mmad_mode = "gemm" if (gevm_mode_flag and gemv_mode_flag) else mmad_mode
        if not only_use_gevm_gemv_flow:
            self.block_in = tbe_platform.BLOCK_VECTOR if mmad_mode == "gevm" else self.block_in
            self.block_out = tbe_platform.BLOCK_VECTOR if mmad_mode == "gemv" else self.block_out
        self.mmad_mode = mmad_mode

    def _get_gevm_flag(self, m0_shape_dict, trans_dict, shape_a, m_index):
        if self.format_a != "ND":
            m0_index = m0_shape_dict.get(self.format_a)
            gevm_mode_flag = shape_a[m0_index] == tbe_platform.BLOCK_VECTOR
            only_use_gevm_gemv_flow = False
        else:
            if self.alpha is not None:
                gevm_mode_flag = (shape_a[m_index] == 1
                    and (shape_a[trans_dict.get(m_index)] % (self.block_in * self.block_reduce) == 0))
                only_use_gevm_gemv_flow = False
            else:
                gevm_mode_flag = (shape_a[m_index] == 1)
                only_use_gevm_gemv_flow = (shape_a[m_index] == 1
                    and (shape_a[trans_dict.get(m_index)] % (self.block_in * self.block_reduce) != 0))
        return gevm_mode_flag, only_use_gevm_gemv_flow

    def _cancel_gevm_mode(self, gevm_mode_flag, shape_a_k, n_shape):
        """The performance of gemm mode is better than gevm in some cases"""
        if not gevm_mode_flag or self.only_use_gevm_gemv_flow:
            return gevm_mode_flag
        multi_ka = 1
        if self.format_a != "ND":
            multi_ka = self.block_reduce
        multi_nb = 1
        if self.format_b != "ND":
            multi_nb = self.block_out
        if (((shape_a_k * multi_ka) >= self.GEVM_MODE_K_DIM_LIMIT)
            or ((shape_a_k * multi_ka, n_shape * multi_nb) in self.GEVM_MODE_LIMIT_LIST)):
            gevm_mode_flag = False
        return gevm_mode_flag

    def _get_gemv_flag(self, n0_shape_dict, trans_dict, shape_b, n_index):
        if self.format_b != "ND":
            n0_index = n0_shape_dict.get(self.format_b)
            gemv_mode_flag = shape_b[n0_index] == tbe_platform.BLOCK_VECTOR
        else:
            gemv_mode_flag = (shape_b[n_index] == 1
                and (shape_b[trans_dict.get(n_index)] % (self.block_out * self.block_reduce) == 0))
        return gemv_mode_flag

    def _get_not_gevm_gemv_flag(self):
        is_int82int32_nd = (self.ops_data_flow_mode == "int82int32"
            and (self.format_a == "ND" or self.format_b == "ND"))
        not_use_gevm_gemv_flag = (self.ops_data_flow_mode == "int82fp32" or is_int82int32_nd) \
                                 and (self.alpha is not None)
        return not_use_gevm_gemv_flag

    def _set_output_format(self):
        """
        get gemm output format
        format_a: the shape format of tensor a
        format_b: the shape format of tensor b
        format_out: the shape format of output, default is FRACTAL_NZ
            if format_a and format_b both ND and user not set format_out, return ND
        """
        format_a = self.format_a
        format_b = self.format_b
        format_out = self.format_out
        default_format = "FRACTAL_NZ"
        if format_out is None:
            if (format_a == "ND") and (format_b == "ND"):
                self.format_out = "ND"
            else:
                self.format_out = default_format
        elif format_out == "NC1HWC0":
            shape_a = self.tensor_a.shape
            if len(shape_a) in (3, 5):
                shape_a_m = self._get_value(shape_a[1])
            else:
                shape_a_m = self._get_value(shape_a[0])
            if shape_a_m != 1:
                self.format_out = "ND"
            else:
                self.format_out = "FRACTAL_NZ"

    def _get_tensor_mmad_with_scale(self, tensor_mmad, scale_drq, scale_drq_tensor, sqrt_out):
        shape_mmad = [self._get_value(i) for i in tensor_mmad.shape]
        if self.mmad_mode in ("gemv", "gevm"):
            shape_mmad[-2] = 1
        nz_b = (self.format_b == "FRACTAL_NZ")
        if scale_drq_tensor is not None:
            tensor_mmad_with_scale = tvm.compute(
                shape_mmad,
                lambda *indices: (tensor_mmad[indices] * scale_drq_tensor[0]),
                name="tensor_mmad_with_scale",
                attrs={"scale_drq": scale_drq,
                       "sqrt_out": sqrt_out,
                       "nz_b": nz_b})
        else:
            tensor_mmad_with_scale = tvm.compute(
                shape_mmad,
                lambda *indices: tensor_mmad(*indices),
                name="tensor_mmad_with_scale",
                attrs={"scale_drq": scale_drq,
                       "sqrt_out": sqrt_out,
                       "nz_b": nz_b})

        return tensor_mmad_with_scale

    def _tensor_b_swap_c1_hw(self):
        if self.op_type == "BatchMatMulV2":
            shape_b_ori = self.tensor_b.op.attrs["ori_shape"]
            shape_b = self.tensor_b.shape
            len_shape_b_ori = len(shape_b_ori)
            len_shape_b = len(shape_b)
            # (c1hw, n1, n0, c0) -> (h, w, c1, n1, n0, c0)
            is_valid = (not in_dynamic()
                        and len_shape_b_ori in (3, 4)
                        and len_shape_b == 4
                        and self.tensor_b.dtype == "int8"
                        and not self.trans_b
                        and self.format_b == "FRACTAL_Z")
            if is_valid:
                if len_shape_b_ori == 4:
                    height, width, _, _ = shape_b_ori
                else:
                    height = 1
                    width, _, _ = shape_b_ori

                c1hw, n1, n0, c0 = shape_b
                c1 = c1hw // (height * width)
                shape_b_swap_c1_hw = [height * width, c1, n1, n0, c0]
                self.tensor_b = tvm.compute(shape_b_swap_c1_hw,
                                            lambda hw_idx, c1_idx, n1_idx, n0_idx, c0_idx:
                                            self.tensor_b(c1_idx * height * width + hw_idx, n1_idx, n0_idx, c0_idx),
                                            name="tensor_b_swap_c1_hw",
                                            attrs={"ori_batch_shape": self.para_dict.get("batch_shape_b")})
                self.shape_b = shape_b_swap_c1_hw

    def _compute_res(self, tensor_gemm):
        res_tag = self._get_ops_tag()
        attrs_dict = self._get_attrs_dict(tensor_gemm)
        shape_gemm = self._get_out_shape(tensor_gemm)
        attrs_dict["shape"] = shape_gemm
        # not_align flag is used for nd out
        not_align = (not self.align_a or not self.align_b) and self.format_out == "ND"
        # current cachetiling scenes both m/k/n axis is mutiply of 16
        not_align = not_align and not self.cache_tiling_flag

        if not_align:
            if self.need_reformat_to_nd and attrs_dict.get("shape")[-1] % self.block_in != 0:
                self.res = tensor_gemm
            elif len(shape_gemm) in (2, 3):
                self.res = tvm.compute(
                    shape_gemm,
                    lambda *indices: tvm.select(
                        indices[-2] < shape_gemm[-2],
                        tvm.select(indices[-1] < shape_gemm[-1], tensor_gemm(*indices))
                    ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                )
            # may needn't
            elif len(shape_gemm) in (4, 5):
                self.res = tvm.compute(
                    shape_gemm,
                    lambda *indices: tvm.select(
                        indices[-4] < shape_gemm[-4],
                        tvm.select(indices[-3] < shape_gemm[-3], tensor_gemm(*indices))
                    ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                )
        else:
            self.res = tvm.compute(
                shape_gemm,
                lambda *indices: tensor_gemm(*indices),
                    name="tensor_c_gm",
                    tag=res_tag,
                    attrs=attrs_dict
            )

    def _get_shapes(self):
        if in_dynamic():
            if self.input_range is None:
                m_shape, shape_a_k, n_shape = 65535, 1, 65535
            else:
                m_shape = self.input_range[-3][1]
                shape_a_k = self.input_range[-2][0]
                n_shape = self.input_range[-1][1]
        else:
            shape_a = [self._get_value(i) for i in self.tensor_a.shape]
            shape_b = [self._get_value(i) for i in self.tensor_b.shape]

            m_index = self.m_shape_dict.get(self.format_a)
            n_index = self.n_shape_dict.get(self.format_b)
            m_index = self.trans_dict.get(m_index) if self.trans_a else m_index
            n_index = self.trans_dict.get(n_index) if self.trans_b else n_index
            ka_index = self.trans_dict.get(m_index)
            m0_index = self.m0_shape_dict.get(self.format_a)
            ka0_index = self.trans_dict.get(m0_index)
            n0_index = self.n0_shape_dict.get(self.format_b)

            m_shape = shape_a[m_index]
            shape_a_k = shape_a[ka_index]
            n_shape = shape_b[n_index]
            if self.format_a != "ND":
                m_shape *= shape_a[m0_index]
                shape_a_k *= shape_a[ka0_index]
            if self.format_b != "ND":
                n_shape *= shape_b[n0_index]
        return (m_shape, shape_a_k, n_shape)

    def _get_attrs_dict(self, tensor_gemm):
        attrs_dict = {
            "shape": tensor_gemm.shape,
            "format": self.format_out,
            "fc_flag": self.fc_flag,
            "is_gemm_new": True
        }
        batch_dims = self._get_batch_dims(self.para_dict)
        attrs_dict["batch_shape_a"] = batch_dims[0]
        attrs_dict["batch_shape_b"] = batch_dims[1]
        attrs_dict["batch_shape"] = batch_dims[2]
        attrs_dict["batch_shape_out"] = batch_dims[2]
        return attrs_dict

    def _get_out_shape(self, tensor_gemm):
        shape_gemm = shape_util.shape_to_list(tensor_gemm.shape)
        shape_a = shape_util.shape_to_list(self.tensor_a.shape)
        shape_b = shape_util.shape_to_list(self.tensor_b.shape)
        shape_gemm_origin = []
        if len(shape_gemm) in (3, 5):
            shape_gemm_origin.append(shape_gemm[0])

        # default m_index is -3, and default n_index is -4
        # A tensor ND is [m, k], FRACTAL_Z is [m1, k1, m0, k0], FRACTAL_NZ is [k1, m1, m0, k0]
        # B tensor ND is [k, n], FRACTAL_Z is [k1, n1, n0, k0], FRACTAL_NZ is [n1, k1, k0, n0]
        m_index = self.m_shape_dict.get(self.format_a, -3)
        n_index = self.n_shape_dict.get(self.format_b, -4)
        m_index = self.trans_dict.get(m_index) if self.trans_a else m_index
        n_index = self.trans_dict.get(n_index) if self.trans_b else n_index
        ori_m_shape = shape_a[m_index]
        ori_n_shape = shape_b[n_index]

        if self.format_out == "ND":
            if self.format_a == "ND":
                shape_gemm_origin.append(ori_m_shape)
            else:
                shape_gemm_origin.append(ori_m_shape * self.block_in)
            if self.format_b == "ND":
                shape_gemm_origin.append(ori_n_shape)
            else:
                shape_gemm_origin.append(ori_n_shape * self.block_out)
        else:
            block_in = self.block_in
            block_out = self.block_out
            if self.format_a == "ND":
                if ori_m_shape == 1:
                    block_in = 1
                ori_m_shape = (ori_m_shape + block_in - 1) // block_in
            if self.format_b == "ND":
                if ori_n_shape == 1:
                    block_out = 1
                ori_n_shape = (ori_n_shape + block_out - 1) // block_out
            shape_gemm_origin.append(ori_n_shape)
            shape_gemm_origin.append(ori_m_shape)
            shape_gemm_origin.append(block_in)
            shape_gemm_origin.append(block_out)
            if self.mmad_mode == "gemv":
                shape_gemm_origin[-3], shape_gemm_origin[-4] = shape_gemm_origin[-4], shape_gemm_origin[-3]
                shape_gemm_origin[-1], shape_gemm_origin[-2] = shape_gemm_origin[-2], shape_gemm_origin[-1]

        return shape_gemm_origin

    def _get_ops_tag(self):
        if self.mmad_mode == "gemv":
            res_tag = "matmul_gemv"
        elif self.mmad_mode == "gevm":
            res_tag = "matmul_gevm"
        else:
            res_tag = "matmul"
        return res_tag

    def _init_func(self):
        # set data type and format, get data shape
        self._set_bias_or_c()
        self._set_info()
        self._set_ori_k()
        self._check_k_dim()
        self._check_n_align()
        self._check_gevm_and_gemv()
        self._set_mmad_mode()
        self._set_alpha_and_beta_mode()
        self._check_quantize_params(self.quantize_params)
        self._need_split_k()

    def _need_split_k(self):
        # this func will replace by the info from ub fusion
        is_gemm  = self.alpha is not None
        have_bias = self.tensor_bias is not None
        not_fp16_in_fp32_out = (self.src_dtype != "float16") or (self.dst_dtype != "float32")
        not_fractal_nz_out = self.format_out != "FRACTAL_NZ"

        shape_a = [self._get_value(i) for i in self.tensor_a.shape]
        shape_b = [self._get_value(i) for i in self.tensor_b.shape]
        have_batch = False
        if (len(shape_a) in (3, 5)) or (len(shape_b) in (3, 5)):
            have_batch = True
        compress_flag = self.compress_index is not None
        not_split_k = (not_fp16_in_fp32_out or not_fractal_nz_out or is_gemm or have_bias or self.is_fusion
                       or self.fc_flag or compress_flag)
        if self.cache_tiling_flag:
            self.split_k = not not_split_k
            return
        not_split_k = not_split_k or have_batch
        if not_split_k:
            return

        blocks = (self.block_in, self.block_reduce, self.block_out)
        compute_perf_core_num = GetPerfCoreNum()
        block_dims = compute_perf_core_num.get_best_perf_factor(self._get_shapes(), blocks)
        self.best_split_k_block_dim = block_dims
        if block_dims[1] != 1:
            self.split_k = True

    def _set_alpha_and_beta_mode(self):
        alpha = self.alpha
        beta = self.beta
        if alpha is None:
            alpha_mode = "none"
        elif not isinstance(alpha, Tensor):
            if math.isclose(alpha, 1.0):
                alpha_mode = "not_mul"
            else:
                alpha_mode = "mul_num"
        else:
            alpha_mode = "mul_tensor"

        if beta is None:
            beta_mode = "none"
        elif not isinstance(beta, Tensor):
            if math.isclose(beta, 1.0):
                beta_mode = "not_mul"
            else:
                beta_mode = "mul_num"
        else:
            beta_mode = "mul_tensor"

        self.beta_mode = beta_mode
        self.alpha_mode = alpha_mode

    def _get_tensor_alpha_mmad(self, tensor_mmad):
        shape_mmad = [self._get_value(i) for i in tensor_mmad.shape]

        if self.alpha_mode == "none":
            return tensor_mmad
        elif self.alpha_mode == "mul_num":
            alpha = self.alpha
            dtype_mmad = tensor_mmad.dtype
            tensor_alpha_mmad = tvm.compute(
                shape_mmad,
                lambda *indices: tvm.convert(alpha).astype(dtype_mmad) * tensor_mmad(*indices),
                name="tensor_alpha_mmad"
            )
        else:
            tensor_alpha = self.alpha
            if tensor_alpha.dtype == "float16":
                tensor_alpha = tvm.compute(
                    tensor_alpha.shape,
                    lambda *indices: shape_util.cast(
                        tensor_alpha(*indices), dtype="float32"
                    ),
                    name="tensor_alpha_f162f32"
                )
            tensor_alpha_mmad = tvm.compute(
                shape_mmad,
                lambda *indices: tensor_mmad(*indices) * tensor_alpha[0],
                name="tensor_alpha_mmad"
            )

        return tensor_alpha_mmad

    def _is_nd_int82fp32(self):
        return (self.format_b == "ND") and (self.format_a == "ND") and (self.ops_data_flow_mode == "int82fp32")

    def _check_align_a(self):
        """
        if src_dtype is fp16, m and k both aligned to 16
        if src_dtype is int8 dst_dtype is fp32, m and k both aligned to 32
        if src_dtype is int8 dst_dtype is int32, m aligned to 16 k aligned to 32
        """
        is_nd_int82fp32 = self._is_nd_int82fp32()
        if self.trans_a:
            index_m = -1
            index_km = -2
        else:
            index_m = -2
            index_km = -1
        ori_m_shape = self._get_value(self.tensor_a.shape[index_m])
        ori_km_shape = self._get_value(self.tensor_a.shape[index_km])
        if is_nd_int82fp32:
            m_shape = int_ceil_div(ori_m_shape, 32) * 32 // 16
            km_shape = int_ceil_div(ori_km_shape, 32) * 32 // 16
        else:
            if self.src_dtype in ("uint8", "int8"):
                m_shape = int_ceil_div(ori_m_shape, 32) * 32 // 16
                if self.int8_not_double_m:
                    m_shape = int_ceil_div(ori_m_shape, self.block_in)
            else:
                m_shape = int_ceil_div(ori_m_shape, self.block_in)
            km_shape = int_ceil_div(ori_km_shape, self.block_reduce)

        shape_a_aligned = [m_shape * self.block_in, km_shape * self.block_reduce]
        align_flag_a = (ori_m_shape == shape_a_aligned[-2]) and (ori_km_shape == shape_a_aligned[-1])
        shape_a_aligned = shape_a_aligned[::-1] if self.trans_a else shape_a_aligned
        if in_dynamic():
            align_flag_a = False
        return align_flag_a, shape_a_aligned

    def _check_align_b(self):
        """
        if src_dtype is fp16, n and k both aligned to 16
        if src_dtype is int8 dst_dtype is fp32, n and k both aligned to 32
        if src_dtype is int8 dst_dtype is int32, n aligned to 16 k aligned to 32
        """
        is_nd_int82fp32 = self._is_nd_int82fp32()
        if self.trans_b:
            index_n = -2
            index_kn = -1
        else:
            index_n = -1
            index_kn = -2
        ori_n_shape = self._get_value(self.tensor_b.shape[index_n])
        ori_kn_shape = self._get_value(self.tensor_b.shape[index_kn])
        if is_nd_int82fp32:
            n_shape = int_ceil_div(ori_n_shape, 32) * 32 // 16
            kn_shape = int_ceil_div(ori_kn_shape, 32) * 32 // 16
        else:
            if self.src_dtype in ("uint8", "int8"):
                n_shape = int_ceil_div(ori_n_shape, 32) * 32 // 16
            else:
                n_shape = int_ceil_div(ori_n_shape, self.block_out)
            kn_shape = int_ceil_div(ori_kn_shape, self.block_reduce)

        shape_b_aligned = [kn_shape * self.block_reduce, n_shape * self.block_out]
        align_flag_b = (ori_n_shape == shape_b_aligned[-1]) and (ori_kn_shape == shape_b_aligned[-2])
        shape_b_aligned = shape_b_aligned[::-1] if self.trans_b else shape_b_aligned
        if in_dynamic():
            align_flag_b = False
        return align_flag_b, shape_b_aligned

    def _do_align_nd_shape(self, tensor_need_align, tensor_name, in_dtype, need_check_align=False):
        """
        do align for a matrix or b matrix, pad zero along the way
        input:
            tensor_need_align: tensor, the tensor need align
            tensor_name: str, a or b
            in_dtype: str, input data type
        return:
            aligned tensor
        """

        if tensor_name == "a":
            is_align, shape_aligned = self._check_align_a()
            self.align_a = is_align
        else:
            is_align, shape_aligned = self._check_align_b()
            self.align_b = is_align
        not_need_align = is_align or (self.mmad_mode in ("gemv", "gevm"))
        not_need_align = False if in_dynamic() else not_need_align
        # current cachetiling scenes both m/k/n axis is mutiply of 16
        not_need_align = True if self.cache_tiling_flag else not_need_align
        use_aligned_pattern = not_need_align or (not need_check_align)
        if in_dynamic():
            tensor_aligned = self._do_shape_aligned_for_dynamic(
                tensor_need_align, shape_aligned, tensor_name, in_dtype, use_aligned_pattern=use_aligned_pattern)
        else:
            tensor_aligned = self._do_shape_aligned_for_static(
                tensor_need_align, shape_aligned, tensor_name, in_dtype, use_aligned_pattern=use_aligned_pattern)
        return tensor_aligned

    def _do_align_nd_shape_for_bias(self, tensor_need_align, in_dtype):
        """
        do align for tensor_bias, pad zero along the way
        input:
            tensor_need_align: tensor, the tensor need align
            in_dtype: str, input data type
        return:
            aligned tensor
        """
        if len(tensor_need_align.shape) in (4, 5):
            is_align = True
            shape_aligned = tensor_need_align.shape
        else:
            ori_m_shape = self._get_value(tensor_need_align.shape[-2])
            ori_n_shape = self._get_value(tensor_need_align.shape[-1])
            m_shape = int_ceil_div(ori_m_shape, self.block_in) * self.block_in
            n_shape = int_ceil_div(ori_n_shape, self.block_out) * self.block_out
            is_align = (ori_m_shape == m_shape) and (ori_n_shape == n_shape)
            shape_aligned = [m_shape, n_shape]

        not_need_align = is_align or (self.mmad_mode in ("gemv", "gevm"))
        use_aligned_pattern = False if in_dynamic() else not_need_align
        tensor_aligned = self._do_shape_aligned_for_static(
            tensor_need_align, shape_aligned, "bias", in_dtype, use_aligned_pattern=use_aligned_pattern)
        return tensor_aligned

    def _do_shape_aligned_for_static(self, tensor_need_align, shape_aligned, tensor_name,
                                     in_dtype, use_aligned_pattern=False):
        """
        do align for tensor_a_zz , tensor_b_zn and tensor_bias
        input:
            tensor_need_align: tensor, the tensor need align
            tensor_name: str, a , b or bias
            in_dtype: str, input data type
            use_aligned_pattern: bool, is this tensor aligned
        return:
            aligned tensor
        """
        if use_aligned_pattern:
            tensor_aligned = tvm.compute(
                tensor_need_align.shape,
                lambda *indices: tensor_need_align(*indices),
                name="tensor_{}_aligned".format(tensor_name)
            )
            if "ori_batch_shape" in tensor_need_align.op.attrs:
                tensor_aligned.op.attrs["ori_batch_shape"] = tensor_need_align.op.attrs["ori_batch_shape"]
            return tensor_aligned

        ori_shape = [self._get_value(i) for i in tensor_need_align.shape]
        shape_aligned = ori_shape[:-2] + shape_aligned

        tensor_aligned = tvm.compute(shape_aligned,
            lambda *indices: tvm.select(indices[-2] < ori_shape[-2],
                tvm.select(
                    indices[-1] < ori_shape[-1],
                    tensor_need_align(*indices),
                    tvm.convert(0).astype(in_dtype)
                ),
                tvm.convert(0).astype(in_dtype)
            ),
            name="tensor_{}_aligned".format(tensor_name)
        )
        if self.int8_not_double_m and tensor_name == "a":
            # virtual_align means use aligned shape, but do not pad zero.
            tensor_aligned.op.attrs["virtual_align"] = True
        return tensor_aligned

    def _do_shape_aligned_for_dynamic(self, tensor_need_align, shape_aligned, tensor_name,
                                      in_dtype, use_aligned_pattern=False):
        """
        do align for tensor_a_zz or tensor_b_zn, pad zero along the way in dynamic mode
        input:
            tensor_need_align: tensor, the tensor need align
            tensor_name: str, a or b
            in_dtype: str, input data type
            use_aligned_pattern: bool is this tensor aligned
        return:
            aligned tensor
        """
        ori_shape = [self._get_value(i) for i in tensor_need_align.shape]
        # Use a virtual Node [tensor_aligned] which will be mapped as "phony_insn".
        # At schedule stage, either aligned or general pattern will be selected for execution,
        # and the remaining one will be mapped as "phony_insn".
        shape_aligned = ori_shape[:-2] + shape_aligned

        tensor_already_aligned = tvm.compute(
            shape_aligned,
            lambda *indices: tensor_need_align(*indices),
            name="tensor_{}_already_aligned".format(tensor_name)
        )
        tensor_do_align = tvm.compute(
            shape_aligned,
            lambda *indices: tvm.select(
                indices[-2] < ori_shape[-2],
                tvm.select(
                    indices[-1] < ori_shape[-1],
                    tensor_need_align(*indices),
                    tvm.convert(0).astype(in_dtype)
                ),
                tvm.convert(0).astype(in_dtype)
            ),
            name="tensor_{}_do_align".format(tensor_name)
        )
        tensor_aligned = tvm.compute(
            shape_aligned,
            lambda *indices: tensor_already_aligned(*indices)
                + tensor_do_align(*indices),
            name="tensor_{}_aligned".format(tensor_name),
            attrs={"use_aligned_pattern": use_aligned_pattern}
        )

        return tensor_aligned

    def _get_tensor_a_zz(self):
        """ compute a matrix for mad
        Input: None
        support func:
            fp16 input:
                (m, k)
        Return:
            the tensor format is Zz for mad
        """
        need_check_align = (self.format_a in ("ND", "NC1HWC0")) and (self.mmad_mode != "gevm")
        not_pad_for_int8_nd = self.tensor_a.dtype in ("int8", "uint8") and \
                              self.format_a == "ND" and (self.alpha is None)
        self.int8_not_double_m = not_pad_for_int8_nd
        if self.format_a in ("ND", "NC1HWC0"):
            tensor_a_aligned = self._do_align_nd_shape(self.tensor_a, "a", self.src_dtype, need_check_align)
        else:
            tensor_a_aligned = self.tensor_a
        shape_a_aligned = [self._get_value(i) for i in tensor_a_aligned.shape]

        # ------------------- GEVM(ND)/GEMV process ------------------- #
        # not support int82fp32
        if self.mmad_mode == "gemv":
            return self._get_tensor_a_zz_gemv(tensor_a_aligned)
        elif self.mmad_mode == "gevm" and self.format_a in ("ND", "NC1HWC0"):
            return self._get_tensor_a_zz_gevm(tensor_a_aligned)

        # ------------------------------------------------------------ #
        if self.ops_data_flow_mode == "int82fp32":
            tensor_a_aligned = tvm.compute(
                shape_a_aligned,
                lambda *indices: shape_util.cast(tensor_a_aligned(*indices), "float16"),
                name="tensor_a_s82f16"
            )

        if self.format_a in ("FRACTAL_Z", "FRACTAL_NZ"):
            # ---------------------- fractal process --------------------- #
            return self._get_tensor_a_frac2zz(tensor_a_aligned)
        # ------------------------ ND process ------------------------ #
        return self._get_tensor_a_nd2zz(tensor_a_aligned, not_pad_for_int8_nd)

    def _get_tensor_a_zz_gemv(self, tensor_a_aligned):
        compute_params_gemv = {"tensor_name": "tensor_a_zz", "trans": self.trans_a}
        if self.format_a == "FRACTAL_NZ":
            compute_params_gemv["mode_info"] = "fractal_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Nz",
                "format_in_a_ub": "none"
            }
            tensor_a_zz = self.fract_change_inner_axis(tensor_a_aligned, compute_params_gemv)
        elif self.format_a == "FRACTAL_Z":
            compute_params_gemv["tensor_name"] = "tensor_a_zz"
            compute_params_gemv["mode_info"] = "fractal_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "none"
            }
            tensor_a_zz = self.fract_change_outer_axis(tensor_a_aligned, compute_params_gemv)
        else:
            compute_params_gemv["trans"] = False
            compute_params_gemv["block_in"] = self.block_in
            compute_params_gemv["block_reduce"] = self.block_reduce
            compute_params_gemv["tensor_name"] = "tensor_a_nd2zz"
            tensor_a_nd2zz = self.compute_nd2zz(tensor_a_aligned, compute_params_gemv)
            compute_params_gemv["trans"] = self.trans_a
            compute_params_gemv["mode_info"] = "nd_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "nd"
            }
            compute_params_gemv["tensor_name"] = "tensor_a_zz"
            tensor_a_zz = self.fract_change_outer_axis(tensor_a_nd2zz, compute_params_gemv)
        return tensor_a_zz

    def _get_tensor_a_zz_gevm(self, tensor_a_aligned):
        compute_params_gemv = {"tensor_name": "tensor_a_zz", "trans": self.trans_a}
        compute_params_gemv["trans"] = self.trans_a
        compute_params_gemv["block_in"] = self.block_in
        compute_params_gemv["block_reduce"] = self.block_reduce
        compute_params_gemv["tensor_name"] = "tensor_a_nd2zz"
        compute_params_gemv["mode_info"] = "nd_gevm"
        compute_params_gemv["format_info"] = {
            "format_in_a_l1": "Zz",
            "format_in_a_ub": "nd"
        }
        tensor_a_zz = self.compute_nd2zz_gevm(tensor_a_aligned, compute_params_gemv)

        return tensor_a_zz

    def _get_tensor_a_frac2zz(self, tensor_a_aligned):
        compute_params_fractal = {"tensor_name": "tensor_a_zz", "trans": self.trans_a}
        # format_a is FRACTAL_Z
        if self.format_a == "FRACTAL_Z":
            if self.trans_a:
                compute_params_fractal["mode_info"] = "Zz_trans"
                compute_params_fractal["format_info"] = {
                    "format_in_a_l1": "Zz",
                    "format_in_a_ub": "none"
                }
                compute_params_fractal["trans"] = False
                tensor_a_aligned = self.fract_change_both_axis(tensor_a_aligned, compute_params_fractal)
            return tensor_a_aligned
        # format_a is FRACTAL_NZ
        if self.ops_data_flow_mode == "int82fp32":
            compute_params_fractal["mode_info"] = "Nz2Zz_int82fp32"
            compute_params_fractal["format_info"] = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "Nz"
            }
            tensor_a_zz = self.compute_nz2zz_int82fp32(tensor_a_aligned, compute_params_fractal)
        else:
            compute_params_fractal["mode_info"] = "Nz2Zz"
            compute_params_fractal["format_info"] = {
                "format_in_a_l1": "Nz",
                "format_in_a_ub": "none"
            }
            tensor_a_zz = self.fract_change_outer_axis(tensor_a_aligned, compute_params_fractal)
            if "ori_batch_shape" in tensor_a_aligned.op.attrs:
                tensor_a_zz.op.attrs["ori_batch_shape"] = tensor_a_aligned.op.attrs["ori_batch_shape"]
        return tensor_a_zz

    def _get_tensor_a_nd2zz(self, tensor_a_aligned, use_normal_func):
        nd_to_zz_normal_flag = self.cube_vector_split or self.ops_data_flow_mode in ("int82int32", "int4int32") or \
                               self.mmad_mode == "gevm" or use_normal_func or in_dynamic()
        nd_to_zz_vnchwconv_flag = self.ops_data_flow_mode not in ("int82int32", "int4int32")

        compute_params = {
            "tensor_name": "tensor_a_zz",
            "block_in": self.block_in,
            "block_reduce": self.block_reduce,
            "data_flow": self.ops_data_flow_mode,
            "trans": self.trans_a
        }
        if nd_to_zz_normal_flag:
            mode_info = "nd2Zz" if (self.ops_data_flow_mode != "int82int32") else "nd2Zz_int8"
            compute_params["mode_info"] = mode_info
            format_in_a_l1 = "Zz"
            if self.trans_a and (self.ops_data_flow_mode != "int82int32"):
                format_in_a_l1 = "Nn"
            compute_params["format_info"] = {
                "format_in_a_l1": format_in_a_l1,
                "format_in_a_ub": "nd"
            }
            tensor_a_zz = self.compute_nd2zz(tensor_a_aligned, compute_params)
        elif nd_to_zz_vnchwconv_flag:
            compute_params["mode_info"] = "nd2Zz_vnchwconv"
            compute_params["format_info"] = {
                "format_in_a_l1": "three_axis",
                "format_in_a_ub": "nd"
            }
            tensor_a_zz = self.compute_nd2zz_vnchwconv(tensor_a_aligned, compute_params)
            if "ori_batch_shape" in tensor_a_aligned.op.attrs:
                tensor_a_zz.op.attrs["ori_batch_shape"] = tensor_a_aligned.op.attrs["ori_batch_shape"]
        return tensor_a_zz

    def _get_tensor_b_zn(self):
        tensor_b = self.tensor_b

        # ---------------------- compress process ---------------------- #
        # noly support fp16 input
        compress_flag = self.compress_index is not None
        self.compress_flag = compress_flag
        if compress_flag:
            tensor_b_zn = self._get_compress_tensor_compute(tensor_b, self.compress_index, "tensor_b_zn")
            self.format_b = "FRACTAL_Z"
            return tensor_b_zn

        need_check_align = (self.format_b in ("ND", "NC1HWC0")) and (self.mmad_mode != "gemv")
        if self.format_b == "ND":
            tensor_b_aligned = self._do_align_nd_shape(tensor_b, "b", self.src_dtype, need_check_align)
        else:
            tensor_b_aligned = tensor_b
        shape_b_aligned = [self._get_value(i) for i in tensor_b_aligned.shape]

        # ---------------------- GEMV process ---------------------- #
        if self.mmad_mode == "gemv":
            return self._get_tensor_b_zn_gemv(tensor_b_aligned)

        # ---------------------------------------------------------- #
        if self.ops_data_flow_mode == "int82fp32":
            tensor_b_aligned = tvm.compute(
                shape_b_aligned,
                lambda *indices: shape_util.cast(tensor_b_aligned(*indices), "float16"),
                name="tensor_b_s82f16"
            )

        # ------------------------ fractal process ----------------------- #
        if self.format_b in ("FRACTAL_Z", "FRACTAL_NZ"):
            return self._get_tensor_b_frac2zn(tensor_b_aligned)

        # -------------------------- ND process -------------------------- #
        return self._get_tensor_b_nd2zn(tensor_b_aligned)

    def _get_tensor_b_zn_gemv(self, tensor_b_aligned):
        compute_params_gemv = {"tensor_name": "tensor_b_zn", "trans": self.trans_b}
        compute_params_gemv["mode_info"] = "fractal_gemv"
        compute_params_gemv["format_info"] = {
            "format_in_b_l1": "Zn",
            "format_in_b_ub": "none"
        }
        if self.format_b in ("FRACTAL_Z", "FRACTAL_NZ"):
            if self.format_b == "FRACTAL_NZ":
                compute_params_gemv["format_info"] = {
                    "format_in_b_l1": "Nz",
                    "format_in_b_ub": "none"
                }
                self.trans_b = not self.trans_b
            compute_params_gemv["trans"] = self.trans_b
            tensor_b_zn = self.fract_change_outer_axis(tensor_b_aligned, compute_params_gemv)
        else:
            compute_params_gemv["tensor_name"] = "tensor_b_nd2zz"
            compute_params_gemv["trans"] = False
            tensor_b_nd2zz = self.compute_nd2zz(tensor_b_aligned, compute_params_gemv)
            compute_params_gemv["mode_info"] = "nd_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_b_l1": "Zz",
                "format_in_b_ub": "nd"
            }
            compute_params_gemv["trans"] = self.trans_b
            compute_params_gemv["tensor_name"] = "tensor_b_zn"
            tensor_b_zn = self.fract_change_both_axis(tensor_b_nd2zz, compute_params_gemv)

        return tensor_b_zn

    def _get_tensor_b_frac2zn(self, tensor_b_aligned):
        compute_params_fractal = {"tensor_name": "tensor_b_zn", "trans": self.trans_b}
        # format_b is FRACTAL_Z
        if self.format_b == "FRACTAL_Z":
            if self.ops_data_flow_mode == "int82fp32":
                compute_params_fractal["mode_info"] = "Zn2Zn_int82fp32"
                compute_params_fractal["format_info"] = {
                    "format_in_b_l1": "Zn",
                    "format_in_b_ub": "Zn"
                }
                tensor_b_zn = self.compute_zn2zn_int82fp32(tensor_b_aligned, compute_params_fractal)
            elif self.trans_b:
                compute_params_fractal["mode_info"] = "Zn_trans"
                compute_params_fractal["format_info"] = {
                    "format_in_b_l1": "Zn",
                    "format_in_b_ub": "none"
                }
                compute_params_fractal["trans"] = False
                tensor_b_zn = self.fract_change_both_axis(tensor_b_aligned, compute_params_fractal)
            else:
                tensor_b_zn = tensor_b_aligned
            return tensor_b_zn
        # format_b is FRACTAL_NZ
        compute_params_fractal["mode_info"] = "Nz2Zn"
        compute_params_fractal["format_info"] = {
            "format_in_b_l1": "Nz",
            "format_in_b_ub": "none"
        }
        tensor_b_zn = self.fract_change_both_axis(tensor_b_aligned, compute_params_fractal)
        if "ori_batch_shape" in tensor_b_aligned.op.attrs:
            tensor_b_zn.op.attrs["ori_batch_shape"] = tensor_b_aligned.op.attrs["ori_batch_shape"]
        return tensor_b_zn

    def _get_tensor_b_nd2zn(self, tensor_b_aligned):
        nd_to_zn_normal_flag = self.cube_vector_split or self.ops_data_flow_mode == "int82int32" or in_dynamic()
        nd_to_zn_vnchwconv_flag = self.ops_data_flow_mode != "int82int32"

        compute_params = {
            "tensor_name": "tensor_b_zn",
            "block_in": self.block_in,
            "block_reduce": self.block_reduce,
            "block_out": self.block_out,
            "data_flow": self.ops_data_flow_mode,
            "trans": self.trans_b
        }
        if nd_to_zn_normal_flag:
            compute_params["mode_info"] = "nd2Zn" if self.ops_data_flow_mode != "int82int32" else "nd2Zn_int8"
            format_in_b_l1 = "Zn"
            if self.trans_b and (self.ops_data_flow_mode != "int82int32"):
                format_in_b_l1 = "Nz"
            compute_params["format_info"] = {
                "format_in_b_l1": format_in_b_l1,
                "format_in_b_ub": "nd"
            }
            tensor_b_zn = self.compute_nd2zn(tensor_b_aligned, compute_params)
        elif nd_to_zn_vnchwconv_flag:
            compute_params["mode_info"] = "nd2Zn_vnchwconv"
            compute_params["format_info"] = {
                "format_in_b_l1": "three_axis",
                "format_in_b_ub": "nd"
            }
            tensor_b_zn = self.compute_nd2zn_vnchwconv(tensor_b_aligned, compute_params)
            if "ori_batch_shape" in tensor_b_aligned.op.attrs:
                tensor_b_zn.op.attrs["ori_batch_shape"] = tensor_b_aligned.op.attrs["ori_batch_shape"]
        return tensor_b_zn

    def _get_tensor_beta_bias(self):
        tensor_bias = self.tensor_c
        tensor_beta = self.beta
        if (tensor_bias is None) or (self.beta_mode not in ("mul_num", "mul_tensor")):
            return None
        if tensor_beta.dtype == "float16":
            tensor_beta = tvm.compute(
                tensor_beta.shape,
                lambda *indices: shape_util.cast(
                    tensor_beta(*indices), dtype="float32"
                ),
                name="tensor_beta_f162f32"
            )

        tensor_bias_aligned = self._do_align_nd_shape_for_bias(tensor_bias, self.dst_dtype)

        shape_bias_aligned = tensor_bias_aligned.shape
        if self.mmad_mode == "gemv":
            shape_bias_aligned_temp = [
                shape_bias_aligned[-3],
                shape_bias_aligned[-4],
                shape_bias_aligned[-1],
                shape_bias_aligned[-2]
            ]
            if len(shape_bias_aligned) == 5:
                shape_bias_aligned_temp.insert(0, shape_bias_aligned[0])
            shape_bias_aligned = shape_bias_aligned_temp

        if tensor_bias_aligned.dtype == "float16":
            tensor_bias_aligned = tvm.compute(
                shape_bias_aligned,
                lambda *indices: shape_util.cast(tensor_bias_aligned(*indices), dtype="float32"),
                name="tensor_b_f32"
            )

        if self.mmad_mode == "gemv":
            tensor_beta_bias = tvm.compute(
                shape_bias_aligned,
                lambda i, j, k, l: tensor_beta[0] * tensor_bias_aligned[j, i, l, k],
                name="tensor_beta_bias"
            )
        else:
            tensor_beta_bias = tvm.compute(
                shape_bias_aligned,
                lambda *indices: tensor_beta[0] * tensor_bias_aligned(*indices),
                name="tensor_beta_bias"
            )
        return tensor_beta_bias

    def _get_shape_bias(self, tensor_bias):
        shape_bias = []
        if self._get_value(self._elecnt_of_shape(tensor_bias.shape)) == 1:
            shape_bias = [1]
        else:
            for i in tensor_bias.shape:
                if shape_bias:
                    shape_bias.append(self._get_value(i))
                elif self._get_value(i) != 0 and self._get_value(i) != 1:
                    # first element value should be > 1
                    shape_bias.append(self._get_value(i))
        return shape_bias

    def _get_tensor_bias_nz(self, tensor_bias, shape_mmad):
        ori_shape = tbe_utils.shape_to_list(tensor_bias.op.attrs['ori_shape'])
        block_out = self.block_out
        shape_bias = self._get_shape_bias(tensor_bias)

        # only support [n], [1,n], [1,1,n]
        def index_bias_of_ori_shape(indices):
            return [0]*(len(shape_bias) - 1) + [indices[-1]]

        def index_bias_of_fractal_nz(indices):
            return [0]*(len(shape_bias) - 1) + [indices[-4]*block_out + indices[-1]]
        # dynamic mode only support bias align to 16
        if in_dynamic() or ori_shape[-1] % 16 == 0:
            tensor_bias_ub = tvm.compute(
                shape_bias, lambda *indices: tensor_bias(*index_bias_of_ori_shape(indices)), name="tensor_bias_ub")
        else:
            tensor_bias_ub = tvm.compute(
                shape_bias, lambda *indices:
                tvm.select(indices[-1] < ori_shape[-1], tensor_bias(*index_bias_of_ori_shape(indices))),
                           name="tensor_bias_ub")
            tensor_init_value_of_bias_ub = tvm.compute(
                shape_bias, lambda *indices:
                tvm.select(indices[-1] >= ori_shape[-1], tvm.const(0, dtype=tensor_bias.op.dtype)),
                           name="tensor_init_value_of_bias_ub")
            tensor_virtual_add_bias = tvm.compute(
                shape_bias, lambda *indices: tensor_bias_ub[indices]+tensor_init_value_of_bias_ub[indices],
                name="tensor_virtual_add_bias")
            tensor_bias_ub = tensor_virtual_add_bias

        # shape_mmad is batch, n1, m1, m0, n0
        if tensor_bias.dtype == "float16" and self.l0c_support_fp32:
            tensor_bias_nz = tvm.compute(
                shape_mmad, lambda *indices: shape_util.cast(
                    tensor_bias_ub(*index_bias_of_fractal_nz(indices)), dtype="float32"), name="tensor_bias_nz")
        else:
            tensor_bias_nz = tvm.compute(
                shape_mmad,
                lambda *indices: tensor_bias_ub(*index_bias_of_fractal_nz(indices)), name="tensor_bias_nz")

        return tensor_bias_nz

    def _get_tensor_mmad(self, tensor_a_zz, tensor_b_zn, tensor_bias):
        """ MatMul calculation
        Input:
            tensor_a_zz: tensor, a matrix in l0a
            tensor_b_zn: tensor, b matrix in l0b
        support func:
            MatMul, Nz->ND
        Return:
            tensor, MatMul result
        """

        attrs_dict = {"ops_format": self.ops_format,
                      "format_a": self.format_a,
                      "format_b": self.format_b,
                      "ops_data_flow_mode": self.ops_data_flow_mode,
                      "kernel_name": self.kernel_name,
                      "mmad_mode": self.mmad_mode,
                      "only_use_gevm_gemv_flow": self.only_use_gevm_gemv_flow,
                      "int8_not_double_m": self.int8_not_double_m,
                      "have_bias": self.have_bias,
                      "have_c": self.have_c,
                      "transpose_a": self.trans_a,
                      "transpose_b": self.trans_b,
                      "align_a": self.align_a,
                      "align_b": self.align_b,
                      "format_out": self.format_out,
                      "placeholder_name": self.placeholder_name,
                      "compress_flag": self.compress_flag,
                      "split_k": int(self.split_k)}

        if self.best_split_k_block_dim != []:
            attrs_dict["custom_block_dim_m"] = self.best_split_k_block_dim[0]
            attrs_dict["custom_block_dim_k"] = self.best_split_k_block_dim[1]
            attrs_dict["custom_block_dim_n"] = self.best_split_k_block_dim[2]

        k_shape_l0 = tensor_b_zn.shape[-3] if self.mmad_mode == "gemv" else tensor_a_zz.shape[-3]
        reduce_kp, reduce_kb = self._get_reduce(k_shape_l0)
        need_add_bias_in_fb = self.have_bias and self.cube_vector_split

        tensor_mmad = self.tvm_compute_mad(
            tensor_a_zz,
            tensor_b_zn,
            tensor_bias,
            reduce_kb,
            reduce_kp,
            self.matrix_type,
            self.mmad_mode,
            "tensor_mmad",
            need_add_bias_in_fb,
            self.offset_x,
            attrs_dict
        )

        need_add_bias_in_l0c = self.have_bias and (not self.cube_vector_split)
        if need_add_bias_in_l0c:
            tensor_bias_nz = self._get_tensor_bias_nz(tensor_bias, tensor_mmad.shape)
            tensor_mmad_with_bias = tvm.compute(tensor_bias_nz.shape, lambda *indices:
                                                tensor_bias_nz[indices] +
                                                tensor_mmad[indices],
                                                name="tensor_mmad_with_bias")
            return tensor_mmad_with_bias
        return tensor_mmad

    def _get_tensor_gemm(self, tensor_alpha_mmad, tensor_beta_bias):
        cur_format_out = "FRACTAL_NZ"
        if tensor_beta_bias is not None:
            shape_beta_bias_aligned = [self._get_value(i) for i in tensor_beta_bias.shape]
            if self.format_out == "ND":
                tensor_gemm = self.tvm_compute_nd_add_nz_to_nd(
                    tensor_beta_bias,
                    tensor_alpha_mmad,
                    "tensor_gemm"
                )
                cur_format_out = "ND"
            else:
                if self.ops_data_flow_mode == "int82int32":
                    tensor_gemm = self.tvm_compute_nd_add_nz_to_nz(
                        tensor_beta_bias,
                        tensor_alpha_mmad,
                        "tensor_gemm"
                    )
                else:
                    tensor_gemm = tvm.compute(
                        shape_beta_bias_aligned,
                        lambda *indices: tensor_beta_bias(*indices) + tensor_alpha_mmad(*indices),
                        "tensor_gemm"
                    )
        else:
            tensor_gemm = tensor_alpha_mmad

        if self.need_cast_to_fp16:
            tensor_gemm = tvm.compute(
                tensor_gemm.shape,
                lambda *indices: shape_util.cast(tensor_gemm(*indices), dtype="float16"),
                name="tensor_gemm_f16"
            )

        need_reformat_to_nd = ((self.format_out == "ND") and (cur_format_out == "FRACTAL_NZ"))
        self.need_reformat_to_nd = need_reformat_to_nd
        if need_reformat_to_nd:
            attrs_dict = self._get_attrs_dict(tensor_gemm)
            shape_gemm = self._get_out_shape(tensor_gemm)
            attrs_dict["shape"] = shape_gemm
            if not self.cache_tiling_flag and attrs_dict.get("shape")[-1] % self.block_in != 0:
                tensor_gemm_nz2nd = self.compute_nz2nd(tensor_gemm, output_shape=shape_gemm,
                                                       tensor_name="tensor_c_gm", res_tag=self._get_ops_tag(),
                                                       attrs_dict=attrs_dict)
            else:
                tensor_gemm_nz2nd = self.compute_nz2nd(tensor_gemm)
            return tensor_gemm_nz2nd
        return tensor_gemm

    def _get_reduce(self, k_shape):
        """get reduce axis for MatMul
        Input:
            k_shape: A matrix's k
        Return:
            axis, two reduce axis
        """

        block_reduce = self.block_reduce
        reduce_kp = tvm.reduce_axis((0, block_reduce), name="kp")
        reduce_kb = tvm.reduce_axis((0, k_shape), name="kb")

        return reduce_kp, reduce_kb

    def _set_info(self):
        """set info about
        1. placeholder name
        2. format
        Input: None
        Return:None
        """
        # ---------- place holder name ---------- #
        placeholder_name = {
            "a": self.tensor_a.op.name,
            "b": self.tensor_b.op.name
        }
        placeholder_name["bias"] = "none"
        if self.tensor_bias is not None:
            placeholder_name["bias"] = self.tensor_bias.op.name
        placeholder_name["c"] = "none"
        if self.tensor_c is not None:
            placeholder_name["c"] = self.tensor_c.op.name
        placeholder_name["alpha"] = "none"
        if self.alpha is not None:
            placeholder_name["alpha"] = self.alpha.op.name
        placeholder_name["beta"] = "none"
        if self.beta is not None:
            placeholder_name["beta"] = self.beta.op.name
        self.placeholder_name = placeholder_name

        self.have_bias = self.tensor_bias is not None
        self.have_c = self.tensor_c is not None
        # ---------- handle format ---------- #
        merge_format_dict = {
            "ND": "ND",
            "NC1HWC0": "ND",
            "fractal": "FRACTAL_Z",
            "FRACTAL_NZ": "FRACTAL_NZ",
            "FRACTAL_Z": "FRACTAL_Z"
        }
        format_a_temp = merge_format_dict.get(self.format_a)
        if format_a_temp is None:
            reason = "The current format_a is not supported, format_a is {}.".format(self.format_a)
            error_manager_cube.raise_err_specific("GEMM", reason)
        self.format_a = format_a_temp

        format_b_temp = merge_format_dict.get(self.format_b)
        if format_b_temp is None:
            reason = "The current format_b is not supported, format_b is {}.".format(self.format_b)
            error_manager_cube.raise_err_specific("GEMM", reason)
        self.format_b = format_b_temp

        self._set_output_format()
        merge_format_output_dict = {
            "ND": "ND",
            "NC1HWC0": "NC1HWC0",
            "FRACTAL_NZ": "FRACTAL_NZ"
        }
        format_out_temp = merge_format_output_dict.get(self.format_out)
        if format_out_temp is None:
            reason = "The current format_out is not supported, format_out is {}.".format(self.format_out)
            error_manager_cube.raise_err_specific("GEMM", reason)
        self.format_out = format_out_temp
        # ---------- handle dataflow ---------- #
        connect_str = "2"
        self.ops_data_flow_mode = connect_str.join([self.type_map.get(self.src_dtype),
                                                    self.type_map.get(self.dst_dtype)])
        self.need_cast_to_fp16 = (self.src_dtype == "float16" and self.dst_dtype == "float16")

        # merge data flow to:
        # 1: fp162fp32
        # 2: fp162fp16
        # 3: int82int32
        # 4: int82fp32
        merge_data_flow_dict = {
            # ---- merge to int82int32 ---- #
            "int82int32": "int82int32",
            "int82fp16": "int82int32",
            "int82int8": "int82int32",
            "uint82fp16": "int82int32",
            "int42int32": "int42int32",
            "int42int4": "int42int32",
            "int42fp16": "int42int32",
            # ---- merge to fp162fp32 ---- #
            "fp162fp32": "fp162fp32",
            # ---- merge to fp162fp16 ---- #
            "fp162fp16": "fp162fp16",
            # ---- merge to int82fp32 ---- #
            "int82fp32": "int82fp32"
        }
        self.ops_data_flow_mode = merge_data_flow_dict.get(self.ops_data_flow_mode)
        if self.ops_data_flow_mode is None:
            reason = ("The current input and output dtype is not supported, "
                      "input dtype is {}, output dtype is {}.".format(self.src_dtype, self.dst_dtype))
            error_manager_cube.raise_err_specific("GEMM", reason)
        block_reduce_dict = {
            "int82int32": tbe_platform.BLOCK_REDUCE_INT8,
            "int42int32": tbe_platform.BLOCK_REDUCE_INT4,
            "fp162fp32": tbe_platform.BLOCK_REDUCE,
            "fp162fp16": tbe_platform.BLOCK_REDUCE,
            "int82fp32": tbe_platform.BLOCK_REDUCE
        }
        self.block_reduce = block_reduce_dict.get(self.ops_data_flow_mode)
        matrix_type_dict = {
            "int82int32": "int32",
            "int42int32": "int32",
            "fp162fp32": "float32",
            "fp162fp16": "float32",
            "int82fp32": "float32"
        }
        self.matrix_type = matrix_type_dict.get(self.ops_data_flow_mode)
        self.l0c_support_fp32 = True
        support_type = tbe_platform.getValue("Intrinsic_mmad")
        if "f162f32" not in support_type:
            self.l0c_support_fp32 = False
        if self.matrix_type == "float32" and (not self.l0c_support_fp32):
            self.matrix_type = "float16"

    def _set_ori_k(self):
        shape_a = [self._get_value(i) for i in self.tensor_a.shape]
        shape_b = [self._get_value(i) for i in self.tensor_b.shape]
        trans_a = (not self.trans_a) if (self.format_a == "FRACTAL_NZ") else self.trans_a
        trans_b = (not self.trans_b) if (self.format_b == "FRACTAL_NZ") else self.trans_b
        offset_a = 0
        if len(shape_a) in (3, 5):
            offset_a = 1
        offset_b = 0
        if len(shape_b) in (3, 5):
            offset_b = 1
        self.ori_km_shape = shape_a[offset_a] if trans_a else shape_a[1+offset_a]
        self.ori_kn_shape = shape_b[1+offset_b] if trans_b else shape_b[offset_b]

    # ----------- check func ---------- #
    def _check_k_dim(self):
        """
        check shape km and kn, should be equal
        Input: None
        ---------------------------------
        Return: None
        """
        if in_dynamic():
            return
        km_shape = self.ori_km_shape
        if self.format_a != "ND":
            km_shape *= self.block_reduce

        kn_shape = self.ori_kn_shape
        if self.format_b != "ND":
            kn_shape *= self.block_reduce

        if km_shape != kn_shape:
            raise RuntimeError(
                f"[E69999] A matrix's k should be equal B matrix's k, but are {km_shape} and {kn_shape}.")

    def _check_n_align(self):
        """
        When the input and output tensors share the memory.
        n not align to block_out is not supported.
        Input: None
        --------------------------
        Return: None
        """
        shape_b = [self._get_value(i) for i in self.tensor_b.shape]
        n_shape = shape_b[0] if self.trans_b else shape_b[1]
        if self.format_b == "ND" and (n_shape % tbe_platform.BLOCK_OUT != 0) and self.alpha is not None:
            reason = ("When the input format is ND, "
                      "the n direction must be aligned to {}.".format(tbe_platform.BLOCK_OUT))
            error_manager_cube.raise_err_specific("GEMM", reason)

    def _check_gevm_and_gemv(self):
        shape_a = self.shape_a
        shape_b = self.shape_b
        if len(shape_a) in (4, 5):
            block_in = shape_a[-2]
            k_shape = shape_a[-4] if self.format_a == "FRACTAL_NZ" else shape_a[-3]
            k_align_flag = (block_in == tbe_platform.BLOCK_VECTOR) and (k_shape % tbe_platform.BLOCK_IN != 0)
            if k_align_flag:
                reason = "For fractal gevm input, K should be multiple of {}.".format(tbe_platform.BLOCK_IN *
                                                                                      self.block_reduce)
                error_manager_cube.raise_err_specific("GEMM", reason)

        if len(shape_b) in (4, 5):
            block_out = shape_b[-1] if self.format_b == "FRACTAL_NZ" else shape_b[-2]
            k_shape = shape_b[-3] if self.format_b == "FRACTAL_NZ" else shape_b[-4]
            k_align_flag = (block_out == tbe_platform.BLOCK_VECTOR) and (k_shape % tbe_platform.BLOCK_OUT != 0)
            if k_align_flag:
                reason = "For fractal gemv input, K should be multiple of {}.".format(tbe_platform.BLOCK_OUT *
                                                                                      self.block_reduce)
                error_manager_cube.raise_err_specific("GEMM", reason)
