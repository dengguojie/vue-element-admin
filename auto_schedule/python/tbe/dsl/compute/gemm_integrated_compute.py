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
from tbe.dsl.compute.util import check_input_tensor_shape
from tbe.dsl.base.operation import in_dynamic
from tbe.common.utils import para_check
from tbe.common.utils import shape_util
from tbe.common.utils.errormgr import error_manager_util
from tbe.tvm.tensor import Tensor
from .gemm_compute_util import FormatCompute

# if K_DIM is  equal or larger than GEVM_MODE_K_DIM_LIMIT in gevm/gemv mode, use gemm mode.
# K_DIM is k * k0
GEVM_MODE_K_DIM_LIMIT = 9216

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
    def __init__(self) -> None:
        pass


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
        "uint8": "uint8"
    }

    def __init__(self, tensor_a, tensor_b, para_dict):
        super(GEMMCompute, self).__init__()
        self.para_dict = para_dict
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.a_shape = [self._get_value(i) for i in tensor_a.shape]
        self.b_shape = [self._get_value(i) for i in tensor_b.shape]
        self.alpha = para_dict.get("alpha")
        self.beta = para_dict.get("beta")
        self.trans_a = para_dict.get("trans_a", False)
        self.trans_b = para_dict.get("trans_b", False)
        self.format_a = para_dict.get("format_a", "ND")
        self.format_b = para_dict.get("format_b", "ND")
        self.offset_x = para_dict.get("offset_a", 0)
        self.offset_w = para_dict.get("offset_b", None)
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
        self.compress_index = para_dict.get("compress_index")
        self.cube_vector_split = tbe_platform_info.get_soc_spec("CUBE_VECTOR_SPLIT")
        self.attrs = para_dict.get("attrs", {})
        self.batch_shape = para_dict.get("batch_shape")
        self.beta_mode, self.alpha_mode = "none", "none"
        self.have_bias, self.have_c = False, False
        self.res = None
        self.placeholder_name = dict()
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

    def _get_batch_dims(self, attrs):
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
        batch_a = tbe_utils.shape_to_list(batch_a)
        batch_b = tbe_utils.shape_to_list(batch_b)
        batch_out = tbe_utils.shape_to_list(batch_out)

        if batch_a or batch_b:
            batch_a, batch_b, batch_l0c = broadcast_shapes(batch_a, batch_b)
        else:
            batch_l0c = list()

        return batch_a, batch_b, batch_out, batch_l0c

    def _init_dynamic_params(self, tensor_c_gm, tensor_l0a, tensor_l0b):
        """
        init dynamic params
        Parameters:
            tensor_c_gm: the tensor res
            tensor_l0a: the matrix a before mmad
            tensor_l0b: the matrix b before mmad
        """
        if not in_dynamic():
            return

        tensor_a_length = len(self.tensor_a.shape)
        tensor_b_length = len(self.tensor_b.shape)
        GEMMComputeParam.batch_a = (tensor_a_length in (3, 5))
        GEMMComputeParam.batch_b = (tensor_b_length in (3, 5))
        if GEMMComputeParam.batch_a or GEMMComputeParam.batch_b:
            GEMMComputeParam.dynamic_mode = "dynamic_mknb"
        else:
            GEMMComputeParam.dynamic_mode = "dynamic_mkn"
        is_gevm = int(self.mmad_mode in ("gemv", "gevm"))
        GEMMComputeParam.tiling_info_dict = {
            "A_shape": self._get_a_shape_in_nc1hwc0(tensor_l0a),
            "B_shape": self._get_b_shape_in_nc1hwc0(tensor_l0b),
            "C_shape": None,
            "A_dtype": self.tensor_a.dtype,
            "B_dtype": self.tensor_b.dtype,
            "C_dtype": tensor_c_gm.dtype,
            "mad_dtype": self.matrix_type,
            "padl": 0,
            "padr": 0,
            "padu": is_gevm,
            "padd": 0,
            "strideH": 1,
            "strideW": 1,
            "strideH_expand": 1,
            "strideW_expand": 1,
            "dilationH": self._get_trans_flag(self.trans_a, self.trans_b),
            "dilationW": 1,
            "group": 1,
            "fused_double_operand_num": 0,
            "bias_flag": (self.tensor_bias is not None),
            "op_tag": "matmul",
            "op_type": "matmul",
            "kernel_name": self.kernel_name,
            "dynamic_shape_flag": True,
            "trans_a": self.trans_a,
            "trans_b": self.trans_b
        }

    def _get_trans_flag(self, transpose_a, transpose_b):
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

    def _get_a_shape_in_nc1hwc0(self, tensor_a_l0a):
        """
        get a shape's format nc1hwc0 inorder to get tiling
        """
        if GEMMComputeParam.batch_a:
            return [tensor_a_l0a.shape[0], tensor_a_l0a.shape[2], tensor_a_l0a.shape[1], 16, 16]
        else:
            return [1, tensor_a_l0a.shape[1], tensor_a_l0a.shape[0], 16, 16]

    def _get_b_shape_in_nc1hwc0(self, tensor_b_l0b):
        """
        get b shape's format nc1hwc0 inorder to get tiling
        """
        if GEMMComputeParam.batch_b:
            return [tensor_b_l0b.shape[1] * 16, tensor_b_l0b.shape[2], 1, 1, 16]
        else:
            return [tensor_b_l0b.shape[0] * 16, tensor_b_l0b.shape[1], 1, 1, 16]

    def _set_bias_or_c(self):
        if (self.alpha is None) or (self.beta is None):
            self.tensor_bias, self.tensor_c = self.tensor_c, self.tensor_bias

    def _set_mmad_mode(self):
        """
        when m or n's length is 1 and k is align to 512Byte, can
        use gemv or gevm mode, else use gemm mode
        """
        a_shape = [self._get_value(i) for i in self.tensor_a.shape]
        b_shape = [self._get_value(i) for i in self.tensor_b.shape]
        mmad_mode = "gemm"
        # The op GEMM not use gevm/gemv now
        if self._get_not_gevm_gemv_flag() or (self.alpha is not None):
            self.mmad_mode = "gemm"
            return

        m_shape_dict = {"ND": -2, "FRACTAL_NZ": -3, "FRACTAL_Z": -4}
        m0_shape_dict = {"FRACTAL_NZ": -2, "FRACTAL_Z": -2}
        n_shape_dict = {"ND": -1, "FRACTAL_NZ": -4, "FRACTAL_Z": -3, "fractal": -3}
        n0_shape_dict = {"FRACTAL_NZ": -1, "FRACTAL_Z": -2, "fractal": -2}
        trans_dict = {-1: -2, -2: -1, -3: -4, -4: -3}

        m_index = m_shape_dict.get(self.format_a)
        n_index = n_shape_dict.get(self.format_b)
        m_index = trans_dict.get(m_index) if self.trans_a else m_index
        n_index = trans_dict.get(n_index) if self.trans_b else n_index
        m_shape = a_shape[m_index]
        n_shape = b_shape[n_index]
        gevm_mode_flag, self.only_use_gevm_gemv_flow = self._get_gevm_flag(m0_shape_dict, trans_dict, a_shape, m_index)
        gemv_mode_flag = self._get_gemv_flag(n0_shape_dict, trans_dict, b_shape, n_index)
        both_fractal = self.format_a != "ND" and self.format_b != "ND"
        self._check_and_renew_block(gevm_mode_flag, gemv_mode_flag, both_fractal, self.only_use_gevm_gemv_flow)

    def _check_and_renew_block(self, gevm_mode_flag, gemv_mode_flag, both_fractal, only_use_gevm_gemv_flow):
        is_vector_mul_vector = gevm_mode_flag and gemv_mode_flag and both_fractal
        if is_vector_mul_vector:
            dict_args = {
                "errorCode": "E61001",
                "op_name": "GEMM",
                "reason": "not support vector mul vector when A and B both fractal."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        mmad_mode = "gevm" if gevm_mode_flag else ("gemv" if gemv_mode_flag else "gemm")
        mmad_mode = "gemm" if (gevm_mode_flag and gemv_mode_flag) else mmad_mode
        if not only_use_gevm_gemv_flow:
            self.block_in = tbe_platform.BLOCK_VECTOR if mmad_mode == "gevm" else self.block_in
            self.block_out = tbe_platform.BLOCK_VECTOR if mmad_mode == "gemv" else self.block_out
        self.mmad_mode = mmad_mode

    def _get_gevm_flag(self, m0_shape_dict, trans_dict, a_shape, m_index):
        if self.format_a != "ND":
            m0_index = m0_shape_dict.get(self.format_a)
            gevm_mode_flag = a_shape[m0_index] == tbe_platform.BLOCK_VECTOR
            only_use_gevm_gemv_flow = False
        else:
            if self.alpha is not None:
                gevm_mode_flag = (a_shape[m_index] == 1
                    and (a_shape[trans_dict.get(m_index)] % (self.block_in * self.block_reduce) == 0))
                only_use_gevm_gemv_flow = False
            else:
                gevm_mode_flag = (a_shape[m_index] == 1)
                only_use_gevm_gemv_flow = (a_shape[m_index] == 1
                    and (a_shape[trans_dict.get(m_index)] % (self.block_in * self.block_reduce) != 0))
        if gevm_mode_flag and (not only_use_gevm_gemv_flow):
            multi = 1
            if self.format_a != "ND":
                multi = self.block_reduce
            if (a_shape[trans_dict.get(m_index)] * multi) >= GEVM_MODE_K_DIM_LIMIT:
                gevm_mode_flag = False
        return gevm_mode_flag, only_use_gevm_gemv_flow

    def _get_gemv_flag(self, n0_shape_dict, trans_dict, b_shape, n_index):
        if self.format_b != "ND":
            n0_index = n0_shape_dict.get(self.format_b)
            gemv_mode_flag = b_shape[n0_index] == tbe_platform.BLOCK_VECTOR
        else:
            gemv_mode_flag = (b_shape[n_index] == 1
                and (b_shape[trans_dict.get(n_index)] % (self.block_out * self.block_reduce) == 0))
        return gemv_mode_flag

    def _get_not_gevm_gemv_flag(self):
        not_use_gevm_gemv_flag = False
        is_int82int32_nd = (self.ops_data_flow_mode == "int82int32"
            and (self.format_a == "ND" or self.format_b == "ND")) and (self.alpha is not None)
        not_use_gevm_gemv_flag = (self.ops_data_flow_mode == "int82fp32" or is_int82int32_nd) and (self.alpha is not None)
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
            a_shape = self.tensor_a.shape
            if len(a_shape) in (3, 5):
                a_shape_dim_m = self._get_value(a_shape[1])
            else:
                a_shape_dim_m = self._get_value(a_shape[0])
            if a_shape_dim_m != 1:
                self.format_out = "ND"
            else:
                self.format_out = "FRACTAL_NZ"

    def _get_c_ub_fract(self, c_matrix, scale_drq, scale_drq_tensor, sqrt_out):
        c_matrix_shape = [self._get_value(i) for i in c_matrix.shape]
        if self.mmad_mode in ("gemv", "gevm"):
            c_matrix_shape[-2] = 1
        nz_b = (self.format_b == "FRACTAL_NZ")
        if scale_drq_tensor is not None:
            tensor_c_ub_fract = tvm.compute(
                c_matrix_shape,
                lambda *indices: (c_matrix[indices] * scale_drq_tensor[0]),
                name = "c_ub_fract",
                attrs={"scale_drq": scale_drq,
                       "sqrt_out": sqrt_out,
                       "nz_b": nz_b})
        else:
            tensor_c_ub_fract = tvm.compute(
                c_matrix_shape,
                lambda *indices: c_matrix(*indices), # pylint: disable=W0108
                name = "c_ub_fract",
                attrs={"scale_drq": scale_drq,
                       "sqrt_out": sqrt_out,
                       "nz_b": nz_b})

        return tensor_c_ub_fract

    def calculate(self):
        """
        the main func of gemm
        """
        self._init_func()
        self.ops_format = self.format_a

        scale_drq, scale_drq_tensor, sqrt_out = self._get_quantize_params(self.quantize_params, self.dst_dtype)
        a_matrix = self._compute_a_matrix()
        b_matrix = self._compute_b_matrix()
        c_matrix = self._compute_c_matrix(a_matrix, b_matrix, self.tensor_bias)
        c_ub_fract = self._get_c_ub_fract(c_matrix, scale_drq, scale_drq_tensor, sqrt_out)

        tensor_alpha_c = self._get_alpha_ab_matrix(c_ub_fract)
        tensor_beta_bias_ub = self._get_beta_c_matrix()
        res = self._compute_alpha_c_add_beta_bias(tensor_alpha_c, tensor_beta_bias_ub)
        self._compute_res(res)

        self._init_dynamic_params(self.res, a_matrix, b_matrix)

    def _compute_res(self, res):
        res_tag = self._get_ops_tag()
        attrs_dict = self._get_attrs_dict(res)
        c_gm_shape = self._get_out_shape(res)
        attrs_dict["shape"] = c_gm_shape
        # not_align flag is used for nd out
        not_align = (not self.align_a) or (not self.align_b)
        not_align = not_align and (self.format_out == "ND")
        a_shape_len = len(self.tensor_a.shape)
        b_shape_len = len(self.tensor_b.shape)

        have_batch = len(c_gm_shape) in (3, 5)

        if not_align:
            if len(c_gm_shape) in (2, 3):
                if have_batch:
                    self.res = tvm.compute(
                        c_gm_shape,
                        lambda batch, i, j: tvm.select(
                            i < c_gm_shape[1],
                            tvm.select(j < c_gm_shape[2], res[batch, i, j])
                        ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                    )
                else:
                    self.res = tvm.compute(
                        c_gm_shape,
                        lambda i, j: tvm.select(
                            i < c_gm_shape[0],
                            tvm.select(j < c_gm_shape[1], res[i, j])
                        ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                    )
            # may needn't
            elif len(c_gm_shape) in (4, 5):
                if have_batch:
                    self.res = tvm.compute(
                        c_gm_shape,
                        lambda batch, i, j, k, l: tvm.select(
                            i < c_gm_shape[1],
                            tvm.select(j < c_gm_shape[2], res[batch, i, j, k, l])
                        ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                    )
                else:
                    self.res = tvm.compute(
                        c_gm_shape,
                        lambda i, j, k, l: tvm.select(
                            i < c_gm_shape[0],
                            tvm.select(j < c_gm_shape[1], res[i, j, k, l])
                        ), name="tensor_c_gm", tag=res_tag, attrs=attrs_dict
                    )
        else:
            self.res = tvm.compute(
                c_gm_shape,
                lambda *indices: res(*indices), # pylint: disable=W0108
                    name="tensor_c_gm",
                    tag=res_tag,
                    attrs=attrs_dict
            )



    def _get_attrs_dict(self, res):
        attrs_dict = {
            "shape": res.shape,
            "format": self.format_out,
            "is_gemm_new": True
        }
        batch_dims = self._get_batch_dims(self.para_dict)
        attrs_dict["batch_shape_a"] = batch_dims[0]
        attrs_dict["batch_shape_b"] = batch_dims[1]
        attrs_dict["batch_shape"] = batch_dims[2]
        attrs_dict["batch_shape_out"] = batch_dims[2]
        return attrs_dict

    def _get_out_shape(self, res):
        c_gm_shape = [self._get_value(i) for i in res.shape]
        c_gm_shape_out = list()
        if len(c_gm_shape)in (3, 5):
            c_gm_shape_out.append(c_gm_shape[0])

        tensor_a_shape = [self._get_value(i) for i in self.tensor_a.shape]
        if self.format_a == "ND":
            ori_m_shape = tensor_a_shape[-1] if self.trans_a else tensor_a_shape[-2]
        elif self.format_a == "FRACTAL_Z":
            ori_m_shape = tensor_a_shape[-3] if self.trans_a else tensor_a_shape[-4]
        else:
            ori_m_shape = tensor_a_shape[-4] if self.trans_a else tensor_a_shape[-3]

        tensor_b_shape = [self._get_value(i) for i in self.tensor_b.shape]
        if self.format_b == "ND":
            ori_n_shape = tensor_b_shape[-2] if self.trans_b else tensor_b_shape[-1]
        elif self.format_b == "FRACTAL_Z":
            ori_n_shape = tensor_b_shape[-4] if self.trans_b else tensor_b_shape[-3]
        else:
            ori_n_shape = tensor_b_shape[-3] if self.trans_b else tensor_b_shape[-4]

        if self.format_out == "ND":
            if self.format_a == "ND":
                c_gm_shape_out.append(ori_m_shape)
            else:
                c_gm_shape_out.append(ori_m_shape * self.block_in)
            if self.format_b == "ND":
                c_gm_shape_out.append(ori_n_shape)
            else:
                c_gm_shape_out.append(ori_n_shape * self.block_out)
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
            c_gm_shape_out.append(ori_n_shape)
            c_gm_shape_out.append(ori_m_shape)
            c_gm_shape_out.append(block_in)
            c_gm_shape_out.append(block_out)
            if self.mmad_mode == "gemv":
                c_gm_shape_out[-3], c_gm_shape_out[-4] = c_gm_shape_out[-4], c_gm_shape_out[-3]
                c_gm_shape_out[-1], c_gm_shape_out[-2] = c_gm_shape_out[-2], c_gm_shape_out[-1]

        return c_gm_shape_out

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

    def _get_alpha_ab_matrix(self, c_matrix):
        c_matrix_shape = [self._get_value(i) for i in c_matrix.shape]

        if self.alpha_mode == "none":
            return c_matrix
        elif self.alpha_mode == "mul_num":
            alpha = self.alpha
            c_matrix_type = c_matrix.dtype
            tensor_alpha_c = tvm.compute(
                c_matrix_shape,
                lambda *indices: tvm.convert(alpha).astype(c_matrix_type) * c_matrix(*indices),
                name="tensor_alpha_c"
            )
        else:
            tensor_alpha = self.alpha
            if tensor_alpha.dtype == "float16":
                tensor_alpha = tvm.compute(
                    tensor_alpha.shape,
                    lambda *indices: shape_util.cast(
                        tensor_alpha(*indices), dtype="float32"
                    ),
                    name="tensor_alpha_fp162fp32"
                )
            tensor_alpha_c = tvm.compute(
                c_matrix_shape,
                lambda *indices: c_matrix(*indices) * tensor_alpha[0],
                name="tensor_alpha_c"
            )

        return tensor_alpha_c

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
            m_shape = math.ceil(ori_m_shape / 32) * 32 // 16
            km_shape = math.ceil(ori_km_shape / 32) * 32 // 16
        else:
            if self.src_dtype in ("uint8", "int8"):
                m_shape = math.ceil(ori_m_shape / 32) * 32 // 16
            else:
                m_shape = math.ceil(ori_m_shape / self.block_in)
            km_shape = math.ceil(ori_km_shape / self.block_reduce)

        aligned_shape_a = [m_shape * self.block_in, km_shape * self.block_reduce]
        align_flag_a = (ori_m_shape == aligned_shape_a[-2]) and (ori_km_shape == aligned_shape_a[-1])
        aligned_shape_a = aligned_shape_a[::-1] if self.trans_a else aligned_shape_a

        return align_flag_a, aligned_shape_a

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
            n_shape = math.ceil(ori_n_shape / 32) * 32 // 16
            kn_shape = math.ceil(ori_kn_shape / 32) * 32 // 16
        else:
            if self.src_dtype in ("uint8", "int8"):
                n_shape = math.ceil(ori_n_shape / 32) * 32 // 16
            else:
                n_shape = math.ceil(ori_n_shape / self.block_out)
            kn_shape = math.ceil(ori_kn_shape / self.block_reduce)

        aligned_shape_b = [kn_shape * self.block_reduce, n_shape * self.block_out]
        align_flag_b = (ori_n_shape == aligned_shape_b[-1]) and (ori_kn_shape == aligned_shape_b[-2])
        aligned_shape_b = aligned_shape_b[::-1] if self.trans_b else aligned_shape_b
        return align_flag_b, aligned_shape_b

    def _do_align_nd_shape(self, tensor_need_align, tensor_name, in_dtype, need_check_align=False):
        """
        do align for a_matrix or b_matrix, pad zero along the way
        input:
            tensor_need_align: tensor, the tensor need align
            tensor_name: str, a or b or bias
            in_dtype: str, input data type
        return:
            aligned tensor
        """

        if tensor_name == "bias":
            if len(tensor_need_align.shape) in (4, 5):
                is_align = True
            else:
                ori_m_shape = self._get_value(tensor_need_align.shape[-2])
                ori_n_shape = self._get_value(tensor_need_align.shape[-1])
                m_shape = math.ceil(ori_m_shape / self.block_in) * self.block_in
                n_shape = math.ceil(ori_n_shape / self.block_out) * self.block_out
                is_align = (ori_m_shape == m_shape) and (ori_n_shape == n_shape)
                aligned_shape = [m_shape, n_shape]
        elif tensor_name == "a":
            is_align, aligned_shape = self._check_align_a()
            self.align_a = is_align
        else:
            is_align, aligned_shape = self._check_align_b()
            self.align_b = is_align
        not_need_align = is_align or (self.mmad_mode in ("gemv", "gevm"))
        not_need_align = False if in_dynamic() else not_need_align
        if not_need_align or (not need_check_align):
            tensor_normalize_ub = tvm.compute(
                tensor_need_align.shape,
                lambda *indices: tensor_need_align(*indices), # pylint: disable=W0108
                name="tensor_{}_normalize_ub".format(tensor_name)
            )
            return tensor_normalize_ub

        ori_shape = [self._get_value(i) for i in tensor_need_align.shape]
        if len(ori_shape) == 2:
            tensor_normalize_ub = tvm.compute(
                aligned_shape,
                lambda i, j: tvm.select(
                    i < ori_shape[0],
                    tvm.select(
                        j < ori_shape[1],
                        tensor_need_align[i, j],
                        tvm.convert(0).astype(in_dtype)
                    ),
                    tvm.convert(0).astype(in_dtype)
                ),
                name="tensor_{}_normalize_ub".format(tensor_name)
            )
        else:
            aligned_shape.insert(0, ori_shape[0])

            tensor_normalize_ub = tvm.compute(
                aligned_shape,
                lambda batch, i, j: tvm.select(
                    i < ori_shape[-2],
                    tvm.select(
                        j < ori_shape[-1],
                        tensor_need_align[batch, i, j],
                        tvm.convert(0).astype(in_dtype)
                    ),
                    tvm.convert(0).astype(in_dtype)
                ),
                name="tensor_{}_normalize_ub".format(tensor_name)
            )

        return tensor_normalize_ub

    def _compute_a_matrix(self):
        """ compute matrix for mad
        Input: None
        support func:
            fp16 input:
                (m, k)
        Return:
            the tensor format is Zn for mad
        """
        need_check_align = (self.format_a in ("ND", "NC1HWC0")) and (self.mmad_mode != "gevm")
        not_pad_for_int8_nd = self.tensor_a.dtype in ("int8", "uint8") and self.format_a == "ND" and (self.alpha is None)
        self.int8_not_double_m = not_pad_for_int8_nd
        need_check_align = need_check_align and (not not_pad_for_int8_nd)
        if self.format_a in ("ND", "NC1HWC0"):
            tensor_a_normalize = self._do_align_nd_shape(self.tensor_a, "a", self.src_dtype, need_check_align)
        else:
            tensor_a_normalize = self.tensor_a
        tensor_a_normalize_shape = [self._get_value(i) for i in tensor_a_normalize.shape]

        # ------------------- GEVM(ND)/GEMV process ------------------- #
        # not support int82fp32
        if self.mmad_mode == "gemv":
            return self._compute_a_matrix_gemv(tensor_a_normalize)
        elif self.mmad_mode == "gevm" and self.format_a in ("ND", "NC1HWC0"):
            return self._compute_a_matrix_gevm(tensor_a_normalize)

        # ------------------------------------------------------------ #
        if self.ops_data_flow_mode == "int82fp32":
            tensor_a_normalize = tvm.compute(
                tensor_a_normalize_shape,
                lambda *indices: shape_util.cast(tensor_a_normalize(*indices), "float16"),
                name="tensor_a_int82fp16"
            )

        if self.format_a in ("FRACTAL_Z", "FRACTAL_NZ"):
            # ---------------------- fractal process --------------------- #
            return self._compute_a_matrix_fractal(tensor_a_normalize)
        else:
            # ------------------------ ND process ------------------------ #
            return self._compute_a_matrix_nd(tensor_a_normalize, not_pad_for_int8_nd)

    def _compute_a_matrix_gemv(self, tensor_a_normalize):
        compute_params_gemv = {"tensor_name": "tensor_a_matrix", "trans": self.trans_a}
        if self.format_a == "FRACTAL_NZ":
            compute_params_gemv["mode_info"] = "fractal_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Nz",
                "format_in_a_ub": "none"
            }
            tensor_a_matrix = self.fract_change_inner_axis(tensor_a_normalize, compute_params_gemv)
        elif self.format_a == "FRACTAL_Z":
            compute_params_gemv["tensor_name"] = "tensor_a_matrix"
            compute_params_gemv["mode_info"] = "fractal_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "none"
            }
            tensor_a_matrix = self.fract_change_outer_axis(tensor_a_normalize, compute_params_gemv)
        else:
            compute_params_gemv["trans"] = False
            compute_params_gemv["block_in"] = self.block_in
            compute_params_gemv["block_reduce"] = self.block_reduce
            compute_params_gemv["tensor_name"] = "tensor_a_fract"
            tensor_a_fract = self.compute_nd2Zz(tensor_a_normalize, compute_params_gemv)
            compute_params_gemv["trans"] = self.trans_a
            compute_params_gemv["mode_info"] = "nd_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_a_l1": "Zz",
                "format_in_a_ub": "nd"
            }
            compute_params_gemv["tensor_name"] = "tensor_a_matrix"
            tensor_a_matrix = self.fract_change_outer_axis(tensor_a_fract, compute_params_gemv)
        return tensor_a_matrix

    def _compute_a_matrix_gevm(self, tensor_a_normalize):
        compute_params_gemv = {"tensor_name": "tensor_a_matrix", "trans": self.trans_a}
        compute_params_gemv["trans"] = self.trans_a
        compute_params_gemv["block_in"] = self.block_in
        compute_params_gemv["block_reduce"] = self.block_reduce
        compute_params_gemv["tensor_name"] = "tensor_a_fract"
        compute_params_gemv["mode_info"] = "nd_gevm"
        compute_params_gemv["format_info"] = {
            "format_in_a_l1": "Zz",
            "format_in_a_ub": "nd"
        }
        tensor_a_matrix = self.compute_nd2Zz_gevm(tensor_a_normalize, compute_params_gemv)

        return tensor_a_matrix

    def _compute_a_matrix_fractal(self, tensor_a_normalize):
        compute_params_fractal = {"tensor_name": "tensor_a_matrix", "trans": self.trans_a}
        if self.format_a == "FRACTAL_Z":
            if self.trans_a:
                compute_params_fractal["mode_info"] = "Zz_trans"
                compute_params_fractal["format_info"] = {
                    "format_in_a_l1": "Zz",
                    "format_in_a_ub": "none"
                }
                compute_params_fractal["trans"] = False
                tensor_a_normalize = self.fract_change_both_axis(tensor_a_normalize, compute_params_fractal)
            return tensor_a_normalize
        elif self.format_a == "FRACTAL_NZ":
            if self.ops_data_flow_mode == "int82fp32":
                compute_params_fractal["mode_info"] = "Nz2Zz_int82fp32"
                compute_params_fractal["format_info"] = {
                    "format_in_a_l1": "Zz",
                    "format_in_a_ub": "Nz"
                }
                tensor_a_matrix = self.compute_Nz2Zz_int82fp32(tensor_a_normalize, compute_params_fractal)
            else:
                compute_params_fractal["mode_info"] = "Nz2Zz"
                compute_params_fractal["format_info"] = {
                    "format_in_a_l1": "Nz",
                    "format_in_a_ub": "none"
                }
                tensor_a_matrix = self.fract_change_outer_axis(tensor_a_normalize, compute_params_fractal)
            return tensor_a_matrix

    def _compute_a_matrix_nd(self, tensor_a_normalize, use_normal_func):
        nd2Zz_normal_flag = False
        nd2Zz_normal_flag = nd2Zz_normal_flag or self.cube_vector_split
        nd2Zz_normal_flag = nd2Zz_normal_flag or (self.ops_data_flow_mode == "int82int32")
        nd2Zz_normal_flag = nd2Zz_normal_flag or (self.mmad_mode == "gevm")
        nd2Zz_normal_flag = nd2Zz_normal_flag or use_normal_func

        nd2Zz_vnchwconv_flag = False
        nd2Zz_vnchwconv_flag = nd2Zz_vnchwconv_flag or (self.ops_data_flow_mode != "int82int32")

        compute_params = {
            "tensor_name": "tensor_a_matrix",
            "block_in": self.block_in,
            "block_reduce": self.block_reduce,
            "data_flow": self.ops_data_flow_mode,
            "trans": self.trans_a
        }
        if nd2Zz_normal_flag:
            mode_info = "nd2Zz" if (self.ops_data_flow_mode != "int82int32") else "nd2Zz_int8"
            compute_params["mode_info"] = mode_info
            format_in_a_l1 = "Zz"
            if self.trans_a and (self.ops_data_flow_mode != "int82int32"):
                format_in_a_l1 = "Nn"
            compute_params["format_info"] = {
                "format_in_a_l1": format_in_a_l1,
                "format_in_a_ub": "nd"
            }
            tensor_a_matrix = self.compute_nd2Zz(tensor_a_normalize, compute_params)
        elif nd2Zz_vnchwconv_flag:
            compute_params["mode_info"] = "nd2Zz_vnchwconv"
            compute_params["format_info"] = {
                "format_in_a_l1": "three_axis",
                "format_in_a_ub": "nd"
            }
            tensor_a_matrix = self.compute_nd2Zz_vnchwconv(tensor_a_normalize, compute_params)
        return tensor_a_matrix

    def _compute_b_matrix(self):
        tensor_b = self.tensor_b

        # ---------------------- compress process ---------------------- #
        # noly support fp16 input
        compress_flag = self.compress_index is not None
        self.compress_flag = compress_flag
        if compress_flag:
            tensor_b_matrix = self._get_compress_tensor_compute(tensor_b, self.compress_index, "tensor_b_matrix")
            self.format_b = "FRACTAL_Z"
            return tensor_b_matrix

        need_check_align = (self.format_b in ("ND", "NC1HWC0")) and (self.mmad_mode != "gemv")
        if self.format_b == "ND":
            tensor_b_normalize = self._do_align_nd_shape(tensor_b, "b", self.src_dtype, need_check_align)
        else:
            tensor_b_normalize = tensor_b
        tensor_b_normalize_shape = [self._get_value(i) for i in tensor_b_normalize.shape]

        # ---------------------- GEMV process ---------------------- #
        if self.mmad_mode == "gemv":
            return self._compute_b_matrix_gemv(tensor_b_normalize)

        # ---------------------------------------------------------- #
        if self.ops_data_flow_mode == "int82fp32":
            tensor_b_normalize = tvm.compute(
                tensor_b_normalize_shape,
                lambda *indices: shape_util.cast(tensor_b_normalize(*indices), "float16"),
                name="tensor_b_int82fp16"
            )

        if self.format_b in ("FRACTAL_Z", "FRACTAL_NZ"):
            # ------------------------ fractal process ----------------------- #
            return self._compute_b_matrix_fractal(tensor_b_normalize)
        else:
            # -------------------------- ND process -------------------------- #
            return self._compute_b_matrix_nd(tensor_b_normalize)

    def _compute_b_matrix_gemv(self, tensor_b_normalize):
        compute_params_gemv = {"tensor_name": "tensor_b_matrix", "trans": self.trans_b}
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
                self.trans_b = False if self.trans_b else True
            compute_params_gemv["trans"] = self.trans_b
            tensor_b_matrix = self.fract_change_outer_axis(tensor_b_normalize, compute_params_gemv)
        else:
            compute_params_gemv["tensor_name"] = "tensor_b_fract"
            compute_params_gemv["trans"] = False
            tensor_b_fract = self.compute_nd2Zz(tensor_b_normalize, compute_params_gemv)
            compute_params_gemv["mode_info"] = "nd_gemv"
            compute_params_gemv["format_info"] = {
                "format_in_b_l1": "Zz",
                "format_in_b_ub": "nd"
            }
            compute_params_gemv["trans"] = self.trans_b
            compute_params_gemv["tensor_name"] = "tensor_b_matrix"
            tensor_b_matrix = self.fract_change_both_axis(tensor_b_fract, compute_params_gemv)

        return tensor_b_matrix

    def _compute_b_matrix_fractal(self, tensor_b_normalize):
        compute_params_fractal = {"tensor_name": "tensor_b_matrix", "trans": self.trans_b}
        if self.format_b == "FRACTAL_Z":
            if self.ops_data_flow_mode == "int82fp32":
                compute_params_fractal["mode_info"] = "Zn2Zn_int82fp32"
                compute_params_fractal["format_info"] = {
                    "format_in_b_l1": "Zn",
                    "format_in_b_ub": "Zn"
                }
                tensor_b_matrix = self.compute_Zn2Zn_int82fp32(tensor_b_normalize, compute_params_fractal)
            elif self.trans_b:
                compute_params_fractal["mode_info"] = "Zn_trans"
                compute_params_fractal["format_info"] = {
                    "format_in_b_l1": "Zn",
                    "format_in_b_ub": "none"
                }
                compute_params_fractal["trans"] = False
                tensor_b_matrix = self.fract_change_both_axis(tensor_b_normalize, compute_params_fractal)
            else:
                tensor_b_matrix = tensor_b_normalize

            return tensor_b_matrix
        elif self.format_b == "FRACTAL_NZ":
            compute_params_fractal["mode_info"] = "Nz2Zn"
            compute_params_fractal["format_info"] = {
                "format_in_b_l1": "Nz",
                "format_in_b_ub": "none"
            }
            tensor_b_matrix = self.fract_change_both_axis(tensor_b_normalize, compute_params_fractal)

            return tensor_b_matrix

    def _compute_b_matrix_nd(self, tensor_b_normalize):
        nd2Zn_normal_flag = False
        nd2Zn_normal_flag = nd2Zn_normal_flag or self.cube_vector_split
        nd2Zn_normal_flag = nd2Zn_normal_flag or (self.ops_data_flow_mode == "int82int32")

        nd2Zn_vnchwconv_flag = False
        nd2Zn_vnchwconv_flag = nd2Zn_vnchwconv_flag or (self.ops_data_flow_mode != "int82int32")

        compute_params = {
            "tensor_name": "tensor_b_matrix",
            "block_in": self.block_in,
            "block_reduce": self.block_reduce,
            "block_out": self.block_out,
            "data_flow": self.ops_data_flow_mode,
            "trans": self.trans_b
        }
        if nd2Zn_normal_flag:
            compute_params["mode_info"] = "nd2Zn" if self.ops_data_flow_mode != "int82int32" else "nd2Zn_int8"
            format_in_b_l1 = "Zn"
            if self.trans_b and (self.ops_data_flow_mode != "int82int32"):
                format_in_b_l1 = "Nz"
            compute_params["format_info"] = {
                "format_in_b_l1": format_in_b_l1,
                "format_in_b_ub": "nd"
            }
            tensor_b_matrix = self.compute_nd2Zn(tensor_b_normalize, compute_params)
        elif nd2Zn_vnchwconv_flag:
            compute_params["mode_info"] = "nd2Zn_vnchwconv"
            compute_params["format_info"] = {
                "format_in_b_l1": "three_axis",
                "format_in_b_ub": "nd"
            }
            tensor_b_matrix = self.compute_nd2Zn_vnchwconv(tensor_b_normalize, compute_params)

        return tensor_b_matrix

    def _get_compress_tensor_compute(self, tensor_src, comp_index, op_name):
        """
        get compress tensor compute
        """
        _, _, _, compress_mode = tbe_platform_info.get_soc_spec("UNZIP")
        comp_size = 8 if compress_mode == 1 else 2

        tile_k_value = tvm.var("tile_L1_k", dtype="int32")
        tile_n_value = tvm.var("tile_L1_n", dtype="int32")

        shape_src = tensor_src.shape
        block_n_num = (shape_src[-3] + tile_n_value - 1) // tile_n_value
        block_k_num = (shape_src[-4] + tile_k_value - 1) // tile_k_value

        # tile_mode is 1 when tile_n < dim_n, or tile_mode is 0
        if len(shape_src) == 4:
            tensor = tvm.compute(shape_src, lambda i, j, k, l:tvm.unzip(
                comp_index((j // tile_n_value * block_k_num + i // tile_k_value) * comp_size),
                tensor_src(i, j, k, l)),
                name=op_name, 
                attrs={"tile_L1_k": tile_k_value,
                        "tile_L1_n": tile_n_value}
            )
        elif len(shape_src) == 5:
            # get multi n block number
            batch_block_num = block_n_num * block_k_num
            tensor = tvm.compute(shape_src, lambda batch, i, j, k, l:tvm.unzip(
                comp_index(((j // tile_n_value * block_k_num + i // tile_k_value)
                            + batch * batch_block_num) * comp_size),
                tensor_src(batch, i, j, k, l)),
                name=op_name, 
                attrs={"tile_L1_k": tile_k_value,
                        "tile_L1_n": tile_n_value}
            )
        return tensor

    def _get_beta_c_matrix(self):
        tensor_c = self.tensor_c
        tensor_beta = self.beta
        if (tensor_c is None) or (self.beta_mode not in ("mul_num", "mul_tensor")):
            return None
        if tensor_beta.dtype == "float16":
            tensor_beta = tvm.compute(
                tensor_beta.shape,
                lambda *indices: shape_util.cast(
                    tensor_beta(*indices), dtype="float32"
                ),
                name="tensor_beta_fp162fp32"
            )

        tensor_c_normalize = self._do_align_nd_shape(tensor_c, "bias", self.dst_dtype, True)

        tensor_c_normalize_shape = tensor_c_normalize.shape
        if self.mmad_mode == "gemv":
            tensor_c_normalize_shape_temp = [
                tensor_c_normalize_shape[-3],
                tensor_c_normalize_shape[-4],
                tensor_c_normalize_shape[-1],
                tensor_c_normalize_shape[-2]
            ]
            if len(tensor_c_normalize_shape) == 5:
                tensor_c_normalize_shape_temp.insert(0, tensor_c_normalize_shape[0])
            tensor_c_normalize_shape = tensor_c_normalize_shape_temp

        if tensor_c_normalize.dtype == "float16":
            tensor_c_normalize = tvm.compute(
                tensor_c_normalize_shape,
                lambda *indices: shape_util.cast(tensor_c_normalize(*indices), dtype="float32"),
                name="tensor_bias_cast_to_fp32"
            )

        if self.mmad_mode == "gemv":
            tensor_beta_bias = tvm.compute(
                tensor_c_normalize_shape,
                lambda i, j, k, l: tensor_beta[0] * tensor_c_normalize[j, i, l, k],
                name="tensor_beta_bias"
            )
        else:
            tensor_beta_bias = tvm.compute(
                tensor_c_normalize_shape,
                lambda *indices: tensor_beta[0] * tensor_c_normalize(*indices),
                name = "tensor_beta_bias"
            )
        return tensor_beta_bias

    def _get_bias_l0c(self, tensor_bias, a_shape, b_shape):

        l0c_shape = [
            b_shape[-3],
            a_shape[-4],
            tbe_platform.BLOCK_OUT,
            tbe_platform.BLOCK_IN
        ]
        a_shape_len = len(a_shape)
        b_shape_len = len(b_shape)
        have_batch = (a_shape_len > 4) or (b_shape_len > 4)
        if self.have_bias and (not self.cube_vector_split):
            if have_batch:
                tensor_bias_l0c = self._get_bias_l0c_with_batch(tensor_bias, a_shape, b_shape, l0c_shape)
            else:
                tensor_bias_l0c = self._get_bias_l0c_without_batch(tensor_bias, l0c_shape)
        else:
            tensor_bias_l0c = tensor_bias

        return tensor_bias_l0c

    def _get_bias_l0c_shape(self, tensor_bias):
        bias_shape = []
        if self._get_value(self._elecnt_of_shape(tensor_bias.shape)) == 1:
            bias_shape = [1]
        else:
            for i in tensor_bias.shape:
                if bias_shape:
                    bias_shape.append(self._get_value(i))
                elif self._get_value(i) != 0 and self._get_value(i) != 1:
                    # first element value should be > 1
                    bias_shape.append(self._get_value(i))
        return bias_shape

    def _get_bias_l0c_with_batch(self, tensor_bias, a_shape, b_shape, l0c_shape):
        block_out = self.block_out
        bias_len = len(tensor_bias.shape)
        a_shape_len = len(a_shape)
        b_shape_len = len(b_shape)
        batch = a_shape[0] if a_shape_len >= b_shape_len else b_shape[0]
        l0c_shape.insert(0, batch)
        bias_shape = self._get_bias_l0c_shape(tensor_bias)
        # tensor_bias shape only be [n,], [1,n] and [1,1,n]
        if len(bias_shape) == 1:
            if bias_len == 1:
                tensor_bias_ub = tvm.compute(
                    bias_shape, lambda i: tensor_bias[i],
                    name="tensor_bias_ub")
            elif bias_len == 2:
                tensor_bias_ub = tvm.compute(
                    bias_shape, lambda i: tensor_bias[0, i],
                    name="tensor_bias_ub"
                )
            else:
                tensor_bias_ub = tvm.compute(
                    bias_shape, lambda i: tensor_bias[0, 0, i],
                    name="tensor_bias_ub"
                )
        elif bias_len == 3:
            # bias_shape only be (batch, 1, n)
            tensor_bias_ub = tvm.compute(
                bias_shape, lambda *indices: tensor_bias[indices],
                name="tensor_bias_ub"
            )

        if tensor_bias.dtype == "float16" and self.l0c_support_fp32:
            tensor_bias_l0c = tvm.compute(
                l0c_shape, lambda batch, i, j, k, l: shape_util.cast(
                    tensor_bias_ub[i * block_out + l], dtype="float32"), name="tensor_bias_l0c")
        else:
            tensor_bias_l0c = tvm.compute(
                l0c_shape,
                lambda batch, i, j, k, l: tensor_bias_ub[i * block_out + l], name="tensor_bias_l0c")

        return tensor_bias_l0c

    def _get_bias_l0c_without_batch(self, tensor_bias, l0c_shape):
        block_out = self.block_out
        bias_len = len(tensor_bias.shape)
        bias_shape = self._get_bias_l0c_shape(tensor_bias)
        # bias only be [n,] and [1,n] for gevm and gemm
        if bias_len == 1:
            tensor_bias_ub = tvm.compute(
                bias_shape, lambda i:
                tensor_bias[i], name="tensor_bias_ub")
        else:
            tensor_bias_ub = tvm.compute(
                bias_shape, lambda i: tensor_bias[0, i],
                name="tensor_bias_ub")

        if tensor_bias.dtype == "float16" and self.l0c_support_fp32:
            tensor_bias_l0c = tvm.compute(
                l0c_shape, lambda i, j, k, l: shape_util.cast(
                    tensor_bias_ub[i * block_out + l],
                    dtype="float32"), name="tensor_bias_l0c")
        else:
            tensor_bias_l0c = tvm.compute(
                l0c_shape, lambda i, j, k, l: tensor_bias_ub[
                    i * block_out + l], name="tensor_bias_l0c")
        return tensor_bias_l0c

    def _compute_c_matrix(self, a_matrix_in, b_matrix_in, tensor_bias):
        """ MatMul calculation
        Input:
            a_matrix_in: tensor, a_matrix in l0a
            b_matrix_in: tensor, b_matrix in l0b
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
                      "compress_flag": self.compress_flag}

        k_shape_l0 = b_matrix_in.shape[-3] if self.mmad_mode == "gemv" else a_matrix_in.shape[-3]
        reduce_kp, reduce_kb = self._get_reduce(k_shape_l0)
        need_add_bias_in_fb = self.have_bias and self.cube_vector_split

        tensor_c_matrix = self.tvm_compute_mad(
            a_matrix_in,
            b_matrix_in,
            tensor_bias,
            reduce_kb,
            reduce_kp,
            self.matrix_type,
            self.mmad_mode,
            "tensor_c_matrix",
            need_add_bias_in_fb,
            self.offset_x,
            attrs_dict
        )

        need_add_bias_in_l0c = self.have_bias and (not self.cube_vector_split)
        if need_add_bias_in_l0c:
            bias_l0c = self._get_bias_l0c(tensor_bias, a_matrix_in.shape, b_matrix_in.shape)
            tensor_c_add_bias = tvm.compute(bias_l0c.shape, lambda *indices:
                                            bias_l0c[indices] +
                                            tensor_c_matrix[indices],
                                            name="tensor_c_add_bias")
            return tensor_c_add_bias
        return tensor_c_matrix

    def _compute_alpha_c_add_beta_bias(self, tensor_alpha_c, tensor_beta_bias):
        cur_format_out = "FRACTAL_NZ"
        if tensor_beta_bias is not None:
            aligned_out_shapes = [self._get_value(i) for i in tensor_beta_bias.shape]
            aligned_out_shapes_len = len(aligned_out_shapes)
            if self.format_out == "ND":
                tensor_alpha_c_add_beta_bias = self.tvm_compute_nd_add_Nz_to_nd(
                    tensor_beta_bias,
                    tensor_alpha_c,
                    "tensor_c_add_bias_ub"
                )
                cur_format_out = "ND"
            else:
                if self.ops_data_flow_mode == "int82int32":
                    tensor_alpha_c_add_beta_bias = self.tvm_compute_nd_add_Nz_to_Nz(
                        tensor_beta_bias,
                        tensor_alpha_c,
                        "tensor_c_add_bias_ub"
                    )
                else:
                    tensor_alpha_c_add_beta_bias = tvm.compute(
                        aligned_out_shapes,
                        lambda *indices: tensor_beta_bias(*indices) + tensor_alpha_c(*indices),
                        "tensor_c_add_bias_ub"
                    )
        else:
            tensor_alpha_c_add_beta_bias = tensor_alpha_c

        if self.need_cast_to_fp16:
            tensor_alpha_c_add_beta_bias = tvm.compute(
                tensor_alpha_c_add_beta_bias.shape,
                lambda *indices: shape_util.cast(tensor_alpha_c_add_beta_bias(*indices), dtype="float16"),
                name="tensor_cast_to_fp16"
            )

        need_reformat_to_nd = ((self.format_out == "ND") and (cur_format_out == "FRACTAL_NZ"))
        if need_reformat_to_nd:
            tensor_out_nd = self.compute_Nz2nd(tensor_alpha_c_add_beta_bias)
            return tensor_out_nd
        return tensor_alpha_c_add_beta_bias

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

    def _get_quantize_params(self, quantize_params=None, out_type="float16"):
        """
        algorithm: check matmul quantize parameters

        Parameters:
        quantize_params: quantization parameters,
                not None means enable quantization, it is dictionary structure

            quantize_alg: quantize mode,
                support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

            scale_mode_a: tensor_a inbound quantization mode,
                    support 'SCALAR' and 'VECTOR'
            scale_mode_b: tensor_b inbound quantization mode,
                    support 'SCALAR' and 'VECTOR'
            scale_mode_out: out tensor quantization mode,
                    support 'SCALAR' and 'VECTOR'

            sqrt_mode_a: tensor_a inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
            sqrt_mode_b: tensor_b inbound sqrt mode, support 'NON_SQRT' and 'SQRT'
            sqrt_mode_out: out tensor sqrt mode, support 'NON_SQRT' and 'SQRT'

            scale_q_a: scale placeholder for tensor_a inbound quantization
            offset_q_a: offset placeholder for tensor_a inbound quantization
            scale_q_b: scale placeholder for tensor_b inbound quantization
            offset_q_b: offset placeholder for tensor_b inbound quantization

            scale_drq: scale placeholder for requantization or dequantization
            offset_drq: scale placeholder for requantization or dequantization

        Returns:
            scale_drq: DISABLE: dequant, ENABLE: requant
            scale_drq_tensor: scale drq tensor
            sqrt_out: NON_SQRT: none sqrt quantize, SQRT: sqrt quantize
        """
        # quantization parameter default value
        sqrt_out = "NON_SQRT"
        scale_drq_tensor = None
        scale_drq = "DISABLE"

        # check input quantization parameter info
        if quantize_params is not None:
            sqrt_out = quantize_params["sqrt_mode_out"]
            if sqrt_out not in ("NON_SQRT", "SQRT"):
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "'scale_mode_out' is {}, should be 'NON_SQRT' or 'SQRT'.".format(sqrt_out)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            # check out dyte and tensor drq
            if out_type == "float16":
                if "scale_drq" not in quantize_params:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "'scale_drq' is None, should not be None" \
                            " while out_dtype is 'float16'."
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
                scale_drq = "ENABLE"
                scale_drq_tensor = quantize_params["scale_drq"]
                if scale_drq_tensor is None:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "scale_drq_tensor is None, need to supply it."
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
                if "offset_drq" in quantize_params:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "'offset_drq' is unnecessary, please delete it."
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            else:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "'dst_dtype' is {}, should be 'float16'".format(out_type)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        return scale_drq, scale_drq_tensor, sqrt_out

    def _check_quantize_params(self, quantize_params=None):  # pylint: disable=R0912
        """
        Parameters:
        quantize_params: quantization parameters,
                not None means enable quantization, it is dictionary structure

            quantize_alg: quantize mode,
                support 'NON_OFFSET' 'HALF_OFFSET_A' 'HALF_OFFSET_B' 'ALL_OFFSET'

            scale_mode_a: tensor_a inbound quantization mode,
                    support 'SCALAR' and 'VECTOR'
            scale_mode_b: tensor_b inbound quantization mode,
                    support 'SCALAR' and 'VECTOR'
            scale_mode_out: out tensor quantization mode,
                    support 'SCALAR' and 'VECTOR'

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

        # check quantize_alg value
        if "quantize_alg" not in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Lack of 'quantize_alg', need to supply it."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        quantize_mode = quantize_params["quantize_alg"]
        if quantize_mode not in ("NON_OFFSET", "HALF_OFFSET_A"):
            dict_args = {
                'errCode': 'E61001',
                'reason': "quantize_alg is {}, it should be 'NON_OFFSET' or" \
                    "'HALF_OFFSET_A'.".format(quantize_mode)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        # check inbound scale mode paras
        if "scale_mode_a" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound scale mode a function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if "scale_mode_b" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound scale mode b function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if "scale_q_a" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound scale quant a function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if "offset_q_a" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound offset quant a function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if "scale_q_b" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound scale quant b function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if "offset_q_b" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound offset quant b function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        # check outbound scale mode paras
        if "scale_mode_out" not in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Lack of 'scale_mode_out', need to supply it."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        scale_out = quantize_params["scale_mode_out"]
        if scale_out not in ("SCALAR", "VECTOR"):
            dict_args = {
                'errCode': 'E61001',
                'reason': "'scale_mode_out' is {}, should be 'SCALAR' or 'VECTOR'.".format(scale_out)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        # check inbound scale mode paras
        if "sqrt_mode_a" in quantize_params or "sqrt_mode_b" in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Inbound sqrt mode function is not supported."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        # check outbound sqrt mode paras
        if "sqrt_mode_out" not in quantize_params:
            dict_args = {
                'errCode': 'E61001',
                'reason': "Lack of 'sqrt_mode_out', need to supply it."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

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
            reason = "The current format_a is not supported, format_a is {}".format(self.format_a)
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        else:
            self.format_a = format_a_temp

        format_b_temp = merge_format_dict.get(self.format_b)
        if format_b_temp is None:
            reason = "The current format_b is not supported, format_b is {}".format(self.format_b)
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        else:
            self.format_b = format_b_temp

        self._set_output_format()
        merge_format_output_dict = {
            "ND": "ND",
            "NC1HWC0": "NC1HWC0",
            "FRACTAL_NZ": "FRACTAL_NZ"
        }
        format_out_temp = merge_format_output_dict.get(self.format_out)
        if format_out_temp is None:
            reason = "The current format_out is not supported, format_out is {}".format(self.format_out)
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
        else:
            self.format_out = format_out_temp
        # ---------- handle dataflow ---------- #
        connect_str = "2"
        self.ops_data_flow_mode = connect_str.join([self.type_map.get(self.src_dtype), self.type_map.get(self.dst_dtype)])
        need_cast_to_fp16 = False
        need_cast_to_fp16 = need_cast_to_fp16 or (self.src_dtype == "float16" and self.dst_dtype == "float16")
        self.need_cast_to_fp16 = need_cast_to_fp16

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
                      "input dtype is {}, output dtype is {}".format(self.src_dtype, self.dst_dtype))
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

        self.block_reduce = (tbe_platform.BLOCK_REDUCE_INT8
            if self.ops_data_flow_mode == "int82int32" else self.block_reduce)
        self.matrix_type = "int32" if self.ops_data_flow_mode == "int82int32" else self.matrix_type
        self.l0c_support_fp32 = True
        support_type = tbe_platform.getValue("Intrinsic_mmad")
        if "f162f32" not in support_type:
            self.l0c_support_fp32 = False
        if self.matrix_type == "float32" and (not self.l0c_support_fp32):
            self.matrix_type = "float16"

    def _set_ori_k(self):
        a_shape = [self._get_value(i) for i in self.tensor_a.shape]
        b_shape = [self._get_value(i) for i in self.tensor_b.shape]
        trans_a = (not self.trans_a) if (self.format_a == "FRACTAL_NZ") else self.trans_a
        trans_b = (not self.trans_b) if (self.format_b == "FRACTAL_NZ") else self.trans_b
        offset_a = 0
        if len(a_shape) in (3, 5):
            offset_a = 1
        offset_b = 0
        if len(b_shape) in (3, 5):
            offset_b = 1
        self.ori_km_shape = a_shape[offset_a] if trans_a else a_shape[1+offset_a]
        self.ori_kn_shape = b_shape[1+offset_b] if trans_b else b_shape[offset_b]

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
            raise RuntimeError("[E69999] A matrix's k should be equal B matrix's k.")

    def _check_n_align(self):
        """
        When the input and output tensors share the memory.
        n not align to block_out is not supported.
        Input: None
        --------------------------
        Return: None
        """
        b_shape = [self._get_value(i) for i in self.tensor_b.shape]
        n_shape = b_shape[0] if self.trans_b else b_shape[1]
        if self.format_b == "ND" and (n_shape % tbe_platform.BLOCK_OUT != 0) and self.alpha is not None:
            reason = ("When the input format is ND, "
                      "the n direction must be aligned to {}.".format(tbe_platform.BLOCK_OUT))
            args_dict = {
                "errCode": "E60108",
                "op_name": "GEMM",
                "reason": reason
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )

    def _check_gevm_and_gemv(self):
        a_shape = self.a_shape
        b_shape = self.b_shape
        if len(a_shape) in (4, 5):
            block_in = a_shape[-2]
            k_shape = a_shape[-4] if self.format_a == "FRACTAL_NZ" else a_shape[-3]
            k_align_flag = (block_in == tbe_platform.BLOCK_VECTOR) and (k_shape % tbe_platform.BLOCK_IN != 0)
            if k_align_flag:
                args_dict = {
                    "errCode": "E61001",
                    "op_name": "GEMM",
                    "reason": "for fractal gevm input, K should be multiple of {}.".format(tbe_platform.BLOCK_IN
                        * self.block_reduce)
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

        if len(b_shape) in (4, 5):
            block_out = b_shape[-1] if self.format_b == "FRACTAL_NZ" else b_shape[-2]
            k_shape = b_shape[-3] if self.format_b == "FRACTAL_NZ" else b_shape[-4]
            k_align_flag = (block_out == tbe_platform.BLOCK_VECTOR) and (k_shape % tbe_platform.BLOCK_OUT != 0)
            if k_align_flag:
                args_dict = {
                    "errCode": "E61001",
                    "op_name": "GEMM",
                    "reason": "for fractal gemv input, K should be multiple of {}.".format(tbe_platform.BLOCK_OUT
                        * self.block_reduce)
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))