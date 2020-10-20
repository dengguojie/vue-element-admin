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
from __future__ import absolute_import  # pylint: disable=too-many-lines

from te.lang.cce.te_compute.util import check_input_tensor_shape
from te.platform import cce_conf
from te.platform import cce_params
from te.tvm import api as tvm
from te.tvm.tensor import Tensor
from te.utils import check_para
from te.utils import operate_shape
from te.utils.error_manager import error_manager_util


def _shape_check(
        tensor_a,  # pylint: disable=C0301, R0912, R0913, R0914, R0915
        tensor_b,
        tensor_bias,
        tensor_alpha,
        tensor_beta,
        trans_a,
        trans_b,
        format_a,
        format_b,
        dst_dtype):
    """
    Check the given input if legal

    Parameters:
    shape_a: list or tuple
            Shape of the first tensor a with rank > 1
    shape_b:  list or tuple
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

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    check_input_tensor_shape(tensor_a)
    check_input_tensor_shape(tensor_b)

    shape_a = [i.value for i in tensor_a.shape]
    shape_b = [i.value for i in tensor_b.shape]
    shape_bias = ()

    shape_len_a = len(shape_a)
    shape_len_b = len(shape_b)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    if tensor_bias is not None:
        shape_bias = [i.value for i in tensor_bias.shape]

    if (in_a_dtype in ("uint8", "int8")) and in_b_dtype == "int8":
        k_block_size = cce_params.BLOCK_REDUCE_INT8
    else:
        k_block_size = cce_params.BLOCK_REDUCE

    if dst_dtype == "int32" and len(shape_bias) == 2:
        for index, value in enumerate(shape_bias):
            if index == 0:
                block = cce_params.BLOCK_IN
            else:
                block = cce_params.BLOCK_OUT
            shape_bias[index] = ((value + block - 1) // block) * block

    def _check_dtype():
        # check type of tensor_alpha and tensor_beta
        if tensor_alpha.dtype != tensor_beta.dtype:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "dtype",
                "param1_name": "alpha",
                "param1_value": "{}".format(tensor_alpha.dtype),
                "param2_name": "beta",
                "param2_value": "{}".format(tensor_beta.dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if dst_dtype != tensor_alpha.dtype:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "dtype",
                "param1_name": "y",
                "param1_value": "{}".format(dst_dtype),
                "param2_name": "alpha",
                "param2_value": "{}".format(tensor_alpha.dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        # ND and fractal support 'float16' and 'b8'
        if not (in_a_dtype == "float16" and in_b_dtype == "float16") and \
                not (in_a_dtype in ("uint8", "int8") and (
                    in_b_dtype == "int8")):
            args_dict = {
                "errCode": "E60005",
                "param_name": "in_a_dtype/in_b_dtype",
                "expected_dtype_list":
                "float16 & float16 and uint8/int8 & int8",
                "dtype": "{}/{}".format(in_a_dtype, in_b_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if dst_dtype not in ("float16", "float32", "int32"):
            args_dict = {
                "errCode": "E60005",
                "param_name": "y",
                "expected_dtype_list": "[float16, float32,int32]",
                "dtype": "{}".format(dst_dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    def _check_fractal():
        if format_a not in ("ND", "fractal"):
            args_dict = {
                "errCode": "E60004",
                "param_name": "a",
                "expected_format_list": "[ND, fractal]",
                "format": "{}".format(format_a)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if format_b not in ("ND", "fractal"):
            args_dict = {
                "errCode": "E60004",
                "param_name": "b",
                "expected_format_list": "[ND, fractal]",
                "format": "{}".format(format_b)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        # fractal and ND not support
        if is_fractal_a and not is_fractal_b:
            args_dict = {
                "errCode":
                "E60114",
                "reason":
                "Not support a is fractal and b is ND!",
                "value":
                "is_fractal_a = {} and is_fractal_b"
                " = {}".format(is_fractal_a, is_fractal_b)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

        if (is_fractal_a == is_fractal_b) and (shape_len_a != shape_len_b):
            args_dict = {
                "errCode": "E60002",
                "attr_name": "dim",
                "param1_name": "a",
                "param1_value": "{}".format(shape_len_a),
                "param2_name": "b",
                "param2_value": "{}".format(shape_len_b)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    _check_dtype()
    _check_fractal()

    def _check_shape():
        if is_fractal_a:
            if shape_len_a not in (4, 5):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor a",
                    "expected_length": "[4,5]",
                    "length": "{}".format(shape_len_a)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
        else:
            if shape_len_a not in (2, 3):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor a",
                    "expected_length": "[2,3]",
                    "length": "{}".format(shape_len_a)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

        if is_fractal_b:
            if shape_len_b not in (4, 5):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor b",
                    "expected_length": "[4,5]",
                    "length": "{}".format(shape_len_b)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
        else:
            if shape_len_b not in (2, 3):
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "tensor b",
                    "expected_length": "[2,3]",
                    "length": "{}".format(shape_len_b)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

        if shape_len_a in (3, 5):
            if tensor_a.shape[0].value != tensor_b.shape[0].value:
                args_dict = {
                    "errCode": "E60002",
                    "attr_name": "shape",
                    "param1_name": "tensor a",
                    "param1_value": "{}".format(tensor_a.shape[0].value),
                    "param2_name": "tensor b",
                    "param2_value": "{}".format(tensor_b.shape[0].value)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

    _check_shape()

    def _check_a_m_k_n():
        is_vector_a = False
        if not is_fractal_a:
            # shape_len_a is 2 or 3
            if trans_a:
                m_shape = shape_a[shape_len_a - 1]
                km_shape = shape_a[shape_len_a - 2]
            else:
                m_shape = shape_a[shape_len_a - 2]
                km_shape = shape_a[shape_len_a - 1]
            real_shape_m = m_shape
        else:
            if trans_a:
                m_shape = shape_a[shape_len_a - 3]
                km_shape = shape_a[shape_len_a - 4]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            else:
                m_shape = shape_a[shape_len_a - 4]
                km_shape = shape_a[shape_len_a - 3]
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            real_shape_m = m_shape * a_block_in

            if a_block_reduce != k_block_size:
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor a",
                    "expected_two_dims":
                    "({},{})".format(cce_params.BLOCK_IN, cce_params.BLOCK_VECTOR),
                    "actual_two_dim": "{}".format(a_block_reduce)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

            if a_block_in not in (cce_params.BLOCK_VECTOR, cce_params.BLOCK_IN):
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor a",
                    "expected_two_dims":
                    "({},{})".format(cce_params.BLOCK_IN, cce_params.BLOCK_VECTOR),
                    "actual_two_dim": "{}".format(a_block_in)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
            if a_block_in == cce_params.BLOCK_VECTOR:
                is_vector_a = True
                if m_shape != cce_params.BLOCK_VECTOR:
                    args_dict = {
                        "errCode":
                        "E60101",
                        "param_name":
                        "tensor a",
                        "expected_two_dims":
                        "({},{})".format(cce_params.BLOCK_IN, cce_params.BLOCK_VECTOR),
                        "actual_two_dim":
                        "{}".format(m_shape)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
                if km_shape % (cce_params.BLOCK_IN) != 0:
                    args_dict = {
                        "errCode":
                        "E60114",
                        "reason":
                        "k should be multiple of {}".format(cce_params.BLOCK_IN *
                                                            k_block_size),
                        "value":
                        "k = {}".format(km_shape)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
        return km_shape, real_shape_m, is_vector_a

    km_shape, real_shape_m, is_vector_a = _check_a_m_k_n()

    def _check_b_m_k_n(is_vector_a):  # pylint: disable=too-many-branches
        is_gemv = False
        b_block_reduce = 1
        b_block_out = 1

        def _get_nd_m_k_n():
            # shape_len_b is 2 or 3
            if trans_b:
                kn_shape = shape_b[shape_len_b - 1]
                n_shape = shape_b[shape_len_b - 2]
            else:
                kn_shape = shape_b[shape_len_b - 2]
                n_shape = shape_b[shape_len_b - 1]

            return kn_shape, n_shape

        if not is_fractal_b:
            kn_shape, n_shape = _get_nd_m_k_n()
        else:
            if trans_b:
                kn_shape = shape_b[shape_len_b - 3]
                n_shape = shape_b[shape_len_b - 4]
                b_block_reduce = shape_b[shape_len_b - 2]
                b_block_out = shape_b[shape_len_b - 1]
            else:
                kn_shape = shape_b[shape_len_b - 4]
                n_shape = shape_b[shape_len_b - 3]
                b_block_reduce = shape_b[shape_len_b - 1]
                b_block_out = shape_b[shape_len_b - 2]

            if b_block_reduce != k_block_size:
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor b",
                    "expected_two_dims": "{}".format(k_block_size),
                    "actual_two_dim": "{}".format(b_block_reduce)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

            if b_block_out not in (cce_params.BLOCK_VECTOR, cce_params.BLOCK_IN):
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor b",
                    "expected_two_dims":
                    "({},{})".format(cce_params.BLOCK_IN, cce_params.BLOCK_VECTOR),
                    "actual_two_dim": "{}".format(b_block_out)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
            if b_block_out == cce_params.BLOCK_VECTOR:
                is_gemv = True
                if is_vector_a:
                    args_dict = {
                        "errCode": "E60114",
                        "reason": "input shape M and N can't both be 1",
                        "value": "input shape M and N are both 1"
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
                if n_shape != 1:
                    args_dict = {
                        "errCode":
                        "E60101",
                        "param_name":
                        "tensor b",
                        "expected_two_dims":
                        "({},{})".format(cce_params.BLOCK_IN, cce_params.BLOCK_VECTOR),
                        "actual_two_dim":
                        "{}".format(n_shape)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
                if kn_shape % (cce_params.BLOCK_IN) != 0:
                    args_dict = {
                        "errCode":
                        "E60114",
                        "reason":
                        "k should be multiple of {}".format(cce_params.BLOCK_IN *
                                                            k_block_size),
                        "value":
                        "k = {}".format(kn_shape)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
                # gemv u8/s8 is transed to gevm(s8/u8), s8/u8 is not support
                # for mad intri
                if in_a_dtype == "uint8" and in_b_dtype == "int8":
                    args_dict = {
                        "errCode": "E60005",
                        "param_name": "in_a_dtype/in_b_dtype",
                        "expected_dtype_list": "int8 & int8",
                        "dtype": "{}/{}".format(in_a_dtype, in_b_dtype)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))

        return is_gemv, b_block_out, kn_shape, n_shape

    is_gemv, b_block_out, kn_shape, n_shape = _check_b_m_k_n(is_vector_a)

    def _check_a_between_b():
        if is_fractal_a == is_fractal_b:
            if km_shape != kn_shape:
                args_dict = {
                    "errCode": "E60114",
                    "reason": "reduce axis not same",
                    "value": "reduce axis not same"
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

    _check_a_between_b()

    def renew_is_gemv(is_gemv):
        if not is_fractal_a and not is_fractal_b:
            is_gemv = n_shape == 1
        return is_gemv

    is_gemv = renew_is_gemv(is_gemv)

    def _check_bias():
        if shape_bias:
            if len(shape_bias) != 2 and len(shape_bias) != 4:
                args_dict = {
                    "errCode": "E60006",
                    "param_name": "c",
                    "expected_length": "2 or 4",
                    "length": "{}".format(len(shape_bias))
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

    _check_bias()


@check_para.check_input_type(Tensor, Tensor,
                             Tensor, Tensor, bool, bool,
                             str, str, str, (type(None), Tensor))
def gemm(
        tensor_a,  # pylint: disable=R1702, R0912, R0913, R0914, R0915
        tensor_b,
        tensor_alpha,
        tensor_beta,
        trans_a=False,
        trans_b=False,
        format_a="ND",
        format_b="ND",
        dst_dtype="float16",
        tensor_bias=None,
        quantize_params=None,
        kernel_name="gemm"):
    """
    algorithm: mmad
    calculating  matrix multiplication, C=alpha_num*A*B+beta_num*C

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b with the same type and shape with a

              If tensor_a/tensor_b is int8/uint8,then L0A must be 16*32,L0B
              must be 32*16.
              If A is transpose , then AShape classification matrix must be
              32*16 in gm/L1,then it is 16*32 in L0A.
              If B is transpose , then BShape classification matrix must be
              16*32 in gm/L1,then it is 32*16 in L0B.

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: If type is bool, a and b's format both be fractal or ND,
                default is ND;
                If type is list, len must be 2, [0] is is_fractal_a,
                [1] is is_fractal_b

    alpha_num: scalar used for multiplication

    beta_num: scalar used for multiplication

    dst_dtype: output data type,support "float16" "float32", default is "float16"

    tensor_bias :the bias with used to init L0C for tensor c

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

    Returns None
    """
    def _get_dtype():
        if not is_fractal_a and not is_fractal_b and tensor_a.dtype == "int8" \
                and tensor_bias.dtype == "float32":
            in_a_dtype = "float16"
            in_b_dtype = "float16"
            is_nd_int82fp32 = True
        else:
            in_a_dtype = tensor_a.dtype
            in_b_dtype = tensor_b.dtype
            is_nd_int82fp32 = False
        return in_a_dtype, in_b_dtype, is_nd_int82fp32

    nz_a = False
    if format_a == "FRACTAL_NZ":
        nz_a = True
        format_a = "fractal"

    nz_b = False
    if format_b == "FRACTAL_NZ":
        nz_b = True
        format_b = "fractal"

    def _compute_alpha_beta():
        if tensor_alpha.dtype == "float16":
            tensor_alpha_temp_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: tensor_alpha(  # pylint: disable=W0108
                    *indices),
                name='tensor_alpha_temp_ub',
            )

            tensor_beta_temp_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: tensor_beta(  # pylint: disable=W0108
                    *indices),
                name='tensor_beta_temp_ub',
            )

            tensor_alpha_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: operate_shape.cast(
                    tensor_alpha_temp_ub(*indices), dtype="float32"),
                name='tensor_alpha_ub',
            )
            tensor_beta_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: operate_shape.cast(
                    tensor_beta_temp_ub(*indices), dtype="float32"),
                name='tensor_beta_ub',
            )
        else:
            tensor_alpha_ub = tvm.compute(
                tensor_alpha.shape,
                lambda *indices: tensor_alpha(  # pylint: disable=W0108
                    *indices),
                name='tensor_alpha_ub',
            )
            tensor_beta_ub = tvm.compute(
                tensor_beta.shape,
                lambda *indices: tensor_beta(  # pylint: disable=W0108
                    *indices),
                name='tensor_beta_ub',
            )
        return tensor_alpha_ub, tensor_beta_ub

    tensor_alpha_ub, tensor_beta_ub = _compute_alpha_beta()

    _shape_check(tensor_a, tensor_b, tensor_bias, tensor_alpha, tensor_beta,
                 trans_a, trans_b, format_a, format_b, dst_dtype)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    in_a_dtype, in_b_dtype, is_nd_int82fp32 = _get_dtype()

    def _get_output_type():
        l0c_support_fp32 = cce_conf.intrinsic_check_support("Intrinsic_mmad", "f162f32")

        def _out_dtype():
            if in_a_dtype == "float16" and in_b_dtype == "float16":
                if dst_dtype not in ("float16", "float32"):
                    args_dict = {
                        "errCode": "E60003",
                        "a_dtype": in_a_dtype,
                        "expected_dtype_list": "float16, float32",
                        "out_dtype": "{}".format(dst_dtype)
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
                out_dtype = "float32"
                if not l0c_support_fp32:
                    out_dtype = "float16"
            elif (in_a_dtype == "int8" and in_b_dtype == "int8") or \
                    (in_a_dtype == "uint8" and in_b_dtype == "int8"):
                out_dtype = "int32"
            else:
                args_dict = {
                    "errCode":
                    "E60114",
                    "reason":
                    "data type of tensor not supported",
                    "value":
                    "in_a_dtype = {},"
                    " in_b_dtype = {}".format(in_a_dtype, in_b_dtype)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
            if (out_dtype == dst_dtype) and (quantize_params is not None):
                args_dict = {
                    "errCode": "E60000",
                    "param_name": "quantize_params",
                    "expected_value": "None",
                    "input_value": "{}".format(quantize_params)
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))

            if dst_dtype not in (out_dtype,
                                 "float16") and not (dst_dtype == "float32"
                                                     and out_dtype == "int32"):
                args_dict = {
                    "errCode":
                    "E60114",
                    "reason":
                    "y_dtype should be float16 for a_dtype ="
                    " {} and b_dtype = {}".format(in_a_dtype, in_b_dtype),
                    "value":
                    dst_dtype
                }
                raise RuntimeError(
                    args_dict, error_manager_util.get_error_message(args_dict))
            return out_dtype

        out_dtype = _out_dtype()

        if (out_dtype not in (dst_dtype, "float32")) and (
                quantize_params is None) and not (dst_dtype == "float32"
                                                  and out_dtype == "int32"):
            args_dict = {"errCode": "E60001", "param_name": "quantize_params"}
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if (quantize_params
                is not None) and (not isinstance(quantize_params, dict)):
            args_dict = {
                "errCode": "E60005",
                "param_name": "quantize_params",
                "expected_dtype_list": "[dict]",
                "dtype": "{}".format(type(quantize_params))
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        if in_a_dtype == "int8" and dst_dtype == "float32":
            out_dtype = "float32"
        return l0c_support_fp32, out_dtype

    l0c_support_fp32, out_dtype = _get_output_type()

    tensor_a_length = len(tensor_a.shape)
    tensor_b_length = len(tensor_b.shape)

    def _get_bias_shape():
        if tensor_bias.dtype != dst_dtype:
            args_dict = {
                "errCode": "E60005",
                "param_name": "c",
                "expected_dtype_list": "[{}]".format(dst_dtype),
                "dtype": "{}".format(tensor_bias.dtype)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))
        bias_shape = list(tensor_bias.shape)
        if len(bias_shape) == 2:
            origin_bias_shape = bias_shape.copy()
            for index, value in enumerate(bias_shape):
                if index == 0:
                    block = cce_params.BLOCK_IN
                else:
                    block = cce_params.BLOCK_OUT
                bias_shape[index] = ((value + block - 1) // block) * block
        else:
            origin_bias_shape = None
        return bias_shape, origin_bias_shape

    if tensor_bias is not None:
        bias_shape, origin_bias_shape = _get_bias_shape()

    def _get_block():
        if in_a_dtype == "float16":
            block_reduce = cce_params.BLOCK_REDUCE
        else:
            block_reduce = cce_params.BLOCK_REDUCE_INT8

        block_in = cce_params.BLOCK_IN
        block_out = cce_params.BLOCK_OUT
        return block_reduce, block_in, block_out

    block_reduce, block_in, block_out = _get_block()
    gm_a_shape_normalize = []

    def _get_a_martix_shape(gm_a_shape_normalize):
        if trans_a:
            if is_fractal_a:
                m_shape = tensor_a.shape[tensor_a_length - 3].value
                m_shape_ori = m_shape
                km_shape = tensor_a.shape[tensor_a_length - 4].value
                km_shape_ori = km_shape
                gm_a_shape_normalize = tensor_a.shape
            else:
                if is_nd_int82fp32:
                    m_shape = (tensor_a.shape[tensor_a_length - 1].value + 32 -
                               1) // 32 * 32 // 16
                    km_shape = (tensor_a.shape[tensor_a_length - 2].value +
                                32 - 1) // 32 * 32 // 16
                else:
                    if in_a_dtype == "int8":
                        m_shape = ((
                            (tensor_a.shape[tensor_a_length - 1].value + 32 -
                             1) // 32) * 32) // 16
                    else:
                        m_shape = (tensor_a.shape[tensor_a_length - 1].value +
                                   block_in - 1) // block_in
                    km_shape = (tensor_a.shape[tensor_a_length - 2].value +
                                block_reduce - 1) // block_reduce
                m_shape_ori = tensor_a.shape[tensor_a_length - 1].value
                km_shape_ori = tensor_a.shape[tensor_a_length - 2].value
                gm_a_shape_normalize.append(km_shape * block_reduce)
                gm_a_shape_normalize.append(m_shape * block_in)
        else:
            if is_fractal_a:
                m_shape = tensor_a.shape[tensor_a_length - 4].value
                m_shape_ori = m_shape
                km_shape = tensor_a.shape[tensor_a_length - 3].value
                km_shape_ori = km_shape
                gm_a_shape_normalize = tensor_a.shape
            else:
                if is_nd_int82fp32:
                    m_shape = (tensor_a.shape[tensor_a_length - 2].value + 32 -
                               1) // 32 * 32 // 16
                    km_shape = (tensor_a.shape[tensor_a_length - 1].value +
                                32 - 1) // 32 * 32 // 16
                else:
                    if in_a_dtype == 'int8':
                        m_shape = ((
                            (tensor_a.shape[tensor_a_length - 2].value + 32 -
                             1) // 32) * 32) // 16
                    else:
                        m_shape = (tensor_a.shape[tensor_a_length - 2].value +
                                   block_in - 1) // block_in
                    km_shape = (tensor_a.shape[tensor_a_length - 1].value +
                                block_reduce - 1) // block_reduce
                m_shape_ori = tensor_a.shape[tensor_a_length - 2].value
                km_shape_ori = tensor_a.shape[tensor_a_length - 1].value
                gm_a_shape_normalize.append(m_shape * block_in)
                gm_a_shape_normalize.append(km_shape * block_reduce)

        return m_shape, m_shape_ori, km_shape, km_shape_ori, \
            gm_a_shape_normalize

    m_shape, m_shape_ori, km_shape, km_shape_ori, \
        gm_a_shape_normalize = _get_a_martix_shape(gm_a_shape_normalize)

    gm_b_shape_normalize = []

    def _get_b_martix_shape(gm_b_shape_normalize):
        if trans_b:
            if is_fractal_b:
                kn_shape = tensor_b.shape[tensor_b_length - 3].value
                kn_shape_ori = kn_shape
                n_shape = tensor_b.shape[tensor_b_length - 4].value
                n_shape_ori = n_shape
                gm_b_shape_normalize = tensor_b.shape
            else:
                if is_nd_int82fp32:
                    kn_shape = (tensor_b.shape[tensor_b_length - 1].value +
                                32 - 1) // 32 * 32 // 16
                    n_shape = (tensor_b.shape[tensor_b_length - 2].value + 32 -
                               1) // 32 * 32 // 16
                else:
                    kn_shape = (tensor_b.shape[tensor_b_length - 1].value +
                                block_reduce - 1) // block_reduce
                    if in_b_dtype == 'int8':
                        n_shape = ((
                            (tensor_b.shape[tensor_b_length - 2].value + 32 -
                             1) // 32) * 32) // 16
                    else:
                        n_shape = (tensor_b.shape[tensor_b_length - 2].value +
                                   block_out - 1) // block_out
                kn_shape_ori = tensor_b.shape[tensor_b_length - 1].value
                n_shape_ori = tensor_b.shape[tensor_b_length - 2].value
                gm_b_shape_normalize.append(n_shape * block_out)
                gm_b_shape_normalize.append(kn_shape * block_reduce)
        else:
            if is_fractal_b:
                kn_shape = tensor_b.shape[tensor_b_length - 4].value
                kn_shape_ori = kn_shape
                n_shape = tensor_b.shape[tensor_b_length - 3].value
                n_shape_ori = n_shape
                gm_b_shape_normalize = tensor_b.shape
            else:
                if is_nd_int82fp32:
                    kn_shape = (tensor_b.shape[tensor_b_length - 2].value +
                                32 - 1) // 32 * 32 // 16
                    n_shape = (tensor_b.shape[tensor_b_length - 1].value + 32 -
                               1) // 32 * 32 // 16
                else:
                    kn_shape = (tensor_b.shape[tensor_b_length - 2].value +
                                block_reduce - 1) // block_reduce
                    if in_b_dtype == 'int8':
                        n_shape = ((
                            (tensor_b.shape[tensor_b_length - 1].value + 32 -
                             1) // 32) * 32) // 16
                    else:
                        n_shape = (tensor_b.shape[tensor_b_length - 1].value +
                                   block_out - 1) // block_out
                kn_shape_ori = tensor_b.shape[tensor_b_length - 2].value
                n_shape_ori = tensor_b.shape[tensor_b_length - 1].value
                gm_b_shape_normalize.append(kn_shape * block_reduce)
                gm_b_shape_normalize.append(n_shape * block_out)

        return kn_shape, n_shape, n_shape_ori, kn_shape_ori, \
            gm_b_shape_normalize

    kn_shape, n_shape, n_shape_ori, kn_shape_ori, gm_b_shape_normalize \
        = _get_b_martix_shape(gm_b_shape_normalize)

    def _check_k():
        # check shape
        if km_shape != kn_shape:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "km_shape",
                "param1_value": "{}".format(km_shape),
                "param2_name": "kn_shape",
                "param2_value": "{}".format(kn_shape)
            }
            raise RuntimeError(args_dict,
                               error_manager_util.get_error_message(args_dict))

    _check_k()

    def _check_shape():
        if is_fractal_a:
            if trans_a:
                if not (tensor_a.shape[tensor_a_length - 1].value
                        == block_reduce and
                        tensor_a.shape[tensor_a_length - 2].value == block_in):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "AShape classification matrix is wrong",
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
            else:
                if not (tensor_a.shape[tensor_a_length - 2].value == block_in
                        and tensor_a.shape[tensor_a_length - 1].value
                        == block_reduce):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "AShape classification matrix is wrong",
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
        if is_fractal_b:
            if trans_b:
                if not (tensor_b.shape[tensor_b_length - 2].value == block_reduce
                        and tensor_b.shape[tensor_b_length - 1].value == block_out):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "BShape classification matrix is wrong",
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))
            else:
                if not (tensor_b.shape[tensor_b_length - 2].value == block_out
                        and tensor_b.shape[tensor_b_length - 1].value == block_reduce):
                    args_dict = {
                        "errCode": "E60108",
                        "reason": "BShape classification matrix is wrong",
                    }
                    raise RuntimeError(
                        args_dict,
                        error_manager_util.get_error_message(args_dict))

    _check_shape()

    def _get_reduce():
        # kBurstAxis and kPointAxis
        if in_a_dtype == "int8" and dst_dtype == "float32":
            reduce_kp = tvm.reduce_axis((0, 16), name='kp')
            reduce_kb = tvm.reduce_axis((0, km_shape * 2), name='kb')
        else:
            reduce_kp = tvm.reduce_axis((0, block_reduce), name='kp')
            reduce_kb = tvm.reduce_axis((0, km_shape), name='kb')
        return reduce_kp, reduce_kb

    reduce_kp, reduce_kb = _get_reduce()

    def _get_optmt_flag():
        optmt_a = 0
        optmt_b = 0
        optmt_c = 0
        if in_a_dtype in {"float16", "int8"}:
            optmt_a = 1
        if in_b_dtype in {"float16", "int8"}:
            optmt_b = 1
        if dst_dtype in ("float16", "float32", "int32"):
            optmt_c = 1
        return optmt_a, optmt_b, optmt_c

    optmt_a, optmt_b, optmt_c = _get_optmt_flag()

    out_shape = (int(n_shape), int(m_shape), int(block_in), int(block_out))
    out_shape_ori = [int(m_shape_ori), int(n_shape_ori)]


    def _do_align(tensor_need_align, shape_aligned, name, in_dtype):
        """
        do align for a_martix or b_martix, We have two way to pad zero.
        do align for a_martix, we pad zero by zero martix.
        do align for b_martix, if (n_axis_len / k_axis_len) > 2, we pad
        zero by zero martix , otherwise pad zero along the way
        input:
            tensor_need_align: tensor, the tensor need align
            shape_aligned: shape, tensor_need_align's aligned shape
            name: str, a or b
            in_dtype: str, input data type
        return:
            aligned tensor
        """
        factor = 32
        if in_dtype == "float16":
            factor = 16

        shape_ori = tensor_need_align.shape
        ax_outer = int(tensor_need_align.shape[0])
        ax_inner = int(tensor_need_align.shape[1])

        use_zero_martix = (name == "a") or ((ax_inner / ax_outer) > 2)

        if use_zero_martix:
            tensor_zero = tvm.compute(
                shape_aligned,
                lambda *indice: tvm.convert(0).astype(in_dtype),
                name="tensor_{}_zero".format(name),
                tag="init_zero")
            tensor_normalize_ub = tvm.compute(
                shape_aligned,
                lambda i, j: tvm.select(
                    i < ax_outer,
                    tvm.select(j < ax_inner, tensor_need_align[i, j],
                               tensor_zero[i, j]), tensor_zero[i, j]),
                name='tensor_{}_normalize_ub'.format(name))

            return tensor_normalize_ub
        else:
            tensor_normalize_ub = tvm.compute(
                shape_aligned,
                lambda i, j: tvm.select(
                    i < ax_outer,
                    tvm.select(j < ax_inner, tensor_need_align[i, j],
                               tvm.convert(0).astype(in_dtype)),
                    tvm.convert(0).astype(in_dtype)
                    ),
                name='tensor_{}_normalize_ub'.format(name))
            return tensor_normalize_ub


    def check_shape_align(shape, factor):
        is_align = True
        for item in shape:
            if item.value % factor != 0:
                is_align = False
                break
        return is_align

    def _compute_bias():
        tensor_bias_ub_fract, tensor_beta_bias_ub, tensor_bias_ub = None, \
            None, None
        if len(bias_shape) == 2:
            if not is_fractal_a:
                bias_m_shape_ori = tensor_bias.shape[0]
                bias_n_shape_ori = tensor_bias.shape[1]
                ub_bias_shape_normalize = [
                    m_shape * block_in, n_shape * block_out]
                tensor_bias_ub = tvm.compute(
                    ub_bias_shape_normalize,
                    lambda i, j: tvm.select(
                        i < bias_m_shape_ori,
                        tvm.select(j < bias_n_shape_ori, tensor_bias[i, j],
                                   tvm.convert(0).astype(tensor_bias.dtype)),
                        tvm.convert(0).astype(tensor_bias.dtype)),
                    name='tensor_bias_ub')
            else:
                tensor_bias_ub = tvm.compute(
                    bias_shape,
                    lambda i, j: tvm.select(
                        j < origin_bias_shape[-1],
                        tvm.select(i < origin_bias_shape[-2], tensor_bias[i, j],
                                   tvm.convert(0).astype(dst_dtype)),
                        tvm.convert(0).astype(dst_dtype)),
                    name='tensor_bias_ub')
                tensor_bias_ub_fract = tvm.compute(
                    out_shape,
                    lambda i, j, k, l: tensor_bias_ub[j * block_in + k, i *
                                                      block_out + l] + 0,
                    name='tensor_bias_ub_fract')
        elif len(bias_shape) == 4:
            tensor_bias_ub = tvm.compute(
                out_shape,
                lambda *indices: tensor_bias(  # pylint: disable=W0108
                    *indices),
                name='tensor_bias_ub')

        if tensor_bias_ub_fract is not None:
            if tensor_beta_ub.dtype == 'float32' and \
                    tensor_bias_ub_fract.dtype == 'float16':
                tensor_float32_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: operate_shape.cast(
                        tensor_bias_ub_fract(*indices), dtype='float32'),
                    name='tensor_float32_bias_ub',
                )
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_float32_bias_ub(*indices),
                    name='tensor_beta_bias_ub',
                )
            else:
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub_fract.shape,
                    lambda *indices: tensor_beta_ub[0] * tensor_bias_ub_fract(
                        *indices),
                    name='tensor_beta_bias_ub',
                )
        else:
            if tensor_beta_ub.dtype == 'float32' and tensor_bias_ub.dtype == 'float16':
                tensor_float32_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: operate_shape.cast(
                        tensor_bias_ub(*indices), dtype='float32'),
                    name='tensor_float32_bias_ub',
                )
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0] *
                    tensor_float32_bias_ub(*indices),
                    name='tensor_beta_bias_ub',
                )
            else:
                tensor_beta_bias_ub = tvm.compute(
                    tensor_bias_ub.shape,
                    lambda *indices: tensor_beta_ub[0] * tensor_bias_ub(
                        *indices),
                    name='tensor_beta_bias_ub',
                )
        return tensor_beta_bias_ub

    tensor_beta_bias_ub = _compute_bias()

    def _a_nd_part_not_trans():
        tensor_a_l1_shape = (m_shape, km_shape, block_in, block_reduce)
        if in_a_dtype == 'int8':
            is_a_align = check_shape_align(tensor_a.shape, 32)
            if not is_a_align:
                tensor_a_normalize_ub = _do_align(tensor_a,
                                                  gm_a_shape_normalize,
                                                  'a',
                                                  in_a_dtype)
            else:
                tensor_a_normalize_ub = tvm.compute(
                    gm_a_shape_normalize,
                    lambda i, j: tensor_a[i, j],
                    name='tensor_a_normalize_ub')
            tensor_a_fract_k_shape = (m_shape, km_shape, block_in, block_reduce)
            tensor_a_fract_k = tvm.compute(
                tensor_a_fract_k_shape,
                lambda i, j, k, l: tensor_a_normalize_ub[i * block_in + k, j *
                                                         block_reduce + l],
                name='a_fract_k')
            tensor_a_l1 = tvm.compute(
                tensor_a_fract_k_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_a_fract_k(*indices),
                name='tensor_a_l1')
            tensor_a_l0a = tvm.compute(
                tensor_a_l1_shape,
                lambda *indices: tensor_a_l1(*indices),  # pylint: disable=W0108
                name='tensor_a_l0a')
        else:
            if is_nd_int82fp32:
                is_a_align = check_shape_align(tensor_a.shape, 32)
                if not is_a_align:
                    tensor_a_normalize_ub = _do_align(tensor_a,
                                                      gm_a_shape_normalize,
                                                      'a',
                                                      'int8')
                else:
                    tensor_a_normalize_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda i, j: tensor_a[i, j],
                        name='tensor_a_normalize_ub')
                tensor_a_normalize_ub = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: operate_shape.cast(
                        tensor_a_normalize_ub(*indices), "float16"),
                    name="tensor_a_float16_normalize_ub",
                )
            else:
                is_a_align = check_shape_align(tensor_a.shape, 16)
                if not is_a_align:
                    tensor_a_normalize_ub = _do_align(tensor_a,
                                                      gm_a_shape_normalize,
                                                      'a',
                                                      in_a_dtype)
                else:
                    tensor_a_normalize_ub = tvm.compute(
                        gm_a_shape_normalize,
                        lambda i, j: tensor_a[i, j],
                        name='tensor_a_normalize_ub'
                    )
            tensor_a_fract_k_shape = (
                m_shape, km_shape * block_reduce, block_in)
            tensor_a_fract_k = tvm.compute(
                tensor_a_fract_k_shape,
                lambda i, j, k: tensor_a_normalize_ub[i * block_in + k, j],
                name='a_fract_k')
            tensor_a_l1 = tvm.compute(
                tensor_a_fract_k_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_a_fract_k(*indices),
                name='tensor_a_l1')
            tensor_a_l0a = tvm.compute(
                tensor_a_l1_shape,
                lambda i, j, k, l: tensor_a_l1[i, j * block_reduce + l, k],
                name='tensor_a_l0a')
        return tensor_a_l0a

    def _a_part_not_trans():
        if is_fractal_a:
            if nz_a:
                tensor_a_l1 = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(
                    (m_shape, km_shape, block_in, block_reduce),
                    lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                    name='tensor_a_l0a')
            else:
                tensor_a_l1 = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l1')
                tensor_a_l0a = tvm.compute(
                    gm_a_shape_normalize,
                    lambda *indices: tensor_a_l1(  # pylint: disable=W0108
                        *indices),
                    name='tensor_a_l0a')
        else:
            tensor_a_l0a = _a_nd_part_not_trans()
        return tensor_a_l0a

    def _compute_a_matrix():  # pylint: disable=too-many-branches
        if not trans_a:
            tensor_a_l0a = _a_part_not_trans()
        else:

            def _part_trans():
                if is_fractal_a:
                    if nz_a:
                        if in_a_dtype == "int8" and dst_dtype == "float32":
                            tensor_a_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices:  # pylint: disable=W0108
                                tensor_a(*indices),
                                name="tensor_a_ub",
                            )
                            tensor_float16_a_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: operate_shape.cast(
                                    tensor_a_ub(*indices), "float16"),
                                name="tensor_float16_a_ub",
                            )
                            new_a_shape = [
                                gm_a_shape_normalize[1],
                                gm_a_shape_normalize[0] * 2,
                                gm_a_shape_normalize[2],
                                gm_a_shape_normalize[3] // 2,
                            ]
                            tensor_zz_a_ub = tvm.compute(
                                new_a_shape,
                                lambda i, j, k, l: tensor_float16_a_ub[
                                    j // 2, i, k, (j * 16 + l) % 32],
                                name="tensor_zz_a_ub",
                            )
                            tensor_a_l1 = tvm.compute(
                                new_a_shape,
                                lambda *indices:  # pylint: disable=W0108
                                tensor_zz_a_ub(*indices),
                                name='tensor_a_l1')
                            tensor_a_l0a = tvm.compute(
                                new_a_shape,
                                lambda *indices:  # pylint: disable=W0108
                                tensor_a_l1(*indices),
                                name='tensor_a_l0a')
                        else:
                            tensor_a_l1 = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: tensor_a(  # pylint: disable=W0108
                                    *indices),
                                name='tensor_a_l1')
                            tensor_a_l0a = tvm.compute(
                                (m_shape, km_shape, block_in, block_reduce),
                                lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                                name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a(  # pylint: disable=W0108
                                *indices),
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')
                else:
                    if in_a_dtype == "float16":
                        tensor_a_l1_shape = (m_shape, km_shape, block_in, block_reduce)
                        if is_nd_int82fp32:
                            is_a_align = check_shape_align(tensor_a.shape, 32)
                            if not is_a_align:
                                tensor_a_normalize_ub = _do_align(tensor_a,
                                                                  gm_a_shape_normalize,
                                                                  'a',
                                                                  'int8')
                            else:
                                tensor_a_normalize_ub = tvm.compute(
                                    gm_a_shape_normalize,
                                    lambda i, j: tensor_a[i, j],
                                    name='tensor_a_normalize_ub')
                            tensor_a_normalize_ub = tvm.compute(
                                gm_a_shape_normalize,
                                lambda *indices: operate_shape.cast(
                                    tensor_a_normalize_ub(*indices), "float16"
                                ),
                                name="tensor_a_float16_normalize_ub",
                            )
                        else:
                            is_a_align = check_shape_align(tensor_a.shape, 16)
                            if not is_a_align:
                                tensor_a_normalize_ub = _do_align(tensor_a,
                                                                  gm_a_shape_normalize,
                                                                  'a',
                                                                  in_a_dtype)
                            else:
                                tensor_a_normalize_ub = tvm.compute(
                                    gm_a_shape_normalize,
                                    lambda i, j: tensor_a[i, j],
                                    name='tensor_a_normalize_ub'
                                )
                        tensor_a_fract_k_shape = (
                            km_shape, m_shape*block_in, block_reduce)
                        tensor_a_fract_k = tvm.compute(
                            tensor_a_fract_k_shape,
                            lambda i, j, k: tensor_a_normalize_ub[
                                i * block_reduce + k, j],
                            name='a_fract_k')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_fract_k_shape,
                            lambda *indices:  # pylint: disable=W0108
                            tensor_a_fract_k(*indices),
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l1_shape,
                            lambda i, j, k, l: tensor_a_l1[j, i * block_in + k,
                                                           l],
                            name='tensor_a_l0a',
                            attrs={"transpose_a": "true"},
                        )
                    else:
                        is_a_align = check_shape_align(tensor_a.shape, 32)
                        tensor_a_ub_shape = (km_shape * block_reduce,
                                             m_shape * block_in)
                        tensor_a_fract_shape = (
                            m_shape,
                            km_shape,
                            block_in,
                            block_reduce,
                        )
                        if not is_a_align:
                            tensor_a_normalize_ub = _do_align(tensor_a,
                                                              gm_a_shape_normalize,
                                                              'a',
                                                              in_a_dtype)
                        else:
                            tensor_a_normalize_ub = tvm.compute(
                                tensor_a_ub_shape,
                                lambda *indices:  # pylint: disable=W0108
                                tensor_a(*indices),
                                name="tensor_a_normalize_ub",
                            )
                        tensor_a_transpose_shape = (
                            m_shape * block_in,
                            km_shape * block_reduce,
                        )
                        tensor_a_transpose = tvm.compute(
                            tensor_a_transpose_shape,
                            lambda i, j: tensor_a_normalize_ub[j, i],
                            name='a_transpose',
                        )
                        tensor_a_fract = tvm.compute(
                            tensor_a_fract_shape,
                            lambda i, j, k, l: tensor_a_transpose[
                                i * block_in + k, j * block_reduce + l],
                            name="a_fract_k",
                        )
                        tensor_a_l1 = tvm.compute(
                            tensor_a_fract_shape,
                            lambda *indices:  # pylint: disable=W0108
                            tensor_a_fract(*indices),
                            name="tensor_a_l1",
                        )
                        tensor_a_l0a = tvm.compute(
                            tensor_a_fract_shape,
                            lambda *indices:  # pylint: disable=W0108
                            tensor_a_l1(*indices),
                            name="tensor_a_l0a",
                            attrs={"transpose_a": "true"},
                        )
                return tensor_a_l0a

            tensor_a_l0a = _part_trans()
        return tensor_a_l0a

    tensor_a_l0a = _compute_a_matrix()

    def _b_nd_part_not_trans():
        if in_b_dtype == 'int8':
            is_b_align = check_shape_align(tensor_b.shape, 32)
            tensor_b_l1_shape = (kn_shape, n_shape, block_out, block_reduce)
            tensor_b_ub_shape = (kn_shape * block_reduce, n_shape * block_out)
            if is_b_align is False:
                tensor_b_normalize_ub = _do_align(tensor_b,
                                                    gm_b_shape_normalize,
                                                    'b',
                                                    'int8')
            else:
                tensor_b_normalize_ub = tvm.compute(
                    tensor_b_ub_shape,
                    lambda i, j: tensor_b[i, j],
                    name='tensor_b_normalize_ub')
            tensor_b_transpose_shape = (n_shape * block_out,
                                        kn_shape * block_reduce)
            tensor_b_transpose = tvm.compute(
                tensor_b_transpose_shape,
                lambda i, j: tensor_b_normalize_ub[j, i],
                name='b_transpose')
            tensor_b_fract = tvm.compute(
                (kn_shape, n_shape, block_out, block_reduce),
                lambda i, j, k, l: tensor_b_transpose[j * block_in + k, i *
                                                      block_reduce + l],
                name='b_fract')
            tensor_b_l1 = tvm.compute(
                tensor_b_l1_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_b_fract(*indices),
                name='tensor_b_l1')
            tensor_b_l0b = tvm.compute(
                (kn_shape, n_shape, block_out, block_reduce),
                lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                name='tensor_b_l0b')
        else:
            if is_nd_int82fp32:
                is_b_align = check_shape_align(tensor_b.shape, 32)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(tensor_b,
                                                      gm_b_shape_normalize,
                                                      'b',
                                                      'int8')
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b[i, j],
                        name='tensor_b_normalize_ub')
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda *indices: operate_shape.cast(
                        tensor_b_normalize_ub(*indices), "float16"),
                    name="tensor_b_float16_normalize_ub",
                )
            else:
                is_b_align = check_shape_align(tensor_b.shape, 16)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(tensor_b,
                                                      gm_b_shape_normalize,
                                                      'b',
                                                      in_b_dtype)
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b[i, j],
                        name='tensor_b_normalize_ub'
                    )

            tensor_b_fract_shape = (
                kn_shape, n_shape * block_out, block_reduce)
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k: tensor_b_normalize_ub[i * block_reduce + k, j],
                name='b_fract')
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices: tensor_b_fract(*indices),  # pylint: disable=W0108
                name='tensor_b_l1')
            tensor_b_l0b = tvm.compute(
                (kn_shape, n_shape, block_out, block_reduce),
                lambda i, j, k, l: tensor_b_l1[i, j * block_reduce + k, l],
                name='tensor_b_l0b')
        return tensor_b_l0b

    def _b_part_not_trans():
        if is_fractal_b:
            if nz_b:
                tensor_b_l1 = tvm.compute(
                    tensor_b.shape,
                    lambda *indices: tensor_b(  # pylint: disable=W0108
                        *indices),
                    name='tensor_b_l1')
                tensor_b_l0b = tvm.compute(
                    tensor_b.shape,
                    lambda *indices: tensor_b_l1(  # pylint: disable=W0108
                        *indices),
                    name='tensor_b_l0b')
            else:
                if in_b_dtype == "int8" and dst_dtype == "float32":
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b(  # pylint: disable=W0108
                            *indices),
                        name="tensor_b_ub",
                    )
                    tensor_float16_b_ub = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: operate_shape.cast(
                            tensor_b_ub(*indices), "float16"),
                        name="tensor_float16_b_ub",
                    )
                    new_b_shape = [
                        tensor_b.shape[0] * 2,
                        tensor_b.shape[1],
                        tensor_b.shape[2],
                        tensor_b.shape[3] // 2,
                    ]
                    tensor_zn_b_ub = tvm.compute(
                        new_b_shape,
                        lambda i, j, k, l: tensor_float16_b_ub[i // 2, j, k, (
                            i * 16 + l) % 32],
                        name="tensor_zn_b_ub",
                    )
                    tensor_b_l1 = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_zn_b_ub(  # pylint: disable=W0108
                            *indices),
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        new_b_shape,
                        lambda *indices: tensor_b_l1(  # pylint: disable=W0108
                            *indices),
                        name='tensor_b_l0b')
                else:
                    tensor_b_l1 = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b(  # pylint: disable=W0108
                            *indices),
                        name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        tensor_b.shape,
                        lambda *indices: tensor_b_l1(  # pylint: disable=W0108
                            *indices),
                        name='tensor_b_l0b')
        else:
            tensor_b_l0b = _b_nd_part_not_trans()
        return tensor_b_l0b

    def _nd_part_trans():
        if in_b_dtype == "float16":
            if is_nd_int82fp32:
                is_b_align = check_shape_align(tensor_b.shape, 32)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(tensor_b,
                                                      gm_b_shape_normalize,
                                                      'b',
                                                      'int8')
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b[i, j],
                        name='tensor_b_normalize_ub')
                if not trans_a and trans_b:
                    transpose_shape = gm_b_shape_normalize[::-1]
                    tensor_b_transpose_ub = tvm.compute(
                        transpose_shape,
                        lambda i, j: tensor_b_normalize_ub[j, i],
                        name="b_transpose_only",
                    )
                    tensor_b_transpose_zero_ub = tvm.compute(
                        transpose_shape,
                        lambda i, j: tvm.select(
                            i < kn_shape_ori,
                            tensor_b_transpose_ub[i, j],
                            tvm.const(0).astype("int8"),
                        ),
                        name="b_transpose_zero",
                    )
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b_transpose_zero_ub[j, i],
                        name="b_after_process",
                    )
                tensor_b_normalize_ub = tvm.compute(
                    gm_b_shape_normalize,
                    lambda *indices: operate_shape.cast(
                        tensor_b_normalize_ub(*indices), "float16"),
                    name="tensor_b_float16_normalize_ub",
                )
            else:
                is_b_align = check_shape_align(tensor_b.shape, 16)
                if not is_b_align:
                    tensor_b_normalize_ub = _do_align(tensor_b,
                                                      gm_b_shape_normalize,
                                                      'b',
                                                      in_b_dtype)
                else:
                    tensor_b_normalize_ub = tvm.compute(
                        gm_b_shape_normalize,
                        lambda i, j: tensor_b[i, j],
                        name='tensor_b_normalize_ub'
                    )
            tensor_b_fract_shape = (
                n_shape, kn_shape * block_reduce, block_out)
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k: tensor_b_normalize_ub[i * block_out + k, j],
                name='b_fract')
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_b_fract(*indices),
                name='tensor_b_l1')
            tensor_b_l0b = tvm.compute(
                (kn_shape, n_shape, block_out, block_reduce),
                lambda i, j, k, l: tensor_b_l1[j, i * block_reduce + l, k],
                name='tensor_b_l0b',
                attrs={"transpose_b": "true"},
            )
        else:
            is_b_align = check_shape_align(tensor_b.shape, 32)
            tensor_b_ub_shape = (n_shape * block_out, kn_shape * block_reduce)
            tensor_b_fract_shape = (
                kn_shape,
                n_shape,
                block_out,
                block_reduce,
            )
            if not is_b_align:
                tensor_b_normalize_ub = _do_align(tensor_b,
                                                    gm_b_shape_normalize,
                                                    'b',
                                                    in_b_dtype)
            else:
                tensor_b_normalize_ub = tvm.compute(
                    tensor_b_ub_shape,
                    lambda *indices:  # pylint: disable=W0108
                    tensor_b(*indices),
                    name="tensor_b_normalize_ub",
                )
            if not trans_a and trans_b:
                transpose_shape = tensor_b_ub_shape[::-1]
                tensor_b_transpose_ub = tvm.compute(
                    transpose_shape,
                    lambda i, j: tensor_b_normalize_ub[j, i],
                    name="b_transpose_only",
                )
                tensor_b_transpose_zero_ub = tvm.compute(
                    transpose_shape,
                    lambda i, j: tvm.select(
                        i < kn_shape_ori,
                        tensor_b_transpose_ub[i, j],
                        tvm.const(0).astype(in_b_dtype),
                    ),
                    name="b_transpose_zero",
                )
                tensor_b_normalize_ub = tvm.compute(
                    tensor_b_ub_shape,
                    lambda i, j: tensor_b_transpose_zero_ub[j, i],
                    name="b_after_process",
                )
            tensor_b_fract = tvm.compute(
                tensor_b_fract_shape,
                lambda i, j, k, l: tensor_b_normalize_ub[j * block_out + k, i *
                                                         block_reduce + l],
                name='b_fract')
            tensor_b_l1 = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_b_fract(*indices),
                name='tensor_b_l1')
            tensor_b_l0b = tvm.compute(
                tensor_b_fract_shape,
                lambda *indices:  # pylint: disable=W0108
                tensor_b_l1(*indices),
                name='tensor_b_l0b',
                attrs={"transpose_b": "true"},
            )
        return tensor_b_l0b

    def _compute_b_matrix():  # pylint: disable=too-many-branches
        if not trans_b:
            tensor_b_l0b = _b_part_not_trans()
        else:

            def _part_trans():
                if is_fractal_b:
                    if nz_b:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape,
                            lambda *indices: tensor_b(  # pylint: disable=W0108
                                *indices),
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape,
                            lambda *indices: tensor_b(  # pylint: disable=W0108
                                *indices),
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            (kn_shape, n_shape, block_out, block_reduce),
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')
                else:
                    tensor_b_l0b = _nd_part_trans()
                return tensor_b_l0b

            tensor_b_l0b = _part_trans()

        return tensor_b_l0b

    tensor_b_l0b = _compute_b_matrix()

    def _compute_c_martix():
        if block_in != cce_params.BLOCK_VECTOR:  # gemm
            # define mad compute
            tensor_c = tvm.compute(
                out_shape,
                lambda nb, mb, mp, np: tvm.sum(
                    (tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] * tensor_b_l0b[
                        reduce_kb, nb, np, reduce_kp]).astype(out_dtype),
                    axis=[reduce_kb, reduce_kp]),
                name='tensor_c',
                attrs={'input_order': 'positive'})
            tensor_c_ub = _get_tensor_c_ub(tensor_c, out_shape, tensor_bias,
                                           tensor_alpha_ub, l0c_support_fp32,
                                           tensor_beta_bias_ub, dst_dtype,
                                           is_fractal_a)

            if is_fractal_a and is_fractal_b:
                tensor_c_gm = tvm.compute(
                    out_shape,
                    lambda *indices: tensor_c_ub(  # pylint: disable=W0108
                        *indices),
                    name='tensor_c_gm',
                    tag='gemm',
                    attrs={'kernel_name': kernel_name})
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda i, j: tvm.select(
                            i < m_shape_ori,
                            tvm.select(j < n_shape_ori, tensor_c_ub[i, j])),
                        name='tensor_c_gm',
                        tag='gemm',
                        attrs={'kernel_name': kernel_name})
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda i, j: tensor_c_ub[j // 16, i // 16, i % 16, j %
                                                 16],
                        name='tensor_c_gm',
                        tag='gemm',
                        attrs={'kernel_name': kernel_name})
        return tensor_c_gm

    tensor_c_gm = _compute_c_martix()
    return tensor_c_gm


def _get_tensor_c_ub(  # pylint: disable=too-many-arguments
        tensor_c, out_shape, tensor_bias, tensor_alpha_ub, l0c_support_fp32,
        tensor_beta_bias_ub, dst_dtype, is_fractal_a):
    """calculate tensor_c_ub"""
    tensor_c_before_mul_ub = tvm.compute(
        out_shape,
        lambda *indices: tensor_c(*indices),  # pylint: disable=W0108
        name='tensor_c_before_mul_ub',
    )
    temp = tensor_c_before_mul_ub
    if temp.dtype == "int32" and dst_dtype == "float32":
        tensor_c_float16_before_mul_ub = tvm.compute(
            out_shape,
            lambda *indices: operate_shape.cast(
                tensor_c_before_mul_ub(*indices), dtype="float16"),
            name="tensor_c_float16_before_mul_ub",
        )
        tensor_c_float32_before_mul_ub = tvm.compute(
            out_shape,
            lambda *indices: operate_shape.cast(
                tensor_c_float16_before_mul_ub(*indices), dtype="float32"),
            name="tensor_c_float32_before_mul_ub",
        )
        temp = tensor_c_float32_before_mul_ub

    if tensor_bias is not None:
        tensor_alpha_c_ub = tvm.compute(
            out_shape,
            lambda *indices: temp(*indices) * tensor_alpha_ub[0],
            name='tensor_alpha_c_ub',
        )
        if not is_fractal_a:
            tensor_c_ub_temp = tvm.compute(
                tensor_beta_bias_ub.shape,
                lambda i, j: tensor_beta_bias_ub[i, j] + tensor_alpha_c_ub[
                    j // 16, i // 16, i % 16, j % 16],
                name='tensor_c_ub_temp',
            )
        else:
            tensor_c_ub_temp = tvm.compute(
                out_shape,
                lambda *indices: tensor_alpha_c_ub(*indices) +
                tensor_beta_bias_ub(*indices),
                name='tensor_c_ub_temp',
            )
    else:
        tensor_c_ub_temp = tvm.compute(
            out_shape,
            lambda *indices: temp(*indices) * tensor_alpha_ub[0],
            name='tensor_c_ub_temp',
        )
    if dst_dtype == 'float16' and l0c_support_fp32:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: operate_shape.cast(
                tensor_c_ub_temp(*indices),
                dtype='float16',
            ),
            name='tensor_c_ub',
        )
    elif dst_dtype == 'float32' and l0c_support_fp32 \
            and not is_fractal_a:
        tensor_c_ub = tensor_c_ub_temp
    elif dst_dtype == 'int32' and not is_fractal_a:
        tensor_c_ub = tensor_c_ub_temp
    else:
        tensor_c_ub = tvm.compute(
            tensor_c_ub_temp.shape,
            lambda *indices: tensor_c_ub_temp(  # pylint: disable=W0108
                *indices),
            name='tensor_c_ub',
        )
    return tensor_c_ub
