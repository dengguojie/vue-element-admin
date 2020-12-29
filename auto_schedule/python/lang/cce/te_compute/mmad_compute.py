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
mmad_compute
"""
# pylint: disable=too-many-lines
from __future__ import absolute_import

from functools import reduce as functools_reduce
from functools import wraps

import te.platform.cce_params as cce
import te.platform.cce_conf as cce_conf
from te.utils.error_manager import error_manager_util
from te.utils.shape_util import shape_to_list
from te.utils import para_check

# pylint: disable=import-error, ungrouped-imports
import topi
from te import tvm
from .util import check_input_tensor_shape
try:
    from te.tvm.dsl_source_info import source_info_decorator
except ImportError:
    def source_info_decorator(depth=1):
        def get_source_info_decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                f_return = func(*args, **kwargs)
                return f_return

            return wrapper

        return get_source_info_decorator


def _elecnt_of_shape(shape):
    """
    calculate reduce shape
    """
    return functools_reduce(lambda x, y: x*y, shape)


# shape limit for matmul
# int32's max value
SHAPE_SIZE_LIMIT = 2**31 - 1
FORMAT_DYNAMIC_THRESHOLD = 1 * 1


def _shape_check(tensor_a, tensor_b,  # pylint: disable=C0301, R0912, R0913, R0914, R0915
                tensor_bias, trans_a, trans_b, format_a, format_b, dst_dtype,
                nz_a):
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

    if format_a not in ("ND", "fractal", "FRACTAL_Z"):
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'tensor_a',
            'expected_format_list': ("ND", "fractal", "FRACTAL_Z"),
            'format': format_a
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if format_b not in ("ND", "fractal", "FRACTAL_Z"):
        dict_args = {
            'errCode': 'E60008',
            'param_name': 'tensor_b',
            'expected_format_list': ("ND", "fractal", "FRACTAL_Z"),
            'format': format_b
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"

    # fractal and ND not support
    if is_fractal_a and not is_fractal_b:
        dict_args = {
            'errCode': 'E61001',
            'reason': 'not support A is fractal and B is ND!'
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    # ND and ND only support 'float16'
    if not is_fractal_a and not is_fractal_b:
        if in_a_dtype != "float16" or in_b_dtype != "float16":
            dict_args = {
                'errCode': 'E61001',
                'reason': "only support 'float16' input datatype for 'ND' and 'ND' format!"
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    # ND and fractal support 'float16' and 'b8'
    else:
        if not (in_a_dtype == "float16" and in_b_dtype == "float16") and \
                not (in_a_dtype in ("uint8", "int8") and (in_b_dtype == "int8")):
            dict_args = {
                'errCode': 'E61001',
                'reason': "only support float16 & float16 and uint8/int8 & int8 intput data type."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if (in_a_dtype in ("uint8", "int8")) and (in_b_dtype == "int8"):
        if not is_fractal_a and is_fractal_b:
            if trans_a:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "Not support A transpose for u8/s8 input and 'ND' & 'fractal'."
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    def _check_shape_len(is_fractal_a, shape_len_a, is_fractal_b, shape_len_b):
        if (is_fractal_a == is_fractal_b) and \
                (shape_len_b not in [shape_len_a - 1, shape_len_a]):
            dict_args = {
                'errCode': 'E61001',
                'reason': "The length of B shape shoud be equal to {} or {}, but now it's " \
                    "equal to {}".format(shape_len_a - 1, shape_len_a, shape_len_b)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if is_fractal_a:
            if shape_len_a not in (4, 5):
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal input data, only support tensor's dim is 4 or 5!"
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        else:
            if shape_len_a not in (2, 3):
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for nd input data, only support tensor's dim is 2 or 3!"
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if is_fractal_b:
            if shape_len_b not in (4, 5):
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal input data, only support tensor's dim is 4 or 5!"
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        else:
            if shape_len_b not in (2, 3):
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for nd input data, only support tensor's dim is 2 or 3!"
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    _check_shape_len(is_fractal_a, shape_len_a, is_fractal_b, shape_len_b)

    if is_fractal_b:
        if shape_len_b not in (4, 5):
            dict_args = {
                'errCode': 'E61001',
                'reason': "for fractal input data, only support tensor's dim is 4 or 5!"
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    else:
        if shape_len_b not in (2, 3):
            dict_args = {
                'errCode': 'E61001',
                'reason': "for nd input data, only support tensor's dim is 2 or 3!"
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    def _check_batch_size():
        if shape_len_a == shape_len_b and \
                tensor_a.shape[0].value != tensor_b.shape[0].value:
            return False
        return True

    batch = None
    if shape_len_a in (3, 5):
        if not _check_batch_size():
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape's batch size",
                "param1_name": "tensor_a",
                "param1_value": tensor_a.shape[0].value,
                "param2_name": "tensor_b",
                "param2_value": tensor_b.shape[0].value,
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        batch = tensor_a.shape[0].value

    if tensor_bias is not None:
        shape_bias = [i.value for i in tensor_bias.shape]

    k_block_size = cce.BLOCK_REDUCE
    if (in_a_dtype in ("uint8", "int8")) and in_b_dtype == "int8":
        k_block_size = cce.BLOCK_REDUCE_INT8

    def _check_dst_dtype(dst_dtype):
        dst_dtype_check_list = ["float16", "float32", "int32", "int8"]
        if dst_dtype not in dst_dtype_check_list:
            args_dict = {
                "errCode": "E60000",
                "param_name": "dst_dtype",
                "expected_value": "dst_dtype_check_list",
                "input_value": dst_dtype
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    _check_dst_dtype(dst_dtype)

    is_gemv = False
    is_vector_a = False
    a_block_in = 1
    if not is_fractal_a:
        # shape_len_a is 2 or 3
        if trans_a:
            m_shape = shape_a[shape_len_a - 1]
            km_shape = shape_a[shape_len_a - 2]
            # non 16 multi result in buffer not align while transport
            if m_shape != cce.BLOCK_VECTOR and m_shape % cce.BLOCK_IN != 0:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for ND input, shape_m must be {} or {} " \
                        "multi when A transport".format(cce.BLOCK_VECTOR, cce.BLOCK_IN)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        else:
            m_shape = shape_a[shape_len_a - 2]
            km_shape = shape_a[shape_len_a - 1]
        real_shape_m = m_shape
    else:
        if trans_a:
            m_shape = shape_a[shape_len_a - 3]
            km_shape = shape_a[shape_len_a - 4]
            if nz_a:
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
            else:
                a_block_reduce = shape_a[shape_len_a - 2]
                a_block_in = shape_a[shape_len_a - 1]
        else:
            m_shape = shape_a[shape_len_a - 4]
            km_shape = shape_a[shape_len_a - 3]
            if nz_a:
                a_block_reduce = shape_a[shape_len_a - 2]
                a_block_in = shape_a[shape_len_a - 1]
            else:
                a_block_reduce = shape_a[shape_len_a - 1]
                a_block_in = shape_a[shape_len_a - 2]
        real_shape_m = m_shape*a_block_in

        if a_block_reduce != k_block_size:
            dict_args = {
                'errCode': 'E61001',
                'reason': "for fractal input,tensor_a's shape last 2 dim must be {} or {}.".format(
                    cce.BLOCK_IN, cce.BLOCK_VECTOR)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if a_block_in not in (cce.BLOCK_VECTOR, cce.BLOCK_IN):
            dict_args = {
                'errCode': 'E61001',
                'reason': "for fractal input,tensor_a's shape last 2 dim must be {} or {}.".format(
                    cce.BLOCK_IN, cce.BLOCK_VECTOR)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if a_block_in == cce.BLOCK_VECTOR:
            is_vector_a = True
            if m_shape != cce.BLOCK_VECTOR:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal input,tensor_a's shape last 2 dim must be {} or {}.".format(
                        cce.BLOCK_IN, cce.BLOCK_VECTOR)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            if km_shape % (cce.BLOCK_IN) != 0:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal gevm input,K should be multiple of {}.".format(cce.BLOCK_IN * k_block_size)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    b_block_reduce = 1
    b_block_out = 1
    if not is_fractal_b:
        # shape_len_b is 2 or 3
        if trans_b:
            kn_shape = shape_b[shape_len_b - 1]
            n_shape = shape_b[shape_len_b - 2]
        else:
            kn_shape = shape_b[shape_len_b - 2]
            n_shape = shape_b[shape_len_b - 1]

        if n_shape != 1 and n_shape % cce.BLOCK_IN != 0:
            dict_args = {
                'errCode': 'E61001',
                'reason': "input shape N should be multiple of {} or 1.".format(cce.BLOCK_IN)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

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
            dict_args = {
                'errCode': 'E61001',
                'reason': "for fractal input,tensor_b's shape last 2 dim must be {}.".format(k_block_size)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if b_block_out not in (cce.BLOCK_VECTOR, cce.BLOCK_IN):
            raise RuntimeError(
                "for fractal input,tensor_b's shape last 2 dim must be %d or %d"
                % (cce.BLOCK_IN, cce.BLOCK_VECTOR))
        if b_block_out == cce.BLOCK_VECTOR:
            is_gemv = True
            if is_vector_a:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "input shape M and N can't both be 1."
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            if n_shape != 1:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal input,tensor_a's shape last 2 dim must be {} or {}.".format(
                        cce.BLOCK_IN, cce.BLOCK_VECTOR)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            if kn_shape % (cce.BLOCK_IN) != 0:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "for fractal gemv input,K should be multiple of {}.".format(
                        cce.BLOCK_IN*k_block_size)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            # gemv u8/s8 is transed to gevm(s8/u8), s8/u8 is not support for mad intri
            if in_a_dtype == "uint8" and in_b_dtype == "int8":
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "b8 gemv only support int8 & int8, current type is {} and {}.".format(
                        in_a_dtype, in_b_dtype)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if is_fractal_a == is_fractal_b:
        if km_shape != kn_shape:
            dict_args = {
                'errCode': 'E61001',
                'reason': "reduce axis not same."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    elif not is_fractal_a and is_fractal_b:
        if km_shape != (kn_shape*b_block_reduce):
            dict_args = {
                'errCode': 'E61001',
                'reason': "Km shape should be equal whit kn*block."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if not is_fractal_a and not is_fractal_b:
        if m_shape == 1 and n_shape == 1:
            dict_args = {
                'errCode': 'E61001',
                'reason': "input shape M and N can't both be 1."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        if n_shape == 1:
            if kn_shape % (cce.BLOCK_IN*cce.BLOCK_IN) != 0:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "input shape K should be multiple of {}.".format(cce.BLOCK_IN * cce.BLOCK_IN)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        elif km_shape % k_block_size != 0:
            dict_args = {
                'errCode': 'E61001',
                'reason': "input shape K should be multiple of {}.".format(cce.BLOCK_IN)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        is_gemv = n_shape == 1

    if in_b_dtype == "int8":
        if is_gemv:
            if trans_a:
                # Load2D intri has error from L1 to L0B transport for b8
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "Not support A transpose for gemv b8 input."
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        else:
            if trans_b:
                # Load2D intri has error from L1 to L0B transport for b8
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "Not support B transpose for gevm or gemm b8 input."
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    # 2d case
    if shape_bias:
        if is_gemv:
            if len(shape_bias) == 1:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "bias shape for gemv must be [m,1] or [1,m,1] or [b,m,1]," \
                        " current is {}.".format(shape_bias)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            if len(shape_bias) == 2:
                if shape_bias != [real_shape_m, 1]:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "bias shape for gemv must be [m,1] or [1,m,1] or [b,m,1]," \
                            " current is {}.".format(shape_bias)
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            elif len(shape_bias) == 3:
                if batch is None:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "tensor A and tensor B lack of batch while bias has batch"
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
                if shape_bias not in ([1, real_shape_m, 1], [batch, real_shape_m, 1]):
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "bias shape for gemv must be [m,1] or [1,m,1] or [b,m,1]," \
                            " current is {}.".format(shape_bias)
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            else:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "bias shape for gemv must be [m,1] or [1,m,1] or [b,m,1]," \
                        " current is {}.".format(shape_bias)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        else:
            if len(shape_bias) == 1:
                if shape_bias[0] != n_shape*b_block_out:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "broadcast bias shape must be equal to shape n"
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            elif len(shape_bias) == 2:
                if shape_bias not in ([1, n_shape*b_block_out], ):
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "bias shape must be [1,n], current is {}".format(shape_bias)
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            elif len(shape_bias) == 3:
                if batch is None:
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "tensor A and tensor B lack of batch while bias has batch"
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
                if shape_bias not in ([1, 1, n_shape*b_block_out], [batch, 1, n_shape*b_block_out]):
                    dict_args = {
                        'errCode': 'E61001',
                        'reason': "bias shape must be [n,] or [1,n] or [1,1,n] or [b,1,n]" \
                            " for gevm and gemm current is {}.".format(shape_bias)
                    }
                    raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
            else:
                dict_args = {
                    'errCode': 'E61001',
                    'reason': "bias shape must be [n,] or [1,n] or [1,1,n] or [b,1,n]" \
                        " for gevm and gemm current is {}.".format(shape_bias)
                }
                raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_quantize_params(quantize_params=None):  # pylint: disable=R0912
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


def _get_quantize_params(quantize_params=None, out_type="float16"):
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


def _get_compress_tensor_compute(tensor_src, comp_index, op_name):
    """
    get compress tensor compute
    """
    _, _, _, compress_mode = cce_conf.get_soc_spec("UNZIP")
    comp_size = 8 if compress_mode == 1 else 2

    tile_k_value = tvm.var("tile_L1_k", dtype="int32")
    tile_n_value = tvm.var("tile_L1_n", dtype="int32")

    shape_src = tensor_src.shape
    block_n_num = (shape_src[-3] + tile_n_value - 1) // tile_n_value
    block_k_num = (shape_src[-4] + tile_k_value - 1) // tile_k_value

    # tile_mode is 1 when tile_n < dim_n, or tile_mode is 0
    if len(shape_src) == 4:
        tensor = tvm.compute(shape_src, lambda i, j, k, l: tvm.unzip(
            comp_index((j // tile_n_value * block_k_num + i // tile_k_value)
                       * comp_size),
            tensor_src(i, j, k, l)),
            name=op_name, attrs={'tile_L1_k': tile_k_value,
                                 'tile_L1_n': tile_n_value})
    elif len(shape_src) == 5:
        # get multi n block number
        batch_block_num = block_n_num * block_k_num
        tensor = tvm.compute(shape_src, lambda batch, i, j, k, l: tvm.unzip(
            comp_index(((j // tile_n_value * block_k_num + i // tile_k_value)
                        + batch * batch_block_num) * comp_size),
            tensor_src(batch, i, j, k, l)),
            name=op_name, attrs={'tile_L1_k': tile_k_value,
                                 'tile_L1_n': tile_n_value})
    return tensor


def _check_and_get_dtype_info(src_dtype_a, src_dtype_b, dst_dtype,
                             l0c_support_fp32, quantize_params):
    """
    check input dtype and output dtype,
    and return L0C dtype and dequant fusion info
    """
    if (src_dtype_a, src_dtype_b) == ("float16", "float16"):
        if dst_dtype not in ("float16", "float32"):
            dict_args = {
                'errCode': 'E60000',
                'param_name': "dst_dtype",
                "expected_value": "'float16', 'float32'",
                "input_value": str(dst_dtype)
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        out_dtype = "float32"
        if l0c_support_fp32 == 0:
            out_dtype = "float16"
    elif (src_dtype_a, src_dtype_b) == ("int8", "int8") or \
            (src_dtype_a, src_dtype_b) == ("uint8", "int8"):
        out_dtype = "int32"
    else:
        dict_args = {
            'errCode': 'E61001',
            'reason': "data type of tensor not supported"
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    if dst_dtype not in (out_dtype, "float16", "int8"):
        dict_args = {
            'errCode': 'E60000',
            'param_name': "dst_dtype",
            "expected_value": "'float16', 'int8' or same as out_dtype",
            "input_value": str(dst_dtype)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    is_fusion_mode = False
    if (src_dtype_a in ("int8", "uint8")) and (quantize_params is None):
        is_fusion_mode = True
    if (out_dtype not in (dst_dtype, "float32", "int32")) and \
            (quantize_params is None) and not is_fusion_mode:
        dict_args = {
            'errCode': 'E61001',
            'reason': "Lack of quantize parameter 'quantize_params'."
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    return out_dtype, is_fusion_mode


def _get_matmul_output_format(format_a, format_out):
    """
    get matmul output format

    Parameters:
        format_a: tensor a format
        out_format: output format, default param is None

    Returns:
        format: matmul output format
    """
    default_format = "FRACTAL_NZ"
    if format_out is not None:
        return format_out
    if format_a == "ND":
        return "NC1HWC0"
    return default_format


@source_info_decorator()
@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, bool, bool, str, str, float, float, str,
                             (type(None), tvm.tensor.Tensor), (type(None), dict), (type(None), str),
                             (type(None), tvm.tensor.Tensor), (type(None), dict), str)
def matmul(tensor_a,  # pylint: disable=W0108, R1702, R0912, R0913, R0914, R0915
           tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
           alpha_num=1.0, beta_num=1.0, dst_dtype="float16", tensor_bias=None,
           quantize_params=None, format_out=None, compress_index=None,
           attrs={}, kernel_name="MatMul"):
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
    out_format: output format
    attrs:
        offset_x: the offset for fmap
        offset_w: the offset for w

    compress_index: index for compressed wights, None means not compress wights
    Returns None
    """
    cube_vector_split = cce_conf.get_soc_spec("CUBE_VECTOR_SPLIT")
    if cube_vector_split:
        result = _matmul_cv_split(tensor_a=tensor_a,
                                  tensor_b=tensor_b,
                                  trans_a=trans_a,
                                  trans_b=trans_b,
                                  format_a=format_a,
                                  format_b=format_b,
                                  dst_dtype=dst_dtype,
                                  tensor_bias=tensor_bias,
                                  format_out=format_out,
                                  kernel_name=kernel_name)
    else:
        result = _matmul_compute(tensor_a=tensor_a,
                                tensor_b=tensor_b,
                                trans_a=trans_a,
                                trans_b=trans_b,
                                format_a=format_a,
                                format_b=format_b,
                                alpha_num=alpha_num,
                                beta_num=beta_num,
                                dst_dtype=dst_dtype,
                                tensor_bias=tensor_bias,
                                quantize_params=quantize_params,
                                format_out=format_out,
                                compress_index=compress_index,
                                attrs=attrs)

    return result


def _matmul_compute( # pylint: disable=W0108, R1702, R0912, R0913, R0914, R0915
        tensor_a,
        tensor_b,
        trans_a=False,
        trans_b=False,
        format_a="ND",
        format_b="ND",
        alpha_num=1.0,
        beta_num=1.0,
        dst_dtype="float16",
        tensor_bias=None,
        quantize_params=None,
        format_out=None,
        compress_index=None,
        attrs={}):
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
    out_format: output format
    attrs:
        offset_x: the offset for fmap
        offset_w: the offset for w

    compress_index: index for compressed wights, None means not compress wights
    Returns None
    """

    nz_a = False
    if format_a == "FRACTAL_NZ":
        nz_a = True
        format_a = "fractal"

    nz_b = False
    if format_b == "FRACTAL_NZ":
        nz_b = True
        format_b = "fractal"

    _shape_check(tensor_a, tensor_b, tensor_bias, trans_a, trans_b, format_a,
                format_b, dst_dtype, nz_a)

    format_out = _get_matmul_output_format(format_a, format_out)

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    tensor_b.op.attrs['trans_b'] = trans_b

    l0c_support_fp32 = 1
    support_type = cce_conf.getValue("Intrinsic_mmad")
    if "f162f32" not in support_type:
        l0c_support_fp32 = 0
    # used for inner_product and ascend_dequant UB fusion
    out_dtype, is_fusion_mode = _check_and_get_dtype_info(
        in_a_dtype, in_b_dtype, dst_dtype, l0c_support_fp32, quantize_params)

    def _check_quant_param_valid():
        if (out_dtype == dst_dtype) and (quantize_params is not None):
            dict_args = {
                'errCode': 'E61001',
                'reason': "When out_dtype = dst_dtype, quantize_params should be None."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

        if (quantize_params is not None) and (not isinstance(quantize_params, dict)):
            dict_args = {
                'errCode': 'E61001',
                'reason': "'quantize_params' should be dict type."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    _check_quant_param_valid()

    tensor_a_length = len(tensor_a.shape)
    tensor_b_length = len(tensor_b.shape)

    is_fractal_a = format_a != "ND"
    is_fractal_b = format_b != "ND"
    is_fractal_out = is_fractal_a and is_fractal_b
    is_fractal_out = is_fractal_out or format_out == "FRACTAL_NZ"
    is_frac_nz_out = format_out == "FRACTAL_NZ"

    if tensor_bias is not None:
        if quantize_params is None and not is_fusion_mode:
            if tensor_bias.dtype != dst_dtype:
                args_dict = {
                    "errCode": "E60000",
                    "param_name": "dtype of tensor_bias",
                    "expected_value": dst_dtype,
                    "input_value": tensor_bias.dtype
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        else:
            if tensor_bias.dtype != out_dtype:
                args_dict = {
                    "errCode": "E60000",
                    "param_name": "dtype of tensor_bias",
                    "expected_value": out_dtype,
                    "input_value": tensor_bias.dtype
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

        bias_shape = []
        if _elecnt_of_shape(tensor_bias.shape).value == 1:
            bias_shape = [1]
        else:
            for i in tensor_bias.shape:
                if bias_shape:
                    bias_shape.append(i.value)
                elif i.value != 0 and i.value != 1:
                    # first element value should be > 1
                    bias_shape.append(i.value)

    def _get_block_reduce():
        if in_a_dtype == "float16":
            return cce.BLOCK_REDUCE
        return cce.BLOCK_REDUCE_INT8

    block_reduce = _get_block_reduce()

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT

    gm_a_shape_normalize = []
    if trans_a:
        if is_fractal_a:
            m_shape = tensor_a.shape[tensor_a_length - 3].value
            m_shape_ori = m_shape
            km_shape = tensor_a.shape[tensor_a_length - 4].value

            def _get_vm_shape():
                if nz_a:
                    return tensor_a.shape[tensor_a_length - 2].value
                return tensor_a.shape[tensor_a_length - 1].value

            vm_shape = _get_vm_shape()

            gm_a_shape_normalize = tensor_a.shape
        else:
            m_shape = (tensor_a.shape[
                tensor_a_length - 1].value + block_in - 1) // block_in
            m_shape_ori = tensor_a.shape[tensor_a_length - 1].value
            km_shape = tensor_a.shape[tensor_a_length - 2].value // block_reduce
            vm_shape = 16
            if tensor_a.shape[tensor_a_length - 1].value == 1:
                m_shape = 1
                vm_shape = 1
            if tensor_a_length in (3, 5):
                gm_a_shape_normalize.append(tensor_a.shape[0])
            gm_a_shape_normalize.append(km_shape*block_reduce)
            gm_a_shape_normalize.append(m_shape*vm_shape)

    else:
        if is_fractal_a:
            m_shape = tensor_a.shape[tensor_a_length - 4].value
            m_shape_ori = m_shape
            km_shape = tensor_a.shape[tensor_a_length - 3].value

            def _get_vm_shape():
                if nz_a:
                    return tensor_a.shape[tensor_a_length - 1].value
                return tensor_a.shape[tensor_a_length - 2].value

            vm_shape = _get_vm_shape()

            gm_a_shape_normalize = tensor_a.shape
        else:
            m_shape = (tensor_a.shape[
                tensor_a_length - 2].value + block_in - 1) // block_in
            m_shape_ori = tensor_a.shape[tensor_a_length - 2].value
            km_shape = tensor_a.shape[tensor_a_length - 1].value // block_reduce
            vm_shape = 16
            if tensor_a.shape[tensor_a_length - 2].value == 1:
                m_shape = 1
                vm_shape = 1

            if tensor_a_length in (3, 5):
                gm_a_shape_normalize.append(tensor_a.shape[0])
            gm_a_shape_normalize.append(m_shape*vm_shape)
            gm_a_shape_normalize.append(km_shape*block_reduce)

    if trans_b:
        if is_fractal_b:
            kn_shape = tensor_b.shape[tensor_b_length - 3].value
            n_shape = tensor_b.shape[tensor_b_length - 4].value
            vn_shape = tensor_b.shape[tensor_b_length - 1].value
        else:
            kn_shape = tensor_b.shape[tensor_b_length - 1].value // block_reduce
            n_shape = tensor_b.shape[tensor_b_length - 2].value // block_out
            vn_shape = 16
            if tensor_b.shape[tensor_b_length - 2].value == 1:
                n_shape = 1
                vn_shape = 1
    else:
        if is_fractal_b:
            kn_shape = tensor_b.shape[tensor_b_length - 4].value
            n_shape = tensor_b.shape[tensor_b_length - 3].value
            vn_shape = tensor_b.shape[tensor_b_length - 2].value
        else:
            kn_shape = tensor_b.shape[tensor_b_length - 2].value // block_reduce
            n_shape = tensor_b.shape[tensor_b_length - 1].value // block_out
            vn_shape = 16
            if tensor_b.shape[tensor_b_length - 1].value == 1:
                n_shape = 1
                vn_shape = 1

    def _gevm_block_in_value(
            is_fractal, m_shape, vm_shape, km_shape, n_shape_ori, block_in_val):
        """
        calculate k!= block_in*block_reduce gevm block_in
        """
        block_in = block_in_val
        block_in_ori = block_in_val
        m_16_shapes = [(25088, 4096), (18432, 4096), (4096, 4096),
                       (9216, 4096), (4096, 1008)]

        if m_shape == 1 and vm_shape == 1:
            block_in_ori = cce.BLOCK_VECTOR
            if km_shape % block_in == 0:
                block_in = cce.BLOCK_VECTOR
                if not is_fractal and \
                        (km_shape*block_reduce, n_shape_ori) in m_16_shapes:
                    block_in = block_in_val
        return block_in, block_in_ori

    block_in, block_in_ori = _gevm_block_in_value(
        is_fractal_a, m_shape, vm_shape, km_shape, n_shape*vn_shape, block_in)

    if n_shape == 1 and vn_shape == 1:
        block_out = cce.BLOCK_VECTOR

    def _check_reduce_shape():
        if km_shape != kn_shape:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape k",
                "param1_name": "tensor_a",
                "param1_value": str(km_shape),
                "param2_name": "tensor_b",
                "param2_value": str(kn_shape),
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    _check_reduce_shape()

    def _check_alpha_beta_numb():
        if alpha_num != 1.0 or beta_num != 1.0:
            dict_args = {
                'errCode': 'E61001',
                'reason': "mmad now only supprot alpha_num = {0} and beta_num = {0}.".format(1.0)
            }
            raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    _check_alpha_beta_numb()

    def _check_fractal_a_value():
        length = tensor_a_length
        if trans_a:
            if nz_a:
                if not (tensor_a.shape[length - 1].value == block_reduce
                        and tensor_a.shape[length - 2].value == block_in):
                    args_dict = {
                        "errCode": "E60101",
                        "param_name": "tensor_a",
                        "expected_two_dims": "{}".format([block_in, block_reduce]),
                        "actual_two_dim": "{}".format(tensor_a.shape[-2:]),
                    }
                    raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
            else:
                if not (tensor_a.shape[length - 2].value == block_reduce
                        and tensor_a.shape[length - 1].value == block_in):
                    args_dict = {
                        "errCode": "E60101",
                        "param_name": "tensor_a",
                        "expected_two_dims": "{}".format([block_reduce, block_in]),
                        "actual_two_dim": "{}".format(tensor_a.shape[-2:]),
                    }
                    raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        else:
            if nz_a:
                if not (tensor_a.shape[length - 1].value == block_in
                        and tensor_a.shape[length - 2].value == block_reduce):
                    args_dict = {
                        "errCode": "E60101",
                        "param_name": "tensor_a",
                        "expected_two_dims": "{}".format([block_reduce, block_in]),
                        "actual_two_dim": "{}".format(tensor_a.shape[-2:]),
                    }
                    raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
            else:
                if not (tensor_a.shape[length - 2].value == block_in
                        and tensor_a.shape[length - 1].value == block_reduce):
                    args_dict = {
                        "errCode": "E60101",
                        "param_name": "tensor_a",
                        "expected_two_dims": "{}".format([block_in, block_reduce]),
                        "actual_two_dim": "{}".format(tensor_a.shape[-2:]),
                    }
                    raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    if is_fractal_a:
        _check_fractal_a_value()

    def _check_fractal_b_value():
        if trans_b:
            if not (tensor_b.shape[tensor_b_length - 2].value == block_reduce
                    and tensor_b.shape[tensor_b_length - 1].value == block_out):
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor_b",
                    "expected_two_dims": "{}".format([block_reduce, block_out]),
                    "actual_two_dim": "{}".format(tensor_b.shape[-2:]),
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))
        else:
            if not (tensor_b.shape[tensor_b_length - 2].value == block_out
                    and tensor_b.shape[tensor_b_length - 1].value == block_reduce):
                args_dict = {
                    "errCode": "E60101",
                    "param_name": "tensor_b",
                    "expected_two_dims": "{}".format([block_out, block_reduce]),
                    "actual_two_dim": "{}".format(tensor_b.shape[-2:]),
                }
                raise RuntimeError(args_dict, error_manager_util.get_error_message(args_dict))

    if is_fractal_b:
        _check_fractal_b_value()

    def _get_offset_info(attrs_dict, dtype_a, dtype_b):
        """
        get offset info, like offset_x and offset_w
        """
        offset_x = 0
        offset_w = 0

        if cce_conf.is_v200_version_new():
            if dtype_a in ("uint8", "int8"):
                if attrs_dict.get("offset_x") is not None:
                    offset_x = attrs_dict.get("offset_x")
            if dtype_b == "int8":
                if attrs_dict.get("offset_w") is not None:
                    offset_w = attrs_dict.get("offset_w")

        return offset_x, offset_w

    offset_x, _ = _get_offset_info(attrs, in_a_dtype, in_b_dtype)
    # define reduce axis
    # kBurstAxis
    reduce_kb = tvm.reduce_axis((0, km_shape), name='kb')
    # kPointAxis
    reduce_kp = tvm.reduce_axis((0, block_reduce), name='kp')

    _check_quantize_params(quantize_params)
    scale_drq, scale_drq_tensor, sqrt_out = \
        _get_quantize_params(quantize_params, dst_dtype)

    # vadds function only support fp16 and fp32
    optmt_a = 0
    optmt_b = 0
    optmt_c = 0
    if in_a_dtype == "float16":
        optmt_a = 1
    if in_b_dtype == "float16":
        optmt_b = 1
    if dst_dtype == "float16":
        optmt_c = 1

    # not gemv
    if block_out != cce.BLOCK_VECTOR:  # pylint: disable=too-many-nested-blocks
        if tensor_a_length in (2, 4):
            l0c_shape = (
                int(n_shape), int(m_shape), int(block_in), int(block_out))
            out_shape = (
                int(n_shape), int(m_shape), int(block_in_ori), int(block_out))
            out_shape_ori = [int(m_shape_ori), int(n_shape*block_out)]
            fusion_out_shape = out_shape_ori

            if is_fractal_out:
                fusion_out_shape = out_shape
                format_out = "FRACTAL_NZ"

            if tensor_bias is not None:
                # bias only be [n,] and [1,n] for gevm and gemm
                if len(tensor_bias.shape) == 1:
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda i:
                        tensor_bias[i], name='tensor_bias_ub')
                else:
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda i: tensor_bias[0, i],
                        name='tensor_bias_ub')
            if not trans_a:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')

                    a_fract_shape = (
                        m_shape, km_shape, block_in_ori, block_reduce)
                    tensor_a_l1_shape = (
                        m_shape, km_shape, block_in, block_reduce)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            a_fract_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_ub_fract')
                        def get_a_l1_tensor(a_ub_fract, a_l1_shape):
                            if a_ub_fract.shape[-2].value == 1:
                                a_l1 = tvm.compute(
                                    a_l1_shape,lambda i, j, k, l:
                                    a_ub_fract[0, j, 0, l],
                                    name='tensor_a_l1')
                            else:
                                a_l1 = tvm.compute(
                                    a_l1_shape,lambda i, j, k, l:
                                    a_ub_fract[i, j, k, l],
                                    name='tensor_a_l1')
                            return a_l1
                        tensor_a_l1 = get_a_l1_tensor(tensor_a_ub_fract, tensor_a_l1_shape)
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        tensor_a_l1_shape,
                        lambda *indices: tensor_a_l1[indices],
                        name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (m_shape, km_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')

                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    if optmt_a == 1:
                        a_fract_shape = (
                            m_shape, km_shape, block_reduce, block_in_ori)
                        tensor_a_l1_shape = (
                            m_shape, km_shape, block_reduce, block_in)
                        tensor_a_ub_fract = tvm.compute(
                            a_fract_shape, lambda i, j, k, l:
                            tensor_a_ub[j*block_reduce + k, i*block_in + l],
                            name='tensor_a_ub_fract')
                        def get_a_l1_tensor(a_ub_fract, a_l1_shape):
                            if a_ub_fract.shape[-1].value == 1:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda i, j, k, l:
                                    a_ub_fract[0, j, k, 0],
                                    name='tensor_a_l1')
                            else:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda i, j, k, l:
                                    a_ub_fract[i, j, k, l],
                                    name='tensor_a_l1')
                            return a_l1
                        tensor_a_l1 = get_a_l1_tensor(tensor_a_ub_fract, tensor_a_l1_shape)

                        tensor_a_l0a_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            km_shape, m_shape, block_reduce, block_in)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda i, j, k, l: tensor_a_l1[j, i, l, k],
                            name='tensor_a_l0a')
            if not trans_b:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    tensor_b_l1_shape = (
                        kn_shape, n_shape, block_reduce, block_out)
                    if optmt_b == 1:
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_ub_fract')

                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_l1')

                    tensor_b_l0b = tvm.compute(
                        (kn_shape, n_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                        name='tensor_b_l0b')
            else:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        (kn_shape, n_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[j*block_out + k, i*block_reduce + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            kn_shape, n_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[j, i, k, l],
                            name='tensor_b_l0b')

            if block_in != cce.BLOCK_VECTOR:  # gemm
                # define mad compute
                tensor_c = tvm.compute(
                    l0c_shape, lambda nb, mb, mp, np: tvm.sum(
                        ((tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] - offset_x) *
                         tensor_b_l0b[reduce_kb, nb, np, reduce_kp]).astype(
                             out_dtype), axis=[reduce_kb, reduce_kp]),
                    name='tensor_c')
                if tensor_bias is None:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: (tensor_c[indices]) *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                else:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub must be [n,]
                        tensor_bias_l0c = tvm.compute(
                            l0c_shape, lambda i, j, k, l: topi.cast(
                                tensor_bias_ub[i*block_out + l],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub must be [n,]
                        tensor_bias_l0c = tvm.compute(
                            l0c_shape, lambda i, j, k, l: tensor_bias_ub[
                                i*block_out + l], name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(l0c_shape, lambda *indices:
                                                    tensor_bias_l0c[indices] +
                                                    tensor_c[indices],
                                                    name='tensor_c_add_bias')

                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: (
                                tensor_c_add_bias[indices] *
                                scale_drq_tensor[0]),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fractal_out:
                    tensor_c_gm = tvm.compute(
                        out_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul',
                        attrs={'shape': fusion_out_shape,
                               'format': format_out}
                    )
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag='matmul',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
            else:  # gevm
                orig_shape = list(out_shape)
                orig_shape[-2] = block_in
                if is_fractal_out:
                    fusion_out_shape = orig_shape
                    format_out = "FRACTAL_NZ"

                # define mad
                tensor_c = tvm.compute(
                    (n_shape, m_shape, block_out, block_out),
                    lambda nb, mb, mp, np: tvm.sum(
                        ((tensor_a_l0a[mb, reduce_kb, mp, reduce_kp] - offset_x) *
                         tensor_b_l0b[reduce_kb, nb, np, reduce_kp]).astype(
                             out_dtype), axis=[reduce_kb, reduce_kp]),
                    name='tensor_c')
                if tensor_bias is not None:
                    # tensor_bias_ub only be [n,]
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape, lambda i, j, k, l: topi.cast(
                                tensor_bias_ub[i*block_out + l],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape, lambda i, j, k, l: tensor_bias_ub[
                                i*block_out + l], name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(
                        tensor_c.shape,
                        lambda *indices: tensor_bias_l0c[indices] + tensor_c[indices],
                        name='tensor_c_add_bias')
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1 and not is_fusion_mode:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fractal_out:
                    tensor_c_gm = tvm.compute(
                        orig_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul_gevm',
                        attrs={'shape': fusion_out_shape,
                               'format': format_out})
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul_gevm',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda i, j: tensor_c_ub[
                                j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag='matmul_gevm',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})

        else:
            # have batch size
            batch_shape = tensor_a.shape[0].value

            l0c_shape = (batch_shape, n_shape, m_shape, block_in, block_out)
            out_shape = (batch_shape, n_shape, m_shape, block_in_ori, block_out)
            out_shape_ori = [int(batch_shape), int(m_shape_ori),
                             int(n_shape*block_out)]
            fusion_out_shape = out_shape_ori
            if is_fractal_out:
                fusion_out_shape = out_shape

            if tensor_bias is not None:
                # tensor_bias shape only be [n,], [1,n] and [1,1,n],
                # bias_shape only be [n,] for gevm and gemm
                if len(bias_shape) == 1:
                    if len(tensor_bias.shape) == 1:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[i],
                            name='tensor_bias_ub')
                    elif len(tensor_bias.shape) == 2:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[0, i],
                            name='tensor_bias_ub')
                    else:
                        tensor_bias_ub = tvm.compute(
                            bias_shape, lambda i: tensor_bias[0, 0, i],
                            name='tensor_bias_ub')
                elif len(bias_shape) == 3:
                    # bias_shape only be (batch, 1, n)
                    tensor_bias_ub = tvm.compute(
                        bias_shape, lambda *indices: tensor_bias[indices],
                        name='tensor_bias_ub')
            if not trans_a:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in, block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[batch, i, j, l, k],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')

                    a_fract_shape = (
                        batch_shape, m_shape, km_shape, block_in_ori, block_reduce)
                    tensor_a_l1_shape = (
                        batch_shape, m_shape, km_shape, block_in, block_reduce)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            a_fract_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_in + k,
                                        j*block_reduce + l],
                            name='tensor_a_ub_fract')

                        def get_a_l1_tensor(a_ub_fract, a_l1_shape):
                            if a_ub_fract.shape[-2].value == 1:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda batch, i, j, k, l:
                                    a_ub_fract[batch, 0, j, 0, l],
                                    name='tensor_a_l1')
                            else:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda batch, i, j, k, l:
                                    a_ub_fract[batch, i, j, k, l],
                                    name='tensor_a_l1')
                            return a_l1
                        tensor_a_l1 = get_a_l1_tensor(tensor_a_ub_fract, tensor_a_l1_shape)

                        tensor_a_l0a = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_l1[indices],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_in + k,
                                        j*block_reduce + l], name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        tensor_a_l1_shape, lambda *indices:
                        tensor_a_l1[indices], name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    if nz_a:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in, block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[batch, j, i, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1 = tvm.compute(
                            gm_a_shape_normalize,
                            lambda *indices: tensor_a[indices], name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (batch_shape, m_shape, km_shape, block_in,
                             block_reduce),
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, l, k], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')

                    if optmt_a == 1:
                        a_fract_shape = (
                            batch_shape, m_shape, km_shape, block_reduce, block_in_ori)
                        tensor_a_l1_shape = (
                            batch_shape, m_shape, km_shape, block_reduce, block_in)
                        tensor_a_ub_fract = tvm.compute(
                            a_fract_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, j*block_reduce + k,
                                        i*block_in + l],
                            name='tensor_a_ub_fract')
                        def get_a_l1_tensor(a_ub_fract, a_l1_shape):
                            if a_ub_fract.shape[-1].value == 1:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda batch, i, j, k, l:
                                    a_ub_fract[batch, 0, j, k, 0],
                                    name='tensor_a_l1')
                            else:
                                a_l1 = tvm.compute(
                                    a_l1_shape, lambda batch, i, j, k, l:
                                    a_ub_fract[batch, i, j, k, l],
                                    name='tensor_a_l1')
                            return a_l1
                        tensor_a_l1 = get_a_l1_tensor(tensor_a_ub_fract, tensor_a_l1_shape)

                        tensor_a_l0a_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        if block_in != cce.BLOCK_VECTOR:
                            def lambda_func(batch, i, j, k, l): return tensor_a_l1[
                                batch, i, j, l, k]
                        else:
                            def lambda_func(batch, i, j, k, l): return tensor_a_l1[
                                batch, i, j, k, l]
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape, lambda_func, name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            batch_shape, km_shape, m_shape, block_reduce, block_in)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[batch, i*block_reduce + k,
                                        j*block_in + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        if block_in != cce.BLOCK_VECTOR:
                            def lambda_func(batch, i, j, k, l): return tensor_a_l1[
                                batch, j, i, l, k]
                        else:
                            def lambda_func(batch, i, j, k, l): return tensor_a_l1[
                                batch, j, i, k, l]
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape, lambda_func, name='tensor_a_l0a')

            if not trans_b:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b_l1[indices],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_not_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l1_shape = (
                                kn_shape, n_shape, block_reduce, block_out)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_ub_fract')

                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda i, j, k, l:
                                    tensor_b_ub[i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                (kn_shape, n_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     i, j, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l1_shape = (
                                batch_shape, kn_shape, n_shape, block_reduce, block_out)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_ub_fract')

                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i*block_reduce + k,
                                                j*block_out + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, kn_shape, n_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, i, j, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_not_fractal()
            else:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')

                    def __get_tensor_l0b_for_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (kn_shape, n_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     j, i, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, kn_shape, n_shape, block_out,
                                 block_reduce),
                                lambda batch, i, j, k, l: tensor_b_l1[
                                    batch, j, i, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[j*block_out + k,
                                                i*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(

                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        i, j, k, l], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[i*block_out + k,
                                                j*block_reduce + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        j, i, k, l], name='tensor_b_l0b')
                        else:
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, j * block_out + k,
                                                i * block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, i, j, k, l], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[batch, i * block_out + k,
                                                j * block_reduce + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, kn_shape, n_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, j, i, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_not_fractal()

            if block_in != cce.BLOCK_VECTOR:
                # define mad compute
                def __get_tensor_c_for_not_block_in_vector():
                    if tensor_b_length in (2, 4):
                        tensor_c = tvm.compute(
                            l0c_shape, lambda batch, nb, mb, mp, np: tvm.sum((
                                (tensor_a_l0a[
                                    batch, mb, reduce_kb, mp, reduce_kp] - offset_x) *
                                tensor_b_l0b[
                                    reduce_kb, nb, np, reduce_kp]).astype(
                                        out_dtype), axis=[reduce_kb, reduce_kp]),
                            name='tensor_c')
                    else:
                        tensor_c = tvm.compute(
                            l0c_shape, lambda batch, nb, mb, mp, np: tvm.sum((
                                (tensor_a_l0a[
                                    batch, mb, reduce_kb, mp, reduce_kp] - offset_x) *
                                tensor_b_l0b[
                                    batch, reduce_kb, nb, np, reduce_kp]).astype(
                                        out_dtype), axis=[reduce_kb, reduce_kp]),
                            name='tensor_c')
                    return tensor_c

                tensor_c = __get_tensor_c_for_not_block_in_vector()

                if tensor_bias is not None:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub shape only be [n,]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                l0c_shape, lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[i*block_out + l],
                                    dtype='float32'), name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                l0c_shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[batch,
                                                         0, i*block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub shape only be [n,]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                l0c_shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    i*block_out + l], name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                l0c_shape, lambda batch, i, j, k, l:
                                tensor_bias_ub[batch,
                                               0, i*block_out + l],
                                name='tensor_bias_l0c')
                    tensor_c_add_bias = tvm.compute(
                        l0c_shape,
                        lambda *indices: tensor_bias_l0c[indices] + tensor_c(
                            *indices), name='tensor_c_add_bias')

                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            out_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            out_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})

                if is_fractal_out:
                    tensor_c_gm = tvm.compute(
                        out_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag="matmul",
                        attrs={'shape': fusion_out_shape,
                               'format': format_out})
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag="matmul",
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
            else:
                # gevm mode, define mad
                def __get_tensor_c_for_block_in_vector():
                    if tensor_b_length in (2, 4):
                        tensor_c = tvm.compute(
                            (batch_shape, n_shape, m_shape, block_out, block_out),
                            lambda batch, nb, mb, mp, np:
                            tvm.sum(((tensor_a_l0a[batch, mb, reduce_kb, mp,
                                                   reduce_kp] - offset_x) *
                                     tensor_b_l0b[reduce_kb, nb, np, reduce_kp])
                                    .astype(out_dtype),
                                    axis=[reduce_kb, reduce_kp]),
                            name='tensor_c')
                    else:
                        tensor_c = tvm.compute(
                            (batch_shape, n_shape, m_shape, block_out, block_out),
                            lambda batch, nb, mb, mp, np:
                            tvm.sum(((tensor_a_l0a[batch, mb, reduce_kb, mp,
                                                   reduce_kp] - offset_x) *
                                     tensor_b_l0b[batch, reduce_kb, nb, np, reduce_kp])
                                    .astype(out_dtype),
                                    axis=[reduce_kb, reduce_kp]),
                            name='tensor_c')
                    return tensor_c

                tensor_c = __get_tensor_c_for_block_in_vector()

                # define reduce
                orig_shape = shape_to_list(tensor_c.shape)
                orig_shape[-2] = block_in

                if tensor_bias is not None:
                    # tensor_bias_ub just is [n,] and [batch, 1, n], no [1, n]
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        # tensor_bias_ub shape only be [n,] and [batch, 1, n]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[i*block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape, lambda batch, i, j, k, l:
                                topi.cast(tensor_bias_ub[batch, 0, i *
                                                         block_out + l],
                                          dtype='float32'),
                                name='tensor_bias_l0c')
                    else:
                        # tensor_bias_ub shape only be [n,] and [batch, 1, n]
                        if len(tensor_bias_ub.shape) == 1:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape, lambda batch, i, j, k, l:
                                tensor_bias_ub[i*block_out + l],
                                name='tensor_bias_l0c')
                        elif len(tensor_bias_ub.shape) == 3:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    batch, 0, i*block_out + l],
                                name='tensor_bias_l0c')

                    tensor_c_add_bias = tvm.compute(
                        tensor_c.shape, lambda *indices:
                        tensor_bias_l0c[indices] + tensor_c[indices],
                        name='tensor_c_add_bias')
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: topi.cast(
                                tensor_c_add_bias[indices], dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: tensor_c_add_bias[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                else:
                    if scale_drq_tensor is not None:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices] *
                            scale_drq_tensor[0], name='tensor_c_ub',
                            attrs={'scale_drq': scale_drq,
                                   'sqrt_out': sqrt_out,
                                   'nz_b': nz_b})
                    elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_c_ub = tvm.compute(
                            orig_shape,
                            lambda *indices: topi.cast(tensor_c[indices],
                                                       dtype='float16'),
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                    else:
                        tensor_c_ub = tvm.compute(
                            orig_shape, lambda *indices: tensor_c[indices],
                            name='tensor_c_ub', attrs={'scale_drq': scale_drq,
                                                       'sqrt_out': sqrt_out,
                                                       'nz_b': nz_b})
                if is_fractal_out:
                    tensor_c_gm = tvm.compute(
                        orig_shape, lambda *indices: tensor_c_ub[indices],
                        name='tensor_c_gm', tag='matmul_gevm',
                        attrs={'shape': orig_shape,
                               'format': format_out})
                else:
                    # ND out shape is dim 2, shape m is original value
                    if optmt_c == 1:
                        tensor_c_ub_fract = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_ub_fract')
                        tensor_c_gm = tvm.compute(
                            out_shape_ori,
                            lambda *indices: tensor_c_ub_fract[indices],
                            name='tensor_c_gm', tag='matmul_gevm',
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
                    else:
                        tensor_c_gm = tvm.compute(
                            out_shape_ori, lambda batch, i, j: tensor_c_ub[
                                batch, j // 16, i // 16, i % 16, j % 16],
                            name='tensor_c_gm', tag="matmul_gevm",
                            attrs={'shape': fusion_out_shape,
                                   'format': format_out})
    else:
        # gemv,c=A*B=(B`A`)`,so B`A` is gevm
        if tensor_a_length in (2, 4):
            if trans_a:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, l, k: tensor_a_l1[i, j, k, l],
                        name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    tensor_a_l1_shape = (
                        km_shape, m_shape, block_reduce, block_in)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_reduce + k, j*block_in + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l: tensor_a_ub[
                                i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, k, l: tensor_a_l1[i, j, l, k],
                        name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (km_shape, m_shape, block_in, block_reduce),
                        lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                        name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            km_shape, m_shape, block_in, block_reduce)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[j*block_in + k, i*block_reduce + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (km_shape, m_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[i, j, k, l],
                            name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda i, j, k, l:
                            tensor_a_ub[i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')
                        tensor_a_l0a = tvm.compute(
                            (km_shape, m_shape, block_in, block_reduce),
                            lambda i, j, k, l: tensor_a_l1[j, i, k, l],
                            name='tensor_a_l0a')

            if trans_b:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        (n_shape, kn_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *i: tensor_b[i],
                        name='tensor_b_ub')
                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l1_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_out + k, j*block_reduce + l],
                            name='tensor_b_l1')
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l1_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, k, l],
                            name='tensor_b_l0b')
            else:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')
                    tensor_b_l0b = tvm.compute(
                        (n_shape, kn_shape, block_out, block_reduce),
                        lambda i, j, k, l: tensor_b_l1[j, i, k, l],
                        name='tensor_b_l0b')
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *i: tensor_b[i],
                        name='tensor_b_ub')
                    if optmt_b == 1:
                        tensor_b_l1_shape = (
                            n_shape, kn_shape, block_reduce, block_out)
                        tensor_b_ub_fract = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[j*block_reduce + k, i*block_out + l],
                            name='tensor_b_ub_fract')
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape,
                            lambda *indices: tensor_b_ub_fract[indices],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[i, j, l, k],
                            name='tensor_b_l0b')
                    else:
                        tensor_b_l1_shape = (
                            kn_shape, n_shape, block_reduce, block_out)
                        tensor_b_l1 = tvm.compute(
                            tensor_b_l1_shape, lambda i, j, k, l:
                            tensor_b_ub[i*block_reduce + k, j*block_out + l],
                            name='tensor_b_l1')
                        tensor_b_l0b_shape = (
                            n_shape, kn_shape, block_out, block_reduce)
                        tensor_b_l0b = tvm.compute(
                            tensor_b_l0b_shape,
                            lambda i, j, k, l: tensor_b_l1[j, i, l, k],
                            name='tensor_b_l0b')

            # define reduce
            orig_shape = [int(m_shape), int(n_shape), int(block_in),
                          int(block_in)]
            orig_shape[-2] = block_out
            out_shape_ori = [int(n_shape*block_out), int(m_shape*block_in)]

            # define mad
            tensor_c = tvm.compute(
                (m_shape, n_shape, block_in, block_in),
                lambda nb, mb, mp, np: tvm.sum(
                    (tensor_b_l0b[mb, reduce_kb, mp, reduce_kp] *
                     (tensor_a_l0a[reduce_kb, nb, np, reduce_kp] - offset_x)
                    ).astype(out_dtype), axis=[reduce_kb, reduce_kp]),
                name='tensor_c')

            if tensor_bias is not None:
                tensor_bias_ub = tvm.compute(
                    tensor_bias.shape, lambda *indices: tensor_bias[indices],
                    name='tensor_bias_ub')
                # bias shape only support [m,1]
                if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_bias_l0c = tvm.compute(
                        tensor_c.shape, lambda i, j, k, l:#orig_shape
                        topi.cast(tensor_bias_ub[i*block_in + l, 0],
                                  dtype='float32'),
                        name='tensor_bias_l0c')
                else:
                    tensor_bias_l0c = tvm.compute(
                        tensor_c.shape, lambda i, j, k, l:
                        tensor_bias_ub[i*block_in + l, 0],
                        name='tensor_bias_l0c')

                tensor_c_add_bias = tvm.compute(
                    tensor_c.shape, lambda *indices:
                    tensor_bias_l0c[indices] + tensor_c[indices],
                    name='tensor_c_add_bias')
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices] *
                        scale_drq_tensor[0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c_add_bias[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            else:
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c[indices]*scale_drq_tensor[
                            0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape, lambda *indices: tensor_c[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            if is_fractal_out:
                tensor_c_gm = tvm.compute(
                    orig_shape, lambda *indices: tensor_c_ub[indices],
                    name='tensor_c_gm', tag='matmul_gemv',
                    attrs={'shape': orig_shape,
                           'format': format_out})
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_ub_fract = tvm.compute(
                        out_shape_ori, lambda i, j:
                        tensor_c_ub[j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_ub_fract')
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda *indices: tensor_c_ub_fract[indices],
                        name='tensor_c_gm', tag='matmul_gemv',
                        attrs={'shape': out_shape_ori,
                               'format': format_out})
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori, lambda i, j:
                        tensor_c_ub[j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_gm', tag="matmul_gemv",
                        attrs={'shape': out_shape_ori,
                               'format': format_out})

        else:
            # have batch size
            batch_shape = tensor_a.shape[0].value
            out_shape_ori = [batch_shape, int(n_shape*block_out),
                             int(m_shape_ori)]

            if trans_a:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce), lambda batch, i, j, l, k:
                        tensor_a_l1[batch, i, j, k, l], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape,
                        lambda *indices: tensor_a[indices], name='tensor_a_ub')
                    tensor_a_l1_shape = (
                        batch_shape, km_shape, m_shape, block_reduce, block_in)
                    if optmt_a == 1:
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_reduce + k, j*block_in + l],
                            name='tensor_a_ub_fract')

                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                    else:
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_reduce + k, j*block_in + l],
                            name='tensor_a_l1')

                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce),
                        lambda batch, i, j, k, l: tensor_a_l1[
                            batch, i, j, l, k], name='tensor_a_l0a')
            else:
                if is_fractal_a:
                    tensor_a_l1 = tvm.compute(
                        gm_a_shape_normalize,
                        lambda *indices: tensor_a[indices], name='tensor_a_l1')
                    tensor_a_l0a = tvm.compute(
                        (batch_shape, km_shape, m_shape, block_in,
                         block_reduce),
                        lambda batch, i, j, k, l: tensor_a_l1[
                            batch, j, i, k, l], name='tensor_a_l0a')
                else:
                    tensor_a_ub = tvm.compute(
                        tensor_a.shape, lambda *i: tensor_a[i],
                        name='tensor_a_ub')
                    if optmt_a == 1:
                        tensor_a_l1_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_ub_fract = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, j*block_in + k, i*block_reduce + l],
                            name='tensor_a_ub_fract')
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape,
                            lambda *indices: tensor_a_ub_fract[indices],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, i, j, k, l], name='tensor_a_l0a')
                    else:
                        tensor_a_l1_shape = (
                            batch_shape, m_shape, km_shape, block_in, block_reduce)
                        tensor_a_l1 = tvm.compute(
                            tensor_a_l1_shape, lambda batch, i, j, k, l:
                            tensor_a_ub[
                                batch, i*block_in + k, j*block_reduce + l],
                            name='tensor_a_l1')
                        tensor_a_l0a_shape = (
                            batch_shape, km_shape, m_shape, block_in, block_reduce)
                        tensor_a_l0a = tvm.compute(
                            tensor_a_l0a_shape,
                            lambda batch, i, j, k, l: tensor_a_l1[
                                batch, j, i, k, l], name='tensor_a_l0a')
            if trans_b:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')

                    def __get_tensor_l0b_for_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (n_shape, kn_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     i, j, l, k], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, n_shape, kn_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, i, j, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l1_shape = (
                                n_shape, kn_shape, block_out, block_reduce)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                tensor_b_l1_shape,
                                lambda i, j, k, l: tensor_b_l1[
                                    i, j, k, l], name='tensor_b_l0b')
                        else:
                            tensor_b_l1_shape = (
                                batch_shape, n_shape, kn_shape, block_out, block_reduce)
                            if optmt_b == 1:
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                            else:
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_out + k, j*block_reduce + l],
                                    name='tensor_b_l1')
                            tensor_b_l0b = tvm.compute(
                                tensor_b_l1_shape,
                                lambda batch, i, j, k, l: tensor_b_l1[
                                    batch, i, j, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_trans_and_not_fractal()
            else:
                if is_fractal_b:
                    if compress_index is not None:
                        tensor_b_l1 = _get_compress_tensor_compute(
                            tensor_b, compress_index, 'tensor_b_l1')
                    else:
                        tensor_b_l1 = tvm.compute(
                            tensor_b.shape, lambda *indices: tensor_b[indices],
                            name='tensor_b_l1')

                    def __get_tensor_l0b_for_not_trans_and_fractal():
                        if tensor_b_length in (2, 4):
                            tensor_b_l0b = tvm.compute(
                                (n_shape, kn_shape, block_out,
                                 block_reduce), lambda i, j, k, l: tensor_b_l1[
                                     j, i, k, l], name='tensor_b_l0b')
                        else:
                            tensor_b_l0b = tvm.compute(
                                (batch_shape, n_shape, kn_shape, block_out,
                                 block_reduce), lambda batch, i, j, k, l: tensor_b_l1[
                                     batch, j, i, k, l], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_fractal()
                else:
                    tensor_b_ub = tvm.compute(
                        tensor_b.shape, lambda *indices: tensor_b[indices],
                        name='tensor_b_ub')

                    def __get_tensor_l0b_for_not_trans_and_not_fractal():
                        if tensor_b_length in (2, 4):
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    n_shape, kn_shape, block_reduce, block_out)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        j*block_reduce + k, i*block_out + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        i, j, l, k], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    kn_shape, n_shape, block_reduce, block_out)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda i, j, k, l:
                                    tensor_b_ub[
                                        i*block_reduce + k, j*block_out + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda i, j, k, l: tensor_b_l1[
                                        j, i, l, k], name='tensor_b_l0b')
                        else:
                            if optmt_b == 1:
                                tensor_b_l1_shape = (
                                    batch_shape, n_shape, kn_shape, block_reduce, block_out)
                                tensor_b_ub_fract = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, j*block_reduce + k, i*block_out + l],
                                    name='tensor_b_ub_fract')
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape,
                                    lambda *indices: tensor_b_ub_fract[indices],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, i, j, l, k], name='tensor_b_l0b')
                            else:
                                tensor_b_l1_shape = (
                                    batch_shape, kn_shape, n_shape, block_reduce, block_out)
                                tensor_b_l1 = tvm.compute(
                                    tensor_b_l1_shape, lambda batch, i, j, k, l:
                                    tensor_b_ub[
                                        batch, i*block_reduce + k, j*block_out + l],
                                    name='tensor_b_l1')
                                tensor_b_l0b_shape = (
                                    batch_shape, n_shape, kn_shape, block_out, block_reduce)
                                tensor_b_l0b = tvm.compute(
                                    tensor_b_l0b_shape,
                                    lambda batch, i, j, k, l: tensor_b_l1[
                                        batch, j, i, l, k], name='tensor_b_l0b')
                        return tensor_b_l0b

                    tensor_b_l0b = __get_tensor_l0b_for_not_trans_and_not_fractal()

            # define mad
            def __get_tensor_c():
                if tensor_b_length in (2, 4):
                    tensor_c = tvm.compute(
                        (batch_shape, m_shape, n_shape, block_in, block_in),
                        lambda batch, nb, mb, mp, np: tvm.sum(
                            (tensor_b_l0b[mb, reduce_kb, mp, reduce_kp] *
                             (tensor_a_l0a[batch, reduce_kb, nb, np, reduce_kp] - \
                             offset_x)).astype(out_dtype),
                            axis=[reduce_kb, reduce_kp]),
                        name='tensor_c')
                else:
                    tensor_c = tvm.compute(
                        (batch_shape, m_shape, n_shape, block_in, block_in),
                        lambda batch, nb, mb, mp, np: tvm.sum(
                            (tensor_b_l0b[batch, mb, reduce_kb, mp, reduce_kp] *
                             (tensor_a_l0a[batch, reduce_kb, nb, np, reduce_kp] - \
                             offset_x)).astype(out_dtype),
                            axis=[reduce_kb, reduce_kp]),
                        name='tensor_c')
                return tensor_c

            tensor_c = __get_tensor_c()
            # define reduce
            orig_shape = shape_to_list(tensor_c.shape)
            orig_shape[-2] = block_out

            if tensor_bias is not None:
                tensor_bias_ub = tvm.compute(
                    tensor_bias.shape, lambda *indices: tensor_bias[indices],
                    name='tensor_bias_ub')

                # bias shape support [m,1] or [1,m,1] or [b,m,1]
                if len(bias_shape) == 2:
                    if len(tensor_bias.shape) == 2:
                        if tensor_bias.dtype == 'float16' and \
                                l0c_support_fp32 == 1:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape,
                                lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[i*block_in + l, 0],
                                    dtype='float32'), name='tensor_bias_l0c')
                        else:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    i*block_in + l, 0],
                                name='tensor_bias_l0c')
                    else:
                        if tensor_bias.dtype == 'float16' and \
                                l0c_support_fp32 == 1:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape,
                                lambda batch, i, j, k, l: topi.cast(
                                    tensor_bias_ub[0, i*block_in + l, 0],
                                    dtype='float32'), name='tensor_bias_l0c')
                        else:
                            tensor_bias_l0c = tvm.compute(
                                tensor_c.shape,
                                lambda batch, i, j, k, l: tensor_bias_ub[
                                    0, i*block_in + l, 0],
                                name='tensor_bias_l0c')
                elif len(bias_shape) == 3:
                    if tensor_bias.dtype == 'float16' and l0c_support_fp32 == 1:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape, lambda batch, i, j, k, l: topi.cast(
                                tensor_bias_ub[batch, i*block_in + l, 0],
                                dtype='float32'), name='tensor_bias_l0c')
                    else:
                        tensor_bias_l0c = tvm.compute(
                            tensor_c.shape,
                            lambda batch, i, j, k, l: tensor_bias_ub[
                                batch, i*block_in + l, 0],
                            name='tensor_bias_l0c')

                tensor_c_add_bias = tvm.compute(
                    tensor_c.shape,
                    lambda *indices: tensor_bias_l0c[indices] + tensor_c(
                        *indices), name='tensor_c_add_bias')
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices] *
                        scale_drq_tensor[0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c_add_bias[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c_add_bias[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            else:
                # gemv/gevm in nd or fractal just copy in continuous
                if scale_drq_tensor is not None:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: tensor_c[indices]*scale_drq_tensor[
                            0], name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                elif dst_dtype == 'float16' and l0c_support_fp32 == 1:
                    tensor_c_ub = tvm.compute(
                        orig_shape,
                        lambda *indices: topi.cast(tensor_c[indices],
                                                   dtype='float16'),
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
                else:
                    tensor_c_ub = tvm.compute(
                        orig_shape, lambda *indices: tensor_c[indices],
                        name='tensor_c_ub',
                        attrs={'scale_drq': scale_drq, 'sqrt_out': sqrt_out, 'nz_b': nz_b})
            if is_fractal_out:
                tensor_c_gm = tvm.compute(
                    orig_shape, lambda *indices: tensor_c_ub[indices],
                    name='tensor_c_gm', tag='matmul_gemv',
                    attrs={'shape': orig_shape,
                           'format': format_out})
            else:
                # ND out shape is dim 2, shape m is original value
                if optmt_c == 1:
                    tensor_c_ub_fract = tvm.compute(
                        out_shape_ori, lambda batch, i, j: tensor_c_ub[
                            batch, j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_ub_fract')
                    tensor_c_gm = tvm.compute(
                        out_shape_ori,
                        lambda *indices: tensor_c_ub_fract[indices],
                        name='tensor_c_gm', tag='matmul_gemv',
                        attrs={'shape': out_shape_ori,
                               'format': format_out})
                else:
                    tensor_c_gm = tvm.compute(
                        out_shape_ori, lambda batch, i, j: tensor_c_ub[
                            batch, j // 16, i // 16, i % 16, j % 16],
                        name='tensor_c_gm', tag="matmul_gemv",
                        attrs={'shape': out_shape_ori,
                               'format': format_out})

    return tensor_c_gm


def _get_tensor_c_out_dtype(tensor_a, tensor_b, dst_dtype):
    """
    get tensor c out date type

    Parameters
    ----------
    tensor_a : input tensor a

    tensor_b : input tensor b

    dst_dtype : input destination date type

    Returns
    -------
    out_dtype : tensor c out date type

    """

    in_a_dtype = tensor_a.dtype
    in_b_dtype = tensor_b.dtype

    l0c_support_fp32 = 1

    support_type = cce_conf.getValue("Intrinsic_mmad")
    if "f162f32" not in support_type:
        l0c_support_fp32 = 0
    if in_a_dtype == "float16" and in_b_dtype == "float16":
        if dst_dtype not in ("float16", "float32"):
            dict_args = {
                'errCode': 'E61001',
                'reason': "dst_dtype must be 'float16' or 'float32'."
            }
            raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
        out_dtype = "float32"
        if l0c_support_fp32 == 0:
            out_dtype = "float16"
    elif (in_a_dtype == "int8" and in_b_dtype == "int8") or \
            (in_a_dtype == "uint8" and in_b_dtype == "int8"):
        out_dtype = "int32"
    else:
        dict_args = {
            'errCode': 'E61001',
            'reason': "data type of tensor not supported."
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    return out_dtype


def _get_block_in_value():
    """
    get block in value

    Parameters
    ----------

    Returns
    -------
    block in value

    """
    return cce.BLOCK_IN


def _get_block_out_value():
    """
    get block out value

    Parameters
    ----------

    Returns
    -------
    block out value

    """
    return cce.BLOCK_OUT


def _get_block_reduce_value(tensor_dtype):
    """
    get block reduce value

    Parameters
    ----------
    tensor_dtype : tensor date type

    Returns
    -------
    block_reduce : block redduce value

    """
    if tensor_dtype == "float16":
        block_reduce = cce.BLOCK_REDUCE
    else:
        block_reduce = cce.BLOCK_REDUCE_INT8
    return block_reduce


def _update_gevm_block_in(m_shape, vm_shape, km_shape, block_in_ori):
    """
    update gevm block in value

    Parameters
    ----------
    m_shape : m axis value

    vm_shape : m axis value

    km_shape : k axis value

    block_in_ori : original block in value

    Returns
    -------
    block_in : block in value

    """
    block_in = block_in_ori
    if m_shape == 1 and vm_shape == 1 and km_shape % block_in == 0:
        block_in = cce.BLOCK_VECTOR
    return block_in


def _update_gevm_block_out(n_shape, vn_shape, block_out_ori):
    """
    update gevm block out value

    Parameters
    ----------
    n_shape : n axis value

    vn_shape : n axis value

    block_out_ori : original block out value

    Returns
    -------
    block_out : block out value

    """
    block_out = block_out_ori
    if n_shape == 1 and vn_shape == 1:
        block_out = cce.BLOCK_VECTOR
    return block_out


def _get_m_k_value(tensor_a, trans_a):
    """
    get tensor a m k axis value

    Parameters
    ----------
    tensor_a : input tensor a

    trans_a : if True, a needs to be transposed

    Returns
    -------
    m_shape : tensor a m axis value

    k_shape : tensor a k axis value

    vm_shape : tensor a m axis value

    """
    tensor_a_length = len(tensor_a.shape)
    if trans_a:
        m_shape = tensor_a.shape[tensor_a_length - 1].value
        km_shape = tensor_a.shape[tensor_a_length - 2].value
        vm_shape = 16
        if tensor_a.shape[tensor_a_length - 1].value == 1:
            m_shape = 1
            vm_shape = 1
    else:
        m_shape = tensor_a.shape[tensor_a_length - 2].value
        km_shape = tensor_a.shape[tensor_a_length - 1].value
        vm_shape = 16
        if tensor_a.shape[tensor_a_length - 2].value == 1:
            m_shape = 1
            vm_shape = 1

    return m_shape, km_shape, vm_shape


def _get_n_k_value(tensor_b, trans_b):
    """
    get tensor b n k axis value

    Parameters
    ----------
    tensor_b : input tensor b

    trans_b : if True, b needs to be transposed

    Returns
    -------
    n_shape : tensor b n axis value

    k_shape : tensor b k axis value

    vn_shape : tensor b n axis value

    """
    tensor_b_length = len(tensor_b.shape)
    if trans_b:
        kn_shape = tensor_b.shape[tensor_b_length - 1].value
        n_shape = tensor_b.shape[tensor_b_length - 2].value
        vn_shape = 16
        if tensor_b.shape[tensor_b_length - 2].value == 1:
            n_shape = 1
            vn_shape = 1
    else:
        kn_shape = tensor_b.shape[tensor_b_length - 2].value
        n_shape = tensor_b.shape[tensor_b_length - 1].value
        vn_shape = 16
        if tensor_b.shape[tensor_b_length - 1].value == 1:
            n_shape = 1
            vn_shape = 1

    return n_shape, kn_shape, vn_shape


def _check_reduce_shape(km_shape, kn_shape):
    """
    check reduce shape valid

    Parameters
    ----------
    km_shape : km shape value

    kn_shape : kn shape value

    Returns
    -------
    None

    """
    if km_shape != kn_shape:
        dict_args = {
            'errCode': 'E61001',
            'reason': "the k shape is wrong in mmad."
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _check_default_param(alpha_num, beta_num, quantize_params, format_out):
    """
    check get perfomance format default param

    Parameters
    ----------
    alpha_num : scalar used for multiplication

    beta_num : scalar used for multiplication

    quantize_params : quantization parameters

    format_out : matmul output format

    Returns
    -------
    None

    """
    if alpha_num != 1.0 or beta_num != 1.0:
        dict_args = {
            'errCode': 'E61001',
            'reason': "mmad now only supprot alpha_num = {0} and beta_num = {0}.".format(1.0)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if quantize_params is not None:
        dict_args = {
            'errCode': 'E61001',
            'reason': "quant parameter should be none."
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))
    if format_out is not None:
        dict_args = {
            'errCode': 'E61001',
            'reason': "format output should be none."
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))


def _get_l0_byte(tensor_a_dtype, tensor_b_dtype, l0c_out_dtype):
    """
    calculate the number of l0 bytes occupied by an element

    Parameters
    ----------
    tensor_a_dtype : tensor a date type

    tensor_b_dtype : tensor b date type

    l0c_out_dtype : tensor c out date type

    Returns
    -------
    l1a_byte : number of l0 bytes occupied by tensor a element

    l1b_byte : number of l0 bytes occupied by tensor b element

    l0c_byte : number of l0 bytes occupied by tensor c element

    """
    l0a_byte = 2 if (tensor_a_dtype == "float16") else 1
    l0b_byte = 2 if (tensor_b_dtype == "float16") else 1
    l0c_byte = 2 if (l0c_out_dtype == "float16") else 4
    return l0a_byte, l0b_byte, l0c_byte


def _get_l1_byte(tensor_a_dtype, tensor_b_dtype):
    """
    calculate the number of l1 bytes occupied by an element

    Parameters
    ----------
    tensor_a_dtype : tensor a date type

    tensor_b_dtype : tensor b date type

    Returns
    -------
    l1a_byte : number of l1 bytes occupied by tensor a element

    l1b_byte : number of l1 bytes occupied by tensor b element

    """
    l1a_byte = 2 if (tensor_a_dtype == "float16") else 1
    l1b_byte = 2 if (tensor_b_dtype == "float16") else 1
    return l1a_byte, l1b_byte


def _is_gemv_mode(block_out):
    """
    Determine if GEMV mode

    Parameters
    ----------
    block_out : block out value

    Returns
    -------
    bool : true or false

    """
    if block_out != cce.BLOCK_VECTOR:
        return False
    return True


def _is_gevm_mode(block_in):
    """
    Determine if GEVM mode

    Parameters
    ----------
    block_in : block in value

    Returns
    -------
    bool : true or false

    """
    if block_in != cce.BLOCK_VECTOR:
        return False
    return True


def _get_core_factor(m_var, n_var):
    """
    get block core process uint

    Parameters
    ----------
    m_var : list, include m axis value, m split value

    n_var : list, include n axis value, n split value

    Returns
    -------
    core_inner_m : block core process m axis uint

    core_inner_n : block core process n axis uint

    """
    m_shape = m_var[0]
    m_factors = m_var[1]
    n_shape = n_var[0]
    n_factors = n_var[1]
    core_inner_m = m_shape
    core_inner_n = n_shape

    block_in = cce.BLOCK_IN
    block_out = cce.BLOCK_OUT
    if m_shape != 1:
        core_inner_m = (((m_shape + block_in - 1) // block_in +
                         (m_factors - 1)) // m_factors) * block_in

    core_inner_n = (((n_shape + block_out - 1) // block_out +
                     (n_factors - 1)) // n_factors) * block_out

    return core_inner_m, core_inner_n


def _get_ub_byte_size(date_type, tensor_ub_fract):
    """
    calculate the number of ub bytes occupied by an element

    Parameters
    ----------
    date_type : element date type

    tensor_ub_fract : bool flag, mark whether data rearrangement is required

    Returns
    -------
    ub_byte : number of bytes occupied by an element

    """
    ub_byte = 0
    if date_type is None:
        return ub_byte
    if date_type == "float16":
        ub_byte = 2
    elif date_type in ("int8", "uint8"):
        ub_byte = 1
    else:
        ub_byte = 4

    if tensor_ub_fract:
        ub_byte = ub_byte * 2

    return ub_byte


def _get_special_l0_factor_tiling(src_shape, m_l0_shape, k_l0_shape, n_l0_shape):
    """
    update l0 tiling parameter

    Parameters
    ----------
    src_shape : m k n shape list

    m_l0_shape : m axis split value in l0

    k_l0_shape : k axis split value in l0

    n_l0_shape : n axis split value in l0

    Returns
    -------
    m_l0_shape : m axis split value in l0

    k_l0_shape : k axis split value in l0

    n_l0_shape : n axis split value in l0
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


def _get_core_num_tiling(m_shape,  # pylint: disable=too-many-locals
                        n_shape, k_shape):
    """
    get block tiling parameter

    Parameters
    ----------
    m_shape : tensor a m axis value

    n_shape : tensor b n axis value

    k_shape : tensor a k axis value

    Returns
    -------
    m_factor : m axis split factor

    n_factor : n axis split factor
    """
    frac_size = 16
    core_num = cce_conf.getValue("Device_core_num")
    m_axis_outer = (m_shape + frac_size - 1) // frac_size
    if m_shape == 1:
        m_axis_outer = 1
        n_axis_outer = (n_shape + frac_size - 1) // frac_size
        if n_axis_outer > core_num:
            return 1, core_num
        return 1, 1

    m_axis_outer = (m_shape + frac_size - 1) // frac_size
    n_axis_outer = (n_shape + frac_size - 1) // frac_size
    if (m_axis_outer * n_axis_outer) <= core_num:
        return m_axis_outer, n_axis_outer
    tensor_a_size = m_shape * k_shape
    tensor_b_size = n_shape * k_shape
    min_copy_size = core_num * (tensor_a_size + tensor_b_size)
    m_factor = m_axis_outer
    n_factor = n_axis_outer

    exp = 1
    if core_num == 32:
        # the exp for 32, 2^(6-1)
        exp = 6
    elif core_num == 2:
        # the exp for 2, 2^(2-1)
        exp = 2
    for i in (2 ** e for e in range(0, exp)):
        # judge cur_factor
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


@para_check.check_input_type(tvm.tensor.Tensor, tvm.tensor.Tensor, bool, bool, str,
                             str, float, float, str, (type(None), tvm.tensor.Tensor),
                             (type(None), dict), (type(None), str))
def get_matmul_performance_format(tensor_a,  # pylint: disable=W0108, R1702, R0912, R0913, R0914, R0915
                                  tensor_b, trans_a=False, trans_b=False,
                                  format_a="ND", format_b="ND", alpha_num=1.0,
                                  beta_num=1.0, dst_dtype="float16",
                                  tensor_bias=None, quantize_params=None,
                                  format_out=None):
    """
    get matmul performance format

    Parameters:
    tensor_a : the first tensor a

    tensor_b : second tensor b

    trans_a : if True, a needs to be transposed

    trans_b : if True, b needs to be transposed

    is_fractal: format is fractal or ND

    alpha_num: scalar used for multiplication

    beta_num: scalar used for multiplication

    dst_dtype: output data type

    tensor_bias :the bias with used to init L0C for tensor c

    quantize_params: quantization parameters

    out_format: output format

    Returns: tensor a format
    """

    l0c_out_dtype = _get_tensor_c_out_dtype(tensor_a, tensor_b, dst_dtype)

    block_in = _get_block_in_value()

    m_shape, km_shape, vm_shape = _get_m_k_value(tensor_a, trans_a)

    frac_n_size = 16
    frac_k_size = 16
    input_dtype = tensor_a.dtype
    if input_dtype in ("int8", "uint8"):
        frac_k_size = 32
    km_shape = (km_shape + frac_k_size - 1) // frac_k_size * frac_k_size

    if m_shape == 1:
        return "NC1HWC0"

    remainder = m_shape % 16
    if remainder * km_shape > FORMAT_DYNAMIC_THRESHOLD:
        return "NC1HWC0"

    n_shape, kn_shape, vn_shape = _get_n_k_value(tensor_b, trans_b)
    kn_shape = (kn_shape + frac_k_size - 1) // frac_k_size * frac_k_size
    n_shape = (n_shape + frac_n_size - 1) // frac_n_size * frac_n_size

    block_in = _update_gevm_block_in(m_shape, vm_shape, km_shape, block_in)

    _check_reduce_shape(km_shape, kn_shape)
    _check_default_param(alpha_num, beta_num, quantize_params, format_out)

    m_factors, n_factors = _get_core_num_tiling(m_shape, n_shape, km_shape)

    m_var = [m_shape, m_factors]
    n_var = [n_shape, n_factors]

    core_inner_m, core_inner_n = _get_core_factor(m_var, n_var)

    l0a_byte, l0b_byte, l0c_byte = _get_l0_byte(tensor_a.dtype,
                                               tensor_b.dtype,
                                               l0c_out_dtype)

    l1a_byte, l1b_byte = _get_l1_byte(tensor_a.dtype,
                                     tensor_b.dtype)

    ub_reserve_buff = 0

    a_ub_byte = _get_ub_byte_size(tensor_a.dtype, True)
    b_ub_byte = _get_ub_byte_size(None, False)
    ub_res_byte = _get_ub_byte_size(l0c_out_dtype, False)

    is_gemv = _is_gemv_mode(block_in)
    if is_gemv:
        a_ub_byte, b_ub_byte = b_ub_byte, a_ub_byte

    get_tiling_shape = tvm.get_global_func("cce.matmul_tiling_gen")
    tiling_shape = get_tiling_shape(core_inner_m, km_shape, core_inner_n,
                                    a_ub_byte,
                                    b_ub_byte, l1a_byte,
                                    l1b_byte, l0a_byte, l0b_byte, l0c_byte,
                                    ub_res_byte, ub_reserve_buff,
                                    False, False)

    if tiling_shape.find('_') == -1:
        dict_args = {
            'errCode': 'E61001',
            'reason': "tiling shape {} is illegal.".format(tiling_shape)
        }
        raise RuntimeError(dict_args, error_manager_util.get_error_message(dict_args))

    tiled_shape = tiling_shape.split('_')
    m_l0_shape = int(tiled_shape[3])
    k_l0_shape = int(tiled_shape[4])
    n_l0_shape = int(tiled_shape[5])

    src_shape = [m_shape, km_shape, n_shape]
    m_l0_shape, k_l0_shape, n_l0_shape = _get_special_l0_factor_tiling(
        src_shape, m_l0_shape, k_l0_shape, n_l0_shape)

    if core_inner_n // n_l0_shape > 1:
        return "FRACTAL_NZ"

    return "NC1HWC0"


def _matmul_cv_split(tensor_a,
    tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
    dst_dtype="float32", tensor_bias=None, format_out=None, kernel_name="MatMul"):
    """
    algorithm: mmad
    calculating matrix multiplication, C=A*B+bias

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

    format_a: the format of tensor a, support FRACTAL_NZ, ND
              default is "ND"

    format_b: the format of tensor b, support FRACTAL_NZ, ND
              default is "ND"

    dst_dtype: output data type, support "float32", default is "float32"

    tensor_bias :the bias with used to add

    format_out: output format, now support ND,Nz

    kernel_name: kernel name, default is "MatMul"

    Returns None
    """
    matmul_object = MatMulCompute(
        tensor_a,
        tensor_b,
        trans_a,
        trans_b,
        format_a,
        format_b,
        dst_dtype,
        tensor_bias,
        format_out,
        kernel_name
    )

    matmul_object._compute_matmul()

    res = matmul_object.c_matrix

    return res


class MatMulCompute:
    """
    algorithm: mmad
    calculating matrix multiplication, C=A*B+bias

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

    format_a: the format of tensor a, support FRACTAL_NZ, ND
              default is "ND"

    format_b: the format of tensor b, support FRACTAL_NZ, ND
              default is "ND"

    dst_dtype: output data type, support "float32", default is "float32"

    tensor_bias :the bias with used to add

    format_out: output format, now support ND,Nz

    kernel_name: kernel name, default is "MatMul"

    Returns None
    """
    def __init__(self, tensor_a,
        tensor_b, trans_a=False, trans_b=False, format_a="ND", format_b="ND",
        dst_dtype="float32", tensor_bias=None, format_out=None, kernel_name="MatMul"):
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.trans_a = trans_a
        self.trans_b = trans_b
        self.format_a = format_a
        self.format_b = format_b
        self.tensor_bias = tensor_bias
        self.src_dtype = tensor_a.dtype
        self.dst_dtype = dst_dtype
        self.kernel_name = kernel_name
        self.block_in = cce.BLOCK_IN
        self.block_out = cce.BLOCK_OUT
        self.block_reduce = cce.BLOCK_REDUCE
        self.matrix_type = "float32"
        self.format_out = format_out

    @staticmethod
    def _ceil_div(dividend, divisor):
        """
        do division and round up to an integer

        """
        if divisor == 0:
            dict_args = {}
            dict_args["errCode"] = "E60108"
            dict_args["reason"] = "Division by zero"
            error_manager_util.raise_runtime_error(dict_args)
        return (dividend + divisor - 1) // divisor

    def _compute_matmul(self):
        """MatMul enter
        Input None
        return result in self.c_matrix
        ---------------------------------
        Return None
        """
        # set data type and format, get data shape
        self._set_info()
        self._get_l1_shape()
        self._check_km_kn()
        # C = A * B
        a_matrix = self._get_a_matrix()
        b_matrix = self._get_b_matrix()
        self.c_matrix = self._compute_c_matrix(a_matrix, b_matrix)

    def _get_a_matrix(self):
        """ compute matrix for mad
        Input : None
        support func:
            fp16 input:
                Nz->Zz
                ND->Nz->Zz
        ---------------------------------
        Return : tensor, Zz matrix for mad
        """

        batch_shape = self.batch_shape
        km_shape = self.km_shape
        m_shape = self.m_shape
        block_reduce = self.block_reduce
        block_in = self.block_in
        temp_tensor_a = self.tensor_a

        # to Nz
        if self.format_a == "ND":
            temp_tensor_a = tvm.compute(
            (km_shape, m_shape * batch_shape, block_in, block_reduce),
            lambda i, j, k, l: temp_tensor_a[j * block_in + k, i * block_reduce + l],
            name="tensor_a_fract")

        # to Zz
        if self.trans_a:
            a_matrix_shape = (km_shape, m_shape * batch_shape, block_reduce, block_in)
            a_matrix = tvm.compute(
                a_matrix_shape,
                lambda i, j, k, l: temp_tensor_a[i, j, l, k],
                name="tensor_a_matrix",
                attrs={"trans": "1"})
        else:
            a_matrix_shape = (m_shape * batch_shape, km_shape, block_in, block_reduce)
            a_matrix = tvm.compute(
                a_matrix_shape,
                lambda i, j, k, l: temp_tensor_a[j, i, k, l],
                name="tensor_a_matrix",
                attrs={"trans": "0"})
        return a_matrix

    def _get_b_matrix(self):
        """ compute matrix for mad
        Input : None
        support func:
            fp16 input:
                Nz->Zn
                ND->Nz->Zn
        ---------------------------------
        Return : tensor, Zn matrix for mad
        """

        batch_shape = self.batch_shape
        n_shape = self.n_shape
        kn_shape = self.kn_shape
        block_out = self.block_out
        block_reduce = self.block_reduce
        temp_tensor_b = self.tensor_b

        # to Nz
        if self.format_b == "ND":
            temp_tensor_b = tvm.compute(
                (n_shape * batch_shape, kn_shape, block_reduce, block_out),
                lambda i, j, k, l: temp_tensor_b[j * block_reduce + k, i * block_out + l],
                name="tensor_b_fract")

        # to Zn
        if self.trans_b:
            b_matrix_shape = (n_shape * batch_shape, kn_shape, block_reduce, block_out)
            b_matrix = tvm.compute(
                b_matrix_shape,
                lambda i, j, k, l: temp_tensor_b[i, j, k, l],
                name="tensor_b_matrix",
                attrs={"trans": "1"})
        else:
            b_matrix_shape = (kn_shape, n_shape * batch_shape, block_out, block_reduce)
            b_matrix = tvm.compute(
                b_matrix_shape,
                lambda i, j, k, l: temp_tensor_b[j, i, l, k],
                name="tensor_b_matrix",
                attrs={"trans": "0"})

        return b_matrix

    def _compute_c_matrix(self, a_matrix_in, b_matrix_in):
        """ MatMul calculation
        Input:
            a_matrix_in: tensor, a_matrix in l0a
            b_matrix_in: tensor, b_matrix in l0b
        support func:
            MatMul, Nz->ND
        ---------------------------------
        Return:
            tensor, MatMul result
        """
        m_shape_l0 = a_matrix_in.shape[0]
        k_shape_l0 = a_matrix_in.shape[1]
        n_shape_l0 = b_matrix_in.shape[1]

        nz2nd_flag = False
        if self.format_out == "ND":
            nz2nd_flag = True
        reduce_kp, reduce_kb = self._get_reduce(k_shape_l0)
        if self.tensor_bias is None:
            tensor_c_matrix = tvm.compute(
                (n_shape_l0 * self.batch_shape, m_shape_l0,
                 self.block_in, self.block_out),
                lambda nb, mb, mp, np: tvm.sum(
                    (a_matrix_in[mb, reduce_kb,
                                 mp, reduce_kp] * b_matrix_in[
                        reduce_kb, nb, np, reduce_kp]).astype(
                        self.matrix_type),
                    axis=[reduce_kb, reduce_kp]),
                name="tensor_c_matrix",
                tag="gemm" if not nz2nd_flag else "",
                attrs={"kernel_name": self.kernel_name})
        else:
            tensor_c_matrix = tvm.compute(
                (self.n_shape * self.batch_shape,
                 self.m_shape, self.block_in, self.block_out),
                lambda nb, mb, mp, np: tvm.sum(tvm.select(
                    tvm.all(reduce_kb.var == 0, reduce_kp.var == 0),
                    (a_matrix_in[mb, reduce_kb, mp, reduce_kp] * b_matrix_in[
                        reduce_kb, nb, np, reduce_kp]).astype(
                        self.matrix_type) +
                    self.tensor_bias[nb * self.block_out + np],
                    (a_matrix_in[mb, reduce_kb, mp, reduce_kp] * b_matrix_in[
                        reduce_kb, nb, np, reduce_kp]).astype(
                        self.matrix_type)),
                    axis=[reduce_kb, reduce_kp]),
                name='tensor_c_matrix', tag="gemm" if not nz2nd_flag else "",
                attrs={"kernel_name": self.kernel_name})

        if nz2nd_flag:
            tensor_c_gm = tvm.compute(
                (m_shape_l0 * self.block_in, n_shape_l0 * self.block_out),
                lambda i, j: tensor_c_matrix[j // self.block_out,
                                             i // self.block_in,
                                             i % self.block_in,
                                             j % self.block_out],
                tag="gemm",
                name="tensor_c_gm",
                attrs={"kernel_name": self.kernel_name})
        else:
            tensor_c_gm = tensor_c_matrix

        return tensor_c_gm

    def _get_reduce(self, k_shape):
        """get reduce axis for MatMul
        Input:
            k_shape: A martix's k
        ---------------------------------
        Return:
            axis, two reduce axis
        """
        # kBurstAxis and kPointAxis
        if self.src_dtype == "int8" and self.dst_dtype == "float32":
            reduce_kp = tvm.reduce_axis((0, self.block_reduce // 2), name="kp")
            reduce_kb = tvm.reduce_axis((0, k_shape * 2), name="kb")
        else:
            reduce_kp = tvm.reduce_axis((0, self.block_reduce), name="kp")
            reduce_kb = tvm.reduce_axis((0, k_shape), name="kb")

        return reduce_kp, reduce_kb

    def _set_info(self):
        """set info about format
        Input: None
        ---------------------------------
        Return: None
        """
        format_a = self.format_a

        if self.src_dtype == "int8":
            self.block_reduce = cce.BLOCK_REDUCE_INT8

        if self.format_out is None:
            if format_a == "ND":
                self.format_out = "ND"
            else:
                self.format_out = "FRACTAL_NZ"

        if self.src_dtype == "int8" and self.dst_dtype == "int32":
            self.matrix_type = "int32"


    def _get_l1_shape(self):
        """ get shape about m,k,n
        Input: None
        ---------------------------------
        Return: None
        """

        # matrix A (M x K)
        self.batch_shape = 1
        if len(self.tensor_a.shape) in (3, 5):
            self.batch_shape = self.tensor_a.shape[0].value

        if self.format_a == "FRACTAL_NZ":
            # [batch, K, M, 16, 16]
            m_shape = self.tensor_a.shape[-3].value
            km_shape = self.tensor_a.shape[-4].value

        else:
            # [batch, M, K]
            m_shape = self._ceil_div(
                self.tensor_a.shape[-2].value, self.block_in)
            km_shape = self._ceil_div(
                self.tensor_a.shape[-1].value, self.block_reduce)

        self.m_shape = m_shape
        self.km_shape = km_shape


        # matrix B (K x N)
        if self.format_b == "FRACTAL_NZ":
            # [batch, N, K, 16, 16]
            kn_shape = self.tensor_b.shape[-3].value
            n_shape = self.tensor_b.shape[-4].value
        else:
            # [batch, K, N]
            kn_shape = self.tensor_b.shape[-2].value // self.block_reduce
            n_shape = self.tensor_b.shape[-1].value // self.block_out

        self.kn_shape = kn_shape
        self.n_shape = n_shape


    # -----------check func ----------- #
    def _check_km_kn(self):
        """
        check shape km and kn, should be equal
        Input: None
        ---------------------------------
        Return: None
        """
        km_shape = self.m_shape if self.trans_a else self.km_shape
        kn_shape = self.n_shape if self.trans_b else self.kn_shape

        if km_shape != kn_shape:
            args_dict = {
                "errCode": "E60002",
                "attr_name": "shape",
                "param1_name": "km_shape",
                "param1_value": "{}".format(km_shape),
                "param2_name": "kn_shape",
                "param2_value": "{}".format(kn_shape)
            }
            raise RuntimeError(
                args_dict, error_manager_util.get_error_message(args_dict)
            )
