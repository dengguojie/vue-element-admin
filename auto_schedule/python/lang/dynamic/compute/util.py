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
compute util
"""
from decorator import decorator
from te import platform as cceconf
from te import tvm
from te.platform.cce_conf import CceProductParams as pver
from ...cce.te_compute import util as cce_util

DTYPE_MAP = {
    "float32": "f32",
    "float16": "f16",
    "int8": "s8",
    "uint8": "u8",
    "int32": "s32",
}

DSL_CHECK_SUPPORT_MAP = cce_util.DSL_CHECK_SUPPORT_MAP


def astype(scalar, dtype):
    """
    :param scalar:
    :param dtype:
    :return:
    """
    if isinstance(scalar, int):
        return tvm.const(scalar, "int").astype(dtype)
    if isinstance(scalar, float):
        return tvm.const(scalar, "float").astype(dtype)
    if isinstance(scalar, (tvm.expr.IntImm, tvm.expr.UIntImm,
                           tvm.expr.FloatImm)):
        return scalar.astype(dtype)
    if isinstance(scalar, tvm.expr.Var):
        return scalar.astype(dtype)
    if isinstance(scalar, tvm.tensor.TensorSlice):
        return scalar
    raise RuntimeError(
        "Scalar must be simple type, but now is {0}".format(type(scalar)))


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        if isinstance(i, tvm.expr.ConstExpr):
            tmp.append(i.value)
        else:
            tmp.append(i)
    return tmp


def check_input_tensor_shape(tensor_shape):
    """
    check_tensor_shape
    """
    shape = tensor_shape
    if isinstance(tensor_shape, tvm.tensor.Tensor):
        shape = shape_to_list(tensor_shape.shape)

    for val in shape:
        if isinstance(val, int) and val <= 0:
            raise RuntimeError(
                "The input shape value must be a positive integer")


def is_cast_support(src_type, dst_type):
    """
    is_cast_support
    """
    if src_type not in DTYPE_MAP:
        raise RuntimeError("%s is unsupported dtype!" % src_type)

    if dst_type not in DTYPE_MAP:
        raise RuntimeError("%s is unsupported dtype!" % dst_type)

    if src_type == dst_type:
        return True

    cast_type = DTYPE_MAP[src_type] + "2" + DTYPE_MAP[dst_type]

    if cast_type == "s322f16":
        cast_type = "deq"

    return cceconf.intrinsic_check_support("Intrinsic_vconv", cast_type)


def judge_var(num):
    """
    judge var if a tvm.var, tvm.const or python data type
    """
    var_dict = {
        "python_const": [int, float],
        "tvm_const": [tvm.expr.IntImm, tvm.expr.UIntImm, tvm.expr.FloatImm],
        "tvm_var": [tvm.expr.Var]
    }
    num_type = type(num)
    for i in var_dict:
        if num_type in var_dict[i]:
            return i
    raise RuntimeError("Input var Error")


def equal(_a, _b):
    """
    :param _a:
    :param _b:
    :return:
    """
    elements1 = {}
    elements2 = {}

    single_types = (int, tvm.expr.Var)
    const_types = (tvm.expr.IntImm,)
    for expr, elements in zip((_a, _b), (elements1, elements2)):
        if isinstance(expr, single_types):
            elements[expr] = elements.get(expr, 0) + 1
        elif isinstance(expr, const_types):
            elements[expr.value] = elements.get(expr.value, 0) + 1
        elif isinstance(expr, tvm.expr.Expr):
            _parse_expr(expr, elements)
        else:
            raise RuntimeError("Unsupported expr: {0}".format(expr))

    return elements1 == elements2


def _parse_expr(expr, elements: dict):
    if isinstance(expr, tvm.expr.Mul):
        _parse_mul(expr, elements)
    else:
        raise RuntimeError("Unsupported expr: {0}".format(expr))


def _parse_mul(expr, elements: dict):
    if not isinstance(expr, tvm.expr.Mul):
        raise RuntimeError("It is not mul expr: {}".format(expr))

    const_types = (tvm.expr.IntImm,)
    var_types = (tvm.expr.Var,)
    for _x in (expr.a, expr.b):
        if isinstance(_x, const_types):
            elements[_x.value] = elements.get(_x.value, 0) + 1
        elif isinstance(_x, var_types):
            elements[_x] = elements.get(_x, 0) + 1
        else:
            _parse_mul(_x, elements)


@decorator
def dtype_check_decorator(func, *args, **kwargs):
    """
    dtype_check_decorator
    """
    func_name = func.__name__
    if func_name == "broadcast":
        if isinstance(args[0], int):
            judge_dtype = "int32"
        elif isinstance(args[0], float):
            judge_dtype = "float16"
        else:
            judge_dtype = args[0].dtype
    elif func_name == "concat":
        if not isinstance(args[0], list):
            raise RuntimeError("The first input type must be list")
        if not isinstance(args[0][0], tvm.tensor.Tensor):
            raise RuntimeError(
                "The first input type must be list of tvm.tensor")
        judge_dtype = args[0][0].dtype
    else:
        if not isinstance(args[0], tvm.tensor.Tensor):
            raise RuntimeError("The first input type must be tvm.tensor")
        judge_dtype = args[0].dtype

    if not dsl_check_support("te.lang.dynamic." + func_name, judge_dtype):
        raise RuntimeError("te.lang.dynamic.%s is not supported %s!"
                           % (func_name, judge_dtype))

    return func(*args, **kwargs)


def dsl_check_support(dsl_api, dtype=None):
    """
    dsl_check_support
    """
    if not dsl_api.startswith("te.lang.dynamic."):
        return False
    if (dtype is not None) and (not isinstance(dtype, str)):
        return False

    dsl_name = dsl_api.split("te.lang.dynamic.")[-1]
    if dsl_name in ("reduce_sum", "sum"):
        dsl_name = "reduce_sum"

    all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    if all_support_dtype is None:
        return False

    soc_ver = pver().get_product_version()
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return False

    if (dtype not in (None, "")) and (dtype not in soc_support_dtype):
        return False

    return True


def refine_axis(axis, shape):
    """
    refine_axis
    """
    if isinstance(axis, (tuple, list)):
        local_axis = axis
    else:
        local_axis = [axis]
    res_axis = []
    shape_len = len(shape)
    for i in local_axis:
        if i < 0:
            laxis = shape_len + i
        else:
            laxis = i
        if (laxis >= shape_len) or (laxis < 0):
            raise RuntimeError("wrong axis.")
        res_axis.append(laxis)
    return sorted(res_axis)


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if not isinstance(value, int):
        raise RuntimeError("type of axis value should be int")
    if value >= shape_len or value < -shape_len:
        raise RuntimeError(
            "input axis is out of range, axis value can be from %d to %d" %
            (-shape_len, shape_len - 1))
    if value < 0:
        value = shape_len + value
    return value


def reduce_axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    axis = list(axis)
    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(shape_len, axis)
        return axis
    for i, _x in enumerate(axis):
        axis[i] = _axis_value_type_check(shape_len, _x)

    axis = list(set(axis))
    axis.sort()
    return axis


def auto_cast_tensor(tensor, intr, supported_types=None):
    """
    auto_cast_tensor
    """
    from .cast_compute import _cast
    if isinstance(tensor, tvm.tensor.Tensor):
        dtype = tensor.dtype
        if supported_types is None:
            intrinsic = "Intrinsic_" + intr
            intr_is_support_dtype = cceconf.intrinsic_check_support(intrinsic,
                                                                    dtype)
            intr_is_support_fp32 = cceconf.intrinsic_check_support(intrinsic,
                                                                   "float32")
        else:
            intr_is_support_dtype = (dtype in supported_types)
            intr_is_support_fp32 = ("float32" in supported_types)

        if not intr_is_support_dtype:
            if intr_is_support_fp32 and is_cast_support(dtype, "float32"):
                tensor = _cast(tensor, "float32")
            else:
                tensor = _cast(tensor, "float16")

    return tensor


def dsl_support_dtype(dsl_name):
    """
    dsl_support_dtype
    """
    if not isinstance(dsl_name, str):
        return []

    if dsl_name in ("reduce_sum", "sum"):
        dsl_name = "reduce_sum"

    all_support_dtype = DSL_CHECK_SUPPORT_MAP.get(dsl_name)
    if all_support_dtype is None:
        return []

    soc_ver = pver().get_product_version()
    soc_support_dtype = all_support_dtype.get(soc_ver)
    if soc_support_dtype is None:
        soc_support_dtype = all_support_dtype.get("AllSoc")
        if soc_support_dtype is None:
            return []

    return list(soc_support_dtype)


def get_tvm_scalar(scalar, dtype):
    """
    get_tvm_scalar
    """
    scalar_type = judge_var(scalar)
    if scalar_type == "tvm_const" and scalar.dtype != dtype:
        scalar = tvm.const(scalar.value, dtype=dtype)

    if scalar_type == "python_const":
        scalar = tvm.const(scalar, dtype=dtype)

    return scalar


def get_vcmpsel_input_type(rhs, slhs, srhs):
    type_strs = []
    if isinstance(rhs, tvm.tensor.Tensor):
        type_strs.append("TENSOR")
    else:
        type_strs.append("SCALAR")

    if isinstance(slhs, tvm.tensor.Tensor):
        type_strs.append("TENSOR")
    else:
        type_strs.append("SCALAR")

    if isinstance(srhs, tvm.tensor.Tensor):
        type_strs.append("TENSOR")
    else:
        type_strs.append("SCALAR")

    return "_".join(type_strs)
