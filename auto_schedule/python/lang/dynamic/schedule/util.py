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
schedule util
"""
from typing import List
from functools import reduce
from operator import mul

from te.platform.cce_conf import CceProductParams as product_params
from te import platform as cce
from te import tvm
from te.tvm.expr import Var
from te.tvm.expr import Reduce
from te.tvm.tensor import Tensor
from te.tvm.tensor import PlaceholderOp
from te.lang.base import operation
from te.utils.error_manager.error_manager_util import get_error_message

from . import BROADCAST_INSNS, SUPPORT_SCALAR_INSNS, \
    NEED_TEMP_SPACE_INSNS, VSEL_INSNS, VCMPSEL_INSNS, NEED_SPACE_WITH_DIFF_TYPE, NEED_EXTENT_NODE_INSNS

VAR_BOUND_LIMIT = 2147483647


def shape_to_list(shape):
    """
    :param shape:
    :return:
    """
    shape0 = []
    for i in shape:
        if isinstance(i, tvm.expr.ConstExpr):
            shape0.append(i.value)
        else:
            shape0.append(i)
    return shape0


def get_dsl_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    tag = tensor.op.tag
    if tensor.op.tag.find("|") != -1:
        insn = tag.split("|")[0]
    else:
        insn = tag
    return insn


def support_scalar(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in SUPPORT_SCALAR_INSNS


def is_v100():
    """
    :return:
    """
    return product_params().is_mini_version() or product_params().is_cloud_version()


def need_temp_space(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    op_tag = get_dsl_insn(tensor)
    return op_tag in NEED_TEMP_SPACE_INSNS or \
           (is_v100() and op_tag in NEED_SPACE_WITH_DIFF_TYPE and tensor.dtype == "int32")


def need_extent_node(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in NEED_EXTENT_NODE_INSNS


def is_vsel_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) == VSEL_INSNS


def is_vcmpsel_insn(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in VCMPSEL_INSNS


def get_tensor_size(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    shape = shape_to_list(tensor.shape)
    if all(isinstance(i, int) for i in shape):
        return reduce(mul, shape_to_list(tensor.shape), 1)
    return -1


def is_vtranspose_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    if not is_broadcast(tensor) or len(tensor.op.input_tensors) != 1:
        return False
    return tensor.dtype == "float16" and not expr_equal(tensor.shape[-1], tensor.op.input_tensors[0].shape[-1])


def is_broadcast(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return get_dsl_insn(tensor) in BROADCAST_INSNS


def is_placeholder(tensor: tvm.tensor.Tensor):
    """
    :param tensor:
    :return:
    """
    return isinstance(tensor.op, tvm.tensor.PlaceholderOp)


def merge_value(map_: dict, key, value):
    """
    :param map_:
    :param key:
    :param value: value container is set
    :return:
    """
    if key not in map_:
        map_[key] = set()
    if isinstance(value, list):
        map_[key].update(value)
    else:
        map_[key].add(value)


def get_bound(expr):
    """
    :param expr:
    :return:
    """
    valid_types = (int, tvm.expr.Expr)
    if not isinstance(expr, valid_types):
        dict_args = dict()
        dict_args["errCode"] = "E90001"
        dict_args["detailed_cause"] = "Only accept (int, expr), but now " \
                                      "is [%s]." % type(expr)
        raise RuntimeError(dict_args, get_error_message(dict_args))

    if isinstance(expr, int):
        return expr, expr
    if isinstance(expr, tvm.expr.IntImm):
        return expr.value, expr.value
    if isinstance(expr, tvm.expr.Var):
        return operation.get_te_var(expr.name).get_bound()

    def _parse(expr_, elements_: list):
        if not isinstance(expr_, tvm.expr.Mul):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Only accept (int, expr), but now " \
                                          "is [%s]." % type(expr_)
            raise RuntimeError(dict_args, get_error_message(dict_args))

        single_types = (tvm.expr.IntImm, tvm.expr.Var)
        for _x in (expr_.a, expr_.b):
            if isinstance(_x, single_types):
                elements_.append(_x)
            else:
                _parse(_x, elements_)

    def _mul(_a, _b):
        if _a is None or _b is None:
            return None
        _bound = _a * _b
        return None if _bound > VAR_BOUND_LIMIT else _bound

    lower, upper = 1, 1
    elements = []
    _parse(expr, elements)
    for _e in elements:
        if isinstance(_e, tvm.expr.IntImm):
            lower, upper = _mul(lower, _e.value), _mul(upper, _e.value)
        elif isinstance(_e, tvm.expr.Var):
            bound = operation.get_te_var(_e.name).get_bound()
            lower, upper = _mul(lower, bound[0]), _mul(upper, bound[1])
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Only accept (int, expr), but now " \
                                          "is [%s]." % type(_e)
            raise RuntimeError(dict_args, get_error_message(dict_args))

    return lower, upper


def get_ub_size():
    """
    :return:
    """
    return cce.get_soc_spec("UB_SIZE")


def get_core_num():
    """
    :return:
    """
    return cce.get_soc_spec("CORE_NUM")


def get_build_cfg():
    """
    :return:
    """
    f_m = cce.fusion_manager.fusion_manager
    return f_m.get_build_cfg()


def equals_one(_x):
    """
    :param _x:
    :return:
    """
    if isinstance(_x, tvm.expr.ConstExpr):
        return _x.value == 1
    if isinstance(_x, int):
        return _x == 1
    return False


def ceil_div(num, factor):
    """
    :param num:
    :param factor:
    :return:
    """
    return (num + factor - 1) // factor


def ceil_align(num, factor):
    """
    :param num:
    :param factor:
    :return:
    """
    return ceil_div(num, factor) * factor


def add_sch_additional_entry(sch, k, v):
    """
    :param sch:
    :param k:
    :param v:
    :return:
    """
    if not hasattr(sch, "addition"):
        sch.addition = {}
    sch.addition[k] = v


def get_sch_additional_entry(sch, k):
    """
    :param sch:
    :param k:
    :return:
    """
    if not hasattr(sch, "addition"):
        return None
    return sch.addition.get(k)


def is_reduce_tensor(tensor: Tensor) -> bool:
    """Check if tensor contains reduce body"""
    if isinstance(tensor.op, PlaceholderOp):
        return False
    if isinstance(tensor.op.body[0], Reduce):
        return True
    return False


def get_reduce_axes(reduce_tensor: Tensor) -> List[Var]:
    """Get reduce axes var of reduce tensor"""
    if not is_reduce_tensor(reduce_tensor):
        raise RuntimeError("Cannot get reduce axes of non-reduce tensor!")
    reduce_tensor_body = reduce_tensor.op.body
    reduce_tensor_axes = list(reduce_tensor_body[0].axis)
    for idx, axis in enumerate(reduce_tensor_axes):
        reduce_tensor_axes[idx] = axis.var
    return reduce_tensor_axes


def get_reduce_all_axes(reduce_tensor: Tensor) -> List[Var]:
    """Get all axes var for reduce tensor"""
    reduce_tensor_body = reduce_tensor.op.body
    return list(reduce_tensor_body[0].source[0].args)


def get_reduce_axis_indices(reduce_tensor: Tensor) -> List[int]:
    """Get all reduce axis index"""
    return [get_reduce_all_axes(reduce_tensor).index(axis) for axis in get_reduce_axes(reduce_tensor)]


def is_keepdims(reduce_tensor: Tensor) -> bool:
    """Check if reduce tensor is keepdims"""
    return len(reduce_tensor.shape) == len(get_reduce_all_axes(reduce_tensor))


def expr_equal(expr_a, expr_b):
    """
    :param expr_a: The first expr
    :param expr_b: The second expr
    :return: bool, compare result
    """
    def _parse_expr(_expr, _elements: dict):
        if isinstance(_expr, tvm.expr.Mul):
            _parse_mul(_expr, _elements)
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Expr parse: unsupported expr: [%s]" % _expr
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def _parse_mul(_expr, _elements: dict):
        if not isinstance(_expr, tvm.expr.Mul):
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Expr parse: it is not mul expr: [%s]" % _expr
            raise RuntimeError(dict_args, get_error_message(dict_args))

        var_types = (tvm.expr.Var,)
        for _x in (_expr.a, _expr.b):
            if isinstance(_x, const_types):
                _elements[_x.value] = _elements.get(_x.value, 0) + 1
            elif isinstance(_x, var_types):
                _elements[_x] = _elements.get(_x, 0) + 1
            else:
                _parse_mul(_x, _elements)

    elements1 = {}
    elements2 = {}
    single_types = (int, float, tvm.expr.Var)
    const_types = (tvm.expr.IntImm,)
    for expr, elements in zip((expr_a, expr_b), (elements1, elements2)):
        if isinstance(expr, single_types):
            elements[expr] = elements.get(expr, 0) + 1
        elif isinstance(expr, const_types):
            elements[expr.value] = elements.get(expr.value, 0) + 1
        elif isinstance(expr, tvm.expr.Expr):
            _parse_expr(expr, elements)
        else:
            dict_args = dict()
            dict_args["errCode"] = "E90001"
            dict_args["detailed_cause"] = "Expr compare: unsupported expr: [%s]" % expr
            raise RuntimeError(dict_args, get_error_message(dict_args))

    return elements1 == elements2
