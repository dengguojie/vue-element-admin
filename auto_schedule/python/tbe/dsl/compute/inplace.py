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
inplace
"""
from tbe import tvm
from tbe.common.testing.dsl_source_info import source_info_decorator
from .util import shape_to_list
from .util import dtype_check_decorator
from .util import get_tvm_scalar


@source_info_decorator()
@dtype_check_decorator
def inplace_add(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] += rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] += rhs; return lhs.
    """

    return _inplace_op(lhs, inplace_ids, rhs, ops="inplace_add")


@source_info_decorator()
@dtype_check_decorator
def inplace_sub(lhs, inplace_ids, rhs):
    """
    calculate inplace sub: computes lhs[inplace_ids, :] -= rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] -= rhs; return lhs.
    """

    return _inplace_op(lhs, inplace_ids, rhs, ops="inplace_sub")


@source_info_decorator()
@dtype_check_decorator
def inplace_update(lhs, inplace_ids, rhs):
    """
    calculate inplace add: computes lhs[inplace_ids, :] = rhs; return lhs.

    Parameters
    ----------
    lhs : wrapped_tensor or tvm.tensor
        left hand tensor

    inplace_ids : a vector. Indices into the left-most dimension of lhs.

    rhs : wrapped_tensor or tvm.tensor
        left hand tensor

    Returns
    -------
    wrapped_tensor : computes lhs[inplace_ids, :] = rhs; return lhs.
    """

    return _inplace_op(lhs, inplace_ids, rhs, ops="inplace_update")


def _inplace_compute(lhs, rhs, ops):
    """
    compute of inplace
    """
    if ops == "inplace_add":
        ret = lhs + rhs
    elif ops == "inplace_sub":
        ret = lhs - rhs
    elif ops == "inplace_update":
        from tbe.common.testing.testing import is_debug_mode
        if is_debug_mode():
            ret = lhs + rhs - lhs
        else:
            ret = lhs - rhs
    else:
        raise RuntimeError("operation %s not support yet" % ops)
    return ret


def _inplace_op(lhs, inplace_ids, rhs, ops):
    """
    factory method of inplace operations
    """
    if _is_tensor_scalar_inplace(lhs, inplace_ids, rhs):
        inplace_ids_list = [inplace_ids]
        scalar = get_tvm_scalar(rhs, lhs.dtype)
        rhs = tvm.compute((1,), lambda *indices: scalar.astype(lhs.dtype), name="reg_mov")
    elif isinstance(inplace_ids, int):
        _inplace_input_check(lhs, inplace_ids, rhs)
        inplace_ids_list = [inplace_ids]
        rhs_reshape = [1, ] + rhs.shape[:]
        rhs = tvm.compute(rhs_reshape, lambda *indices: rhs[indices[1:]],
                          name="reshapeComputeinlineOp")
    elif isinstance(inplace_ids, list):
        _inplace_input_check(lhs, inplace_ids, rhs)
        inplace_ids_list = inplace_ids
    else:
        raise RuntimeError("inplace_ids not support this type")

    def _inplace_lambda_func(indices):
        """
        compute_func of inplace operator
        """
        unique_id = []
        for i in inplace_ids_list:
            if i not in unique_id:
                unique_id.append(i)

        def __compute_outer_dim(i):
            inplace_ids_copy = list(inplace_ids_list)[:]
            if i in unique_id:
                idx = inplace_ids_copy.index(i)
                inplace_ids_copy[idx] = -1
                ret = _inplace_compute(lhs(*indices), rhs[(idx,) + indices[1:]], ops)
                for _ in range(inplace_ids_list.count(i) - 1):
                    idx = inplace_ids_copy.index(i)
                    inplace_ids_copy[idx] = -1
                    ret = _inplace_compute(ret, rhs[(idx,) + indices[1:]], ops)
                return ret
            return None

        res = lhs(*indices)
        for i in unique_id:
            res = tvm.select(indices[0] == i, __compute_outer_dim(i), res)
        return res

    lambda_func = lambda *indices: _inplace_lambda_func(indices)

    lshape = shape_to_list(lhs.shape)
    name = ops + '_' + lhs.name.split("_")[-1]
    str_inplace_ids = ",".join([str(i) for i in inplace_ids_list])
    with tvm.tag_scope(ops + "|" + str_inplace_ids):
        res = tvm.compute(lshape, lambda_func, name=name)

    return res


def _is_tensor_scalar_inplace(lhs, ids, rhs):
    if not isinstance(lhs, tvm.tensor.Tensor):
        raise RuntimeError("The lhTensor input type must be tvm.tensor")
    lshape = shape_to_list(lhs.shape)
    if len(lshape) == 1 and not isinstance(rhs, tvm.tensor.Tensor) and isinstance(ids, int):
        ret = True
    elif len(lshape) == 1 and \
            not isinstance(rhs, tvm.tensor.Tensor) and \
            not isinstance(ids, int):
        raise RuntimeError("inplace_ids must be int type!")
    else:
        ret = False

    return ret


# pylint: disable=too-many-branches
def _inplace_input_check(lhs, ids, rhs):
    if not (isinstance(lhs, tvm.tensor.Tensor) and isinstance(rhs, tvm.tensor.Tensor)):
        raise RuntimeError("The input type must be tvm.tensor")

    if lhs.dtype != rhs.dtype:
        raise RuntimeError("dtype must be the same, while lhs is %s, rhs is %s" % (
            lhs.dtype, rhs.dtype))

    lshape = shape_to_list(lhs.shape)
    rshape = shape_to_list(rhs.shape)
    # check shape:
    for shapei in lshape:
        if shapei <= 0 or not isinstance(shapei, int):
            raise RuntimeError("The lhs input shape value must be a positive integer")
    for shapei in rshape:
        if shapei <= 0 or not isinstance(shapei, int):
            raise RuntimeError("The rhs input shape value must be a positive integer")

    if isinstance(ids, int):
        if (len(lshape) - 1) != len(rshape):
            raise RuntimeError(
                "When inplace_ids is int, the ndim of lhs(1:) %d must be equal to the rhs %d"
                % (len(lshape) - 1, len(rshape)))
        if lshape[1:] != rshape[0:]:
            raise RuntimeError(
                "When inplace_ids is int, the lhs(1:) must be same dimension sizes as rhs")
        if ids >= lshape[0] or ids < 0:
            raise RuntimeError("the ids should be less than the rank of lhs's first dimension")
    elif isinstance(ids, list):
        if len(lshape) != len(rshape):
            raise RuntimeError(
                "The ndim of lhs %d must be equal to the rhs %d" % (len(lshape), len(rshape)))
        if lshape[1:] != rshape[1:]:
            raise RuntimeError(
                "The lhs must be same dimension sizes as rhs except the first dimension")
        for i in ids:
            if i < 0 or not isinstance(i, int):
                raise RuntimeError("The ids list must be a positive integer")
        # pylint: disable=len-as-condition
        # ids length check
        if len(ids) != rshape[0] or len(ids) < 1:
            raise RuntimeError("len(ids) should be equal to rhs.shape[0]")
        # max ids check
        if max(ids) >= lshape[0] or min(ids) < 0:
            raise RuntimeError(
                "the max of ids should be less than the rank of lhs's first dimension")
        _inplace_rhs_shape_check(rshape)
    else:
        raise RuntimeError("inplace_ids type not support!")


def _inplace_rhs_shape_check(rshape):
    if rshape[0] > (248*1024//32-2):
        raise RuntimeError(
            "inpalce rhTensor shape[0]=%s is larger than 7394, "
            "resulting in insufficient ub space" % rshape[0])
