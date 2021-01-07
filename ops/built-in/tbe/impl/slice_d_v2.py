#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2020 Huawei Technologies Co., Ltd
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
slice_d_v2
"""
# pylint: disable=too-many-statements,invalid-name,too-many-branches,unused-argument,too-many-locals
from types import MethodType
import math

from te.utils import para_check
from te import tik
from impl import common_util
from impl import constant_util as constant

from impl.dynamic.strided_slice import StridedSlice
from impl.dynamic.strided_slice import ceil_32bytes_align_count


# pylint: disable=undefined-variable
# pylint: disable=too-many-instance-attributes,useless-object-inheritance
# pylint: disable=too-many-arguments,unused-argument,unneeded-not,invalid-name
def _data_move(tik_instance: tik.Tik, dest: tik.Tensor, src: tik.Tensor, count):
    dtype_size = common_util.get_data_size(src.dtype)
    burst = math.ceil(count * dtype_size / constant.BLOCK_SIZE)
    tik_instance.data_move(dest, src, 0, 1, burst, 0, 0)


def _check_parameters(shape, size, kernel_name):
    """
    check the parameters including shape, dtype, begin, size and kernel_name

    """
    para_check.check_shape(shape, param_name="x")

    if not len(shape) == len(size):
        expected_value = "must be equal to shape!"
        real_value = "not equal to shape!"
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "length of shape and size",
                                                           expected_value, real_value)

    for i, (shape_i, size_i) in enumerate(zip(shape, size)):
        if not (isinstance(size[i], int) and -1 <= size_i <= shape_i
                and size_i != 0):
            expected_value = "greater than or equal to -1," \
                             "less than or equal to input shape, and cannot be equal to 0!"
            real_value = "greater than input shape, and equal to 0!"
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "value of size must be int",
                                                               expected_value, real_value)


# pylint: disable=locally-disabled,too-many-arguments
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.KERNEL_NAME)
def slice_d_v2(x, offsets, y, size, kernel_name="slice_d_v2"):
    """
    algorithm: slice
    calculating: this operation extracts a slice of size
                 from a tensor input
                 starting at the location specified by begin.

    Parameters
    ----------
    x: dict
        contains shape and dtype information of input tensor
    y: dict
        contains shape and dtype information of output tensor
    offsets: dict
        represents the index of the first value to select
    size: list or tuple
        represents the shape of output tensor
    kernel_name: str
        cce kernel name, default value is "slice_d_v2".

    Returns
    -------
    tik instance
    """
    if not isinstance(size, list):
        size = list(size)

    # dynamic slice does not use offsets, end params.
    def init_tiling(tiling_inst: StridedSlice.TilingParam):
        begin_dtype = offsets.get("dtype")

        def gen_shape(name, index):
            name += str(index)
            return tiling_inst.tik_instance.Scalar(begin_dtype, name=name)

        tiling_inst.begin = tuple(map(lambda x: gen_shape("begin_", x[0]), enumerate(tiling_inst.input_shape)))

        with tiling_inst.tik_instance.new_stmt_scope():
            need_ub_size = ceil_32bytes_align_count(len(tiling_inst.input_shape), tiling_inst.dtype)
            ub = tiling_inst.tik_instance.Tensor(begin_dtype, (need_ub_size,), name="begin_ub", scope=tik.scope_ubuf)
            _data_move(tiling_inst.tik_instance, ub, strided_slice_instance.begin_gm, need_ub_size)
            # set begin values
            for index, value in enumerate(tiling_inst.begin):
                value.set_as(ub[index])

        # set input shape
        input_shape = x.get("shape")

        # set end
        for index, value in enumerate(size):
            if value == -1:
                tiling_inst.end[index].set_as(input_shape[index])
                size[index] = input_shape[index]
            else:
                tiling_inst.end[index].set_as(tiling_inst.begin[index] + value)

        # set stride
        for index, value in enumerate(tiling_inst.stride):
            value.set_as(1)

        tmp_begin = tuple(map(lambda x: gen_shape("begin_", x[0]), enumerate(tiling_inst.input_shape)))
        for index, value in enumerate(tmp_begin):
            value.set_as(tiling_inst.begin[index])

        tmp_end = tuple(map(lambda x: gen_shape("end_", x[0]), enumerate(tiling_inst.end)))
        for index, value in enumerate(tmp_end):
            value.set_as(tiling_inst.end[index])

        def _make_perf_params(output_shape, input_shape, input_begin, input_end):
            last_same = False
            perf_size = 0
            perf_output_shape = []
            perf_input_shape = []
            perf_input_begin = []
            perf_input_end = []
            for i, _ in enumerate(input_shape):
                if input_shape[i] != output_shape[i]:
                    last_same = False
                    perf_output_shape.append(output_shape[i])
                    perf_input_shape.append(input_shape[i])
                    perf_input_begin.append(input_begin[i])
                    perf_input_end.append(input_end[i])
                    perf_size += 1
                    continue

                if not last_same:
                    last_same = True
                    perf_output_shape.append(output_shape[i])
                    perf_input_shape.append(input_shape[i])
                    perf_input_begin.append(input_begin[i])
                    perf_input_end.append(input_end[i])
                    perf_size += 1
                    continue

                index = perf_size - 1
                perf_output_shape[index] *= output_shape[i]
                perf_input_shape[index] *= input_shape[i]
                perf_input_begin[index] = 0
                perf_input_end[index] = perf_input_shape[index]

            if len(perf_input_shape) > 1 and perf_input_shape[-1] == perf_output_shape[-1]:
                index = -2
                perf_output_shape[index] *= perf_output_shape[-1]
                perf_input_shape[index] *= perf_input_shape[-1]
                perf_input_begin[index] *= perf_input_shape[-1]
                perf_input_end[index] *= perf_input_shape[-1]

                perf_output_shape.pop(-1)
                perf_input_shape.pop(-1)
                perf_input_begin.pop(-1)
                perf_input_end.pop(-1)

            if len(input_shape) > len(perf_input_shape):
                delta = len(input_shape) - len(perf_input_shape)
                output_shape = [1] * delta
                input_shape = [1] * delta
                input_begin = [0] * delta
                input_end = [1] * delta
                output_shape += perf_output_shape
                input_shape += perf_input_shape
                input_begin += perf_input_begin
                input_end += perf_input_end

            return output_shape, input_shape, input_begin, input_end

        perf_output_shape, perf_input_shape, perf_input_begin, perf_input_end = _make_perf_params(size,
                                                                                                  input_shape,
                                                                                                  tmp_begin,
                                                                                                  tmp_end
                                                                                                  )

        # set input shape
        for index, value in enumerate(perf_input_shape):
            tiling_inst.input_shape[index].set_as(value)

        # set begin
        for index, value in enumerate(perf_input_begin):
            tiling_inst.begin[index].set_as(value)

        # set end
        for index, value in enumerate(perf_input_end):
            tiling_inst.end[index].set_as(value)

        # set output shape
        for index, value in enumerate(perf_output_shape):
            tiling_inst.output_shape[index].set_as(value)

        # set out_dim
        tiling_inst.out_dim.set_as(1)
        for index, dim in enumerate(tiling_inst.output_shape):
            if index != len(tiling_inst.output_shape) - 1:
                tiling_inst.out_dim.set_as(tiling_inst.out_dim * dim)

    input_shape = x.get("shape")
    input_dtype = x.get("dtype").lower()
    check_list = ("float32", "float16", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64")
    para_check.check_dtype(input_dtype, check_list, param_name="x")
    strided_slice_instance = StridedSlice(x, None, 0, 0, 0, 0, 0, kernel_name)
    strided_slice_instance.begin_gm = strided_slice_instance.tik_instance.Tensor(offsets.get("dtype"),
                                                                                 offsets.get("shape"),
                                                                                 scope=tik.scope_gm,
                                                                                 name="begin_gm")

    strided_slice_instance.input_gm = strided_slice_instance.tik_instance.Tensor(x.get("dtype"),
                                                                                 x.get("shape"),
                                                                                 scope=tik.scope_gm,
                                                                                 name="input_gm")
    _check_parameters(input_shape, size, kernel_name)
    strided_slice_instance.tiling_param.init = MethodType(init_tiling, strided_slice_instance.tiling_param)
    strided_slice_instance.strided_slice()
    inst = strided_slice_instance.tik_instance
    opt_config = {"out_of_bound_sync_check": True}
    inst.BuildCCE(kernel_name=strided_slice_instance.kernel_name,
                  inputs=(strided_slice_instance.input_gm,
                          strided_slice_instance.begin_gm),
                  outputs=(strided_slice_instance.output_gm,),
                  config=opt_config,
                  enable_l2=False)
    return inst
