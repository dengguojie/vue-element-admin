#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

is_finite
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl import common_util


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    # ub size count
    UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
    # aicore count
    CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
    # byte count one block
    BLOCK_BYTE_COUNT = 32
    # repeat up limit for mte
    REPEAT_LIMIT = 255
    # max int64 value
    MAX_INT64_VALUE = 2**64 - 1
    # parameters for moving tiling data
    TILING_CTRL_PARAM = ("int64", 64, 4)
    # the number that fp16 data should to vand with(Binary 11111)
    FLAG_FP16 = 31
    # the number that fp32 data should to vand with(Binary 11111111)
    FLAG_FP32 = 255
    # the bits of fp16 shift right
    SHR_FP16_NUM = 10
    # the bits of fp32 shift right
    SHR_FP32_NUM = 23
    # the dtype that fp16 should to reinterpret_cast_to before shift right
    SHR_FP16_DTYPE = "uint16"
    # the dtype that fp32 should to reinterpret_cast_to before shift right
    SHR_FP32_DTYPE = "uint32"
    # the dtype that fp16 vand result should to reinterpret_cast_to
    VAND_RESULT_FP16_DTYPE = "int16"
    # the dtype that fp32 vand result should to reinterpret_cast_to
    VAND_RESULT_FP32_DTYPE = "int32"
    # the finite result need to be converted to 1
    BOOL_FLAG_TRUE = 1
    # the constants related to input dtype
    CONSTANT_MAP = {
        "float16": {
            "shr_dtype": SHR_FP16_DTYPE,
            "shr_num": SHR_FP16_NUM,
            "shr_flag": FLAG_FP16,
            "vand_result_dtype": VAND_RESULT_FP16_DTYPE,
            "flag_tensor_multiple": 1
        },
        "float32": {
            "shr_dtype": SHR_FP32_DTYPE,
            "shr_num": SHR_FP32_NUM,
            "shr_flag": FLAG_FP32,
            "vand_result_dtype": VAND_RESULT_FP32_DTYPE,
            "flag_tensor_multiple": 2
        }
    }


# 'pylint: disable=invalid-name
def _ceil_div(value_x, value_y):
    """
    do ceil division
    """
    return (value_x + value_y - 1) // value_y


def _get_element_cnt_one_block(dtype):
    """
    get element count in a block
    """
    byte_len = common_util.get_data_size(dtype)
    element_cnt = Constant.BLOCK_BYTE_COUNT // byte_len

    return element_cnt


def _get_max_element_in_ub(dtype, ub_part):
    """
    get the up limit elements in UB
    """
    byte_len = common_util.get_data_size(dtype)

    ub_upper_limit = ((Constant.UB_SIZE - 2 * 1024) // 2) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def _check_input_params(input_dtype, output_dtype):
    """
    check whether the input parameters is valid or not
    """
    para_check.check_dtype(input_dtype, ("float16", "float32"), param_name="x")
    para_check.check_dtype(output_dtype, ("int8",), param_name="y")


def _get_ub_max_size(input_dtype, output_dtype):
    """
    output 32 bytes align
    """
    ub_x_size = _get_max_element_in_ub(input_dtype, 1) // 2
    return ub_x_size - ub_x_size % (Constant.BLOCK_BYTE_COUNT // common_util.get_data_size(output_dtype))


# 'pylint:disable=too-many-arguments
def _scalar_vector_func(tik_inst, vec_func, dst, src, scalar, data_len, data_type):
    """
    do scalar vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    if not isinstance(data_len, int):
        repeat = tik_inst.Scalar("int64", "scalar_vector_func_repeat")
        repeat.set_as(data_len // repeat_data_num)
    else:
        repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    if not isinstance(repeat, int) or repeat != 0:
        with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
            with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
                vec_func(repeat_data_num, dst[offset], src[offset], scalar, Constant.REPEAT_LIMIT, 1, 1, 8, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    if not isinstance(left_repeat, int) or left_repeat != 0:
        with tik_inst.if_scope(left_repeat > 0):
            offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src[offset], scalar, left_repeat, 1, 1, 8, 8)
    if not isinstance(repeat_tail, int) or repeat_tail != 0:
        with tik_inst.if_scope(repeat_tail > 0):
            offset = repeat * repeat_data_num
            vec_func(repeat_tail, dst[offset], src[offset], scalar, 1, 1, 1, 8, 8)


# 'pylint:disable=too-many-arguments
def _vector_single_src_func(tik_inst, vec_func, dst, src, data_len, data_type):
    """
    do vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    if not isinstance(data_len, int):
        repeat = tik_inst.Scalar("int64", "vector_func_repeat")
        repeat.set_as(data_len // repeat_data_num)
    else:
        repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    if not isinstance(repeat, int) or repeat != 0:
        with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
            with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
                vec_func(repeat_data_num, dst[offset], src[offset], Constant.REPEAT_LIMIT, 1, 1, 8, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    if not isinstance(left_repeat, int) or left_repeat != 0:
        with tik_inst.if_scope(left_repeat > 0):
            offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src[offset], left_repeat, 1, 1, 8, 8)
    if not isinstance(repeat_tail, int) or repeat_tail != 0:
        with tik_inst.if_scope(repeat_tail > 0):
            offset = repeat * repeat_data_num
            vec_func(repeat_tail, dst[offset], src[offset], 1, 1, 1, 8, 8)


# 'pylint:disable=too-many-arguments
def _vector_double_src_func(tik_inst, vec_func, dst, src1, src2, data_len, data_type):
    """
    do vector operator
    """
    data_one_block = _get_element_cnt_one_block(data_type)
    repeat_data_num = 8 * data_one_block
    if not isinstance(data_len, int):
        repeat = tik_inst.Scalar("int64", "vector_func_repeat")
        repeat.set_as(data_len // repeat_data_num)
    else:
        repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    if not isinstance(repeat, int) or repeat != 0:
        with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
            with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
                vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], Constant.REPEAT_LIMIT, 1, 1, 1, 8, 8,
                         8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    if not isinstance(left_repeat, int) or left_repeat != 0:
        with tik_inst.if_scope(left_repeat > 0):
            offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            vec_func(repeat_data_num, dst[offset], src1[offset], src2[offset], left_repeat, 1, 1, 1, 8, 8, 8)
    if not isinstance(repeat_tail, int) or repeat_tail != 0:
        with tik_inst.if_scope(repeat_tail > 0):
            offset = repeat * repeat_data_num
            vec_func(repeat_tail, dst[offset], src1[offset], src2[offset], 1, 1, 1, 1, 8, 8, 8)


def _vector_dup_func(tik_inst, dst, scalar, data_len):
    """
    do vec_dup
    """
    data_one_block = _get_element_cnt_one_block(dst.dtype)
    repeat_data_num = 8 * data_one_block
    if not isinstance(data_len, int):
        repeat = tik_inst.Scalar("int32", "vector_dup_repeat")
        repeat.set_as(data_len // repeat_data_num)
    else:
        repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT

    if not isinstance(repeat, int) or repeat != 0:
        with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
            with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
                tik_inst.vec_dup(repeat_data_num, dst[offset], scalar, Constant.REPEAT_LIMIT, 8)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    if not isinstance(left_repeat, int) or left_repeat != 0:
        with tik_inst.if_scope(left_repeat > 0):
            offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            tik_inst.vec_dup(repeat_data_num, dst[offset], scalar, left_repeat, 8)
    if not isinstance(repeat_tail, int) or repeat_tail != 0:
        with tik_inst.if_scope(repeat_tail > 0):
            offset = repeat * repeat_data_num
            tik_inst.vec_dup(repeat_tail, dst[offset], scalar, 1, 8)


# 'pylint:disable=too-many-locals
def _vconv_func(tik_inst, dst, src, round_mode, data_len):
    dst_data_one_block = _get_element_cnt_one_block(dst.dtype)
    src_data_one_block = _get_element_cnt_one_block(src.dtype)
    data_one_block = dst_data_one_block if dst_data_one_block <= src_data_one_block else src_data_one_block
    repeat_data_num = 8 * data_one_block
    dst_rep_stride = repeat_data_num // dst_data_one_block
    src_rep_stride = repeat_data_num // src_data_one_block
    if not isinstance(data_len, int):
        repeat = tik_inst.Scalar("int32", "vconv_repeat")
        repeat.set_as(data_len // repeat_data_num)
    else:
        repeat = data_len // repeat_data_num
    repeat_tail = data_len % repeat_data_num
    loop_repeat_cnt = repeat // Constant.REPEAT_LIMIT
    deq_scalar = 1.0 if src.dtype == "int32" and dst.dtype == "float16" else None

    if not isinstance(repeat, int) or repeat != 0:
        with tik_inst.if_scope(repeat >= Constant.REPEAT_LIMIT):
            with tik_inst.for_range(0, loop_repeat_cnt) as repeat_lp_cnt:
                offset = repeat_lp_cnt * Constant.REPEAT_LIMIT * repeat_data_num
                tik_inst.vconv(repeat_data_num, round_mode, dst[offset], src[offset], Constant.REPEAT_LIMIT, 1, 1,
                               dst_rep_stride, src_rep_stride, deq_scalar)
    left_repeat = repeat - loop_repeat_cnt * Constant.REPEAT_LIMIT
    if not isinstance(left_repeat, int) or left_repeat != 0:
        with tik_inst.if_scope(left_repeat > 0):
            offset = loop_repeat_cnt * Constant.REPEAT_LIMIT * repeat_data_num
            tik_inst.vconv(repeat_data_num, round_mode, dst[offset], src[offset], left_repeat, 1, 1, dst_rep_stride,
                           src_rep_stride, deq_scalar)
    if not isinstance(repeat_tail, int) or repeat_tail != 0:
        with tik_inst.if_scope(repeat_tail > 0):
            offset = repeat * repeat_data_num
            tik_inst.vconv(repeat_tail, round_mode, dst[offset], src[offset], 1, 1, 1, dst_rep_stride, src_rep_stride,
                           deq_scalar)


def _data_move(tik_inst, dst, src, data_len):
    element_one_block = _get_element_cnt_one_block(src.dtype)
    tik_inst.data_move(dst, src, 0, 1, _ceil_div(data_len, element_one_block), 0, 0)


class IsFinite:
    """
    is_finite: Determine whether data of tensor is a finite number
    calculation process:
    1. Shift the exponent bit data right to the low bit
        (1). use reinterpret_cast_to to convert the data to uint
        (2). vshr(data, shr_num)
    2. Use vand to compare data and flags
        (1). vand(data, flag)
    3. Convert the comparison result to bool
        (1). data = data - flag
        (2). data = abs(data)
        (3). data = vmins(data, 1)
    """

    def __init__(self, input_x, output_y, kernel_name="is_finite"):
        self.tik_inst = tik.Tik()
        self.input_ub = None
        self.cache_ub = None
        self._init_inner_params(input_x, output_y, kernel_name)
        self._init_gm()
        self._init_tiling_params()

    def _init_inner_params(self, input_x, output_y, kernel_name):
        self.kernel_name = kernel_name
        self.input_dtype = input_x.get("dtype").lower()
        self.output_dtype = output_y.get("dtype").lower()

        if self.output_dtype == "bool":
            self.output_dtype = "int8"

        _check_input_params(self.input_dtype, self.output_dtype)

        constant_map = Constant.CONSTANT_MAP.get(self.input_dtype)
        self.shr_dtype = constant_map.get("shr_dtype")
        self.shr_num = constant_map.get("shr_num")
        self.shr_flag = constant_map.get("shr_flag")
        self.vand_result_dtype = constant_map.get("vand_result_dtype")
        self.flag_tensor_multiple = constant_map.get("flag_tensor_multiple")
        self.per_loop_size = _get_ub_max_size(self.input_dtype, self.output_dtype)

    def _init_gm(self):
        self.data_input = self.tik_inst.Tensor(self.input_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm,
                                               "data_input")
        self.data_out = self.tik_inst.Tensor(self.output_dtype, (Constant.MAX_INT64_VALUE,), tik.scope_gm, "data_out")
        self.data_tiling = self.tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                                tik.scope_gm, "data_tiling")

    def _init_ub(self):
        ub_max_size = _get_max_element_in_ub(self.input_dtype, 2)
        self.input_ub = self.tik_inst.Tensor(self.input_dtype, (ub_max_size,), tik.scope_ubuf, "input_ub")
        self.cache_ub = self.tik_inst.Tensor(self.input_dtype, (ub_max_size,), tik.scope_ubuf, "cache_ub")

    def _init_tiling_params(self):
        self.ub_tiling = self.tik_inst.Tensor(Constant.TILING_CTRL_PARAM[0], (Constant.TILING_CTRL_PARAM[1],),
                                              tik.scope_ubuf, "ub_tiling")
        self.need_core_num = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "need_core_num")
        self.total_element_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "total_element_size")
        self.per_core_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_core_size")
        self.core_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_size")
        self.core_loop_cnt = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_loop_cnt")
        self.core_left_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "core_left_size")
        self.real_per_loop_size = self.tik_inst.Scalar(Constant.TILING_CTRL_PARAM[0], "per_loop_size")

    def _get_tiling_params(self, block_idx):
        _data_move(self.tik_inst, self.ub_tiling, self.data_tiling, Constant.TILING_CTRL_PARAM[1])
        self.need_core_num.set_as(self.ub_tiling[0])
        self.total_element_size.set_as(self.ub_tiling[1])
        self.per_core_size.set_as(self.ub_tiling[2])

        with self.tik_inst.if_scope(tik.all(block_idx == self.need_core_num - 1, self.ub_tiling[5] != 0)):
            self.core_size.set_as(self.ub_tiling[5])
            self.core_loop_cnt.set_as(self.ub_tiling[6])
            self.core_left_size.set_as(self.ub_tiling[7])
        with self.tik_inst.else_scope():
            self.core_size.set_as(self.ub_tiling[2])
            self.core_loop_cnt.set_as(self.ub_tiling[3])
            self.core_left_size.set_as(self.ub_tiling[4])

        with self.tik_inst.if_scope(self.core_loop_cnt == 0):
            self.real_per_loop_size.set_as(0)
        with self.tik_inst.else_scope():
            self.real_per_loop_size.set_as((self.core_size - self.core_left_size) // self.core_loop_cnt)

    def build(self):
        """
        build cce
        """
        opt_config = {"out_of_bound_sync_check": True, "enable_const_fold": True}
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=[self.data_input],
                               outputs=[self.data_out],
                               flowtable=[self.data_tiling],
                               config=opt_config)
        return self.tik_inst

    def compute(self):
        """
        do is_finite
        """
        with self.tik_inst.for_range(0, Constant.CORE_NUM, block_num=Constant.CORE_NUM) as block_idx:
            self._get_tiling_params(block_idx)
            core_offset = block_idx * self.per_core_size
            self._schedule(core_offset, self.core_loop_cnt, self.core_left_size)

    def _inner_compute(self, offset, element_size):
        self._data_move_in(offset, element_size)
        self._prepare_data_input(element_size)
        self._get_exponent_data(element_size)
        self._binarize_result(element_size)
        self._data_move_out(offset, element_size)

    def _schedule(self, core_offset, core_loop_cnt, core_left_size):
        with self.tik_inst.if_scope(core_loop_cnt > 0):
            with self.tik_inst.for_range(0, core_loop_cnt, thread_num=2) as lp_cnt:
                self._init_ub()
                lp_offset = core_offset + lp_cnt * self.per_loop_size
                self._inner_compute(lp_offset, self.per_loop_size)
        with self.tik_inst.if_scope(core_left_size > 0):
            self._init_ub()
            offset = core_offset + core_loop_cnt * self.real_per_loop_size
            self._inner_compute(offset, core_left_size)

    def _prepare_data_input(self, element_size):
        input_ub = self.input_ub.reinterpret_cast_to(self.shr_dtype)
        # cache reuse
        shr_ub = self.cache_ub.reinterpret_cast_to(self.shr_dtype)
        _scalar_vector_func(self.tik_inst, self.tik_inst.vshr, shr_ub, input_ub, self.shr_num, element_size,
                            self.shr_dtype)

    def _get_exponent_data(self, element_size):
        shr_ub = self.cache_ub.reinterpret_cast_to("uint16")
        # cache reuse
        flag_tensor_ub = self.input_ub.reinterpret_cast_to("uint16")
        flag_tensor_size = element_size * self.flag_tensor_multiple
        _vector_dup_func(self.tik_inst, flag_tensor_ub, self.shr_flag, flag_tensor_size)
        _vector_double_src_func(self.tik_inst, self.tik_inst.vand, shr_ub, shr_ub, flag_tensor_ub, flag_tensor_size,
                                "uint16")

    def _binarize_result(self, element_size):
        vand_result = self.cache_ub.reinterpret_cast_to(self.vand_result_dtype)
        # cache reuse
        vconv_result = self.input_ub.reinterpret_cast_to("float16")
        _vconv_func(self.tik_inst, vconv_result, vand_result, "none", element_size)
        _scalar_vector_func(self.tik_inst, self.tik_inst.vadds, vconv_result, vconv_result, -self.shr_flag,
                            element_size, vconv_result.dtype)
        _vector_single_src_func(self.tik_inst, self.tik_inst.vabs, vconv_result, vconv_result, element_size,
                                vconv_result.dtype)
        _scalar_vector_func(self.tik_inst, self.tik_inst.vmins, vconv_result, vconv_result, Constant.BOOL_FLAG_TRUE,
                            element_size, vconv_result.dtype)

    def _data_move_out(self, offset, element_size):
        result = self.input_ub.reinterpret_cast_to("float16")
        # cache reuse
        out_ub = self.cache_ub.reinterpret_cast_to("int8")
        _vconv_func(self.tik_inst, out_ub, result, "to-zero", element_size)
        _data_move(self.tik_inst, self.data_out[offset], out_ub, element_size)

    def _data_move_in(self, offset, element_size):
        # move input data
        _data_move(self.tik_inst, self.input_ub, self.data_input[offset], element_size)


@register_operator("IsFinite")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def is_finite(input_x, output_y, kernel_name="is_finite"):
    """
    Determine whether data of tensor is a finite number

    ----------
    input_x : dict
        shape and dtype of input, only support float16, float32
    output_y: dict
        shape and dtype of output, should be same shape as input,
        and the dtype should be bool
    kernel_name : str
        cce kernel name, default value is is_finite

    Returns
    ----------
    None
    """
    is_finite_instance = IsFinite(input_x, output_y, kernel_name)
    is_finite_instance.compute()
    ub_size = _get_max_element_in_ub(is_finite_instance.input_dtype, 1)
    tbe_context.get_context().add_compile_info(
        "vars", {
            "ub_size": ub_size,
            "core_num": Constant.CORE_NUM,
            "input_data_byte": common_util.get_data_size(is_finite_instance.input_dtype)
        })
    return is_finite_instance.build()
