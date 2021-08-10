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
trans_data_common_func
"""


from math import gcd as func_gcd
from functools import reduce as func_reduce
from te import platform as tbe_platform


CORE_UB_SIZE = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)
# AICORE count
CORE_DIM_NUM = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
# C0 length except int8 and uint8
C0_16 = 16
# C0 length int8 and uint8
C0_32 = 32
# Ni length
NI_16 = 16
# bytes in one block
BLOCK_BYTE_SIZE = 32
# repeat up limit for vector command
REPEAT_LIMIT_VECT = 255
# repeat up limit for mte
REPEAT_LIMIT_MTE = 4095
# strides up limit for mte
STRIDE_LIMIT_MTE = 65535
# mask value for float32
MASK_64 = 64
# mask value for float16
MASK_128 = 128
# max int64 value
MAX_INT64_VALUE = 2 ** 62 - 1
# used for vnchwconv
ADDR_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15)
# used for scalar
REG_IDX_LIST = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31)
# vnchwconv line count
VNC_LINES = 16


def ceil_div(value_x, value_y):
    """
    do ceil division

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the ceiling of value_x as an Integral
    """

    if value_y == 0:
        return value_x
    result = (value_x + value_y - 1) // value_y

    return result


def ceil_fill(value_x, value_y):
    """
    do ceiling product

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the ceiling product of value_x and value_y as an Integral
    """

    if value_y == 0:
        return value_x
    result = (value_x + value_y - 1) // value_y * value_y

    return result


def floor_div(value_x, value_y):
    """
    do floor division

    Parameters
    ----------
    value_x : int
        dividend
    value_y : int
        divider

    Returns
    -------
    the floor of value_x as an Integral
    """

    if value_y == 0:
        return value_x
    result = value_x // value_y

    return result


def lcm(value_x, value_y):
    """
    get lcm of value_x and value_y

    Parameters
    ----------
    value_x : int
        one integral input
    value_y : int
        another integral input

    Returns
    -------
    the lcm of value_x and value_y as an Integral
    """

    if value_x == 0 or value_y == 0:
        return 0
    result = value_x * value_y // func_gcd(value_x, value_y)

    return result


def get_c0_len(dtype):
    """
    get c0 length according to dtype

    Parameters
    ----------
    dtype : str
        data type name

    Returns
    -------
    the c0 length of dtype as an Integral
    """

    c0_len = C0_32 if dtype.lower() in ("int8", "uint8") else C0_16

    return c0_len


def clean_ubuf(tik_inst, src, src_offset, dup_len):
    """
    set ubuf to zero

    Parameters
    ----------
    tik_inst : tik instance
        used to generate cce code
    src : tensor
        ub tensor
    src_offset: int
        ub offset
    dup_len: int
        ub length to be cleaned

    Returns
    -------
    None
    """

    dtype = src.dtype.lower()
    if dtype in ("float16", "int16", "uint16"):
        dtype_factor = 2
    elif dtype in ("float32", "int32", "uint32"):
        dtype_factor = 1
    batch_size = MASK_64

    with tik_inst.new_stmt_scope():
        dup_len_reg = tik_inst.Scalar()
        dup_len_reg.set_as(dup_len)

        with tik_inst.if_scope(dup_len_reg > 0):
            repeat = dup_len_reg // (batch_size * dtype_factor)
            left_elem = dup_len_reg % (batch_size * dtype_factor)
            repeat_loop = repeat // REPEAT_LIMIT_VECT
            repeat_left = repeat % REPEAT_LIMIT_VECT
            dup_value = tik_inst.Scalar(dtype=dtype)
            dup_value.set_as(0)

            with tik_inst.if_scope(repeat_loop > 0):
                with tik_inst.for_range(0, repeat_loop) as rpt_idx:
                    tik_inst.vector_dup(MASK_64 * dtype_factor,
                                        src[src_offset + rpt_idx * REPEAT_LIMIT_VECT * batch_size * dtype_factor],
                                        dup_value, REPEAT_LIMIT_VECT, 1, 8)

            with tik_inst.if_scope(repeat_left > 0):
                tik_inst.vector_dup(MASK_64 * dtype_factor,
                                    src[src_offset + repeat_loop * REPEAT_LIMIT_VECT * batch_size * dtype_factor],
                                    dup_value, repeat_left, 1, 8)

            with tik_inst.if_scope(left_elem > 0):
                tik_inst.vector_dup(left_elem, src[src_offset + repeat * batch_size * dtype_factor],
                                    dup_value, 1, 1, 8)


def get_shape_size(sub_shape):
    """
    return shape size

    Parameters
    ----------
    sub_shape : tuple/list
        input tuple or list

    Returns
    -------
    the product of all values in sub_shape as an Integral
    """

    shape_size = func_reduce(lambda x, y: x * y, sub_shape)

    return shape_size


def get_dtype_len(in_dtype):
    """
    get the byte count of certain dtype

    Parameters
    ----------
    in_dtype : str
        data type name

    Returns
    -------
    the block count of in_dtype as an Integral
    """

    temp_dtype = in_dtype.lower()

    if temp_dtype in ("int8", "uint8"):
        byte_len = 1
    elif temp_dtype in ("float16", "int16", "uint16"):
        byte_len = 2
    elif temp_dtype in ("float32", "int32", "uint32"):
        byte_len = 4
    elif temp_dtype in ("int64", "uint64"):
        byte_len = 8

    return byte_len


def get_max_element_in_ub(in_dtype, ub_part, deduct_size=1024):
    """
    get the up limit elements in one part of UB

    Parameters
    ----------
    in_dtype : str
        data type name
    ub_part : int
        count of UB will be split
    deduct_size: int
        count of UB will be deducted

    Returns
    -------
    the up limit elements in one part of UB as an Integral
    """

    if ub_part == 0:
        return CORE_UB_SIZE

    byte_len = get_dtype_len(in_dtype)
    max_ub_size = 256 * 1024  # the unit is Byte
    if CORE_UB_SIZE >= max_ub_size:
        ub_upper_limit = (max_ub_size - deduct_size) // ub_part
    else:
        ub_upper_limit = (CORE_UB_SIZE - deduct_size) // ub_part
    element_size = ub_upper_limit // byte_len

    return element_size


def get_dtype_factor(dtype):
    """
    return 2 for float32, 1 for float16

    Parameters
    ----------
    in_dtype : str
        data type name

    Returns
    -------
    the data type factor as an Integral
    """

    size_factor = 2 if dtype.lower() in ("float32", "int32", "uint32") else 1

    return size_factor
