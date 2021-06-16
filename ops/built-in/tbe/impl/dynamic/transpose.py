"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

transpose
"""

from impl.util import util_common
from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from tbe.common.platform.platform_info import api_check_support

UB_SIZE = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
CORE_NUM = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
ACCU_BLOCK_SIZE = 128  # should less than 240 for both 310 and 910
ROW_UNIT = 128

# scenario_0
S0_FIXED_PART_SCALA_MAX_NUM = 100

# scenario_1
S1_FIXED_PART_SCALA_MAX_NUM = 100
S1_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_3
S3_FIXED_PART_SCALA_MAX_NUM = 100
S3_PERCORE_PART_SCALA_MAX_NUM = 100

# scenario_7
S7_FIXED_PART_SCALA_MAX_NUM = 100
S7_PERCORE_PART_SCALA_MAX_NUM = 100

TILING_MAX_PARAM_NUM = 512
TILING_MAX_SIZE_GM = 2048  # 16KB
MAX_INT64_VALUE = 2 ** 64 - 1
BLOCK_SIZE = 32
TRANSPOSE_MAX_AXIS_NUM = 8
RESERVED_UB = 4  # 4KB
EPB16 = 16
EPB32 = 32 
ELE_NUM_PER_BLOCK_FP32 = 8
ELE_NUM_PER_BLOCK_INT64 = 4
TILING_HEAD_LEN = 4
TILING_FIXED_MAX_LEN = 1024


def _fuzzy_match(shape_t):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    white_list_shape_fuzzy =  [
                               [-1, 12, 197, 64], [-1, 197, 12, 64], [-1, 197, 768], [-1, 768, 196],
                               [-1, 768, 197], [768, -1, 197], [128, 197, 12, -1], [128, 12, 197, -1]
                              ]
    for shape_w in white_list_shape_fuzzy:
        if len(shape_t) != len(shape_w):
            continue
        count = 0
        for i in range(len(shape_t)):
            if shape_w[i] == -1 or shape_t[i] == shape_w[i]:
                count = count + 1
                continue
            else:
                break
        if count == len(shape_t):
            return True
    return False


def _by_dynamic_static_union_version(shape, core_num):
    """
    temporary function, for dynamic & static union version not fully verified
    """
    if core_num == 1:
        white_list_shape_lhisi = [[1, 24, 3, 20]]
        shape_t_lhisi = list(shape)
        if shape_t_lhisi in white_list_shape_lhisi:
            return True
        return False

    white_list_shape = [
                         [2, 512, 1024], [1024, 91], [2, 512, 1024], [256, 784, 91],
                         [1024, 364], [2, 128, 91, 28, 28], [2, 128, 28, 28, 91],
                         [1024, 1024], [2, 512, 1024], [12544, 1024], [2, 512, 12544],
                         [4, 2, 4, 2, 3, 64], [1100, 1100], [2, 100, 1], [200, 116, 116, 4],
                         [1100], [1100, 512], [1, 512, 1, 24], [1, 512, 24], [38, 67, 512], [67, 38, 512],
                         [1, 24, 5, 5], [1, 486, 5, 5], [1, 24, 10, 10], [1, 486, 10, 10],
                         [1, 24, 20, 20], [1, 486, 20, 20], [1, 24, 40, 40], [1, 486, 40, 40],
                         [1, 24, 80, 80], [1, 486, 80, 80], [12, 8, 8, 36, 120],
                         [1, 100, 28, 28, 91], [4, 100, 28, 28, 91], [8, 100, 28, 28, 91], [16, 100, 28, 28, 91],
                         [80, 8, 1, 240], [80, 240, 8], [80, 240, 1, 8], [8, 80, 240], [240, 8, 64], [80, 8, 84],
                         [8, 80, 64], [1, 4, 1080, 1920, 3], [2, 100, 28, 28, 91], [2560, 26, 512],
                         [16, 40, 3, 14, 14], [16, 80, 3, 7, 7], [16, 20, 3, 28, 28],
                         [32, 3, 76, 76, 85], [32, 3, 38, 38, 85], [32, 3, 19, 19, 85], [32, 3, 85, 76, 76],
                         [32, 3, 85, 38, 38], [32, 3, 85, 19, 19], [512, 512, 9],
                         [3, 256, 1024], [48, 56, 64], [3, 16, 256, 64], [3, 1024, 256], [3, 3, 16, 16, 16, 16],
                         [3, 256, 16, 64], [48, 256, 64], [48, 256, 256], [768, 768], [3072, 768], [768, 197, 197],
                         [768, 3072]
                       ]
    shape_t = list(shape)
    if shape_t in white_list_shape:
        return True

    if _fuzzy_match(shape_t):
        return True

    return False


# pylint: disable=unused-argument
def check_supported(input_x, perm, output_y, kernel_name="dynamic_transpose"):
    """
    dynamic transpose is selected when any condition is true: \n
        -1 in input_x shape \n
        -1 in output_y shape \n
        -2 in input_x shape \n
    """
    x_shape = input_x.get("ori_shape")
    x_dtype = input_x.get("dtype")

    check_list = ["int8", "uint8", "bool", "float", "float32", "int32", "uint32", "int16", "uint16", "float16"]
    if x_dtype not in check_list:
        reason = "x_dtype [%s] not in %s" %(x_dtype, str(check_list))
        return False, reason

    if util_common.is_unknown([input_x, perm, output_y]):
        return True, ""

    if _by_dynamic_static_union_version(x_shape, CORE_NUM):
        return True, ""

    if tbe_context.get_context():
        if hasattr(tbe_context.get_context(), "get_build_type"):
            if tbe_context.get_context().get_build_type() == "fuzzily_build":
                return True, ""

    return False, ""


def _assert(tik_inst, ub_input, p, v):
    index = tik_inst.Scalar("int64", init_value=1)
    with tik_inst.if_scope(p != v):
        tik_inst.data_move(ub_input, ub_input[index], 0, 1, 1, 0, 0)


def _get_half_ub():
    return (UB_SIZE - RESERVED_UB * 1024) // 2 // 2


# pylint: disable=unused-argument,invalid-name, too-many-arguments, unused-variable, too-many-locals
# pylint: disable=too-many-statements, invalid-name, no-self-use, protected-access
# pylint: disable=too-many-instance-attributes, too-few-public-methods
class Transpose(object):
    """
    Transpose
    """

    class TilingParamS0(object):
        """
        TilingParamS0
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64):
            """
            get tiling parameters
            """
            # part 2: fixed

            for i in range(2):
                tiling_reg_list[i].set_as(ub_input_64_t[TILING_HEAD_LEN + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]

            #part 3 : percore
            reg_base = S0_FIXED_PART_SCALA_MAX_NUM
            self.base = tiling_reg_list[reg_base + 0]
            self.ele_num = tiling_reg_list[reg_base + 1]
            self.major_loop = tiling_reg_list[reg_base + 2]
            self.major_num = tiling_reg_list[reg_base + 3]
            self.tail_num = tiling_reg_list[reg_base + 4]
            self.not_align_ele = tiling_reg_list[reg_base + 5]

            self.base.set_as(ub_input_64[0])
            self.ele_num.set_as(ub_input_64[1])
            self.major_loop.set_as(ub_input_64[2])
            self.major_num.set_as(ub_input_64[3])
            self.tail_num.set_as(ub_input_64[4])
            self.not_align_ele.set_as(ub_input_64[5])

    class TilingParamS1(object):
        """
        TilingParamS1
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(6):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]

            reg_base = 6
            ub_offset.set_as(TILING_HEAD_LEN + 6)
            cycle = 3
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S1_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])

            ub_offset.set_as(1)
            reg_base = reg_base + 1
            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS2(object):
        """
        TilingParamS2
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            for i in range(9):
                tiling_reg_list[i].set_as(ub_input_64_t[TILING_HEAD_LEN + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]
            self.src_stride = tiling_reg_list[6]
            self.back_num = tiling_reg_list[7]
            self.skip_ele = tiling_reg_list[8]

            reg_base = 9
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN + 9)
            cycle = 4
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            self.dst_jump_factor_mod = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_factor_mod.append(tiling_reg_list[reg_base + i * cycle + 3])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor_mod[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM

            self.base = tiling_reg_list[reg_base]
            self.base.set_as(ub_input_64[ub_offset])
            ub_offset.set_as(ub_offset + 1)
            reg_base = reg_base + 1

            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])
            ub_offset.set_as(ub_offset + 1)
            reg_base = reg_base + 1

            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            reg_base = reg_base + TRANSPOSE_MAX_AXIS_NUM
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            reg_base = reg_base + TRANSPOSE_MAX_AXIS_NUM

            self.head_major_loop = tiling_reg_list[reg_base + 0]
            self.head_major_num = tiling_reg_list[reg_base + 1]
            self.head_tail_num = tiling_reg_list[reg_base + 2]
            self.body_loop = tiling_reg_list[reg_base + 3]
            self.body_major_loop = tiling_reg_list[reg_base + 4]
            self.body_major_num = tiling_reg_list[reg_base + 5]
            self.body_tail_num = tiling_reg_list[reg_base + 6]
            self.tail_major_loop = tiling_reg_list[reg_base + 7]
            self.tail_major_num = tiling_reg_list[reg_base + 8]
            self.tail_tail_num = tiling_reg_list[reg_base + 9]

            self.head_major_loop.set_as(ub_input_64[ub_offset + 0])
            self.head_major_num.set_as(ub_input_64[ub_offset + 1])
            self.head_tail_num.set_as(ub_input_64[ub_offset + 2])
            self.body_loop.set_as(ub_input_64[ub_offset + 3])
            self.body_major_loop.set_as(ub_input_64[ub_offset + 4])
            self.body_major_num.set_as(ub_input_64[ub_offset + 5])
            self.body_tail_num.set_as(ub_input_64[ub_offset + 6])
            self.tail_major_loop.set_as(ub_input_64[ub_offset + 7])
            self.tail_major_num.set_as(ub_input_64[ub_offset + 8])
            self.tail_tail_num.set_as(ub_input_64[ub_offset + 9])
            ub_offset.set_as(ub_offset + 10)

            # part 4: variable
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S1_FIXED_PART_SCALA_MAX_NUM + S1_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS3(object):
        """
        TilingParamS3
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """
            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(10):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])
            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.last_axis_len = tiling_reg_list[2]
            self.last_axis_burst_len = tiling_reg_list[3]
            self.align_ele = tiling_reg_list[4]
            self.trans_axis_num = tiling_reg_list[5]
            self.major_loop_num = tiling_reg_list[6]
            self.major_blocks = tiling_reg_list[7]
            self.tail_blocks = tiling_reg_list[8]
            self.back_ele = tiling_reg_list[9]

            reg_base = 10
            ub_offset.set_as(ub_offset + reg_base)
            cycle = 3
            self.src_jump_stride = []
            self.dst_jump_stride = []
            self.dst_jump_factor = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.trans_axis_num)

            # part 3 : percore
            ub_offset.set_as(0)
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM
            self.loop_num = tiling_reg_list[reg_base]
            self.loop_num.set_as(ub_input_64[ub_offset])

            ub_offset.set_as(1)
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + 1
            self.init_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple.append(tiling_reg_list[reg_base + i])
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_tuple[i].set_as(ub_input_64[ub_offset + i])

            # part 4: variable
            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + S3_PERCORE_PART_SCALA_MAX_NUM
            cycle = 1
            self.rt_tuple = []
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])

            reg_base = S3_FIXED_PART_SCALA_MAX_NUM + S3_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM
            self.src_addr = tiling_reg_list[reg_base]
            self.dst_addr = tiling_reg_list[reg_base + 1]

    class TilingParamS7(object):
        """
        TilingParamS7
        """
        def __init__(self, tiling_reg_list, ub_input_64_t, ub_input_64, tik_inst):
            """
            get tiling parameters
            """

            # part 2: fixed
            ub_offset = tik_inst.Scalar("int32", init_value=TILING_HEAD_LEN)

            for i in range(6):
                tiling_reg_list[i].set_as(ub_input_64_t[ub_offset + i])

            self.core_num = tiling_reg_list[0]
            self.ub_size = tiling_reg_list[1]
            self.n_axis_num = tiling_reg_list[2]
            self.dst_axis_num = tiling_reg_list[3]
            self.src_axis_num = tiling_reg_list[4]
            self.right_part_vol = tiling_reg_list[5]

            self.n_jump_factor = []
            self.n_jump_stride = []
            self.dst_jump_factor = []
            self.dst_jump_stride = []
            self.src_jump_factor = []
            self.src_jump_stride = []

            reg_base = 6
            ub_offset.set_as(ub_offset + reg_base)
            cycle = 6
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.n_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.dst_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.dst_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 3])
                self.src_jump_factor.append(tiling_reg_list[reg_base + i * cycle + 4])
                self.src_jump_stride.append(tiling_reg_list[reg_base + i * cycle + 5])


            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.n_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.n_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.n_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.dst_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.dst_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.dst_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_factor[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.src_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.src_jump_stride[i].set_as(ub_input_64_t[ub_offset + i])
            ub_offset.set_as(ub_offset + self.src_axis_num)


            # part 3: per core
            per_core_front = 12
            ub_offset.set_as(0)
            reg_base = S7_FIXED_PART_SCALA_MAX_NUM
            for i in range(per_core_front):
                tiling_reg_list[reg_base + i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(per_core_front)

            self.loop_on_n = tiling_reg_list[reg_base + 0]
            self.n_offset_actual = tiling_reg_list[reg_base + 1]
            self.col_per_mc = tiling_reg_list[reg_base + 2]
            self.loop_on_mc = tiling_reg_list[reg_base + 3]
            self.col_tc = tiling_reg_list[reg_base + 4]
            self.col_offset = tiling_reg_list[reg_base + 5]
            self.back_step_left = tiling_reg_list[reg_base + 6]
            self.row_per_mr = tiling_reg_list[reg_base + 7]
            self.loop_on_mr = tiling_reg_list[reg_base + 8]
            self.row_tr = tiling_reg_list[reg_base + 9]
            self.row_offset = tiling_reg_list[reg_base + 10]
            self.back_step_up = tiling_reg_list[reg_base + 11]
            #if add line here, should change "per_core_front"

            self.init_n_tuple = []
            self.init_dst_tuple = []
            self.tail_dst_tuple = []
            self.init_src_tuple = []
            self.tail_src_tuple = []

            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + per_core_front
            cycle = 5
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.init_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.tail_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.init_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 3])
                self.tail_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 4])

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_n_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.n_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_dst_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.dst_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.tail_dst_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.dst_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.init_src_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.src_axis_num)

            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.tail_src_tuple[i].set_as(ub_input_64[ub_offset + i])
            ub_offset.set_as(ub_offset + self.src_axis_num)


            # part 4: variable
            self.rt_n_tuple = []
            self.rt_src_tuple = []
            self.rt_dst_tuple = []
            self.rt_dst_tuple_backup = []

            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + S7_PERCORE_PART_SCALA_MAX_NUM
            cycle = 4
            for i in range(TRANSPOSE_MAX_AXIS_NUM):
                self.rt_n_tuple.append(tiling_reg_list[reg_base + i * cycle + 0])
                self.rt_dst_tuple.append(tiling_reg_list[reg_base + i * cycle + 1])
                self.rt_src_tuple.append(tiling_reg_list[reg_base + i * cycle + 2])
                self.rt_dst_tuple_backup.append(tiling_reg_list[reg_base + i * cycle + 3])

            reg_base = S7_FIXED_PART_SCALA_MAX_NUM + S7_PERCORE_PART_SCALA_MAX_NUM + cycle * TRANSPOSE_MAX_AXIS_NUM

            self.fp16_offset_1 = tiling_reg_list[reg_base + 0]
            self.fp16_offset_2 = tiling_reg_list[reg_base + 1]
            self.fp16_offset_3 = tiling_reg_list[reg_base + 2]
            self.src_stride_reorder = tiling_reg_list[reg_base + 3]
            self.dst_stride_reorder = tiling_reg_list[reg_base + 4]
            self.col_reorder = tiling_reg_list[reg_base + 5]
            self.row_reorder = tiling_reg_list[reg_base + 6]
            self.rt_dst_addr = tiling_reg_list[reg_base + 7]

    def _init_mem_make_ub_allocated(self, ub_input):
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        self.tik_inst.vector_dup(128, ub_input_b16, 0, 1, 1, 0)

    def __init__(self, tik_inst, x_dtype, tensor_list, kernel_name):
        self.tik_inst = tik_inst
        self.x_dtype = x_dtype
        self.kernel_name = kernel_name
        self.data_in, self.data_perm, self.data_out, self.data_workspace, self.data_tiling = tensor_list
        self.ub_size = self._get_ub_size_by_dtype()
        self.ub_size_64 = self._get_ub_size_by_int64()
        self.ub_input_64_t = self.tik_inst.Tensor("int64", (128,), tik.scope_ubuf, "ub_input_64_t")
        self.ub_input_b16_vor = self.tik_inst.Tensor("int16", (128,), tik.scope_ubuf, "ub_input_b16_vor")
        self._init_mem_make_ub_allocated(self.ub_input_b16_vor)
        self.ub_input_64 = self.tik_inst.Tensor("int64", (self.ub_size_64,), tik.scope_ubuf, "ub_input_64")
        self._init_mem_make_ub_allocated(self.ub_input_64)
        self.tiling_reg_list = [self.tik_inst.Scalar("int64") for i in range(TILING_MAX_PARAM_NUM)]
        self.element_per_block = self._element_per_block(self.x_dtype)
        self.fp16_times = (self._sizeof_dtype(x_dtype) + 1) // self._sizeof_dtype("float16") # fp32/int32:2  fp16/int16:1
        self.ele_per_block = BLOCK_SIZE // self._sizeof_dtype(x_dtype)
        tik_inst.data_move(self.ub_input_64_t, self.data_tiling, 0, 1, TILING_FIXED_MAX_LEN // BLOCK_SIZE, 0, 0)

    def _sizeof_dtype(self, dtype):
        if dtype in ("int8", "uint8", "bool"):
            return 1
        if dtype in ("float16", "int16", "uint16"):
            return 2
        if dtype in ("float", "float32", "int32", "uint32"):
            return 4
        if dtype in ("int64", "uint64", "double"):
            return 8
        return 8

    def _element_per_block(self, dtype):
        if dtype in ("int8", "uint8", "bool"):
            return 32
        if dtype in ("float16", "int16", "uint16"):
            return 16
        if dtype in ("float", "float32", "int32", "uint32"):
            return 8
        if dtype in ("int64", "uint64", "double"):
            return 4
        return 4

    def _get_ub_size_by_dtype(self):
        return (UB_SIZE - RESERVED_UB * 2048) // self._sizeof_dtype(self.x_dtype)

    def _get_ub_size_by_int64(self):
        return (UB_SIZE - RESERVED_UB * 1024) // self._sizeof_dtype("int64")

    def _get_src_size(self):
        if UB_SIZE == 256 * 1024:
            return 3968 - 16  # 910 & 310
        if UB_SIZE == 192 * 1024:
            return 2848  # cs
        if UB_SIZE == 128 * 1024:
            return 1861  # es
        return 3968 - 16

    def _get_dst_size(self):
        if UB_SIZE == 256 * 1024:
            return 247  # 910 & 301,  247 to avoid bank conflict
        if UB_SIZE == 192 * 1024:
            return 191 # cs
        if UB_SIZE == 128 * 1024:
            return 123 # es
        return 247

    def _get_offset_1(self):
        if CORE_NUM > 2 and UB_SIZE == 256 * 1024:
            return 3952 # 910
        if CORE_NUM == 2 and UB_SIZE == 256 * 1024:
            return 3824 # 310
        if CORE_NUM == 1 and UB_SIZE == 192 * 1024:
            return 2832 # cs es
        return 3952

    def _get_offset_2(self, offset_1):
        return offset_1 + offset_1 * EPB16

    #       cycle_num_wsp: 4
    #                loop_0       loop_1    tail
    #             ------------|------------|----
    #    cycle_0  -   -   -   |-   -   -   |-
    #    cycle_1   -   -   -  | -   -   -  | -
    #    cycle_2    -   -   - |  -   -   - |  -
    #    cycle_3     -   -   -|   -   -   -|   -

    # pylint: disable=too-many-arguments, unused-argument, invalid-name
    def _copy_input_2_workspace(self, tiling_param, ub_input, src_pos_wsp, dst_pos_wsp):
        tp = tiling_param
        with self.tik_inst.for_range(0, tp.cycle_num_wsp) as i:
            with self.tik_inst.for_range(0, tp.loop_num_wsp) as j:
                self.tik_inst.data_move(ub_input,
                                        self.data_in[src_pos_wsp +\
                                                j * tp.nburst_wsp * tp.last_axis_ele * self.element_per_block +\
                                                i * tp.last_axis_ele],
                                        0,
                                        tp.nburst_wsp,
                                        tp.burst_len,
                                        tp.src_stride_wsp,
                                        0)
                self.tik_inst.data_move(self.data_workspace[dst_pos_wsp +\
                                            j * tp.nburst_wsp * tp.last_axis_ele_a * self.element_per_block +\
                                            i * tp.last_axis_ele_a],
                                        ub_input,
                                        0,
                                        tp.nburst_wsp,
                                        tp.burst_len,
                                        0,
                                        tp.dst_stride_wsp)
            with self.tik_inst.if_scope(tp.nburst_tail_wsp != 0):
                self.tik_inst.data_move(ub_input,
                                        self.data_in[src_pos_wsp +\
                                                     tp.loop_num_wsp * tp.nburst_wsp * tp.last_axis_ele *\
                                                     self.element_per_block +\
                                                     i * tp.last_axis_ele],
                                        0,
                                        tp.nburst_tail_wsp,
                                        tp.burst_len,
                                        tp.src_stride_wsp,
                                        0)
                self.tik_inst.data_move(self.data_workspace[dst_pos_wsp +\
                                                         tp.loop_num_wsp * tp.nburst_wsp * tp.last_axis_ele_a *\
                                                         self.element_per_block +\
                                                         i * tp.last_axis_ele_a],
                                        ub_input,
                                        0,
                                        tp.nburst_tail_wsp,
                                        tp.burst_len,
                                        0,
                                        tp.dst_stride_wsp)

    def _ele_per_block(self):
        return BLOCK_SIZE // self._sizeof_dtype(self.x_dtype)

    def _move_data_s0(self, tp, ub_input_64):
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        with self.tik_inst.for_range(0, tp.major_loop) as i:
            self.tik_inst.data_move(ub_input, self.data_in[tp.base + i * tp.major_num * self.ele_per_block],
                                    0, 1, tp.major_num, 0, 0)
            self.tik_inst.data_move(self.data_out[tp.base + i * tp.major_num * self.ele_per_block],
                                    ub_input, 0, 1, tp.major_num, 0, 0)

        with self.tik_inst.if_scope(tp.tail_num != 0):
            self.tik_inst.data_move(ub_input,
                                    self.data_in[tp.base + tp.major_loop * tp.major_num * self.ele_per_block],
                                    0, 1, tp.tail_num, 0, 0)
            self.tik_inst.data_move(self.data_out[tp.base + tp.major_loop * tp.major_num * self.ele_per_block],
                                    ub_input, 0, 1, tp.tail_num, 0, 0)

        with self.tik_inst.if_scope(tp.not_align_ele != 0):
            with self.tik_inst.if_scope(tik.all(tp.major_loop == 0, tp.tail_num == 0)):
                self.tik_inst.data_move(ub_input, self.data_in[tp.base], 0, 1, 1, 0, 0)
                self.tik_inst.data_move(self.data_out[tp.base], ub_input, 0, 1, 1, 0, 0)
            with self.tik_inst.else_scope():
                self.tik_inst.data_move(ub_input, self.data_in[tp.base + tp.ele_num - self.ele_per_block],
                                        0, 1, 1, 0, 0)
                self.tik_inst.data_move(self.data_out[tp.base + tp.ele_num - self.ele_per_block], ub_input,
                                        0, 1, 1, 0, 0)

    def _get_src_addr_s1(self, tp):
        with self.tik_inst.if_scope(tp.trans_axis_num == 7):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.src_jump_stride[4] + \
                               tp.rt_tuple[5] * tp.src_jump_stride[5] + \
                               tp.rt_tuple[6] * tp.src_jump_stride[6])

        with self.tik_inst.if_scope(tp.trans_axis_num == 6):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.src_jump_stride[4] + \
                               tp.rt_tuple[5] * tp.src_jump_stride[5])

        with self.tik_inst.if_scope(tp.trans_axis_num == 5):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.src_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.src_jump_stride[4])

        with self.tik_inst.if_scope(tp.trans_axis_num == 4):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.src_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.src_jump_stride[3])

        with self.tik_inst.if_scope(tp.trans_axis_num == 3):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.src_jump_stride[2])

        with self.tik_inst.if_scope(tp.trans_axis_num == 2):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.src_jump_stride[1])

        with self.tik_inst.if_scope(tp.trans_axis_num == 1):
            tp.src_addr.set_as(tp.rt_tuple[0] * tp.src_jump_stride[0])

    def _get_dst_addr_s1(self, tp):
        with self.tik_inst.if_scope(tp.trans_axis_num == 7):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.dst_jump_stride[4] + \
                               tp.rt_tuple[5] * tp.dst_jump_stride[5] + \
                               tp.rt_tuple[6] * tp.dst_jump_stride[6])

        with self.tik_inst.if_scope(tp.trans_axis_num == 6):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.dst_jump_stride[4] + \
                               tp.rt_tuple[5] * tp.dst_jump_stride[5])

        with self.tik_inst.if_scope(tp.trans_axis_num == 5):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.dst_jump_stride[3] + \
                               tp.rt_tuple[4] * tp.dst_jump_stride[4])

        with self.tik_inst.if_scope(tp.trans_axis_num == 4):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.dst_jump_stride[2] + \
                               tp.rt_tuple[3] * tp.dst_jump_stride[3])

        with self.tik_inst.if_scope(tp.trans_axis_num == 3):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1] + \
                               tp.rt_tuple[2] * tp.dst_jump_stride[2])

        with self.tik_inst.if_scope(tp.trans_axis_num == 2):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0] + \
                               tp.rt_tuple[1] * tp.dst_jump_stride[1])

        with self.tik_inst.if_scope(tp.trans_axis_num == 1):
            tp.dst_addr.set_as(tp.rt_tuple[0] * tp.dst_jump_stride[0])

    def _init_tuple_common(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_tuple[i].set_as(tp.init_tuple[i])

    def _copy_in_s1(self, tp, ub_input, burst_len, ub_offset):
        self._get_src_addr_s1(tp)
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, burst_len, 0, 0)

    def _copy_out_s1(self, tp, ub_input, burst_len, ub_offset):
        self._get_dst_addr_s1(tp)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, burst_len, 0, 0)

    def _copy_anti_overlap_s1(self, tp, ub_input):
        skip_offset = self.tik_inst.Scalar("int32")
        skip_offset.set_as((tp.last_axis_burst_len - 1) * self.ele_per_block)
        skip_offset.set_as(skip_offset - (self.ele_per_block - (tp.last_axis_len - skip_offset)))
        scalar_value = self.tik_inst.Scalar(self.x_dtype)
        with self.tik_inst.for_range(0, self.ele_per_block) as i:
            scalar_value.set_as(ub_input[skip_offset + i])
            ub_input[i] = scalar_value
        self.tik_inst.data_move(self.data_out[tp.dst_addr + skip_offset], ub_input, 0, 1, 1, 0, 0)

    def _move_data_s1(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32") # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_tuple_common(tp)
        with self.tik_inst.if_scope(tp.align_ele == 0):
            with self.tik_inst.for_range(0, tp.loop_num) as ln:
                self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)
        with self.tik_inst.else_scope():
            with self.tik_inst.for_range(0, tp.loop_num - 1) as ln:
                self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len, ub_offset)
                self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)
            self._copy_in_s1(tp, ub_input, tp.last_axis_burst_len, ub_offset)
            self._copy_out_s1(tp, ub_input, tp.last_axis_burst_len - 1, ub_offset)
            self._copy_anti_overlap_s1(tp, ub_input)

    # pylint: disable=too-many-arguments, unused-argument, invalid-name
    def _reorder_s2(self, tp, ub_input, ub_offset, ub_offset_exclude_pad):
        if self.x_dtype in ("int8", "uint8", "bool"):
            b8_offset = _get_half_ub() * 2
            ub_input_b8 = ub_input.reinterpret_cast_to("int8")
            src_ele_num_in_b8 = self._get_src_size() * 2 # avoid bank conflict
            src_list = [ub_input_b8[src_ele_num_in_b8 * i] for i in range(EPB16)]
            dst_list_low = [ub_input_b8[b8_offset + EPB32 * i] for i in range(EPB16)]
            dst_list_high = [ub_input_b8[b8_offset + EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset == 1):
                self.tik_inst.vnchwconv(False, False, dst_list_low, src_list, 1, 0, 0)
                self.tik_inst.vnchwconv(False, True, dst_list_high, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset != 1):
                self.tik_inst.vnchwconv(False, False, dst_list_low, src_list, ub_offset, EPB32, 1)
                self.tik_inst.vnchwconv(False, True, dst_list_high, src_list, ub_offset, EPB32, 1)
            
            # step2. erase unused elements aligned
            all_line_number = tp.last_axis_burst_len * EPB32
            pad_line_number = tp.align_ele * self.fp16_times
            nburst = ub_offset // tp.last_axis_burst_len
            burst_len = all_line_number - pad_line_number
            self.tik_inst.data_move(ub_input_b8, ub_input_b8[b8_offset], 0, nburst, burst_len, pad_line_number, 0)

            # step3. make all elements in the first col be in memory of contiguous
            ub_offset_exclude_pad.set_as(((all_line_number - pad_line_number) * nburst + EPB32 - 1)// EPB32)
            src_list_low = [ub_input_b8[EPB32 * i] for i in range(EPB16)]
            src_list_high = [ub_input_b8[EPB32 * i + EPB32 * EPB16] for i in range(EPB16)]
            dst_list = [ub_input_b8[b8_offset + self._get_dst_size() * EPB32 * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset_exclude_pad == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, 1, 0, 0)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset_exclude_pad > 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list_low, ub_offset_exclude_pad, 1, EPB32)
                self.tik_inst.vnchwconv(True, False, dst_list, src_list_high, ub_offset_exclude_pad, 1, EPB32)
            self.tik_inst.data_move(ub_input_b8, ub_input_b8[b8_offset], 0, 1, ub_offset_exclude_pad, 0, 0)
        else:
            # step1. make all elements in the first col
            fp16_offset_1 = ACCU_BLOCK_SIZE * 32
            fp16_offset_2 = ACCU_BLOCK_SIZE * 32 + ACCU_BLOCK_SIZE * 32 * 16
            ub_input_fp16 = ub_input.reinterpret_cast_to("float16")
            src_ele_num_in_fp16 = self._get_src_size() # avoid bank conflict
            src_list = [ub_input_fp16[src_ele_num_in_fp16 * i] for i in range(EPB16)]
            dst_list = [ub_input_fp16[fp16_offset_1 + EPB16 * i] for i in range(EPB16)]
            with self.tik_inst.if_scope(ub_offset == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset != 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, ub_offset, EPB16, 1)

            # step2. erase unused elements aligned
            all_line_number = tp.last_axis_burst_len * EPB16
            pad_line_number = tp.align_ele * self.fp16_times
            nburst = ub_offset // tp.last_axis_burst_len
            burst_len = all_line_number - pad_line_number
            self.tik_inst.data_move(ub_input_fp16[fp16_offset_2], ub_input_fp16[fp16_offset_1],
                                    0, nburst, burst_len, pad_line_number, 0)

            # step3. make all elements in the first col be in memory of contiguous
            ub_offset_exclude_pad.set_as(((all_line_number - pad_line_number) * nburst + EPB16 - 1)// EPB16)
            src_list = [ub_input_fp16[fp16_offset_2 + EPB16 * i] for i in range(EPB16)]
            # 247 avoid bank conflict
            dst_list = [ub_input_fp16[self._get_dst_size() * EPB16 * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(ub_offset_exclude_pad == 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, 1, 0, 0)
            with self.tik_inst.if_scope(ub_offset_exclude_pad > 1):
                self.tik_inst.vnchwconv(False, False, dst_list, src_list, ub_offset_exclude_pad, 1, EPB16)

    def _get_src_addr_s2(self, tp):
        self._get_src_addr_s1(tp)

    def _get_dst_addr_s2(self, tp, steps):
        tp.dst_addr.set_as((tp.base + steps) * tp.last_axis_len)

    def _copy_out_s2(self, tp, ub_input, accu_blocks, backup_steps, steps):
        ub_offset_exclude_pad = self.tik_inst.Scalar("int32") # unit : block
        ub_offset_exclude_pad.set_as(accu_blocks)
        with self.tik_inst.if_scope(tp.align_ele != 0):
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
        self._get_dst_addr_s2(tp, backup_steps)
        self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, ub_offset_exclude_pad, 0, 0)
        backup_steps.set_as(steps)
        accu_blocks.set_as(0)

    def _copy_common_s2(self, tp, ub_input, steps, accu_blocks, major_loop, major_num, tail_num):
        backup_steps = self.tik_inst.Scalar("int64", init_value=0)
        backup_steps.set_as(steps)
        tik_inst = self.tik_inst
        with tik_inst.for_range(0, major_loop):
            self._get_src_addr_s2(tp)
            tik_inst.data_move(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr], 0, major_num,
                               tp.last_axis_burst_len, tp.src_stride, 0)
            steps.set_as(steps + major_num)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            accu_blocks.set_as(accu_blocks + major_num * tp.last_axis_burst_len)
            with self.tik_inst.if_scope(accu_blocks >= ACCU_BLOCK_SIZE):  # 64=2KB, 128=4KB, 200=6.4KB
                self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

        with tik_inst.if_scope(tail_num != 0):
            self._get_src_addr_s2(tp)
            tik_inst.data_move(ub_input[accu_blocks * self.ele_per_block], self.data_in[tp.src_addr], 0, tail_num,
                               tp.last_axis_burst_len, tp.src_stride, 0)
            steps.set_as(steps + tail_num)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            accu_blocks.set_as(accu_blocks + tail_num * tp.last_axis_burst_len)
            with self.tik_inst.if_scope(accu_blocks >= ACCU_BLOCK_SIZE):  # 64=2KB, 128=4KB, 200=6.4KB
                self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

        with self.tik_inst.if_scope(accu_blocks != 0):
            self._copy_out_s2(tp, ub_input, accu_blocks, backup_steps, steps)

    def _copy_head_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.head_major_loop, tp.head_major_num, tp.head_tail_num)

    def _copy_body_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.for_range(0, tp.body_loop):
            self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.body_major_loop,
                                 tp.body_major_num, tp.body_tail_num)

    def _copy_tail_s2_aligned(self, tp, ub_input, steps, accu_blocks):
        self._copy_common_s2(tp, ub_input, steps, accu_blocks, tp.tail_major_loop, tp.tail_major_num, tp.tail_tail_num)

    def _copy_tiny_data_lt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            ub_offset_exclude_pad = self.tik_inst.Scalar("int32")  # unit : block
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(0)
            accu_blocks.set_as(0)
            self._get_dst_addr_s2(tp, steps)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            with self.tik_inst.for_range(0, tp.loop_num):
                self._get_src_addr_s2(tp)
                self.tik_inst.data_move(ub_input[accu_blocks * self.ele_per_block],
                                        self.data_in[tp.src_addr], 0, 1, 1, 0, 0)
                steps.set_as(steps + 1)
                self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor,
                                              tp.dst_jump_factor_mod, tp.base, steps)
                accu_blocks.set_as(accu_blocks + 1)
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
            with self.tik_inst.for_range(0, tp.loop_num) as i:
                scalar_value.set_as(ub_input[i])
                ub_input[i] = scalar_value
            self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, 1, 0, 0)
            # for int32 skip_ele = 0 if last_axis is 1,2,4
            with self.tik_inst.if_scope(tp.skip_ele != 0):
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    scalar_value.set_as(ub_input[tp.skip_ele + i])
                    ub_input[i] = scalar_value
                self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.skip_ele], ub_input, 0, 1, 1, 0, 0)

    def _copy_anti_overlap_lt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            ub_offset_exclude_pad = self.tik_inst.Scalar("int32")  # unit : block
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(tp.loop_num - tp.back_num)
            accu_blocks.set_as(0)
            self._get_dst_addr_s2(tp, steps)
            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)
            with self.tik_inst.for_range(0, tp.back_num):
                self._get_src_addr_s2(tp)
                self.tik_inst.data_move(ub_input[accu_blocks * self.ele_per_block],
                                        self.data_in[tp.src_addr], 0, 1, 1, 0, 0)
                steps.set_as(steps + 1)
                self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor,
                                              tp.dst_jump_factor_mod, tp.base, steps)
                accu_blocks.set_as(accu_blocks + 1)
            self._reorder_s2(tp, ub_input, accu_blocks, ub_offset_exclude_pad)
            with self.tik_inst.for_range(0, self.ele_per_block) as i:
                scalar_value.set_as(ub_input[i])
                ub_input[i] = scalar_value
            self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, 1, 0, 0)
            # for int32 skip_ele = 0 if last_axis is 1,2,4
            with self.tik_inst.if_scope(tp.skip_ele != 0):
                with self.tik_inst.for_range(0, self.ele_per_block) as i:
                    scalar_value.set_as(ub_input[tp.skip_ele + i])
                    ub_input[i] = scalar_value
                self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.skip_ele], ub_input, 0, 1, 1, 0, 0)

    def _copy_anti_overlap_gt_blk_s2(self, tp, ub_input, steps, accu_blocks):
        with self.tik_inst.if_scope(tp.loop_num != 0):
            scalar_value = self.tik_inst.Scalar(self.x_dtype)
            steps.set_as(tp.loop_num - 1)
            self._get_dst_addr_s2(tp, steps)
            self._get_src_addr_s2(tp)

            self._update_tuple_with_steps(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor, tp.dst_jump_factor_mod,
                                          tp.base, steps)

            self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr], 0, 1, tp.last_axis_burst_len, 0, 0)
            self.tik_inst.data_move(self.data_out[tp.dst_addr], ub_input, 0, 1, tp.last_axis_burst_len - 1, 0, 0)

            with self.tik_inst.for_range(0, self.ele_per_block) as i:
                scalar_value.set_as(ub_input[tp.last_axis_len - self.ele_per_block + i])
                ub_input[i] = scalar_value
            self.tik_inst.data_move(self.data_out[tp.dst_addr + tp.last_axis_len - self.ele_per_block],
                                    ub_input, 0, 1, 1, 0, 0)

    def _move_data_s2(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        steps = self.tik_inst.Scalar("int64", init_value=0)
        accu_blocks = self.tik_inst.Scalar("int32", init_value=0)  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        #                   <----------------this core data---------------->
        #   <----------------------->|<----------------------->|<----------------------->

        #   -----------------------------------------------------------------------------
        #   |               |        |                         |           |            |
        #   |               | head   |      body               |      tail |            |
        #   |               |        |                         |           |            |
        #   -----------------------------------------------------------------------------

        self._init_tuple_common(tp)
        self._copy_head_s2_aligned(tp, ub_input, steps, accu_blocks)
        self._copy_body_s2_aligned(tp, ub_input, steps, accu_blocks)
        self._copy_tail_s2_aligned(tp, ub_input, steps, accu_blocks)

        with self.tik_inst.if_scope(tik.all(tp.head_major_loop == 0,
                                            tp.body_loop == 0,
                                            tp.tail_major_loop == 0,
                                            tp.last_axis_len < self.ele_per_block)):
            self._copy_tiny_data_lt_blk_s2(tp, ub_input, steps, accu_blocks)
        with self.tik_inst.else_scope():
            with self.tik_inst.if_scope(tp.last_axis_len < self.ele_per_block):
                self._copy_anti_overlap_lt_blk_s2(tp, ub_input, steps, accu_blocks)
            with self.tik_inst.if_scope(tp.align_ele != 0):
                with self.tik_inst.if_scope(tp.last_axis_len > self.ele_per_block):
                    self._copy_anti_overlap_gt_blk_s2(tp, ub_input, steps, accu_blocks)

    def _copy_in_major_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr + last_axis_offset], 0, 1, tp.major_blocks, 0, 0)

    def _copy_out_major_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(self.data_out[tp.dst_addr + last_axis_offset], ub_input, 0, 1, tp.major_blocks, 0, 0)

    def _update_last_axis_offset(self, tp, last_axis_offset):
        last_axis_offset.set_as(last_axis_offset + tp.major_blocks * self.ele_per_block)

    def _copy_in_tail_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(ub_input, self.data_in[tp.src_addr + last_axis_offset - tp.back_ele],
                                0, 1, tp.tail_blocks, 0, 0)

    def _copy_out_tail_s3(self, tp, ub_input, last_axis_offset):
        self.tik_inst.data_move(self.data_out[tp.dst_addr + last_axis_offset - tp.back_ele], ub_input,
                                0, 1, tp.tail_blocks, 0, 0)

    def _get_src_addr_s3(self, tp):
        self._get_src_addr_s1(tp)

    def _get_dst_addr_s3(self, tp):
        self._get_dst_addr_s1(tp)

    def _move_data_s3(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        last_axis_offset = self.tik_inst.Scalar("int32")  # unit : ele
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        tik_inst = self.tik_inst

        self._init_tuple_common(tp)
        with tik_inst.for_range(0, tp.loop_num):
            last_axis_offset.set_as(0)
            self._get_src_addr_s3(tp)
            self._get_dst_addr_s3(tp)
            with tik_inst.for_range(0, tp.major_loop_num):
                self._copy_in_major_s3(tp, ub_input, last_axis_offset)
                self._copy_out_major_s3(tp, ub_input, last_axis_offset)
                self._update_last_axis_offset(tp, last_axis_offset)
            with tik_inst.if_scope(tp.tail_blocks != 0):
                self._copy_in_tail_s3(tp, ub_input, last_axis_offset)
                self._copy_out_tail_s3(tp, ub_input, last_axis_offset)
            self._update_tuple(tp.trans_axis_num, tp.rt_tuple, tp.dst_jump_factor)

    def _init_n_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_n_tuple[i].set_as(tp.init_n_tuple[i])

    def _init_dst_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.init_dst_tuple[i])
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple_backup[i].set_as(tp.init_dst_tuple[i])

    def _restore_dst_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.rt_dst_tuple_backup[i])

    def _backup_dst_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple_backup[i].set_as(tp.rt_dst_tuple[i])

    def _tail_dst_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_dst_tuple[i].set_as(tp.tail_dst_tuple[i])

    def _init_src_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_src_tuple[i].set_as(tp.init_src_tuple[i])

    def _tail_src_tuple(self, tp):
        for i in range(TRANSPOSE_MAX_AXIS_NUM - 1):
            tp.rt_src_tuple[i].set_as(tp.tail_src_tuple[i])

    def _update_tuple(self, axis_num, rt_tuple, jump_factor):
        with self.tik_inst.if_scope(axis_num == 7):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                    rt_tuple[1].set_as(0)
                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                        rt_tuple[2].set_as(0)
                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                            rt_tuple[3].set_as(0)
                            with self.tik_inst.if_scope(rt_tuple[4] == jump_factor[4] - 1):
                                rt_tuple[4].set_as(0)
                                with self.tik_inst.if_scope(rt_tuple[5] == jump_factor[5] - 1):
                                    rt_tuple[5].set_as(0)
                                    rt_tuple[6].set_as(rt_tuple[6] + 1)
                                with self.tik_inst.else_scope():
                                    rt_tuple[5].set_as(rt_tuple[5] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[4].set_as(rt_tuple[4] + 1)
                        with self.tik_inst.else_scope():
                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 6):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                    rt_tuple[1].set_as(0)
                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                        rt_tuple[2].set_as(0)
                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                            rt_tuple[3].set_as(0)
                            with self.tik_inst.if_scope(rt_tuple[4] == jump_factor[4] - 1):
                                rt_tuple[4].set_as(0)
                                rt_tuple[5].set_as(rt_tuple[5] + 1)
                            with self.tik_inst.else_scope():
                                rt_tuple[4].set_as(rt_tuple[4] + 1)
                        with self.tik_inst.else_scope():
                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 5):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                    rt_tuple[1].set_as(0)
                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                        rt_tuple[2].set_as(0)
                        with self.tik_inst.if_scope(rt_tuple[3] == jump_factor[3] - 1):
                            rt_tuple[3].set_as(0)
                            rt_tuple[4].set_as(rt_tuple[4] + 1)
                        with self.tik_inst.else_scope():
                            rt_tuple[3].set_as(rt_tuple[3] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 4):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                    rt_tuple[1].set_as(0)
                    with self.tik_inst.if_scope(rt_tuple[2] == jump_factor[2] - 1):
                        rt_tuple[2].set_as(0)
                        rt_tuple[3].set_as(rt_tuple[3] + 1)
                    with self.tik_inst.else_scope():
                        rt_tuple[2].set_as(rt_tuple[2] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 3):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                with self.tik_inst.if_scope(rt_tuple[1] == jump_factor[1] - 1):
                    rt_tuple[1].set_as(0)
                    rt_tuple[2].set_as(rt_tuple[2] + 1)
                with self.tik_inst.else_scope():
                    rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 2):
            with self.tik_inst.if_scope(rt_tuple[0] == jump_factor[0] - 1):
                rt_tuple[0].set_as(0)
                rt_tuple[1].set_as(rt_tuple[1] + 1)
            with self.tik_inst.else_scope():
                rt_tuple[0].set_as(rt_tuple[0] + 1)

        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as(rt_tuple[0] + 1)

    def _update_tuple_with_steps(self, axis_num, rt_tuple, jump_factor, jump_factor_mod, base, steps):
        with self.tik_inst.if_scope(axis_num == 7):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])
            rt_tuple[5].set_as((base + steps) / jump_factor_mod[5] % jump_factor[5])
            rt_tuple[6].set_as((base + steps) / jump_factor_mod[6] % jump_factor[6])

        with self.tik_inst.if_scope(axis_num == 6):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])
            rt_tuple[5].set_as((base + steps) / jump_factor_mod[5] % jump_factor[5])

        with self.tik_inst.if_scope(axis_num == 5):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])
            rt_tuple[4].set_as((base + steps) / jump_factor_mod[4] % jump_factor[4])

        with self.tik_inst.if_scope(axis_num == 4):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])
            rt_tuple[3].set_as((base + steps) / jump_factor_mod[3] % jump_factor[3])

        with self.tik_inst.if_scope(axis_num == 3):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])
            rt_tuple[2].set_as((base + steps) / jump_factor_mod[2] % jump_factor[2])

        with self.tik_inst.if_scope(axis_num == 2):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])
            rt_tuple[1].set_as((base + steps) / jump_factor_mod[1] % jump_factor[1])

        with self.tik_inst.if_scope(axis_num == 1):
            rt_tuple[0].set_as((base + steps) / jump_factor_mod[0] % jump_factor[0])

    def _get_n_src_offset(self, tp):
        n_src_offset = self.tik_inst.Scalar("int64", init_value=0)
        with self.tik_inst.if_scope(tp.n_axis_num == 5):
            n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride[0] + \
                                tp.rt_n_tuple[1] * tp.n_jump_stride[1] + \
                                tp.rt_n_tuple[2] * tp.n_jump_stride[2] + \
                                tp.rt_n_tuple[3] * tp.n_jump_stride[3] + \
                                tp.rt_n_tuple[4] * tp.n_jump_stride[4])

        with self.tik_inst.if_scope(tp.n_axis_num == 4):
            n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride[0] + \
                                tp.rt_n_tuple[1] * tp.n_jump_stride[1] + \
                                tp.rt_n_tuple[2] * tp.n_jump_stride[2] + \
                                tp.rt_n_tuple[3] * tp.n_jump_stride[3])

        with self.tik_inst.if_scope(tp.n_axis_num == 3):
            n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride[0] + \
                                tp.rt_n_tuple[1] * tp.n_jump_stride[1] + \
                                tp.rt_n_tuple[2] * tp.n_jump_stride[2])

        with self.tik_inst.if_scope(tp.n_axis_num == 2):
            n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride[0] + \
                                tp.rt_n_tuple[1] * tp.n_jump_stride[1])

        with self.tik_inst.if_scope(tp.n_axis_num == 1):
            n_src_offset.set_as(tp.rt_n_tuple[0] * tp.n_jump_stride[0])

        with self.tik_inst.if_scope(tp.n_axis_num == 0):
            n_src_offset.set_as(0)

        return n_src_offset

    def _get_src_addr(self, tp, ln, lc, lr, bsl):
        src_addr = self.tik_inst.Scalar("int64", init_value=0)

        with self.tik_inst.if_scope(tp.src_axis_num == 7):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] + \
                            tp.rt_src_tuple[5] * tp.src_jump_stride[5] + \
                            tp.rt_src_tuple[6] * tp.src_jump_stride[6] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 6):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] + \
                            tp.rt_src_tuple[5] * tp.src_jump_stride[5] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 5):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] + \
                            tp.rt_src_tuple[4] * tp.src_jump_stride[4] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 4):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] + \
                            tp.rt_src_tuple[3] * tp.src_jump_stride[3] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 3):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] + \
                            tp.rt_src_tuple[2] * tp.src_jump_stride[2] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 2):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] + \
                            tp.rt_src_tuple[1] * tp.src_jump_stride[1] - \
                            bsl + self._get_n_src_offset(tp))

        with self.tik_inst.if_scope(tp.src_axis_num == 1):
            src_addr.set_as(tp.col_offset + lc * tp.col_per_mc + \
                            tp.rt_src_tuple[0] * tp.src_jump_stride[0] - \
                            bsl + self._get_n_src_offset(tp))

        return src_addr

    def _get_dst_addr(self, tp, ln, lc, lr, col_id, bsl, bsu):
        dst_addr = self.tik_inst.Scalar("int64")

        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] + \
                            tp.rt_dst_tuple[5] * tp.dst_jump_stride[5] + \
                            tp.rt_dst_tuple[6] * tp.dst_jump_stride[6] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] + \
                            tp.rt_dst_tuple[5] * tp.dst_jump_stride[5] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] + \
                            tp.rt_dst_tuple[4] * tp.dst_jump_stride[4] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 4):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] + \
                            tp.rt_dst_tuple[3] * tp.dst_jump_stride[3] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 3):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] + \
                            tp.rt_dst_tuple[2] * tp.dst_jump_stride[2] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 2):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] + \
                            tp.rt_dst_tuple[1] * tp.dst_jump_stride[1] - \
                            bsu + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            dst_addr.set_as(tp.n_offset_actual + tp.row_offset + lr * tp.row_per_mr + \
                            tp.rt_dst_tuple[0] * tp.dst_jump_stride[0] - \
                            bsu + ln * tp.right_part_vol)

        return dst_addr

    def _init_dst_addr(self, tp, ln):
        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.init_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.init_dst_tuple[6] * tp.dst_jump_stride[6] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.init_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.init_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 4):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.init_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 3):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.init_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 2):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.init_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.row_offset + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.init_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.row_offset + ln * tp.right_part_vol)

    def _tail_dst_addr_f2t(self, tp, ln):  # need merge
        with self.tik_inst.if_scope(tp.dst_axis_num == 7):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.tail_dst_tuple[5] * tp.dst_jump_stride[5] + \
                                  tp.tail_dst_tuple[6] * tp.dst_jump_stride[6] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 6):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + \
                                  tp.tail_dst_tuple[5] * tp.dst_jump_stride[5] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 5):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + \
                                  tp.tail_dst_tuple[4] * tp.dst_jump_stride[4] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 4):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + \
                                  tp.tail_dst_tuple[3] * tp.dst_jump_stride[3] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 3):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + \
                                  tp.tail_dst_tuple[2] * tp.dst_jump_stride[2] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 2):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  tp.tail_dst_tuple[1] * tp.dst_jump_stride[1] + ln * tp.right_part_vol)

        with self.tik_inst.if_scope(tp.dst_axis_num == 1):
            tp.rt_dst_addr.set_as(tp.n_offset_actual + tp.tail_dst_tuple[0] * tp.dst_jump_stride[0] + \
                                  ln * tp.right_part_vol)

    def _update_dst_addr_f2t(self, tp):
        tp.rt_dst_addr.set_as(tp.rt_dst_addr + tp.col_per_mc * tp.row_per_mr)

    def _update_src_tuple_t2f(self, tp, lr):
        with self.tik_inst.if_scope(tp.src_axis_num == 4):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]) % \
                                      tp.src_jump_stride[0])
            tp.rt_src_tuple[2].set_as(((tp.row_offset + lr * tp.row_per_mr) // \
                                       (tp.src_jump_stride[0] * tp.src_jump_stride[1])) % tp.src_jump_stride[1])
            tp.rt_src_tuple[3].set_as((tp.row_offset + lr * tp.row_per_mr) // \
                                      (tp.src_jump_stride[0] * tp.src_jump_stride[1] * tp.src_jump_stride[2]))

        with self.tik_inst.if_scope(tp.src_axis_num == 3):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]) % \
                                      tp.src_jump_stride[0])
            tp.rt_src_tuple[2].set_as((tp.row_offset + lr * tp.row_per_mr) // \
                                      (tp.src_jump_stride[0] * tp.src_jump_stride[1]))

        with self.tik_inst.if_scope(tp.src_axis_num == 2):
            tp.rt_src_tuple[0].set_as((tp.row_offset + lr * tp.row_per_mr) % tp.src_jump_stride[0])
            tp.rt_src_tuple[1].set_as(((tp.row_offset + lr * tp.row_per_mr) // tp.src_jump_stride[0]))

        with self.tik_inst.if_scope(tp.src_axis_num == 1):
            tp.rt_src_tuple[0].set_as(tp.row_offset + lr * tp.row_per_mr)

    def _update_dst_addr_t2f(self, tp, lr, bsu):
        tp.rt_dst_addr.set_as(tp.rt_dst_addr + lr * tp.row_per_mr - bsu)

    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #                         |                        |
    #             A           |          A             |  B
    # --------------------------------------------------------
    #             C           |          C             |  D
    # --------------------------------------------------------

    # A:   major_col_major_batch
    # B:   tail_col_major_batch
    # C:   major_col_tail_batch
    # D:   tail_col_tail_batch

    def _reorder_s7_b16(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        tp.fp16_offset_1.set_as(248 * 256)
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        with self.tik_inst.if_scope(is_tc == True):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr == True):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        repeat_cnt = tp.col_reorder // EPB16
        with self.tik_inst.for_range(0, tp.row_reorder // EPB16) as loop:
            src_addr_list = [ub_input_fp16[loop * tp.col_reorder * EPB16 + tp.col_reorder * i] for i in range(EPB16)]
            dst_addr_list = [ub_input_fp16[tp.fp16_offset_1 + loop * EPB16 + ROW_UNIT * i] for i in range(EPB16)]

            with self.tik_inst.if_scope(repeat_cnt == 1):
                tp.src_stride_reorder.set_as(0)
                tp.dst_stride_reorder.set_as(0)
            with self.tik_inst.else_scope():
                tp.src_stride_reorder.set_as(1)
                tp.dst_stride_reorder.set_as(ROW_UNIT)

            self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                    tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s7_b32(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        tp.fp16_offset_1.set_as(248 * 256)

        with self.tik_inst.if_scope(is_tc == True):
            tp.col_reorder.set_as(tp.col_tc)
        with self.tik_inst.else_scope():
            tp.col_reorder.set_as(tp.col_per_mc)

        with self.tik_inst.if_scope(is_tr == True):
            tp.row_reorder.set_as(tp.row_tr)
        with self.tik_inst.else_scope():
            tp.row_reorder.set_as(tp.row_per_mr)

        # do hwc to chw transfer
        inner_hw_len = 16 // self.fp16_times
        fp16_inner_hwc_len = 8 * tp.col_reorder * self.fp16_times
        ub_input_fp16 = ub_input.reinterpret_cast_to("float16")

        # first vnchwconv
        src_addr_list = [ub_input_fp16[fp16_inner_hwc_len * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.fp16_offset_1 + EPB16 * i] for i in range(EPB16)]
        repeat_cnt = tp.col_reorder
        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(1)
            tp.dst_stride_reorder.set_as(16)

        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

        # do hwc to chw transfer
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, inner_hw_len) as i:
                self.tik_inst.data_move(ub_input_fp16[i * self.fp16_times * EPB16],
                                        ub_input_fp16[tp.fp16_offset_1 + i * tp.col_reorder * self.fp16_times * EPB16],
                                        0, tp.col_reorder, self.fp16_times, 0, (inner_hw_len - 1) * self.fp16_times)

        # second vnchwconv
        src_addr_list = [ub_input_fp16[EPB16 * i] for i in range(EPB16)]
        dst_addr_list = [ub_input_fp16[tp.fp16_offset_1 + EPB16 * i] for i in range(EPB16)]
        with self.tik_inst.if_scope(repeat_cnt == 1):
            tp.src_stride_reorder.set_as(0)
            tp.dst_stride_reorder.set_as(0)
        with self.tik_inst.else_scope():
            tp.src_stride_reorder.set_as(16)
            tp.dst_stride_reorder.set_as(16)
        self.tik_inst.vnchwconv(False, False, dst_addr_list, src_addr_list, repeat_cnt,
                                tp.dst_stride_reorder, tp.src_stride_reorder)

    def _reorder_s7(self, tp, ub_input, ub_offset, is_tc=False, is_tr=False):
        with self.tik_inst.if_scope(self.fp16_times == 2):  # fp32/int32
            self._reorder_s7_b32(tp, ub_input, ub_offset, is_tc, is_tr)
        with self.tik_inst.else_scope():  # fp16/int16
            self._reorder_s7_b16(tp, ub_input, ub_offset, is_tc, is_tr)

    def _copy_in_major_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, lr, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc // self.ele_per_block)

    def _copy_in_major_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_tr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, lr, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc // self.ele_per_block)

    def _copy_in_tail_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, lr, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc // self.ele_per_block)

    def _copy_in_tail_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        ub_offset.set_as(0)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_tr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, lr, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc // self.ele_per_block)

    def _copy_out_major_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, lc, lr, col_id, 0, 0)],
                                        ub_input[tp.fp16_offset_1 // self.fp16_times + col_id * ROW_UNIT],
                                        0, 1, tp.row_per_mr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_major_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, lc, lr, col_id, 0, tp.back_step_up)],
                                        ub_input[tp.fp16_offset_1 // self.fp16_times + col_id * ROW_UNIT],
                                        0, 1, tp.row_tr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_tail_col_major_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_tc) as col_id:
                with self.tik_inst.if_scope(col_id >= tp.back_step_left):
                    self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, lc, lr, col_id,
                                                                             tp.back_step_left, 0)],
                                            ub_input[tp.fp16_offset_1 // self.fp16_times + col_id * ROW_UNIT],
                                            0, 1, tp.row_per_mr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_out_tail_col_tail_row(self, tp, ub_input, ub_offset, ln, lc, lr):
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_tc) as col_id:
                with self.tik_inst.if_scope(col_id >= tp.back_step_left):
                    self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, lc, lr, col_id,
                                                                             tp.back_step_left, tp.back_step_up)],
                                            ub_input[tp.fp16_offset_1 // self.fp16_times + col_id * ROW_UNIT],
                                            0, 1, tp.row_tr // self.ele_per_block, 0, 0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _reorder_university_f2t(self, tp, ub_input, ub_offset, col_ele_num, row_ele_num, mode):

        # step1. make all elements in the first col
        tp.fp16_offset_1.set_as(self._get_offset_1())
        tp.fp16_offset_2.set_as(self._get_offset_2(self._get_offset_1()))
        ub_input_b16 = ub_input.reinterpret_cast_to("int16")
        src_ele_num_in_fp16 = self._get_src_size()  # minus 16 avoid bank conflict
        src_list = [ub_input_b16[src_ele_num_in_fp16 * i] for i in range(EPB16)]
        dst_list_intermediate = [ub_input_b16[tp.fp16_offset_1 + EPB16 * i] \
                                 for i in range(EPB16)]
        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_intermediate, src_list, ub_offset, EPB16, 1)

        # step2. move output elements together
        with self.tik_inst.if_scope(mode == 0):
            #f2t
            with self.tik_inst.if_scope(tik.all(self.fp16_times == 1, row_ele_num < 32)):
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, row_ele_num) as i:
                        self.tik_inst.vor(128,
                                          ub_input_b16[tp.fp16_offset_2 + i * EPB16],
                                          ub_input_b16[tp.fp16_offset_1 + i * col_ele_num * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num // 8,
                                          row_ele_num,
                                          1,
                                          1,
                                          row_ele_num * 8,
                                          8,
                                          0)

            with self.tik_inst.if_scope(tik.all(self.fp16_times == 1, row_ele_num >= 32)):
                loop_num = self.tik_inst.Scalar("int32")
                tail_num = self.tik_inst.Scalar("int32")
                loop_num.set_as(row_ele_num // 8)
                tail_num.set_as(row_ele_num % 8)
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, loop_num) as i:
                        self.tik_inst.vor(128,
                                          ub_input_b16[tp.fp16_offset_2 + i * 8 * EPB16],
                                          ub_input_b16[tp.fp16_offset_1 + i * col_ele_num * 8 * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num,
                                          1,
                                          col_ele_num,
                                          1,
                                          row_ele_num,
                                          1,
                                          0)
                    with self.tik_inst.if_scope(tail_num != 0):
                        self.tik_inst.vor(tail_num * 16,
                                          ub_input_b16[tp.fp16_offset_2 + loop_num * 8 * EPB16],
                                          ub_input_b16[tp.fp16_offset_1 + loop_num * col_ele_num * 8 * EPB16],
                                          self.ub_input_b16_vor,
                                          col_ele_num,
                                          1,
                                          col_ele_num,
                                          1,
                                          row_ele_num,
                                          1,
                                          0)

            with self.tik_inst.if_scope(self.fp16_times == 2):
                with self.tik_inst.new_stmt_scope(disable_sync=True):
                    with self.tik_inst.for_range(0, row_ele_num) as i:
                        self.tik_inst.data_move(ub_input_b16[tp.fp16_offset_2 + i * self.fp16_times * EPB16],
                                                ub_input_b16[tp.fp16_offset_1 + \
                                                              i * col_ele_num * self.fp16_times * EPB16],
                                                0,
                                                col_ele_num,
                                                self.fp16_times,
                                                0,
                                                row_ele_num * self.fp16_times - self.fp16_times)
        with self.tik_inst.else_scope():
            # t2f
            with self.tik_inst.new_stmt_scope(disable_sync=True):
                with self.tik_inst.for_range(0, col_ele_num) as i:
                    self.tik_inst.data_move(ub_input_b16[tp.fp16_offset_2 + i * row_ele_num * self.fp16_times * EPB16],
                                            ub_input_b16[tp.fp16_offset_1 + i * self.fp16_times * EPB16],
                                            0,
                                            row_ele_num,
                                            self.fp16_times,
                                            col_ele_num * self.fp16_times - self.fp16_times,
                                            0)

        # step3. make all elements in the first col be in memory of contiguous
        src_list_intermediate = [ub_input_b16[tp.fp16_offset_2 + EPB16 * i]\
                for i in range(EPB16)]
        dst_list_finally = [ub_input_b16[self._get_dst_size() * 16 * i] for i in range(EPB16)]

        with self.tik_inst.if_scope(ub_offset == 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, 1, 0, 0)
        with self.tik_inst.if_scope(ub_offset != 1):
            self.tik_inst.vnchwconv(False, False, dst_list_finally, src_list_intermediate, ub_offset, 1, EPB16)

    def _copy_in_major_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, 0, 0)],
                                        0,
                                        1,
                                        tp.col_per_mc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_per_mc // self.ele_per_block)

    def _copy_out_major_col_f2t(self, tp, ub_input, ub_offset, lc):
        self.tik_inst.data_move(self.data_out[tp.rt_dst_addr],
                                ub_input,
                                0,
                                1,
                                (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)
        self._update_dst_addr_f2t(tp)

    def _copy_in_tail_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        ub_offset.set_as(0)
        self._init_src_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.row_per_mr) as line:
                self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                        self.data_in[self._get_src_addr(tp, ln, lc, 0, tp.back_step_left)],
                                        0,
                                        1,
                                        tp.col_tc // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.src_axis_num, tp.rt_src_tuple, tp.src_jump_factor)
                ub_offset.set_as(ub_offset + tp.col_tc // self.ele_per_block)

    def _copy_out_tail_col_f2t(self, tp, ub_input, ub_offset, ln, lc):
        self._tail_dst_addr_f2t(tp, ln)
        self.tik_inst.data_move(self.data_out[tp.rt_dst_addr],
                                ub_input,
                                0,
                                1,
                                (tp.col_tc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)

    def _copy_in_major_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        ub_offset.set_as(0)
        self._update_src_tuple_t2f(tp, lr)
        self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                self.data_in[self._get_src_addr(tp, ln, 0, lr, 0)],
                                0,
                                1,
                                (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block,
                                0,
                                0)
        ub_offset.set_as(ub_offset + (tp.col_per_mc * tp.row_per_mr) // self.ele_per_block)

    def _copy_out_major_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, 0, lr, col_id, 0, 0)],
                                        ub_input[col_id * tp.row_per_mr],
                                        0,
                                        1,
                                        tp.row_per_mr // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _copy_in_tail_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        ub_offset.set_as(0)
        self._tail_src_tuple(tp)
        self.tik_inst.data_move(ub_input[ub_offset * self.ele_per_block],
                                self.data_in[self._get_src_addr(tp, ln, 0, lr, 0)],
                                0,
                                1,
                                (tp.col_per_mc * tp.row_tr) // self.ele_per_block,
                                0,
                                0)
        ub_offset.set_as(ub_offset + (tp.col_per_mc * tp.row_tr) // self.ele_per_block)

    def _copy_out_tail_row_t2f(self, tp, ub_input, ub_offset, ln, lr):
        self._init_dst_tuple(tp)
        with self.tik_inst.new_stmt_scope(disable_sync=True):
            with self.tik_inst.for_range(0, tp.col_per_mc) as col_id:
                self.tik_inst.data_move(self.data_out[self._get_dst_addr(tp, ln, 0, lr, col_id, 0, tp.back_step_up)],
                                        ub_input[col_id * tp.row_tr],
                                        0,
                                        1,
                                        tp.row_tr // self.ele_per_block,
                                        0,
                                        0)
                self._update_tuple(tp.dst_axis_num, tp.rt_dst_tuple, tp.dst_jump_factor)

    def _move_data_last_axis_university(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)

        self._init_n_tuple(tp)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_dst_tuple(tp)

            with self.tik_inst.for_range(0, tp.loop_on_mc) as lc:
                self._init_src_tuple(tp)
                with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                    self._restore_dst_tuple(tp)
                    self._copy_in_major_col_major_row(tp, ub_input, ub_offset, ln, lc, lr)
                    self._reorder_s7(tp, ub_input, ub_offset, False, False)
                    self._copy_out_major_col_major_row(tp, ub_input, ub_offset, ln, lc, lr)

                with self.tik_inst.if_scope(tp.row_tr != 0):
                    self._tail_src_tuple(tp)
                    self._restore_dst_tuple(tp)
                    self._copy_in_major_col_tail_row(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)
                    self._reorder_s7(tp, ub_input, ub_offset, False, True)
                    self._copy_out_major_col_tail_row(tp, ub_input, ub_offset, ln, lc, tp.loop_on_mr)
                self._backup_dst_tuple(tp)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                self._init_src_tuple(tp)
                with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                    self._tail_dst_tuple(tp)
                    self._copy_in_tail_col_major_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)
                    self._reorder_s7(tp, ub_input, ub_offset, True, False)
                    self._copy_out_tail_col_major_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, lr)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                with self.tik_inst.if_scope(tp.row_tr != 0):
                    self._tail_src_tuple(tp)
                    self._tail_dst_tuple(tp)
                    self._copy_in_tail_col_tail_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)
                    self._reorder_s7(tp, ub_input, ub_offset, True, True)
                    self._copy_out_tail_col_tail_row(tp, ub_input, ub_offset, ln, tp.loop_on_mc, tp.loop_on_mr)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    def _move_data_last_axis_fat_2_thin(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_n_tuple(tp)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_dst_tuple(tp)
            self._init_dst_addr(tp, ln)
            with self.tik_inst.for_range(0, tp.loop_on_mc) as lc:
                self._copy_in_major_col_f2t(tp, ub_input, ub_offset, ln, lc)
                self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 0)
                self._copy_out_major_col_f2t(tp, ub_input, ub_offset, lc)

            with self.tik_inst.if_scope(tp.col_tc != 0):
                self._copy_in_tail_col_f2t(tp, ub_input, ub_offset, ln, tp.loop_on_mc)
                self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_tc, tp.row_per_mr, 0)
                self._copy_out_tail_col_f2t(tp, ub_input, ub_offset, ln, tp.loop_on_mc)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    def _move_data_last_axis_thin_2_fat(self, tp, ub_input_64):
        ub_offset = self.tik_inst.Scalar("int32")  # unit : block
        ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
        self._init_n_tuple(tp)

        with self.tik_inst.for_range(0, tp.loop_on_n) as ln:
            self._init_src_tuple(tp)
            with self.tik_inst.for_range(0, tp.loop_on_mr) as lr:
                self._copy_in_major_row_t2f(tp, ub_input, ub_offset, ln, lr)
                self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_per_mr, 1)
                self._copy_out_major_row_t2f(tp, ub_input, ub_offset, ln, lr)

            with self.tik_inst.if_scope(tp.row_tr != 0):
                self._copy_in_tail_row_t2f(tp, ub_input, ub_offset, ln, tp.loop_on_mr)
                self._reorder_university_f2t(tp, ub_input, ub_offset, tp.col_per_mc, tp.row_tr, 1)
                self._copy_out_tail_row_t2f(tp, ub_input, ub_offset, ln, tp.loop_on_mr)
            self._update_tuple(tp.n_axis_num, tp.rt_n_tuple, tp.n_jump_factor)

    # -------------------------------------------------------------------------------------------------
    #                                    scenario_8
    # -------------------------------------------------------------------------------------------------
    def _move_data_s8(self, ub_input_64):
        # 4 255 3 8 -> 3 255 4 8
        if api_check_support("tik.vcopy", "int16"):
            ub_input = ub_input_64.reinterpret_cast_to(self.x_dtype)
            tik_inst = self.tik_inst
            tik_inst.data_move(ub_input, self.data_in, 0, 1, 4 * 255 * 3, 0, 0)
            with tik_inst.for_range(0, 3) as i:
                tik_inst.vcopy(4 * 8,
                               ub_input[32 * 1024 + i * 255 * 4 * 8],
                               ub_input[0 + i * 8],
                               255,
                               1,
                               255 * 3,
                               4,
                               3,
                               "counter")
            tik_inst.data_move(self.data_out, ub_input[32 * 1024], 0, 1, 4 * 255 * 3, 0, 0)

    def _do_tiling_s0(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        self.tik_inst.data_move(ub_input_64[0],
                                self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                0, 1, per_core_len // ELE_NUM_PER_BLOCK_INT64 + 1, 0, 0)
        tp = self.TilingParamS0(tiling_reg_list, ub_input_64_t, ub_input_64)
        return tp

    def _do_tiling_s1(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        self.tik_inst.data_move(ub_input_64[0],
                                self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                0, 1, per_core_len // ELE_NUM_PER_BLOCK_INT64 + 1, 0, 0)
        tp = self.TilingParamS1(tiling_reg_list, ub_input_64_t, ub_input_64, self.tik_inst)
        return tp

    def _do_tiling_s2(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        self.tik_inst.data_move(ub_input_64[0],
                                self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                0, 1, per_core_len // ELE_NUM_PER_BLOCK_INT64 + 1, 0, 0)
        tp = self.TilingParamS2(tiling_reg_list, ub_input_64_t, ub_input_64, self.tik_inst)
        return tp

    def _do_tiling_s3(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        self.tik_inst.data_move(ub_input_64[0],
                                self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                0, 1, per_core_len // ELE_NUM_PER_BLOCK_INT64 + 1, 0, 0)
        tp = self.TilingParamS3(tiling_reg_list, ub_input_64_t, ub_input_64, self.tik_inst)
        return tp

    def _do_tiling_s7(self, block_idx, tiling_reg_list, ub_input_64_t, ub_input_64, fixed_len, per_core_len):
        self.tik_inst.data_move(ub_input_64[0],
                                self.data_tiling[TILING_HEAD_LEN + fixed_len + block_idx * per_core_len],
                                0, 1, per_core_len // ELE_NUM_PER_BLOCK_INT64 + 1, 0, 0)
        tp = self.TilingParamS7(tiling_reg_list, ub_input_64_t, ub_input_64, self.tik_inst)
        return tp

    def _decode_tiling_head(self):
        scenario = self.tik_inst.Scalar("int64")
        fixed_len = self.tik_inst.Scalar("int64")
        per_core_len = self.tik_inst.Scalar("int64")
        sub_scenario = self.tik_inst.Scalar("int64")
        scenario.set_as(self.ub_input_64_t[0])
        fixed_len.set_as(self.ub_input_64_t[1])
        per_core_len.set_as(self.ub_input_64_t[2])
        sub_scenario.set_as(self.ub_input_64_t[3])
        return scenario, fixed_len, per_core_len, sub_scenario

    def compute_tiling(self):
        """
        execution function
        """
        scenario, fixed_len, per_core_len, sub_scenario = self._decode_tiling_head()

        with self.tik_inst.for_range(0, CORE_NUM, block_num=CORE_NUM) as block_idx:
            with self.tik_inst.if_scope(scenario == 7):
                tp = self._do_tiling_s7(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                        self.ub_input_64, fixed_len, per_core_len)
                with self.tik_inst.if_scope(sub_scenario == 0):
                    self._move_data_last_axis_university(tp, self.ub_input_64)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(sub_scenario == 1):
                        self._move_data_last_axis_fat_2_thin(tp, self.ub_input_64)
                    with self.tik_inst.else_scope():
                        self._move_data_last_axis_thin_2_fat(tp, self.ub_input_64)
            with self.tik_inst.else_scope():
                with self.tik_inst.if_scope(scenario == 1):
                    tp = self._do_tiling_s1(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                            self.ub_input_64, fixed_len, per_core_len)
                    self._move_data_s1(tp, self.ub_input_64)
                with self.tik_inst.else_scope():
                    with self.tik_inst.if_scope(tik.any(scenario == 2, scenario == 6)):
                        tp = self._do_tiling_s2(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                self.ub_input_64, fixed_len, per_core_len)
                        self._move_data_s2(tp, self.ub_input_64)
                    with self.tik_inst.else_scope():
                        with self.tik_inst.if_scope(scenario == 3):
                            tp = self._do_tiling_s3(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                    self.ub_input_64, fixed_len, per_core_len)
                            self._move_data_s3(tp, self.ub_input_64)
                        with self.tik_inst.if_scope(scenario == 0):  # scenario == 0
                            tp = self._do_tiling_s0(block_idx, self.tiling_reg_list, self.ub_input_64_t,
                                                    self.ub_input_64, fixed_len, per_core_len)
                            self._move_data_s0(tp, self.ub_input_64)
                        with self.tik_inst.if_scope(scenario == 8):
                            self._move_data_s8(self.ub_input_64)

    def compute(self, input_list):
        """
        entrance function
        """
        self.compute_tiling()
        tbe_context.get_context().add_compile_info("vars", {
            "ub_size": UB_SIZE // BLOCK_SIZE, "core_num": CORE_NUM, "dtype": self.x_dtype})
        # this "global_variable_link" flag suggest ccec.py do link without "-r" option
        # which will result in global variable in cce file with wrong address
        tbe_context.get_context().add_compile_info("global_variable_link", True)
        self.tik_inst.BuildCCE(kernel_name=self.kernel_name,
                               inputs=input_list,
                               outputs=[self.data_out],
                               flowtable=[self.data_tiling])
        return {"compile_info": tbe_context.get_context().get_compile_info()}


@register_operator("Transpose")
def transpose(x, perm, y, kernel_name="transpose"):
    """
    do transpose by perm attribute

    Parameters
    ----------
    x : dict
        shape and dtype of input
    perm : list or tuple
        permutation of the dimension of tensor
    y : dict
        shape and dtype of output, the dtype should be same as input
    kernel_name : str
        kernel name, default value is "transpose"

    Returns
    -------
    compile info
    """
    x_dtype = x.get("dtype").lower()
    p_dtype = perm.get("dtype").lower()
    y_dtype = y.get("dtype").lower()
    tik_inst = tik.Tik()
    data_in = tik_inst.Tensor(x_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "x")
    data_perm = tik_inst.Tensor(p_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "perm")
    data_out = tik_inst.Tensor(y_dtype, (MAX_INT64_VALUE,), tik.scope_gm, "y")
    data_workspace = tik_inst.Tensor(y_dtype, (1024, ), tik.scope_gm, "data_workspace", is_workspace=True)
    data_tiling = tik_inst.Tensor("int64", (TILING_MAX_SIZE_GM,), tik.scope_gm, "data_tiling")
    tensor_list = [data_in, data_perm, data_out, data_workspace, data_tiling]
    input_list = [data_in, data_perm]
    transpose_instance = Transpose(tik_inst, x_dtype, tensor_list, kernel_name)
    return transpose_instance.compute(input_list)
