# Copyright 2019 Huawei Technologies Co., Ltd
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
strided_slice_for_last_dim
"""
import functools

from te import tik
from te import platform as tbe_platform
from impl import common_util
from impl.util.util_tik_comm_func import floor_align
from impl.util.util_tik_comm_func import ceil_align
from impl.util.util_tik_comm_func import ceil_div
from impl.util.util_tik_comm_func import gm2ub
from impl.util.util_tik_comm_func import ub2gm


# constant 8
NUM_EIGHT = 8

VNCHW_BLOCK_SIZE = 512
VNCHW_ELEMENT_FP16 = VNCHW_BLOCK_SIZE // 2


def strided_slice_last_dim(input_shape, dtype, output_shape, begin, end, stride, kernel_name):
    """
    strided slice for only last dim to slice

    Returns
    -------
    tik_instance: tik_instance
    """
    if _can_do_with_vnchw_conv(input_shape, dtype, begin, end, stride):
        return strided_slice_last_dim_with_vnchw_conv(input_shape, dtype, begin, end, stride, kernel_name)

    return strided_slice_last_dim_with_scalar(input_shape, dtype, output_shape, begin, end, stride, kernel_name)

# pylint: disable=invalid-name, too-many-locals, unused-argument
# pylint: disable=too-many-arguments, unused-variable, too-many-return-statements
# pylint: disable=too-many-branches, too-many-statements
def strided_slice_last_dim_with_scalar(input_shape, dtype, output_shape, begin, end, stride, kernel_name):
    """
    strided slice for only last dim to slice with scalar

    Returns
    -------
    tik_instance: tik_instance
    """
    vmul_support = tbe_platform.cce_conf.api_check_support("tik.vmul", "float32")
    if not vmul_support:
        return False
    tik_instance = tik.Tik()
    aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
    input_size = 1
    for i in input_shape:
        input_size = input_size * i
    if dtype == "float16":
        type_block_num = 16
    elif dtype == "float32":
        type_block_num = 8
    else:
        return False

    output_size = 1
    output_length = len(output_shape)
    for i in range(0, output_length):
        output_size = output_size * output_shape[i]

    if len(output_shape) != len(input_shape):
        consecutive_num = 1
    else:
        consecutive_num = output_shape[len(output_shape) - 1]

    ouput_core_data_num = output_size // aicore_num
    input_core_data_num = input_size // aicore_num

    output_group_1 = ouput_core_data_num
    input_group_1 = ouput_core_data_num // consecutive_num * input_shape[len(input_shape) - 1]
    output_group_2 = 0
    input_group_2 = 0

    tail_core_num = 0
    total_core_num = aicore_num
    tail_flag = False
    if output_size % aicore_num != 0 or ouput_core_data_num % type_block_num != 0:
        aicore_num = aicore_num - 1
        ouput_core_data_num = output_size // aicore_num
        input_core_data_num = input_size // aicore_num
        if output_size % aicore_num != 0 or ouput_core_data_num % type_block_num != 0:
            if output_size // aicore_num == 0:
                aicore_num = 1
            output_group_1 = ouput_core_data_num
            output_group_1 = (output_group_1 // type_block_num) * type_block_num

            output_group_2 = output_size - (output_group_1 * aicore_num)
            input_group_1 = output_group_1 // consecutive_num * input_shape[len(input_shape) - 1]
            input_group_2 = input_size - (input_group_1 * aicore_num)

            tail_core_num = 1
            total_core_num = aicore_num + tail_core_num
            tail_flag = True
        else:
            output_group_1 = ouput_core_data_num
            input_group_1 = ouput_core_data_num // consecutive_num * input_shape[len(input_shape) - 1]
            output_group_2 = 0
            input_group_2 = 0
            total_core_num = aicore_num

    def _get_ub_block_num():
        """
        get the ub_size for dtype, get the block_size for dtype
        """
        ub_size_bytes = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) - 1024
        # Convert byts to Bytes
        dtype_bytes_size = tbe_platform.cce_intrin.get_bit_len(dtype) // 8
        ub_number = ub_size_bytes // dtype_bytes_size
        block_number = 32 // dtype_bytes_size

        return ub_number, block_number

    def _get_split_axis(input_size, output_size):
        ub_number, block_number = _get_ub_block_num()
        total_num = (input_size + output_size)
        result = 1
        find_flag = False
        for result in range(1, output_size):
            if (total_num // result) <= ub_number and (output_size % result) == 0 and (
                    (output_size // result) % type_block_num) == 0 and \
                    (output_size // consecutive_num) % result == 0:
                find_flag = True
                break

        return result, find_flag

    # internal split factor
    split_factor_group_1, find_flag_1 = _get_split_axis(input_group_1, output_group_1)

    split_factor_group_2, find_flag_2 = _get_split_axis(input_group_2, output_group_2)

    if not find_flag_1:
        return False

    if split_factor_group_2 != 1:
        return False

    if output_group_1 % consecutive_num != 0 or output_group_2 % consecutive_num != 0:
        return False

    if input_group_1 % input_shape[len(input_shape) - 1] != 0:
        return False

    if input_group_2 % input_shape[len(input_shape) - 1] != 0:
        return False

    if input_group_1 == 0 or output_group_1 == 0:
        return False

    if consecutive_num > 100:
        return False

    # can't change
    start_num = begin[len(begin) - 1]
    # can't change
    len_burst = input_shape[len(input_shape) - 1]

    # gm_size
    output_data = tik_instance.Tensor(dtype, (output_size,), name="output_data",
                                      scope=tik.scope_gm)
    input_data = tik_instance.Tensor(dtype, (input_size,), name="input_data",
                                     scope=tik.scope_gm)

    input_ub_size0 = ((input_group_2 + type_block_num - 1) // type_block_num) * type_block_num
    output_ub_size0 = ((output_group_2 + type_block_num - 1) // type_block_num) * type_block_num

    input_ub_size1 = input_size // aicore_num // split_factor_group_1
    output_ub_size1 = output_size // aicore_num // split_factor_group_1

    input_ub_size = max(input_ub_size0, input_ub_size1)
    output_ub_size = max(output_ub_size0, output_ub_size1)

    dtype_size = common_util.get_data_size(dtype)
    total_ub_size = tik.Dprofile().get_unified_buffer_size()
    if input_ub_size + output_ub_size > total_ub_size // dtype_size:
        return False

    # ub_size change
    input_data_ub = tik_instance.Tensor(dtype,
                                        (input_ub_size,),
                                        name="input_data_ub",
                                        scope=tik.scope_ubuf)
    # ub_size change
    output_data_ub = tik_instance.Tensor(dtype,
                                         (output_ub_size,),
                                         name="output_data_ub",
                                         scope=tik.scope_ubuf)

    if (output_group_2 % type_block_num) == 0:
        group_2_length = output_group_2 // type_block_num
    else:
        group_2_length = (output_group_2 // type_block_num) + 1

    if (input_group_2 % type_block_num) == 0:
        group_2_input_length = input_group_2 // type_block_num
    else:
        group_2_input_length = (input_group_2 // type_block_num) + 1

    with tik_instance.for_range(0, total_core_num,
                                block_num=total_core_num) as total_cycle:
        with tik_instance.if_scope(total_cycle < aicore_num):
            with tik_instance.for_range(0, split_factor_group_1)as axis_outer:
                if ((input_group_1 // split_factor_group_1) % type_block_num) == 0:
                    stride_length = input_group_1 // split_factor_group_1 // type_block_num
                else:
                    stride_length = (input_group_1 // split_factor_group_1 // type_block_num) + 1
                tik_instance.data_move(input_data_ub,
                                       input_data[(total_cycle * input_group_1)
                                                  + (axis_outer * input_group_1
                                                     // split_factor_group_1)],
                                       0, 1,
                                       stride_length,
                                       0, 0)
                max_num = output_group_1 // consecutive_num // split_factor_group_1
                with tik_instance.for_range(0, max_num)as group:
                    for cur_num in range(0, consecutive_num):
                        output_data_ub[group * consecutive_num + cur_num].set_as(
                            input_data_ub[group * len_burst + start_num + cur_num])
                gm_deviation = axis_outer * output_group_1 // split_factor_group_1
                output_data_src1 = total_cycle * output_group_1 + gm_deviation

                if ((output_group_1 // split_factor_group_1) % type_block_num) == 0:
                    output_stride_length = (output_group_1 // split_factor_group_1) // type_block_num
                else:
                    output_stride_length = ((output_group_1 // split_factor_group_1) // type_block_num) + 1
                tik_instance.data_move(output_data[output_data_src1],
                                       output_data_ub, 0, 1,
                                       output_stride_length,
                                       0, 0)
        if tail_flag:
            with tik_instance.else_scope():
                with tik_instance.for_range(0, split_factor_group_2)as axis_outer:
                    input_deviation = axis_outer * input_group_2 // split_factor_group_2
                    stride_input_dev = group_2_input_length // split_factor_group_2
                    tik_instance.data_move(input_data_ub, input_data[
                        (aicore_num * input_group_1) + input_deviation],
                                           0, 1, stride_input_dev, 0, 0)
                    max_num = output_group_2 // consecutive_num // split_factor_group_2
                    with tik_instance.for_range(0, max_num)as group:
                        for cur_num in range(0, consecutive_num):
                            output_data_ub[group * consecutive_num + cur_num].set_as(
                                input_data_ub[group * len_burst + start_num + cur_num])
                    tik_instance.data_move(output_data[aicore_num * output_group_1 + (
                        axis_outer * output_group_2 // split_factor_group_2)], output_data_ub,
                                           0, 1, group_2_length, 0, 0)

    tik_instance.BuildCCE(kernel_name, inputs=[input_data], outputs=[output_data])

    return tik_instance


def _can_do_with_vnchw_conv(input_shape, dtype, begin, end, stride):
    """
    Determining if it can use vnchw_conv to do.
    """
    dtype_size = common_util.get_data_size(dtype)
    if dtype_size % 2 != 0:
        return False

    float16_type_size = common_util.get_data_size("float16")
    input_inner_dims = input_shape[-1] * dtype_size // float16_type_size
    need_ub_size = (16 * input_inner_dims * 2) * float16_type_size
    element_each_block = common_util.constant.BLOCK_SIZE // float16_type_size

    total_ub_size = tik.Dprofile().get_unified_buffer_size()
    if len(input_shape) < 2:
        return False

    input_out_dims = functools.reduce(lambda x, y: x * y, input_shape[0:-1])
    output_shape = list(map(lambda x, y, z: (x - y) // z, end, begin, stride))
    output_out_dims = functools.reduce(lambda x, y: x * y, output_shape[0:-1])
    if input_out_dims != output_out_dims:
        return False

    # in this case, scalar make better performance
    if input_shape[-1] // output_shape[-1] > 32:
        return False

    # if not double buffer the performance will not better than do with scalar
    double_buffer_mutil_times = 2
    return need_ub_size * element_each_block * double_buffer_mutil_times <= total_ub_size and input_out_dims > 2


# pylint: disable=invalid-name, too-many-locals, unused-argument
# pylint: disable=too-many-arguments, unused-variable, too-many-return-statements
# pylint: disable=too-many-branches, too-many-statements
def strided_slice_last_dim_with_vnchw_conv(input_shape, dtype, begin, end, stride, kernel_name):
    """
    strided slice for only last dim to slice with vnchw_conv
    """
    dtype_size = common_util.get_data_size(dtype)
    output_shape = list(map(lambda x, y, z: (x - y) // z, end, begin, stride))
    float16_type_size = common_util.get_data_size("float16")
    multi_times = dtype_size // float16_type_size
    input_inner_dims = input_shape[-1] * multi_times
    output_inner_dims = (end[-1] - begin[-1]) // stride[-1] * multi_times
    out_dims = functools.reduce(lambda x, y: x * y, input_shape[0:-1])
    begin_value = begin[-1] * multi_times
    end_value = end[-1] * multi_times
    profile = tik.Dprofile()
    total_ub_length = profile.get_unified_buffer_size() // float16_type_size
    need_ub_size_one_row = 16 * input_inner_dims * 2
    element_each_block = common_util.constant.BLOCK_SIZE // float16_type_size
    thread_num = 2
    ub_size = floor_align(total_ub_length // 2 // thread_num, element_each_block)
    max_rows_in_ub = floor_align(ub_size // (input_inner_dims * 16), element_each_block)
    aicore_num = profile.get_aicore_num()
    output_32byes_align_rows = element_each_block
    if element_each_block % output_inner_dims == 0:
        output_32byes_align_rows = element_each_block // output_inner_dims
    elif output_inner_dims % element_each_block == 0:
        output_32byes_align_rows = 1

    rows_each_core = ceil_align(ceil_div(out_dims, aicore_num), output_32byes_align_rows)
    aicore_num_used = ceil_div(out_dims, rows_each_core)
    tail_rows = out_dims % rows_each_core
    if aicore_num_used == 1:
        rows_each_core = out_dims
        tail_rows = 0

    tik_instance = tik.Tik()
    input_gm = tik_instance.Tensor("float16", (out_dims, input_inner_dims), scope=tik.scope_gm, name="input_gm")
    output_gm = tik_instance.Tensor("float16", (out_dims, output_inner_dims), scope=tik.scope_gm, name="output_gm")

    def slice_each_core(blk_idx, to_do_rows, thread_num):
        thread_num = min(to_do_rows, thread_num)
        repeat_times = ceil_align(ceil_div(to_do_rows, max_rows_in_ub), thread_num)
        rows_each_repeat = ceil_align(to_do_rows // repeat_times, output_32byes_align_rows)
        if rows_each_repeat > max_rows_in_ub:
            rows_each_repeat = floor_align(to_do_rows // repeat_times, output_32byes_align_rows)
        repeat_times = ceil_div(to_do_rows, rows_each_repeat)
        repeat_tail_count = to_do_rows % rows_each_repeat

        input_addr = rows_each_core * input_inner_dims * blk_idx
        output_addr = rows_each_core * output_inner_dims * blk_idx

        roll_back_rows = tik_instance.Scalar(dtype="int64", name="roll_back_rows", init_value=0)
        curr_rows = tik_instance.Scalar(dtype="int64", name="curr_rows", init_value=rows_each_repeat)
        if repeat_tail_count * output_inner_dims % element_each_block != 0:
            roll_back_rows.set_as(ceil_align(repeat_tail_count, element_each_block) - repeat_tail_count)

        with tik_instance.new_stmt_scope():
            thread_num = min(repeat_times, thread_num)
            with tik_instance.for_range(0, repeat_times, thread_num=thread_num) as repeat_idx:
                rows_idx = tik_instance.Scalar(dtype="int64", name="rows_idx", init_value=repeat_idx * rows_each_repeat)
                if repeat_tail_count * output_inner_dims % element_each_block != 0 and repeat_times != 1:
                    with tik_instance.if_scope(repeat_idx == repeat_times - 1):
                        rows_idx.set_as(rows_idx - roll_back_rows)
                        curr_rows.set_as(repeat_tail_count + roll_back_rows)

                input_ub = tik_instance.Tensor("float16", (ub_size, ), scope=tik.scope_ubuf, name="input_ub")
                vnchw_conv_ub = tik_instance.Tensor("float16", (ub_size, ), scope=tik.scope_ubuf, name="vnchw_conv_ub")
                gm2ub(tik_instance, input_ub, input_gm[input_addr + rows_idx * input_inner_dims],
                      curr_rows * input_inner_dims)
                dst_list = [vnchw_conv_ub[i * element_each_block] for i in range(16)]
                src_list = [input_ub[i * element_each_block] for i in range(16)]
                vnchw_conv_repeat_times = ceil_div(curr_rows * input_inner_dims, 16)
                with tik_instance.if_scope(vnchw_conv_repeat_times == 1):
                    tik_instance.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
                with tik_instance.else_scope():
                    tik_instance.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 16, 1)
                tik_instance.data_move(input_ub, vnchw_conv_ub[begin_value * 16],
                                       0, curr_rows, output_inner_dims, input_inner_dims - output_inner_dims, 0)

                dst_list = [vnchw_conv_ub[i * element_each_block] for i in range(16)]
                src_list = [input_ub[i * element_each_block] for i in range(16)]
                vnchw_conv_repeat_times = ceil_div(curr_rows * output_inner_dims, 16)
                with tik_instance.if_scope(vnchw_conv_repeat_times == 1):
                    tik_instance.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 0, 0)
                with tik_instance.else_scope():
                    tik_instance.vnchwconv(False, False, dst_list, src_list, vnchw_conv_repeat_times, 1, 16)

                ub2gm(tik_instance, output_gm[output_addr + rows_idx * output_inner_dims],
                      vnchw_conv_ub, curr_rows * output_inner_dims)

    with tik_instance.for_range(0, aicore_num_used, block_num=aicore_num_used) as blk_idx:
        if tail_rows != 0:
            with tik_instance.if_scope(blk_idx < aicore_num_used -1):
                slice_each_core(blk_idx, rows_each_core, thread_num)
            with tik_instance.else_scope():
                slice_each_core(blk_idx, tail_rows, thread_num)
        else:
            slice_each_core(blk_idx, rows_each_core, thread_num)

    opt_config = {"out_of_bound_sync_check": True,
                  "enable_const_fold": True}
    tik_instance.BuildCCE(kernel_name, inputs=[input_gm], outputs=[output_gm], config=opt_config)

    return tik_instance
