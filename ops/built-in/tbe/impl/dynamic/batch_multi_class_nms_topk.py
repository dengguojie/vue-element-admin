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
batch_multi_class_nms_topk
"""
import math
from impl.util.platform_adapter import tik
from impl import common_util


# 'pylint: disable=too-many-arguments
def _vconcat(instance, dst, src, mode, cnt, dst_offset=0,
             src_offset=0):
    """
    _vconcat
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    with instance.if_scope(repeat_255 > 0):
        with instance.for_range(0, repeat_255) as i:
            instance.vconcat(dst[dst_offset + i * 255 * 16 * 8],
                             src[src_offset + i * 255 * 16], 255, mode)
    if repeat_remain > 0:
        instance.vconcat(
            dst[dst_offset + 255 * 16 * 8 * repeat_255],
            src[src_offset + 255 * 16 * repeat_255],
            repeat_remain, mode)


# 'pylint: disable=too-many-arguments
def _vrpsort16(instance, dst, src, cnt, dst_offset=0, src_offset=0):
    """
    _vrpsort16
    """
    repeat_255 = cnt // (16 * 255)
    repeat_remain = (cnt - repeat_255 * 16 * 255) // 16
    if repeat_255 > 0:
        with instance.for_range(0, repeat_255) as i:
            instance.vrpsort16(dst[dst_offset + i * 255 * 16 * 8],
                               src[src_offset + i * 255 * 16 * 8], 255)
    if repeat_remain > 0:
        instance.vrpsort16(dst[dst_offset + 255 * 16 * 8 * repeat_255],
                           src[src_offset + 255 * 16 * 8 * repeat_255], repeat_remain)


def _vrpsort16_scalar(
        instance,
        dst,
        src,
        cnt,
        dst_offset=0,
        src_offset=0):
    repeat_255 = instance.Scalar(
        init_value=cnt // (16 * 255), dtype="int32", name="repeat_255")
    repeat_remain = instance.Scalar(
        init_value=(
            cnt - repeat_255 * 16 * 255) // 16,
        dtype="int32",
        name="repeat_remain")
    with instance.if_scope(repeat_255 > 0):
        with instance.for_range(0, repeat_255, name='i0') as i:
            instance.vrpsort16(
                dst[dst_offset + i * 255 * 16 * 8], src[src_offset + i * 255 * 16 * 8], 255)
    with instance.if_scope(repeat_remain > 0):
        instance.vrpsort16(dst[dst_offset + 255 * 16 * 8 * repeat_255], src[src_offset + 255 *
                            16 * 8 * repeat_255], repeat_remain)


# 'pylint: disable=too-many-arguments
def _merge_region(instance, out_ub, dst, src, rows, cols):
    """
    merge_region
    """
    cols_padding = ((cols + 15) // 16) * 16
    with instance.for_range(0, rows, name='merge_i0') as i:
        result_ub = _merge_recur(instance,
                                 out_ub,
                                 dst,
                                 src,
                                 cols, (cols + 15) // 16,
                                 1,
                                 region_offset=i * cols_padding * 8)
    return result_ub


def _merge_region_scalar(instance, loop_times, dst, src, rows, cols):
    """
    merge_region
    """
    cols_padding = instance.Scalar(dtype='int32', name='cols_padding')
    cols_padding.set_as(((cols + 15) // 16) * 16)
    with instance.for_range(0, rows, name='merge_i0') as i:
        _merge_loop_scalar(instance,
                           loop_times,
                           src,
                           dst,
                           cols,
                           (cols + 15) // 16,
                           region_offset=i * cols_padding * 8)


def _merge_loop_scalar(
        tik_instance,
        loop_times,
        src_ub,
        dst_ub,
        last_dim,
        total_region_list,
        region_offset=0):
    """
    _merge_loop_scalar
    """
    region_list_reg = tik_instance.Scalar(
        init_value=total_region_list,
        dtype="int32",
        name="region_list_reg")
    with tik_instance.for_range(0, loop_times) as i:
        with tik_instance.if_scope(i % 2 == 0):
            _merge_recur_scalar(
                tik_instance,
                src_ub,
                dst_ub,
                last_dim,
                region_list_reg,
                (i + 1),
                region_offset)

        with tik_instance.else_scope():
            _merge_recur_scalar(
                tik_instance,
                dst_ub,
                src_ub,
                last_dim,
                region_list_reg,
                (i + 1),
                region_offset)
        region_list_reg.set_as((region_list_reg + 3) // 4)


# 'pylint:disable=locally-disabled,too-many-arguments,too-many-locals,too-many-statements
def _merge_recur(instance,
                 out_ub,
                 dst_ub,
                 src_ub,
                 last_dim,
                 total_region_list,
                 level,
                 region_offset=0):
    """
    _merge_recur
    merge multi sorted region proposal list to one sorted region proposal list
    """

    # vmrgsort4 can merger at most 4 sorted region list
    def is_next_to_last_merge():
        """
        is_next_to_last_merge
        """
        return 1 < math.ceil(total_region_list / 4) <= 4

    loops = total_region_list // 4
    remain = total_region_list % 4

    if is_next_to_last_merge() and dst_ub.name == out_ub.name:
        dst_ub = instance.Tensor(out_ub.dtype, out_ub.shape,
                                 scope=tik.scope_ubuf, name="ub_merge_recur")

    merge_n0 = 16 * (4 ** (level - 1))
    merge_n1 = merge_n0
    merge_n2 = merge_n0
    merge_n3 = merge_n0
    merge_repeat = loops
    need_tail_process = False
    if loops > 0 and remain == 0:
        if merge_n0 * 4 * loops > last_dim:
            merge_repeat = loops - 1
            n012 = merge_n0 + merge_n1 + merge_n2
            merge_left = last_dim - ((merge_n0 * 4 * (loops - 1)) + n012)
            need_tail_process = True
    if merge_repeat > 0:
        ub_offset = region_offset
        src_list = (src_ub[ub_offset],
                    src_ub[ub_offset + merge_n0 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8])
        element_count_list = (merge_n0, merge_n1, merge_n2, merge_n3)
        valid_bit = 15
        instance.vmrgsort4(dst_ub[ub_offset], src_list, element_count_list,
                           False, valid_bit, merge_repeat)

    if need_tail_process:
        tail_offset = 4 * merge_n0 * merge_repeat * 8
        ub_offset = region_offset + tail_offset
        src_list = (src_ub[ub_offset],
                    src_ub[ub_offset + merge_n0 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8 +
                           merge_n2 * 8])
        element_count_list = (merge_n0, merge_n1, merge_n2, merge_left)
        valid_bit = 15
        instance.vmrgsort4(dst_ub[ub_offset], src_list, element_count_list,
                           False, valid_bit, repeat_times=1)

    if loops > 0:
        offset = 4 * loops * 16 * (4 ** (level - 1))
    else:
        offset = 0

    if remain == 3:
        merge_n0 = 16 * (4 ** (level - 1))
        merge_n1 = merge_n0
        merge_n2 = last_dim - (offset + merge_n0 + merge_n1)
        ub_offset = region_offset + offset * 8
        src_list = (src_ub[ub_offset],
                    src_ub[ub_offset + merge_n0 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8])
        element_count_list = (merge_n0, merge_n1, merge_n2, 0)
        valid_bit = 2 ** remain - 1
        instance.vmrgsort4(dst_ub[ub_offset], src_list, element_count_list,
                           False, valid_bit, repeat_times=1)
    elif remain == 2:
        merge_n0 = 16 * (4 ** (level - 1))
        merge_n1 = last_dim - (offset + merge_n0)
        ub_offset = region_offset + offset * 8
        src_list = (src_ub[ub_offset],
                    src_ub[ub_offset + merge_n0 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8],
                    src_ub[ub_offset + merge_n0 * 8 + merge_n1 * 8 + merge_n2 * 8])
        element_count_list = (merge_n0, merge_n1, 0, 0)
        valid_bit = 2 ** remain - 1
        instance.vmrgsort4(dst_ub[ub_offset], src_list, element_count_list,
                           False, valid_bit, repeat_times=1)
    elif remain == 1:
        merge_n0 = last_dim - offset
        num_blocks_write = (merge_n0 * 8 * common_util.get_data_size(
            src_ub.dtype) + 31) // 32
        ub_offset = region_offset + offset * 8
        instance.data_move(dst_ub[ub_offset], src_ub[ub_offset], 0, 1,
                           num_blocks_write, 0, 0)

    next_total_region_list = math.ceil(total_region_list / 4)
    if next_total_region_list <= 1:
        return dst_ub

    if is_next_to_last_merge():
        src_ub = out_ub

    return _merge_recur(instance, out_ub, src_ub, dst_ub,
                        last_dim, next_total_region_list, level + 1,
                        region_offset)


# 'pylint:disable=locally-disabled,too-many-arguments,too-many-locals
def _merge_recur_scalar(tik_instance,
                        src_ub,
                        dst_ub,
                        last_dim,
                        total_region_list,
                        level,
                        region_offset=0):
    """
    _merge_recur
    merge multi sorted region proposal list to one sorted region proposal list
    """

    # vmrgsort4 can merger at most 4 sorted region list
    mid_var = total_region_list // 4
    loops = tik_instance.Scalar(
        init_value=(mid_var),
        dtype="int32",
        name="loops")
    remain = tik_instance.Scalar(
        init_value=(
            total_region_list -
            loops *
            4),
        dtype="int32",
        name="remain")

    level_reg = tik_instance.Scalar(
        init_value=level,
        dtype="int32",
        name="level_reg")

    merge_n0_reg = tik_instance.Scalar(init_value=1, name="merge_n0_reg")
    merge_n0_reg.set_as(merge_n0_reg << (4 + 2 * level_reg - 2))

    merge_n1_reg = tik_instance.Scalar(
        init_value=merge_n0_reg, name="merge_n1_reg")
    merge_n2_reg = tik_instance.Scalar(
        init_value=merge_n0_reg, name="merge_n2_reg")
    merge_n3_reg = tik_instance.Scalar(
        init_value=merge_n0_reg, name="merge_n3_reg")
    merge_repeat = tik_instance.Scalar(init_value=loops, name="merge_repeat")

    need_tail_process = tik_instance.Scalar(
        init_value=0, name="need_tail_process")
    merge_left = tik_instance.Scalar(init_value=0, name="merge_left")
    n012 = tik_instance.Scalar(init_value=0, name="n012")

    with tik_instance.if_scope(tik.all(loops > 0, remain == 0)):
        with tik_instance.if_scope(merge_n0_reg * 4 * loops > last_dim):
            merge_repeat.set_as(loops - 1)
            n012.set_as(merge_n0_reg + merge_n1_reg + merge_n2_reg)
            merge_left.set_as(
                last_dim - ((merge_n0_reg * 4 * (loops - 1)) + n012))
            need_tail_process.set_as(1)

    with tik_instance.if_scope(merge_repeat > 0):
        src_list = [
            src_ub[region_offset], src_ub[region_offset + merge_n0_reg * 8],
            src_ub[region_offset + merge_n0_reg * 8 + merge_n1_reg * 8],
            src_ub[region_offset + merge_n0_reg * 8 + merge_n1_reg * 8 + merge_n2_reg * 8]]
        tik_instance.vmrgsort4(
            dst_ub[region_offset],
            src_list,
            (merge_n0_reg,
             merge_n1_reg,
             merge_n2_reg,
             merge_n3_reg),
            False,
            15,
            merge_repeat)

    with tik_instance.if_scope(need_tail_process == 1):
        tail_offset = tik_instance.Scalar(init_value=(4 * merge_n0_reg * merge_repeat * 8), name="tail_offset")
        src_list = [src_ub[region_offset + tail_offset], src_ub[region_offset + tail_offset +
                    merge_n0_reg * 8], src_ub[region_offset + tail_offset + merge_n0_reg * 8 +
                    merge_n1_reg * 8], src_ub[region_offset + tail_offset + merge_n0_reg * 8 + merge_n1_reg * 8 +
                                              merge_n2_reg * 8]]
        tik_instance.vmrgsort4(dst_ub[region_offset + tail_offset], src_list, (merge_n0_reg, merge_n1_reg, merge_n2_reg,
                                merge_left), False, 15, 1)

    offset_reg = tik_instance.Scalar(dtype="int32", init_value=0, name="offset_reg")
    offset_reg.set_as(4 * loops * merge_n0_reg)

    with tik_instance.if_scope(remain == 3):
        merge_n2_reg.set_as(last_dim - (offset_reg + merge_n0_reg + merge_n1_reg))

        src_list = [src_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg * 8 + merge_n0_reg *
                    8], src_ub[region_offset + offset_reg * 8 + merge_n0_reg * 8 + merge_n1_reg * 8], src_ub[0]]

        tik_instance.vmrgsort4(dst_ub[region_offset + offset_reg * 8], src_list,
                               (merge_n0_reg, merge_n1_reg, merge_n2_reg, 16), False, 7, 1)
    with tik_instance.if_scope(remain == 2):
        merge_n1_reg.set_as(last_dim - (offset_reg + merge_n0_reg))
        src_list = [src_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg *
                    8 + merge_n0_reg * 8], src_ub[0], src_ub[0]]

        tik_instance.vmrgsort4(dst_ub[region_offset + offset_reg * 8], src_list,
                               (merge_n0_reg, merge_n1_reg, 16, 16), False, 3, 1)

    with tik_instance.if_scope(remain == 1):
        merge_n0_reg.set_as(last_dim - offset_reg)
        num_blocks_write_reg = tik_instance.Scalar(init_value=(
            (merge_n0_reg * 16 + 31) // 32), name="merge_n1_reg")
        tik_instance.data_move(dst_ub[region_offset + offset_reg * 8], src_ub[region_offset + offset_reg * 8],
                               0, 1, num_blocks_write_reg, 0, 0)


def trans_to_proposol(instance, proposol, boxes_ub, scores_ub, cnt):
    """
    trans_to_proposol
    """
    _vconcat(instance, proposol, scores_ub, mode=4, cnt=cnt)
    _vconcat(instance, proposol, boxes_ub[cnt * 0], mode=0, cnt=cnt)
    _vconcat(instance, proposol, boxes_ub[cnt * 1], mode=1, cnt=cnt)
    _vconcat(instance, proposol, boxes_ub[cnt * 2], mode=2, cnt=cnt)
    _vconcat(instance, proposol, boxes_ub[cnt * 3], mode=3, cnt=cnt)
    return proposol


def sort_within_ub(instance, src, cols):
    """
    sort_within_ub
    """
    with instance.new_stmt_scope():
        dst = instance.Tensor(src.dtype, src.shape, scope=tik.scope_ubuf, name="ub_sort_within")
        _vrpsort16(instance, dst, src, cnt=cols)
        if cols > 16:
            result_ub = _merge_region(instance, out_ub=src, dst=src, src=dst, rows=1, cols=cols)
        else:
            result_ub = dst

        if result_ub.name != src.name:
            burst = math.ceil(cols * src.shape[1] * common_util.get_data_size(src.dtype) / 32)
            instance.data_move(src, result_ub, 0, 1, burst, 0, 0)
    return src


def sort_within_ub_scalar(instance, src, cols, loop_times):
    """
    sort_within_ub
    """
    with instance.new_stmt_scope():
        dst = instance.Tensor(src.dtype, src.shape, scope=tik.scope_ubuf, name="dst")
        _vrpsort16_scalar(instance, dst, src, cnt=cols)
        with instance.if_scope(cols > 16):
            _merge_region_scalar(instance, loop_times, dst=src, src=dst, rows=1, cols=cols)
        with instance.if_scope(tik.any(cols <= 16, loop_times % 2 == 0)):
            burst = instance.Scalar(dtype='int32', name='burst')
            burst.set_as((cols * src.shape[1] * common_util.get_data_size(src.dtype) - 1) // 32 + 1)
            instance.data_move(src, dst, 0, 1, burst, 0, 0)


def sort_with_ub(instance, src_ub_list, dst_ub, sorted_num):
    """
    sort_with_ub
    """
    ub_count = len(src_ub_list)
    if ub_count < 4:
        src_ub_list += [src_ub_list[-1]] * (4 - ub_count)
    element_count_list = [sorted_num] * 4
    valid_bit = 2 ** ub_count - 1
    instance.vmrgsort4(dst_ub, src_ub_list, element_count_list, False, valid_bit, repeat_times=1)
