#!/usr/bin/python
# -*- coding: utf-8 -*-
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
sort
"""

# pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from te.platform.fusion_manager import fusion_manager
from te import tik
from topi.cce import util
from functools import reduce as functools_reduce
from impl.util.platform_adapter import error_manager_vector

PROPOSAL_NUM = 8
FP16_BYTE = 2
MAX_NUM = 7040
BLOCK = 16
REPEAT_MAX = 255
DATA_MAX = 4080


@fusion_manager.register("sort")
def check(x, y1, y2, axis, kernel_name):
    """
    Function: Check parameters (eg: shape dtype etc).
    Modify : 2020-08-03
    """
    util.check_kernel_name(kernel_name)

    shape = y1.get("shape")
    dtype = y1.get("dtype").lower()
    util.check_dtype_rule(dtype, ("float16"))
    util.check_shape_rule(shape)

    shape = y2.get("shape")
    dtype = y2.get("dtype").lower()
    util.check_dtype_rule(dtype, ("int32"))
    util.check_shape_rule(shape)

    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    util.check_dtype_rule(dtype, ("float16"))
    util.check_shape_rule(shape)

    if axis == -1:
        axis = len(shape) - 1

    if axis != len(shape) - 1:
        error_manager_vector.raise_err_specific_reson("sort", "Dim should take the last one.")

    num = shape[axis]

    if num > MAX_NUM:
        error_manager_vector.raise_err_specific_reson("sort", "Num in dim is too big (>7040).")

    return shape, dtype, num


def vbs16(tik_instance, num, total, input_ub, descending):
    """
    Function: Sort every 16 numsi in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num: The number of effective object.
    total: The number of all object (16 alignment).
    input_ub: UB
    ----------
    """
    Max = tik_instance.Scalar('float16', init_value=65504)
    Min = tik_instance.Scalar('float16', init_value=-65504)
    # Add ineffective object for 16 alignment
    if descending:
        with tik_instance.for_range(0, total - num) as i:
            input_ub[(num + i) * PROPOSAL_NUM + 4].set_as(Min)
    else:
        with tik_instance.for_range(0, total - num) as i:
            input_ub[(num + i) * PROPOSAL_NUM + 4].set_as(Max)

    # dest position in UB
    dest_pos_ub = total * PROPOSAL_NUM
    n_repeat_total = total // BLOCK

    if n_repeat_total > REPEAT_MAX:
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=REPEAT_MAX)
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub + REPEAT_MAX * BLOCK * PROPOSAL_NUM],
                               src=input_ub[REPEAT_MAX * BLOCK * PROPOSAL_NUM],
                               repeat_times=n_repeat_total - REPEAT_MAX)
    else:
        tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)

    return input_ub, dest_pos_ub


def merge4(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 4 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM],
                input_ub[
                    src_pos_ub + (offset + num_list[index] + num_list[index + 1] + num_list[index + 2]) * PROPOSAL_NUM]]

    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
    # merge 4 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
    # update the lists info : Merge the four element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 4])
    a = num_list[:index + 1:]
    b = num_list[index + 4::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge3(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 3 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * PROPOSAL_NUM], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
    # merge 3 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
    # update the lists info : Merge the three element values and record them in a(num_list)
    num_list[index] = sum(num_list[index:index + 3])
    a = num_list[:index + 1:]
    b = num_list[index + 3::]
    a.extend(b)
    offset += a[index]

    return a, input_ub, offset


def merge2(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 2 lists in UB.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    num_list: record the lists info
    offset: used space
    src_pos_ub, dest_pos_ub: position info
    input_ub: UB
    ----------
    """
    src_list = [input_ub[src_pos_ub + offset * PROPOSAL_NUM],
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM],
                input_ub[0], input_ub[0]]
    src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
    # merge 2 lists
    tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * PROPOSAL_NUM], src_list, src_list_lengths,
                           if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

    # update the lists info : Merge the two element values and record them in num_list
    num_list[index] += num_list[index + 1]
    del num_list[index + 1]
    offset += num_list[index]

    return num_list, input_ub, offset


def vms4(tik_instance, total, input_ub, dest_pos_ub):
    """
    Function: Merge all lists into one.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    total: The number of all object (16 alignment).
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    ----------
    """
    # record the lists info
    length = total // BLOCK
    num_list = [BLOCK] * length

    # over 4096
    if length > 256:
        # leftset rightset : num_list's valid room
        input_ub, _, num_list1 = vms4core(tik_instance, input_ub, dest_pos_ub, 0, length // 2, num_list)
        input_ub, _, num_list2 = vms4core(tik_instance, input_ub, dest_pos_ub, length // 2, length, num_list)

        num_list1.extend(num_list2)

        src_pos_ub, dest_pos_ub = dest_pos_ub, 0
        _, input_ub, _ = merge2(tik_instance, num_list1, input_ub, 0, src_pos_ub, 0, dest_pos_ub)
        return input_ub, dest_pos_ub

    input_ub, dest_pos_ub, num_list = vms4core(tik_instance, input_ub, dest_pos_ub, 0, length, num_list)
    return input_ub, dest_pos_ub


def vms4core(tik_instance, input_ub, dest_pos_ub, leftset, rightset, num_list):
    """
    Function: Merge core.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    leftset, rightset: The valid room.
    num_list : Lists info
    ----------
    """
    src_pos_ub = 0
    num_list = num_list[leftset:rightset]
    offset_temp = leftset * BLOCK
    while len(num_list) > 1:
        src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
        index = 0
        offset = offset_temp
        while True:
            res = len(num_list) - index
            if res > 3:
                num_list, input_ub, offset = merge4(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 3:
                num_list, input_ub, offset = merge3(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 2:
                num_list, input_ub, offset = merge2(tik_instance, num_list, input_ub,
                                                    offset, src_pos_ub, index, dest_pos_ub)
            elif res == 1:
                tik_instance.data_move(input_ub[dest_pos_ub + offset * PROPOSAL_NUM],
                                       input_ub[src_pos_ub + offset * PROPOSAL_NUM], 0, 1,
                                       num_list[index] * PROPOSAL_NUM // BLOCK, 0, 0)
            else:
                break
            index += 1

    return input_ub, dest_pos_ub, num_list


def moveout(tik_instance, descending, num_16, num, data_out, offset_out, input_ub, dest_pos_ub, data_indices,
            version):
    """
    Function: Move UB to GM, and trans y2 from fp16 to int32.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    descending, offset_out, num_16, num, dest_pos_ub : for index compute
    data_out, input_ub, data_indices : for data move
    ----------
    """
    int_list = tik_instance.Tensor("int32", [num_16], name="data_indices_ub_list", scope=tik.scope_ubuf)
    src_pos_ub = num_16 * PROPOSAL_NUM if dest_pos_ub == 0 else 0
    # ascend
    with tik_instance.if_scope(descending is False):
        # data is continuous in GM & gather scattered data together
        with tik_instance.for_range(0, num) as i2:
            input_ub[i2 + src_pos_ub].set_as(input_ub[(num_16 - 1 - i2) * PROPOSAL_NUM + 4 + dest_pos_ub])
            input_ub[i2 + src_pos_ub + num_16].set_as(input_ub[(num_16 - 1 - i2) * PROPOSAL_NUM + dest_pos_ub])

        # conv indices (float16->int32) , and move from UB to GM
        if num_16 > DATA_MAX:
            tik_instance.vec_conv(BLOCK, "round", int_list, input_ub[src_pos_ub + num_16], REPEAT_MAX, 2, 1)
            tik_instance.vec_conv(BLOCK, "round", int_list[DATA_MAX], input_ub[src_pos_ub + num_16 + DATA_MAX],
                                  (num_16 % DATA_MAX) // BLOCK, 2, 1)
        else:
            tik_instance.vec_conv(BLOCK, "round", int_list, input_ub[src_pos_ub + num_16], num_16 // BLOCK, 2, 1)

        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[offset_out], input_ub[src_pos_ub], 0, 1, num_16 // BLOCK, 0, 0)
        tik_instance.data_move(data_indices[offset_out], int_list, 0, 1, num_16 // 8, 0, 0)

    # descend
    with tik_instance.else_scope():
        # data is continuous in GM & gather scattered data together
        if version == "mini":
            with tik_instance.for_range(0, num) as i2:
                input_ub[i2 + src_pos_ub].set_as(input_ub[i2 * PROPOSAL_NUM + 4 + dest_pos_ub])
                input_ub[i2 + src_pos_ub + num_16].set_as(input_ub[i2 * PROPOSAL_NUM + dest_pos_ub])
        elif version == "cloud":
            if num_16 > DATA_MAX:
                tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub], REPEAT_MAX, 4)
                tik_instance.vextract(input_ub[src_pos_ub + DATA_MAX], input_ub[dest_pos_ub + DATA_MAX * PROPOSAL_NUM],
                                      (num_16 % DATA_MAX) // BLOCK, 4)

                tik_instance.vextract(input_ub[src_pos_ub + num_16], input_ub[dest_pos_ub], REPEAT_MAX, 0)
                tik_instance.vextract(input_ub[src_pos_ub + num_16 + DATA_MAX],
                                      input_ub[dest_pos_ub + DATA_MAX * PROPOSAL_NUM],
                                      (num_16 % DATA_MAX) // BLOCK, 0)
            else:
                tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub], num_16 // BLOCK, 4)
                tik_instance.vextract(input_ub[src_pos_ub + num_16], input_ub[dest_pos_ub], num_16 // BLOCK, 0)
        else:
            error_manager_vector.raise_err_specific_reson("sort", "Unexcepted version.")

        if num_16 > DATA_MAX:
            tik_instance.vec_conv(BLOCK, "round", int_list, input_ub[src_pos_ub + num_16], REPEAT_MAX, 2, 1)
            tik_instance.vec_conv(BLOCK, "round", int_list[DATA_MAX], input_ub[src_pos_ub + num_16 + DATA_MAX],
                                  (num_16 % DATA_MAX) // BLOCK, 2, 1)
        else:
            tik_instance.vec_conv(BLOCK, "round", int_list, input_ub[src_pos_ub + num_16], num_16 // BLOCK, 2, 1)
        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[offset_out], input_ub[src_pos_ub], 0, 1, num_16 // BLOCK, 0, 0)
        tik_instance.data_move(data_indices[offset_out], int_list, 0, 1, num_16 // 8, 0, 0)

    return data_out, data_indices


def sort_compute(tik_instance, dtype, num_16, i0, descending, num, data_out, data_indices, input_gm):
    """
    Function: sortcompute in UB.
    Modify : 2020-08-03

    Attention : This way is unstable (can't compare two scalar).
    Init base parameters
    Parameters
    ----------
    dtype, num_16, i0, descending, num, distance, shape, big_distance, L : for index compute
    data_out, data_indices, input_gm : for data move
    ----------
    """
    input_ub = tik_instance.Tensor(dtype, [num_16 * PROPOSAL_NUM * 2], name="input_ub", scope=tik.scope_ubuf)
    version = tik.Dprofile().get_product_name()
    offset_in = i0 * num
    offset_out = i0 * num_16
    dest_pos_ub = num_16 * PROPOSAL_NUM
    # 1. Move data from OUT to UB
    tik_instance.data_move(input_ub[dest_pos_ub], input_gm[offset_in], 0, 1, num_16 // BLOCK, 0, 0)

    if version == "cloud":
        idx = tik_instance.Scalar(dtype="float32", init_value=num)
        with tik_instance.for_range(0, num) as i2:
            idx.set_as(idx - 1)
            input_ub[(num - 1 - i2) * PROPOSAL_NUM].set_as(idx)
    elif version == "mini":
        data_out_ub_ = tik_instance.Tensor(dtype, [BLOCK], name="data_out_ub_", scope=tik.scope_ubuf)
        data_indices_ub_int_ = tik_instance.Tensor("int32", [BLOCK], name="data_indices_ub_int_", scope=tik.scope_ubuf)
        with tik_instance.for_range(0, num) as i2:
            data_indices_ub_int_.set_as(num - 1 - i2)
            tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
            input_ub[(num - 1 - i2) * PROPOSAL_NUM].set_as(data_out_ub_[0])
    else:
        error_manager_vector.raise_err_specific_reson("sort", "Unexcepted version.")

    if num_16 > DATA_MAX:
        tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], REPEAT_MAX, 4)
        tik_instance.vconcat(input_ub[DATA_MAX * PROPOSAL_NUM], input_ub[dest_pos_ub + DATA_MAX],
                             (num_16 % DATA_MAX) // BLOCK, 4)
    else:
        tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], num_16 // BLOCK, 4)
    # 2. vbs16
    input_ub, dest_pos_ub = vbs16(tik_instance, num, num_16, input_ub, descending)
    # 3. vms4
    input_ub, dest_pos_ub = vms4(tik_instance, num_16, input_ub, dest_pos_ub)
    # 4. Move Data from UB to OUT
    data_out, data_indices = moveout(tik_instance, descending, num_16, num, data_out, offset_out, input_ub,
                        dest_pos_ub, data_indices, version)

    return data_out, data_indices


@util.check_input_type(dict, dict, dict, int, bool, str)
def sort(x, y1, y2, axis=-1, descending=False, kernel_name="sort"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.
    Modify : 2020-08-03

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y1): dict
        data of output
    indices(y2): dict
        data of indices
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    shape, dtype, num = check(x, y1, y2, axis, kernel_name)
    allnum = functools_reduce(lambda x, y: x * y, shape)

    tik_instance = tik.Tik(tik.Dprofile())

    rounds = allnum // num

    num_16 = (num + BLOCK - 1) // BLOCK * BLOCK

    input_gm = tik_instance.Tensor(dtype, shape, name="x", scope=tik.scope_gm)
    data_out = tik_instance.Tensor(dtype, [rounds * num_16], name="data_out", scope=tik.scope_gm, is_workspace=True)
    data_indices = tik_instance.Tensor("int32", [rounds * num_16], name="data_indices", scope=tik.scope_gm,
                                       is_workspace=True)
    data_out_ = tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm)
    data_indices_ = tik_instance.Tensor("int32", shape, name="data_indices_", scope=tik.scope_gm)

    available_aicore_num = tik.Dprofile().get_aicore_num()
    used_aicore_num = available_aicore_num if rounds > available_aicore_num else rounds
    batch_num_per_aicore = rounds // used_aicore_num
    batch_tail = rounds % used_aicore_num

    with tik_instance.for_range(0, used_aicore_num, block_num=used_aicore_num) as i:
        with tik_instance.for_range(0, batch_num_per_aicore) as k:
            data_out, data_indices = sort_compute(tik_instance, dtype, num_16, i + k * used_aicore_num, descending,
                                                  num, data_out, data_indices, input_gm)
        with tik_instance.if_scope(i < batch_tail):
            data_out, data_indices = sort_compute(tik_instance, dtype, num_16,
                                                  batch_num_per_aicore * used_aicore_num + i,
                                                  descending, num, data_out, data_indices, input_gm)
            
    availabel_ub_size = tik.Dprofile().get_unified_buffer_size()
    threadNum = 2 if rounds > 1 else 1
    threadNum = 1 if num_16 * 12 > availabel_ub_size else threadNum
    with tik_instance.for_range(0, rounds, thread_num=threadNum) as i:
        float_ub = tik_instance.Tensor("float16", [num_16], name="float_ub", scope=tik.scope_ubuf)
        int_ub = tik_instance.Tensor("int32", [num_16], name="int_ub", scope=tik.scope_ubuf)

        with tik_instance.for_range(0, rounds) as i:
            tik_instance.data_move(float_ub[0], data_out[i * num_16], 0, 1, num_16 // 16, 0, 0)
            tik_instance.data_move(data_out_[i * num], float_ub[0], 0, 1, num_16 // 16, 0, 0)

            tik_instance.data_move(int_ub[0], data_indices[i * num_16], 0, 1, num_16 // 8, 0, 0)
            tik_instance.data_move(data_indices_[i * num], int_ub[0], 0, 1, num_16 // 8, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_gm], outputs=[data_out_, data_indices_])

    return tik_instance
