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
sort_v2
"""
import functools
from te import tik
from te.utils import para_check
import te.platform as tbe_platform

RESERVE_SIZE = 2 * 1024
PROPOSAL_NUM = 8


def check(x, y, axis, kernel_name):
    """
    Function: Check parameters (eg: shape dtype etc).
    Modify : 2020-11-11
    """
    para_check.check_kernel_name(kernel_name)

    shape = y.get("shape")
    dtype = y.get("dtype").lower()
    para_check.check_dtype(dtype, ("float16"), param_name="y")
    para_check.check_shape(shape, param_name="y")

    shape = x.get("shape")
    dtype = x.get("dtype").lower()
    para_check.check_dtype(dtype, ("float16"), param_name="x")
    para_check.check_shape(shape, param_name="x")

    if axis == -1:
        axis = len(shape) - 1

    if axis != len(shape) - 1:
        raise RuntimeError("Dim should take the last one.")

    num = shape[axis]

    return shape, dtype, num


def merge4(tik_instance, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
    """
    Function: Merge 4 lists in UB.
    Modify : 2020-11-11

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
    Modify : 2020-11-11

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
    Modify : 2020-11-11

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
                input_ub[src_pos_ub + (offset + num_list[index]) * PROPOSAL_NUM], input_ub[0], input_ub[0]]

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
    Modify : 2020-11-11

    Init base parameters
    Parameters
    ----------
    num: The number of effective object.
    total: The number of all object (16 alignment).
    input_ub: UB
    dest_pos_ub: The dest position in UB.
    ----------
    """
    # record the lists info
    length = total // 16
    num_list = [16] * length

    src_pos_ub = 0
    while len(num_list) > 1:
        src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
        index = 0
        offset = 0
        while True:
            res = len(num_list) - index
            if res > 3:
                num_list, input_ub, offset = merge4(tik_instance, num_list, input_ub, offset, src_pos_ub, index,
                                                    dest_pos_ub)
            elif res == 3:
                num_list, input_ub, offset = merge3(tik_instance, num_list, input_ub, offset, src_pos_ub, index,
                                                    dest_pos_ub)
            elif res == 2:
                num_list, input_ub, offset = merge2(tik_instance, num_list, input_ub, offset, src_pos_ub, index,
                                                    dest_pos_ub)
            elif res == 1:
                tik_instance.data_move(input_ub[dest_pos_ub + offset * PROPOSAL_NUM],
                                       input_ub[src_pos_ub + offset * PROPOSAL_NUM], 0, 1,
                                       num_list[index] * PROPOSAL_NUM // 16, 0, 0)
            else:
                break
            index += 1

    return input_ub, dest_pos_ub


def pick(tik_instance, descending, temp, offset, i0, k, num, data_out, data_out_, input_ub, num_gm):
    """
    Function: pick value from proposal.
    Modify : 2020-11-11
    ----------
    descending, temp, offset, i0, k : for index compute
    data_out, input_ub2, num_gm : for data move
    ----------
    """
    # dest position in UB
    dest_pos_ub = k * PROPOSAL_NUM
    repeat_times = k // 16
    # ascend
    with tik_instance.if_scope(descending is False):
        tik_instance.data_move(input_ub[0], temp[offset + (num % k - k) * PROPOSAL_NUM], 0, 1,
                               (k * PROPOSAL_NUM) // 16, 0, 0)
        # data is continuous in GM & gather scattered data together
        with tik_instance.for_range(0, k) as i2:
            input_ub[i2 + dest_pos_ub].set_as(input_ub[(k - 1 - i2) * PROPOSAL_NUM + 4])

        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[i0 * k], input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)
    
        with tik_instance.for_range(1, num_gm) as i:
            tik_instance.data_move(input_ub[0], temp[offset + (k * (i - 1) + num % k) * PROPOSAL_NUM], 0, 1,
                                   (k * PROPOSAL_NUM) // 16, 0, 0)
            # data is continuous in GM & gather scattered data together
            with tik_instance.for_range(0, k) as i2:
                input_ub[i2 + dest_pos_ub].set_as(input_ub[(k - 1 - i2) * PROPOSAL_NUM + 4])

            # move output (float16) from UB to GM
            tik_instance.data_move(data_out_[i0 * num + k * (num_gm - 1 - i)], input_ub[dest_pos_ub], 0, 1,
                                   repeat_times, 0, 0)

    # descend
    with tik_instance.else_scope():
        with tik_instance.for_range(0, num_gm - 1) as i:
            tik_instance.data_move(input_ub[0], temp[offset + k * i * PROPOSAL_NUM], 0, 1,
                                   (k * PROPOSAL_NUM) // 16, 0, 0)
            # data is continuous in GM & gather scattered data together
            tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, 4)

            # move output (float16) from UB to GM
            tik_instance.data_move(data_out_[i0 * num + k * i], input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)
        
        tik_instance.data_move(input_ub[0], temp[offset + k * (num_gm - 1) * PROPOSAL_NUM], 0, 1,
                               (k * PROPOSAL_NUM) // 16, 0, 0)
        # data is continuous in GM & gather scattered data together
        tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, 4)

        # move output (float16) from UB to GM
        tik_instance.data_move(data_out[i0 * k], input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    return data_out, data_out_


def sort_in_ub(tik_instance, input_ub, num, i, k, input_gm, temp, index, offset):
    """
    Function: sort in ub.
    Modify : 2020-11-16

    Init base parameters
    Parameters
    ----------
    num, i, k, index, offset : for index compute
    input_gm, temp, dtype : for data move
    ----------
    """
    # dest position in UB
    dest_pos_ub = k * PROPOSAL_NUM
    repeat_times = k // 16
    # 1. Move data from OUT to UB
    tik_instance.data_move(input_ub[dest_pos_ub], input_gm[index + i * k], 0, 1, repeat_times, 0, 0)

    with tik_instance.if_scope(num < (i + 1) * k):
        # aline for k
        aline = k - num % k
        Min = tik_instance.Scalar('float16', init_value=-65504)
        # Add ineffective object for 16 alignment
        with tik_instance.for_range(0, aline % 16) as j:
            input_ub[dest_pos_ub + num % k + j].set_as(Min)
        # Add ineffective object for k alignment
        with tik_instance.if_scope(aline > 15):
            tik_instance.vec_dup(16, input_ub[dest_pos_ub + num % k + aline % 16], Min, aline // 16, 1)

    tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, 4)

    # 2. vbs16
    tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)

    # 3. vms4
    input_ub, dest_pos_ub = vms4(tik_instance, k, input_ub, dest_pos_ub)

    # 4. Move Data from UB to OUT
    tik_instance.data_move(temp[offset + i * k * PROPOSAL_NUM], input_ub[dest_pos_ub], 0, 1,
                           k * PROPOSAL_NUM // 16, 0, 0)

    return temp


def sort_in_gm(tik_instance, k, temp, num_gm, batchsize, input_ub, offset):
    """
    Function: sort in gm.
    Modify : 2020-11-16

    Init base parameters
    Parameters
    ----------
    k, num_gm, batchsize, offset : for index compute
    temp, input_ub : for data move
    ----------
    """
    src_pos_ub = tik_instance.Scalar("int32")
    dest_pos_ub = tik_instance.Scalar("int32")

    with tik_instance.for_range(0, num_gm - 1) as tail:
        src_pos_ub.set_as(0)
        dest_pos_ub.set_as(batchsize * PROPOSAL_NUM)

        tik_instance.data_move(input_ub[src_pos_ub + k * PROPOSAL_NUM], temp[offset], 0, 1, (k * PROPOSAL_NUM) // 16, 0,
                               0)
        with tik_instance.for_range(1, num_gm - tail) as i:
            tik_instance.data_move(input_ub[src_pos_ub], temp[offset + k * i * PROPOSAL_NUM], 0, 1,
                                   (k * PROPOSAL_NUM) // 16, 0, 0)

            tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                   [input_ub[src_pos_ub], input_ub[src_pos_ub + k * PROPOSAL_NUM],
                                    input_ub[0], input_ub[0]], [k, k, 0, 0], if_exhausted_suspension=False,
                                   valid_bit="0011", repeat_times=1)

            tik_instance.data_move(temp[offset + k * (i - 1) * PROPOSAL_NUM], input_ub[dest_pos_ub], 0, 1,
                                   (k * PROPOSAL_NUM) // 16, 0, 0)
            
            dest_pos_ub.set_as(src_pos_ub)
            src_pos_ub.set_as(batchsize * PROPOSAL_NUM - dest_pos_ub)

        # Move Data from UB to GM
        tik_instance.data_move(temp[offset + k * (num_gm - tail - 1) * PROPOSAL_NUM],
                               input_ub[src_pos_ub + k * PROPOSAL_NUM], 0, 1, (k * PROPOSAL_NUM) // 16, 0, 0)

    return temp


@tbe_platform.fusion_manager.fusion_manager.register("sort_v2")
def sort_compute(tik_instance, dtype, num, i0, used_core_num, descending, k, data_out, data_out_,
                 input_gm, temp, num_gm, batchsize, totalnum):
    """
    Function: compute.
    Modify : 2020-11-11

    Init base parameters
    Parameters
    ----------
    dtype, total, i0, descending, num, distance, shape : for index compute
    data_out, input_gm : for data move
    ----------
    """

    offset = (i0 % used_core_num) * num_gm * k * PROPOSAL_NUM

    index = i0 * num

    input_ub = tik_instance.Tensor(dtype, [batchsize * PROPOSAL_NUM * 2], name="input_ub", scope=tik.scope_ubuf)

    # SORT IN UB
    with tik_instance.for_range(0, num_gm) as i:
        temp = sort_in_ub(tik_instance, input_ub, num, i, k, input_gm, temp, index, offset)

    # SORT IN GM
    temp = sort_in_gm(tik_instance, k, temp, num_gm, batchsize, input_ub, offset)

    # Pick Data from GM to GM
    data_out, data_out_ = pick(tik_instance, descending, temp, offset, i0, k, num, data_out, data_out_,
                               input_ub, num_gm)

    return data_out, data_out_


@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
def sort_v2(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    """
    Function: Sorts the elements of the input tensor along a given dimension in ascending order by value.
    Modify : 2020-11-11

    Init base parameters
    Parameters
    ----------
    input(x): dict
        data of input
    output(y): dict
        data of output
    dim(axis): int
    descending: bool
    kernel_name: str
        the name of the operator
    ----------
    """
    shape, dtype, num = check(x, y, axis, kernel_name)
    allnum = functools.reduce(lambda x, y: x * y, shape)

    tik_instance = tik.Tik(tik.Dprofile('cloud'))
    batchsize = (tik.Dprofile().get_unified_buffer_size() - RESERVE_SIZE) // 32 // 32 * 32
    k = batchsize // 2
    num_gm = num // k
    num_gm = num_gm if (num % k) == 0 else (num_gm + 1)
    totalnum = num_gm * k
    rounds = allnum // num

    input_gm = tik_instance.Tensor(dtype, shape, name="input_gm", scope=tik.scope_gm)
    data_out = tik_instance.Tensor(dtype, [rounds * k], name="data_out", scope=tik.scope_gm, is_workspace=True)
    data_out_ = tik_instance.Tensor(dtype, shape, name="data_out_", scope=tik.scope_gm)

    available_core_num = tik.Dprofile().get_aicore_num()
    used_core_num = available_core_num if rounds > available_core_num else rounds
    batch_num_per_core_process = rounds // used_core_num
    batch_tail = rounds % used_core_num

    temp = tik_instance.Tensor(dtype, [used_core_num * num_gm * k * PROPOSAL_NUM], name="temp", scope=tik.scope_gm,
                               is_workspace=True)

    with tik_instance.for_range(0, used_core_num, block_num=used_core_num) as i:
        with tik_instance.for_range(0, batch_num_per_core_process) as j:
            data_out, data_out_ = sort_compute(tik_instance, dtype, num, i + j * used_core_num, used_core_num,
                                               descending, k, data_out, data_out_, input_gm, temp, num_gm,
                                               batchsize, totalnum)

        with tik_instance.if_scope(i < batch_tail):
            data_out, data_out_ = sort_compute(tik_instance, dtype, num, batch_num_per_core_process * used_core_num + i,
                                               used_core_num, descending, k, data_out, data_out_, input_gm, temp,
                                               num_gm, batchsize, totalnum)

    # fine tune data in GM
    data_ub = tik_instance.Tensor(dtype, [k], name="data_ub", scope=tik.scope_ubuf)
    tail_ub = tik_instance.Tensor(dtype, [k], name="tail_ub", scope=tik.scope_ubuf)
    temp_ub = tik_instance.Tensor(dtype, [k], name="temp_ub", scope=tik.scope_ubuf)
    valid_data = num % k
    for i in range(rounds):
        if valid_data > 15:
            tik_instance.data_move(data_ub[0], data_out[i * k], 0, 1, valid_data // 16, 0, 0)
            tik_instance.data_move(data_out_[i * num + (num_gm - 1) * k], data_ub[0], 0, 1, valid_data // 16, 0, 0)
        if valid_data % 16 != 0:
            tik_instance.data_move(tail_ub[0], data_out[i * k + valid_data // 16 * 16], 0, 1, 1, 0, 0)
            tik_instance.data_move(temp_ub[0], data_out_[i * num + (num_gm - 1) * k + valid_data // 16 * 16],
                                   0, 1, 1, 0, 0)
            # Add ineffective object for 16 alignment
            for idx in range(valid_data % 16):
                temp_ub[idx].set_as(tail_ub[idx])
            tik_instance.data_move(data_out_[i * num + (num_gm - 1) * k + valid_data // 16 * 16], temp_ub[0],
                                   0, 1, 1, 0, 0)

    tik_instance.BuildCCE(kernel_name=kernel_name, inputs=[input_gm], outputs=[data_out_])

    return tik_instance
