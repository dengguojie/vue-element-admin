#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright 2022 Huawei Technologies Co., Ltd
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

# 'pylint: disable=invalid-name,too-many-locals,too-many-arguments,unused-argument
from functools import reduce as functools_reduce
import te.platform as tbe_platform
from te import tik
from te.utils import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # proposal struct contains 8 elements
    PROPOSAL_NUM = 8
    # use this idx in proposal struct for val
    VAL_IDX = 4
    # use this idx in proposal struct for idx's merchant part
    INT_IDX = 0
    # use this idx in proposal struct for idx's remainder part
    REM_IDX = 1
    # min val in fp16
    MIN_VAL = -65504
    # max val in fp16
    MAX_VAL = 65504
    # mask for vadds
    MASK = 128
    # sorting threshold for normal data volume
    BLOCK = 16
    # sorting threshold for data volume over 2048
    NUM_BLOCK = 2048
    # sorting limit for data volume
    DATA_LIMITE = 100000


def check_supported(x, y1, y2, axis, descending, kernel_name="sort"):
    """
    check the op support situation.
    Go to AICPU when the date in sort axis is over 100K.
    """
    input_shape = x.get("shape")
    if input_shape[-1] > Constant.DATA_LIMITE:
        reason = "The date in sort axis is over 100K."
        return False, reason

    return True, ""


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class Sort(object):
    def __init__(self, x, y1, y2, axis, descending, kernel_name):
        self.cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        self.is_old_version = True
        shape, self.dtype, self.num_per_task = self.check(x, y1, y2, axis, kernel_name)
        self.kernel_name = kernel_name
        self.descending = descending
        allnum = functools_reduce(lambda x, y: x * y, shape)
        self.task_num = allnum // self.num_per_task
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.align = 4 if self.dtype == "float16" else 2
        self.num_offset = Constant.PROPOSAL_NUM if self.is_old_version else self.align

        self.sort_size = 16 if self.is_old_version else 32
        self.struce_len = 16 if self.is_old_version else 8

        self.num_16 = (self.num_per_task + Constant.BLOCK - 1) // Constant.BLOCK * Constant.BLOCK
        self.num_2048 = (self.num_per_task + Constant.NUM_BLOCK - 1) // Constant.NUM_BLOCK * Constant.NUM_BLOCK
        self.batch_per_task = self.num_2048 // Constant.NUM_BLOCK
        self.num_per_batch = self.num_16 if self.num_per_task <= Constant.NUM_BLOCK else Constant.NUM_BLOCK

        self.input_gm = self.tik_instance.Tensor(self.dtype, shape, name="input_gm", scope=tik.scope_gm)
        self.data_out = self.tik_instance.Tensor(self.dtype, shape, name="data_out", scope=tik.scope_gm)
        self.data_indices = self.tik_instance.Tensor("int32", shape, name="data_indices", scope=tik.scope_gm)

        self.available_core_num = tik.Dprofile().get_aicore_num()
        self.used_core_num = self.available_core_num if self.task_num > self.available_core_num else self.task_num
        self.batch_num_per_core_process = self.task_num // self.used_core_num
        self.batch_tail = self.task_num % self.used_core_num

        self.tmp_workspace = None
        if self.num_per_task <= Constant.NUM_BLOCK:
            self.data_out_align = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_16],
                                                           name="data_out_align",
                                                           scope=tik.scope_gm, is_workspace=True)
            self.data_indices_align = self.tik_instance.Tensor("int32", [self.task_num * self.num_16],
                                                               name="data_indices_align",
                                                               scope=tik.scope_gm, is_workspace=True)

        else:
            self.data_out_align = self.tik_instance.Tensor(self.dtype, [self.task_num * self.num_2048],
                                                           name="data_out_align",
                                                           scope=tik.scope_gm, is_workspace=True)
            self.data_indices_align = self.tik_instance.Tensor("int32", [self.task_num * self.num_2048],
                                                               name="data_indices_align",
                                                               scope=tik.scope_gm, is_workspace=True)
            self.tmp_workspace = self.tik_instance.Tensor(
                self.dtype, [self.used_core_num * self.batch_per_task * self.num_2048 * self.num_offset],
                name="tmp_workspace", scope=tik.scope_gm, is_workspace=True)

    def check(self, x, y1, y2, axis, kernel_name):
        """
        Function: Check parameters (eg: shape dtype etc).
        """
        para_check.check_kernel_name(kernel_name)

        shape = y1.get("shape")
        dtype = y1.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("float16", "float32"))
        para_check.check_shape_rule(shape)

        shape = y2.get("shape")
        dtype = y2.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("int32"))
        para_check.check_shape_rule(shape)

        shape = x.get("shape")
        dtype = x.get("dtype").lower()
        para_check.check_dtype_rule(dtype, ("float16", "float32"))
        para_check.check_shape_rule(shape)

        if axis == -1:
            axis = len(shape) - 1

        if axis != len(shape) - 1:
            raise RuntimeError("Dim should take the last one.")

        num_per_task = shape[axis]

        return shape, dtype, num_per_task

    def sort_compute(self):
        """
        Function: sort compute.
        """
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_core_process) as j:
                if self.num_per_task <= Constant.NUM_BLOCK:
                    self.sort_mini_num(i + j * self.used_core_num)
                else:
                    self.sort_large_num(i + j * self.used_core_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                if self.num_per_task <= Constant.NUM_BLOCK:
                    self.sort_mini_num(self.batch_num_per_core_process * self.used_core_num + i)
                else:
                    self.sort_large_num(self.batch_num_per_core_process * self.used_core_num + i)

        self.tune()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm],
                                   outputs=[self.data_out, self.data_indices])

        return self.tik_instance

    def sort_mini_num(self, core_idx):
        """
        Function: fix num_per_task less than 2048.
        """
        idx_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)
        input_ub = self.tik_instance.Tensor(self.dtype, [self.num_16 * Constant.PROPOSAL_NUM * 2], name="input_ub",
                                            scope=tik.scope_ubuf)

        offset_in = core_idx * self.num_per_task
        offset_out = core_idx * self.num_16
        dest_pos_ub = self.num_16 * Constant.PROPOSAL_NUM
        n_repeat_total = self.num_16 // Constant.BLOCK
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[offset_in], 0, 1, n_repeat_total, 0, 0)
        max_num = self.tik_instance.Scalar('float16', init_value=Constant.MAX_VAL)
        min_num = self.tik_instance.Scalar('float16', init_value=Constant.MIN_VAL)
        # Add ineffective object for 16 alignment
        if self.descending:
            with self.tik_instance.for_range(0, self.num_16 - self.num_per_task) as i:
                input_ub[(self.num_per_task + i) + dest_pos_ub].set_as(min_num)
        else:
            with self.tik_instance.for_range(0, self.num_16 - self.num_per_task) as i:
                input_ub[(self.num_per_task + i) + dest_pos_ub].set_as(max_num)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], n_repeat_total, Constant.VAL_IDX)

        if self.cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="data_out_ub_",
                                                    scope=tik.scope_ubuf)
            data_indices_ub_int_ = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="data_indices_ub_int_",
                                                            scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                data_indices_ub_int_.set_as(self.num_per_task - 1 - i2)
                self.tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[(self.num_per_task - 1 - i2)].set_as(data_out_ub_[0])
        else:
            idx = self.tik_instance.Scalar(dtype="float32", init_value=self.num_per_task)
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                idx.set_as(idx - 1)
                idx_ub[(self.num_per_task - 1 - i2)].set_as(idx)
        self.tik_instance.vconcat(input_ub[0], idx_ub[0], n_repeat_total, Constant.INT_IDX)

        # 2. vbs16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=n_repeat_total)
        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        self.moveout_mini_num(offset_out, input_ub, dest_pos_ub)

    def sort_large_num(self, core_idx):
        """
        Function: fix num_per_task more than 2048.
        """
        idx_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_BLOCK], name="idx_ub", scope=tik.scope_ubuf)
        tmp_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_BLOCK], name="tmp_ub", scope=tik.scope_ubuf)

        if self.cce_product == tbe_platform.ASCEND_310:
            data_out_ub_ = self.tik_instance.Tensor(self.dtype, [Constant.BLOCK], name="data_out_ub_",
                                                    scope=tik.scope_ubuf)
            data_indices_ub_int_ = self.tik_instance.Tensor("int32", [Constant.BLOCK], name="data_indices_ub_int_",
                                                            scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, Constant.NUM_BLOCK // 2) as i2:
                data_indices_ub_int_.set_as(i2)
                self.tik_instance.vec_conv(1, "none", data_out_ub_, data_indices_ub_int_, 1, 0, 0, deqscale=1.0)
                idx_ub[i2].set_as(data_out_ub_[0])
            self.tik_instance.vec_adds(Constant.MASK, idx_ub[Constant.NUM_BLOCK // 2], idx_ub, 1024.0, 8, 8, 8)
        else:
            idx = self.tik_instance.Scalar(dtype="float32", init_value=0)
            with self.tik_instance.for_range(0, Constant.NUM_BLOCK // 2) as i2:
                idx_ub[i2].set_as(idx)
                idx.set_as(idx + 1)
            self.tik_instance.vec_adds(Constant.MASK, idx_ub[Constant.NUM_BLOCK // 2], idx_ub, 1024.0, 8, 8, 8)

        offset = (core_idx % self.used_core_num) * self.batch_per_task * Constant.NUM_BLOCK * Constant.PROPOSAL_NUM
        ori_offset = core_idx * self.num_per_task
        input_ub = self.tik_instance.Tensor(self.dtype, [Constant.NUM_BLOCK * 2 * Constant.PROPOSAL_NUM * 2],
                                            name="input_ub", scope=tik.scope_ubuf)

        # SORT IN UB
        for i in range(self.batch_per_task):
            self.sort_in_ub(input_ub, idx_ub, tmp_ub, i, ori_offset, offset)

        # SORT IN GM
        self.sort_in_gm(input_ub, offset)

        # Pick Data from GM to GM
        self.moveout_large_num(offset, core_idx, input_ub)

    def vms4(self, input_ub, dest_pos_ub):
        """
        Function: Merge all lists into one.
        """
        # record the lists info
        length = self.num_per_batch // self.sort_size
        num_list = [self.sort_size] * length
        # Let them merge evenly, to avoid exceeding the limit of the number of single list.
        src_pos_ub = 0
        while len(num_list) > 1:
            src_pos_ub, dest_pos_ub = dest_pos_ub, src_pos_ub
            index = 0
            offset = 0
            while True:
                res = len(num_list) - index
                if res > 3:
                    num_list, input_ub, offset = self.merge4(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 3:
                    num_list, input_ub, offset = self.merge3(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 2:
                    num_list, input_ub, offset = self.merge2(num_list, input_ub, offset, src_pos_ub, index,
                                                             dest_pos_ub)
                elif res == 1:
                    self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.num_offset],
                                                input_ub[src_pos_ub + offset * self.num_offset], 0, 1,
                                                num_list[index] * self.struce_len // 32, 0, 0)
                else:
                    break
                index += 1

        return input_ub, dest_pos_ub

    def merge4(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 4 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * self.num_offset],
                    input_ub[
                        src_pos_ub + (
                                offset + num_list[index] + num_list[index + 1] + num_list[
                            index + 2]) * self.num_offset]]

        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
        # merge 4 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                       if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the four element values and record them in a(num_list)
        num_list[index] = sum(num_list[index:index + 4])
        a = num_list[:index + 1:]
        b = num_list[index + 4::]
        a.extend(b)
        offset += a[index]

        return a, input_ub, offset

    def merge3(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 3 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * self.num_offset],
                    input_ub[0]]
        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
        # merge 3 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:3],
                                       src_list_lengths[0:3], if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the three element values and record them in a num_list
        num_list[index] = sum(num_list[index:index + 3])
        a = num_list[:index + 1:]
        b = num_list[index + 3::]
        a.extend(b)
        offset += a[index]

        return a, input_ub, offset

    def merge2(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 2 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * self.num_offset],
                    input_ub[src_pos_ub + (offset + num_list[index]) * self.num_offset], input_ub[0], input_ub[0]]

        src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
        # merge 2 lists
        if self.is_old_version:
            self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * self.num_offset], src_list, src_list_lengths,
                                        if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)
        else:
            self.tik_instance.vmrgsort(input_ub[dest_pos_ub + offset * self.num_offset], src_list[0:2],
                                       src_list_lengths[0:2], if_exhausted_suspension=False, repeat_times=1)
        # update the lists info : Merge the two element values and record them in num_list
        num_list[index] += num_list[index + 1]
        del num_list[index + 1]
        offset += num_list[index]

        return num_list, input_ub, offset

    def sort_in_ub(self, input_ub, idx_ub, tmp_ub, i, ori_offset, offset):
        """
        Function: sort in ub.
        """
        # dest position in UB
        dest_pos_ub = Constant.NUM_BLOCK * Constant.PROPOSAL_NUM
        repeat_times = Constant.NUM_BLOCK // Constant.BLOCK
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[ori_offset + i * Constant.NUM_BLOCK], 0, 1,
                                    repeat_times, 0, 0)

        self.tik_instance.vector_dup(Constant.BLOCK, tmp_ub[0], i, repeat_times, 1, 1)
        # index // 2048
        self.tik_instance.vconcat(input_ub[0], tmp_ub[0], repeat_times, Constant.INT_IDX)
        # index % 2048
        self.tik_instance.vconcat(input_ub[0], idx_ub[0], repeat_times, Constant.REM_IDX)

        if self.num_per_task < (i + 1) * Constant.NUM_BLOCK:
            # aline for NUM_BLOCK
            aline = Constant.NUM_BLOCK - self.num_per_task % Constant.NUM_BLOCK
            if self.descending:
                tmp = self.tik_instance.Scalar('float16', init_value=Constant.MIN_VAL)
            # descend
            else:
                tmp = self.tik_instance.Scalar('float16', init_value=Constant.MAX_VAL)
            # Add ineffective object for 16 alignment
            for j in range(aline % Constant.BLOCK):
                input_ub[dest_pos_ub + self.num_per_task % Constant.NUM_BLOCK + j].set_as(tmp)
            # Add ineffective object for NUM_BLOCK alignment
            if aline > Constant.BLOCK - 1:
                self.tik_instance.vec_dup(Constant.BLOCK, input_ub[
                    dest_pos_ub + self.num_per_task % Constant.NUM_BLOCK + aline % Constant.BLOCK],
                                          tmp, aline // Constant.BLOCK, 1)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, Constant.VAL_IDX)
        # 2. vrpsort16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)
        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)
        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.tmp_workspace[offset + i * Constant.NUM_BLOCK * Constant.PROPOSAL_NUM],
                                    input_ub[dest_pos_ub], 0, 1,
                                    Constant.NUM_BLOCK * Constant.PROPOSAL_NUM // Constant.BLOCK, 0, 0)

    def sort_in_gm(self, input_ub, offset):
        """
        Function: sort in gm.
        ----------
        """
        src_pos_ub = self.tik_instance.Scalar("int32")
        dest_pos_ub = self.tik_instance.Scalar("int32")

        with self.tik_instance.for_range(0, self.batch_per_task - 1) as tail:
            src_pos_ub.set_as(0)
            dest_pos_ub.set_as(Constant.NUM_BLOCK * 2 * Constant.PROPOSAL_NUM)

            self.tik_instance.data_move(input_ub[src_pos_ub + Constant.NUM_BLOCK * Constant.PROPOSAL_NUM],
                                        self.tmp_workspace[offset], 0,
                                        1, (Constant.NUM_BLOCK * Constant.PROPOSAL_NUM) // Constant.BLOCK, 0, 0)
            with self.tik_instance.for_range(1, self.batch_per_task - tail) as i:
                self.tik_instance.data_move(input_ub[src_pos_ub],
                                            self.tmp_workspace[
                                                offset + Constant.NUM_BLOCK * i * Constant.PROPOSAL_NUM],
                                            0, 1, (Constant.NUM_BLOCK * Constant.PROPOSAL_NUM) // Constant.BLOCK, 0, 0)

                self.tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                            [input_ub[src_pos_ub],
                                             input_ub[src_pos_ub + Constant.NUM_BLOCK * Constant.PROPOSAL_NUM],
                                             input_ub[0], input_ub[0]],
                                            [Constant.NUM_BLOCK, Constant.NUM_BLOCK, 0, 0],
                                            if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

                self.tik_instance.data_move(
                    self.tmp_workspace[offset + Constant.NUM_BLOCK * (i - 1) * Constant.PROPOSAL_NUM],
                    input_ub[dest_pos_ub], 0, 1, (Constant.NUM_BLOCK * Constant.PROPOSAL_NUM) // Constant.BLOCK, 0, 0)

                dest_pos_ub.set_as(src_pos_ub)
                src_pos_ub.set_as(Constant.NUM_BLOCK * 2 * Constant.PROPOSAL_NUM - dest_pos_ub)

            # Move Data from UB to GM
            self.tik_instance.data_move(
                self.tmp_workspace[
                    offset + Constant.NUM_BLOCK * (self.batch_per_task - tail - 1) * Constant.PROPOSAL_NUM],
                input_ub[src_pos_ub + Constant.NUM_BLOCK * Constant.PROPOSAL_NUM], 0, 1,
                (Constant.NUM_BLOCK * Constant.PROPOSAL_NUM) // Constant.BLOCK, 0, 0)

    def moveout_mini_num(self, offset_out, input_ub, dest_pos_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        int_list = self.tik_instance.Tensor("int32", [self.num_16], name="int_list", scope=tik.scope_ubuf)
        src_pos_ub = self.num_16 * Constant.PROPOSAL_NUM if dest_pos_ub == 0 else 0
        # ascend
        with self.tik_instance.if_scope(self.descending is False):
            # data is continuous in GM & gather scattered data together
            with self.tik_instance.for_range(0, self.num_per_task) as i2:
                input_ub[i2 + src_pos_ub].set_as(
                    input_ub[(self.num_16 - 1 - i2) * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])
                input_ub[i2 + src_pos_ub + self.num_16].set_as(
                    input_ub[(self.num_16 - 1 - i2) * Constant.PROPOSAL_NUM + dest_pos_ub])

        # descend
        with self.tik_instance.else_scope():
            # data is continuous in GM & gather scattered data together
            if self.cce_product == tbe_platform.ASCEND_310:
                with self.tik_instance.for_range(0, self.num_per_task) as i2:
                    input_ub[i2 + src_pos_ub].set_as(
                        input_ub[i2 * Constant.PROPOSAL_NUM + Constant.VAL_IDX + dest_pos_ub])
                    input_ub[i2 + src_pos_ub + self.num_16].set_as(
                        input_ub[i2 * Constant.PROPOSAL_NUM + dest_pos_ub])
            else:
                self.tik_instance.vextract(input_ub[src_pos_ub], input_ub[dest_pos_ub],
                                           self.num_16 // Constant.BLOCK, Constant.VAL_IDX)
                self.tik_instance.vextract(input_ub[src_pos_ub + self.num_16], input_ub[dest_pos_ub],
                                           self.num_16 // Constant.BLOCK, Constant.INT_IDX)

        # conv indices (float16->int32) , and move from UB to GM
        self.tik_instance.vec_conv(Constant.BLOCK, "round", int_list, input_ub[src_pos_ub + self.num_16],
                                   self.num_16 // Constant.BLOCK, 2, 1)

        # move output (float16) from UB to GM
        self.tik_instance.data_move(self.data_out_align[offset_out], input_ub[src_pos_ub], 0, 1,
                                    self.num_16 // Constant.BLOCK, 0, 0)
        self.tik_instance.data_move(self.data_indices_align[offset_out], int_list, 0, 1,
                                    2 * self.num_16 // Constant.BLOCK, 0, 0)

    def moveout_large_num(self, offset, core_idx, input_ub):
        """
        Function: pick value from proposal. Move UB to GM, and trans y2 from fp16 to int32.
        """
        # dest position in UB
        dest_pos_ub = Constant.NUM_BLOCK * Constant.PROPOSAL_NUM
        repeat_times = Constant.NUM_BLOCK // Constant.BLOCK

        with self.tik_instance.for_range(0, self.batch_per_task) as i:
            self.tik_instance.data_move(input_ub[0],
                                        self.tmp_workspace[offset + Constant.NUM_BLOCK * i * Constant.PROPOSAL_NUM],
                                        0, 1, (Constant.NUM_BLOCK * Constant.PROPOSAL_NUM) // Constant.BLOCK, 0, 0)
            int_list_1 = self.tik_instance.Tensor("int32", [Constant.NUM_BLOCK], name="int_list_1",
                                                  scope=tik.scope_ubuf)
            int_list_2 = self.tik_instance.Tensor("int32", [Constant.NUM_BLOCK], name="int_list_2",
                                                  scope=tik.scope_ubuf)
            int_list_3 = self.tik_instance.Tensor("int32", [Constant.NUM_BLOCK], name="int_list_3",
                                                  scope=tik.scope_ubuf)
            int_list_4 = self.tik_instance.Tensor("int32", [Constant.NUM_BLOCK], name="int_list_4",
                                                  scope=tik.scope_ubuf)

            self.tik_instance.vector_dup(Constant.BLOCK, int_list_4, Constant.NUM_BLOCK, repeat_times, 1, 2)

            self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.INT_IDX)
            self.tik_instance.vextract(input_ub[dest_pos_ub + Constant.NUM_BLOCK], input_ub[0], repeat_times,
                                       Constant.REM_IDX)
            self.tik_instance.vec_conv(Constant.BLOCK, "round", int_list_1, input_ub[dest_pos_ub], repeat_times, 2, 1)
            self.tik_instance.vec_conv(Constant.BLOCK, "round", int_list_2, input_ub[dest_pos_ub + Constant.NUM_BLOCK],
                                       repeat_times, 2, 1)

            self.tik_instance.vec_mul(Constant.BLOCK, int_list_3, int_list_1, int_list_4, repeat_times, 2, 2, 2)
            self.tik_instance.vec_add(Constant.BLOCK, int_list_1, int_list_2, int_list_3, repeat_times, 2, 2, 2)

            # data is continuous in GM & gather scattered data together

            if self.cce_product == tbe_platform.ASCEND_310:
                with self.tik_instance.for_range(0, Constant.NUM_BLOCK) as i2:
                    input_ub[dest_pos_ub + i2].set_as(input_ub[i2 * Constant.PROPOSAL_NUM + Constant.VAL_IDX])
            else:
                self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.VAL_IDX)
            # move output (float16) from UB to GM
            # ascend
            with self.tik_instance.if_scope(self.descending is False):
                with self.tik_instance.for_range(0, Constant.NUM_BLOCK) as i2:
                    int_list_2[i2].set_as(int_list_1[Constant.NUM_BLOCK - i2 - 1])
                    input_ub[dest_pos_ub + Constant.NUM_BLOCK + i2].set_as(
                        input_ub[dest_pos_ub + Constant.NUM_BLOCK - i2 - 1])
                self.tik_instance.data_move(
                    self.data_indices_align[
                        core_idx * self.num_2048 + Constant.NUM_BLOCK * (self.batch_per_task - i - 1)],
                    int_list_2, 0, 1, 2 * repeat_times, 0, 0)
                self.tik_instance.data_move(
                    self.data_out_align[core_idx * self.num_2048 + Constant.NUM_BLOCK * (self.batch_per_task - i - 1)],
                    input_ub[dest_pos_ub + Constant.NUM_BLOCK], 0, 1, repeat_times, 0, 0)
            # descend
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.data_indices_align[core_idx * self.num_2048 + Constant.NUM_BLOCK * i],
                                            int_list_1, 0, 1, 2 * repeat_times, 0, 0)
                self.tik_instance.data_move(self.data_out_align[core_idx * self.num_2048 + Constant.NUM_BLOCK * i],
                                            input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

    def tune(self):
        """
        Function: remove min.
        """

        if self.num_per_task <= Constant.NUM_BLOCK:
            repeat_times = self.num_16 // Constant.BLOCK
            float_ub = self.tik_instance.Tensor("float16", [self.num_16], name="float_ub", scope=tik.scope_ubuf)
            int_ub = self.tik_instance.Tensor("int32", [self.num_16], name="int_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.task_num) as i:
                self.tik_instance.data_move(float_ub[0], self.data_out_align[i * self.num_16], 0, 1, repeat_times, 0, 0)
                self.tik_instance.data_move(self.data_out[i * self.num_per_task], float_ub[0], 0, 1, repeat_times,
                                            0, 0)

                self.tik_instance.data_move(int_ub[0], self.data_indices_align[i * self.num_16], 0, 1, 2 * repeat_times,
                                            0, 0)
                self.tik_instance.data_move(self.data_indices[i * self.num_per_task], int_ub[0], 0, 1, 2 * repeat_times,
                                            0, 0)
        else:
            repeat_times = Constant.NUM_BLOCK // Constant.BLOCK
            float_ub = self.tik_instance.Tensor("float16", [Constant.NUM_BLOCK], name="float_ub",
                                                scope=tik.scope_ubuf)
            int_ub = self.tik_instance.Tensor("int32", [Constant.NUM_BLOCK], name="int_ub", scope=tik.scope_ubuf)
            with self.tik_instance.for_range(0, self.task_num) as i:
                with self.tik_instance.for_range(0, self.batch_per_task) as j:
                    self.tik_instance.data_move(float_ub[0],
                                                self.data_out_align[i * self.num_2048 + j * Constant.NUM_BLOCK],
                                                0, 1, repeat_times, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * Constant.NUM_BLOCK],
                                                float_ub[0], 0, 1, repeat_times, 0, 0)

                    self.tik_instance.data_move(int_ub[0],
                                                self.data_indices_align[i * self.num_2048 + j * Constant.NUM_BLOCK],
                                                0, 1, 2 * repeat_times, 0, 0)
                    self.tik_instance.data_move(self.data_indices[i * self.num_per_task + j * Constant.NUM_BLOCK],
                                                int_ub[0], 0, 1, 2 * repeat_times, 0, 0)


# 'pylint: disable=too-few-public-methods
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_BOOL, para_check.KERNEL_NAME)
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
    op_obj = Sort(x, y1, y2, axis, descending, kernel_name)
    return op_obj.sort_compute()
