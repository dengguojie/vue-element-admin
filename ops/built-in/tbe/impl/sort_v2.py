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
sort_v2
"""
import functools
from te import tik
from te.utils import para_check
import te.platform as tbe_platform
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    # aicore num in 910
    CLOULD_CORE = 32
    RESERVE_SIZE = 2 * 1024
    PROPOSAL_NUM = 8
    VAL_INDEX = 4
    REPEAT_MAX = 255


def op_select_format(x, y, axis=-1, descending=False, kernel_name="sort_v2"):
    cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)

    op_dtype = "float16" if cce_product != tbe_platform.ASCEND_920A else "float16,float32"
    op_format = "ND" if cce_product != tbe_platform.ASCEND_920A else "ND,ND"

    input0 = gen_param(classify="input0", name="x", datatype=op_dtype, format=op_format)
    output0 = gen_param(classify="output0", name="y", datatype=op_dtype, format=op_format)

    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)
    return param_dynamic_in_json


# pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
class SortV2(object):
    def __init__(self, x, y, axis, descending, kernel_name):
        cce_product = tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION)
        self.is_clould = True if cce_product != tbe_platform.ASCEND_920A else False
        shape, self.dtype, self.num_per_task = self.check(x, y, axis, kernel_name)
        allnum = functools.reduce(lambda x, y: x * y, shape)
        self.kernel_name = kernel_name
        self.descending = descending
        self.align = 4 if self.dtype == "float16" else 2
        self.num_offset = Constant.PROPOSAL_NUM if self.is_clould else self.align
        self.min = -65504 if self.dtype == "float16" else -3.4e38
        self.block_size = 16 if self.dtype == "float16" else 8
        self.tik_instance = tik.Tik(tik.Dprofile())
        self.struce_len = 16 if self.is_clould else 8
        self.sort_size = 16 if self.is_clould else 32
        self.num_in_ub = (tik.Dprofile().get_unified_buffer_size() - Constant.RESERVE_SIZE) // self.struce_len // 2

        self.num_per_batch = self.num_in_ub // 2 // self.sort_size * self.sort_size
        self.batch_per_task = (self.num_per_task + self.num_per_batch - 1) // self.num_per_batch
        self.totalnum_per_task = self.batch_per_task * self.num_per_batch

        self.task_num = allnum // self.num_per_task

        self.available_core_num = tik.Dprofile().get_aicore_num()
        self.used_core_num = self.available_core_num if self.task_num > self.available_core_num else self.task_num
        self.batch_num_per_core_process = self.task_num // self.used_core_num
        self.batch_tail = self.task_num % self.used_core_num

        self.input_gm = self.tik_instance.Tensor(self.dtype, shape, name="input_gm", scope=tik.scope_gm)
        self.data_out_align = self.tik_instance.Tensor(self.dtype, [self.task_num, self.totalnum_per_task],
                                                       name="data_out_align",
                                                       scope=tik.scope_gm, is_workspace=True)
        self.data_out = self.tik_instance.Tensor(self.dtype, shape, name="data_out", scope=tik.scope_gm)

        self.tmp_workspace = self.tik_instance.Tensor(
            self.dtype, [self.used_core_num * self.batch_per_task * self.num_in_ub * self.num_offset],
            name="tmp_workspace", scope=tik.scope_gm, is_workspace=True)

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
    def check(self, x, y, axis, kernel_name):
        """
        Function: Check parameters
        """
        para_check.check_kernel_name(kernel_name)

        shape = y.get("shape")
        dtype = y.get("dtype").lower()
        para_check.check_dtype(dtype, ("float16", "float32"), param_name="y")
        para_check.check_shape(shape, param_name="y")

        shape = x.get("shape")
        dtype = x.get("dtype").lower()
        para_check.check_dtype(dtype, ("float16", "float32"), param_name="x")
        para_check.check_shape(shape, param_name="x")

        if axis == -1:
            axis = len(shape) - 1

        if axis != len(shape) - 1:
            raise RuntimeError("Dim should take the last one.")

        num_per_task = shape[axis]

        return shape, dtype, num_per_task

    def sort_v2_compute(self):
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_core_process) as j:
                self.sort_compute(i + j * self.used_core_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.sort_compute(self.batch_num_per_core_process * self.used_core_num + i)

        # fine tune data in GM
        self.tune()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm], outputs=[self.data_out])

        return self.tik_instance

    def sort_compute(self, task_idx):
        """
        Function: compute.
        """
        # offset in self.tmp_workspace
        ws_offset = (task_idx % self.used_core_num) * self.totalnum_per_task * self.num_offset
        # offset in self.input_gm
        in_offset = task_idx * self.num_per_task

        if self.is_clould:
            input_ub = self.tik_instance.Tensor(self.dtype, [self.num_in_ub * self.num_offset * 2],
                                                name="input_ub", scope=tik.scope_ubuf)
            # SORT IN UB
            with self.tik_instance.for_range(0, self.batch_per_task) as batch_idx:
                self.sort_in_ub_910(input_ub, batch_idx, in_offset, ws_offset)

            # SORT IN GM
            self.sort_in_gm(input_ub, ws_offset)

            # Pick Data from GM to GM
            self.pick(ws_offset, task_idx, input_ub)
        else:
            # SORT IN UB
            with self.tik_instance.for_range(0, self.batch_per_task) as batch_idx:
                self.sort_in_ub_920(batch_idx, in_offset, ws_offset)

            input_ub = self.tik_instance.Tensor(self.dtype, [self.num_in_ub * self.num_offset * 2],
                                                name="input_ub", scope=tik.scope_ubuf)
            # SORT IN GM
            self.sort_in_gm(input_ub, ws_offset)

            # Pick Data from GM to GM
            self.pick(ws_offset, task_idx, input_ub)

    def sort_in_ub_910(self, input_ub, batch_idx, in_offset, ws_offset):
        # dest position in UB
        dest_pos_ub = self.num_per_batch * self.num_offset
        repeat_times = self.num_per_batch // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub],
                                    self.input_gm[in_offset + batch_idx * self.num_per_batch], 0, 1, repeat_times,
                                    0, 0)

        with self.tik_instance.if_scope(self.num_per_task < (batch_idx + 1) * self.num_per_batch):
            # aline for k
            aline = self.num_per_batch - self.num_per_task % self.num_per_batch
            Min = self.tik_instance.Scalar(self.dtype, init_value=self.min)
            repeat_num_max = self.block_size * Constant.REPEAT_MAX
            # Add ineffective object for 16 alignment
            with self.tik_instance.for_range(0, aline % self.block_size) as j:
                input_ub[dest_pos_ub + self.num_per_task % self.num_per_batch + j].set_as(Min)
            # Add ineffective object for k alignment
            with self.tik_instance.if_scope(aline >= self.block_size):
                with self.tik_instance.if_scope(aline > repeat_num_max):
                    with self.tik_instance.for_range(0, aline // repeat_num_max) as repeat_idx:
                        self.tik_instance.vec_dup(self.block_size,
                                                  input_ub[dest_pos_ub + self.num_per_task % self.num_per_batch +
                                                           aline % self.block_size +
                                                           repeat_idx * repeat_num_max],
                                                  Min, Constant.REPEAT_MAX, 1)
                    self.tik_instance.vec_dup(self.block_size,
                                              input_ub[dest_pos_ub + self.num_per_task % self.num_per_batch +
                                                       aline % self.block_size +
                                                       aline // repeat_num_max * repeat_num_max],
                                              Min, aline // self.block_size -
                                              aline // repeat_num_max * Constant.REPEAT_MAX, 1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_dup(self.block_size,
                                              input_ub[dest_pos_ub + self.num_per_task % self.num_per_batch +
                                                       aline % self.block_size],
                                              Min, aline // self.block_size, 1)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, Constant.VAL_INDEX)
        # 2. vbs16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)

        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)

        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.tmp_workspace[ws_offset + batch_idx * self.num_per_batch * self.num_offset],
                                    input_ub[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

    def sort_in_ub_920(self, batch_idx, in_offset, ws_offset):
        input_ub_tmp = self.tik_instance.Tensor(self.dtype, [self.num_per_batch * self.num_offset * 2],
                                                name="input_ub_tmp", scope=tik.scope_ubuf)
        val_ub = self.tik_instance.Tensor(self.dtype, [self.num_per_batch], name="val_ub",
                                          scope=tik.scope_ubuf)
        src1_ub = self.tik_instance.Tensor("uint32", [self.num_per_batch], name="src1_ub", scope=tik.scope_ubuf)

        # dest position in UB
        repeat_times = self.num_per_batch // self.block_size
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(val_ub,
                                    self.input_gm[in_offset + batch_idx * self.num_per_batch], 0, 1, repeat_times,
                                    0, 0)
        dest_pos_ub = self.num_per_batch * self.num_offset
        with self.tik_instance.if_scope(self.num_per_task < (batch_idx + 1) * self.num_per_batch):
            # aline for k
            aline = self.num_per_batch - self.num_per_task % self.num_per_batch
            Min = self.tik_instance.Scalar(self.dtype, init_value=self.min)
            repeat_num_max = self.block_size * Constant.REPEAT_MAX
            # Add ineffective object for 16 alignment
            with self.tik_instance.for_range(0, aline % self.block_size) as j:
                val_ub[self.num_per_task % self.num_per_batch + j].set_as(Min)
            # Add ineffective object for k alignment
            with self.tik_instance.if_scope(aline >= self.block_size):
                with self.tik_instance.if_scope(aline > repeat_num_max):
                    with self.tik_instance.for_range(0, aline // repeat_num_max) as repeat_idx:
                        self.tik_instance.vec_dup(self.block_size,
                                                  val_ub[self.num_per_task % self.num_per_batch +
                                                         aline % self.block_size +
                                                         repeat_idx * repeat_num_max],
                                                  Min, Constant.REPEAT_MAX, 1)
                    self.tik_instance.vec_dup(self.block_size,
                                              val_ub[self.num_per_task % self.num_per_batch +
                                                     aline % self.block_size +
                                                     aline // repeat_num_max * repeat_num_max],
                                              Min, aline // self.block_size -
                                              aline // repeat_num_max * Constant.REPEAT_MAX, 1)
                with self.tik_instance.else_scope():
                    self.tik_instance.vec_dup(self.block_size,
                                              val_ub[self.num_per_task % self.num_per_batch +
                                                     aline % self.block_size],
                                              Min, aline // self.block_size, 1)

        # 2. vbs32
        self.tik_instance.vsort32(input_ub_tmp[dest_pos_ub], val_ub, src1_ub, self.num_per_batch // self.sort_size)

        # 3. vms4
        input_ub_tmp, dest_pos_ub = self.vms4(input_ub_tmp, dest_pos_ub)

        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.tmp_workspace[ws_offset + batch_idx * self.num_per_batch * self.num_offset],
                                    input_ub_tmp[dest_pos_ub], 0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)

    def vms4(self, input_ub, dest_pos_ub):
        """
        Function: Merge all lists into one.
        """
        # record the lists info
        length = self.num_per_batch // self.sort_size
        num_list = [self.sort_size] * length

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
                    self.tik_instance.data_move(input_ub[dest_pos_ub + offset * self.align],
                                                input_ub[src_pos_ub + offset * self.align], 0, 1,
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
        if self.is_clould:
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
        if self.is_clould:
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
        if self.is_clould:
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

    def sort_in_gm(self, input_ub, ws_offset):
        src_pos_ub = self.tik_instance.Scalar("int32")
        dest_pos_ub = self.tik_instance.Scalar("int32")

        with self.tik_instance.for_range(0, self.batch_per_task - 1) as tail:
            src_pos_ub.set_as(0)
            dest_pos_ub.set_as(self.num_in_ub * self.num_offset)

            self.tik_instance.data_move(input_ub[src_pos_ub + self.num_per_batch * self.num_offset],
                                        self.tmp_workspace[ws_offset], 0, 1,
                                        self.num_per_batch * self.struce_len // 32, 0, 0)
            with self.tik_instance.for_range(1, self.batch_per_task - tail) as i:
                self.tik_instance.data_move(input_ub[src_pos_ub],
                                            self.tmp_workspace[ws_offset + self.num_per_batch * i * self.num_offset],
                                            0, 1, self.num_per_batch * self.struce_len // 32, 0, 0)
                if self.is_clould:
                    self.tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                                [input_ub[src_pos_ub],
                                                 input_ub[src_pos_ub + self.num_per_batch * self.num_offset],
                                                 input_ub[0], input_ub[0]],
                                                [self.num_per_batch, self.num_per_batch, 0, 0],
                                                if_exhausted_suspension=False,
                                                valid_bit="0011", repeat_times=1)
                else:
                    self.tik_instance.vmrgsort(input_ub[dest_pos_ub],
                                               [input_ub[src_pos_ub],
                                                input_ub[src_pos_ub + self.num_per_batch * self.num_offset]],
                                               [self.num_per_batch, self.num_per_batch],
                                               if_exhausted_suspension=False, repeat_times=1)
                self.tik_instance.data_move(
                    self.tmp_workspace[ws_offset + self.num_per_batch * (i - 1) * self.num_offset],
                    input_ub[dest_pos_ub], 0,
                    1, self.num_per_batch * self.struce_len // 32, 0, 0)

                dest_pos_ub.set_as(src_pos_ub)
                src_pos_ub.set_as(self.num_in_ub * self.num_offset - dest_pos_ub)

            # Move Data from UB to GM
            self.tik_instance.data_move(
                self.tmp_workspace[ws_offset + self.num_per_batch * (self.batch_per_task - tail - 1) * self.num_offset],
                input_ub[src_pos_ub + self.num_per_batch * self.num_offset], 0, 1,
                self.num_per_batch * self.struce_len // 32, 0, 0)

    def pick(self, ws_offset, task_idx, input_ub):
        # dest position in UB
        dest_pos_ub = self.num_in_ub * self.num_offset
        repeat_times = self.num_per_batch // self.block_size
        with self.tik_instance.for_range(0, self.batch_per_task) as i:
            self.tik_instance.data_move(input_ub[0],
                                        self.tmp_workspace[ws_offset + self.num_per_batch * i * self.num_offset], 0, 1,
                                        self.num_per_batch * self.struce_len // 32, 0, 0)

            if self.is_clould:
                # ascend
                with self.tik_instance.if_scope(self.descending is False):
                    # data is continuous in GM & gather scattered data together
                    with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                        input_ub[i2 + dest_pos_ub].set_as(
                            input_ub[(self.num_per_batch - 1 - i2) * Constant.PROPOSAL_NUM + Constant.VAL_INDEX])

                    # move output (float16) from UB to GM
                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.totalnum_per_task +
                                            self.num_per_batch * (self.batch_per_task - 1 - i)],
                        input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)
                # descend
                with self.tik_instance.else_scope():
                    # data is continuous in GM & gather scattered data together
                    self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.VAL_INDEX)

                    # move output (float16) from UB to GM
                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.totalnum_per_task + self.num_per_batch * i],
                        input_ub[dest_pos_ub],
                        0, 1, repeat_times, 0, 0)
            else:
                # ascend
                with self.tik_instance.if_scope(self.descending is False):
                    # data is continuous in GM & gather scattered data together
                    with self.tik_instance.for_range(0, self.num_per_batch) as i2:
                        input_ub[i2 + dest_pos_ub].set_as(input_ub[(self.num_per_batch - 1 - i2) * self.align])

                    # move output (float16) from UB to GM
                    self.tik_instance.data_move(self.data_out_align[
                                                    task_idx * self.totalnum_per_task + self.num_per_batch * (
                                                            self.batch_per_task - 1 - i)],
                                                input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

                # descend
                with self.tik_instance.else_scope():
                    # move output from UB to GM
                    with self.tik_instance.if_scope(self.align == 4):
                        self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 3,
                                                    repeat_times, 1, 8, 0)
                    with self.tik_instance.else_scope():
                        self.tik_instance.vreducev2(None, input_ub[dest_pos_ub], input_ub[0], 1,
                                                    repeat_times // 2, 1, 8, 0)
                    self.tik_instance.data_move(
                        self.data_out_align[task_idx * self.totalnum_per_task + self.num_per_batch * i],
                        input_ub[dest_pos_ub],
                        0, 1, repeat_times, 0, 0)

    def tune(self):
        """
        Function: remove min.
        """
        offset = self.num_per_batch - self.num_per_task % self.num_per_batch
        num_max = self.num_in_ub * 2 * self.align
        use_num = self.totalnum_per_task if self.totalnum_per_task > num_max else num_max

        batchs = (self.totalnum_per_task + use_num - 1) // use_num
        float_ub = self.tik_instance.Tensor(self.dtype, [use_num], name="float_ub", scope=tik.scope_ubuf)

        with self.tik_instance.for_range(0, self.task_num) as i:
            with self.tik_instance.for_range(0, batchs) as j:
                with self.tik_instance.if_scope(self.descending is False):
                    self.tik_instance.data_move(float_ub[0],
                                                self.data_out_align[i * self.totalnum_per_task + offset + j * use_num],
                                                0, 1, use_num // self.block_size, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * use_num], float_ub[0], 0, 1,
                                                use_num // self.block_size, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(float_ub[0],
                                                self.data_out_align[i * self.totalnum_per_task + j * use_num], 0, 1,
                                                use_num // self.block_size, 0, 0)
                    self.tik_instance.data_move(self.data_out[i * self.num_per_task + j * use_num], float_ub[0], 0, 1,
                                                use_num // self.block_size, 0, 0)


# 'pylint: disable=too-few-public-methods
@tbe_platform.fusion_manager.fusion_manager.register("sort_v2")
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
    op_obj = SortV2(x, y, axis, descending, kernel_name)
    return op_obj.sort_v2_compute()
