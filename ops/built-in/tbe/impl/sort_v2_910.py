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
sort_v2_910
"""
import functools
from te import tik
from te.utils import para_check


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    RESERVE_SIZE = 2 * 1024
    PROPOSAL_NUM = 8
    BOLCK_SIZE = 16
    VAL_INDEX = 4
    MIN_VAL = -65504


class SortV2In910(object):
    def __init__(self, x, y, axis, descending, kernel_name):
        shape, self.dtype, self.num = self.check(x, y, axis, kernel_name)
        allnum = functools.reduce(lambda x, y: x * y, shape)
        self.kernel_name = kernel_name
        self.descending = descending
        self.tik_instance = tik.Tik(tik.Dprofile('cloud'))
        self.batchsize = (tik.Dprofile().get_unified_buffer_size() - Constant.RESERVE_SIZE) // 32 // 32 * 32
        self.k = self.batchsize // 2
        self.num_gm = (self.num + self.k - 1) // self.k
        self.totalnum = self.num_gm * self.k

        big_shape = list(shape)
        big_shape[-1] = self.totalnum
        self.rounds = allnum // self.num

        self.available_core_num = tik.Dprofile().get_aicore_num()
        self.used_core_num = self.available_core_num if self.rounds > self.available_core_num else self.rounds
        self.batch_num_per_core_process = self.rounds // self.used_core_num
        self.batch_tail = self.rounds % self.used_core_num

        self.input_gm = self.tik_instance.Tensor(self.dtype, shape, name="input_gm", scope=tik.scope_gm)
        self.data_out = self.tik_instance.Tensor(self.dtype, big_shape, name="data_out", scope=tik.scope_gm,
                                                 is_workspace=True)
        self.data_out_ = self.tik_instance.Tensor(self.dtype, shape, name="data_out_", scope=tik.scope_gm)
        self.temp = self.tik_instance.Tensor(self.dtype,
                                             [self.used_core_num * self.num_gm * self.k * Constant.PROPOSAL_NUM],
                                             name="temp", scope=tik.scope_gm, is_workspace=True)

    def sort_v2_compute(self):
        with self.tik_instance.for_range(0, self.used_core_num, block_num=self.used_core_num) as i:
            with self.tik_instance.for_range(0, self.batch_num_per_core_process) as j:
                self.sort_compute(i + j * self.used_core_num)
            with self.tik_instance.if_scope(i < self.batch_tail):
                self.sort_compute(self.batch_num_per_core_process * self.used_core_num + i)

        # fine tune data in GM
        self.tune()

        self.tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.input_gm], outputs=[self.data_out_])

        return self.tik_instance

    # pylint: disable=invalid-name,too-many-arguments,too-many-locals,unused-argument
    def check(self, x, y, axis, kernel_name):
        """
        Function: Check parameters
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

    def merge4(self, num_list, input_ub, offset, src_pos_ub, index, dest_pos_ub):
        """
        Function: Merge 4 lists in UB.
        """
        src_list = [input_ub[src_pos_ub + offset * Constant.PROPOSAL_NUM],
                    input_ub[src_pos_ub + (offset + num_list[index]) * Constant.PROPOSAL_NUM],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * Constant.PROPOSAL_NUM],
                    input_ub[
                        src_pos_ub + (offset + num_list[index] + num_list[index + 1] + num_list[
                            index + 2]) * Constant.PROPOSAL_NUM]]

        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], num_list[index + 3]]
        # merge 4 lists
        self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * Constant.PROPOSAL_NUM], src_list, src_list_lengths,
                                    if_exhausted_suspension=False, valid_bit="1111", repeat_times=1)
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
        src_list = [input_ub[src_pos_ub + offset * Constant.PROPOSAL_NUM],
                    input_ub[src_pos_ub + (offset + num_list[index]) * Constant.PROPOSAL_NUM],
                    input_ub[src_pos_ub + (offset + num_list[index] + num_list[index + 1]) * Constant.PROPOSAL_NUM],
                    input_ub[0]]
        src_list_lengths = [num_list[index], num_list[index + 1], num_list[index + 2], 0]
        # merge 3 lists
        self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * Constant.PROPOSAL_NUM], src_list, src_list_lengths,
                                    if_exhausted_suspension=False, valid_bit="0111", repeat_times=1)
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
        src_list = [input_ub[src_pos_ub + offset * Constant.PROPOSAL_NUM],
                    input_ub[src_pos_ub + (offset + num_list[index]) * Constant.PROPOSAL_NUM], input_ub[0], input_ub[0]]

        src_list_lengths = [num_list[index], num_list[index + 1], 0, 0]
        # merge 2 lists
        self.tik_instance.vmrgsort4(input_ub[dest_pos_ub + offset * Constant.PROPOSAL_NUM], src_list, src_list_lengths,
                                    if_exhausted_suspension=False, valid_bit="0011", repeat_times=1)

        # update the lists info : Merge the two element values and record them in num_list
        num_list[index] += num_list[index + 1]
        del num_list[index + 1]
        offset += num_list[index]

        return num_list, input_ub, offset

    def vms4(self, input_ub, dest_pos_ub):
        """
        Function: Merge all lists into one.
        """
        # record the lists info
        length = self.k // Constant.BOLCK_SIZE
        num_list = [Constant.BOLCK_SIZE] * length

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
                    self.tik_instance.data_move(input_ub[dest_pos_ub + offset * Constant.PROPOSAL_NUM],
                                                input_ub[src_pos_ub + offset * Constant.PROPOSAL_NUM], 0, 1,
                                                num_list[index] * Constant.PROPOSAL_NUM // Constant.BOLCK_SIZE, 0, 0)
                else:
                    break
                index += 1

        return input_ub, dest_pos_ub

    def pick(self, offset, task_idx, input_ub):
        # dest position in UB
        dest_pos_ub = self.k * Constant.PROPOSAL_NUM
        repeat_times = self.k // Constant.BOLCK_SIZE
        with self.tik_instance.for_range(0, self.num_gm) as i:
            self.tik_instance.data_move(input_ub[0], self.temp[offset + self.k * i * Constant.PROPOSAL_NUM], 0, 1,
                                        (self.k * Constant.PROPOSAL_NUM) // Constant.BOLCK_SIZE, 0, 0)
            # ascend
            with self.tik_instance.if_scope(self.descending is False):
                # data is continuous in GM & gather scattered data together
                with self.tik_instance.for_range(0, self.k) as i2:
                    input_ub[i2 + dest_pos_ub].set_as(
                        input_ub[(self.k - 1 - i2) * Constant.PROPOSAL_NUM + Constant.VAL_INDEX])

                # move output (float16) from UB to GM
                self.tik_instance.data_move(self.data_out[task_idx * self.totalnum + self.k * (self.num_gm - 1 - i)],
                                            input_ub[dest_pos_ub], 0, 1, repeat_times, 0, 0)

            # descend
            with self.tik_instance.else_scope():
                # data is continuous in GM & gather scattered data together
                self.tik_instance.vextract(input_ub[dest_pos_ub], input_ub[0], repeat_times, Constant.VAL_INDEX)

                # move output (float16) from UB to GM
                self.tik_instance.data_move(self.data_out[task_idx * self.totalnum + self.k * i], input_ub[dest_pos_ub],
                                            0, 1, repeat_times, 0, 0)

    def sort_in_ub(self, input_ub, i, index, offset):
        # dest position in UB
        dest_pos_ub = self.k * Constant.PROPOSAL_NUM
        repeat_times = self.k // Constant.BOLCK_SIZE
        # 1. Move data from OUT to UB
        self.tik_instance.data_move(input_ub[dest_pos_ub], self.input_gm[index + i * self.k], 0, 1, repeat_times, 0, 0)

        with self.tik_instance.if_scope(self.num < (i + 1) * self.k):
            # aline for k
            aline = self.k - self.num % self.k
            Min = self.tik_instance.Scalar('float16', init_value=-65504)
            # Add ineffective object for 16 alignment
            with self.tik_instance.for_range(0, aline % Constant.BOLCK_SIZE) as j:
                input_ub[dest_pos_ub + self.num % self.k + j].set_as(Min)
            # Add ineffective object for k alignment
            with self.tik_instance.if_scope(aline > Constant.BOLCK_SIZE - 1):
                self.tik_instance.vec_dup(Constant.BOLCK_SIZE,
                                          input_ub[dest_pos_ub + self.num % self.k + aline % Constant.BOLCK_SIZE],
                                          Min, aline // Constant.BOLCK_SIZE, 1)

        self.tik_instance.vconcat(input_ub[0], input_ub[dest_pos_ub], repeat_times, Constant.VAL_INDEX)

        # 2. vbs16
        self.tik_instance.vrpsort16(dst=input_ub[dest_pos_ub], src=input_ub[0], repeat_times=repeat_times)

        # 3. vms4
        input_ub, dest_pos_ub = self.vms4(input_ub, dest_pos_ub)

        # 4. Move Data from UB to OUT
        self.tik_instance.data_move(self.temp[offset + i * self.k * Constant.PROPOSAL_NUM], input_ub[dest_pos_ub], 0, 1,
                                    self.k * Constant.PROPOSAL_NUM // Constant.BOLCK_SIZE, 0, 0)

    def sort_in_gm(self, input_ub, offset):
        src_pos_ub = self.tik_instance.Scalar("int32")
        dest_pos_ub = self.tik_instance.Scalar("int32")

        with self.tik_instance.for_range(0, self.num_gm - 1) as tail:
            src_pos_ub.set_as(0)
            dest_pos_ub.set_as(self.batchsize * Constant.PROPOSAL_NUM)

            self.tik_instance.data_move(input_ub[src_pos_ub + self.k * Constant.PROPOSAL_NUM], self.temp[offset], 0, 1,
                                        (self.k * Constant.PROPOSAL_NUM) // Constant.BOLCK_SIZE, 0, 0)
            with self.tik_instance.for_range(1, self.num_gm - tail) as i:
                self.tik_instance.data_move(input_ub[src_pos_ub],
                                            self.temp[offset + self.k * i * Constant.PROPOSAL_NUM], 0, 1,
                                            (self.k * Constant.PROPOSAL_NUM) // Constant.BOLCK_SIZE, 0, 0)

                self.tik_instance.vmrgsort4(input_ub[dest_pos_ub],
                                            [input_ub[src_pos_ub],
                                             input_ub[src_pos_ub + self.k * Constant.PROPOSAL_NUM],
                                             input_ub[0], input_ub[0]], [self.k, self.k, 0, 0],
                                            if_exhausted_suspension=False,
                                            valid_bit="0011", repeat_times=1)

                self.tik_instance.data_move(self.temp[offset + self.k * (i - 1) * Constant.PROPOSAL_NUM],
                                            input_ub[dest_pos_ub],
                                            0, 1,
                                            (self.k * Constant.PROPOSAL_NUM) // Constant.BOLCK_SIZE, 0, 0)

                dest_pos_ub.set_as(src_pos_ub)
                src_pos_ub.set_as(self.batchsize * Constant.PROPOSAL_NUM - dest_pos_ub)

            # Move Data from UB to GM
            self.tik_instance.data_move(self.temp[offset + self.k * (self.num_gm - tail - 1) * Constant.PROPOSAL_NUM],
                                        input_ub[src_pos_ub + self.k * Constant.PROPOSAL_NUM], 0, 1,
                                        (self.k * Constant.PROPOSAL_NUM) // Constant.BOLCK_SIZE, 0, 0)

    def tune(self):
        """
        Function: remove min.
        """
        offset = self.k - self.num % self.k
        use_num = self.totalnum
        if self.totalnum > self.batchsize * Constant.BOLCK_SIZE:
            use_num = self.batchsize * Constant.BOLCK_SIZE 

        batchs = self.totalnum // use_num
        batchs = batchs if self.totalnum % use_num == 0 else (batchs + 1)

        float_ub = self.tik_instance.Tensor("float16", [use_num], name="float_ub", scope=tik.scope_ubuf)
        with self.tik_instance.for_range(0, self.rounds) as i:
            with self.tik_instance.for_range(0, batchs) as j:
                with self.tik_instance.if_scope(self.descending is False):
                    self.tik_instance.data_move(float_ub[0], self.data_out[i * self.totalnum + offset + j * use_num], 0,
                                                1,
                                                use_num // Constant.BOLCK_SIZE, 0, 0)
                    self.tik_instance.data_move(self.data_out_[i * self.num + j * use_num], float_ub[0], 0, 1,
                                                use_num // Constant.BOLCK_SIZE, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.data_move(float_ub[0], self.data_out[i * self.totalnum + j * use_num], 0, 1,
                                                use_num // Constant.BOLCK_SIZE, 0, 0)
                    self.tik_instance.data_move(self.data_out_[i * self.num + j * use_num], float_ub[0], 0, 1,
                                                use_num // Constant.BOLCK_SIZE, 0, 0)

    def sort_compute(self, task_idx):
        offset = (task_idx % self.used_core_num) * self.num_gm * self.k * Constant.PROPOSAL_NUM

        index = task_idx * self.num

        input_ub = self.tik_instance.Tensor(self.dtype, [self.batchsize * Constant.PROPOSAL_NUM * 2], name="input_ub",
                                            scope=tik.scope_ubuf)

        # SORT IN UB
        with self.tik_instance.for_range(0, self.num_gm) as i:
            self.sort_in_ub(input_ub, i, index, offset)

        # SORT IN GM
        self.sort_in_gm(input_ub, offset)

        # Pick Data from GM to GM
        self.pick(offset, task_idx, input_ub)
