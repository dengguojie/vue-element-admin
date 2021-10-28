# Copyright 2021 Huawei Technologies Co., Ltd
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
roll
"""
from te.utils import para_check
from te import tik
from te.platform import cce_intrin as te_irn
from te.platform import cce_conf as te_cnf
import functools as fctool
import math


class Roll(object):
    """
    Implementation of roll
    """
    def __init__(self, input_x, shifts, dims, kernel_name):
        """
        init of roll
        """
        self.tik_instance = tik.Tik()
        self.input_x_shape = list(input_x.get("shape"))
        self.input_x_dtype = input_x.get("dtype")
        self.dims = dims
        self.shifts = shifts
        self.flag = False
        # flag is true when dims is the default value
        if len(self.dims) == 0:
            self.flag = True
        self.kernel_name = kernel_name
        self.dtype_bytes_size_x = te_irn.get_bit_len(self.input_x_dtype) // 8
        self.data_each_block_x = 32 // self.dtype_bytes_size_x
        self.output_y_dtype = self.input_x_dtype
        self.output_y_shape = self.input_x_shape
        self.dtype_bytes_size_y = te_irn.get_bit_len(self.output_y_dtype) // 8
        self.data_each_block_y = 32 // self.dtype_bytes_size_y
        self.ub_size = te_cnf.get_soc_spec(te_cnf.UB_SIZE) - 64
        self.ub_tensor_size = self.ub_size // self.dtype_bytes_size_x // 2 // 2 // 32 * 32
        self.input_x_gm = self.tik_instance.Tensor(self.input_x_dtype,
                                                   self.input_x_shape,
                                                   name="input_x_gm",
                                                   scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.output_y_dtype,
                                                    self.output_y_shape,
                                                    name="output_y_gm",
                                                    scope=tik.scope_gm)
        self.in_num = fctool.reduce(lambda x, y: x * y, self.input_x_shape)
        if not self.flag:
            shape_tmp = self.input_x_shape
            self.dim = self.dims[0]
            self.shift = self.shifts[0]
            shape_dim = self.input_x_shape[self.dim]
            if self.shift < 0:
                self.shift = self.shift + shape_dim
            self.shift = self.shift % shape_dim
            after_shape = shape_tmp[self.dim + 1:]
            if len(after_shape) == 0:
                self.after_num = 1
            else:
                self.after_num = fctool.reduce(lambda x, y: x * y, after_shape)
        else:
            self.shift = self.shifts[0]
            if self.shift < 0:
                self.shift = self.shift + self.in_num
            self.shift = self.shift % self.in_num

        if self.in_num <= self.ub_tensor_size:
            self.ai_core_num = 1
        else:
            self.ai_core_num = te_cnf.get_soc_spec(te_cnf.CORE_NUM)

        if self.ai_core_num == 1:
            self.num_each_core = self.in_num
        else:
            self.num_each_core = self.in_num // self.ai_core_num // 32 * 32
        self.last_core_num = self.in_num - self.num_each_core * self.ai_core_num

    def roll(self):
        """
        Calculate total entrance
        """
        self.roll_compute()
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_x_gm],
                                   outputs=[self.output_y_gm])
        return self.tik_instance

    def roll_compute(self):
        """
        compute entrance
        """
        with self.tik_instance.for_range(0, self.ai_core_num,
                                         block_num=self.ai_core_num) as core_id:
            move_offset = self.num_each_core * core_id
            self.roll_compute_each_core(move_offset,
                                        self.num_each_core)
        move_offset = self.num_each_core * self.ai_core_num
        if self.last_core_num > 0:
            self.roll_compute_each_core(move_offset,
                                        self.last_core_num)

    def roll_compute_each_core(self, core_move_offset, core_move_num):
        """
        Compute on each core
        """
        loop_time = core_move_num // self.ub_tensor_size
        move_offset = core_move_offset
        need_db = True
        if loop_time < 2:
            need_db = False
        if loop_time > 0:
            if need_db:
                with self.tik_instance.for_range(0, loop_time, thread_num=2) as loop_id:
                    move_offset = loop_id * self.ub_tensor_size + core_move_offset
                    self.roll_compute_each_loop(move_offset,
                                                self.ub_tensor_size)
            else:
                with self.tik_instance.for_range(0, loop_time) as loop_id:
                    move_offset = loop_id * self.ub_tensor_size + core_move_offset
                    self.roll_compute_each_loop(move_offset,
                                                self.ub_tensor_size)
            move_offset = loop_time * self.ub_tensor_size + core_move_offset
        last_num = core_move_num % self.ub_tensor_size
        if last_num > 0:
            self.roll_compute_each_loop(move_offset, last_num)

    def roll_compute_each_loop(self, move_offset, move_num):
        """
        compute each loop
        move_num <= ub_tensor_size
        """
        self.init_ub_tensor_and_scalar()
        self.loop_first_offset.set_as(move_offset)
        burse_len = math.ceil(move_num / self.data_each_block_x)
        if not self.flag:
            self.begin.set_as(move_offset // self.after_num)
            with self.tik_instance.if_scope((move_offset + move_num) % self.after_num == 0):
                self.end.set_as((move_offset + move_num) // self.after_num)
            with self.tik_instance.else_scope():
                self.end.set_as((move_offset + move_num) // self.after_num + 1)
            with self.tik_instance.for_range(self.begin, self.end) as i:
                with self.tik_instance.if_scope(i == move_offset // self.after_num):
                    self.offset_this_dim.set_as(move_offset)
                    with self.tik_instance.if_scope((move_offset + move_num) >= (self.after_num * (i + 1))):
                        self.num_this_dim.set_as(self.after_num * (i + 1) - move_offset)
                    with self.tik_instance.else_scope():
                        self.num_this_dim.set_as(move_num)
                with self.tik_instance.else_scope():
                    self.offset_this_dim.set_as(self.after_num * i)
                    with self.tik_instance.if_scope((move_offset + move_num) >= (self.after_num * (i + 1))):
                        self.num_this_dim.set_as(self.after_num)
                    with self.tik_instance.else_scope():
                        self.num_this_dim.set_as((move_offset + move_num) % self.after_num)
                self.roll_each_dim()
            self.tik_instance.data_move(self.output_y_gm[move_offset],
                                        self.input_x_ub,
                                        0, 1, burse_len, 0, 0)
        else:
            self.ori_first_offset.set_as(move_offset - self.shift)
            with self.tik_instance.if_scope(self.ori_first_offset < 0):
                self.ori_first_offset.set_as(self.ori_first_offset + self.in_num)
            self.ori_last_offset.set_as(self.ori_first_offset + move_num)
            # cut off
            with self.tik_instance.if_scope(self.ori_last_offset > self.in_num):
                self.ori_last_offset.set_as(self.ori_last_offset % self.in_num)
                # the front section
                self.num_front.set_as(self.in_num - self.ori_first_offset)
                self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_front)
                self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
                self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
                self.tik_instance.data_move(self.tmp_ub[0],
                                            self.input_x_gm[self.ori_first_offset],
                                            0, 1, self.burse_len, 0, 0)
                with self.tik_instance.for_range(0, self.num_front) as n_id:
                    self.input_x_ub[move_offset + n_id].set_as(self.tmp_ub[n_id])
                # the back section
                self.num_back.set_as(self.ori_last_offset)
                self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_back)
                self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
                self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
                self.tik_instance.data_move(self.tmp_ub[0],
                                            self.input_x_gm[0],
                                            0, 1, self.burse_len, 0, 0)
                with self.tik_instance.for_range(0, self.num_back) as n_id:
                    self.input_x_ub[move_offset + self.num_front + n_id].set_as(self.tmp_ub[n_id])
                self.tik_instance.data_move(self.output_y_gm[move_offset],
                                            self.input_x_ub,
                                            0, 1, burse_len, 0, 0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(self.input_x_ub[0],
                                            self.input_x_gm[self.ori_first_offset],
                                            0, 1, burse_len, 0, 0)
                self.tik_instance.data_move(self.output_y_gm[move_offset],
                                            self.input_x_ub[0],
                                            0, 1, burse_len, 0, 0)

    def init_ub_tensor_and_scalar(self):
        """
        init tensor and scalar in ub
        """
        self.input_x_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                                   (self.ub_tensor_size,),
                                                   name="input_x_ub",
                                                   scope=tik.scope_ubuf)
        self.tmp_ub = self.tik_instance.Tensor(self.input_x_dtype,
                                               (self.ub_tensor_size,),
                                               name="tmp_ub",
                                               scope=tik.scope_ubuf)
        self.offset_this_dim = self.tik_instance.Scalar(dtype="int32")
        self.num_this_dim = self.tik_instance.Scalar(dtype="int32")
        self.begin = self.tik_instance.Scalar(dtype="int32")
        self.end = self.tik_instance.Scalar(dtype="int32")
        self.compute_burse_len_num = self.tik_instance.Scalar(dtype="float32")
        self.burse_len = self.tik_instance.Scalar(dtype="int32")
        self.first_offset = self.tik_instance.Scalar(dtype="int32")
        self.last_offset = self.tik_instance.Scalar(dtype="int32")
        self.ori_first_offset = self.tik_instance.Scalar(dtype="int32")
        self.ori_last_offset = self.tik_instance.Scalar(dtype="int32")
        self.num_need_make_up = self.tik_instance.Scalar(dtype="int32")
        self.num_front = self.tik_instance.Scalar(dtype="int32")
        self.num_back = self.tik_instance.Scalar(dtype="int32")
        self.burse_num = self.tik_instance.Scalar(dtype="int32")
        self.loop_first_offset = self.tik_instance.Scalar(dtype="int32")

    def roll_each_dim(self):
        """
        roll compute on each dim
        """
        self.tik_instance.scalar_conv('', self.compute_burse_len_num, self.num_this_dim)
        self.compute_burse_len_num.set_as(self.compute_burse_len_num / self.data_each_block_x)
        self.tik_instance.scalar_conv('ceil', self.burse_len, self.compute_burse_len_num)
        self.ori_first_offset.set_as((self.offset_this_dim - self.shift * self.after_num))
        with self.tik_instance.if_scope(self.ori_first_offset < 0):
            self.ori_first_offset.set_as(self.ori_first_offset + self.in_num)
        with self.tik_instance.else_scope():
            self.ori_first_offset.set_as(self.ori_first_offset)
        self.ori_last_offset.set_as(self.ori_first_offset + self.num_this_dim)
        self.tik_instance.data_move(self.tmp_ub[0],
                                    self.input_x_gm[self.ori_first_offset],
                                    0, 1, self.burse_len, 0, 0)
        with self.tik_instance.for_range(0, self.num_this_dim) as n_id:
            self.input_x_ub[self.offset_this_dim - self.loop_first_offset
                            + n_id].set_as(self.tmp_ub[n_id])


#pylint: disable=unused argument
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT,
                            para_check.OPTION_ATTR_LIST_INT, para_check.OPTION_ATTR_LIST_INT,
                            para_check.KERNEL_NAME)
def roll(x, y, shifts, dims, kernel_name="roll"):
    """
    roll the data according to the shifts and dims

    Parameters
    ----------
    x : dict
    shape and dtype of input_x
    y : dict
    shape and dtype of output_y, should be same shape as input, dtype is same as the quantified type
    shifts: list
    the processed shifts
    dims: list
    the processed dim
    kernel_name : str
    kernel name, default value is "roll"

    Returns
    -------
    None
    """
    shifts = list(shifts)
    dims = list(dims)
    shape_x = x.get("shape")
    dtype_x = x.get("dtype")
    check_x_tuple = ("float16", "float32", "int32", "uint32", "int8", "uint8")
    dims_len = len(shape_x)
    ne_dims = dims_len * -1
    if dtype_x not in check_x_tuple:
        raise RuntimeError("X only support %s while dtype is %s" %
                           (",".join(check_x_tuple), dtype_x))
    if len(dims) == 0:
        if len(shifts) != 1:
            raise RuntimeError("Len(dims) is 0 while len(shifts) > 1")
        for shift in shifts:
            dtype_shift = type(shift)
            if dtype_shift != int:
                raise RuntimeError("Shifts only support {} while dtype is {}".format(int, dtype_shift))
    else:
        if len(dims) != len(shifts):
            raise RuntimeError("Len(dims) should be equal to len(shifts), while len(dims) is {}, len(shifts) is {}"
                               .format(len(dims), len(shifts)))
        for shift in shifts:
            dtype_shift = type(shift)
            if dtype_shift != int:
                raise RuntimeError("Shifts only support {} while dtype is {}".format(int, dtype_shift))

        for i in range(len(dims)):
            dim = dims[i]
            dtype_dim = type(dim)
            if dtype_dim != int:
                raise RuntimeError("Dims only support {} while dtype is {}".format(int, dtype_dim))
            if dim < ne_dims or dim > (dims_len - 1):
                raise RuntimeError("Only support {} =< dim <= {} while dim is {}".format(ne_dims, dims_len - 1, dim))
            if dim < 0:
                dims[i] = dims[i] + dims_len

    para_check.check_shape_rule(shape_x)
    para_check.check_shape_size(shape_x)
    para_check.check_kernel_name(kernel_name)
    roll_instance = Roll(x, shifts, dims, kernel_name)
    return roll_instance.roll()
