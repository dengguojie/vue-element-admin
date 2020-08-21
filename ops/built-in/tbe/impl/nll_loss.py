"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

nll_loss
"""
# pylint: disable=ungrouped-imports,import-error
import math
from te import tik
from topi.cce import util
from te import platform as tbe_platform
from impl.constant_util import MASK64
from impl.constant_util import BLOCK_SIZE
from impl.constant_util import DATA_SIZE_FOUR
from impl.constant_util import DATA_SIZE_EIGHT

DIM2 = 2
NEGATIVE = -1
NUM_UB_SIZE = MASK64
ONE_KB = 1024
NUM_SIXTYFOUR = MASK64


# pylint: disable=locally-disabled,unused-argument,too-many-locals,invalid-name
def _shape_and_dtype_check(x, target, weight, kernel_name):
    x_shape = x.get("shape")
    x_dtype = x.get("dtype").lower()
    target_shape = target.get("shape")
    target_dtype = target.get("dtype").lower()
    weight_shape = weight.get("shape")
    weight_dtype = weight.get("dtype").lower()

    util.check_shape_rule(x_shape)
    util.check_shape_rule(target_shape)
    util.check_shape_rule(weight_shape)
    util.check_tensor_shape_size(x_shape)
    util.check_tensor_shape_size(target_shape)
    util.check_tensor_shape_size(weight_shape)
    util.check_kernel_name(kernel_name)
    util.check_dtype_rule(x_dtype, "float32")
    util.check_dtype_rule(target_dtype, "int32")
    util.check_dtype_rule(weight_dtype, "float32")
    if len(x_shape) > DIM2:
        raise RuntimeError("The dimension of x should be equal to"
                           "or less than 2")
    if len(target_shape) != 1:
        raise RuntimeError("The dimension of target only support 1")
    if len(x_shape) == DIM2 and x_shape[0] != target_shape[0]:
        raise RuntimeError("The first dimension of x and"
                           " target should be equal")
    if len(weight_shape) != 1:
        raise RuntimeError("The dimension of weight only support 1")
    if x_shape[-1] != weight_shape[0]:
        raise RuntimeError("The last dimension of x and the first dimension"
                           "of weight should be equal")


# pylint: disable=too-many-instance-attributes,too-many-arguments
# pylint: disable=too-many-locals,too-many-statements
# pylint: disable=attribute-defined-outside-init
class nll_loss_compute:
    """
    NLLLOSS

    Returns
    -------
    None
    """
    def __init__(self, x, target, weight, reduction, kernel_name):
        self.init_tik_instance()
        self.in_burst_len = 1
        self.out_burst_len = 1
        self.target_burst_len = 1
        self.weight_burst_len = 1
        self.target_ub_size = 1
        self.weight_ub_size = 1
        self.last_burst_len = 1
        self.max_line_in_ub = 0
        self.move_times = 1
        self.x_ub_size = 1
        self.target = target
        self.weight = weight
        self.reduction = reduction
        self.kernel_name = kernel_name
        self.x_dtype = x.get("dtype").lower()
        self.x_shape = x.get("shape")
        self.target_shape = target.get("shape")
        self.target_dtype = target.get("dtype").lower()
        self.weight_shape = weight.get("shape")
        self.weight_dtype = weight.get("dtype").lower()
        self.x_dim = len(self.x_shape)
        self.init_size()
        self.init_gm()

    def init_tik_instance(self):
        """
        init the tik_instance.

        Parameters
        ----------

        Returns
        -------
        None
        """
        profile = tik.Dprofile()
        self.tik_instance = tik.Tik(profile)

    def init_size(self):
        """
        init the size of args.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x_gm_size = 1
        self.core_num = self.x_shape[0]
        for i in self.x_shape:
            self.x_gm_size = i * self.x_gm_size
        self.target_gm_size = self.target_shape[0]
        self.weight_gm_size = self.weight_shape[0]
        if self.x_dim == DIM2 and self.reduction == "none":
            self.output_gm_size = self.x_shape[0]
        else:
            self.output_gm_size = 1
        self.stride_len = self.x_shape[1]
        self.total_weight_size = 1
        self.target_ub_size = math.ceil(self.x_shape[0] /
                                        DATA_SIZE_EIGHT)*BLOCK_SIZE
        self.weight_ub_size = math.ceil(self.weight_shape[0] /
                                        DATA_SIZE_EIGHT)*BLOCK_SIZE

    def init_gm(self):
        """
        init the gm of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.data_x = self.tik_instance.Tensor(self.x_dtype, [self.x_gm_size],
                                               name="data_x",
                                               scope=tik.scope_gm)
        self.data_target = self.tik_instance.Tensor(self.target_dtype,
                                                    [self.target_gm_size],
                                                    name="data_target",
                                                    scope=tik.scope_gm)
        self.data_weight = self.tik_instance.Tensor(self.weight_dtype,
                                                    [self.weight_gm_size],
                                                    name="data_weight",
                                                    scope=tik.scope_gm)
        self.index_x = self.tik_instance.Scalar(dtype="int32")
        self.output = self.tik_instance.Tensor(self.x_dtype,
                                               [self.output_gm_size],
                                               name="output",
                                               scope=tik.scope_gm,
                                               is_atomic_add=True)
        if self.x_dim == DIM2 and (self.reduction == "sum"):
            self.total_weight = self.tik_instance.Tensor(
                self.x_dtype, [self.total_weight_size], name="total_weight",
                scope=tik.scope_gm, is_atomic_add=True)
        else:
            self.total_weight = self.tik_instance.Tensor(
                self.x_dtype, [self.total_weight_size], name="total_weight",
                scope=tik.scope_gm)

    def init_ub(self):
        """
        init the ub of input and output.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.x_ub = self.tik_instance.Tensor(self.x_dtype, [self.x_ub_size],
                                             name="x_ub",
                                             scope=tik.scope_ubuf)
        if len(self.x_shape) == 1:
            self.target_ub = self.tik_instance.Tensor(self.target_dtype,
                                                      [NUM_UB_SIZE],
                                                      name="target_ub",
                                                      scope=tik.scope_ubuf)
        else:
            self.target_ub = self.tik_instance.Tensor(
                self.target_dtype, [self.target_ub_size/DATA_SIZE_FOUR],
                name="target_ub", scope=tik.scope_ubuf)

        self.weight_ub = self.tik_instance.Tensor(
            self.x_dtype, [self.weight_ub_size/DATA_SIZE_FOUR],
            name="weight_ub", scope=tik.scope_ubuf)

    def one_dim_compute(self):
        """
        calculate process when input is 1D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.init_ub()
        self.tik_instance.data_move(self.target_ub, self.data_target,
                                    1, 1, 1, 0, 0)
        self.index_x.set_as(self.target_ub(0))
        self.tik_instance.data_move(self.x_ub, self.data_x[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.weight_ub,
                                    self.data_weight[self.index_x],
                                    1, 1, 1, 0, 0)
        self.tik_instance.vmul(1, self.x_ub, self.x_ub, self.weight_ub,
                               1, 1, 1, 1, 1, 0, 0)
        self.tik_instance.vmuls(1, self.x_ub, self.x_ub, NEGATIVE,
                                1, 1, 1, 0, 0)
        self.tik_instance.data_move(self.output, self.x_ub, 1, 1, 1, 0, 0)

    def calculate_tiling(self):
        """
        init the gm of input and output

        Parameters
        ----------

        Returns
        -------
        None
        """
        shape = self.x_shape
        ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer") - ONE_KB
        self.target_burst_len = math.ceil(shape[0]/DATA_SIZE_EIGHT)
        self.weight_burst_len = math.ceil(shape[1]/DATA_SIZE_EIGHT)
        res_ub_size = ub_size_bytes - self.weight_ub_size*2 - \
                      self.target_ub_size*2
        self.move_times = math.ceil((shape[0]*shape[1]*DATA_SIZE_FOUR)/
                                    res_ub_size)
        if shape[0]*shape[1]*DATA_SIZE_FOUR > res_ub_size:
            num = res_ub_size//(shape[1]*DATA_SIZE_FOUR)
            total_num = num*shape[1]
            self.in_burst_len = math.ceil(total_num/DATA_SIZE_EIGHT)
            self.last_burst_len = math.ceil((shape[0] -
                                             (self.move_times - 1)*num) /
                                            DATA_SIZE_EIGHT)
            self.max_line_in_ub = num
            self.x_ub_size = res_ub_size
        else:
            total_num = shape[0]*shape[1]
            self.last_burst_len = math.ceil(total_num/DATA_SIZE_EIGHT)
            self.x_ub_size = total_num

    def two_dim_compute(self):
        """
        calculate process when x is 2D.

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.reduction == "none":
            self.calculate_tiling()
            self.init_ub()
            self.temp_ub = self.tik_instance.Tensor(
                self.x_dtype, [self.target_ub_size/DATA_SIZE_FOUR],
                name="temp_ub", scope=tik.scope_ubuf)
            self.temp_weight_ub = self.tik_instance.Tensor(
                self.x_dtype, [self.target_ub_size /
                               DATA_SIZE_FOUR],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            self.tik_instance.data_move(self.target_ub, self.data_target, 0, 1,
                                        self.target_burst_len, 0, 0)
            self.tik_instance.data_move(self.weight_ub, self.data_weight, 0, 1,
                                        self.weight_burst_len, 0, 0)
            for cycle in range(self.move_times-1):
                self.tik_instance.data_move(
                    self.x_ub, self.data_x[cycle*self.max_line_in_ub *
                                           self.x_shape[1]],
                    0, 1, self.in_burst_len, 0, 0)
                for i in range(self.max_line_in_ub):
                    self.index_x.set_as(
                        self.target_ub(i + cycle*self.max_line_in_ub))
                    self.tik_instance.data_move(
                        self.temp_ub[i + cycle*self.max_line_in_ub],
                        self.x_ub[self.x_shape[1]*i + self.index_x],
                        0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        self.temp_weight_ub[i + cycle*self.max_line_in_ub],
                        self.weight[self.index_x], 0, 1, 1, 0, 0)
            last_start_addr = (self.move_times - 1)*self.max_line_in_ub * \
                              self.x_shape[1]
            last_target_num = self.x_shape[0] - (self.move_times - 1) * \
                              self.max_line_in_ub
            self.tik_instance.data_move(self.x_ub, self.data_x(last_start_addr),
                                        0, 1, self.last_burst_len, 0, 0)
            for i in range(last_target_num):
                self.index_x.set_as(
                    self.target_ub(i+(self.move_times - 1)*self.max_line_in_ub))
                self.temp_ub[
                    i+(self.move_times-1)*self.max_line_in_ub].set_as \
                    (self.x_ub[self.x_shape[1]*i + self.index_x])
                self.temp_weight_ub[
                    i+(self.move_times-1)*self.max_line_in_ub].set_as \
                    (self.weight_ub[self.index_x])
            vmul_repeat = math.ceil(self.x_shape[0]/MASK64)
            self.tik_instance.vmul(MASK64, self.temp_ub, self.temp_ub,
                                   self.temp_weight_ub,
                                   vmul_repeat, 1, 1, 1, 8, 8, 8)
            self.tik_instance.vmuls(MASK64, self.temp_ub, self.temp_ub,
                                    NEGATIVE, vmul_repeat, 1, 1, 8, 8)
            self.tik_instance.data_move(self.output, self.temp_ub, 0, 1,
                                        self.target_burst_len, 0, 0)

        elif self.reduction == "sum":
            with self.tik_instance.for_range(0, self.core_num,
                                             block_num=self.core_num) as cycle:
                self.init_ub()
                self.tik_instance.data_move(self.target_ub,
                                            self.data_target[cycle],
                                            0, 1, 1, 0, 0)
                self.index_x.set_as(self.target_ub[0])
                self.tik_instance.data_move(
                    self.x_ub, self.data_x[(self.stride_len*cycle)
                                           + self.index_x],
                    0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.weight_ub,
                                            self.data_weight[self.index_x],
                                            0, 1, 1, 0, 0)
                self.tik_instance.vmul(1, self.x_ub, self.x_ub, self.weight_ub,
                                       1, 1, 1, 1, 1, 0, 0)
                self.tik_instance.vmuls(1, self.x_ub, self.x_ub, NEGATIVE,
                                        1, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output, self.x_ub,
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.total_weight, self.weight_ub,
                                            0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def reduction_is_mean_tiling(self):
        """
        calculate size when x is 2D and resuction is mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        self.big_target = False
        self.ub_size_bytes = tbe_platform.CceProductParams().getParams(
            "Unified_Buffer") - ONE_KB
        self.x_size = math.ceil(self.x_shape[0]*self.x_shape[-1]/64)*64
        self.target_ub_size = math.ceil(self.target_shape[0]/NUM_SIXTYFOUR) * \
                              NUM_SIXTYFOUR
        self.refactor_weight_size = self.target_ub_size
        self.weight_ub_size = math.ceil(
            self.weight_shape[0]/NUM_SIXTYFOUR+1)*NUM_SIXTYFOUR
        self.work_tensor_size = math.ceil(self.target_shape[0]/512)*8
        self.no_align_last_ub_size = self.ub_size_bytes/4 - \
            self.target_ub_size - self.refactor_weight_size - \
            self.weight_ub_size - self.work_tensor_size
        self.last_ub_size = (self.no_align_last_ub_size//64)*64
        self.move_max_line = (self.last_ub_size//(self.x_shape[-1]+1)//8)*8
        self.refactor_x_size = math.ceil(self.move_max_line/64)*64
        if self.move_max_line < 8:
            self.big_target = True
            self.no_align_last_ub_size = self.ub_size_bytes/4 - \
                self.weight_ub_size
            self.last_ub_size = (self.no_align_last_ub_size//64)*64
            self.move_max_line = self.last_ub_size // (self.x_shape[-1]+4)
            self.target_ub_size = math.ceil(
                self.move_max_line/NUM_SIXTYFOUR)*NUM_SIXTYFOUR
            self.refactor_x_size = self.target_ub_size
            self.refactor_weight_size = self.target_ub_size
            self.work_tensor_size = math.ceil(self.move_max_line/512)*8
        self.move_max_burst = math.ceil(self.move_max_line*self.x_shape[-1]/8)
        self.move_last_line = self.target_shape[0] % self.move_max_line
        self.max_vmul_repeat = math.ceil(self.move_max_line/64)
        self.last_vmul_repeat = math.ceil(self.move_last_line/64)
        self.compute_tail_mask = self.max_vmul_repeat*64 - self.move_max_line
        self.compute_offset = (self.max_vmul_repeat - 1)*64
        if self.move_last_line:
            self.move_last_burst = math.ceil(self.move_last_line *
                                             self.x_shape[-1]/8)
        else:
            self.move_last_line = self.move_max_line
            self.move_last_burst = self.move_max_burst
            self.last_vmul_repeat = self.max_vmul_repeat
        self.move_times = math.ceil(self.target_shape[0]/self.move_max_line)
        self.x_offset = self.move_max_line*self.x_shape[-1]
        if self.move_times == 1:
            self.max_vmul_repeat = self.last_vmul_repeat
            self.move_max_line = self.move_last_line
            self.move_max_burst = self.move_last_burst

    def sum_compute(self, sum_shape, dst_ub, src_ub, work_ub):
        """
        calculate tensor sum.

        Parameters
        ----------

        Returns
        -------
        None
        """
        vcadd_repeat_time = math.ceil(sum_shape/64)
        head_repeat = vcadd_repeat_time - 1
        tail_mask = int(sum_shape % 64)
        if tail_mask == 0:
            self.tik_instance.vec_reduce_add(MASK64, dst_ub, src_ub,
                                             work_ub, vcadd_repeat_time, 8)

        else:
            self.tik_instance.vcadd(
                tail_mask, dst_ub,
                src_ub[head_repeat*64], 1, 1, 1, 8)
            if head_repeat:
                self.tik_instance.vec_reduce_add(MASK64, dst_ub[64], src_ub,
                                                 work_ub, head_repeat, 8)
                self.tik_instance.vadd(1, dst_ub, dst_ub,
                                       dst_ub[64], 1, 1, 1, 1, 8, 8, 8)

    def reduction_is_mean_compute(self):
        """
        calculate process when x is 2D and resuction is mean.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.move_times,
                                         block_num=1) as cycle:
            refactor_weight_ub = self.tik_instance.Tensor(
                "float32", [self.refactor_weight_size],
                name="refactor_weight_ub",
                scope=tik.scope_ubuf)
            refactor_x_ub = self.tik_instance.Tensor(
                "float32", [self.refactor_x_size],
                name="refactor_x_ub",
                scope=tik.scope_ubuf)
            temp_weight_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            work_tensor_ub = self.tik_instance.Tensor(
                "float32", [self.work_tensor_size],
                name="work_tensor_ub",
                scope=tik.scope_ubuf)
            target_ub = self.tik_instance.Tensor("float32",
                                                 [self.target_ub_size],
                                                 name="target_ub",
                                                 scope=tik.scope_ubuf)
            x_ub = self.tik_instance.Tensor("float32", [self.move_max_burst*8],
                                            name="x_ub", scope=tik.scope_ubuf)

            self.tik_instance.data_move(temp_weight_ub, self.data_weight, 0, 1,
                                        self.weight_ub_size//8, 0, 0)
            self.tik_instance.data_move(target_ub, self.data_target, 0, 1,
                                        self.target_ub_size//8, 0, 0)
            with self.tik_instance.for_range(0, self.x_shape[0]) as i:
                self.index_x.set_as(target_ub[i])
                refactor_weight_ub[i].set_as(temp_weight_ub[self.index_x])

            self.sum_compute(self.x_shape[0], temp_weight_ub,
                             refactor_weight_ub, work_tensor_ub)
            with self.tik_instance.if_scope(cycle < self.move_times-1):
                self.tik_instance.data_move(
                    x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_max_burst, 8, 8)
                with self.tik_instance.for_range(0, self.move_max_line) as i:
                    self.index_x.set_as(target_ub[i])
                    refactor_x_ub[i].set_as(x_ub[self.index_x +
                                                 i*self.x_shape[-1]])
                self.tik_instance.vmuls(MASK64, refactor_x_ub, refactor_x_ub,
                                        NEGATIVE, self.max_vmul_repeat,
                                        1, 1, 8, 8)
                self.tik_instance.vmul(
                    MASK64, refactor_x_ub, refactor_x_ub,
                    refactor_weight_ub[cycle*self.move_max_line],
                    self.max_vmul_repeat, 1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_max_line, x_ub,
                                 refactor_x_ub, work_tensor_ub)
                self.tik_instance.vdiv(1, x_ub, x_ub, temp_weight_ub,
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output, x_ub, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)
            with self.tik_instance.else_scope():
                self.tik_instance.data_move(
                    x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_last_burst, 8, 8)
                with self.tik_instance.for_range(0, self.move_last_line) as i:
                    self.index_x.set_as(target_ub[i])
                    refactor_x_ub[i].set_as(x_ub[self.index_x +
                                                 i*self.x_shape[-1]])
                self.tik_instance.vmuls(MASK64, refactor_x_ub, refactor_x_ub,
                                        NEGATIVE, self.last_vmul_repeat,
                                        1, 1, 8, 8)
                self.tik_instance.vmul(
                    MASK64, refactor_x_ub, refactor_x_ub,
                    refactor_weight_ub[cycle*self.move_max_line],
                    self.last_vmul_repeat, 1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_last_line, x_ub, refactor_x_ub,
                                 work_tensor_ub)
                self.tik_instance.vdiv(1, x_ub, x_ub, temp_weight_ub,
                                       1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.data_move(self.total_weight, temp_weight_ub,
                                            0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(1)
                self.tik_instance.data_move(self.output, x_ub, 0, 1, 1, 0, 0)
                self.tik_instance.set_atomic_add(0)

    def big_target_compute(self):
        """
        calculate while target cannot move to ub one time.

        Parameters
        ----------

        Returns
        -------
        None
        """
        with self.tik_instance.for_range(0, self.move_times) as cycle:
            refactor_weight_ub = self.tik_instance.Tensor(
                "float32", [self.refactor_weight_size],
                name="refactor_weight_ub",
                scope=tik.scope_ubuf)
            refactor_x_ub = self.tik_instance.Tensor(
                "float32", [self.refactor_x_size],
                name="refactor_x_ub",
                scope=tik.scope_ubuf)
            temp_weight_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            temp_total_x_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            temp_total_weight_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            temp_output_ub = self.tik_instance.Tensor(
                "float32", [self.weight_ub_size],
                name="temp_weight_ub",
                scope=tik.scope_ubuf)
            work_tensor_ub = self.tik_instance.Tensor(
                "float32", [self.work_tensor_size],
                name="work_tensor_ub",
                scope=tik.scope_ubuf)
            target_ub = self.tik_instance.Tensor("float32",
                                                 [self.target_ub_size],
                                                 name="target_ub",
                                                 scope=tik.scope_ubuf)
            x_ub = self.tik_instance.Tensor("float32", [self.move_max_burst*8],
                                            name="x_ub", scope=tik.scope_ubuf)
            self.tik_instance.data_move(temp_weight_ub, self.data_weight, 0, 1,
                                        self.weight_ub_size//8, 0, 0)
            with self.tik_instance.if_scope(cycle < self.move_times - 1):
                self.tik_instance.data_move(
                    x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_max_burst, 0, 0)
                self.tik_instance.data_move(
                    target_ub, self.data_target[cycle*self.move_max_line],
                    0, 1, self.target_ub_size//8, 0, 0)
                with self.tik_instance.for_range(0, self.move_max_line) as i:
                    self.index_x.set_as(target_ub[i])
                    refactor_weight_ub[i].set_as(temp_weight_ub[self.index_x])
                    refactor_x_ub[i].set_as(
                        x_ub[self.index_x + i*self.x_shape[-1]])
                self.tik_instance.vmuls(MASK64, refactor_x_ub, refactor_x_ub,
                                        NEGATIVE, self.max_vmul_repeat,
                                        1, 1, 8, 8)
                self.tik_instance.vmul(
                    MASK64, refactor_x_ub, refactor_x_ub,
                    refactor_weight_ub, self.max_vmul_repeat, 1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_max_line, x_ub, refactor_x_ub,
                                 work_tensor_ub)
                self.sum_compute(self.move_max_line, temp_total_weight_ub,
                                 refactor_weight_ub, work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        temp_output_ub, temp_total_weight_ub, 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        temp_total_x_ub, x_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(1, temp_output_ub, temp_output_ub,
                                           temp_total_weight_ub, 1, 1, 1,
                                           1, 8, 8, 8)
                    self.tik_instance.vadd(1, temp_total_x_ub, temp_total_x_ub,
                                           x_ub, 1, 1, 1, 1, 8, 8, 8)
            with self.tik_instance.if_scope(cycle == self.move_times - 1):
                self.tik_instance.data_move(
                    x_ub, self.data_x[cycle*self.x_offset], 0, 1,
                    self.move_last_burst, 0, 0)
                self.tik_instance.data_move(
                    target_ub, self.data_target[cycle*self.move_max_line],
                    0, 1, math.ceil(self.move_last_line/8), 0, 0)
                with self.tik_instance.for_range(0, self.move_last_line) as i:
                    self.index_x.set_as(target_ub[i])
                    refactor_weight_ub[i].set_as(temp_weight_ub[self.index_x])
                    refactor_x_ub[i].set_as(
                        x_ub[self.index_x + i*self.x_shape[-1]])
                self.tik_instance.vmuls(MASK64, refactor_x_ub, refactor_x_ub,
                                        NEGATIVE, self.last_vmul_repeat,
                                        1, 1, 8, 8)
                self.tik_instance.vmul(
                    MASK64, refactor_x_ub, refactor_x_ub,
                    refactor_weight_ub, self.last_vmul_repeat,
                    1, 1, 1, 8, 8, 8)
                self.sum_compute(self.move_last_line, x_ub,
                                 refactor_x_ub, work_tensor_ub)
                self.sum_compute(self.move_last_line, temp_total_weight_ub,
                                 refactor_weight_ub, work_tensor_ub)
                with self.tik_instance.if_scope(cycle == 0):
                    self.tik_instance.data_move(
                        temp_output_ub, temp_total_weight_ub, 0, 1, 1, 0, 0)
                    self.tik_instance.data_move(
                        temp_total_x_ub, x_ub, 0, 1, 1, 0, 0)
                with self.tik_instance.else_scope():
                    self.tik_instance.vadd(1, temp_output_ub, temp_output_ub,
                                           temp_total_weight_ub, 1, 1, 1,
                                           1, 8, 8, 8)
                    self.tik_instance.vadd(1, temp_total_x_ub, temp_total_x_ub,
                                           x_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.vdiv(1, temp_total_x_ub, temp_total_x_ub,
                                       temp_output_ub, 1, 1, 1, 1, 8, 8, 8)
                self.tik_instance.data_move(self.output, temp_total_x_ub,
                                            0, 1, 1, 0, 0)
                self.tik_instance.data_move(self.total_weight, temp_output_ub,
                                            0, 1, 1, 0, 0)

    def nll_loss_compute_start(self):
        """
        Different calculation methods

        Parameters
        ----------

        Returns
        -------
        None
        """
        if self.x_dim == 1:
            self.one_dim_compute()
        elif self.x_dim == DIM2 and self.reduction != "mean":
            self.two_dim_compute()
        elif self.x_dim == DIM2 and self.reduction == "mean":
            self.reduction_is_mean_tiling()
            if self.big_target:
                self.big_target_compute()
            else:
                self.reduction_is_mean_compute()

        if self.x_dim == DIM2 and (self.reduction == "sum" or
                                   self.reduction == "mean"):
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.data_x, self.data_target,
                                               self.data_weight],
                                       outputs=[self.output, self.total_weight])
        else:
            self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                       inputs=[self.data_x, self.data_target,
                                               self.data_weight],
                                       outputs=[self.output])
        return self.tik_instance


@util.check_input_type(dict, dict, dict, dict, dict, str, str)
def nll_loss(x, target, weight, y, total_weight, reduction="mean",
             kernel_name="nll_loss"):
    """
    calculating data

    Parameters
    ----------
    x : dict
        shape and dtype of input, the length of shape should be two or one.
    target : dict
        shape and dtype of input, the length of shape only support one.
    weight : dict or None
        the length of shape only support one when weight is dict.
    y:dict
        It’s a tensor with shape(minibatch, ) when reduction == ‘none’ and
        the input is 2D. Otherwise, the output is a scalar.
    total_weight:
        shape and dtype of output, should be same type as weight
    reduction: str
        default value is "mean"
    kernel_name : str
        kernel name, default value is "nll_loss"

    Returns
    -------
    None
    """
    _shape_and_dtype_check(x, target, weight, kernel_name)
    nll_loss_function = nll_loss_compute(x, target, weight,
                                         reduction, kernel_name)
    return nll_loss_function.nll_loss_compute_start()
