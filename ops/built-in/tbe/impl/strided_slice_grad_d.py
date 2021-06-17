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
strided_slice_grad_d
"""
import math
import functools
import te.platform as tbe_platform
from te.utils import para_check
from te import tik
from te.utils.error_manager import error_manager_vector as error_manager
from impl import pad_d
from impl import copy_only
from impl.strided_slice_d import _init_parameter
from impl import common_util
from impl.util.util_tik_comm_func import ceil_align

# General limitation of the reduce size for input shape: 2**31
SHAPE_SIZE_LIMIT = 2147483648
BLOCK_SIZE = 32

VNCHW_BLOCK_SIZE = 512
VNCHW_ELEMENT_FP16 = VNCHW_BLOCK_SIZE // 2

TRANS_MIN_BLKS = 16


# pylint: disable=unused-argument
# pylint: disable=consider-using-in,unnecessary-pass
def check_supported(dy, output, shape, begin, end, strides, begin_mask=0,
                    end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                    kernel_name="strided_slice_grad_d"):
    """
    verify the types of cast supported by tbe
    """
    check_result = True, ""

    supported_new_axis_mask = {0, 2, 4}
    supported_shrink_axis_mask = {0, 1, 2, 4}

    if new_axis_mask not in supported_new_axis_mask:
        reason = "the new_axis_mask is not supported, new_axis_mask:%s, supported_new_axis_mask:%s" \
                 % (new_axis_mask, supported_new_axis_mask)
        check_result = False, reason

    if shrink_axis_mask not in supported_shrink_axis_mask:
        reason = "the shrink_axis_mask is not supported, shrink_axis_mask:%s, supported_shrink_axis_mask:%s" \
                 % (shrink_axis_mask, supported_shrink_axis_mask)
        check_result = False, reason

    if len(shape) == 0:
        reason = "should not be empty shape, shape:%s" % str(shape)
        check_result = False, reason

    return check_result


# pylint: disable=invalid-name, too-many-instance-attributes
# pylint: disable=too-many-arguments, useless-object-inheritance
# pylint: disable=too-many-locals, too-many-statements
# pylint: disable=attribute-defined-outside-init, unused-argument
# pylint: disable=attribute-defined-outside-init, chained-comparison
# pylint: disable=consider-using-in,protected-access
class StridedSliceGradLastDimCompute(object):
    """
    the compute for stridedslicegrad in last dim situation
    """
    def __init__(self, shape, begin, size, dtype, kernel_name):
        self.shape = shape
        self.dim_product = 1
        self.input_dim_last = 1
        self.output_dim_last = 1
        self.begin_last = 1
        self.dtype = dtype
        self.kernel_name = kernel_name
        self.tik_profiling = tik.Dprofile()
        self.ele_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        # align size for product dim, to make sure out data is 32B align
        self.product_dim_align_size = BLOCK_SIZE // self.ele_size

        # check only last dim to be sliced
        for i, (shape_i, begin_i, size_i) in \
                enumerate(zip(reversed(shape),
                              reversed(begin), reversed(size))):
            if i != 0:
                if shape_i != size_i:
                    self.check_result = False
                    return

                self.dim_product *= shape_i
            else:
                if begin_i < 0:
                    begin_i += shape_i
                self.input_dim_last = shape_i
                self.begin_last = begin_i
                self.output_dim_last = size_i

        # for moving data continuously, only small last dim is allowed
        # last dim data size <= 340B
        if self.input_dim_last * self.ele_size > 340:
            self.check_result = False
            return

        # for dividing cores easily, only big product dim is allowed
        # product dim >= aicore_num * 32 // ele_size
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        if self.dim_product < self.product_dim_align_size * aicore_num:
            self.check_result = False
            return

        self.check_result = True

    def check(self):
        """
        return check_result
        """
        return self.check_result

    def check_perf(self):
        """
        return if can enter performance template
        """
        is_satisfied_perf = (self.output_dim_last % self.product_dim_align_size == 0) and \
                            (self.input_dim_last % self.product_dim_align_size == 0) and \
                            (self.begin_last % self.product_dim_align_size == 0)
        return is_satisfied_perf

    def _get_block_tiling(self, product, core, block_idx):
        """
        get_block_tiling
        """
        task_size = self.product_dim_align_size
        if product % task_size == 0:
            tasks = product // task_size
        else:
            tasks = product // task_size + 1

        begin = self.tik_instance.Scalar(dtype="int64")
        size = self.tik_instance.Scalar(dtype="int64")
        if tasks % core == 0:
            begin.set_as(block_idx * (tasks // core) * task_size)
            size.set_as((tasks // core) * task_size)
        else:
            pack1 = tasks // core + 1
            pack2 = tasks // core
            with self.tik_instance.if_scope(block_idx >= tasks % core):
                begin.set_as(pack1 * block_idx * task_size - (block_idx - tasks % core) * task_size)
                size.set_as(pack2 * task_size)
            with self.tik_instance.else_scope():
                begin.set_as(pack1 * block_idx * task_size)
                size.set_as(pack1 * task_size)

        with self.tik_instance.if_scope(block_idx == (core - 1)):
            size.set_as(product - begin)
        return begin, size

    def can_do_with_vnchwconv(self):
        dtype_size = common_util.get_data_size(self.dtype)
        if dtype_size % 2 != 0:
            return False

        need_ub_length_of_float16 = ceil_align(self.input_dim_last * 16 * dtype_size, VNCHW_BLOCK_SIZE) * 3
        if need_ub_length_of_float16 < (self.tik_profiling.get_unified_buffer_size() // 2):
            return True
        return False

    def strided_slice_grad(self):
        """
        schedule for strided_slice_grad
        """
        if not self.check_result:
            error_manager.raise_err_specific_reson("strided_slice_grad_d",
                                                   "conditions of SliceLastDimCompute are not fulfilled")
        can_do_with_vnchwconv = self.can_do_with_vnchwconv()
        if can_do_with_vnchwconv:
            obj = StridedSliceGradLastDimWithVnchwConv(self.input_dim_last, self.output_dim_last, self.dim_product,
                                                       self.begin_last, self.dtype, self.kernel_name)
            return obj.strided_slice_grad_compute()

        return self.strided_slice_grad_scalar()

    def strided_slice_grad_scalar(self):
        """
        strided_slice_grad with scalar
        """
        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        ub_size = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.UB_SIZE)

        pad_value = tik_instance.Scalar(dtype=self.dtype, init_value=0)
        x = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.input_dim_last),
                                name="x", scope=tik.scope_gm)
        y = tik_instance.Tensor(self.dtype,
                                (self.dim_product, self.output_dim_last),
                                name="y", scope=tik.scope_gm)

        with tik_instance.for_range(0, aicore_num,
                                    block_num=aicore_num) as block_idx:
            dim_product_begin, dim_product_size = \
                self._get_block_tiling(self.dim_product, aicore_num, block_idx)
            max_dim_product = ub_size // self.ele_size \
                              // (self.input_dim_last + self.output_dim_last) \
                              // self.product_dim_align_size * self.product_dim_align_size
            loops = tik_instance.Scalar(dtype="int64")
            loops.set_as(dim_product_size // max_dim_product)
            with tik_instance.if_scope(dim_product_size % max_dim_product == 0):
                loops.set_as(loops - 1)

            with tik_instance.for_range(0, loops) as i:
                dim_product_begin_in_loop = i * max_dim_product
                dim_product_size_in_loop = max_dim_product

                x_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.input_dim_last), \
                                           name="x_ub", scope=tik.scope_ubuf)
                y_ub = tik_instance.Tensor(self.dtype,
                                           (max_dim_product,
                                            self.output_dim_last), \
                                           name="y_ub", scope=tik.scope_ubuf)

                output_size_in_loop = dim_product_size_in_loop \
                                      * self.output_dim_last * self.ele_size
                burst_length_out = output_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(y_ub,
                                       y[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.output_dim_last],
                                       0, 1, burst_length_out, 0, 0)

                with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                    idx_x = j * self.input_dim_last
                    idx_y = j * self.output_dim_last
                    for k in range(self.input_dim_last):
                        max_num = self.begin_last + self.output_dim_last
                        if (k >= self.begin_last) and (k < max_num):
                            x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                        else:
                            x_ub[idx_x + k] = pad_value

                input_size_in_loop = dim_product_size_in_loop \
                                     * self.input_dim_last * self.ele_size
                burst_length = input_size_in_loop // BLOCK_SIZE
                tik_instance.data_move(x[(dim_product_begin +
                                          dim_product_begin_in_loop)
                                         * self.input_dim_last],
                                       x_ub,
                                       0, 1, burst_length, 0, 0)

            # last loop
            i = loops
            dim_product_begin_in_loop = i * max_dim_product
            dim_product_size_in_loop = dim_product_size - dim_product_begin_in_loop

            x_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.input_dim_last), \
                                       name="x_ub", scope=tik.scope_ubuf)
            y_ub = tik_instance.Tensor(self.dtype, (max_dim_product, self.output_dim_last), \
                                       name="y_ub", scope=tik.scope_ubuf)

            output_size_in_loop = dim_product_size_in_loop * self.output_dim_last * self.ele_size
            burst_length_out = tik_instance.Scalar(dtype="int64")
            burst_length_out.set_as(output_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(output_size_in_loop % BLOCK_SIZE != 0):
                burst_length_out.set_as(burst_length_out + 1)
            tik_instance.data_move(y_ub,
                                   y[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.output_dim_last],
                                   0, 1, burst_length_out, 0, 0)

            with tik_instance.for_range(0, dim_product_size_in_loop) as j:
                idx_x = j * self.input_dim_last
                idx_y = j * self.output_dim_last
                for k in range(self.input_dim_last):
                    max_num = (self.begin_last + self.output_dim_last)
                    if (k >= self.begin_last) and (k < max_num):
                        x_ub[idx_x + k] = y_ub[idx_y + k - self.begin_last]
                    else:
                        x_ub[idx_x + k] = pad_value

            input_size_in_loop = dim_product_size_in_loop * self.input_dim_last * self.ele_size
            burst_length = tik_instance.Scalar(dtype="int64")
            burst_length.set_as(input_size_in_loop // BLOCK_SIZE)
            with tik_instance.if_scope(input_size_in_loop % BLOCK_SIZE != 0):
                burst_length.set_as(burst_length + 1)
            tik_instance.data_move(x[(dim_product_begin +
                                      dim_product_begin_in_loop)
                                     * self.input_dim_last],
                                   x_ub,
                                   0, 1, burst_length, 0, 0)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[y], outputs=[x])

    def strided_slice_grad_perf(self):
        """
        high performance schedule for strided_slice_grad
        self.input_dim_last, self.input_dim_last and self.begin_last should divided by block size
        """
        if not self.check_result:
            error_manager.raise_err_specific_reson("strided_slice_grad_d",
                                                   "conditions of SliceLastDimCompute are not fullfilled")

        tik_instance = tik.Tik()
        self.tik_instance = tik_instance
        aicore_num = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.CORE_NUM)
        ub_size = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE)

        self.pad_value = tik_instance.Scalar(dtype=self.dtype, init_value=0)
        self.x = tik_instance.Tensor(self.dtype,
                                     (self.dim_product, self.input_dim_last),
                                     name="x", scope=tik.scope_gm)
        self.y = tik_instance.Tensor(self.dtype,
                                     (self.dim_product, self.output_dim_last),
                                     name="y", scope=tik.scope_gm)

        self.max_dim_product = ub_size // self.ele_size \
            // self.input_dim_last \
            // self.product_dim_align_size * self.product_dim_align_size

        product_per_core = self.dim_product // aicore_num
        is_same_core = 0 if self.dim_product % aicore_num == 0 else 1
        product_last_core = product_per_core if is_same_core == 0 else self.dim_product % aicore_num
        aicore_num += is_same_core

        self.x_ub = self.tik_instance.Tensor(self.dtype, (self.max_dim_product, self.input_dim_last), \
                    name="x_ub", scope=tik.scope_ubuf)
        self.vector_mask = 256 // self.ele_size

        with self.tik_instance.for_range(0, aicore_num, block_num=aicore_num) as block_idx:
            with self.tik_instance.if_scope(block_idx != aicore_num - 1):
                self._compute_each_core(block_idx * product_per_core, product_per_core)
            with self.tik_instance.else_scope():
                self._compute_each_core(block_idx * product_per_core, product_last_core)

        tik_instance.BuildCCE(kernel_name=self.kernel_name, inputs=[self.y], outputs=[self.x])
        return tik_instance

    def _compute_each_core(self, move_product_offset, move_product_len):
        product_repeat = move_product_len // self.max_dim_product
        if product_repeat > 0:
            with self.tik_instance.for_range(0, product_repeat) as product_index:
                self.compute_each_loop(move_product_offset + product_index * \
                self.max_dim_product, self.max_dim_product)

        product_tail = move_product_len % self.max_dim_product
        if product_tail > 0:
            self.compute_each_loop(move_product_offset + product_repeat * self.max_dim_product, product_tail)

    def compute_each_loop(self, move_product_offset, move_product_len):
        """
        compute_each_loop
        """
        # vector dup 0 to x_ub
        repeat_loop = (move_product_len * self.input_dim_last) // (self.vector_mask * 255)
        if repeat_loop > 0:
            with self.tik_instance.for_range(0, repeat_loop) as repeat_index:
                self.tik_instance.vector_dup(self.vector_mask, self.x_ub[repeat_index * self.vector_mask * 255], \
                self.pad_value, 255, 1, 8)
        repeat_tail = (move_product_len * self.input_dim_last) % (self.vector_mask * 255) // self.vector_mask
        if repeat_tail > 0:
            self.tik_instance.vector_dup(self.vector_mask, self.x_ub[repeat_loop * self.vector_mask * 255], \
            self.pad_value, repeat_tail, 1, 8)
        mask_tail = (move_product_len * self.input_dim_last) % self.vector_mask
        if mask_tail > 0:
            self.tik_instance.vector_dup(mask_tail, self.x_ub[repeat_loop * self.vector_mask * 255 + \
            repeat_tail * self.vector_mask], self.pad_value, 1, 1, 8)

        # move y to x_ub and pad
        mv_stride = (self.input_dim_last - self.output_dim_last) // self.product_dim_align_size
        move_loop = move_product_len // 4095
        if move_loop > 0:
            with self.tik_instance.for_range(0, move_loop) as move_index:
                self.tik_instance.data_move(self.x_ub[self.begin_last + move_index * 4095 * self.input_dim_last], \
                self.y[(move_product_offset + move_index * 4095) * self.output_dim_last], 0, 4095, \
                self.output_dim_last // self.product_dim_align_size, 0, mv_stride)
        move_tail = move_product_len % 4095
        if move_tail > 0:
            self.tik_instance.data_move(self.x_ub[self.begin_last + move_loop * 4095 * self.input_dim_last], \
            self.y[(move_product_offset + move_loop * 4095) * self.output_dim_last], 0, \
            move_tail, self.output_dim_last // self.product_dim_align_size, 0, mv_stride)

        # move x_ub to x
        burst_len = (move_product_len * self.input_dim_last) // self.product_dim_align_size
        if burst_len > 65535:
            nburst = burst_len // 65535
            burst_len = 65535
        else:
            nburst = 1
        self.tik_instance.data_move(self.x[move_product_offset * self.input_dim_last], \
            self.x_ub, 0, nburst, burst_len, 0, 0)


class StridedSliceGradLastDimWithVnchwConv:
    """
    StridedSliceGrad LastDim With VnchwConv
    """
    def __init__(self, input_dim_last, output_dim_last, dim_product, begin_last, dtype, kernel_name):
        self.dtype = dtype
        self.dtype_size = common_util.get_data_size(self.dtype)
        self.inner_dtype = "float16"
        self.inner_bytes_size = common_util.get_data_size(self.inner_dtype)
        self.kernel_name = kernel_name
        self.tik_profiling = tik.Dprofile()
        self.tik_instance = tik.Tik()
        self.core_nums = self.tik_profiling.get_aicore_num()

        self.dim_product = dim_product
        self.input_dim_last = input_dim_last * self.dtype_size // self.inner_bytes_size
        self.output_dim_last = output_dim_last * self.dtype_size // self.inner_bytes_size
        self.begin_last = begin_last * self.dtype_size // self.inner_bytes_size

        self.block_num = BLOCK_SIZE // self.inner_bytes_size

        self.tiling_dtype = "int64"
        self.core_outer_num = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_num", init_value=0)
        self.core_outer_start = self.tik_instance.Scalar(self.tiling_dtype, "core_outer_start", init_value=0)
        self.core_inner_num = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_num", init_value=0)
        self.core_inner_start = self.tik_instance.Scalar(self.tiling_dtype, "core_inner_start", init_value=0)

        self.input_offset = []
        self.output_offset = []
        self.pad_scalar = 0

        self.ori_input_gm = self.tik_instance.Tensor(dtype,
                                                     (dim_product * output_dim_last,),
                                                     name="input_gm",
                                                     scope=tik.scope_gm)

        self.ori_output_gm = self.tik_instance.Tensor(dtype,
                                                      (dim_product * input_dim_last,),
                                                      name="output_gm",
                                                      scope=tik.scope_gm,
                                                      is_atomic_add=True)

        if self.dtype != self.inner_dtype:
            self.input_gm = self.ori_input_gm.reinterpret_cast_to(self.inner_dtype)
            self.output_gm = self.ori_output_gm.reinterpret_cast_to(self.inner_dtype)
        else:
            self.input_gm = self.ori_input_gm
            self.output_gm = self.ori_output_gm

    def core_scedule_args(self, core_index):
        """
        core_scedule_args
        """
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 0):
            core_outer_all = self.tiling_input_shape[-1]
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
            self.core_outer_num.set_as(self.core_outer_num * self.block_num)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(self.core_outer_start + self.core_outer_num > core_outer_all):
                self.core_outer_num.set_as(core_outer_all - self.core_outer_start)
                with self.tik_instance.if_scope(self.core_outer_num % self.block_num != 0):
                    self.core_outer_num.set_as((self.core_outer_num + self.block_num - 1) // self.block_num)
                    self.core_outer_num.set_as(self.core_outer_num * self.block_num)
                self.core_outer_start.set_as(core_outer_all - self.core_outer_num)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 1):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:5])
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + core_outer_all % self.core_nums)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 2):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:4])
            with self.tik_instance.if_scope(self.tiling_output_dim_5 * self.tiling_output_dim_4 < self.block_num):
                # the last two is less one block, only can process use one core
                self.core_outer_num.set_as(0)
                self.core_outer_start.set_as(0)
                with self.tik_instance.if_scope(core_index == 0):
                    self.core_outer_num.set_as(core_outer_all)
            with self.tik_instance.else_scope():
                self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
                self.core_outer_start.set_as(core_index * self.core_outer_num)
                with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                    with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                        self.core_outer_num.set_as(self.core_outer_num - 1)
                        self.core_outer_start.set_as(core_index * self.core_outer_num
                                                     + core_outer_all % self.core_nums)
        with self.tik_instance.if_scope(self.tiling_input_dim_cut_axis == 3):
            core_outer_all = functools.reduce(lambda x, y: x * y, self.tiling_input_shape[:3])
            self.core_outer_num.set_as((core_outer_all + self.core_nums - 1) // self.core_nums)
            self.core_outer_start.set_as(core_index * self.core_outer_num)
            with self.tik_instance.if_scope(core_outer_all % self.core_nums != 0):
                with self.tik_instance.if_scope(core_index >= core_outer_all % self.core_nums):
                    self.core_outer_num.set_as(self.core_outer_num - 1)
                    self.core_outer_start.set_as(core_index * self.core_outer_num + core_outer_all % self.core_nums)
        for i, _ in enumerate(self.tiling_input_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "input_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_input_shape[i:]))
            self.input_offset.append(scalar)
        for i, _ in enumerate(self.tiling_output_shape):
            scalar = self.tik_instance.Scalar(self.tiling_dtype, "output_offset_" + str(i), init_value=0)
            scalar.set_as(functools.reduce(lambda x, y: x * y, self.tiling_output_shape[i:]))
            self.output_offset.append(scalar)

    def tiling_args(self):
        """
        when input shape is less 6, will. expand to 6
        tiling info:
            tiling_key:
            tiling_input_dim_0
            tiling_input_dim_1
            tiling_input_dim_2
            tiling_input_dim_3
            tiling_input_dim_4
            tiling_input_dim_5
            tiling_pading_00
            tiling_pading_01
            tiling_pading_10
            tiling_pading_11
            tiling_pading_20
            tiling_pading_21
            tiling_pading_30
            tiling_pading_31
            tiling_pading_40
            tiling_pading_41
            tiling_pading_50
            tiling_pading_51
            tiling_input_dim_cut_axis: which dim will be cut
        """
        aicore_num = self.tik_profiling.get_aicore_num()
        for i in reversed(range(1, aicore_num + 1)):
            if self.dim_product % i == 0:
                multi_core_dim_value = i
                break

        self.tiling_input_dim_0 = 1
        self.tiling_input_dim_1 = 1
        self.tiling_input_dim_2 = 1
        self.tiling_input_dim_3 = multi_core_dim_value
        self.tiling_input_dim_4 = self.dim_product // multi_core_dim_value
        self.tiling_input_dim_5 = self.output_dim_last
        self.tiling_pading_00 = 0
        self.tiling_pading_01 = 0
        self.tiling_pading_10 = 0
        self.tiling_pading_11 = 0
        self.tiling_pading_20 = 0
        self.tiling_pading_21 = 0
        self.tiling_pading_30 = 0
        self.tiling_pading_31 = 0
        self.tiling_pading_40 = 0
        self.tiling_pading_41 = 0
        self.tiling_pading_50 = self.begin_last
        self.tiling_pading_51 = self.input_dim_last - self.output_dim_last - self.begin_last
        self.tiling_input_dim_cut_axis = 2

        self.tiling_input_shape = [self.tiling_input_dim_0, self.tiling_input_dim_1, self.tiling_input_dim_2,
                                   self.tiling_input_dim_3, self.tiling_input_dim_4, self.tiling_input_dim_5]
        self.tiling_output_shape = [0] * 6

        self.tiling_pading_value = [[self.tiling_pading_00, self.tiling_pading_01],
                                    [self.tiling_pading_10, self.tiling_pading_11],
                                    [self.tiling_pading_20, self.tiling_pading_21],
                                    [self.tiling_pading_30, self.tiling_pading_31],
                                    [self.tiling_pading_40, self.tiling_pading_41],
                                    [self.tiling_pading_50, self.tiling_pading_51]]

        # calcu output_dim
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = self.tiling_input_shape[i]
            pad_left = self.tiling_pading_value[i][0]
            pad_right = self.tiling_pading_value[i][1]
            self.tiling_output_shape[i] = input_dims + pad_left + pad_right

        self.tiling_output_dim_0, self.tiling_output_dim_1, \
            self.tiling_output_dim_2, self.tiling_output_dim_3, \
            self.tiling_output_dim_4, self.tiling_output_dim_5 = self.tiling_output_shape

    def get_output_outer_idx(self, in_idx, outer_num=5):
        """
        get_output_outer_idx use in_idx
        """
        input_dim_0 = in_idx // self.input_offset[1]
        input_dim_1 = (in_idx % self.input_offset[1]) // self.input_offset[2]
        input_dim_2 = (in_idx % self.input_offset[2]) // self.input_offset[3]
        input_dim_3 = (in_idx % self.input_offset[3]) // self.input_offset[4]
        input_dim_4 = (in_idx % self.input_offset[4]) // self.input_offset[5]
        input_dim_5 = in_idx % self.input_offset[5]

        input_list = [input_dim_0, input_dim_1, input_dim_2, input_dim_3, input_dim_4, input_dim_5]
        output_list = []
        for i, _ in enumerate(self.tiling_input_shape):
            input_dims = input_list[i]
            pad_left = self.tiling_pading_value[i][0]
            output_dims = input_dims + pad_left
            output_list.append(output_dims)

        output_idx = 0
        for i in range(outer_num):
            output_idx = output_idx + output_list[i] * self.output_offset[i + 1]
        return output_idx

    def data_move(self, gm_src_info, gm_dst_info, copy_len, used_ub):
        """
        func for data_move
        """
        input_gm, input_offset = gm_src_info
        output_gm, output_offset = gm_dst_info
        bursn_len = (copy_len + self.block_num - 1) // self.block_num
        self.tik_instance.data_move(used_ub,
                                    input_gm[input_offset],
                                    0, 1, bursn_len, 0, 0)
        self.tik_instance.data_move(output_gm[output_offset],
                                    used_ub,
                                    0, 1, bursn_len, 0, 0)

    def strided_slice_grad_compute(self):
        """
        strided_slice_grad compute
        """
        core_nums = self.tik_profiling.get_aicore_num()
        with self.tik_instance.for_range(0, core_nums, block_num=core_nums) as core_index:
            self.tiling_args()
            self.core_scedule_args(core_index)
            with self.tik_instance.new_stmt_scope():
                self._strided_slice_grad_vnchwconv()

        opt_config = {"out_of_bound_sync_check": True,
                      "enable_const_fold": True
                      }
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.ori_input_gm],
                                   outputs=self.ori_output_gm,
                                   config=opt_config
                                   )

    def _strided_slice_grad_vnchwconv(self):
        """
        strided_slice_grad with vnchwconv
        """
        max_line_in_ub = 16
        max_output_size = 480 * 2
        second_dim_input_num = self.tiling_input_shape[-2]
        third_dim_input_num = self.tiling_input_shape[-1]
        third_dim_output_num = self.tiling_output_shape[-1]

        first_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="first_dim_cut_num")
        second_dim_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_cut_num")

        second_dim_cut_num.set_as(max_output_size // third_dim_output_num)
        with self.tik_instance.if_scope(second_dim_cut_num > second_dim_input_num):
            second_dim_cut_num.set_as(second_dim_input_num)

        first_dim_cut_num.set_as(max_line_in_ub * second_dim_cut_num)

        # cut inner first dim and second dim info
        second_dim_total_loop_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_num")
        second_dim_total_loop_tail = self.tik_instance.Scalar(dtype="int64", name="second_dim_total_loop_tail")
        second_dim_total_loop_num.set_as(second_dim_input_num // second_dim_cut_num)
        second_dim_total_loop_tail.set_as(second_dim_input_num % second_dim_cut_num)

        second_dim_outer_cut_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_cut_num")
        second_dim_outer_cut_num.set_as(max_line_in_ub)
        with self.tik_instance.if_scope(second_dim_total_loop_num < max_line_in_ub):
            second_dim_outer_cut_num.set_as(second_dim_total_loop_num)

        second_dim_outer_loop_num_ceil = \
            (second_dim_total_loop_num + second_dim_outer_cut_num - 1) // second_dim_outer_cut_num
        second_dim_outer_loop_num_floor = second_dim_total_loop_num // second_dim_outer_cut_num

        second_dim_outer_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                               name="second_dim_outer_sigment_ub",
                                                               scope=tik.scope_ubuf)
        second_dim_outer_sigment_ub[0].set_as(second_dim_outer_cut_num)
        second_dim_outer_sigment_ub[1].set_as(second_dim_total_loop_num % second_dim_outer_cut_num)

        second_dim_sigment_ub = self.tik_instance.Tensor("int64", (4,),
                                                         name="second_dim_sigment_ub",
                                                         scope=tik.scope_ubuf)
        second_dim_sigment_ub[0].set_as(second_dim_cut_num)
        second_dim_sigment_ub[1].set_as(second_dim_input_num % second_dim_cut_num)

        loop_align_tail = self.tik_instance.Scalar(dtype="int64", name="loop_align_tail")
        tail_align_tail = self.tik_instance.Scalar(dtype="int64", name="tail_align_tail")
        loop_align_tail.set_as((second_dim_cut_num * third_dim_output_num) % self.block_num)
        tail_align_tail.set_as((second_dim_total_loop_tail * third_dim_output_num) % self.block_num)
        with self.tik_instance.if_scope(self.tiling_output_shape[-1] * self.tiling_output_shape[-2] <= self.block_num):
            loop_align_tail.set_as(0)
            tail_align_tail.set_as(0)

        vnchw_src_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride0", init_value=1)
        vnchw_dst_stride0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride0", init_value=16)
        vnchw_src_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_src_stride1", init_value=16)
        vnchw_dst_stride1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_dst_stride1", init_value=1)
        vnchw_repeat0 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat0", init_value=1)
        vnchw_repeat1 = self.tik_instance.Scalar(dtype="int32", name="vnchw_repeat1", init_value=1)
        vnchw_repeat0.set_as(((second_dim_cut_num * third_dim_input_num) + self.block_num - 1) // self.block_num)
        vnchw_repeat1.set_as(((second_dim_cut_num * third_dim_output_num) + self.block_num - 1) // self.block_num)
        with self.tik_instance.if_scope(vnchw_repeat0 == 1):
            vnchw_src_stride0.set_as(0)
            vnchw_dst_stride0.set_as(0)
        with self.tik_instance.if_scope(vnchw_repeat1 == 1):
            vnchw_src_stride1.set_as(0)
            vnchw_dst_stride1.set_as(0)

        def run_outer_by_outer(second_dim_start, do_inner_num, do_outer_num, align_tail):
            def _run_one_outer(_outer_num_idx, ub_list):
                origin_data_ub, vnchw_data_ub, vnchw_output_data_ub, _, _ = ub_list
                _, _, _, origin_output_data_ub, origin_output_tail_data_ub = ub_list
                input_outer_idx = _outer_num_idx + self.core_outer_start
                input_gm_offset = input_outer_idx * self.input_offset[4]
                output_outer_offset = self.tik_instance.Scalar(dtype="int64", name="output_outer_offset")
                output_outer_offset.set_as(self.get_output_outer_idx(input_gm_offset, 4))

                # step1. copy 16 dims in origin_data_ub
                with self.tik_instance.for_range(0, do_outer_num) as _copy_idx:
                    burst_len = ((do_inner_num * third_dim_input_num) + self.block_num - 1) // self.block_num
                    src_offset = (second_dim_start + _copy_idx * do_inner_num) * third_dim_input_num
                    self.tik_instance.data_move(origin_data_ub[_copy_idx * max_output_size],
                                                self.input_gm[input_gm_offset + src_offset],
                                                0, 1, burst_len, 0, 0)
                # step2. vnchw 16 dims origin_data_ub to vnchw_data_ub
                origin_data_ub_list = [origin_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                vnchw_data_ub_list = [vnchw_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False,
                                            vnchw_data_ub_list,
                                            origin_data_ub_list,
                                            vnchw_repeat0,
                                            vnchw_dst_stride0, vnchw_src_stride0)

                pad_left = self.tiling_pading_value[-1][0]
                pad_right = self.tiling_pading_value[-1][1]
                # step3. rearange vnchw_data_ub to vnchw_output_data_ub
                # step3.0 copy input data to vnchw_output_data_ub with datamove
                burst_num = do_inner_num
                burst_len = third_dim_input_num
                src_offset = 0
                dst_offset = pad_left * self.block_num
                src_stride = 0
                dst_stride = pad_left + pad_right
                self.tik_instance.data_move(vnchw_output_data_ub[dst_offset],
                                            vnchw_data_ub[src_offset],
                                            0, burst_num, burst_len, src_stride, dst_stride)

                # step4. vnchw vnchw_output_data_ub to 16 dims origin_output_data_ub
                origin_output_data_ub_list = \
                    [origin_output_data_ub[i * max_output_size] for i in range(0, TRANS_MIN_BLKS)]
                vnchw_output_data_ub_list = \
                    [vnchw_output_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                self.tik_instance.vnchwconv(False, False,
                                            origin_output_data_ub_list,
                                            vnchw_output_data_ub_list,
                                            vnchw_repeat1,
                                            vnchw_dst_stride1, vnchw_src_stride1)

                # step5. copy 16 dims to output
                # step5.1 copy do_outer_num - 1 lines to output use ceil_div block
                with self.tik_instance.for_range(0, do_outer_num - 1) as _copy_idx:
                    burst_len = (do_inner_num * third_dim_output_num + self.block_num - 1) // self.block_num
                    dst_offset = \
                        output_outer_offset + \
                        (self.tiling_pading_value[4][0] + second_dim_start + _copy_idx * do_inner_num) \
                        * self.output_offset[5]
                    self.tik_instance.data_move(self.output_gm[dst_offset],
                                                origin_output_data_ub[_copy_idx * max_output_size],
                                                0, 1, burst_len, 0, 0)
                # step5.1 copy the last do_outer_num lines to output use floor_div block
                burst_len = (do_inner_num * third_dim_output_num) // self.block_num
                dst_offset = \
                    output_outer_offset + \
                    (self.tiling_pading_value[4][0] + second_dim_start + (do_outer_num - 1) * do_inner_num) \
                    * self.output_offset[5]
                self.tik_instance.data_move(self.output_gm[dst_offset],
                                            origin_output_data_ub[(do_outer_num - 1) * max_output_size],
                                            0, 1, burst_len, 0, 0)

                # step6. process tail for the last line
                with self.tik_instance.if_scope(align_tail != 0):
                    origin_output_data_ub_list = \
                        [origin_output_tail_data_ub[i * 16] for i in range(0, TRANS_MIN_BLKS)]
                    vnchw_output_data_ub_list = \
                        [vnchw_output_data_ub[i * 16 + (do_inner_num * third_dim_output_num - 16) * 16]
                         for i in range(0, TRANS_MIN_BLKS)]
                    self.tik_instance.vnchwconv(False, False,
                                                origin_output_data_ub_list,
                                                vnchw_output_data_ub_list,
                                                1, 0, 0)
                    burst_len = 1
                    dst_offset = \
                        output_outer_offset \
                        + (self.tiling_pading_value[4][0] + second_dim_start + do_outer_num * do_inner_num) \
                        * self.output_offset[5] \
                        - self.block_num
                    self.tik_instance.data_move(self.output_gm[dst_offset],
                                                origin_output_tail_data_ub[(do_outer_num - 1)* 16],
                                                0, 1, burst_len, 0, 0)

            origin_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_ping", scope=tik.scope_ubuf)
            vnchw_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

            vnchw_output_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (max_line_in_ub * max_output_size,),
                                                                 name="vnchw_output_data_ub_ping",
                                                                 scope=tik.scope_ubuf)
            origin_output_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype,
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_ping = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_ping, self.pad_scalar,
                                         max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)

            origin_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                           name="origin_data_ub_pang", scope=tik.scope_ubuf)
            vnchw_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (max_line_in_ub * max_output_size,),
                                                          name="vnchw_data_ub_ping", scope=tik.scope_ubuf)

            vnchw_output_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype,
                                                                 (max_line_in_ub * max_output_size,),
                                                                 name="vnchw_output_data_ub_ping",
                                                                 scope=tik.scope_ubuf)
            origin_output_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype,
                                                                  (max_line_in_ub * max_output_size,),
                                                                  name="origin_output_data_ub_ping",
                                                                  scope=tik.scope_ubuf)
            origin_output_tail_data_ub_pang = self.tik_instance.Tensor(self.inner_dtype, (16 * 16,),
                                                                       name="origin_output_tail_data_ub_ping",
                                                                       scope=tik.scope_ubuf)
            self.tik_instance.vector_dup(self.block_num * 8, vnchw_output_data_ub_pang, self.pad_scalar,
                                         max_line_in_ub * max_output_size // self.block_num // 8, 1, 8)
            ping_ub_list = [origin_data_ub_ping, vnchw_data_ub_ping, vnchw_output_data_ub_ping,
                            origin_output_data_ub_ping, origin_output_tail_data_ub_ping]
            pang_ub_list = [origin_data_ub_pang, vnchw_data_ub_pang, vnchw_output_data_ub_pang,
                            origin_output_data_ub_pang, origin_output_tail_data_ub_pang]
            with self.tik_instance.for_range(0, self.core_outer_num // 2) as _outer_idx:
                _run_one_outer(_outer_idx * 2, ping_ub_list)
                _run_one_outer(_outer_idx * 2 + 1, pang_ub_list)
            with self.tik_instance.if_scope(self.core_outer_num % 2 == 1):
                _run_one_outer(self.core_outer_num - 1, ping_ub_list)

        with self.tik_instance.for_range(0, second_dim_outer_loop_num_ceil) as second_dim_outer_idx:
            second_dim_outer_start = second_dim_outer_idx * second_dim_outer_cut_num * second_dim_cut_num
            second_dim_outer_process_num = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_process_num")
            second_dim_outer_process_num.set_as(
                second_dim_outer_sigment_ub[second_dim_outer_idx // second_dim_outer_loop_num_floor])
            run_outer_by_outer(second_dim_outer_start, second_dim_cut_num,
                               second_dim_outer_process_num, loop_align_tail)

        with self.tik_instance.if_scope(second_dim_total_loop_tail != 0):
            second_dim_outer_tail_start = self.tik_instance.Scalar(dtype="int64", name="second_dim_outer_tail_start")
            second_dim_outer_tail_start.set_as((second_dim_input_num // second_dim_cut_num) * second_dim_cut_num)
            with self.tik_instance.if_scope(second_dim_total_loop_tail * third_dim_output_num < self.block_num):
                new_tail_num = (self.block_num + third_dim_output_num - 1) // third_dim_output_num
                second_dim_outer_tail_start.set_as(
                    second_dim_outer_tail_start - new_tail_num + second_dim_total_loop_tail)
                second_dim_total_loop_tail.set_as(new_tail_num)

            run_outer_by_outer(second_dim_outer_tail_start, second_dim_total_loop_tail, 1, tail_align_tail)


def _update_begin_end(input_shape, begin, end, begin_mask, end_mask):
    """
    Calculate the value of padding by input parameters.

    Parameters
    ----------
    input_shape: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    begin_mask: int
        a bit mask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.

    Returns
    -------
    begin_shape: list.
        shape of 'begin' after mask handle
    end_shape: list.
        shape of 'end' after mask handle
    """
    begin_shape = list(begin)
    end_shape = list(end)

    if end_shape[-1] > input_shape[-1]:
        end_shape[-1] = input_shape[-1]

    # If the ith bit of begin_mask is set, begin[i] is ignored,
    # and the fullest possible range in that dimension is used instead.
    # end_mask works analogously, except with the end range.
    for i, _ in enumerate(zip(input_shape, begin_shape, end_shape)):
        # process begin_mask
        if (begin_mask & 2**i) == 2**i:
            begin_shape[i] = 0
        # process end_mask
        if (end_mask & 2**i) == 2**i:
            end_shape[i] = input_shape[i]

    return begin_shape, end_shape


def _get_paddings(shape_x, begin_shape, end_shape):
    """
    Calculate the value of padding by input parameters.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    begin_shape: list or tuple.
        represents the index of the first value to select.
    end_shape: list or tuple.
        represents the index of the last value to select.

    Returns
    -------
    paddings: list.
        indicates how many zeros to add after the contents of `shape_dy` in every dimension
    """
    paddings = []
    for begin_i, shape_x_i, end_i in zip(begin_shape, shape_x, end_shape):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        paddings.append([begin_i, shape_x_i - end_i])

    return paddings


def _check_shape_parameter(shape_x, shape_dy, begin, end, strides):
    """
    Check whether the input shape meets the requirements.

    Parameters
    ----------
    shape_x: list or tuple.
        shape of output.
    shape_dy: list or tuple.
        shape of input.
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.

    Returns
    -------
    None.
    """
    # length of 'shape_x, shape_dy, begin, end, strides' must be the same
    if not (len(end) == len(begin) and \
            len(shape_x) == len(begin) and \
            len(shape_x) == len(strides)):
        error_manager.raise_err_specific_reson("strided_slice_grad_d", "shape length mismatch!")

    # value of begin must less equal to end, and it's range is (0, shape_x_i).
    for i, (shape_x_i, begin_i, end_i) in enumerate(zip(shape_x, begin, end)):
        if begin_i < 0:
            begin_i += shape_x_i
        if end_i < 0:
            end_i += shape_x_i
        if not ((begin_i >= 0) and (end_i <= shape_x_i)
                and (begin_i <= end_i)):
            error_manager.raise_err_specific_reson("strided_slice_grad_d",
                                                    "Bound Over: begin[" + str(i) + "]:" + str(begin[i]) +
                                                    ", end[" + str(i) + "]:" + str(end[i]) + ", shape_x[" +
                                                    str(i) + "]:" + str(shape_x_i))

    # value of strides must all be 1.
    for i, strides_i in enumerate(strides):
        if strides_i != 1:
            error_manager.raise_err_input_value_invalid("strided_slice_grad_d", "strides[" + str(i) + "]",
                                                        "1", str(strides_i))


def _check_is_not_aligned_shape(shape, begin, ellipsis_mask, shrink_axis_mask):
    """
    Check whether the shape of begin and shape is not equal,
       and masks are not 0
    Parameters
    ----------
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th
        position is actually an ellipsis.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th
        specification should shrink the dimensionality.
    Returns
    -------
    bool result
    """
    is_check_pass = False
    if len(shape) > len(begin) and len(begin) == 2 and \
            ellipsis_mask == 1 and shrink_axis_mask == 2:
        is_check_pass = True

    return is_check_pass, begin


# pylint: disable=locally-disabled,too-many-arguments,too-many-locals

@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_LIST_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT, para_check.OPTION_ATTR_INT,
                            para_check.KERNEL_NAME)
def strided_slice_grad_d(dy, output, shape, begin, end, strides, begin_mask=0,
                         end_mask=0, ellipsis_mask=0, new_axis_mask=0, shrink_axis_mask=0,
                         kernel_name="strided_slice_grad_d"):

    """
    Since `StridedSlice` cuts out pieces of its `input` which is size`shape_dy`, its gradient
    will have the same shape (which is passed here as `shape_x`). The gradient will be zero in any
    element that the slice does not select.

    Parameters
    ----------
    dy : dict
        shape and dtype of input
    output_x : dict
        shape and dtype of out
    shape : list or tuple.
        shape of input
    begin: list or tuple.
        represents the index of the first value to select.
    end: list or tuple.
        represents the index of the last value to select.
    strides: list or tuple.
        step length to select.
    begin_mask: int
        a bitmask where a bit i being 1 means to ignore the begin value and instead use the
        largest interval possible.
    end_mask: int
        analogous to `begin_mask`.
    ellipsis_mask: int
        a bitmask where bit `i` being 1 means the `i`th position is actually an ellipsis.
    new_axis_mask: int
        a bitmask where bit `i` being 1 means the `i`th specification creates a
        new shape 1 dimension.
    shrink_axis_mask: int
        a bitmask where bit `i` implies that the `i`th specification should shrink
        the dimensionality.
    kernel_name : str
        cce kernel name, default value is "strided_slice_grad_d"

    Returns
    -------
    None.
    """
    shape_dy = dy.get("shape")
    ori_shape_dy = dy.get("ori_shape")
    format_dy = dy.get("format")
    ori_format_dy = dy.get("ori_format")
    input_dtype = dy.get("dtype").lower()

    para_check.check_dtype(input_dtype, ("float16", "float32", "int8", "uint8", "int32"), param_name="dy")
    para_check.check_shape(shape, param_name="shape")
    para_check.check_shape(shape_dy, param_name="dy")

    is_not_aligned, ori_begin = _check_is_not_aligned_shape(shape, begin,
                                                            ellipsis_mask,
                                                            shrink_axis_mask)

    shape = list(shape)
    begin = list(begin)
    end = list(end)
    strides = list(strides)
    shape, begin_shape, end_shape, stride_shape = _init_parameter(shape, begin, end,
                                                                  strides, begin_mask, end_mask,
                                                                  ellipsis_mask, new_axis_mask,
                                                                  shrink_axis_mask)
    shape_dy = list(map(lambda x, y, z: math.ceil((x - y) / (1 if z == 0 else z)),
                        end_shape, begin_shape, stride_shape))
    _check_shape_parameter(shape, shape_dy, begin_shape, end_shape, stride_shape)

    last_dim_compute = StridedSliceGradLastDimCompute(shape,
                                                      begin_shape,
                                                      shape_dy,
                                                      input_dtype, kernel_name)

    if last_dim_compute.check():
        if last_dim_compute.output_dim_last == last_dim_compute.input_dim_last:
            copy_only.copy_only(dy, dy, kernel_name)
            return
        if last_dim_compute.check_perf() and ellipsis_mask == 0:
            last_dim_compute.strided_slice_grad_perf()
        else:
            last_dim_compute.strided_slice_grad()
    elif is_not_aligned:
        shape_dy = list(shape_dy)
        shape_dy += [1]
        paddings = [[0, 0]] * (len(shape) - 1) + \
                   [[ori_begin[1], shape[-1] - ori_begin[1] - 1]]
        dy_dict = {"shape": shape_dy, "ori_shape": ori_shape_dy,
                   "format": format_dy, "ori_format": ori_format_dy,
                   "dtype": input_dtype}
        pad_d(dy_dict, dy_dict, paddings, kernel_name)
    else:
        paddings = _get_paddings(shape, begin_shape, end_shape)

        # Call the pad operator due to gradient of 'StridedSlice' is the same as 'pad'
        # when the strides is 1.
        # pad.pad_cce(shape_dy, paddings, dtype, "CONSTANT", pad_value, kernel_name, need_build,
        #            need_print)
        dy_dict = {"shape": shape_dy, "ori_shape": ori_shape_dy,
                   "format": format_dy, "ori_format": ori_format_dy, "dtype": input_dtype}
        pad_d(dy_dict, dy_dict, paddings, kernel_name)
