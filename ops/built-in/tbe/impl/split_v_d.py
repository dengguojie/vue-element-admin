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
split_v_d
"""

import functools
import te.lang.cce
from te import tik
from te import tvm
from te import platform as tbe_platform
from te.utils import para_check
from te.utils import shape_util
from te.platform.cce_build import build_config
from impl.split_last_dim import check_use_last_dim_branch
from impl.split_last_dim import split_last_dim
from impl.split_last_dim import SplitWith5HD
from impl.split_d import SplitMov
from impl.split_d import SplitLastDimVnv
from impl.util import util_common
from impl.util.util_select_op_base import gen_param
from impl.util.util_select_op_base import get_dynamic_param_in_json
from impl.copy_only import copy_only
from te.utils.error_manager import error_manager_vector
from impl.util.util_select_op_base import SplitInput
from impl.util.util_select_op_base import SplitOutput
from impl.util.util_select_op_base import get_op_cal_info

# vtranspose can deal 16*16
TRANSPOSE_SIZE = 256


# pylint: disable = unused-argument
def get_op_support_info(input_value,
                        output_data,
                        size_splits,
                        split_dim,
                        num_split,
                        kernel_name="split_v_d"):
    """
    get_op_support_info
    """
    shape_value_len = len(input_value.get("shape"))
    if split_dim < 0:
        split_dim += shape_value_len
    format_value = input_value.get("format").upper()
    if format_value == "ND" or format_value == "NC1HWC0" or format_value == "FRACTAL_NZ":
        axis_split_matrix=[]
        for i in range(0, shape_value_len-1):
            if i != split_dim:
                output_list = []
                for j in range(0, num_split):
                    output_0 = [j, [i]]
                    output_list.append(output_0)
                split_0 = [SplitInput([0, [i], [-1], [-1]]), SplitOutput(*output_list)]
                axis_split_matrix.append(split_0)
        axis_reduce_list = None
    else:
        axis_split_matrix = None
        axis_reduce_list = None
    op_cal_info_in_json = get_op_cal_info(axis_split_matrix, axis_reduce_list, 0, 0)
    return op_cal_info_in_json


def ceil(int_x, int_y):
    """
    get cel for int_x and int_y
    """
    if int_x == 0:
        return 1
    res = int_x // int_y
    if int_x % int_y == 0:
        return res

    return res + 1


# pylint: disable=locally-disabled,unused-argument,too-many-arguments,too-many-locals,too-many-return-statements
# pylint: disable=too-many-statements,too-many-branches,too-many-instance-attributes,too-many-boolean-expressions
class SplitNotEqual():
    """Function: use to finish SplitNotEqual main functions to reset data
    """

    def __init__(self, shape, dtype, split_dim, size_splits, kernel_name):
        """init SplitNotEqual parameters
        """
        self.shape = shape
        self.dtype = dtype
        self.split_dim = split_dim
        self.size_splits = size_splits
        self.kernel_name = kernel_name
        self.first_dim = self.shape[0]
        self.last_dim = self.shape[1]
        self.dtype_size = tbe_platform.cce_intrin.get_bit_len(self.dtype) // 8
        self.block_ele = 32 // self.dtype_size
        self.ub_ele = tbe_platform.cce_conf.get_soc_spec(tbe_platform.cce_conf.UB_SIZE) // self.dtype_size - \
                      self.block_ele
        self.core_num = tbe_platform.cce_conf.get_soc_spec(
            tbe_platform.cce_conf.CORE_NUM)
        self.tik_instance = tik.Tik()
        self.input_tensor = self.tik_instance.Tensor(self.dtype,
                                                     self.shape,
                                                     name="gm_in",
                                                     scope=tik.scope_gm)
        self.output_tensor_list = []
        for _, i in enumerate(range(len(self.size_splits))):
            output_tensor = self.tik_instance.Tensor(
                self.dtype, [self.first_dim, size_splits[i]],
                name="gm_out" + str(i),
                scope=tik.scope_gm)
            self.output_tensor_list.append(output_tensor)

    def process_align(self, core_ele, core_offset):
        """process_align function
        """
        max_ele = 8 * self.block_ele
        loop_num = core_ele // max_ele
        last_ele = core_ele - loop_num * max_ele

        def _run_idx(loop_ele, loop_idx):
            in_offset = core_offset * self.last_dim + loop_idx * max_ele * self.last_dim
            # copy from gm to ub
            data_ub = self.tik_instance.Tensor(self.dtype,
                                               [max_ele * self.last_dim],
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            in_burst = ceil(loop_ele * self.last_dim, self.block_ele)
            self.tik_instance.data_move(data_ub, self.input_tensor[in_offset],
                                        0, 1, in_burst, 0, 0)

            # scalar ub and copy from ub to gm
            out_ub = self.tik_instance.Tensor(self.dtype,
                                              [max_ele * self.last_dim],
                                              name="out_ub",
                                              scope=tik.scope_ubuf)
            ub_offset = 0
            for _, i in enumerate(range(len(self.size_splits))):
                if (ub_offset == 0 or ub_offset % self.block_ele == 0) and \
                        self.size_splits[i] % self.block_ele == 0:
                    src_stride = self.last_dim // self.block_ele - self.size_splits[
                        i] // self.block_ele
                    ub_burst = ceil(self.size_splits[i], self.block_ele)
                    self.tik_instance.data_move(out_ub, data_ub[ub_offset], 0,
                                                loop_ele, ub_burst, src_stride,
                                                0)
                else:
                    with self.tik_instance.for_range(0, loop_ele) as idx_1:
                        with self.tik_instance.for_range(
                                0, self.size_splits[i]) as idx_2:
                            out_ub[idx_1 * self.size_splits[i] + idx_2].set_as(
                                data_ub[ub_offset + idx_1 * self.last_dim +
                                        idx_2])

                out_offset = core_offset * self.size_splits[
                    i] + loop_idx * max_ele * self.size_splits[i]
                out_burst = ceil(loop_ele * self.size_splits[i], self.block_ele)
                self.tik_instance.data_move(
                    self.output_tensor_list[i][out_offset], out_ub, 0, 1,
                    out_burst, 0, 0)
                ub_offset = ub_offset + self.size_splits[i]

        with self.tik_instance.for_range(0, loop_num) as loop_idx:
            _run_idx(max_ele, loop_idx)
        if last_ele != 0:
            _run_idx(last_ele, loop_num)

    def process_not_align(self, core_ele, core_offset):
        """process_not_align function
        """
        max_ele = 16 * self.block_ele
        loop_num = core_ele // max_ele
        thread = 1
        if loop_num > 1:
            thread = 2
        with self.tik_instance.for_range(0, loop_num,
                                         thread_num=thread) as loop_idx:
            # define ub scope
            data_ub = self.tik_instance.Tensor(self.dtype,
                                               [max_ele * self.last_dim],
                                               name="data_ub",
                                               scope=tik.scope_ubuf)
            out_ub = self.tik_instance.Tensor(self.dtype,
                                              [max_ele * self.block_ele],
                                              name="out_ub",
                                              scope=tik.scope_ubuf)
            # mov for size_splits[0] ~ size_splits[-2]
            offset_1 = core_offset * self.last_dim + loop_idx * max_ele * self.last_dim
            with self.tik_instance.for_range(0, 16) as idx:
                offset_2 = offset_1 + idx * self.last_dim
                self.tik_instance.data_move(data_ub[idx * self.block_ele],
                                            self.input_tensor[offset_2], 0,
                                            max_ele // 16, 1, self.last_dim - 1,
                                            15)
            ub_offset = 0
            for _, i in enumerate(range(len(self.size_splits) - 1)):
                with self.tik_instance.for_range(0, max_ele) as idx_1:
                    with self.tik_instance.for_range(
                            0, self.size_splits[i]) as idx_2:
                        out_ub[idx_1 * self.size_splits[i] + idx_2].set_as(
                            data_ub[ub_offset + idx_1 * self.block_ele + idx_2])
                offset_3 = core_offset * self.size_splits[
                    i] + loop_idx * max_ele * self.size_splits[i]
                self.tik_instance.data_move(
                    self.output_tensor_list[i][offset_3], out_ub, 0, 1,
                    max_ele * self.size_splits[i] // self.block_ele, 0, 0)
                ub_offset = ub_offset + self.size_splits[i]

            # mov for size_splits[-1]
            offset_1 = core_offset * self.last_dim + loop_idx * max_ele * self.last_dim + ub_offset
            with self.tik_instance.for_range(0, 16) as idx:
                offset_2 = offset_1 + idx * self.last_dim
                self.tik_instance.data_move(
                    data_ub[idx * self.size_splits[-1]],
                    self.input_tensor[offset_2], 0, max_ele // 16,
                    self.size_splits[-1] // self.block_ele,
                    self.last_dim - self.size_splits[-1] // self.block_ele,
                    self.size_splits[-1] * 15 // self.block_ele)
            offset_3 = core_offset * self.size_splits[
                -1] + loop_idx * max_ele * self.size_splits[-1]
            self.tik_instance.data_move(
                self.output_tensor_list[-1][offset_3], data_ub, 0, 1,
                max_ele * self.size_splits[-1] // self.block_ele, 0, 0)
        if core_ele // max_ele != 0:
            with self.tik_instance.for_range(0, 1):
                # define ub scope
                data_ub = self.tik_instance.Tensor(self.dtype,
                                                   [max_ele * self.last_dim],
                                                   name="data_ub",
                                                   scope=tik.scope_ubuf)
                out_ub = self.tik_instance.Tensor(self.dtype,
                                                  [max_ele * self.block_ele],
                                                  name="out_ub",
                                                  scope=tik.scope_ubuf)
                # mov for size_splits[0] ~ size_splits[-2]
                offset_1 = (self.first_dim - max_ele) * self.last_dim
                with self.tik_instance.for_range(0, 16) as idx:
                    offset_2 = offset_1 + idx * self.last_dim
                    self.tik_instance.data_move(data_ub[idx * self.block_ele],
                                                self.input_tensor[offset_2], 0,
                                                max_ele // 16, 1,
                                                self.last_dim - 1, 15)
                ub_offset = 0
                for _, i in enumerate(range(len(self.size_splits) - 1)):
                    with self.tik_instance.for_range(0, max_ele) as idx_1:
                        with self.tik_instance.for_range(
                                0, self.size_splits[i]) as idx_2:
                            out_ub[idx_1 * self.size_splits[i] + idx_2].set_as(
                                data_ub[ub_offset + idx_1 * self.block_ele +
                                        idx_2])
                    offset_3 = (self.first_dim - max_ele) * self.size_splits[i]
                    self.tik_instance.data_move(
                        self.output_tensor_list[i][offset_3], out_ub, 0, 1,
                        max_ele * self.size_splits[i] // self.block_ele, 0, 0)
                    ub_offset = ub_offset + self.size_splits[i]

                # mov for size_splits[-1]
                offset_1 = (self.first_dim -
                            max_ele) * self.last_dim + ub_offset
                with self.tik_instance.for_range(0, 16) as idx:
                    offset_2 = offset_1 + idx * self.last_dim
                    self.tik_instance.data_move(
                        data_ub[idx * self.size_splits[-1]],
                        self.input_tensor[offset_2], 0, max_ele // 16,
                        self.size_splits[-1] // self.block_ele,
                        self.last_dim - self.size_splits[-1] // self.block_ele,
                        self.size_splits[-1] * 15 // self.block_ele)
                offset_3 = (self.first_dim - max_ele) * self.size_splits[-1]
                self.tik_instance.data_move(
                    self.output_tensor_list[-1][offset_3], data_ub, 0, 1,
                    max_ele * self.size_splits[-1] // self.block_ele, 0, 0)

    def run(self):
        """run function
        """
        if self.last_dim % self.block_ele == 0:
            unit = self.block_ele
            process_fuc = self.process_align
        else:
            unit = 16 * self.block_ele
            process_fuc = self.process_not_align
        block_len = ceil(self.first_dim, unit)
        core_len = ceil(block_len, self.core_num)

        core_ele = core_len * unit
        core_used = self.first_dim // core_ele
        if self.first_dim % core_ele != 0:
            core_used = core_used + 1
        last_core_ele = self.first_dim - (core_used - 1) * core_ele
        with self.tik_instance.for_range(0, core_used,
                                         block_num=core_used) as core_index:
            core_offset = core_index * core_ele
            if last_core_ele != core_ele:
                with self.tik_instance.if_scope(core_index < (core_used - 1)):
                    process_fuc(core_ele, core_offset)

                with self.tik_instance.else_scope():
                    process_fuc(last_core_ele, core_offset)
            else:
                process_fuc(core_ele, core_offset)
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
                                   inputs=[self.input_tensor],
                                   outputs=self.output_tensor_list)
        return self.tik_instance

    def check_support(self):
        """check if input shape can select this branch
        """
        is_supported = True
        size = 0
        for _, i in enumerate(range(len(self.size_splits) - 1)):
            size = size + self.size_splits[i]
        if self.last_dim % self.block_ele == 0:
            if self.last_dim > 300:
                return False
            if (self.size_splits[0] % self.block_ele != 0 or
                self.size_splits[1] % self.block_ele != 0):
                return False
        else:
            if self.last_dim > 128:
                return False
            if size >= self.block_ele:
                return False
            if self.size_splits[-1] % self.block_ele != 0:
                return False
        return is_supported


def split_v_d_compute(input_value,
                      output_data,
                      size_splits,
                      split_dim,
                      num_split,
                      kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: TVM tensor
        input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    output_shape_list: list
        the list of output shapes.
    output_tensor_list: list
        the list of output tensors, output tensor type is TVM tensor.
    """
    output_shape_list, output_tensor_list = te.lang.cce.split_compute_com(
        input_value, split_dim, size_splits)
    return output_shape_list, output_tensor_list


def op_select_format(input_value,
                     output_data,
                     size_splits,
                     split_dim,
                     num_split,
                     kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    None.
    """
    dtype = input_value.get("dtype").lower()
    if dtype == "int8":
        c0_len = 32
    else:
        c0_len = 16
    output_org_shape_list = []
    output_org_format_list = []
    is_support_hd = True
    support_ori_format = \
        util_common.get_fused_format_str(["N", "D", "H", "W", "C"]) \
        + util_common.get_fused_format_str(["N", "H", "W", "C"])
    input_ori_shape = input_value.get("ori_shape")
    input_ori_shape = shape_util.scalar2tensor_one(input_ori_shape)
    input_ori_format = input_value.get("ori_format")
    split_dim = split_dim % len(input_ori_shape)

    for _, output_dict in enumerate(output_data):
        ori_format = output_dict.get("ori_format").upper()
        ori_shape = output_dict.get("ori_shape")
        ori_shape = shape_util.scalar2tensor_one(ori_shape)
        output_org_shape_list.append(ori_shape)
        output_org_format_list.append(ori_format)

        if ori_format not in support_ori_format or len(input_ori_shape) != len(input_ori_format) \
                or len(ori_format) != len(ori_shape):
            is_support_hd = False
            break

        # when split_d by N, H, W, support NDC1HWC0
        if ori_format[split_dim] != "C":
            break

        # when split_d by C, but output size not C0 align donot support NC1HWC0
        if ori_shape[split_dim] % c0_len != 0:
            is_support_hd = False
            break

    is_support_nz = False
    if input_ori_format[0] == "N" and split_dim == 0 and len(input_ori_shape) > 2:
        is_support_nz = True

    split_with_5hd_not_align = SplitWith5HD(input_value, output_data, split_dim,
                                            num_split, kernel_name)
    is_support_other_5hd = split_with_5hd_not_align.check_op_select()
    size_equal = len(set(size_splits))
    if size_equal != 1:
        is_support_other_5hd = False

    dtype_base = [
        "float16", "float", "int32", "int8", "int16", "int64", "uint8",
        "uint16", "uint32", "uint64"
    ]
    dtype_5hd = [
        "float16", "float", "int32", "int8", "int16", "uint16", "uint32"
    ]
    dtype_base_out = dtype_base.copy()
    format_base_out = ["ND"] * len(dtype_base)

    if is_support_hd:
        other_format = "NC1HWC0" if len(input_ori_shape) == 4 else "NDC1HWC0"
        dtype_base_out = dtype_base_out + dtype_5hd
        format_base_out = format_base_out + [other_format] * len(dtype_5hd)

    if is_support_nz:
        dtype_base_out = dtype_base_out + dtype_base
        format_base_out = format_base_out + ["FRACTAL_NZ"] * len(dtype_base)

    if is_support_other_5hd:
        dtype_base_out = dtype_base_out + ["float16", "int16", "uint16"]
        format_base_out = format_base_out + ["NC1HWC0"] * 3

    dtype_str = ','.join(dtype_base_out)
    format_str = ','.join(format_base_out)

    input0 = gen_param(classify="input0",
                       name="x",
                       datatype=dtype_str,
                       format=format_str)
    output0 = gen_param(classify="output0",
                        name="y",
                        datatype=dtype_str,
                        format=format_str)
    param_list = [input0, output0]
    param_dynamic_in_json = get_dynamic_param_in_json(param_list)

    return param_dynamic_in_json


@para_check.check_op_params(para_check.REQUIRED_INPUT,
                            para_check.DYNAMIC_OUTPUT,
                            para_check.REQUIRED_ATTR_LIST_INT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT,
                            para_check.KERNEL_NAME)
def split_v_d(input_value,
              output_data,
              size_splits,
              split_dim,
              num_split,
              kernel_name="split_v_d"):
    """Split a tensor into len(size_splits) tensors along one dimension.

    Parameters
    ----------
    input_value: dict
        the dict of input tensor.
    output_data: list or tuple
        the list of output tensor.
    size_splits: list or tuple
        a Python list containing the sizes of each output tensor
        along `split_dim`.
    split_dim: int
        the dimension along which to split_d.
    num_split: int
        used to specify the number of outputs.
    kernel_name: str
        cce kernel name, default value is "split_v_d".

    Returns
    -------
    None.
    """
    input_format = input_value.get("format")
    shape = input_value.get("shape")
    ori_format = input_value.get("ori_format")
    ori_shape = input_value.get("ori_shape")
    split_dim = util_common.update_axis_for_other_format(ori_shape, split_dim, input_format, ori_format)
    if input_format == "NC1HWC0":
        split_with_5hd_not_align = SplitWith5HD(input_value, output_data,
                                                split_dim, num_split,
                                                kernel_name)
        if split_with_5hd_not_align.check_5hd_vnchw():
            split_with_5hd_not_align.do_5hd_split_cut_by_batch()
            return
    if (split_dim == 1 and input_format == "NC1HWC0") or (split_dim == 2 and input_format == "NDC1HWC0"):
        size_splits = list(size_splits)
        size_splits = [size // 16 for size in size_splits]

    if input_format == "FRACTAL_NZ" and ori_format != "FRACTAL_NZ" and split_dim >= len(shape) - 4:
        size_splits = list(size_splits)
        size_splits = [size // 16 for size in size_splits]

    dtype = input_value.get("dtype")
    dtype_lower = dtype.lower()
    check_list = ("int8", "int16", "int32", "int64", "uint8", "uint16",
                  "uint32", "uint64", "float16", "float32")

    para_check.check_shape(shape, param_name="input_value")
    para_check.check_dtype(dtype_lower, check_list, param_name="input_value")

    shape_len = len(shape)
    split_dim = shape_util.axis_check(shape_len, split_dim)

    dim = shape[split_dim]
    if len(size_splits) + 1 == num_split or len(size_splits) == 0:
        spilt_list = []
        split_sum = 0
        if len(size_splits) != 0:
            for i, _ in enumerate(size_splits):
                spilt_list.append(size_splits[i])
                split_sum = split_sum + size_splits[i]
            if dim - split_sum > 0:
                spilt_list.append(dim - split_sum)
        else:
            batch = dim / num_split
            for i in range(0, num_split):
                spilt_list.append(int(batch))
        size_splits = spilt_list

    size_splits = list(size_splits)
    size_splits_sum = 0
    for size in size_splits:
        if size != -1:
            size_splits_sum += size
    if dim != size_splits_sum:
        for i, _ in enumerate(size_splits):
            if _ == -1:
                size_splits[i] = dim - size_splits_sum

    size_sum = 0
    for size in size_splits:
        if size < 1:
            expected_value = "The size of size_splits must be greater or equal to 1"
            real_value = "less to 1"
            error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits",
                                                           expected_value, real_value)
        size_sum = size_sum + size
    if size_sum != shape[split_dim]:
        expected_value = "The sum size of size_splits must be equal to the length of split_dim"
        real_value = "The sum size is (%d) and the length of split_dim is (%d)"\
                     % (size_sum, shape[split_dim])
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits",
                                                           expected_value, real_value)
    if len(size_splits) != num_split:
        expected_value = "The length of size_splits must be equal to num_split"
        real_value = "The length of size_splits is (%d) and the num_split is (%d)" \
                     % (len(size_splits), num_split)
        error_manager_vector.raise_err_input_value_invalid(kernel_name, "size_splits",
                                                           expected_value, real_value)

    if num_split == 1:
        copy_only(input_value, input_value, kernel_name)
        return

    split_mov = SplitMov(shape, dtype_lower, split_dim, num_split, size_splits,
                         kernel_name)
    new_shape = split_mov.input_shape
    new_split_dim = split_mov.split_dim
    new_size_splits = split_mov.size_splits
    new_output_shapes = split_mov.output_shapes
    input_size = functools.reduce(lambda x, y: x * y, new_shape)
    last_dim_same = True
    input_last_dim = new_output_shapes[0][-1]
    for i, _ in enumerate(new_output_shapes):
        if input_last_dim != new_output_shapes[i][-1]:
            last_dim_same = False
            break

    if dtype_lower == "float16" and new_split_dim == len(new_shape) - 1 and last_dim_same and \
            new_size_splits[0] == 1 and num_split <= 16 and input_size >= TRANSPOSE_SIZE * num_split:
        split_vnc = SplitLastDimVnv(new_shape, dtype_lower, new_output_shapes,
                                    new_split_dim, num_split, kernel_name)
        split_vnc.split_last_dim_vnc_compute()
        return

    if check_use_last_dim_branch(new_shape, dtype_lower, new_split_dim,
                                 num_split, new_size_splits):
        split_last_dim(new_shape, dtype_lower, new_split_dim, num_split,
                       new_size_splits, kernel_name)
        return

    if split_mov.check_whether_use_split_mov():
        split_mov.split_mov_compute()
        return

    if new_split_dim == 1:
        split_not_equal = SplitNotEqual(new_shape, dtype_lower, new_split_dim,
                                        new_size_splits, kernel_name)
        if split_not_equal.check_support():
            split_not_equal.run()
            return

    data = tvm.placeholder(shape, name="data", dtype=dtype_lower)
    output_shape_list, output_tensor_list = split_v_d_compute(
        data, output_data, size_splits, split_dim, num_split, kernel_name)

    sch, build_list = te.lang.cce.split_schedule_com(data, split_dim,
                                                     output_shape_list,
                                                     output_tensor_list)

    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name)
