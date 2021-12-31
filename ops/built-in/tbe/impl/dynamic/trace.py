"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

trace
"""

from impl.util.platform_adapter import tik
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context


# 'pylint:disable=too-few-public-methods,too-many-instance-attributes
class Constant:
    """
    The class for constant
    """
    TILING_ARG_NUM = 4
    MAX_SHAPE_SIZE = 2 ** 32 - 1
    TILING_MODE_1 = 1
    TILING_MODE_2 = 2


def get_bytes_len(dtype):
    """
    Parameters
    ----------
    dtype:  input dtype

    Returns
    -------
    dtype Btypes
    """

    index = 0
    for i in dtype:
        if i.isdigit():
            break
        index += 1
    return int(dtype[index:]) // 8


def ceil_value(value, factor):
    """
    if not divide exactly then plus 1

    Parameters
    ----------
    value:  input number
    factor: factor

    Returns
    -------
    ceil value
    """

    return (value + factor - 1) // factor


class TikTrace():
    """
    trace init
    """

    def __init__(self, input_data, kernel_name):
        self.tik_instance = tik.Tik()
        self.input_data_shape = input_data.get("shape")
        self.kernel_name = kernel_name

        self.ub_size_bytes = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.block_byte_size = 32

        self.input_dtype = input_data.get("dtype")
        self.input_dtype_bytes_size = get_bytes_len(self.input_dtype)
        self.data_each_block = (self.block_byte_size
            // self.input_dtype_bytes_size)
        self.input_x_gm = self.tik_instance.Tensor(self.input_dtype,
            (Constant.MAX_SHAPE_SIZE,), name="input_x_gm", scope=tik.scope_gm)
        self.output_y_gm = self.tik_instance.Tensor(self.input_dtype,
            (1,), name="output_y_gm", scope=tik.scope_gm)

        # tiling params
        self.tiling_dtype = "int64"
        self.tiling_block_num = ceil_value(get_bytes_len(self.tiling_dtype)
            * Constant.TILING_ARG_NUM, self.block_byte_size)
        self.tiling_gm = self.tik_instance.Tensor(self.tiling_dtype,
            (Constant.TILING_ARG_NUM,), name="tiling_gm", scope=tik.scope_gm)

        self.tiling_ub = None
        self.input_h = None
        self.input_w = None
        self.tiling_mode = None
        self.need_core_num = None
        self.metrix_rank = None
        self.aicore_num = None
        self.aicore_output_gm = None
        self.data_num_each_core = None
        self.aicore_comp_ub = None
        self.metrix_sum_ub = None
        self.aicore_proc_cnt = None

    def _get_tiling_args(self):
        """
        get runtime params from tiling data

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.tiling_ub = self.tik_instance.Tensor(self.tiling_dtype,
            (self.tiling_block_num * Constant.TILING_ARG_NUM,), name="tiling_ub",
            scope=tik.scope_ubuf)
        self.tik_instance.data_move(self.tiling_ub, self.tiling_gm, 0, 1,
            self.tiling_block_num, 0, 0)

        self.input_h = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="input_h")
        self.input_w = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="input_w")
        self.tiling_mode = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="tiling_mode")
        self.need_core_num = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="need_core_num")

        self.input_h.set_as(self.tiling_ub[0])
        self.input_w.set_as(self.tiling_ub[1])
        self.tiling_mode.set_as(self.tiling_ub[2])
        self.need_core_num.set_as(self.tiling_ub[3])

    def _init_process_args(self):
        """
        get process params from input

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self.metrix_rank = self.tik_instance.Scalar(dtype=self.tiling_dtype,
            name="metrix_rank")
        self.metrix_rank = min(self.input_h, self.input_w)

        self.aicore_num = self.tik_instance.Scalar(dtype="int32",
            name="aicore_num", init_value=self.need_core_num)

        # Define temporary gm space in multi-core processing
        self.aicore_output_gm = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.core_num, self.data_each_block),
            name="aicore_output_gm", scope=tik.scope_gm, is_workspace=True)
        self.data_num_each_core = self.tik_instance.Scalar(dtype="int32",
            name="data_num_each_core")
        self.data_num_each_core.set_as(ceil_value(self.metrix_rank,
            self.aicore_num))
        # The maximum number of blocks processed by aicore at a time
        self.aicore_proc_cnt = 4095

# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
    def trace_computer(self):
        """
        main process of trace dynamic shape

        Parameters
        ----------
        None

        Returns
        -------
        None
        """

        self._get_tiling_args()
        self._init_process_args()

        with self.tik_instance.for_range(0, self.core_num,
            block_num=self.core_num) as index:
            with self.tik_instance.if_scope(index < self.need_core_num):
                zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
                    init_value=0)
                # aicore_comp_ub is used to store data moved from gm to ubuf
                self.aicore_comp_ub = self.tik_instance.Tensor(self.input_dtype,
                    shape=(self.aicore_proc_cnt, self.data_each_block),
                    name="aicore_comp_ub", scope=tik.scope_ubuf)
                self.metrix_sum_ub = self.tik_instance.Tensor(self.input_dtype,
                    shape=(self.data_each_block, 1), name="metrix_sum_ub",
                    scope=tik.scope_ubuf)
                self.metrix_sum_ub[0, 0].set_as(zero_scalar)

                move_offset = index * self.data_num_each_core
                process_limit = self.tik_instance.Scalar(dtype="int32",
                    name="process_limit",
                    init_value=move_offset + self.data_num_each_core)
                with self.tik_instance.if_scope(process_limit <= self.metrix_rank):
                    self._trace_computer_each_core(index, move_offset, self.data_num_each_core)
                with self.tik_instance.else_scope():
                    tail_cnt = self.tik_instance.Scalar(
                        dtype="int32", name="tail_cnt",
                        init_value=self.metrix_rank - ((self.aicore_num - 1) * self.data_num_each_core))
                    self._trace_computer_each_core(index, move_offset, tail_cnt)

        self._trace_computer_all_core()
        tbe_context.get_context().add_compile_info("vars", {
            "core_num": self.core_num
        })
        self.tik_instance.BuildCCE(kernel_name=self.kernel_name,
            inputs=[self.input_x_gm], outputs=[self.output_y_gm],
            flowtable=[self.tiling_gm])
        return self.tik_instance

    def _trace_computer_each_core(self, index, move_offset, proc_cnt):
        """
        Calculate the matrix data in each ai core

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        origin_move_offset = move_offset
        loop_time = proc_cnt // self.aicore_proc_cnt
        with self.tik_instance.if_scope(loop_time > 0):
            with self.tik_instance.for_range(0, loop_time) as loop_index:
                move_offset += loop_index * self.aicore_proc_cnt
                self._trace_computer_each_matrix(move_offset,
                    self.aicore_proc_cnt)
            move_offset = origin_move_offset + loop_time * self.aicore_proc_cnt
        last_cnt = proc_cnt % self.aicore_proc_cnt
        with self.tik_instance.if_scope(last_cnt > 0):
            self._trace_computer_each_matrix(move_offset, last_cnt)

        self.tik_instance.data_move(self.aicore_output_gm[index, 0],
            self.metrix_sum_ub, 0, 1, 1, 0, 0)

    def _trace_tiling_proc_big(self, move_offset, proc_cnt):
        """
        When the mode is tiling_mode_2 or tiling_mode_3,
        Process the bigger inputs

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        comp_line_num = proc_cnt - 1
        with self.tik_instance.for_range(0, comp_line_num) as i:
            diag_idx = self.tik_instance.Scalar(dtype="int64", name="diag_idx",
                init_value=((i + move_offset) * (self.input_w + 1)))
            self.tik_instance.data_move(self.aicore_comp_ub[i, 0],
                self.input_x_gm[diag_idx], 0, 1, 1, 0, 0)

        last_data_idx = self.tik_instance.Scalar(
            dtype="int64", name="last_data_idx",
            init_value=(comp_line_num + move_offset) * (self.input_w + 1) - self.data_each_block + 1)
        self.tik_instance.data_move(
            self.aicore_comp_ub[comp_line_num, 0], self.input_x_gm[last_data_idx], 0, 1, 1, 0, 0)
        self.aicore_comp_ub[comp_line_num, 0].set_as(
            self.aicore_comp_ub[comp_line_num, self.data_each_block - 1])

    def _trace_tiling_proc_small(self, move_offset, proc_cnt):
        """
        When the mode is tiling_mode_1, Process the smaller inputs

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of matrix data
        """

        block_num = ceil_value(proc_cnt * self.input_w, self.data_each_block)
        tmp_ub = self.tik_instance.Tensor(self.input_dtype,
            shape=(block_num * self.data_each_block, 1), name="tmp_ub",
            scope=tik.scope_ubuf)
        burst_len = proc_cnt * self.input_w // self.data_each_block
        with self.tik_instance.if_scope(burst_len > 0):
            move_start_idx = self.tik_instance.Scalar(dtype="int64",
                name="move_start_idx", init_value=move_offset * self.input_w)
            self.tik_instance.data_move(tmp_ub,
                self.input_x_gm[move_start_idx], 0, 1, burst_len, 0, 0)
            last_num = proc_cnt * self.input_w % self.data_each_block

            with self.tik_instance.if_scope(last_num > 0):
                ub_base_offset = burst_len * self.data_each_block
                gm_base_offset = (move_offset * self.input_w
                    + burst_len * self.data_each_block)
                gm_back_offset = last_num - self.data_each_block
                gm_last_start_idx = gm_base_offset + gm_back_offset
                self.tik_instance.data_move(tmp_ub[ub_base_offset],
                    self.input_x_gm[gm_last_start_idx], 0, 1, 1, 0, 0)
                with self.tik_instance.for_range(0, last_num) as i:
                    tmp_ub[ub_base_offset + i, 0].set_as(
                        tmp_ub[ub_base_offset - gm_back_offset + i, 0])
        with self.tik_instance.else_scope():
            self.tik_instance.data_move(tmp_ub,
                self.input_x_gm[move_offset], 0, 1, 1, 0, 0)

        with self.tik_instance.for_range(0, proc_cnt) as i:
            diag_idx = self.tik_instance.Scalar(dtype="int64",
                name="diag_idx", init_value=move_offset +
                i * (self.input_w + 1))
            self.aicore_comp_ub[i, 0].set_as(tmp_ub[diag_idx])

    def _trace_computer_each_matrix(self, move_offset, proc_cnt):
        """
        Calculate the part matrix data in each ai core

        Parameters
        ----------
        move_offset: The beginning index of the input data
        proc_cnt: The count of the input data

        Returns
        -------
        Sum of part matrix data
        """

        work_tensor_ub = self.tik_instance.Tensor(self.input_dtype,
            shape=(proc_cnt,), name="work_tensor_ub", scope=tik.scope_ubuf)
        zero_scalar = self.tik_instance.Scalar(dtype=self.input_dtype,
            init_value=0)
        add_tensor_a = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.data_each_block, 1), name="add_tensor_a",
            scope=tik.scope_ubuf)
        add_tensor_b = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.data_each_block, 1), name="add_tensor_b",
            scope=tik.scope_ubuf)

        self.tik_instance.vec_dup(self.data_each_block, add_tensor_a,
            zero_scalar, 1, 0)
        self.tik_instance.vec_dup(self.data_each_block, add_tensor_b,
            zero_scalar, 1, 0)

        with self.tik_instance.if_scope(self.tiling_mode ==
            Constant.TILING_MODE_1):
            self._trace_tiling_proc_small(move_offset, proc_cnt)
        with self.tik_instance.if_scope(self.tiling_mode ==
            Constant.TILING_MODE_2):
            self._trace_tiling_proc_big(move_offset, proc_cnt)

        self.tik_instance.vec_reduce_add(1, add_tensor_a, self.aicore_comp_ub,
            work_tensor_ub, proc_cnt, 1)
        self.tik_instance.vec_add(1, add_tensor_a, add_tensor_a, add_tensor_b,
            1, 0, 0, 0)
        self.tik_instance.vec_add(1, self.metrix_sum_ub, add_tensor_a,
            self.metrix_sum_ub, 1, 0, 0, 0)

    def _trace_computer_all_core(self):
        """
        Sum the output of each core

        Parameters
        ----------
        None

        Returns
        -------
        sum data
        """

        aicore_all_input = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.aicore_num, self.data_each_block),
            name="aicore_all_input", scope=tik.scope_ubuf)
        self.tik_instance.data_move(aicore_all_input,
            self.aicore_output_gm, 0, 1, self.aicore_num, 0, 0)
        work_tensor_ub = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.aicore_num, self.data_each_block),
            name="work_tensor_ub", scope=tik.scope_ubuf)
        sum_tensor = self.tik_instance.Tensor(self.input_dtype,
            shape=(self.data_each_block, 1), name="sum_tensor",
            scope=tik.scope_ubuf)
        self.tik_instance.vec_reduce_add(1, sum_tensor, aicore_all_input,
            work_tensor_ub, self.aicore_num, 1)
        self.tik_instance.data_move(self.output_y_gm[0], sum_tensor,
            0, 1, 1, 0, 0)


# 'pylint: disable=locally-disabled,too-many-arguments,unused-argument
@register_operator("Trace")
@para_check.check_op_params(para_check.REQUIRED_INPUT,
                    para_check.REQUIRED_OUTPUT, para_check.KERNEL_NAME)
def trace(input_x, output_y, kernel_name="trace"):
    """
    Operation for trace.

    Parameters
    ----------
    input_data: 2D metrix of input, include shape and dtype, dtype support float16, float
    kernel_name: cce kernel name, default value is trace

    Returns
    -------
    tik_instance
    """

    trace_instance = TikTrace(input_x, kernel_name)
    tik_instance = trace_instance.trace_computer()
    return tik_instance
