#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use
this file except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic unpack
"""
# pylint: disable=too-many-locals,too-few-public-methods,too-many-branches,unused-argument
# pylint: disable=too-many-arguments,too-many-instance-attributes
from enum import Enum
from enum import unique
import copy

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import buildcfg
from tbe.tvm.cce_build_module import build_fatbin
from tbe.tvm.build_module import build_config


class Constant:
    """
    The class for constant.
    """

    # When right_dim < MIN_RIGHT_DIM,Multi output go special tiling.
    MIN_RIGHT_DIM = 8


@unique
class TilingStrategy(Enum):
    """
    Enum tiling cases
    """
    SINGLE_OUTPUT = 0
    SMALL_SHAPE = 1
    BIG_SHAPE = 2
    LESS_32B = 3
    LAST_DIM_SMALL = 4


class CompileVar:
    """
    Compile var
    """

    def __init__(self, name, bound):
        self.tvm_var = tvm.var(name)
        self.name = name
        self.bound = bound

    def get_tvm_var(self):
        """
        get self.tvm_var
        """
        return self.tvm_var

    def get_name(self):
        """
        get self.name
        """
        return self.name

    def get_bound(self):
        """
        get self.bound
        """
        return self.bound


# pylint: disable=too-many-instance-attributes,too-few-public-methods
class Unpack:
    """
    Base Class for Unpack Op, includes Unpack op info.
    """

    # pylint: disable=too-many-arguments
    def __init__(self, input_x, output_y, num, axis, kernel_name):
        self.input_x = input_x
        self.output_num = num
        self.kernel_name = kernel_name
        self.dtype = input_x.get("dtype").lower()
        self.axis = axis

        self.dim_info_vars = []
        self.ub_tensor_list = []
        self.res_tensor_list = []
        self.gm2ub_tensor = None
        self.virtual_node = None
        self.single_out = None
        self.sch_list = []
        self.arg_list = []
        self.rules = []
        self.compile_vars = {}
        self.dim_vars = []
        self.dim_bounds = []
        self.output_shape = []
        self.x_reshape = None
        self.left_range = None
        self.right_range = None

        self._input_placeholder = None
        self.block_idx = None
        self.ub_size = None
        self.core_num = None
        self.dtype_size = None
        self.ele_per_block = None
        self.bound_upper = None
        self.axis_at_last_dim = False
        self.special_tiling = False
        self.new_axis = 1

        self._check_params(axis)
        self._init_params()
        self._trans_input_shape(axis)
        self._init_flag()

    def _check_params(self, axis):
        x_shape = list(self.input_x["shape"])
        if self.output_num is not None and x_shape[axis] != -1:
            if self.output_num != x_shape[axis]:
                error_manager_vector.raise_err_specific_reson(self.kernel_name,
                                                              "the num must be equal to x_shape[axis]")
        if self.output_num is None:
            self.output_num = x_shape[axis]
        if self.output_num == -1:
            error_manager_vector.raise_err_specific_reson(self.kernel_name,
                                                          "the number of outputs is unknown, do not support")
        # 1536B means stack holding the param provided to the platform,
        # 1 param takes 8 bytes, needs Multiple output param and 1 input param and 1 input dynamic_param
        # mini has more parameters (offset, index) than cloud
        compile_platform = tbe_platform.get_soc_spec("SOC_VERSION")
        if compile_platform in ("Ascend310",):
            max_num = (1536 // 3) // 8 - 2
        else:
            max_num = 1536 // 8 - 2
        if self.output_num > max_num:
            error_manager_vector.raise_err_input_param_not_in_range(self.kernel_name, 'num',
                                                                    1, max_num, self.output_num)

    def _init_params(self):
        """
        Init params info of unpack op
        """
        self.block_idx = tvm.thread_axis('blockIdx.x')
        self.ub_size = tbe_platform.get_soc_spec(tbe_platform.UB_SIZE)
        self.core_num = tbe_platform.get_soc_spec(tbe_platform.CORE_NUM)
        self.dtype_size = tbe_platform.get_bit_len(self.dtype) // 8
        one_block_bytes_size = tbe_platform.VECTOR_INST_BLOCK_WIDTH // tbe_platform.VECTOR_INST_BLOCK_NUM
        self.ele_per_block = one_block_bytes_size // self.dtype_size
        # each output requires 192B for reg buf
        self.bound_upper = (self.ub_size - self.output_num * 192) // self.dtype_size

    def _trans_input_shape(self, axis):
        """
        trans the input shape into three dimensions (left, mid, right) and
        get the range of left and right.
        """
        x_shape = list(self.input_x["shape"])
        x_range = list(self.input_x["range"])

        if len(x_shape) != len(x_range):
            error_manager_vector.raise_err_specific_reson(self.kernel_name, "input_x shape is invalid")

        real_axis = axis + len(x_shape) if axis < 0 else axis
        left_upper = 1
        right_upper = 1
        self.left_range = (1, self.core_num)
        self.axis_at_last_dim = bool(real_axis == len(x_shape) - 1)

        for idx in range(real_axis):
            if not x_range[idx][1]:
                left_upper = None
                break
            left_upper *= x_range[idx][1]

        for idx in range(real_axis + 1, len(x_shape)):
            if not x_range[idx][1]:
                right_upper = None
                break
            right_upper *= x_range[idx][1]

        left_dim_upper = (1, left_upper)
        right_dim_upper = (1, right_upper)

        self._set_dim_var("left_dim", left_dim_upper)
        self._set_dim_var("right_dim", right_dim_upper)

        if self.output_num == 1:
            self.x_reshape = (self.dim_vars[0] * self.dim_vars[1], )
            self.right_range = (1, left_upper * right_upper) if left_upper and right_upper else (1, None)
        else:
            self.x_reshape = (self.dim_vars[0], self.output_num, self.dim_vars[1])
            self.right_range = (1, right_upper)

    def _init_flag(self):
        if not self.right_range[1]:
            self.special_tiling = self.axis_at_last_dim and self.output_num > 1
        else:
            self.special_tiling = (
                self.axis_at_last_dim or self.right_range[1] < Constant.MIN_RIGHT_DIM) and self.output_num > 1

    def _set_dim_var(self, dim_name, dim_range):
        """
        Let dimension be represented by tvm.var.
        """
        dim_info_var = CompileVar(dim_name, dim_range)
        self.dim_info_vars.append(dim_info_var)
        self.dim_vars.append(dim_info_var.get_tvm_var())

    def _index_offset(self, offset, *index):
        """
        Compute the output offset in input_tensor
        """
        input_index = list(index)
        output_index = ()
        for idx, _ in enumerate(self.output_shape):
            if idx == self.new_axis:
                input_index[idx] = input_index[idx] + offset
            output_index += (input_index[idx],)
        return output_index

    # pylint: disable=unnecessary-lambda
    def _multi_output_common_compute(self):
        """
        Multi output compute function for common cases
        """
        offset = 0
        for i in range(self.output_num):
            tensor_ub = tvm.compute(self.output_shape,
                                    lambda *index: self._input_placeholder(*self._index_offset(offset, *index)),
                                    name="tensor" + str(i))
            self.ub_tensor_list.append(tensor_ub)

            res_tensor = tvm.compute(self.output_shape, lambda *index: tensor_ub(*index), name="res" + str(i))
            self.res_tensor_list.append(res_tensor)
            offset = offset + self.output_shape[self.new_axis]

        # create virtual node
        def _add_compute(*index):
            virtual_tensor = self.res_tensor_list[0](*index)
            for res_tensor in self.res_tensor_list[1:]:
                virtual_tensor += res_tensor(*index)
            return virtual_tensor

        self.virtual_node = tvm.compute(self.output_shape, lambda *index: _add_compute(*index), name="virtual_node")

    # pylint: disable=unnecessary-lambda
    def _multi_output_special_compute(self):
        """
        Multi output compute function for special cases
        """

        self.gm2ub_tensor = tvm.compute(self.x_reshape,
                                        lambda *index: self._input_placeholder(*index),
                                        name="gm2ub_tensor")
        offset = 0
        self.res_tensor_list = []
        for i in range(self.output_num):
            ub2ub_tensor = tvm.compute(self.output_shape,
                                       lambda *index: self.gm2ub_tensor(*self._index_offset(offset, *index)),
                                       name="tensor" + str(i))
            self.ub_tensor_list.append(ub2ub_tensor)
            res_tensor = tvm.compute(self.output_shape,
                                     lambda *index, tensor_in=ub2ub_tensor: tensor_in(*index),
                                     name="res" + str(i))
            self.res_tensor_list.append(res_tensor)
            offset = offset + self.output_shape[self.new_axis]

        # create virtual node
        def _add_compute(*index):
            virtual_tensor = self.res_tensor_list[0](*index)
            for ub2gm_tensor in self.res_tensor_list[1:]:
                virtual_tensor += ub2gm_tensor(*index)
            return virtual_tensor

        self.virtual_node = tvm.compute(self.output_shape, lambda *index: _add_compute(*index), name="virtual_node")

    # pylint: disable=unnecessary-lambda
    def _single_output_compute(self):
        """
        Single output compute function
        """
        tensor_ub = tvm.compute(self.output_shape, lambda *index: self._input_placeholder(*index), name="tensor_ub")
        self.ub_tensor_list.append(tensor_ub)
        self.single_out = tvm.compute(self.output_shape, lambda *index: tensor_ub(*index), name="res_tensor")
        self.res_tensor_list.append(self.single_out)

    def _compute(self):
        """
        Unpack compute function
        """
        self._input_placeholder = tvm.placeholder(self.x_reshape, dtype=self.dtype, name="input_x")

        if self.output_num == 1:
            self.output_shape = [self._input_placeholder.shape[0]]
            self._single_output_compute()
        else:
            self.output_shape = [self._input_placeholder.shape[0], 1, self._input_placeholder.shape[2]]
            if self.special_tiling:
                self._multi_output_special_compute()
            else:
                self._multi_output_common_compute()

    # pylint: disable=too-many-locals,too-many-branches
    def _multi_output_schedule(self, left_dim_out, right_dim_in, ub_tiling_axis, split_factor):
        """
        unpack schedule function for multi_output
        Parameters
        ----------
        left_dim_out: tvm.var
            the var identify spilt_factor for block_tiling on left_dim axis
        right_dim_in: tvm.var
            the var identify spilt_factor for block_tiling on right_dim axis
        ub_tiling_axis: int
            identify spilt axis for ub_tiling
        split_factor: tvm.var
            the var identify spilt_factor
        Returns
        ---------
        sch: tvm.schedule
            the compute schedule
        build_list: list
            include tvm.tensor of input and tvm.tensor of res
        """
        build_list = [self._input_placeholder]
        for res_tensor in self.res_tensor_list:
            build_list.append(res_tensor)

        sch = tvm.create_schedule(self.virtual_node.op)
        sch.sequential_malloc(tbe_platform.scope_ubuf)

        if self.special_tiling:
            sch[self.gm2ub_tensor].set_scope(tbe_platform.scope_ubuf)

        for tensor in self.ub_tensor_list:
            sch[tensor].set_scope(tbe_platform.scope_ubuf)

        tensor_ub = self.ub_tensor_list[0]
        for _, tensor in enumerate(self.ub_tensor_list[1:]):
            sch[tensor_ub].reused_by(tensor)
        if ub_tiling_axis == -1:
            left_dim_outer, left_dim_inner = sch[self.virtual_node].split(self.virtual_node.op.axis[0], nparts=1)
            axis_outer, axis_inner = sch[self.virtual_node].split(left_dim_inner, factor=1)
            for i in range(self.output_num):
                sch[self.ub_tensor_list[i]].compute_at(sch[self.virtual_node], axis_outer)
                sch[self.res_tensor_list[i]].compute_at(sch[self.virtual_node], axis_outer)
                sch[self.ub_tensor_list[i]].emit_insn(self.ub_tensor_list[i].op.axis[0], tbe_platform.DMA_COPY)
                sch[self.res_tensor_list[i]].emit_insn(self.res_tensor_list[i].op.axis[0], tbe_platform.DMA_COPY)
            sch[self.virtual_node].emit_insn(axis_inner, tbe_platform.PHONY_INSN)
            return sch, build_list
        if self.special_tiling:
            left_dim_outer, left_dim_inner = sch[self.virtual_node].split(self.virtual_node.op.axis[0],
                                                                          nparts=left_dim_out)
            sch[self.virtual_node].bind(left_dim_outer, self.block_idx)
            axis_outer, axis_inner = sch[self.virtual_node].split(left_dim_inner, factor=split_factor)
        else:
            left_dim_outer, left_dim_inner = sch[self.virtual_node].split(self.virtual_node.op.axis[0],
                                                                          nparts=left_dim_out)
            right_dim_outer, right_dim_inner = sch[self.virtual_node].split(self.virtual_node.op.axis[2],
                                                                            factor=right_dim_in)
            sch[self.virtual_node].reorder(left_dim_outer, right_dim_outer, left_dim_inner, right_dim_inner)
            fused_axis = sch[self.virtual_node].fuse(left_dim_outer, right_dim_outer)
            sch[self.virtual_node].bind(fused_axis, self.block_idx)

            if ub_tiling_axis == 0:
                axis_outer, axis_inner = sch[self.virtual_node].split(left_dim_inner, factor=split_factor)
            else:
                axis_outer, axis_inner = sch[self.virtual_node].split(right_dim_inner, factor=split_factor)

        ub_tensor_emit = tbe_platform.DMA_COPY
        is_need_align = True
        if self.special_tiling:
            sch[self.gm2ub_tensor].compute_at(sch[self.virtual_node], axis_outer)
            sch[self.gm2ub_tensor].emit_insn(self.gm2ub_tensor.op.axis[ub_tiling_axis], tbe_platform.DMA_COPY)
            ub_tensor_emit = tbe_platform.DATA_MOV
            is_need_align = False

        for i in range(self.output_num):
            if is_need_align:
                sch[self.ub_tensor_list[i]].set_buffer_size(self.bound_upper)
                sch[self.ub_tensor_list[i]].storage_align(self.ub_tensor_list[i].op.axis[0], self.ele_per_block, 0)
            sch[self.ub_tensor_list[i]].compute_at(sch[self.virtual_node], axis_outer)
            sch[self.res_tensor_list[i]].compute_at(sch[self.virtual_node], axis_outer)
            sch[self.ub_tensor_list[i]].emit_insn(self.ub_tensor_list[i].op.axis[ub_tiling_axis], ub_tensor_emit)
            sch[self.res_tensor_list[i]].emit_insn(self.res_tensor_list[i].op.axis[ub_tiling_axis],
                                                   tbe_platform.DMA_COPY)

        sch[self.virtual_node].emit_insn(axis_inner, tbe_platform.PHONY_INSN)

        return sch, build_list

    def _single_output_schedule(self, right_dim_in, split_factor):
        """
        unpack schedule function for one output
        Parameters
        ----------
        right_dim_in: tvm.var
            the var identify spilt_factor for block_tiling on right_dim axis
        split_factor: tvm.var
            the var identify spilt_factor
        Returns
        ---------
        sch: tvm.schedule
            the compute schedule
        build_list: list
            include tvm.tensor of input and tvm.tensor of res
        """
        build_list = [self._input_placeholder]
        for res_tensor in self.res_tensor_list:
            build_list.append(res_tensor)

        sch = tvm.create_schedule(self.single_out.op)
        sch.sequential_malloc(tbe_platform.scope_ubuf)

        for tensor in self.ub_tensor_list:
            sch[tensor].set_scope(tbe_platform.scope_ubuf)

        right_dim_outer, right_dim_inner = sch[self.single_out].split(self.single_out.op.axis[0], factor=right_dim_in)
        sch[self.single_out].bind(right_dim_outer, self.block_idx)

        axis_outer, axis_inner = sch[self.single_out].split(right_dim_inner, factor=split_factor)

        sch[self.ub_tensor_list[0]].compute_at(sch[self.single_out], axis_outer)
        sch[self.ub_tensor_list[0]].emit_insn(self.ub_tensor_list[0].op.axis[0], tbe_platform.DMA_COPY)
        sch[self.res_tensor_list[0]].emit_insn(axis_inner, tbe_platform.DMA_COPY)

        return sch, build_list

    def _unpack_schedule(self, right_dim_in, ub_tiling_axis, split_factor, left_dim_out):
        """
        unpack schedule function
        Parameters
        ----------
        right_dim_in: tvm.var
            the var identify spilt_factor for block_tiling on right_dim axis
        ub_tiling_axis: int
            identify spilt axis for ub_tiling
        split_factor: tvm.var
            the var identify spilt_factor
        Returns
        ---------
        sch: tvm.schedule
            the compute schedule
        build_list: list
            include tvm.tensor of input and tvm.tensor of res
        """
        if self.output_num == 1:
            sch, build_list = self._single_output_schedule(right_dim_in, split_factor)
        else:
            sch, build_list = self._multi_output_schedule(left_dim_out, right_dim_in, ub_tiling_axis, split_factor)

        return sch, build_list

    def _calc_tiling_case(self):
        """
        calc different tiling strategy
        """
        tiling_cases = []
        ub_ele_num = self.ub_size // self.dtype_size
        tiling_strategy = [
            TilingStrategy.SINGLE_OUTPUT, TilingStrategy.SMALL_SHAPE, TilingStrategy.BIG_SHAPE, TilingStrategy.LESS_32B,
            TilingStrategy.LAST_DIM_SMALL
        ]
        ub_factor_bound = (1, ub_ele_num)
        for _, key in enumerate(tiling_strategy):
            if self.output_num == 1 and key != TilingStrategy.SINGLE_OUTPUT:
                continue
            if key == TilingStrategy.BIG_SHAPE:
                ub_tiling_axis = 2
            elif key == TilingStrategy.LESS_32B:
                ub_tiling_axis = -1
            else:
                ub_tiling_axis = 0
            tiling_cases.append({"key": key, "ub_tiling_axis": ub_tiling_axis, "ub_factor_bound": ub_factor_bound})
        return tiling_cases

    def _build_unpack_cce(self):
        """
        Build cce
        """
        self._compute()
        tiling_cases = self._calc_tiling_case()
        for case in tiling_cases:
            if (case.get("key") == TilingStrategy.SINGLE_OUTPUT) and self.output_num != 1:
                continue
            if self.special_tiling != (case.get("key") == TilingStrategy.LAST_DIM_SMALL):
                continue
            tvm_vars = self.dim_info_vars.copy()
            left_dim_out = CompileVar("left_dim_out", self.left_range)
            tvm_vars.append(left_dim_out)
            right_dim_in = CompileVar("right_dim_in", self.right_range)
            tvm_vars.append(right_dim_in)
            split_factor = CompileVar("split_factor", case.get("ub_factor_bound"))
            tvm_vars.append(split_factor)

            var_list = [var.get_tvm_var() for var in tvm_vars]
            sch, tensor_list = self._unpack_schedule(right_dim_in.get_tvm_var(), case.get("ub_tiling_axis"),
                                                     split_factor.get_tvm_var(), left_dim_out.get_tvm_var())

            # set var bound
            for var in tvm_vars:
                sch.set_var_range(var.get_tvm_var(), *(var.get_bound()))

            self.sch_list.append(sch)
            self.arg_list.append(var_list + tensor_list)

            self.rules.append(case.get("key").value)
            self.compile_vars[case.get("key").value] = [var.get_name() for var in tvm_vars]

        build_config_items = {"parse_ddr_args": True, "build_fatbin": True}
        dynamic_config = buildcfg.default_buildcfg.dynamic_build_config_dict
        with buildcfg.build_config(**dynamic_config):
            upper_config = buildcfg.get_current_build_config("all")
        upper_config.update(build_config_items)

        build_configs = []
        for sch in self.sch_list:
            dynamic_single_sch_build_config = copy.deepcopy(upper_config)
            build_configs.append(build_config(**dynamic_single_sch_build_config))
        build_fatbin(build_configs, self.sch_list, self.arg_list, self.rules, self.kernel_name)

        # Add compile info
        tbe_context.get_context().add_compile_info(
            "compile_vars", {
                "core_num": self.core_num,
                "ub_size": self.ub_size,
                "output_num": self.output_num,
                "axis": self.axis,
                "is_special_tiling": self.special_tiling
            })
        tbe_context.get_context().add_compile_info("vars", self.compile_vars)

    def build_cce(self):
        """
        build_cce
        """
        self._build_unpack_cce()


@register_operator("Unpack")
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.DYNAMIC_OUTPUT, para_check.OPTION_ATTR_INT,
                            para_check.REQUIRED_ATTR_INT, para_check.KERNEL_NAME)
def unpack(input_x, output_y, num=None, axis=0, kernel_name="unpack"):
    """
    unpacks the given dimension of a rank R tensor into rank (R-1) tensors.

    Parameters
    ----------
    input_x : dict.
        shape, dtype and format of value to be unpacked.
    output_y: tuple or list
        the list of output tensor.
    num : int.
        the length of the dim axis, automatically inferred if None(default).
    axis: int.
        the axis to unpack along.
    kernel_name : str
        cce kernel name, default value is "unpack".

    Returns
    -------
    None
    """
    unpack_obj = Unpack(input_x, output_y, num, axis, kernel_name)
    unpack_obj.build_cce()
