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

ascend_quant
"""
from functools import reduce as function_reduce
import copy
import te.lang.cce
from te import tvm
from te import platform as cceconf
from te.platform.cce_conf import CceProductParams as pver
from te.platform import cce_emitinsn_params
from .elewise_schedule_new import ElewiseSchedule
from .util import dfs_tensor_graph

# define the tensor name
CAST_F16_NAME = "cast_f16_ub"
INPUT_NAME = "input_ub"
VMULS_REFORM_NAME = "reform_by_vmuls"
SQRT_NAME = "scale_sqrt_ub"
OFFSET_NAME = "offset_ub"
CAST_I8_NAME = "cast_i8_ub"
VADDS_REFORM_NAME = "reform_by_vadds"

# define the Maximum number of cores
MAXIMUM_CORE_NUM = 65535

# define the map of dtype size
DTYPE_SIZE_MAP = {"float16": 2,
                  "float32": 4}


def _tilling_axis(shape, dtype_size, tensor_num, res):
    """
    get the split axis and factor by ub size

    Parameters
    ----------
    shape: the shape of input
    dtype_size: the dtype size
    tensor_num: the number of tensor size

    Returns
    -------
    split_axis and split_factor
    """
    shape_new = list(shape).copy()
    total_size = (cceconf.get_soc_spec("UB_SIZE") - 1024) // dtype_size
    max_ub_count = total_size // tensor_num
    total_ele = max_ub_count // 2
    split_axis = 0
    split_factor = 1
    block_num = _get_block_num(res)
    val_cnt = 1
    index_cnt = 0

    for i in range(0, len(shape_new) - 1):
        val_cnt = val_cnt * shape_new[i]
        index_cnt = i
        if val_cnt >= block_num:
            break

    block_size = val_cnt // block_num * \
                 function_reduce(lambda x, y: x * y, shape_new[index_cnt + 1:])
    if 256 <= block_size <= total_ele:
        total_ele = block_size

    for index, _ in enumerate(shape_new):
        ele_cnt = function_reduce(lambda x, y: x * y, shape_new[index:])
        if ele_cnt <= total_ele:
            split_axis = index - 1
            split_factor = total_ele // ele_cnt
            break
    size = function_reduce(lambda x, y: x * y, shape_new[1:])
    if split_axis == 0 and size <= total_ele:
        split_axis = 0
        split_factor = 1
    if split_axis < 0:
        split_axis = 0
        split_factor = 1
    return split_axis, split_factor


def _round_emit_insn(round_mode):
    """
    Obtains the conv instruction by the round mode attr

    Parameters
    ----------
    round_mode: the attr of round mode

    Returns
    -------
    instruction
    """
    emit_insn_str = 'vector_conv_%s' % round_mode.value.lower()
    if pver().is_mini_version():
        # mini
        emit_insn_str = 'vector_conv'
    if round_mode == "Round":
        emit_insn_str = 'vector_conv'
    return emit_insn_str


def _reorder_by_split_c0(tensor):
    """
    reorder tensor by c1 axis

    Parameters
    ----------
    tensor: the tensor to be split

    Returns
    -------
    None
    """
    num = len(tensor.op.axis)
    factor = 16
    if num == 4:
        c0o, c0i = tensor.split(tensor.op.axis[3], factor)
        tensor.reorder(tensor.op.axis[0],
                       tensor.op.axis[1],
                       tensor.op.axis[2],
                       c0o,
                       c0i)
    else:
        c0o, c0i = tensor.split(tensor.op.axis[4], factor)
        tensor.reorder(tensor.op.axis[0],
                       tensor.op.axis[1],
                       tensor.op.axis[2],
                       tensor.op.axis[3],
                       c0o,
                       c0i)


def _reorder_by_split_c1(tensor):
    """
    reorder tensor by c0 axis

    Parameters
    ----------
    tensor: the tensor to be split

    Returns
    -------
    None
    """
    factor = 2
    c1o, c1i = tensor.split(tensor.op.axis[1], factor)
    tensor.reorder(tensor.op.axis[0],
                   c1o,
                   tensor.op.axis[2],
                   c1i,
                   tensor.op.axis[3])


def _set_buffer_scope(sch, tensor_map):
    """
    set the scope for tensors

    Parameters
    ----------
    sch: the schedule
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    for _, value in tensor_map.items():
        sch[value].set_scope(cceconf.scope_ubuf)


def _set_buffer_compute_at(sch, res, tensor_map, axis_outer):
    """
    set the compute axis for tensors

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors
    axis_outer: the axis to be set

    Returns
    -------
    None
    """
    for _, value in tensor_map.items():
        sch[value].compute_at(sch[res], axis_outer)


def _reorder_buffer(sch, res, tensor_map):
    """
    reorder all tensors to the same shape

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    None
    """
    for key, value in tensor_map.items():
        if key in [INPUT_NAME, CAST_F16_NAME]:
            _reorder_by_split_c1(sch[value])
        else:
            _reorder_by_split_c0(sch[value])
    _reorder_by_split_c0(sch[res])


def _set_buffer_emit_insn(sch, tensor_list, axis_inner, attr_dic):
    """
    instruction mapping

    Parameters
    ----------
    sch: the schedule
    tensor_list: the list of tensors
    axis_inner: the inner axis
    attr_dic: the dict of attr

    Returns
    -------
    None
    """
    res = tensor_list[0]
    tensor_map = tensor_list[1]
    round_emit_insn = _round_emit_insn(attr_dic.get("round_mode"))
    input_c1 = attr_dic.get("input_c1")
    if input_c1 % 2 == 0:
        in_dma = "dma_copy"
    else:
        in_dma = "dma_padding"
    if CAST_F16_NAME in tensor_map:
        sch[tensor_map.get(CAST_F16_NAME)].emit_insn(
            sch[tensor_map.get(CAST_F16_NAME)].op.axis[0], 'vector_conv')
    if OFFSET_NAME in tensor_map:
        sch[tensor_map.get(OFFSET_NAME)].emit_insn(
            sch[tensor_map.get(OFFSET_NAME)].op.axis[0], 'vector_adds')
    if SQRT_NAME in tensor_map:
        sch[tensor_map.get(SQRT_NAME)].emit_insn(
            sch[tensor_map.get(SQRT_NAME)].op.axis[0], 'vector_muls')
    if VMULS_REFORM_NAME in tensor_map:
        sch[tensor_map.get(VMULS_REFORM_NAME)].emit_insn(
            sch[tensor_map.get(VMULS_REFORM_NAME)].op.axis[0], 'vector_muls')
    if VADDS_REFORM_NAME in tensor_map:
        sch[tensor_map.get(VADDS_REFORM_NAME)].emit_insn(
            sch[tensor_map.get(VADDS_REFORM_NAME)].op.axis[0], 'vector_adds')
    sch[tensor_map.get(CAST_I8_NAME)].emit_insn(
        sch[tensor_map.get(CAST_I8_NAME)].op.axis[0], round_emit_insn)
    sch[tensor_map.get(INPUT_NAME)].emit_insn(
        sch[tensor_map.get(INPUT_NAME)].op.axis[0], in_dma)
    align_length = attr_dic.get("HWC0")
    l1_fusion_flag = attr_dic.get("l1_fusion_flag")
    if l1_fusion_flag != -1 and align_length > 0:
        sch[res].bind_buffer(res.op.axis[1], align_length, 0)
    sch[res].emit_insn(axis_inner, 'dma_copy')


def _get_fuse_info(sch, res, res_split_shape, split_info):
    """
    get the fuse info

    Parameters
    ----------
    sch: the schedule
    res: the placeholder of result
    res_split_shape: the output shape
    split_info: split_axis and split_factor

    Returns
    -------
    fused_value, fused_list, axis_outer_num
    """
    split_axis = split_info[0]
    split_factor = split_info[1]
    if res_split_shape[split_axis] % split_factor > 0:
        axis_outer_num = res_split_shape[split_axis] // split_factor + 1
    else:
        axis_outer_num = res_split_shape[split_axis] // split_factor
    origin_list = [res_split_shape[i] for i in range(split_axis)]
    fused_value = 1
    for _, item in enumerate(origin_list):
        fused_value *= item
    fused_list = [sch[res].op.axis[i] for i in range(split_axis)]
    return fused_value, fused_list, axis_outer_num


def _bind_fuse(fused_value, fused_list, axis_outer_num, sch, res,
               axis_outer, res_split_shape):
    """
    bind the fused axis.
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")
    bind_axis = axis_outer
    if fused_list:
        if fused_value * axis_outer_num <= core_num:
            fused_list.append(axis_outer)
            bind_axis = sch[res].fuse(*fused_list)
            axis_outer = bind_axis
        elif fused_value < core_num:
            num = core_num // fused_value
            thread_outer, axis_outer = sch[res].split(axis_outer,
                                                      nparts=num)
            fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*fused_list)
        else:
            val_cnt = 1
            index = 0
            for i in range(len(fused_list)):
                val_cnt = val_cnt * res_split_shape[i]
                if val_cnt >= core_num:
                    index = i
                    break
            num = core_num // (val_cnt // res_split_shape[index])
            thread_outer, _ = sch[res].split(res.op.axis[index], nparts=num)
            new_fused_list = fused_list[:index]
            new_fused_list.append(thread_outer)
            bind_axis = sch[res].fuse(*new_fused_list)
    sch[res].bind(bind_axis, tvm.thread_axis("blockIdx.x"))
    return axis_outer


def _bind_core(out_shape, sch, res, tensor_map):
    """
    bind multi-core

    Parameters
    ----------
    out_shape: the output shape
    sch: the schedule
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    axis_outer, axis_inner
    """
    res_split_shape = (out_shape[0],
                       out_shape[1],
                       out_shape[2],
                       2,
                       out_shape[3] // 2)
    core_num = _get_block_num(res)
    split_axis, split_factor = _tilling_axis(
        res_split_shape,
        DTYPE_SIZE_MAP.get(tensor_map.get(INPUT_NAME).dtype.lower()),
        4, res)
    axis_outer, axis_inner = sch[res].split(res.op.axis[split_axis],
                                            factor=split_factor)
    bind_axis = 0
    can_bind = False
    for i in range(split_axis):
        if res_split_shape[i] >= core_num:
            bind_axis = i
            can_bind = True
            break
    fused_value, fused_list, axis_outer_num = _get_fuse_info(
        sch, res, res_split_shape, (split_axis, split_factor))

    if can_bind:
        thread_outer, _ = sch[res].split(res.op.axis[bind_axis],
                                         nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    elif axis_outer_num >= core_num:
        thread_outer, axis_outer = sch[res].split(axis_outer,
                                                  nparts=core_num)
        sch[res].bind(thread_outer, tvm.thread_axis("blockIdx.x"))
    else:
        axis_outer = _bind_fuse(fused_value, fused_list, axis_outer_num, sch,
                                res, axis_outer, res_split_shape)
    sch[tensor_map.get(INPUT_NAME)].double_buffer()
    return axis_outer, axis_inner


def _get_tensor_map(res, tensor_map):
    """
    get the compute tensors

    Parameters
    ----------
    res: the placeholder of result
    tensor_map: the compute tensors

    Returns
    -------
    is_fuse_flag
    """
    is_fuse_flag = False
    if res is None:
        return False
    stack = [res]
    visited_list = []
    while stack:
        cur_tensor = stack.pop()
        visited_list.append(cur_tensor)
        for in_tensor in cur_tensor.op.input_tensors:
            if in_tensor not in visited_list:
                stack.append(in_tensor)
                tensor_map[in_tensor.name] = in_tensor
                tag = in_tensor.op.tag
                if tag and tag.find("elewise") != -1:
                    is_fuse_flag = True
                    break
    return is_fuse_flag


def _get_block_num(res):
    """
    get the core number
    """
    core_num = cceconf.get_soc_spec("CORE_NUM")
    l1_fusion_flag = res.op.attrs['l1_fusion_flag'].value
    if l1_fusion_flag != -1:
        return 1
    return core_num


def ascend_quant_schedule(res, input_tensors):
    """
    the schedule processes of quant

    Parameters
    ----------
    res: the placeholder of result
    input_tensors: the placeholder of input

    Returns
    -------
    the result of schedule
    """
    out_shape = te.lang.cce.util.shape_to_list(res.shape)
    sch = tvm.create_schedule(res.op)
    tensor_map = {}
    is_fuse_flag = _get_tensor_map(res, tensor_map)
    if is_fuse_flag:
        schedule = QuantSchedule()
        schedule.do_schedule([res], [sch], [])
    else:
        attr_dic = {
            "scale": res.op.attrs['scale'],
            "sqrt_mode": res.op.attrs['sqrt_mode'],
            "offset": res.op.attrs['offset'],
            "round_mode": res.op.attrs['round_mode'],
            "input_c1": res.op.attrs['c1_dim'].value,
            "l1_fusion_flag": res.op.attrs['l1_fusion_flag'].value,
            'input_format': res.op.attrs['input_format'],
            'addr_type': res.op.attrs['addr_type'].value,
            'HWC0': res.op.attrs['HWC0'].value
        }
        l1_fusion_flag = attr_dic.get("l1_fusion_flag")
        if "input_x" in tensor_map:
            tensor = tensor_map.pop("input_x")
            attr_type = 0
            if tensor.op.attrs:
                if 'addr_type' in tensor.op.attrs:
                    attr_type = tensor.op.attrs["addr_type"].value
            if l1_fusion_flag != -1 and attr_type == 1:
                sch[tensor].set_scope(cceconf.scope_cbuf_fusion)
        out_addr_type = attr_dic.get("addr_type")
        if l1_fusion_flag != -1 and out_addr_type == 1:
            sch[res].set_scope(cceconf.scope_cbuf_fusion)
        _set_buffer_scope(sch, tensor_map)
        _reorder_buffer(sch, res, tensor_map)
        axis_outer, axis_inner = _bind_core(out_shape, sch, res, tensor_map)
        _set_buffer_compute_at(sch, res, tensor_map, axis_outer)
        _set_buffer_emit_insn(sch, (res, tensor_map), axis_inner, attr_dic)
    return sch


class QuantSchedule(ElewiseSchedule):
    """
    class of cce quant schedule

    Parameters
    ----------
    ElewiseSchedule: base class of elewise schedule

    Returns
    -------
    QuantSchedule_instance : instance of QuantSchedule
    """

    def __init__(self):
        ElewiseSchedule.__init__(self, True)
        self.attrs = {}

    def _get_res_attrs(self):
        """
        get the attrs carried by the tensor
        """
        self.attrs["scale"] = self._last_output_tensor.op.attrs['scale']
        self.attrs["sqrt_mode"] = self._last_output_tensor.op.attrs[
            'sqrt_mode']
        self.attrs["offset"] = self._last_output_tensor.op.attrs['offset']
        self.attrs["round_mode"] = self._last_output_tensor.op.attrs[
            'round_mode']
        self.attrs["input_format"] = self._last_output_tensor.op.attrs[
            'input_format']
        self.attrs["c1_dim"] = self._last_output_tensor.op.attrs[
            'c1_dim'].value
        self.attrs["addr_type"] = self._last_output_tensor.op.attrs[
            'addr_type']
        self.attrs["HWC0"] = self._last_output_tensor.op.attrs['HWC0']

    def _calculate_emit_insn_map(self, tensor):
        """
        Get the instruction map of tensor

        Parameters:
        ----------
        tensor: the tensor

        Returns
        -------
        Instruction map string
        """
        round_emit_insn = _round_emit_insn(self.attrs.get("round_mode"))
        if tensor.op.tag.find("|") != -1:
            str_list = tensor.op.tag.split("|")
            insn = self._insn_map.get(str_list[0])
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(str_list[0])
        else:
            insn = self._insn_map.get(tensor.op.tag)
            if insn and self._check_cast_support(tensor):
                return insn
            insn = self._reg_insn_map.get(tensor.op.tag)
            if insn is None:
                if tensor.op.tag == "quant":
                    insn = "dma_copy"
                elif tensor.op.name == "cast_i8_ub":
                    insn = round_emit_insn
                elif tensor.op.name == "input_ub" and \
                        self.attrs["c1_dim"] % 2 != 0:
                    insn = "dma_padding"
                else:
                    insn = "vector_auto"
        return insn

    def _calculate_cache_write(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._mid_tensors:
            if i.op.name == "input_ub" and self.attrs["c1_dim"] % 2 == 0:
                self._cache_write_exclude_tensors.append(i)
            if i not in self._cache_write_exclude_tensors:
                self._cache_write_tensors.append(i)

        if self._last_output_tensor not in self._cache_write_exclude_tensors:
            # in order to avoid last axis broadcast tensor as output tensor
            # do cache_write
            if self._last_output_tensor.op.tag == "elewise_binary_phony":
                self._elewise_binary_phony_as_output = True
                return

    def _do_cache_write(self):
        """
        cache write operations

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        for i in self._cache_write_tensors:
            self._cache_write_tensors_and_buffer_map[i] = i
            self._schedule[i].set_scope(cceconf.scope_ubuf)
        if self._out_tensors:
            visited, input_tensors, mid_tensors, tensor_map = dfs_tensor_graph(
                self._out_tensors[0])
            for tensor in self._out_tensors[1:]:
                dfs_tensor_graph(tensor, True, visited, input_tensors,
                                 mid_tensors, tensor_map)
            kernel_name = input_tensors[0].name.split("__")[-1]
            reuse_dict = cce_emitinsn_params.cceEmitParamsIns.get_param(
                "InputReuseDict_" + kernel_name)
            if reuse_dict is not None:
                for i, j in reuse_dict.items():
                    out_tensor = j
                    if j in self._temp_out_tensors:
                        out_tensor = self._temp_out_tensors[j]
                    self._schedule[i].reused_by(out_tensor)

    def do_schedule(self, out_tensors, sch_list, spec_node_list):
        """
        auto_schedule for cce AI-CORE

        Parameters
        ----------
        out_tensors : the out tvm.tensor

        sch : schedule, the computation schedule for the op

        spec_node_list : special node list

        Returns
        -------
        Bool, now is true

        """
        self._spec_node_list = spec_node_list
        self._out_tensors = copy.copy(out_tensors)

        if sch_list[0] is not None:
            self._schedule = sch_list[0]

        is_success = self._construct_compute_graph(out_tensors, [])
        if not is_success:
            return False

        self._get_res_attrs()

        self._calculate_cache_read()
        self._do_cache_read()

        self._calculate_cache_write()
        self._do_cache_write()

        self.reorder_buffer()

        self._calculate_tiling()
        self._do_tiling()

        self._do_buffer_tile()

        self._calculate_compute_inline()
        self._do_compute_inline()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._calculate_double_buffer()
        self._do_double_buffer()

        return self._schedule_valid

    def reorder_buffer(self):
        """
        reorder reform tensors
        """
        split_c0_list = ["reform_by_vmuls", "reform_by_vadds"]
        for i in self._cache_write_tensors:
            name = i.op.name
            if name is not None and name in split_c0_list:
                _reorder_by_split_c0(self._schedule[i])

    def _calculate_compute_inline(self):
        """
        Calculate the tensor that needs compute inline

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        self._compute_inline_tensors.extend(self._cache_write_exclude_tensors)
