#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

dynamic bn_training_reduce
"""
import copy
from enum import Enum
from enum import auto

from impl.util.platform_adapter import tvm
from impl.util.platform_adapter import tbe
from impl.util.platform_adapter import para_check
from impl.util.platform_adapter import shape_util
from impl.util.platform_adapter import op_tiling
from impl.util.platform_adapter import tbe_context
from impl.util.platform_adapter import OpPatternMode
from impl.util.platform_adapter import classify
from impl.util.platform_adapter import register_operator
from impl.util.platform_adapter import operation
from impl.util.platform_adapter import tbe_platform
from impl.util.platform_adapter import error_manager_vector
from impl.util.platform_adapter import tuple_sum


# 'pylint: disable=too-few-public-methods
class Constant:
    """
    The class for constant
    """
    CONST = "const"
    BN_REDUCE = "bn_reduce"
    VAR_BOUND_LIMIT = 2147483647
    BLOCK_SIZE_BYTE = 32

# <dsl insn, pass insn> mapping
    INSN_MAPPING = {
        "elewise_binary_mul": "vector_mul",
        "elewise_single_cast": "vector_conv",
        "elewise_empty_intrin": "phony_insn",
        "tuple_reduce_sum": "vector_reduce_sum",
        "reduce_sum": "vector_reduce_sum",
        "dma_copy": "dma_copy"
    }

    DTYPE_BYTE_MAPPING = {
        "uint1": 0.125,
        "bool": 1,
        "int8": 1,
        "uint8": 1,
        "float16": 2,
        "int16": 2,
        "uint16": 2,
        "float32": 4,
        "int32": 4,
        "uint32": 4,
        "int64": 8,
        "uint64": 8,
    }


@operation.register_schedule(pattern=Constant.BN_REDUCE)
def schedule(outs, tiling_case):
    '''
    :param outs:
    :param tiling_case:
    :return:
    '''
    [outs].clear()
    # Get Compute Graph Info
    graph_info = operation.get_context().get_current_compute().get("compute_graph_info")
    single_reduce_info = operation.get_context().get_current_compute().get("single_reduce_info")
    if tiling_case.is_customised:
        reduce_sch = BnReduceCustomisedSchedule(tiling_case, graph_info, single_reduce_info)
        real_schedule = reduce_sch.do_schedule(outs)
        real_schedule.tiling_key = tiling_case.tiling_key
    elif tiling_case.is_atomic:
        reduce_sch = BnReduceAtomicSchedule(tiling_case, graph_info, single_reduce_info)
        reduce_sch.init(outs)
        if single_reduce_info.is_reduce_all_axes():
            reduce_sch.reduce_case = 1
        elif single_reduce_info.is_reduce_not_last_axis():
            reduce_sch.reduce_case = 2
        else:
            reduce_sch.reduce_case = 3
        real_schedule = reduce_sch.do_schedule(outs)
        real_schedule.tiling_key = tiling_case.tiling_key
    else:
        raise NotImplementedError("Reduce schedule received invalid type: %s" % str(tiling_case.type))
    return real_schedule


@operation.register_tiling_case(pattern=Constant.BN_REDUCE)
def calc_tiling_case(outs, options=None):
    """
    bn training reduce tiling case interface
    :param outs:
    :param options:
    :return:
    """
    [options].clear()
    outs = list(outs) if isinstance(outs, (list, tuple)) else [outs]

    current_compute = operation.get_context().get_current_compute()

    # construct information of graph
    compute_graph_info = ComputeGraphInfo(outs)
    single_reduce_info = BNReduceInfo(compute_graph_info)
    current_compute.add("compute_graph_info", compute_graph_info)
    current_compute.add("single_reduce_info", single_reduce_info)
    if not compute_graph_info.reduce_tensor_set:
        error_detail = "Couldn't find reduce node for ReduceSchedule"
        error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

    tiling_instance = GenBnReduceTilingCase()
    tiling_case_list = []
    if single_reduce_info.reduce_axis_indices == [0, 2, 3]:
        tiling_case_list += tiling_instance.calculate_customised_tiling_cases(single_reduce_info)
    else:
        tiling_case_list += tiling_instance.calculate_atomic_tiling_cases(single_reduce_info)

    if operation.get_context().get("_mode") == Constant.CONST:
        if single_reduce_info.reduce_axis_indices == [0, 2, 3]:
            return tiling_instance.gen_const_tiling_case(single_reduce_info, compute_graph_info, True)

        return tiling_instance.gen_const_tiling_case(single_reduce_info, compute_graph_info)

    tiling_instance.apply_compile_info(single_reduce_info, compute_graph_info)
    # calc_tiling_key
    for tiling_case in tiling_case_list:
        tiling_instance.calc_tiling_key(single_reduce_info, tiling_case)

    return tiling_case_list


# 'pylint: disable=unused-argument,invalid-name
# 'pylint: disable=redefined-builtin
def bn_training_reduce_compute(x, sum, square_sum, axis, kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: TVM tensor
        contains x data
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    res: TVM tensor list
        the result of bn_training_reduce compute
    """
    dtype = x.dtype.lower()
    if dtype == "float16":
        x = tbe.cast_to(x, "float32")

    square_x = tbe.vmul(x, x)
    sum_x, square_sum_x = tuple_sum([x, square_x], axis, True)
    res = [sum_x, square_sum_x]

    return res


# 'pylint: disable=too-many-locals, too-many-statements
@register_operator("BNTrainingReduce", pattern=Constant.BN_REDUCE)
@para_check.check_op_params(para_check.REQUIRED_INPUT, para_check.REQUIRED_OUTPUT, para_check.REQUIRED_OUTPUT,
                            para_check.KERNEL_NAME)
def bn_training_reduce(x, sum, square_sum, kernel_name="bn_training_reduce"):
    """
    algorithm: part of fused_batch_norm_v2
    The first step of batch_norm
    which to calculate the sum and square sum of x.
    The major component of this operator is reduce operation.

    Parameters
    ----------
    x: dict
        dict of input, A 5HD Tensor for input data.
    sum: dict
        dict of sum, A `Tensor`. Sum of x.
    square_sum: dict
        dict of square_sum, A `Tensor`. Square sum of x.
    kernel_name: str
        kernel name, default value is "bn_training_reduce"

    Returns
    -------
    None
    """

    def generate_reduce_input(inputs_before_reduce):
        """
        obtain the shape and range to classify
        """

        def _process_all_unknown_shape(shape_list, range_list):
            """
            process input include shape -2
            """
            all_unknown_shape_len = 8
            for single_shape in shape_list:
                if tuple(single_shape) != (-2,):
                    all_unknown_shape_len = len(single_shape)
                    break

            for idx, single_shape in enumerate(shape_list):
                if tuple(single_shape) == (-2,):
                    shape_list[idx] = [-1] * all_unknown_shape_len
                    range_list[idx] = [(0, None)] * all_unknown_shape_len

        def _get_dim(i):
            return max((s[i] for s in shape_local))

        def _select_min_upper_bound(input_list):
            min_ele = Constant.VAR_BOUND_LIMIT + 1
            for ele in input_list:
                if ele is None:
                    continue
                if ele < min_ele:
                    min_ele = ele
            return min_ele if min_ele != Constant.VAR_BOUND_LIMIT + 1 else None

        def _get_range(i):
            if shape_out[i] != -1:
                return shape_out[i], shape_out[i]

            return max((r[i][0] for r in range_local)), _select_min_upper_bound((r[i][1] for r in range_local))

        shape_local = [x["shape"] for x in inputs_before_reduce]
        range_local = [
            x.get("range") if x.get("range") else [(1, None)] * len(shape_local[0]) for x in inputs_before_reduce
        ]

        _process_all_unknown_shape(shape_local, range_local)

        shape_out = [_get_dim(i) for i in range(len(shape_local[0]))]

        range_out = [_get_range(i) for i in range(len(range_local[0]))]

        for index, _ in enumerate(shape_out):
            if range_out[index][0] == range_out[index][1]:
                shape_out[index] = range_out[index][0]

        return {"shape": shape_out, "range": range_out}

    def simply_input(ins):
        """
        :param ins:
        :return:
        """
        for (x, axis) in ins:
            if x['mode'] == Constant.CONST and axis['value'][0] == 0:
                x['shape'].insert(0, 1)
                axis['value'] = [x + 1 for x in axis['value']]
        return ins

    data_format = x.get("format").upper()
    origin_format = x.get("ori_format").upper()
    dtype = x.get("dtype").lower()

    ori_reduce_axis = [0, 2, 3]
    x['rel_pos_to_reduce'] = 'before'
    tbe_context.get_context().add_compile_info('ori_axis', ori_reduce_axis)

    # check and format
    check_list = ("NC1HWC0",)
    para_check.check_format(data_format, check_list, param_name="x")
    if data_format == "NCHW" and origin_format not in ("NCHW",):
        error_detail = "The origin format only supports NCHW when format is NCHW, origin_format:", origin_format
        error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

    if data_format == "NC1HWC0" and x['shape'][-1] < 0:
        error_detail = "The input x dim C0 should be a const value. x['shape'][-1]：", x['shape'][-1]
        error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

    # check dtype
    check_list = ("float16", "float32")
    para_check.check_dtype(dtype, check_list, param_name="x")

    schedules = []
    tensors = []

    dup_x = copy.deepcopy(x)
    normalised_x = generate_reduce_input([x])
    dup_x['shape'] = normalised_x['shape']
    dup_x['range'] = normalised_x['range']
    is_const = all(x > 0 for x in dup_x['shape'])
    original = "original"
    if is_const:
        dup_x['mode'] = Constant.CONST
        tbe_context.get_context().add_compile_info("ori_const_shape", dup_x['shape'])
    else:
        dup_x['mode'] = original
        dup_x['shape'][0] = -1
        dup_x['range'][0] = (1, None)

    shape_len = len(x['shape'])

    axis_checked = shape_util.axis_check(shape_len, ori_reduce_axis)
    input_axis = {'shape': [len(axis_checked), ], 'value': axis_checked, 'rel_pos_to_reduce': 'axis'}
    ins = classify([x, input_axis], OpPatternMode.REDUCE, {'keepdims': True})
    ins = simply_input(ins)

    ins.insert(0, [dup_x, input_axis])

    for (_x, _axis) in ins:
        input_shape_len = len(_x['shape'])
        last_reduce_axis = _axis['value'][-1]

        if data_format == 'NC1HWC0' and last_reduce_axis == (input_shape_len - 1):
            continue

        # compute
        with tbe.compute():
            shape_var_new = shape_util.variable_shape([_x, _axis], op_mode='reduce')[0]
            data_input = tvm.placeholder(shape_var_new, name="data_input", dtype=dtype)
            res = bn_training_reduce_compute(data_input, sum, square_sum, _axis.get('value'), kernel_name=kernel_name)
            tensor_list = [data_input] + list(res)
            tensors.append(tensor_list)

        # schedule
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
            schedules.append(sch)
            if _x['mode'] == Constant.CONST and sch:
                break

    # build
    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.build(schedules, config)


# 'pylint: disable=too-many-instance-attributes
class ComputeGraphInfo:
    """
    Operator Compute Graph Info collector and container
    """

    def __init__(self, output_tensors):
        """
        Initialize containers and try to collect info
        """
        # Basic Info
        self.output_tensor_set = None
        self.tensor_consumers_map = None
        self.tensor_producers_map = None
        self.tensor_list = None
        # Extra info initialized by hooks
        self.reduce_tensor_set = set()
        self.input_tensor_set = set()
        self.non_gm_input_tensor_set = set()
        # Extra info initialized after pre-initialization
        self.mid_output_tensor_set = set()
        self.mid_tensor_set = set()
        self.endpoint_output_tensor_set = set()
        self.max_single_tensor_ub_size = None
        self.tensors_before_reduce = []
        self.tensors_after_reduce = []

        # Do info collection
        self._collect_info(output_tensors)
        self._init_max_ub_count(output_tensors)

    def _collect_info(self, output_tensors):
        """
        :param output_tensors:
        :return:
        """
        self.output_tensor_set = set(output_tensors)
        self.tensor_list, self.tensor_consumers_map, self.tensor_producers_map = \
            self.dfs_compute_graph(self.output_tensor_set)
        # Initialize non-hookable info
        self.gen_mid_tensor_sets()
        # endpoint_output_tensor_set
        self.gen_endpoint_output_tensor_set()

    def gen_endpoint_output_tensor_set(self):
        """
        :return:
        """
        for output_tensor in self.output_tensor_set:
            if not self.tensor_consumers_map[output_tensor]:
                self.endpoint_output_tensor_set.add(output_tensor)

    def gen_mid_tensor_sets(self):
        """
        :return:
        """
        # mid_output_tensor_set
        # mid_tensor_set
        for tensor in self.tensor_list:
            if tensor in self.output_tensor_set and self.tensor_consumers_map[tensor]:
                # Tensor in output and has consumers is middle_out_tensor
                self.mid_output_tensor_set.add(tensor)
                self.mid_tensor_set.add(tensor)
            elif tensor not in self.output_tensor_set | self.input_tensor_set | self.non_gm_input_tensor_set:
                self.mid_tensor_set.add(tensor)

    def dfs_compute_graph(self, root_tensor):
        """
        :param root_tensor:
        :param hooks:
        :return:
        """

        def _recursive_func(_root_tensor, _visited_list, _tensor_consumers_map, _tensor_producers_map):
            """
            :param _root_tensor:
            :param _visited_list:
            :param _tensor_consumers_map:
            :param _tensor_producers_map:
            :return:
            """
            _visited_list.add(_root_tensor)
            _tensor_producers_map.setdefault(_root_tensor, set())
            _tensor_consumers_map.setdefault(_root_tensor, set())

            if isinstance(_root_tensor.op, tvm.tensor.PlaceholderOp):
                self.input_tensor_set.add(_root_tensor)
            else:
                self.non_gm_input_tensor_set.add(_root_tensor)

            if _root_tensor.op.tag.find("reduce") != -1:
                self.reduce_tensor_set.add(_root_tensor)

            for in_tensor in _root_tensor.op.input_tensors:
                _tensor_consumers_map.setdefault(in_tensor, set())
                _tensor_consumers_map[in_tensor].add(_root_tensor)
                _tensor_producers_map[_root_tensor].add(in_tensor)
                _recursive_func(in_tensor, _visited_list, _tensor_consumers_map, _tensor_producers_map)

        visited_list = set()
        tensor_consumers_map = {}
        tensor_producers_map = {}
        if isinstance(root_tensor, (list, tuple, set)):
            for tensor in root_tensor:
                _recursive_func(tensor, visited_list, tensor_consumers_map, tensor_producers_map)
        elif isinstance(root_tensor, tvm.tensor.Tensor):
            _recursive_func(root_tensor, visited_list, tensor_consumers_map, tensor_producers_map)
        else:
            error_manager_vector.raise_err_input_format_invalid("bn_training_reduce", "dfs_compute_graph()", \
            ["list, tuple, Tensor"], str(type(root_tensor)))
        return list(visited_list), tensor_consumers_map, tensor_producers_map

    def get_all_tensors_before_reduce(self, output_tensors):
        """
        :param output_tensors:
        :return:
        """
        sum_x = output_tensors[0]
        square_sum_x = output_tensors[1]

        data_mul = square_sum_x.op.input_tensors[1]
        data = sum_x.op.input_tensors[0]

        cast_0 = None
        if isinstance(data.op, tvm.tensor.PlaceholderOp):
            cast_0 = None
        else:
            cast_0 = data
            data = cast_0.op.input_tensors[0]
        self.tensors_before_reduce.append(data)
        self.tensors_before_reduce.append(data_mul)
        if cast_0 is not None:
            self.tensors_before_reduce.append(cast_0)

        self.tensors_after_reduce.append(sum_x)
        self.tensors_after_reduce.append(square_sum_x)

    def _init_max_ub_count(self, output_tensors):
        """
        :param output_tensors:
        :return:
        """
        self.get_all_tensors_before_reduce(output_tensors)
        soc_ub_size = tbe_platform.get_soc_spec("UB_SIZE")
        soc_ub_size = soc_ub_size // 4
        soc_ub_size = soc_ub_size // 2  # double buffer

        total_width = 4

        max_bound = total_width * 128
        max_ub_count = int(soc_ub_size // max_bound * 128)

        self.max_single_tensor_ub_size = max_ub_count
        self.tensor_ub_size_before_reduce = max_ub_count
        self.tensor_ub_size_after_reduce = max_ub_count


class TilingStrategy(Enum):
    """
    Tiling strategy
    """
    CUT_C1 = auto()
    CUT_H = auto()
    CUT_N = auto()
    NONE_CUT = auto()


# 'pylint: disable=too-many-instance-attributes
class BNReduceInfo:
    """
    bn training reduce info object calss
    """

    def __init__(self, compute_graph_info):
        self.reduce_tensor = tuple(compute_graph_info.reduce_tensor_set)
        self.all_axes = self.get_reduce_all_axes(self.reduce_tensor[0])
        self.reduce_axes = self.get_reduce_axes(self.reduce_tensor[0])
        self.shape_before_reduce = list(self.reduce_tensor[0].op.input_tensors[0].shape)
        self.shape_after_reduce = list(self.reduce_tensor[0].shape)
        self.reduce_axis_indices = self.get_reduce_axis_indices(self.reduce_tensor[0])
        self.keepdims = len(self.shape_before_reduce) == len(self.shape_after_reduce)
        self.graph_info = compute_graph_info
        self.reduce_index_map = {}
        self.reduce_axis_map = {}
        self.record_reduce_info(self.reduce_tensor[0])

    def is_reduce_not_last_axis(self):
        """
        :return:
        """
        is_not_last_axis = self.all_axes[-1] not in self.reduce_axes
        return is_not_last_axis

    def is_reduce_last_axis(self):
        """
        :return:
        """
        return self.all_axes[-1] in self.reduce_axes

    def is_reduce_all_axes(self):
        """
        :return:
        """
        return set(self.all_axes) == set(self.reduce_axes)

    def record_reduce_info(self, tensor):
        """
        :param tensor:
        :return:
        """
        tensor_op = tensor.op
        reduce_axis_var = []
        for i in tensor_op.reduce_axis:
            reduce_axis_var.append(i)
        data_axis_var = tensor_op.body[0].source[0].args
        for ax_item in reduce_axis_var:
            for index in range(0, len(data_axis_var), 1):
                if data_axis_var[index].same_as(ax_item.var):
                    self.reduce_axis_map[index] = ax_item

        for i, ele in enumerate(self.reduce_axis_indices):
            self.reduce_index_map[ele] = i

    def get_reduce_axes(self, reduce_tensor):
        """
        Get reduce axes var of reduce tensor
        :param reduce_tensor:
        :return:
        """
        if not self.is_reduce_tensor(reduce_tensor):
            error_detail = 'Cannot get reduce axes of non-reduce tensor!'
            error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)
        reduce_tensor_body = reduce_tensor.op.body
        reduce_tensor_axes = list(reduce_tensor_body[0].axis)
        for idx, axis in enumerate(reduce_tensor_axes):
            reduce_tensor_axes[idx] = axis.var
        return reduce_tensor_axes

    def get_reduce_axis_indices(self, reduce_tensor):
        """
        Get all reduce axis index
        :param reduce_tensor:
        :return:
        """
        return [self.get_reduce_all_axes(reduce_tensor).index(axis) for axis in self.get_reduce_axes(reduce_tensor)]

    @staticmethod
    def get_reduce_all_axes(reduce_tensor):
        """
        Get all axes var for reduce tensor
        :param reduce_tensor:
        :return:
        """
        reduce_tensor_body = reduce_tensor.op.body
        return list(reduce_tensor_body[0].source[0].args)

    @staticmethod
    def is_reduce_tensor(tensor):
        """Check if tensor contains reduce body"""
        if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
            return False
        if isinstance(tensor.op.body[0], tvm.expr.Reduce):
            return True
        return False

    @staticmethod
    def find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce
        :param reduce_axis_index
        :return the last axis or the last serials axises that are not in reduce_axis
        """
        # `shape_before_reduce:(ak+1,rk,...,r2,a2,r1,a1) or (ak,rk,...,r2,a1,r1)`
        # find a1 position, a1 may contain continues axis
        a1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i not in reduce_axis_index:
                a1_end_index = i
                break
        a1_start_index = a1_end_index
        if a1_end_index is None:
            return a1_start_index, a1_end_index
        for i in range(a1_end_index, -1, -1):
            if i in reduce_axis_index:
                a1_start_index = i + 1
                break
            if i == 0:
                a1_start_index = i

        return a1_start_index, a1_end_index

    @staticmethod
    def reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index):
        """
        reorder shape (r4,a4,r3,a3,r2,a2,r1,a1) to (a4,a3,a2, r4,r3,r2,r1,a1)
        :param shape_before_reduce: like (r4,a4,r3,a3,r2,a2,r1,a1)
        :param reduce_axis_index
        :return:
        """
        # shape_before_reduce: (ak+1,rk,..,r2,a2,r1,a1)
        # find the last none-reduce axis a1

        a1_start_index, _ = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
        last_none_reduce_axis = a1_start_index

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}
        # (ak+1,ak,...,a2, rk,..,r2,r1,a1)
        reordered_shape = list(shape_before_reduce)
        temp_axis = last_none_reduce_axis - 1
        for i in range(len(reduce_axis_index) - 1, -1, -1):
            reordered_shape[temp_axis] = shape_before_reduce[reduce_axis_index[i]]
            reorder_to_orignal_axis_map[temp_axis] = reduce_axis_index[i]
            orignal_to_reorder_axis_map[reduce_axis_index[i]] = temp_axis
            temp_axis = temp_axis - 1
        for i in range(last_none_reduce_axis - 1, -1, -1):
            if i not in reduce_axis_index:
                reordered_shape[temp_axis] = shape_before_reduce[i]
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis - 1

        for i in range(last_none_reduce_axis, len(shape_before_reduce)):
            reorder_to_orignal_axis_map[i] = i
            orignal_to_reorder_axis_map[i] = i

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map

    # 'pylint: disable=too-many-locals
    @staticmethod
    def reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index):
        """
        reorder shape (a4,r4,a3,r3,a2,r2,a1,r1) to (a4,a3,a2,a1,r4,r3,r2,r1)
        :param shape_before_reduce: like(a4,r4,a3,r3,a2,r2,a1,r1)
        :param reduce_axis_index
        :return:
        """
        # `shape_before_reduce: (a4,r4,a3,r3,a2,r2,a1,r1)`

        orignal_to_reorder_axis_map = {}
        reorder_to_orignal_axis_map = {}

        reordered_shape = []
        temp_axis = 0
        for i, ele in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        for i, ele in enumerate(shape_before_reduce):
            if i in reduce_axis_index:
                reordered_shape.append(ele)
                reorder_to_orignal_axis_map[temp_axis] = i
                orignal_to_reorder_axis_map[i] = temp_axis
                temp_axis = temp_axis + 1

        return reordered_shape, reorder_to_orignal_axis_map, orignal_to_reorder_axis_map


# 'pylint: disable=too-few-public-methods
class ReduceTilingCase:
    """
    bn training reduce tiling case object class
    """

    def __init__(self):
        self.is_atomic = False
        self.block_split_axis_index = None
        self.block_factor = None
        self.ub_split_axis_index = None
        self.ub_factor = None
        self.multi_core = None
        self.tiling_key = None
        self.is_customised = False
        self.tiling_strategy = None
        self.is_fuse_hn = False
        self.ub_factor_bound = None

    def __repr__(self):
        """
        :return:
        """
        segment0 = "ATOMIC" if self.is_atomic else "NORMAL"
        segment1 = "ENABLED" if self.multi_core else "DISABLED"
        return "%s REDUCE: (%d, %d) with multicore %s" % (segment0, self.block_split_axis_index,
                                                          self.ub_split_axis_index, segment1)


class GenBnReduceTilingCase:
    """
    bn training reduce tiling case class : generate all the tiling cases
    """
    def __init__(self):
        return

    def apply_compile_info(self, reduce_info, graph_info, do_customised=True):
        """
        add the compile info to context. It is necessary for host tiling
        :param reduce_info:
        :param graph_info:
        :param do_customised:
        :return:
        """
        can_atomic = self.check_atomic_add_support(reduce_info)
        max_ub_count = graph_info.max_single_tensor_ub_size
        core_num = tbe_platform.get_soc_spec("CORE_NUM")
        keep_dims = 1
        reduce_block_size = self.get_block_size(reduce_info.reduce_tensor[0].dtype)
        common_info = [max_ub_count, core_num, keep_dims, reduce_block_size, can_atomic, do_customised]
        tbe_context.get_context().add_compile_info("_common_info", common_info)

    def calc_tiling_key(self, reduce_info, tiling):
        """
        :param reduce_info:
        :param tiling:
        :return:
        """
        # `tiling: single_case`
        shape = reduce_info.shape_before_reduce
        reduce_axis_idx = reduce_info.reduce_axis_indices
        block_split_axis = tiling.block_split_axis_index
        ub_split_axis = tiling.ub_split_axis_index
        atomic = tiling.is_atomic
        is_customised = tiling.is_customised
        is_fuse_hn = tiling.is_fuse_hn

        shape_type, db_flag = 0, 0

        if operation.get_context().get("_mode") == Constant.CONST:
            ori_axis = tbe_context.get_context().get_compile_info().get("ori_axis")
            tiling_key = self._gen_const_tiling_key(ori_axis)
        else:
            tiling_key = self._get_tiling_key(atomic, db_flag, shape_type, block_split_axis, ub_split_axis, shape,
                                              reduce_axis_idx, is_customised, is_fuse_hn)

        tiling.tiling_key = tiling_key

    # 'pylint: disable=too-many-locals
    def gen_const_tiling_case(self, single_reduce_info, compute_graph_info, customised_input=False):
        """
        :param single_reduce_info:
        :param compute_graph_info:
        :param customised_input:
        :return:
        """
        self.apply_compile_info(single_reduce_info, compute_graph_info, customised_input)
        tbe_context.get_context().add_compile_info("_reduce_shape_known", True)
        const_tiling_case = ReduceTilingCase()
        shape_after_reduce = shape_util.shape_to_list(single_reduce_info.shape_after_reduce)
        input_dtype = tuple(compute_graph_info.input_tensor_set)[0].dtype
        output_dtype = tuple(compute_graph_info.output_tensor_set)[0].dtype

        ori_const_shape = tbe_context.get_context().get_compile_info().get("ori_const_shape")

        # invoking op_tiling interface during compilation need axis info in sch
        inputs = [{"shape": ori_const_shape, "dtype": input_dtype}]

        outputs = [{"shape": shape_after_reduce, "dtype": output_dtype}]
        # the flag of invoking op_tiling interface during compilation
        tbe_context.get_context().add_compile_info("_const_shape_post", False)
        run_info = op_tiling.do_op_tiling(operation.get_context().get_op_type(),
                                          tbe_context.get_context().get_compile_info(), inputs, outputs)

        tiling_format = {
            "block_axis": "int",
            "block_factor": "int",
            "ub_axis": "int",
            "ub_factor": "int",
            "is_customised": "int",
            "is_fuse_hn": "int"
        }

        tiling_data = op_tiling.decode(run_info["tiling_data"], tiling_format)
        const_tiling_case.block_split_axis_index = tiling_data["block_axis"]
        const_tiling_case.block_factor = tiling_data["block_factor"]
        const_tiling_case.ub_split_axis_index = tiling_data["ub_axis"]
        const_tiling_case.ub_factor = tiling_data["ub_factor"]
        const_tiling_case.is_atomic = run_info["clear_atomic"]
        const_tiling_case.multi_core = run_info["block_dim"] > 1

        if customised_input and const_tiling_case.is_atomic:
            return []

        const_tiling_case.is_customised = tiling_data["is_customised"]
        const_tiling_case.is_fuse_hn = tiling_data["is_fuse_hn"]
        strategy_list = [
            TilingStrategy.CUT_N, TilingStrategy.CUT_C1, TilingStrategy.CUT_H, None, TilingStrategy.NONE_CUT
        ]
        block_axis = tiling_data["block_axis"]
        const_tiling_case.tiling_strategy = strategy_list[block_axis]

        self.calc_tiling_key(single_reduce_info, const_tiling_case)
        # the flag of invoking op_tiling interface during running
        tbe_context.get_context().add_compile_info("_const_shape_post", True)
        # invoking op_tiling interface during running need axis info in ops
        block_dims = tbe_context.get_context().get_compile_info().get("_block_dims")
        if block_dims is None:
            block_dims = {}
            tbe_context.get_context().add_compile_info("_block_dims", block_dims)
        block_dims[str(const_tiling_case.tiling_key)] = run_info["block_dim"]
        atomic_flags = tbe_context.get_context().get_compile_info().get("_atomic_flags")
        if atomic_flags is None:
            atomic_flags = {}
            tbe_context.get_context().add_compile_info("_atomic_flags", atomic_flags)
        atomic_flags[str(const_tiling_case.tiling_key)] = run_info["clear_atomic"]

        return [const_tiling_case]

    def calculate_atomic_tiling_cases(self, info):
        """
        :param info:
        :return:
        """
        tiling_case_list = []
        if self.check_atomic_add_support(info):
            shape_before_reduce = info.shape_before_reduce
            reduce_axis_index = info.reduce_axis_indices
            if info.is_reduce_all_axes():
                tiling_case_list += self._gen_atomic_tiling_case_reduce_all(shape_before_reduce)

            elif info.is_reduce_not_last_axis():
                tiling_case_list += self._gen_atomic_tiling_case_not_last_axis(shape_before_reduce, reduce_axis_index)

            elif info.is_reduce_last_axis():
                tiling_case_list += self._gen_atomic_tiling_case_last_axis(shape_before_reduce, reduce_axis_index)
        return tiling_case_list

    @staticmethod
    def _gen_atomic_tiling_case_not_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        reordered_shape, reorder_to_orignal_axis_map, _ = \
            BNReduceInfo.reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)
        tiling_case_list = []
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis in reduce_axis_index:
                block_split_axis = orignal_axis
                for j in range(0, len(reordered_shape)):
                    orignal_axis = reorder_to_orignal_axis_map[j]
                    if orignal_axis in reduce_axis_index and j < i:
                        continue
                    ub_split_axis = reorder_to_orignal_axis_map[j]
                    tiling_case = ReduceTilingCase()
                    tiling_case.is_atomic = True
                    tiling_case.block_split_axis_index = block_split_axis
                    tiling_case.ub_split_axis_index = ub_split_axis
                    tiling_case.multi_core = True
                    tiling_case_list.append(tiling_case)
        return tiling_case_list

    @staticmethod
    def _gen_atomic_tiling_case_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        reordered_shape, reorder_to_orignal_axis_map, _ = \
            BNReduceInfo.reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)
        tiling_case_list = []
        for i in range(0, len(reordered_shape)):
            orignal_axis = reorder_to_orignal_axis_map[i]
            if orignal_axis in reduce_axis_index:
                block_split_axis = orignal_axis
                for j in range(0, len(reordered_shape)):
                    orignal_axis = reorder_to_orignal_axis_map[j]
                    if orignal_axis in reduce_axis_index and j < i:
                        continue
                    ub_split_axis = reorder_to_orignal_axis_map[j]
                    tiling_case = ReduceTilingCase()
                    tiling_case.is_atomic = True
                    tiling_case.block_split_axis_index = block_split_axis
                    tiling_case.ub_split_axis_index = ub_split_axis
                    tiling_case.multi_core = True
                    tiling_case_list.append(tiling_case)
        return tiling_case_list

    @staticmethod
    def get_block_size(dtype):
        """
        :param dtype:
        :return:
        """
        if dtype in ["float32", "fp32", "int32"]:
            block_size = 8
        elif dtype in ["bool", "int8", "uint8"]:
            block_size = 32
        elif dtype in ["float16", "fp16"]:
            block_size = 16
        elif dtype in ["int64"]:
            block_size = 4
        else:
            excepted_dtype_list = ["float32", "fp32", "int32", "bool", "int8", "uint8", "float16", "fp16", "int64"]
            error_manager_vector.raise_err_input_dtype_not_supported("bn_training_reduce", "dtype", \
            excepted_dtype_list, dtype)
        return block_size

    @staticmethod
    def check_atomic_add_support(reduce_info):
        """
        :param reduce_info:
        :return:  False or True
        """
        if tbe_platform.get_soc_spec(tbe_platform.SOC_VERSION) != tbe_platform.ASCEND_910:
            return False
        reduce_tensor = reduce_info.reduce_tensor[0]
        if reduce_tensor is None:
            return False
        dtype = reduce_tensor.dtype
        if dtype != "float32":
            return False
        output_tensors = reduce_info.graph_info.output_tensor_set
        for output_tensor in output_tensors:
            dtype = output_tensor.dtype
            if dtype != "float32":
                return False
        tag = reduce_tensor.op.tag
        if tag.find("sum") == -1:
            return False
        return True

    @staticmethod
    def _gen_atomic_tiling_case_reduce_all(shape_before_reduce):
        """
        :param shape_before_reduce:
        :return:
        """
        tiling_case_list = []
        for i in range(0, len(shape_before_reduce)):
            block_split_axis = i
            for j in range(i, len(shape_before_reduce)):
                ub_split_axis = j
                tiling_case = ReduceTilingCase()
                tiling_case.is_atomic = True
                tiling_case.block_split_axis_index = block_split_axis
                tiling_case.ub_split_axis_index = ub_split_axis
                tiling_case.multi_core = True
                tiling_case_list.append(tiling_case)
        return tiling_case_list

    # 'pylint: disable=too-many-locals, too-many-arguments
    @staticmethod
    def _get_tiling_key(atomic, db_flag, shape_type, block_split_axis, ub_split_axis, shape, reduce_idx_list,
                        is_customised, is_fuse_hn):
        """
        :param atomic: "True": atomic_reduce, "False": normal_reduce.
        :param db: int number in [0,1]. "0": enable db, "1": close db.
        :param shape_type: int number in [0,99]. Diff numbers represent diff types of
               shapes. Example: "0": normal shape, "1": const shape, "2": special shape.
        :param block_split_axis: int number in [0,7] that represent index of split axis
        :param ub_split_axis: int number in [0,7] that represent index of split axis.
        :param shape: shape before reduce
        :param reduce_idx_list:

        :return: key(int32)
        """

        def _check(idx, value):
            """
            :param idx:
            :param value:
            :return:
            """
            rule = [range(2), range(100), range(9), range(9), range(1000), range(2), range(2)]
            name = ["db_flag", "shape_type", "block_split_axis", "ub_split_axis", "pattern", "customised", "fuse_hn"]
            if value not in rule[idx]:
                error_manager_vector.raise_err_input_value_invalid("bn_training_reduce", name[idx], \
                str(rule[idx]), value)

        def _get_pattern_key(_shape, _reduce_idx_list):
            """
            :param _shape:
            :param _reduce_idx_list:
            :return:
            """
            pattern_key = 0
            length = len(_shape)
            for i in range(length):
                if i in _reduce_idx_list:
                    pattern_key += 2 * 2**(length - i - 1)
                else:
                    pattern_key += 2**(length - i - 1)

            return pattern_key

        pattern = _get_pattern_key(shape, reduce_idx_list)
        pos = (db_flag, shape_type, block_split_axis, ub_split_axis, pattern, is_customised, is_fuse_hn)
        val = (10**9, 10**7, 10**6, 10**5, 10**2, 10, 1)
        key = 0
        for item, value in enumerate(pos):
            _check(item, value)
            key += value * val[item]
        if not atomic:
            key *= -1
        return key

    @staticmethod
    def _gen_const_tiling_key(reduce_axis):
        """
        generate dict key from reduce_axis
        :param reduce_axis:
        :return:
        """
        if not reduce_axis:
            return -1
        reduce_axis_local = list(reduce_axis)[:]
        reduce_axis_local = sorted(reduce_axis_local)
        dict_key = 0
        for i in reduce_axis_local:
            dict_key = 10 * dict_key + i + 1

        return dict_key

    @staticmethod
    def calculate_customised_tiling_cases(info):
        """
        :param info:
        :return:
        """
        tiling_case_list = []
        reduce_tensor = info.reduce_tensor[0]
        input_tensor = reduce_tensor.op.input_tensors[0]
        dim_len = len(input_tensor.shape)
        input_format = "NC1HWC0" if dim_len == 5 else "NCHW"
        if not input_format == "NC1HWC0":
            return tiling_case_list

        # strategy: cut c1
        ub_split_axis = [0, 2, 3]
        is_mte3_opt = [0, 1]
        for axis in ub_split_axis:
            for flag in is_mte3_opt:
                tiling_case = ReduceTilingCase()
                tiling_case.block_split_axis_index = 1
                tiling_case.ub_split_axis_index = axis
                tiling_case.is_customised = True
                tiling_case.is_fuse_hn = flag
                tiling_case.tiling_strategy = TilingStrategy.CUT_C1
                tiling_case_list.append(tiling_case)

        # strategy: cut H twice
        is_fuse_hn = [0, 1]
        for fuse_flag in is_fuse_hn:
            tiling_case = ReduceTilingCase()
            tiling_case.block_split_axis_index = 2
            tiling_case.ub_split_axis_index = 2
            tiling_case.is_customised = True
            tiling_case.is_fuse_hn = fuse_flag
            tiling_case.tiling_strategy = TilingStrategy.CUT_H
            tiling_case_list.append(tiling_case)

        # strategy: cut batch
        ub_split_axis = [0, 2, 3]
        is_c1_too_big = [0, 1]
        for axis in ub_split_axis:
            for flag in is_c1_too_big:
                tiling_case = ReduceTilingCase()
                tiling_case.block_split_axis_index = 0
                tiling_case.ub_split_axis_index = axis
                tiling_case.is_customised = True
                tiling_case.is_fuse_hn = flag
                tiling_case.tiling_strategy = TilingStrategy.CUT_N
                tiling_case_list.append(tiling_case)

        # strategy: cut general
        ub_split_axis = [0, 2, 3]
        for axis in ub_split_axis:
            tiling_case = ReduceTilingCase()
            tiling_case.block_split_axis_index = 4
            tiling_case.ub_split_axis_index = axis
            tiling_case.is_customised = True
            tiling_case.is_fuse_hn = False
            tiling_case.tiling_strategy = TilingStrategy.NONE_CUT
            tiling_case_list.append(tiling_case)

        return tiling_case_list


# 'pylint: disable=too-many-instance-attributes
class BnReduceCustomisedSchedule:
    """
    customised schedule
    """

    def __init__(self, tiling_case, graph_info, single_reduce_info):
        self._schedule = None
        self._tiling_case = tiling_case
        self._tiling_strategy = self._tiling_case.tiling_strategy
        self._mode = "dynamic"
        self._scope = "local.UB"
        self._out_tensors = set()

        self.graph_info = graph_info
        self.reduce_info = single_reduce_info
        self.max_dtype_bytes = 4
        self._tensor_space = None

        self._block_tiling_vars = {}
        self._ub_tiling_vars = {}
        self._inner_shape = []
        self.sum_x = None
        self.square_sum_x = None
        self.data_mul = None
        self.data = None
        self.cast_0 = None
        self.data_ub = None
        self.sum_x_ub = None
        self.cast_0_ub = None
        self.data_mul_ub = None
        self.sum_square_x_ub = None
        self.is_keep_dim = True
        self.need_db = True

    def do_schedule(self, outs):
        '''
        :return:
        '''
        self._out_tensors = copy.copy(outs)
        self._construct_compute_graph(outs)
        self._schedule = tvm.create_schedule(outs[0].op)

        self._schedule.tiling_key = self._tiling_case.tiling_key

        self._do_cache_read()

        self._do_storage_bound()

        self._calc_tiling()
        self._do_tiling()

        for i in range(0, len(outs)):
            outs.pop()
        for i in self._out_tensors:
            outs.append(i)

        return self._schedule

    def _construct_compute_graph(self, outs):
        """
        :param outs:
        :return:
        """
        self.sum_x = outs[0]
        self.square_sum_x = outs[1]

        self.data_mul = self.square_sum_x.op.input_tensors[1]
        self.data = self.sum_x.op.input_tensors[0]

        self.cast_0 = None
        if isinstance(self.data.op, tvm.tensor.PlaceholderOp):
            self.cast_0 = None
        else:
            self.cast_0 = self.data
            self.data = self.cast_0.op.input_tensors[0]

        shape_input = shape_util.shape_to_list(self.data.shape)
        shape_res = shape_util.shape_to_list(self.sum_x.shape)
        self.is_keep_dim = True
        if len(shape_input) != len(shape_res):
            self.is_keep_dim = False

    def _do_cache_read(self):
        """
        create cache read ub tensor
        :return:
        """
        if self.cast_0 is not None:
            self.data_ub = self._schedule.cache_read(self.data, self._scope, [self.cast_0])
            self.cast_0_ub = self._schedule.cache_read(self.cast_0, self._scope, [self.data_mul, self.sum_x])
        else:
            self.data_ub = self._schedule.cache_read(self.data, self._scope, [self.data_mul, self.sum_x])
            self.cast_0_ub = None

        self.data_mul_ub = self._schedule.cache_read(self.data_mul, self._scope, [self.sum_x])

        self._schedule[self.data_mul].compute_inline()
        if self.cast_0 is not None:
            self._schedule[self.cast_0].compute_inline()

    def _do_storage_bound(self):
        """
        schedule do storage bound
        :return:
        """
        storage_bound = self.graph_info.max_single_tensor_ub_size
        if self.cast_0 is not None:
            self._schedule[self.data_ub].set_buffer_size(storage_bound)
            self._schedule[self.cast_0_ub].set_buffer_size(storage_bound)
        else:
            self._schedule[self.data_ub].set_buffer_size(storage_bound)
            self.cast_0_ub = None

        self._schedule[self.data_mul_ub].set_buffer_size(storage_bound)
        self._schedule[self.sum_x].set_buffer_size(storage_bound)

    def _calc_tiling(self):
        """
        :return:
        """
        funcs = {
            TilingStrategy.CUT_C1: self._calc_tiling_cut_c1,
            TilingStrategy.CUT_H: self._calc_tiling_cut_c1,
            TilingStrategy.CUT_N: self._calc_tiling_cut_c1,
            TilingStrategy.NONE_CUT: self._calc_tiling_cut_c1,
        }
        funcs[self._tiling_strategy]()

    def _do_tiling(self):
        """
        :return:
        """
        funcs = {
            TilingStrategy.CUT_C1: self._do_tiling_cut_c1,
            TilingStrategy.CUT_H: self._do_tiling_cut_h,
            TilingStrategy.CUT_N: self._do_tiling_cut_n,
            TilingStrategy.NONE_CUT: self._do_tiling_cut_g,
        }
        funcs[self._tiling_strategy]()

    def _calc_tiling_cut_c1(self):
        """
        create block factor and ub factor var if it's necessary
        :return:
        """
        shape_before_reduce = self.reduce_info.shape_before_reduce
        shape = shape_util.shape_to_list(shape_before_reduce)
        b_i = self._tiling_case.block_split_axis_index
        u_i = self._tiling_case.ub_split_axis_index
        b_bound = (1, self.get_bound(shape[b_i])[1])
        u_bound = self._tiling_case.ub_factor_bound
        if u_bound is None:
            u_bound = (1, self.get_bound(shape[u_i])[1])

        block_factor = self._tiling_case.block_factor
        ub_factor = self._tiling_case.ub_factor

        if block_factor is None:
            self._block_tiling_vars[b_i] = operation.var("block_factor_" + str(b_i), b_bound)
        else:
            self._block_tiling_vars[b_i] = block_factor

        if ub_factor is None:
            self._ub_tiling_vars[u_i] = operation.var("ub_factor_" + str(u_i), u_bound)
        else:
            self._ub_tiling_vars[u_i] = ub_factor

    @staticmethod
    def get_reduce_axis_from_split_axis(ub_split_axis):
        """
        :param ub_split_axis:
        :return:
        """
        if ub_split_axis == 0:
            ub_split_reduce_axis = 0
        elif ub_split_axis == 2:
            ub_split_reduce_axis = 1
        else:
            ub_split_reduce_axis = 2
        return ub_split_reduce_axis

    # 'pylint: disable=too-many-locals, too-many-branches, too-many-statements
    def _do_tiling_cut_c1(self):
        """
        block split on C1 axis
        :return:
        """
        sch = self._schedule

        _, sum_x_ub = sch.cache_write([self.square_sum_x, self.sum_x], self._scope)

        storage_bound = self.graph_info.max_single_tensor_ub_size
        sch[sum_x_ub].set_buffer_size(storage_bound)

        sum_x_c1_axis = self.sum_x.op.axis[1]
        sum_x_c0_axis = self.sum_x.op.axis[4]

        sum_x_ub_n_axis = sum_x_ub.op.axis[0]
        sum_x_ub_c1_axis = sum_x_ub.op.axis[1]
        sum_x_ub_h_axis = sum_x_ub.op.axis[2]
        sum_x_ub_w_axis = sum_x_ub.op.axis[3]
        sum_x_ub_c0_axis = sum_x_ub.op.axis[4]

        sum_x_ub_n_reduce_axis = sum_x_ub.op.reduce_axis[0]
        sum_x_ub_h_reduce_axis = sum_x_ub.op.reduce_axis[1]
        sum_x_ub_w_reduce_axis = sum_x_ub.op.reduce_axis[2]

        b_idx = self._tiling_case.block_split_axis_index
        u_idx = self._tiling_case.ub_split_axis_index
        block_factor = self._block_tiling_vars[b_idx]
        split_factor = self._ub_tiling_vars[u_idx]

        sum_x_block_outer, sum_x_block_inner = \
            sch[self.sum_x].split(sum_x_c1_axis, factor=block_factor)

        sum_x_block_inner_outer = None
        sum_x_block_inner_inner = None
        is_mte3_opt = self._tiling_case.is_fuse_hn
        if is_mte3_opt:
            sum_x_block_inner_outer, sum_x_block_inner_inner = sch[self.sum_x].split(sum_x_block_inner, nparts=1)

        ub_split_reduce_axis = self.get_reduce_axis_from_split_axis(u_idx)
        sum_x_ub_outer, sum_x_ub_inner = \
            sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[ub_split_reduce_axis],
                                factor=split_factor)

        if ub_split_reduce_axis == 0:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_h_axis, sum_x_ub_w_axis, sum_x_ub_outer,
                                  sum_x_ub_inner, sum_x_ub_h_reduce_axis, sum_x_ub_w_reduce_axis, sum_x_ub_c0_axis)

        elif ub_split_reduce_axis == 1:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_outer, sum_x_ub_inner, sum_x_ub_w_axis, sum_x_ub_w_reduce_axis,
                                  sum_x_ub_c0_axis)

        else:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_h_reduce_axis, sum_x_ub_w_axis, sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_c0_axis)

        # 'pylint: disable=too-many-arguments
        def do_compute_at(sch, data_ub, sum_x, sum_x_ub, sum_x_ub_outer, cast_0_ub, data_mul_ub, is_mte3_opt,
                          sum_x_block_inner_outer, sum_x_block_inner):
            """
            :param sch:
            :param data_ub:
            :param sum_x:
            :param sum_x_ub:
            :param sum_x_ub_outer:
            :param cast_0_ub:
            :param data_mul_ub:
            :param is_mte3_opt:
            :param sum_x_block_inner_outer:
            :param sum_x_block_inner:
            :return:
            """
            sch[data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
            if cast_0_ub is not None:
                sch[cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
            sch[data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

            if is_mte3_opt:
                sch[sum_x_ub].compute_at(sch[sum_x], sum_x_block_inner_outer)
            else:
                sch[sum_x_ub].compute_at(sch[sum_x], sum_x_block_inner)

        do_compute_at(sch, self.data_ub, self.sum_x, sum_x_ub, sum_x_ub_outer, self.cast_0_ub, self.data_mul_ub,
                      is_mte3_opt, sum_x_block_inner_outer, sum_x_block_inner)

        block = tvm.thread_axis("blockIdx.x", sch)
        sch[self.sum_x].bind(sum_x_block_outer, block)

        if self.need_db:
            sch[self.data_ub].double_buffer()

        # 'pylint: disable=too-many-arguments
        def do_emit_insn(sch, sum_x, sum_x_ub, sum_x_ub_inner, sum_x_c0_axis, data_ub, cast_0_ub, data_mul_ub):
            """
            :param sch:
            :param sum_x:
            :param sum_x_ub:
            :param sum_x_ub_inner:
            :param sum_x_c0_axis:
            :param data_ub:
            :param cast_0_ub:
            :param data_mul_ub:
            :return:
            """
            sch[sum_x_ub].emit_insn(sum_x_ub_inner, "vector_reduce_sum")

            sch[data_ub].emit_insn(data_ub.op.axis[0], "dma_copy")
            if cast_0_ub is not None:
                sch[cast_0_ub].emit_insn(cast_0_ub.op.axis[0], "vector_conv")
            sch[data_mul_ub].emit_insn(data_mul_ub.op.axis[0], "vector_mul")

            if is_mte3_opt:
                sch[sum_x].emit_insn(sum_x_block_inner_inner, "dma_copy")
            else:
                sch[sum_x].emit_insn(sum_x_c0_axis, "dma_copy")

        do_emit_insn(sch, self.sum_x, sum_x_ub, sum_x_ub_inner, sum_x_c0_axis, self.data_ub, self.cast_0_ub,
                     self.data_mul_ub)

        self._schedule = sch

    def _do_tiling_cut_h(self):
        """
        block split on h dim
        :return:
        """
        sch = self._schedule
        core_num = tbe_platform.get_soc_spec("CORE_NUM")
        reduce_axis_loc = [1, 2, 3]

        if self._tiling_case.is_fuse_hn:
            nparts_num = core_num // 2
        else:
            nparts_num = core_num

        u_i = self._tiling_case.ub_split_axis_index
        ub_factor = self._ub_tiling_vars[u_i]
        sum_x_block_outer, sum_x_block_inner = sch[self.sum_x].split(self.sum_x.op.reduce_axis[1], nparts=nparts_num)

        sch[self.sum_x].split(sum_x_block_inner, factor=ub_factor)

        if isinstance(ub_factor, tvm.expr.Var):
            sch.set_constraint(ub_factor > 0)
        if self._tiling_case.is_fuse_hn:
            fused = sch[self.sum_x].fuse(self.sum_x.op.reduce_axis[0], sum_x_block_outer)
            sum_x_ub_rf, _ = sch.rfactor(self.sum_x, fused)
        else:
            sum_x_ub_rf, _ = sch.rfactor(self.sum_x, sum_x_block_outer)

        sch[sum_x_ub_rf].set_buffer_size(self.graph_info.max_single_tensor_ub_size)
        if isinstance(ub_factor, tvm.expr.Var):
            sch.set_constraint(ub_factor > 0)

        sum_x_global, square_sum_x_global = sch.cache_write([self.sum_x, self.square_sum_x], "")
        self.sum_x_ub = sum_x_global
        self.sum_square_x_ub = square_sum_x_global
        sch[sum_x_ub_rf].set_scope(self._scope)
        sch[self.sum_x_ub].set_buffer_size(self.graph_info.max_single_tensor_ub_size)

        self._out_tensors[0] = sum_x_global
        self._out_tensors[1] = square_sum_x_global

        if self._tiling_case.is_fuse_hn:
            sch[sum_x_global].reorder(sum_x_global.op.reduce_axis[0], sum_x_global.op.axis[1], sum_x_global.op.axis[0],
                                      sum_x_global.op.axis[2], sum_x_global.op.axis[3], sum_x_global.op.axis[4])
            sch[sum_x_ub_rf].reorder(sum_x_ub_rf.op.axis[0], sum_x_ub_rf.op.axis[1], sum_x_ub_rf.op.reduce_axis[1],
                                     sum_x_ub_rf.op.reduce_axis[2], sum_x_ub_rf.op.reduce_axis[0],
                                     sum_x_ub_rf.op.axis[5])
        else:
            self._schedule_cut_h_twice_do_reorder(sch, sum_x_global, sum_x_ub_rf)

        if self.is_keep_dim:
            sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.axis[1])
        else:
            sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.axis[0])

        if self._tiling_case.is_fuse_hn:
            sch[self.data_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])
            if self.cast_0_ub is not None:
                sch[self.cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])
            sch[self.data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[1])
        else:
            sch[self.data_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[reduce_axis_loc[1]])
            if self.cast_0_ub is not None:
                sch[self.cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[reduce_axis_loc[1]])
            sch[self.data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_ub_rf.op.reduce_axis[reduce_axis_loc[1]])

        block = tvm.thread_axis("blockIdx.x")
        sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

        if self.need_db:
            sch[self.data_ub].double_buffer()

        sch[self.data_ub].emit_insn(self.data_ub.op.axis[0], "dma_copy")
        if self.cast_0_ub is not None:
            sch[self.cast_0_ub].emit_insn(self.cast_0_ub.op.axis[0], "vector_conv")
        sch[self.data_mul_ub].emit_insn(self.data_mul_ub.op.axis[0], "vector_mul")

        if self._tiling_case.is_fuse_hn:
            sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[2], "vector_reduce_sum")
            sch[sum_x_global].emit_insn(sum_x_global.op.axis[2], "dma_copy")
        else:
            sch[sum_x_ub_rf].emit_insn(sum_x_ub_rf.op.reduce_axis[reduce_axis_loc[2]], "vector_reduce_sum")
            sch[sum_x_global].emit_insn(sum_x_global.op.axis[4], "dma_copy")

        sch[self.sum_x].emit_insn(sch[self.sum_x].op.axis[0], "phony_insn")

        self._schedule = sch

    def _do_tiling_cut_n(self):
        """
        :return:
        """
        sch = self._schedule
        core_num = tbe_platform.get_soc_spec("CORE_NUM")

        sum_x_block_outer, _ = sch[self.sum_x].split(self.sum_x.op.reduce_axis[0], nparts=core_num)

        sum_x_ub_rf, _ = sch.rfactor(self.sum_x, sum_x_block_outer)

        sum_x_global, square_sum_x_global = sch.cache_write([self.sum_x, self.square_sum_x], "")

        self._out_tensors = [sum_x_global, square_sum_x_global]

        sch[sum_x_ub_rf].set_scope(self._scope)
        storage_bound = self.graph_info.max_single_tensor_ub_size
        sch[sum_x_ub_rf].set_buffer_size(storage_bound)

        ub_split_axis = self._tiling_case.ub_split_axis_index
        split_factor = self._ub_tiling_vars[ub_split_axis]

        if ub_split_axis == 0:
            sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(sum_x_ub_rf.op.reduce_axis[-1], factor=split_factor)
        else:
            sum_x_rf_outer, sum_x_rf_inner = sch[sum_x_ub_rf].split(sum_x_ub_rf.op.reduce_axis[ub_split_axis - 2],
                                                                    factor=split_factor)

        if isinstance(split_factor, tvm.expr.Var):
            sch.set_constraint(split_factor <= storage_bound)

        sch[sum_x_global].reorder(
            sum_x_global.op.reduce_axis[0],
            sum_x_global.op.axis[0],
            sum_x_global.op.axis[1],  # C1 axis
            sum_x_global.op.axis[2],
            sum_x_global.op.axis[3],
            sum_x_global.op.axis[4])  # C0 axis
        if ub_split_axis == 0:
            sch[sum_x_ub_rf].reorder(
                sum_x_ub_rf.op.axis[0],  # N axis
                sum_x_ub_rf.op.axis[1],
                sum_x_ub_rf.op.axis[2],  # C1 axis
                sum_x_ub_rf.op.axis[3],
                sum_x_ub_rf.op.axis[4],
                sum_x_rf_outer,
                sum_x_rf_inner,
                sum_x_ub_rf.op.reduce_axis[0],
                sum_x_ub_rf.op.reduce_axis[1],
                sum_x_ub_rf.op.axis[5])  # C0 axis
        elif ub_split_axis == 2:
            sch[sum_x_ub_rf].reorder(
                sum_x_ub_rf.op.axis[0],  # N axis
                sum_x_ub_rf.op.reduce_axis[2],
                sum_x_ub_rf.op.axis[1],
                sum_x_ub_rf.op.axis[2],  # C1 axis
                sum_x_ub_rf.op.axis[3],
                sum_x_ub_rf.op.axis[4],
                sum_x_rf_outer,
                sum_x_rf_inner,
                sum_x_ub_rf.op.reduce_axis[1],
                sum_x_ub_rf.op.axis[5])  # C0 axis
        elif ub_split_axis == 3:
            sch[sum_x_ub_rf].reorder(
                sum_x_ub_rf.op.axis[0],  # N axis
                sum_x_ub_rf.op.reduce_axis[2],
                sum_x_ub_rf.op.axis[1],
                sum_x_ub_rf.op.axis[2],  # C1 axis
                sum_x_ub_rf.op.axis[3],
                sum_x_ub_rf.op.axis[4],
                sum_x_ub_rf.op.reduce_axis[0],
                sum_x_rf_outer,
                sum_x_rf_inner,
                sum_x_ub_rf.op.axis[5])  # C0 axis

        is_c1_too_big = self._tiling_case.is_fuse_hn
        if is_c1_too_big:
            sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.axis[1])
        else:
            sch[sum_x_ub_rf].compute_at(sch[sum_x_global], sum_x_global.op.reduce_axis[0])

        sch[self.data_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

        if self.cast_0_ub is not None:
            sch[self.cast_0_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)
        sch[self.data_mul_ub].compute_at(sch[sum_x_ub_rf], sum_x_rf_outer)

        block = tvm.thread_axis("blockIdx.x")
        sch[sum_x_global].bind(sum_x_global.op.reduce_axis[0], block)

        if self.need_db:
            sch[self.data_ub].double_buffer()

        sch[sum_x_ub_rf].emit_insn(sum_x_rf_inner, "vector_reduce_sum")

        sch[self.data_ub].emit_insn(self.data_ub.op.axis[0], "dma_copy")
        if self.cast_0_ub is not None:
            sch[self.cast_0_ub].emit_insn(self.cast_0_ub.op.axis[0], "vector_conv")
        sch[self.data_mul_ub].emit_insn(self.data_mul_ub.op.axis[0], "vector_mul")

        if is_c1_too_big:
            sch[sum_x_global].emit_insn(sum_x_global.op.axis[2], "dma_copy")
        else:
            sch[sum_x_global].emit_insn(sum_x_global.op.axis[1], "dma_copy")

        sch[self.sum_x].emit_insn(sch[self.sum_x].op.axis[0], "phony_insn")

        self._schedule = sch

    def _do_tiling_cut_g(self):
        """
        tiling schedule main interface
        :return:
        """
        sch = self._schedule

        ub_split_axis = self._tiling_case.ub_split_axis_index
        ub_factor = self._ub_tiling_vars[ub_split_axis]

        ub_split_reduce_axis = self.get_reduce_axis_from_split_axis(ub_split_axis)

        _, sum_x_ub = sch.cache_write([self.square_sum_x, self.sum_x], self._scope)

        sum_x_ub_outer, sum_x_ub_inner = sch[sum_x_ub].split(sum_x_ub.op.reduce_axis[ub_split_reduce_axis],
                                                             factor=ub_factor)

        sum_x_c1_axis = self.sum_x.op.axis[1]
        sum_x_c0_axis = self.sum_x.op.axis[4]
        sum_x_ub_n_axis = sum_x_ub.op.axis[0]
        sum_x_ub_c1_axis = sum_x_ub.op.axis[1]
        sum_x_ub_h_axis = sum_x_ub.op.axis[2]
        sum_x_ub_w_axis = sum_x_ub.op.axis[3]
        sum_x_ub_c0_axis = sum_x_ub.op.axis[4]

        sum_x_ub_n_reduce_axis = sum_x_ub.op.reduce_axis[0]
        sum_x_ub_h_reduce_axis = sum_x_ub.op.reduce_axis[1]
        sum_x_ub_w_reduce_axis = sum_x_ub.op.reduce_axis[2]

        if ub_split_reduce_axis == 0:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_outer, sum_x_ub_inner, sum_x_ub_h_axis,
                                  sum_x_ub_w_axis, sum_x_ub_h_reduce_axis, sum_x_ub_w_reduce_axis, sum_x_ub_c0_axis)
        elif ub_split_reduce_axis == 1:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_outer, sum_x_ub_inner, sum_x_ub_w_axis, sum_x_ub_w_reduce_axis,
                                  sum_x_ub_c0_axis)
        else:
            sch[sum_x_ub].reorder(sum_x_ub_n_axis, sum_x_ub_c1_axis, sum_x_ub_n_reduce_axis, sum_x_ub_h_axis,
                                  sum_x_ub_h_reduce_axis, sum_x_ub_w_axis, sum_x_ub_outer, sum_x_ub_inner,
                                  sum_x_ub_c0_axis)

        sch[self.data_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
        if self.cast_0_ub is not None:
            sch[self.cast_0_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)
        sch[self.data_mul_ub].compute_at(sch[sum_x_ub], sum_x_ub_outer)

        sch[sum_x_ub].compute_at(sch[self.sum_x], sum_x_c1_axis)

        if self.need_db:
            sch[self.data_ub].double_buffer()

        block = tvm.thread_axis("blockIdx.x")
        sch[self.sum_x].bind(sum_x_c1_axis, block)

        sch[self.data_ub].emit_insn(self.data_ub.op.axis[0], "dma_copy")
        if self.cast_0_ub is not None:
            sch[self.cast_0_ub].emit_insn(self.cast_0_ub.op.axis[0], "vector_conv")
        sch[self.data_mul_ub].emit_insn(self.data_mul_ub.op.axis[0], "vector_mul")
        sch[sum_x_ub].emit_insn(sum_x_ub_inner, "vector_reduce_sum")
        sch[self.sum_x].emit_insn(sum_x_c0_axis, "dma_copy")

        self._schedule = sch

    @staticmethod
    def _schedule_cut_h_twice_do_reorder(sch, sum_x_global, sum_x_ub_rf):
        """
        :param sch:
        :param sum_x_global:
        :param sum_x_ub_rf:
        :return:
        """
        sch[sum_x_global].reorder(
            sum_x_global.op.reduce_axis[0],
            sum_x_global.op.axis[0],
            sum_x_global.op.axis[1],  # C1 axis
            sum_x_global.op.axis[2],
            sum_x_global.op.axis[3],
            sum_x_global.op.axis[4])  # C0 axis

        sch[sum_x_ub_rf].reorder(
            sum_x_ub_rf.op.axis[0],
            sum_x_ub_rf.op.axis[1],  # N axis
            sum_x_ub_rf.op.axis[2],  # C1 axis
            sum_x_ub_rf.op.axis[3],
            sum_x_ub_rf.op.axis[4],
            sum_x_ub_rf.op.reduce_axis[0],
            sum_x_ub_rf.op.reduce_axis[2],
            sum_x_ub_rf.op.reduce_axis[3],
            sum_x_ub_rf.op.reduce_axis[1],
            sum_x_ub_rf.op.axis[5])  # C0 axis

    @staticmethod
    def get_bound(expr):
        """
        :param expr:
        :return:
        """
        valid_types = (int, tvm.expr.Expr)
        if not isinstance(expr, valid_types):
            error_manager_vector.raise_err_input_dtype_not_supported("bn_training_reduce", "expr", "(int, expr)", \
            type(expr))

        if isinstance(expr, int):
            return expr, expr
        if isinstance(expr, tvm.expr.IntImm):
            return expr.value, expr.value
        if isinstance(expr, tvm.expr.Var):
            return operation.get_te_var(expr.name).get_bound()

        def _mul(_a, _b):
            if _a is None or _b is None:
                return None
            _bound = _a * _b
            return None if _bound > Constant.VAR_BOUND_LIMIT else _bound

        def _max(_a, _b):
            if _a is None or _b is None:
                return None
            return max(_a, _b)

        def _min(_a, _b):
            if _a is None or _b is None:
                return None
            return min(_a, _b)

        def _parse(_expr):
            if isinstance(_expr, tvm.expr.ConstExpr):
                return _expr.value, _expr.value
            if isinstance(_expr, tvm.expr.Var):
                bound = operation.get_te_var(_expr.name).get_bound()
                return bound[0], bound[1]
            if isinstance(_expr, tvm.expr.Mul):
                left_lower, left_upper = _parse(_expr.a)
                right_lower, right_upper = _parse(_expr.b)
                _lower, _upper = _mul(left_lower, right_lower), _mul(left_upper, right_upper)
            elif isinstance(_expr, tvm.expr.Max):
                left_lower, left_upper = _parse(_expr.a)
                right_lower, right_upper = _parse(_expr.b)
                _lower, _upper = _min(left_lower, right_lower), _max(left_upper, right_upper)
            else:
                error_manager_vector.raise_err_input_dtype_not_supported("bn_training_reduce", "_expr", \
                "(ConstExpr, Var, Mul, Max)", type(_expr))
            return _lower, _upper

        return _parse(expr)


# 'pylint: disable=too-few-public-methods, too-many-instance-attributes
class BnReduceAtomicSchedule:
    """
    ReduceAtomicSchedule: base class of bn training reduce atomic schedule
    Returns
    """
    # 'pylint: disable=too-many-statements
    def __init__(self, tiling_case, graph_info, reduce_info):
        self._op = []
        self._origin_op = []
        self._res_tensor = None
        self._last_output_tensors = []
        self._input_tensors = []
        self._input_tensor_dst_tensor_map = {}
        self._mid_tensors = []  # exclude _input_tensors and last_output_tensor
        self._mid_tensor_dst_tensor_map = {}  # {mid_tensor->dst_tensor}
        self._mid_output_tensors = []
        self._mid_output_tensors_dst_tensor_map = {}
        self._cache_write_exclude_tensors = []

        self._broadcast_last_axis_tensors = []
        self._broadcast_scalars = []
        self._broadcast_scalar_dst_tensor_map = {}
        self._broadcast_not_last_axis_tensors = []

        self._tuple_reduce_tensor_out_list = []
        self._tensor_list_before_reduce = []
        self._reduce_tensors = []
        self._broadcast_tensors = []
        self._vector_dup_tensors = []  # broadcast scalar in ub
        self._tensor_dst_tensor_map = {}  # {tensor->dst_tensor(next_tensor)}
        self._tensor_scaler_operator = ["elewise_binary_mul", "elewise_binary_add"]
        # 0: reserved; 1: reduce all; 2: reduce nlast; 3: reduce last
        self.reduce_case = 0
        self._tiling_factor_vars = []
        self._total_size = 0
        self._have_reduce = False
        self._final_out_tensor_global = None
        self._final_out_tensor_global_emit_axis = 0
        self._final_out_tensor_ub_rf = None

        self._need_multi_core = True

        self._tiling_case = tiling_case
        self.graph_info = graph_info
        self.reduce_info = reduce_info

        self._schedule = None
        self._schedule_valid = True
        self._need_db = False
        self._need_multi_core = True
        self._multi_core_bind_tensor = None
        self._multi_core_fused_axis = None
        self._out_tensors = []

        self._cache_read_tensors_and_readers_map = {}
        self._cache_read_tensors_and_buffer_map = {}
        self._cache_write_tensors = []
        self._cache_write_tensors_and_buffer_map = {}
        self._compute_inline_tensors = []
        self._double_buffer_tensors = []
        self._double_buffer_map = {}
        self._tiling_tensor = None
        self._insn_map = {}
        self._reg_insn_map = {}
        self._tiling_para = {"block_tiling": {"axis": 0, "factor": 1}, "ub_tiling": {"axis": 0, "factor": 1}}

        self._tiling_result = {}
        self._compute_at_map = {}
        self._emit_insn_map = {}
        self._scope = "local.UB"

        # reduce_axis_map: key:reduce_axis_index, value:reduce_axis_var
        # reduce_index_map: key:reduce_axis_index in original index,
        #                   value:reduce_axis_index in reduce axis
        self._reduce_info = {
            "reduce_tensor": None,
            "reduce_axis_map": {},
            "reduce_axis_index": [],
            "reduce_index_map": [],
            "shape_before_reduce": None,
            "keep_dims": True,
            "dtype": None
        }

        self._reduce_tiling_para = {
            "block_tiling": {
                "tiling_tensor": None,
                "axis": 0,
                "axis_var": None,
                "factor": 1
            },
            "ub_tiling": [{
                "tiling_tensor": None,
                "axis": 0,
                "axis_var": None,
                "factor": 1
            }]
        }

        self._reduce_tiling_result = {"block_tiling": {}, "ub_tiling": [{}]}

        self._storage_align_para = {}
        self._axis_offset = 0

    def do_schedule(self, outs):
        """
        main interface for atomic schedule
        :param outs:
        :param tiling_case:
        :param graph_info:
        :param reduce_info:
        :return:
        """
        self._res_tensor = outs[0]
        self._schedule = tvm.create_schedule([self._res_tensor.op])

        self._out_tensors = copy.copy(outs)

        self._do_cache_read()
        self._do_cache_write()
        self._do_compute_inline()

        self._calculate_tiling()
        self._do_tiling()

        self._do_reorder()

        self._do_storage_bound()
        self._do_set_constraint()

        self._caculate_storage_align()
        self._do_storage_align()

        self._calculate_multi_core()
        self._do_multi_core()

        self._calculate_compute_at()
        self._do_compute_at()

        self._calculate_emit_insn()
        self._do_emit_insn()

        self._do_double_buffer()

        for i in range(0, len(outs)):
            outs.pop()
        for i in self._out_tensors:
            outs.append(i)

        return self._schedule

    def init(self, out_tensors):
        """
        :param out_tensors:
        :param spec_node_list:
        :return:
        """
        self._out_tensors = copy.copy(out_tensors)

        is_success = self._construct_compute_graph(out_tensors)
        if not is_success:
            return False

        self._calculate_cache_read()
        self._calculate_cache_write()
        self._calculate_compute_inline()

        return True

    def _do_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        self._double_buffer_tensors.clear()

        for i in self._cache_read_tensors_and_readers_map:
            readers = self._cache_read_tensors_and_readers_map[i]
            read_buffer = self._schedule.cache_read(i, self._scope, readers)

            self._cache_read_tensors_and_buffer_map[i] = read_buffer

            self._double_buffer_tensors.append(read_buffer)

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
            write_buffer = self._schedule.cache_write(i, self._scope)
            self._cache_write_tensors_and_buffer_map[i] = write_buffer

    def _do_compute_inline(self):
        """
        compute inline operations
        """
        for i in self._compute_inline_tensors:
            self._schedule[i].compute_inline()

    def _do_multi_core(self):
        """
        :return:
        """
        if self._need_multi_core:
            res = self._multi_core_bind_tensor
            block = tvm.thread_axis("blockIdx.x")
            self._schedule[res].bind(self._multi_core_fused_axis, block)

    def _do_compute_at(self):
        """
        :return:
        """
        for stage in self._compute_at_map:
            parent_stage = self._compute_at_map[stage]["parent"]
            scope_iter_var = self._compute_at_map[stage]["scope"]
            self._schedule[stage].compute_at(parent_stage, scope_iter_var)

    def _do_double_buffer(self):
        """
        double buffer operations
        read_buffer : the all read_cache for input in ub, type is list
        """
        temp_write_buffer = []
        if self._need_db:
            for i in self._double_buffer_tensors:
                self._schedule[i].double_buffer()
                # just for ternary instruction
                if i in self._double_buffer_map:
                    buffers = list(set(self._double_buffer_map[i]))
                    for buffer in buffers:
                        temp_write_buffer.append(buffer)
                        self._schedule[buffer].double_buffer()
            if temp_write_buffer:
                self._recursive_double_buffer(temp_write_buffer)

    def _do_emit_insn(self):
        for stage in self._emit_insn_map:
            scope_iter_var = self._emit_insn_map[stage]["scope"]
            instruction = self._emit_insn_map[stage]["instruction"]
            self._schedule[stage].emit_insn(scope_iter_var, instruction)

    @staticmethod
    def _map_apend(input_map, key, value):
        """
        :param input_map:
        :param key:
        :param value:
        :return:
        """
        if input_map.get(key):
            if isinstance(value, list):
                for tmp_value in value:
                    if tmp_value not in input_map[key]:
                        input_map[key].append(tmp_value)
            else:
                if value not in input_map[key]:
                    input_map[key].append(value)
        else:
            if isinstance(value, list):
                input_map[key] = value
            else:
                input_map[key] = [value]

    def get_dst_tensor_map(self, reslist, tensor_map):
        """
        get the dst_tensor list of the tensor with more than one dst_tensor
        tensor_map = {input: outputlist}
        """
        for out_tensor in reslist:
            for in_tensor in list(out_tensor.op.input_tensors):
                if in_tensor in tensor_map:
                    if out_tensor not in tensor_map[in_tensor]:
                        tensor_map[in_tensor].append(out_tensor)
                else:
                    tensor_map[in_tensor] = [out_tensor]
                    self.get_dst_tensor_map([in_tensor], tensor_map)

    # 'pylint: disable=too-many-locals
    def _construct_compute_graph(self, out_tensors):
        """
        record relate context imformations of operations

        """
        # find the last out tensor
        last_output_tensor = out_tensors[0]

        visited_list = []
        tensor_list = []

        visited_list.append(last_output_tensor)
        tensor_list.append(last_output_tensor)
        self.__gen_reversed_subgraph_list(last_output_tensor, tensor_list, visited_list)

        self._last_output_tensors = out_tensors

        # tensor classification
        self._tensor_classify(out_tensors, tensor_list)

        self._res_tensor = self._last_output_tensors[0]
        self._record_reduce_info(self._res_tensor)

        self._tensor_list_before_reduce = self.graph_info.tensors_before_reduce

        return True

    def _tensor_classify(self, out_tensors, tensor_list):
        """
        :param out_tensors:
        :param tensor_list:
        :return:
        """

        for tensor in tensor_list:
            if isinstance(tensor.op, tvm.tensor.PlaceholderOp):
                self._input_tensors.append(tensor)
                if tensor in self._tensor_dst_tensor_map.keys():
                    self._input_tensor_dst_tensor_map[tensor] = \
                        self._tensor_dst_tensor_map[tensor]
            else:
                if tensor.op.tag.find("reduce") != -1:
                    self._reduce_tensors.append(tensor)
                if tensor.op.tag.find("broadcast") != -1:
                    if tensor.op.tag == "unified_broadcast":
                        self._broadcast_tensors.append(tensor)
                    else:
                        self._vector_dup_tensors.append(tensor)
                if tensor in out_tensors:
                    if tensor in self._tensor_dst_tensor_map.keys():
                        self._mid_output_tensors.append(tensor)
                        self._mid_output_tensors_dst_tensor_map[tensor] = \
                            self._tensor_dst_tensor_map[tensor]
                        self._mid_tensors.append(tensor)
                else:
                    self._mid_tensors.append(tensor)

    def __gen_reversed_subgraph_list(self, tensor, tensor_list, visited_list):
        """traverse tensors by Depth-First-Search

        Parameters
        ----------
        tensor : tensor
            traverse tensors from this tensor,
            traversing its input tensors recursively.

        tensor_list : list
            record tensors in the order of Depth-First-Search.

        visited_list : list
            record tensors which has been visited.
        """
        for in_tensor in list(tensor.op.input_tensors):
            self._map_apend(self._tensor_dst_tensor_map, in_tensor, tensor)
            if in_tensor not in visited_list:
                visited_list.append(in_tensor)
                tensor_list.append(in_tensor)

            self.__gen_reversed_subgraph_list(in_tensor, tensor_list, visited_list)

    def _calculate_cache_read(self):
        """
        cache read operations

        Parameters
        ----------
        None

        Returns
        -------
        list : read buffers
        """
        for i in self._input_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i, self._input_tensor_dst_tensor_map[i])

        for i in self._mid_output_tensors:
            self._map_apend(self._cache_read_tensors_and_readers_map, i, self._mid_output_tensors_dst_tensor_map[i])

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
            if i not in self._cache_write_exclude_tensors:
                self._cache_write_tensors.append(i)

    @staticmethod
    def _is_reduce_all_axis(shape_before_reduce, reduce_axis_index):
        """
        :return:
        """
        for i, _ in enumerate(shape_before_reduce):
            if i not in reduce_axis_index:
                return False
        return True

    @staticmethod
    def _is_reduce_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        return reduce_axis_index[-1] == len(shape_before_reduce) - 1

    @staticmethod
    def _is_reduce_not_last_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        return reduce_axis_index[-1] != len(shape_before_reduce) - 1

    def _do_storage_bound(self):
        """
        :return:
        """

        def _get_tensor_space(_tensor):
            if _tensor in self.graph_info.tensors_after_reduce:
                _space = self.graph_info.tensor_ub_size_after_reduce
            elif _tensor in self.graph_info.tensors_before_reduce:
                _space = self.graph_info.tensor_ub_size_before_reduce
            else:
                error_detail = 'undefined tensor, _tensor:%s' % _tensor
                error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

            return _space

        for tensor in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
            self._schedule[read_buffer].set_buffer_size(_get_tensor_space(tensor))

        for tensor in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
            self._schedule[write_buffer].set_buffer_size(_get_tensor_space(tensor))

        for tensor in self._mid_output_tensors:
            self._schedule[tensor].set_buffer_size(_get_tensor_space(tensor))

        # final output must be reduce_shape
        _bound = self.graph_info.tensor_ub_size_after_reduce
        self._schedule[self._final_out_tensor_ub_rf].set_buffer_size(_bound)
        self._schedule[self._final_out_tensor_global].set_buffer_size(_bound)

    def _caculate_storage_align(self):
        """
        :return:
        """
        self._storage_align_para.clear()
        if not self._need_storage_align():
            return

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)
        if a1_end_index is None:
            return

        def _construct_storage_align_para(tensor_list, align_axis, mid_out_align_axis):
            """
            :param tensor_list:
            :param align_axis:
            :param align_factor:
            :param mid_out_align_axis:
            :return:
            """
            for i in self._cache_read_tensors_and_buffer_map:
                if i in tensor_list:
                    read_buffer = self._cache_read_tensors_and_buffer_map[i]
                    align_factor = Constant.BLOCK_SIZE_BYTE // Constant.DTYPE_BYTE_MAPPING.get(read_buffer.dtype)
                    para = {
                        "align_axis_var": read_buffer.op.axis[align_axis],
                        "align_factor": align_factor,
                        "offset": 0
                    }
                    self._storage_align_para[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                if i in tensor_list:
                    write_buffer = self._cache_write_tensors_and_buffer_map[i]
                    align_factor = Constant.BLOCK_SIZE_BYTE // Constant.DTYPE_BYTE_MAPPING.get(write_buffer.dtype)
                    para = {
                        "align_axis_var": write_buffer.op.axis[align_axis],
                        "align_factor": align_factor,
                        "offset": 0
                    }
                    self._storage_align_para[write_buffer] = para
            for tensor in self._mid_output_tensors:
                if tensor in tensor_list:
                    align_factor = Constant.BLOCK_SIZE_BYTE // Constant.DTYPE_BYTE_MAPPING.get(tensor.dtype)
                    para = {
                        "align_axis_var": tensor.op.axis[mid_out_align_axis],
                        "align_factor": align_factor,
                        "offset": 0
                    }
                    self._storage_align_para[tensor] = para

        if self.reduce_case == 2:
            align_axis = a1_start_index - 1
            if align_axis < 0:
                align_axis = a1_end_index

            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_storage_align_para(tensor_list_before_reduce, align_axis, align_axis)

            is_keep_dims = self._reduce_info["keep_dims"]

            res_a1_start_index = a1_start_index
            if not is_keep_dims:
                res_a1_start_index = a1_start_index - len(reduce_axis_index)
            if res_a1_start_index == 0:
                return
            res_align_axis = res_a1_start_index - 1

            align_factor = Constant.BLOCK_SIZE_BYTE // \
            Constant.DTYPE_BYTE_MAPPING.get(self._final_out_tensor_ub_rf.dtype)            
            para = {
                "align_axis_var": self._final_out_tensor_ub_rf.op.axis[res_align_axis + self._axis_offset],
                "align_factor": align_factor,
                "offset": 0
            }
            self._storage_align_para[self._final_out_tensor_ub_rf] = para

            para = {
                "align_axis_var": self._final_out_tensor_global.op.axis[res_align_axis],
                "align_factor": align_factor,
                "offset": 0
            }
            self._storage_align_para[self._final_out_tensor_global] = para

        else:
            align_axis = a1_end_index
            tensor_list_before_reduce = self._tensor_list_before_reduce
            _construct_storage_align_para(tensor_list_before_reduce, align_axis, align_axis)

    def _do_storage_align(self):
        """
        :param hape_before_reduce:
        :param reduce_axis_index:
        :return:
        """

        for stage in self._storage_align_para:
            scope_iter_var = self._storage_align_para[stage]["align_axis_var"]
            align_factor = self._storage_align_para[stage]["align_factor"]
            offset = self._storage_align_para[stage]["offset"]
            self._schedule[stage].storage_align(scope_iter_var, align_factor, offset)

    def _need_storage_align(self):
        """
        :return:
        """
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_split_axis = ub_tiling_para["axis"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        # for shape(r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, do not need storage_align
        if self.reduce_case == 2:
            a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce,
                                                                                   reduce_axis_index)
            if a1_end_index is None:
                return False
            if a1_start_index <= ub_split_axis <= a1_end_index:
                return False

        elif self.reduce_case == 3:
            r1_start_index, r1_end_index = self._find_last_reduce_axis(shape_before_reduce, reduce_axis_index)
            if r1_end_index is None:
                return False
            # for shape(a4,r4,a3,r3,a2,r2,a1,r1), if ub split r1, do not need storage_align
            if r1_start_index <= ub_split_axis <= r1_end_index:
                return False
        else:
            return False

        return True

    @staticmethod
    def _find_last_reduce_axis(shape_before_reduce, reduce_axis_index):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        # shape_before_reduce:(ak+1,rk,..,r2,a2,r1,a1) or (ak,rk,..,r2,a1,r1),
        # find r1 position, r1 may contain continues axis
        r1_end_index = None
        for i in range(len(shape_before_reduce) - 1, -1, -1):
            if i in reduce_axis_index:
                r1_end_index = i
                break
        r1_start_index = r1_end_index
        if r1_end_index is None:
            return r1_start_index, r1_end_index
        for i in range(r1_end_index, -1, -1):
            if i not in reduce_axis_index:
                r1_start_index = i + 1
                break
            if i == 0:
                r1_start_index = i

        return r1_start_index, r1_end_index

    # 'pylint: disable=too-many-locals
    def _calculate_tiling(self):
        """
        calculate tiling strategy

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        self._tiling_factor_vars.clear()

        tiling_case = self._tiling_case

        block_split_axis = tiling_case.block_split_axis_index
        block_factor = tiling_case.block_factor
        ub_split_axis = tiling_case.ub_split_axis_index
        ub_factor = tiling_case.ub_factor

        if block_factor is None:

            block_inner = operation.var("block_factor", (1, None))
            self._tiling_factor_vars.append(block_inner)
        else:
            block_inner = block_factor

        if ub_factor is None:
            ub_inner = operation.var("ub_factor", (1, None))
            self._tiling_factor_vars.append(ub_inner)
        else:
            ub_inner = ub_factor

        res_tensor = self._res_tensor
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if block_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[block_split_axis]
            block_tiling_para = {
                "tiling_tensor": res_tensor,
                "axis": block_split_axis,
                "axis_var": axis_var,
                "factor": block_inner
            }
        else:
            block_tiling_para = {
                "tiling_tensor": res_tensor,
                "axis": block_split_axis,
                "axis_var": None,
                "factor": block_inner
            }
        # if ub tiling is performed along a certain reduce axis,
        # need to pass the reduce axis as the split itervar parameter
        if ub_split_axis in reduce_axis_index:
            reduce_axis = self._reduce_info["reduce_axis_map"]
            axis_var = reduce_axis[ub_split_axis]
            ub_tiling_para = [{
                "tiling_tensor": res_tensor,
                "axis": ub_split_axis,
                "axis_var": axis_var,
                "factor": ub_inner
            }]
        else:
            ub_tiling_para = [{
                "tiling_tensor": res_tensor,
                "axis": ub_split_axis,
                "axis_var": None,
                "factor": ub_inner
            }]

        self._reduce_tiling_para["block_tiling"] = block_tiling_para
        self._reduce_tiling_para["ub_tiling"] = ub_tiling_para

    def _do_tiling(self):
        """
        :return:
        """
        self._do_block_tiling()

        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_outer = block_tiling_result["outer_itervar"]

        self._atomic_additonal_schedule(block_tiling_tensor, block_split_axis, res_block_outer)

        self._do_ub_tiling()

    def _do_block_tiling(self):
        """
        :return:
        """
        block_tiling_para = self._reduce_tiling_para["block_tiling"]
        block_tiling_tensor = block_tiling_para["tiling_tensor"]
        block_split_axis = block_tiling_para["axis"]
        block_split_inner = block_tiling_para["factor"]

        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        if block_split_axis not in reduce_axis_index:
            error_manager_vector.raise_err_input_value_invalid("bn_training_reduce", "block_split_axis", \
            reduce_axis_index, block_split_axis)

        if "axis_var" in block_tiling_para.keys() and \
                block_tiling_para["axis_var"] is not None:
            axis_var = block_tiling_para["axis_var"]
        else:
            axis_var = block_tiling_tensor.op.axis[block_split_axis]

        res_block_outer, res_block_inner = \
            self._schedule[block_tiling_tensor].split(axis_var,
                                                      factor=block_split_inner)
        block_tiling_result = {
            "tiling_tensor": block_tiling_tensor,
            "axis": block_split_axis,
            "parent_itervar": axis_var,
            "outer_itervar": res_block_outer,
            "inner_itervar": res_block_inner
        }
        self._reduce_tiling_result["block_tiling"] = block_tiling_result

    # 'pylint: disable=too-many-locals
    def _do_ub_tiling(self):
        """
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_tiling_tensor = block_tiling_result["tiling_tensor"]
        block_split_axis = block_tiling_result["axis"]
        res_block_inner = block_tiling_result["inner_itervar"]

        ub_tiling_result_list = []
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_tiling_tensor = ub_tiling_para["tiling_tensor"]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        if ub_tiling_tensor is not None:
            if block_tiling_tensor is not None and block_split_axis == ub_split_axis \
                    and ub_tiling_tensor == block_tiling_tensor:
                res_ub_outer, res_ub_inner = self._schedule[ub_tiling_tensor].split(res_block_inner,
                                                                                    factor=ub_split_inner)
                ub_tiling_result = {
                    "tiling_tensor": ub_tiling_tensor,
                    "axis": ub_split_axis,
                    "parent_itervar": res_block_inner,
                    "outer_itervar": res_ub_outer,
                    "inner_itervar": res_ub_inner
                }
            else:
                # if the axis_var is not empty,
                # the axis_var is used as the split parameter first,
                # otherwise the split_axis of the tilting_tensor is used as
                # the split parameter
                if "axis_var" in ub_tiling_para.keys() and \
                        ub_tiling_para["axis_var"] is not None:
                    axis_var = ub_tiling_para["axis_var"]
                else:
                    if self.reduce_case > 0:
                        shape_before_reduce = self._reduce_info["shape_before_reduce"]
                        reduce_axis_index = self._reduce_info["reduce_axis_index"]
                        is_keep_dim = self._reduce_info["keep_dims"]
                        none_reduce_index_map = self._find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index,
                                                                                is_keep_dim)
                        axis = none_reduce_index_map[ub_split_axis]
                        axis_var = ub_tiling_tensor.op.axis[axis + self._axis_offset]
                    else:
                        axis_var = ub_tiling_tensor.op.axis[ub_split_axis]
                if self.reduce_case > 0 and block_split_axis == ub_split_axis:
                    res_ub_outer, res_ub_inner = self._schedule[ub_tiling_tensor].split(
                        ub_tiling_tensor.op.reduce_axis[-1], factor=ub_split_inner)
                else:
                    res_ub_outer, res_ub_inner = self._schedule[ub_tiling_tensor].split(axis_var, factor=ub_split_inner)

                ub_tiling_result = {
                    "tiling_tensor": ub_tiling_tensor,
                    "axis": ub_split_axis,
                    "parent_itervar": axis_var,
                    "outer_itervar": res_ub_outer,
                    "inner_itervar": res_ub_inner
                }
            ub_tiling_result_list.append(ub_tiling_result)

        self._reduce_tiling_result["ub_tiling"] = ub_tiling_result_list

    def _atomic_additonal_schedule(self, block_tiling_tensor, block_split_axis, block_outer_var):
        """
        :param block_tiling_tensor:
        :param block_split_axis:
        :param block_outer_var:
        :return:
        """

        fused_list = []
        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        for i in range(0, reduce_block_axis):
            fused_list.append(block_tiling_tensor.op.reduce_axis[i])

        fused_list.append(block_outer_var)
        fused = self._schedule[block_tiling_tensor].fuse(*fused_list)

        factor_axis = 0
        final_out_tensor_ub_rf = self._schedule.rfactor(block_tiling_tensor, fused, factor_axis=factor_axis)
        if factor_axis == 0:
            self._axis_offset = 1
        else:
            self._axis_offset = 0

        if len(self._last_output_tensors) > 1:
            final_out_tensor_ub_rf = final_out_tensor_ub_rf[0]
        self._schedule[final_out_tensor_ub_rf].set_scope(tbe_platform.scope_ubuf)
        self._reduce_tiling_para["ub_tiling"][0]["tiling_tensor"] = \
            final_out_tensor_ub_rf

        final_out_tensor_global_list = self._schedule.cache_write(self._last_output_tensors, "")
        final_out_tensor_global = final_out_tensor_global_list[0]
        self._final_out_tensor_global = final_out_tensor_global
        self._final_out_tensor_ub_rf = final_out_tensor_ub_rf
        self.__replace_out_tensors(final_out_tensor_global_list)

    def _do_set_constraint(self):
        """
        :return:
        """
        if operation.get_context().get("_mode") == Constant.CONST:
            return
        ub_tiling_para_list = self._reduce_tiling_para["ub_tiling"]
        ub_tiling_para = ub_tiling_para_list[0]
        ub_split_axis = ub_tiling_para["axis"]
        ub_split_inner = ub_tiling_para["factor"]

        shape_before_reduce = self._reduce_info["shape_before_reduce_expr"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        max_ub_count = self.graph_info.tensor_ub_size_before_reduce

        if self.reduce_case == 2:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                BNReduceInfo.reorder_reduce_nlast_shape(shape_before_reduce, reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self._schedule.set_constraint(reordered_shape[i] <= max_ub_count)

        else:
            reordered_shape, _, orignal_to_reorder_axis_map = \
                BNReduceInfo.reorder_reduce_last_shape(shape_before_reduce, reduce_axis_index)
            axis = orignal_to_reorder_axis_map[ub_split_axis]
            shape_in_ub = ub_split_inner
            if isinstance(shape_in_ub, tvm.expr.Var):
                self._schedule.set_constraint(ub_split_inner <= max_ub_count)
            for i in range(axis + 1, len(reordered_shape)):
                shape_in_ub = shape_in_ub * reordered_shape[i]
                if isinstance(shape_in_ub, tvm.expr.Var):
                    self._schedule.set_constraint(reordered_shape[i] <= max_ub_count)

        self._schedule.set_constraint(shape_in_ub <= max_ub_count)

    def _do_reorder(self):
        """
        :return:
        """

        final_out_tensor_global = self._final_out_tensor_global
        final_out_tensor_ub_rf = self._final_out_tensor_ub_rf

        if self.reduce_case == 1:
            self._reorder_atomic_reduce_all(final_out_tensor_ub_rf, final_out_tensor_global)
        if self.reduce_case == 2:
            self._reorder_reduce_not_last_axis_before_reduce()
            # for shape(r4,a4,r3,a3,r2,a2,r1,a1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak,r1,.rb-1,rbi,rb+1,,.rn)
            self._reorder_atomic_reduce_not_last_axis(final_out_tensor_ub_rf, final_out_tensor_global)
        if self.reduce_case == 3:

            self._reorder_reduce_last_axis_before_reduce()
            # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
            # reorder ir (a1,a2,..ak,rbo,r1,.,rb-1,rb+1,..rn,rbi) to
            # (rbo,a1,a2,..ak-1,r1,.rb-1,rbi,rb+1,,.,ak,rn)
            self._reorder_atomic_reduce_last_axis(final_out_tensor_ub_rf, final_out_tensor_global)

    def _reorder_reduce_not_last_axis_before_reduce(self):
        """
        :return:
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)

        if a1_end_index is None:
            error_detail = "errCode:E90001, detailed_cause:a1_end_index can not be none!"
            error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

        # reorder tensor before reduce,
        # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
        # the orignal ir is (r4,a4,r3,a3,r2,a2,r1,a1),
        # reorder orignal ir to (a4,a3,a2,r4,r3,r2,r1,a1)
        tensor_list_before_reduce = self._tensor_list_before_reduce
        tensor_list = tensor_list_before_reduce

        def __get_reorder_list(tensor):
            """
            :param tensor:
            :return:
            """
            reordered_axis_list = []
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    reordered_axis_list.append(tensor.op.axis[i])

            for i in reduce_axis_index:
                reordered_axis_list.append(tensor.op.axis[i])

            return reordered_axis_list

        for tensor in self._cache_read_tensors_and_buffer_map:
            if tensor in tensor_list:
                read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
                reordered_axis_list = __get_reorder_list(read_buffer)
                self._schedule[read_buffer].reorder(*(reordered_axis_list))

        for tensor in self._cache_write_tensors_and_buffer_map:
            if tensor in tensor_list:
                write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
                reordered_axis_list = __get_reorder_list(write_buffer)
                self._schedule[write_buffer].reorder(*(reordered_axis_list))

        for tensor in self._mid_output_tensors:
            if tensor in tensor_list:
                reordered_axis_list = __get_reorder_list(tensor)
                self._schedule[tensor].reorder(*(reordered_axis_list))

    # 'pylint: disable=too-many-branches
    def _reorder_reduce_last_axis_before_reduce(self):
        """
        :return:
        """
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        shape_before_reduce = self._reduce_info["shape_before_reduce"]

        # reorder tensor before reduce,
        # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
        # the orignal ir is(a4,r4,a3,r3,a2,r2,a1,r1),
        # reorder orignal ir to (a4,a3,a2,a1,r4,r3,r2,r1)

        tensor_list_before_reduce = self._tensor_list_before_reduce
        tensor_list = tensor_list_before_reduce

        for tensor in self._cache_read_tensors_and_buffer_map:
            if tensor in tensor_list:
                read_buffer = self._cache_read_tensors_and_buffer_map[tensor]
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(read_buffer.op.axis[i])

                for i in range(0, len(shape_before_reduce)):
                    if i in reduce_axis_index:
                        reordered_axis_list.append(read_buffer.op.axis[i])

                self._schedule[read_buffer].reorder(*(reordered_axis_list))

        for tensor in self._cache_write_tensors_and_buffer_map:
            if tensor in tensor_list:
                write_buffer = self._cache_write_tensors_and_buffer_map[tensor]
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(write_buffer.op.axis[i])
                self._schedule[write_buffer].reorder(*(reordered_axis_list))

        for tensor in self._mid_output_tensors:
            if tensor in tensor_list:
                reordered_axis_list = []
                for i in range(0, len(shape_before_reduce)):
                    if i not in reduce_axis_index:
                        reordered_axis_list.append(tensor.op.axis[i])
                self._schedule[tensor].reorder(*(reordered_axis_list))

    def __replace_out_tensors(self, final_out_tensor_global_list):
        """
        :param final_out_tensor_global_list:
        :return:
        """
        final_out_tensor_list_index = []
        for tensor in self._last_output_tensors:
            for i in range(0, len(self._out_tensors)):
                if tensor == self._out_tensors[i]:
                    final_out_tensor_list_index.append(i)
                    break
        for i, _ in enumerate(final_out_tensor_global_list):
            self._out_tensors[final_out_tensor_list_index[i]] = \
                final_out_tensor_global_list[i]

    def _reorder_atomic_reduce_all(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_block_axis = reduce_index_map[block_split_axis]
        reduce_ub_axis = reduce_index_map[ub_split_axis]

        if block_split_axis != ub_split_axis:
            # rbi
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])
        for i in range(reduce_block_axis, reduce_ub_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        ub_rf_reordered_axis_list.append(res_ub_outer)
        ub_rf_reordered_axis_list.append(res_ub_inner)
        for i in range(reduce_ub_axis, len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
            reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
            ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

    # 'pylint: disable=too-many-locals, too-many-statements
    def _reorder_atomic_reduce_not_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # reorder (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak+1,ak,..a2,rk,.,rb-1,rbi,rb+1,..r2,r1,a1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..r2,r1,a1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        # rbo axis of out_tensor_global
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*(global_reordered_axis_list))

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce, reduce_axis_index)

        is_keep_dim = self._reduce_info["keep_dims"]

        reduce_index_map = self._reduce_info["reduce_index_map"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]

        none_reduce_index_map = self._find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, is_keep_dim)

        # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
        # reorder ir (ak+1,ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo_fused, ak,..a2,rbi,rb-1,..r2,r1,a1)
        # append rbo_fused
        ub_rf_reordered_axis_list = [out_tensor_ub_rf.op.axis[-1 + self._axis_offset]]

        def __reorder_case_1(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])

            # add a1 outer, a1 may be continous
            for i in range(a1_start_index, ub_split_axis):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                          self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1 inner, a1 may be continous
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                          self._axis_offset])

            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        def __reorder_case_2(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            reduce_ub_axis = reduce_index_map[ub_split_axis]
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis, len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                          self._axis_offset])
            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        def __reorder_case_3(ub_rf_reordered_axis_list):
            # add axis (ak,..a2)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])
            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(ub_split_axis + 1, a1_start_index):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])

            # add rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

            # add a1
            for i in range(a1_start_index, a1_end_index + 1):
                none_reduce_index = none_reduce_index_map[i]
                ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                          self._axis_offset])

            self._schedule[out_tensor_ub_rf].reorder(*(ub_rf_reordered_axis_list))

        # if ub split axis in(a1)
        reduce_block_axis = reduce_index_map[block_split_axis]
        if a1_start_index <= ub_split_axis <= a1_end_index:
            __reorder_case_1(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (rbi,rb-1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            __reorder_case_2(ub_rf_reordered_axis_list)
            return

        # if ub split axis in (ak,..a2)
        if ub_split_axis not in reduce_axis_index:
            __reorder_case_3(ub_rf_reordered_axis_list)

    def _reorder_atomic_reduce_last_axis(self, out_tensor_ub_rf, out_tensor_global):
        """
        :param out_tensor_ub_rf:
        :param out_tensor_global:
        :return:
        """
        # for shape (a4,r4,a3,r3,a2,r2,a1,r1),
        # reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,rk,.,rb-1,rbi,rb+1,..a1,r1) or
        # (rbo_fused, ak,..a2,rbi,rb+1,..a1,r1) if need fused
        block_tiling_result = self._reduce_tiling_result["block_tiling"]
        block_split_axis = block_tiling_result["axis"]

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_split_axis = ub_tiling_result["axis"]
        res_ub_outer = ub_tiling_result["outer_itervar"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        shape_before_reduce = self._reduce_info["shape_before_reduce"]
        reduce_axis_index = self._reduce_info["reduce_axis_index"]
        is_keep_dim = self._reduce_info["keep_dims"]

        # reorder ir (ak,..a2,a1,rbo) to (rbo,ak,..a2,a1)
        global_reordered_axis_list = [out_tensor_global.op.reduce_axis[0]]
        for i in range(0, len(out_tensor_global.op.axis)):
            axis = out_tensor_global.op.axis[i]
            global_reordered_axis_list.append(axis)

        self._schedule[out_tensor_global].reorder(*global_reordered_axis_list)

        none_reduce_index_map = self._find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, is_keep_dim)

        ub_rf_reordered_axis_list = []
        reduce_index_map = self._reduce_info["reduce_index_map"]

        # 'reorder (ak,..a2,a1,rbo,rk,.,rb+1,rb-1,..r1,rbi) to
        # (rbo, ak,..a2,a1, rk,.,rb-1,rbi,rb+1,..r2,r1) or
        # (rbo_fused, ak,..a2,a1, rbi,rb+1,..r2,r1) if need fused

        # rbo
        ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.axis[-1 + self._axis_offset])

        # 'if ub split axis in (rbi,rb+1,..r2,r1)
        if ub_split_axis in reduce_axis_index:
            # add axis (ak,..a2,a1)
            for i in range(0, len(self._schedule[out_tensor_ub_rf].op.axis) - 1):
                ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[i + self._axis_offset])

            # 'append rbi
            if block_split_axis != ub_split_axis:
                ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            reduce_block_axis = reduce_index_map[block_split_axis]
            reduce_ub_axis = reduce_index_map[ub_split_axis]
            for i in range(reduce_block_axis, reduce_ub_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)
            for i in range(reduce_ub_axis, len(out_tensor_ub_rf.op.reduce_axis) + reduce_block_axis - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i - reduce_block_axis]
                ub_rf_reordered_axis_list.append(reduce_axis)

        # 'if ub split axis in (ak,..a2,a1)
        else:
            # add axis (ak,..a2,a1)
            for i in range(0, ub_split_axis - 1):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])

            ub_rf_reordered_axis_list.append(res_ub_outer)
            ub_rf_reordered_axis_list.append(res_ub_inner)

            for i in range(ub_split_axis + 1, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index = none_reduce_index_map[i]
                    ub_rf_reordered_axis_list.append(self._schedule[out_tensor_ub_rf].op.axis[none_reduce_index +
                                                                                              self._axis_offset])

            # 'rbi
            ub_rf_reordered_axis_list.append(out_tensor_ub_rf.op.reduce_axis[-1])

            # add axis (rb-1,..r2,r1)
            for i in range(0, len(out_tensor_ub_rf.op.reduce_axis) - 1):
                reduce_axis = out_tensor_ub_rf.op.reduce_axis[i]
                ub_rf_reordered_axis_list.append(reduce_axis)

        self._schedule[out_tensor_ub_rf].reorder(*ub_rf_reordered_axis_list)

    @staticmethod
    def _find_none_reduce_axis_map(shape_before_reduce, reduce_axis_index, keep_dims):
        """
        :param shape_before_reduce:
        :param reduce_axis_index:
        :return:
        """
        none_reduce_index_map = {}
        if keep_dims:
            for i in range(0, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index_map[i] = i
        else:
            count = 0
            for i in range(0, len(shape_before_reduce)):
                if i not in reduce_axis_index:
                    none_reduce_index_map[i] = count
                    count += 1

        return none_reduce_index_map

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
        for i in self._mid_tensors:
            if i not in self._mid_output_tensors:
                self._compute_inline_tensors.append(i)

    def _calculate_multi_core(self):
        """
        Calculate fuse and bind axis of multicore

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        if self._need_multi_core:
            self._multi_core_fused_axis = \
                self._final_out_tensor_global.op.reduce_axis[0]
            self._multi_core_bind_tensor = self._final_out_tensor_global

    def _calculate_compute_at(self):
        """
        Calculate the tensor that needs compute at

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """

        self._compute_at_map.clear()

        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]

        for ub_tiling_result in ub_tiling_result_list:
            if "tiling_tensor" not in ub_tiling_result.keys() or \
                    "outer_itervar" not in ub_tiling_result.keys():
                continue
            ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
            res_ub_outer = ub_tiling_result["outer_itervar"]
            if self.reduce_case == 2:
                ub_split_axis = ub_tiling_result["axis"]
                # for shape (r4,a4,r3,a3,r2,a2,r1,a1), if ub split a1, compute at r1
                # when a1 is continous,
                reduce_axis_index = self._reduce_info["reduce_axis_index"]
                shape_before_reduce = self._reduce_info["shape_before_reduce"]
                a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(
                    shape_before_reduce, reduce_axis_index)
                if a1_end_index is None:
                    error_detail = 'errCode:E90001, detailed_cause:a1_end_index can not be none!'
                    error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)
                if a1_start_index <= ub_split_axis <= a1_end_index:
                    if len(ub_tiling_tensor.op.reduce_axis) > 1:
                        res_ub_outer = ub_tiling_tensor.op.reduce_axis[-2]
                    else:
                        res_ub_outer = ub_tiling_tensor.op.reduce_axis[-1]

            for i in self._cache_read_tensors_and_buffer_map:
                read_buffer = self._cache_read_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[read_buffer] = para
            for i in self._cache_write_tensors_and_buffer_map:
                write_buffer = self._cache_write_tensors_and_buffer_map[i]
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[write_buffer] = para
            for i in self._mid_output_tensors:
                para = {"parent": self._schedule[ub_tiling_tensor], "scope": res_ub_outer}
                self._compute_at_map[i] = para

        para = {
            "parent": self._schedule[self._final_out_tensor_global],
            "scope": self._final_out_tensor_global.op.reduce_axis[0]
        }
        self._compute_at_map[self._final_out_tensor_ub_rf] = para


    def _calculate_emit_insn(self):
        """
        Calculate the instruction map of tensor

        Parameters:
        ----------
        None

        Returns
        -------
        None
        """
        self._emit_insn_map.clear()
        ub_tiling_result_list = self._reduce_tiling_result["ub_tiling"]
        ub_tiling_result = ub_tiling_result_list[0]
        ub_tiling_tensor = ub_tiling_result["tiling_tensor"]
        res_ub_inner = ub_tiling_result["inner_itervar"]

        def get_insn(tensor_):
            """
            :param tensor_:
            :return:
            """
            tag = tensor_.op.tag
            if tensor_.op.tag.find("|") != -1:
                insn = tag.split("|")[0]
            else:
                insn = tag
            return Constant.INSN_MAPPING.get(insn, insn)

        ub_split_axis = ub_tiling_result["axis"]
        for i in self._cache_read_tensors_and_buffer_map:
            read_buffer = self._cache_read_tensors_and_buffer_map[i]
            para = {"scope": read_buffer.op.axis[ub_split_axis], "instruction": 'dma_copy'}
            self._emit_insn_map[read_buffer] = para

        for i in self._cache_write_tensors_and_buffer_map:
            write_buffer = self._cache_write_tensors_and_buffer_map[i]
            insn = get_insn(write_buffer)
            para = {"scope": write_buffer.op.axis[0], "instruction": insn}
            self._emit_insn_map[write_buffer] = para

        for out_tensor in self._mid_output_tensors:
            para = {"scope": out_tensor.op.axis[0], "instruction": 'dma_copy'}
            self._emit_insn_map[out_tensor] = para

        # ub_tiling_tensor must be reduce_tensor
        res_tensor = self._res_tensor
        extra_space = self.graph_info.tensor_ub_size_before_reduce

        if self.reduce_case == 2:
            ub_split_axis = ub_tiling_result["axis"]
            # for shape (r4,a4,r3,a3,r2,a2,r1,a1),
            # the ir order (a4,a3,a2,r4,r3,r2,r1,a1)
            # if ub split a2,a3 or a4, emit insn should target at r4
            # when a1 is continous
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            shape_before_reduce = self._reduce_info["shape_before_reduce"]
            a1_start_index, a1_end_index = BNReduceInfo.find_last_none_reduce_axis(shape_before_reduce,
                                                                                   reduce_axis_index)
            if a1_end_index is None:
                error_detail = "errCode:E90001, detailed_cause:a1_end_index can not be none!"
                error_manager_vector.raise_err_specific_reson("bn_training_reduce", error_detail)

            if ub_split_axis < a1_start_index and \
                    ub_split_axis not in reduce_axis_index:
                res_ub_inner = ub_tiling_tensor.op.reduce_axis[-1]

            self._emit_insn_map[ub_tiling_tensor] = {
                "scope": res_ub_inner,
                "instruction": 'vector_reduce_sum',
                "extra_space": extra_space
            }
        elif self.reduce_case == 3:
            reduce_axis_index = self._reduce_info["reduce_axis_index"]
            ub_split_axis = ub_tiling_result["axis"]
            # ub cut ak (none reduce axis),
            if ub_split_axis not in reduce_axis_index:
                self._emit_insn_map[ub_tiling_tensor] = {
                    "scope": ub_tiling_tensor.op.reduce_axis[-1],
                    "instruction": 'vector_reduce_sum',
                    "extra_space": extra_space
                }
            else:
                self._emit_insn_map[ub_tiling_tensor] = {
                    "scope": res_ub_inner,
                    "instruction": 'vector_reduce_sum',
                    "extra_space": extra_space
                }
        else:
            self._emit_insn_map[ub_tiling_tensor] = {
                "scope": res_ub_inner,
                "instruction": 'vector_reduce_sum',
                "extra_space": extra_space
            }

        self._emit_insn_map[self._final_out_tensor_global] = {
            "scope": self._final_out_tensor_global.op.axis[0],
            "instruction": 'dma_copy'
        }
        self._emit_insn_map[res_tensor] = {"scope": self._schedule[res_tensor].op.axis[0], "instruction": 'phony_insn'}

    def _record_reduce_info(self, tensor):
        """
        :param tensor:
        :return:
        """
        self._reduce_info["reduce_axis_index"] = self.reduce_info.reduce_axis_indices
        self._reduce_info["reduce_index_map"] = self.reduce_info.reduce_index_map
        self._reduce_info["reduce_axis_map"] = self.reduce_info.reduce_axis_map
        self._reduce_info["shape_before_reduce"] = self.reduce_info.shape_before_reduce
        self._reduce_info["shape_after_reduce"] = self.reduce_info.shape_after_reduce
        self._reduce_info["shape_before_reduce_expr"] = self.reduce_info.shape_before_reduce
        self._reduce_info["keep_dims"] = True
