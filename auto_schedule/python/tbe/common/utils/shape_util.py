#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Copyright 2019-2020 Huawei Technologies Co., Ltd
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
common function for check ops parameter
"""
from functools import reduce
from itertools import chain
from tbe.common.context import get_context
from tbe.common.utils import para_check
from tbe.common.utils.errormgr import get_error_message
from tbe.dsl.base import operation
from tbe.dsl.base.expr_compare import expr_equal
from tbe.tvm import api as tvm
from tbe.tvm import expr as _expr
from tbe.tvm import make as _make
from tbe.tvm import tensor as _tensor


def squeeze_shape(shape):
    """
    squeeze shape
    """
    squeezed_shape = [i for i in shape if i > 1]
    if not squeezed_shape:
        squeezed_shape = [1]

    return squeezed_shape


def wrap_axes_to_positive(axes, rank):
    """
    wrap axis to positive
    """
    if isinstance(axes, (tuple, list)):
        local_axes = axes
    else:
        local_axes = [axes]
    res_axes = []
    for axis in local_axes:
        if rank <= axis or axis < -rank:
            raise RuntimeError("Axis must between [-%d, %d)." % (rank, rank))
        if axis < 0:
            laxis = axis + rank
        else:
            laxis = axis
        res_axes.append(laxis)

    return res_axes


def refine_shape_axes(shape, axes):
    """
    refine shape and axes for reduce ops, fused reduced axes,
    and fused not reduced axes
    result is a tuple of (shape, axes)
    for example:
        input: shape is (2,3,4,5,6), axes is (1, -3)
        output: (2, 12, 30), (1,)

    Parameters
    ----------
    shape : shape which need refine

    axes : axes which need refine

    Returns
    -------
    shape : list
        refined shape

    axes : list
        refined axes

    """
    if len(shape) == 1:
        return shape, axes
    wrapped_axes = wrap_axes_to_positive(axes, len(shape))
    wrapped_axes = sorted(wrapped_axes)
    refined_axes = []
    reduce_flag = -1
    refined_shape = []
    for idx, dim in enumerate(shape):
        if dim == 1:
            # dim is one, not need reduce skip
            continue
        tmp_flag = 1 if idx in wrapped_axes else 0
        if reduce_flag == 1 and tmp_flag == 1:
            # continues reduce
            refined_shape[-1] *= dim
        elif reduce_flag == 0 and tmp_flag == 0:
            # continues no reduce
            refined_shape[-1] *= dim
        else:
            refined_shape.append(dim)
            if tmp_flag == 1:
                refined_axes.append(idx)
            reduce_flag = tmp_flag

    if not refined_shape:
        refined_shape.append(1)

    return refined_shape, refined_axes


def unify_broadcast_shapes(shapes: list, op_name=para_check.OP_NAME):
    """
    produce broadcast shape
    for example:
        input: shape is [[2, 3], [3, 2, 1], [3, 1, 3]]
        output: [1, 2, 3], [3, 2, 1], [3, 1, 3], [3, 2, 3]

    Parameters
    ----------
    shapes : all input shapes

    op_name : operator name

    Returns
    -------
    shape : list
        completed input shapes and max shape

    """
    # refresh value of OP_NAME after the assignment
    if not op_name:
        op_name = para_check.OP_NAME

    def _greater_one(_value):
        if isinstance(_value, (_expr.IntImm, _expr.UIntImm)):
            return _value.value > 1
        if isinstance(_value, int):
            return _value > 1
        return False

    def _max(_shape):
        no_one_shape = [s for s in _shape if not expr_equal(s, 1)]
        if len(no_one_shape) == 0:
            max_value = 1
        elif len(no_one_shape) == 1:
            max_value = no_one_shape[0]
        else:
            max_value = tvm.max(*no_one_shape)
            for value in no_one_shape:
                if _greater_one(value):
                    max_value = value
                    break
        return max_value

    max_dim_length = max(len(list(shape)) for shape in shapes)
    input_shapes = []
    for shape in shapes:
        input_shapes.append([1] * (max_dim_length - len(shape)) + list(shape))
    input_shapes = list(map(list, zip(*input_shapes)))
    max_shape = [_max(shape) for shape in input_shapes]
    const_type = (_expr.IntImm, _expr.UIntImm, int)
    for value, shape in zip(max_shape, input_shapes):
        if isinstance(value, const_type):
            for _shape in shape:
                if isinstance(_shape, const_type) and not expr_equal(_shape, value) and not expr_equal(_shape, 1):
                    error_info = {
                        'errCode': para_check.OP_ERROR_CODE_013, 'op_name': op_name,
                        'input1_shape': ",".join(str(i) for i in shape),
                        'input2_shape': ",".join(str(i) for i in max_shape)}
                    raise RuntimeError(
                        error_info,
                        "In op[%s], the inputs[%s] could not be broadcast "
                        "together with shapes[%s]."
                        % (op_name, error_info['input1_shape'], error_info['input2_shape']))
    input_shapes = list(map(list, zip(*input_shapes)))
    return (*input_shapes, max_shape)


def broadcast_shapes(shape1, shape2, op_name=para_check.OP_NAME,
                     param_name_input1='', param_name_input2=''):
    """
    two input shapes produce third output shape
    """
    # refresh value of OP_NAME after the assignment
    if not op_name:
        op_name = para_check.OP_NAME

    if operation.in_dynamic():
        return unify_broadcast_shapes([shape1, shape2], op_name)

    shape1 = list(shape1)
    shape2 = list(shape2)
    swapped = False
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        swapped = True

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    out_shape = []
    for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
        if not expr_equal(shape1_i, shape2_i) and \
                (isinstance(shape1_i, int) and shape1_i != 1) \
                and (isinstance(shape2_i, int) and shape2_i != 1):
            error_info = {
                'errCode': para_check.OP_ERROR_CODE_013, 'op_name': op_name,
                'input1_name': param_name_input1,
                'input2_name': param_name_input2,
                'input1_shape': ",".join(str(i) for i in shape1),
                'input2_shape': ",".join(str(i) for i in shape2)}
            raise RuntimeError(
                error_info,
                "In op[%s], the inputs[%s][%s] could not be broadcast "
                "together with shapes[%s][%s]."
                % (op_name, param_name_input1, param_name_input2,
                   error_info['input1_shape'], error_info['input2_shape']))
        out_shape.append(shape1_i if expr_equal(shape2_i, 1) else shape2_i)

    if swapped:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    def _const_refine_shapes_for_broadcast(shape1, shape2):
        def _delete_one(shape1, shape2):
            # delete 1 when both 1
            shape1_new = []
            shape2_new = []
            for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
                if (shape1_i != shape2_i) or \
                        (shape1_i == shape2_i and shape1_i != 1):
                    shape1_new.append(shape1[i])
                    shape2_new.append(shape2[i])
            if shape1_new == [] and shape2_new == []:
                shape1_new = [1]
                shape2_new = [1]
            return shape1_new, shape2_new

        shape1, shape2 = _delete_one(shape1, shape2)

        fused_shape1 = []
        fused_shape2 = []
        fused_shape1.append(shape1[0])
        fused_shape2.append(shape2[0])
        j = 0
        for i, (shape1_i, shape2_i) in enumerate(zip(shape1, shape2)):
            if i == 0:
                pass
            elif shape1_i == shape2_i and shape1[i - 1] == shape2[i - 1]:
                fused_shape1[j] *= shape1[i]
                fused_shape2[j] *= shape2[i]
            elif shape1_i != shape2_i and \
                    shape1[i - 1] != shape2[i - 1] and \
                    (shape1_i == shape1[i - 1] or shape2_i == shape2[i - 1]):
                fused_shape1[j] *= shape1[i]
                fused_shape2[j] *= shape2[i]
            else:
                j += 1
                if i != 0:
                    fused_shape1.append(shape1[i])
                    fused_shape2.append(shape2[i])

        return fused_shape1, fused_shape2

    from tbe.common.buildcfg import get_current_build_config
    if get_current_build_config("enable_op_prebuild"):
        return shape1, shape2

    shape1, shape2 = list(shape1), list(shape2)
    swapped = False
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        swapped = True

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    fused_shape1, fused_shape2 = \
        _const_refine_shapes_for_broadcast(shape1, shape2)

    if swapped:
        fused_shape1, fused_shape2 = fused_shape2, fused_shape1

    return fused_shape1, fused_shape2


def _get_input_range_nchw(op_type, in_shape, in_format, in_range):
    """
    get input range of nchw format
    """
    pos_n = in_format.find('N')
    pos_c = in_format.find('C')
    pos_h = in_format.find('H')
    pos_w = in_format.find('W')

    n_dim = 0
    h_dim = 2
    w_dim = 3
    format_nchw_dim = 4
    format_nc1hwc0_dim = 5

    if len(in_range) == format_nchw_dim:
        in_range = [in_range[pos_n], in_range[pos_c], in_range[pos_h], in_range[pos_w]]
    # range in NC1HWC0 format sometimes
    elif len(in_range) == format_nc1hwc0_dim:
        in_range = [in_range[n_dim], (in_shape[pos_c], in_shape[pos_c]), in_range[h_dim], in_range[w_dim]]
    else:
        err_man.raise_err_specific_user(op_type, "dimension of range should be 4 or 5.")
    for r in in_range:
        if not isinstance(r, (tuple, list)):
            err_man.raise_err_specific_user(op_type, "each dim of range must be tuple or list.")
    return [tuple(r) if r else r for r in in_range]


def _produce_convout_shape(inputs: list):
    """
    produce another input shape for conv + add fusion
    """
    shape_out = []
    n_dim = 0
    h_dim = 2
    w_dim = 3
    hw_dim = 2
    dynamic_flag = -1
    for input_x in inputs:
        in_shape = input_x.get("shape")
        ori_shape = list(input_x.get("ori_shape"))
        ori_format = input_x.get("ori_format")
        ori_range = input_x.get("ori_range")
        in_range_nchw = _get_input_range_nchw("conv2d", ori_shape, ori_format, ori_range)

        if input_x.get("format") != "NC1HWC0":
            return []
        if in_shape[n_dim] == dynamic_flag:
            if operation.get_te_var("batch_n"):
                in_shape[n_dim] = operation.get_te_var("batch_n").get_tvm_var()
            else:
                in_shape[n_dim] = operation.var("batch_n", in_range_nchw[n_dim])
                operation.add_exclude_bound_var(in_shape[n_dim])
        if in_shape[hw_dim] == dynamic_flag:
            if operation.get_te_var("ho"):
                ho_var = operation.get_te_var("ho").get_tvm_var()
            else:
                ho_var = operation.var("ho", in_range_nchw[h_dim])
                operation.add_exclude_bound_var(ho_var)
            if operation.get_te_var("wo"):
                wo_var = operation.get_te_var("wo").get_tvm_var()
            else:
                wo_var = operation.var("wo", in_range_nchw[w_dim])
                operation.add_exclude_bound_var(wo_var)
            in_shape[hw_dim] = ho_var*wo_var
        shape_out.append(in_shape)
    return shape_out


def _cube_variable_shape(inputs: list):
    shape_out = []
    n_dim = 0
    h_dim = 2
    w_dim = 3
    hw_dim = 2
    dynamic_flag = -1
    unknown_flag = -2

    groups = 1
    for op_info in get_context().get_op_info():
        if op_info.pattern == "Convolution":
            groups = op_info.attrs[3]
            break

    if inputs and inputs[0].get("input_pattern") != "cube":
        # vector inputs process for conv + eltwise（double input）fusion
        return _produce_convout_shape(inputs)

    weight_ori_format = inputs[1].get("ori_format")
    c_index = weight_ori_format.index("C")
    weight_cin = inputs[1].get("ori_shape")[c_index]

    fmap_cin = weight_cin * groups

    for i, input in enumerate(inputs):
        if i == 0:
            ori_shape = list(input.get("ori_shape"))
            ori_format = input.get("ori_format")
            in_range = input.get("range")
            if ori_shape == [unknown_flag]:
                in_range_nchw = [(1, None), (1, 1), (1, None), (1, None)]
            else:
                in_range_nchw = _get_input_range_nchw("cube", ori_shape, ori_format, in_range)

            in_shape = input.get("shape")
            if input.get("format") != "NC1HWC0":
                return []
            if in_shape[1] == -1:
                fmap_ci0 = in_shape[-1]
                in_shape[1] = (fmap_cin + fmap_ci0 - 1) // fmap_ci0
            if in_shape[n_dim] == dynamic_flag:
                in_shape[n_dim] = operation.var("batch_n", in_range_nchw[n_dim])
                operation.add_exclude_bound_var(in_shape[n_dim])
            if in_shape[h_dim] == dynamic_flag:
                in_shape[h_dim] = operation.var("fmap_h", in_range_nchw[h_dim])
                operation.add_exclude_bound_var(in_shape[h_dim])
            if in_shape[w_dim] == dynamic_flag:
                in_shape[w_dim] = operation.var("fmap_w", in_range_nchw[w_dim])
                operation.add_exclude_bound_var(in_shape[w_dim])

            shape_out.append(in_shape[:])
        else:
            shape_out.append(input.get("shape")[:])

    return shape_out


def _dw_variable_shape(inputs: list):
    """
    get the input shape of fusion node trans_data_trans_data_conv2d_backprop_filter
    """
    shape_out = []
    dedx_index, dedy_index = 0, 1
    dedx_ori_format = inputs[dedx_index].get("ori_format")
    n_index = dedx_ori_format.index("N")
    c_index = dedx_ori_format.index("C")
    h_index = dedx_ori_format.index("H")
    w_index = dedx_ori_format.index("W")
    batch_var = operation.var("batch", [1, None])
    for i, input_member in enumerate(inputs):
        input_shape = input_member.get("shape")
        if i in (dedx_index, dedy_index):
            input_name = "fmap" if i == dedx_index else "dedy"
            input_shape[n_index] = batch_var
            input_shape[c_index] = operation.var("{}_c".format(str(input_name)), [1, None])
            input_shape[h_index] = operation.var("{}_h".format(str(input_name)), [1, None])
            input_shape[w_index] = operation.var("{}_w".format(str(input_name)), [1, None])
            operation.add_exclude_bound_var(input_shape[n_index])
            operation.add_exclude_bound_var(input_shape[c_index])
            operation.add_exclude_bound_var(input_shape[h_index])
            operation.add_exclude_bound_var(input_shape[w_index])
        shape_out.append(input_shape[:])
    return shape_out


def _get_input_no_range_dx(in_shape, in_format, in_range):
    """
    get input no range
    """
    tmp = in_shape[:]
    format_fractal_z_dim = 4
    format_nc1hwc0_dim = 5
    if in_format == "FRACTAL_Z" and len(in_range) == format_fractal_z_dim:
        axis_ci1hw, axis_co1 = 0, 1
        # no add_exclude_bound_var for parallel compilation ERROR
        tmp[axis_ci1hw] = operation.var("filter_ci1hw")
        tmp[axis_co1] = operation.var("filter_co1")
    elif in_format == "NC1HWC0" and len(in_range) == format_nc1hwc0_dim:
        axis_n, axis_c1, axis_h, axis_w = 0, 1, 2, 3
        tmp[axis_n] = operation.var("dedy_n")
        tmp[axis_c1] = operation.var("dedy_c1")
        tmp[axis_h] = operation.var("dedy_h")
        tmp[axis_w] = operation.var("dedy_w")
    return tmp


def _dx_variable_shape(inputs: list):
    """
    get no range variable shape of conv2d_backprop_input
    """
    shape_out = []
    for input_member in inputs:
        in_shape = input_member.get("shape")
        in_format = input_member.get("format")
        in_range = input_member.get("range")
        in_shape_var = _get_input_no_range_dx(in_shape, in_format, in_range)
        shape_out.append(in_shape_var[:])
    return shape_out


def _transpose_variable_shape(inputs: list):
    if len(inputs) != 1:
        dict_args = {"errCode": "E90001", "detailed_cause": "input numbers error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    shape_x = inputs[0].get("shape")
    range_x = inputs[0].get("range")
    shape_out = []
    for i, x in enumerate(shape_x):
        if x == -1:
            _var = operation.var_inner(f"_dim_{i}", range_x[i])
            shape_out.append(_var)
        else:
            shape_out.append(x)
    return shape_out


def _gather_variable_shape(inputs):

    params_info = inputs[0]
    indices_info = inputs[1]

    gather_mode = operation.get_context().get("_gather_mode")

    if gather_mode == "gather":
        axis_info = inputs[2]
        batch_dims_info = inputs[3]
        rank_info = 1
    else:
        axis_info = 0
        batch_dims_info = inputs[2]
        rank_info = indices_info["shape"][-1]

    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_axis", axis_info)
    current_compute.add("_rank", rank_info)
    current_compute.add("_params_shape", params_info["shape"])
    current_compute.add("_indices_shape", indices_info["shape"])

    # zeros shape condition
    if 0 in chain(list(params_info["shape"]) + list(indices_info["shape"][:-1])):
        if gather_mode == "gather":
            return params_info["shape"], indices_info["shape"], axis_info, batch_dims_info
        return params_info["shape"], indices_info["shape"], batch_dims_info

    # new params var
    params_shape = []
    for index, value in enumerate(params_info["shape"]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_params_dim_{}".format(index), params_info["range"][index])
            params_shape.append(_var)
        else:
            params_shape.append(value)

    # new indices var
    indices_shape = []
    # batch dims must same
    indices_shape.extend(params_shape[:batch_dims_info])
    for index, value in enumerate(indices_info["shape"][batch_dims_info:]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_indices_dim_{}".format(index + batch_dims_info),
                                       indices_info["range"][index + batch_dims_info])
            indices_shape.append(_var)
        else:
            indices_shape.append(value)

    if gather_mode == "gather":
        return params_shape, indices_shape, axis_info, batch_dims_info
    return params_shape, indices_shape, batch_dims_info


def _slice_variable_shape(inputs: list):
    x_info = inputs[0]
    begin_info = inputs[1]
    end_info = inputs[2]

    current_compute = operation.get_context().get_current_compute()
    if 0 in x_info["shape"]:
        current_compute.add("_zero_shape", True)
    else:
        current_compute.add("_zero_shape", False)

    # new x var
    x_shape = []
    for index, value in enumerate(x_info["shape"]):
        _var = None
        if value == -1:
            _var = operation.var_inner("_x_dim_{}".format(index), x_info["range"][index])
            x_shape.append(_var)
        else:
            x_shape.append(value)

    dim_len = len(x_info["shape"])
    begin_list = []
    is_begin_list = isinstance(begin_info, list)
    end_list = []
    is_end_list = isinstance(end_info, list)
    for _idx in range(dim_len):
        if is_begin_list:
            begin_value = begin_info[_idx]
        else:
            begin_value = operation.var_inner("_begin_dim_{}".format(_idx), (1, None))
        begin_list.append(begin_value)

        if is_end_list:
            end_value = end_info[_idx]
        else:
            end_value = begin_value + operation.var_inner("_size_dim_{}".format(_idx), (1, None))
        end_list.append(end_value)

    return x_shape, begin_list, end_list


def _transdata_variable_shape(inputs: list):
    """
    Eg:
    src is [30, 1024]
    dst is [64, 32, 16]
    map is {0: [1, ],
            1: [0, 2]}
    In map has three model: int:int, tuple:int, int:tuple
    int:int is do nothing
    tuple:int is do de-pad
    int:tuple is do pad
    Return src_shape, dst_shape
    """
    if len(inputs) != 3:
        dict_args = {"errCode": "E90001", "detailed_cause": "inputs' size error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))

    def is_const_model(_input):
        return -1 not in _input

    def reversed_map(_map):
        result = {}
        for key, value in _map.items():
            value = tuple(value) if isinstance(value, list) else value
            key = list(key) if isinstance(key, tuple) else key
            result[value] = key
        return result

    def init_shape(_input):
        if is_const:
            return _input

        result = []
        for k, v in enumerate(_input):
            # set pad-dim-n as var but not 1 while dynamic
            result.append(v if v != -1 and [k, v] != [0, 1] else operation.var_inner(f"_dim_{k}", [1, None]))
        return result

    def src_infer_dst(in_shape, out_shape, _map):
        for key, value in enumerate(in_shape):
            map_value = _map.get(key)
            if isinstance(map_value, int):
                out_shape[map_value] = value
            elif isinstance(map_value, (list, tuple)):
                if len(map_value) == 1:
                    out_shape[map_value[0]] = tvm.floordiv(value + pad_factor - 1, pad_factor) * pad_factor
                else:
                    out_shape[map_value[0]] = tvm.floordiv(value + pad_factor - 1, pad_factor)
                    out_shape[map_value[1]] = pad_factor

        return out_shape

    def infer_shape_main(_input, _output, _map):
        _input = init_shape(_input)
        _output = src_infer_dst(_input, _output, _map)
        return _input, _output

    axes_map = inputs[2]
    is_forward = inputs[0].get("is_forward")
    src_shape = inputs[0].get("shape")
    dst_shape = inputs[1]

    if not is_forward:
        src_shape, dst_shape = dst_shape, src_shape
        axes_map = reversed_map(axes_map)

    current_compute = operation.get_context().get_current_compute()
    # transdata_category that help to choose different computation.
    # 32bit-Tensor would be reinterpret as 16bit-Tensor in schedule(ori_bit).
    pad_factor = operation.get_compile_info().get("_pad_factor")
    is_const = is_const_model(src_shape)
    current_compute.add("_pad_factor", pad_factor)
    current_compute.add("_const_model", is_const)
    current_compute.add("_transdata_category", inputs[0].get("transdata_category", None))
    current_compute.add("_ori_bit", inputs[0].get("ori_bit", None))

    src_shape, dst_shape = infer_shape_main(src_shape, dst_shape, axes_map)
    if not is_forward:
        src_shape, dst_shape = dst_shape, src_shape

    return src_shape, dst_shape


def _concat_variable_shape(inputs: list):
    def add_zero_axis_var(_is_first_in, _first_var):
        if shape_x[0] == -1:
            if _is_first_in:
                _first_var = operation.var_inner(f"_dim_{index}_0", range_x[0])
                _is_first_in = False
            cur_shape.append(_first_var)
        else:
            cur_shape.append(shape_x[0])
        return _is_first_in, _first_var

    def add_one_axis_var():
        if shape_x[1] == -1:
            _var = operation.var_inner(f"_dim_{index}_1", range_x[1])
            cur_shape.append(_var)
        else:
            cur_shape.append(shape_x[1])

    if len(inputs) != 1:
        dict_args = {"errCode": "E90001", "detailed_cause": "concat input numbers error"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    is_first_in = True
    shape_out = []
    first_var = -1
    for index, x in enumerate(inputs[0]):
        shape_x = x.get("shape")
        range_x = x.get("range")
        cur_shape = []
        is_first_in, first_var = add_zero_axis_var(is_first_in, first_var)
        if len(shape_x) > 1:
            add_one_axis_var()
        shape_out.append(cur_shape)
    current_compute = operation.get_context().get_current_compute()
    mode = inputs[0][0].get("mode")
    current_compute.add("_mode", mode)
    return shape_out


def _split_variable_shape(inputs: list):
    def check_params():
        if len(inputs) != 2:
            dict_args = {"errCode": "E90001", "detailed_cause": "split variable shape requires two input parameters:"
                         "input tensor and split numbers"}
            raise RuntimeError(dict_args, get_error_message(dict_args))
        split_size = inputs[1]
        split_input_number_limit = 63
        if len(split_size) > split_input_number_limit or len(split_size) <= 0:
            dict_args = {
                "errCode": "E90001",
                "detailed_cause": f"split numbers error, split numbers must be "
                                  f"greater 0 and less equal {split_input_number_limit}, now, it is {split_num}"}
            raise RuntimeError(dict_args, get_error_message(dict_args))

    def shape_variable():
        input_vars = []
        for index, (shape_value, range_value) in enumerate(zip(shape_x, range_x)):
            if shape_value == -1:
                _var = operation.var_inner(f"_dim_{index}", range_value)
                input_vars.append(_var)
            else:
                input_vars.append(shape_value)
            if index == 0:
                current_compute.add("_ori_var", input_vars[-1])
                input_vars[-1] = (input_vars[-1] + split_factor - 1) // split_factor * split_factor
        return input_vars

    def get_split_range(r_value, split_num):
        if r_value is None:
            new_r_value = None
        elif r_value < split_num:
            new_r_value = 1
        else:
            new_r_value = r_value // split_num
        return new_r_value

    def split_value_variable():
        avg_split = operation.get_context().get("_avg_split")
        split_num = len(inputs[1])
        if avg_split:
            if shape_x[1] > -1:
                _var = shape_x[1] // split_num
                current_compute.add("_split_is_const", True)
            else:
                low_r = get_split_range(range_x[1][0], split_num)
                high_r = get_split_range(range_x[1][1], split_num)
                _var = operation.var_inner("_split_0", (low_r, high_r))
            split_vars = [_var for _ in range(split_num)]
        else:
            if split_num == 0 and shape_x[1] > -1:
                split_vars = [shape_x[1]]
            elif min(inputs[1]) > -1:
                return list(inputs[1])
            else:
                split_vars = []
                for index in range(split_num):
                    _var = operation.var_inner(f"_split_{index}", (min(1, range_x[1][0]), range_x[1][1]))
                    split_vars.append(_var)
        return split_vars

    check_params()

    input_x = inputs[0]
    shape_x = shape_to_list(input_x.get("shape"))
    if len(shape_x) != 2:
        dict_args = {"errCode": "E90001", "detailed_cause": "split variable shape must be two dims"}
        raise RuntimeError(dict_args, get_error_message(dict_args))
    range_x = input_x.get("range")
    mode = input_x.get("mode")
    split_factor = input_x.get("split_factor")
    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_mode", mode)

    return shape_variable(), split_value_variable()


def _tuple_reduce_variable_shape(inputs: list):
    """
    variable shape for tuple_reduce ops
    :param inputs: all inputs
    :return:
    """
    shapes = [tensor.get("shape") for tensor in inputs]
    shapes_transpose = list(map(list, zip(*shapes)))
    tuple_reduce_vars = [operation.var_inner("_dim_{}".format(j), (1, None))
                         if -1 in col else None for j, col in enumerate(shapes_transpose)]
    return [[tuple_reduce_vars[j] if v == -1 else v for j, v in enumerate(_shape)] for _shape in shapes]


def variable_shape(inputs: list, op_mode="elewise"):
    """
    :param inputs: all inputs
    :param op_mode: operator mode, default is elewise
    :return:
    """
    op_type = ""
    if len(inputs) > 0 and isinstance(inputs[0], dict):
        op_type = inputs[0].get("input_op_type", "")
    if op_type == "Conv2DBackpropInput":
        return _dx_variable_shape(inputs)

    if op_mode == "conv2d_backprop_filter":
        return _dw_variable_shape(inputs)

    if op_mode == "cube":
        return _cube_variable_shape(inputs)

    if op_mode == "reduce":
        return _reduce_variable_shape(inputs)
		
    if op_mode == "tuple_reduce":
        return _tuple_reduce_variable_shape(inputs)

    if op_mode == "norm":
        return _norm_variable_shape(inputs)

    if op_mode == "gather":
        return _gather_variable_shape(inputs)

    if op_mode == "slice":
        return _slice_variable_shape(inputs)

    if op_mode == "transpose":
        return _transpose_variable_shape(inputs)

    if op_mode == "transdata":
        return _transdata_variable_shape(inputs)

    if op_mode == "concat":
        return _concat_variable_shape(inputs)

    if op_mode == "split":
        return _split_variable_shape(inputs)

    return _elewise_varibale_shape(inputs, op_mode)


def _elewise_varibale_shape(inputs: list, op_mode="elewise"):
    def _get_range_intersection(ranges):
        def _range_intersection(range_a, range_b):
            if range_a is None or range_b is None:
                return None
            a_lower, a_upper = range_a
            b_lower, b_upper = range_b
            if max(a_lower, b_lower) > min(a_upper, b_upper):
                return None
            return max(a_lower, b_lower), min(a_upper, b_upper)

        return reduce(_range_intersection, ranges)

    def _update_range(shapes, ranges):
        def _fixed_shape_range(shapes, ranges):
            for _range in ranges:
                for i, (r0, r1) in enumerate(_range):
                    if r0 is None and r1 is None:
                        _range[i] = (para_check.MAX_UNKNOWN_SHAPE_NUM, para_check.MAX_UNKNOWN_SHAPE_NUM)
                    elif r0 is None:
                        _range[i] = (para_check.MAX_UNKNOWN_SHAPE_NUM, r1)
                    elif r1 is None:
                        _range[i] = (r0, para_check.MAX_UNKNOWN_SHAPE_NUM)
            for _shape, _range in zip(shapes, ranges):
                for i, (s, (r0, r1)) in enumerate(zip(_shape, _range)):
                    if s != -1:
                        _range[i] = (s, s)
                    elif r0 == r1:
                        _shape[i] = r0

        _fixed_shape_range(shapes, ranges)
        t_shapes = list(map(list, zip(*shapes)))
        t_ranges = list(map(list, zip(*ranges)))
        for _shape, _range in zip(t_shapes, t_ranges):
            no_one_range = [r for r in _range if r[0] > 1]
            if len(no_one_range) > 0:
                mied_range = _get_range_intersection(no_one_range)
                if mied_range is None:
                    dict_args = {}
                    dict_args["errCode"] = "E90001"
                    dict_args["detailed_cause"] = "input shape error, shape range no intersection"
                    raise RuntimeError(dict_args, get_error_message(dict_args))
                for i, r in enumerate(_range):
                    if 1 in r:
                        if r[1] < mied_range[0]:
                            _range[i] = (1, 1)
                        elif r[1] > mied_range[1]:
                            _range[i] = (1, mied_range[1])
                    else:
                        _range[i] = mied_range
        shapes = list(map(list, zip(*t_shapes)))
        ranges = list(map(list, zip(*t_ranges)))
        _fixed_shape_range(shapes, ranges)
        return shapes, ranges

    def _get_dim(_i, _shapes):
        return max(s[_i] for s in _shapes)

    def _extract(_inputs):
        def _complete(_in):
            shapes, ranges = [], []
            for x in _in:
                _shape, _range = list(x["shape"]), x.get("range")
                d_v = dim_length - len(_shape)
                x_shape = [1] * d_v + _shape
                x_range = [(1, 1)] * d_v + list(_range)
                shapes.append(x_shape)
                ranges.append(x_range)
            return shapes, ranges

        if support_broadcast:
            dim_length = max(len(s["shape"]) for s in _inputs)
            shapes, ranges = _complete(_inputs)
            shapes, ranges = _update_range(shapes, ranges)
            return shapes, ranges

        _shapes, _ranges = [], []
        for _input in inputs:
            _shapes.append(_input["shape"])
            _ranges.append(_input["range"])
        _shape = [_get_dim(_i, _shapes) for _i in range(len(_shapes[0]))]
        _shapes = [_shape.copy() for _ in range(len(_shapes))]

        return _shapes, _ranges

    def _maybe_broadcast():
        if support_broadcast:
            for _r in ranges:
                if _r[i][0] <= 1:
                    return True
        return False

    def _mode_process():
        if mode == para_check.CONST:
            if support_broadcast:
                input1 = inputs[0]["const_shape"]
                input2 = inputs[1]["const_shape"]
                const_shape = [a & b for a, b in zip(input1, input2)]
            else:
                const_shape = inputs[0]["shape"]
            operation.get_context().get_current_compute().add("_const_shape", const_shape)
        elif mode == para_check.SPECIAL and inputs[0].get("pattern"):
            pattern = inputs[0].get("pattern")
            operation.get_context().get_current_compute().add("_pattern", pattern)
            for i, _pattern in enumerate(pattern):
                if _pattern == para_check.COMMON:
                    for j in range(len(shapes)):
                        if shapes[j][i] == -1:
                            # mark this dimension dose not exist broadcast
                            shapes[j][i] = -77
        elif mode == para_check.SPECIAL_SCALAR:
            pattern = inputs[0].get("pattern")
            operation.get_context().get_current_compute().add("_pattern", pattern)

    if len(inputs) < 1:
        return []
    mode = inputs[0].get("mode") or para_check.ORIGINAL
    current_compute = operation.get_context().get_current_compute()
    current_compute.add("_mode", mode)
    support_broadcast = operation.get_context().get("_support_broadcast") or False

    shapes, ranges = _extract(inputs)
    _mode_process()

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast()
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1 and _range[i][0] == _range[i][1]:
                d_shape.append(_range[i][0])
            elif shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var_inner("_dim_{}_{}".format(str(i), str(_suffix)), _range[i])
                d_shape.append(_var)
            elif shape[i] == -77:
                # no broadcast
                if _var is None:
                    _var = operation.var_inner("_dim_{}_{}".format(str(i), str(_suffix)), _range[i])
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes


def _reduce_variable_shape(inputs: list):
    """
    variable shape for reduce ops
    """
    inputs_before_reduce, inputs_after_reduce, input_axis = [], [], []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "axis":
            input_axis.append(single_input)
        elif input_type == "after":
            inputs_after_reduce.append(single_input)
        else:
            inputs_before_reduce.append(single_input)

    axis = input_axis[0].get("value")

    if len(inputs) < 1:
        return []
    mode = inputs_before_reduce[0].get("mode")
    if mode is None:
        mode = para_check.ORIGINAL
    operation.get_context().add("_mode", mode)
    current_compute = operation.get_context().get_current_compute()
    if current_compute:
        current_compute.add("_mode", mode)
        current_compute.add("_shape", inputs_before_reduce[0]["shape"])
        ori_axis = input_axis[0].get("ori_axis")
        if ori_axis is not None:
            current_compute.add("_ori_axis", ori_axis)
        axis_dtype = input_axis[0].get("axis_dtype")
        if axis_dtype is not None:
            current_compute.add("_axis_dtype", axis_dtype)

    shape_local = [x["shape"] for x in inputs_before_reduce]
    range_local = [x.get("range") if x.get("range") else [(1, None)]*len(shape_local[0]) for x in inputs_before_reduce]
    shape_before_reduce, shape_after_reduce = [], []
    for index in range(len(shape_local[0])):
        _var = None
        if shape_local[0][index] == -1:
            _var = operation.var_inner("_dim_{}".format(str(index)), range_local[0][index])
            shape_before_reduce.append(_var)
        else:
            shape_before_reduce.append(shape_local[0][index])

    def _gen_shape_after_reduce():
        for idx in range(len(shape_before_reduce)):
            if idx in axis:
                if not len(inputs_after_reduce[0]["shape"]) == len(inputs_before_reduce[0]["shape"]):
                    continue
                shape_after_reduce.append(1)
            else:
                shape_after_reduce.append(shape_before_reduce[idx])

    if inputs_after_reduce:
        _gen_shape_after_reduce()

    shape_out = []
    for single_input in inputs:
        input_type = single_input.get("rel_pos_to_reduce")
        if input_type == "axis":
            shape_out.append(input_axis[0].get("shape")[:])
        elif input_type == "after":
            shape_out.append(shape_after_reduce[:])
        else:
            shape_out.append(shape_before_reduce[:])

    return shape_out


def _norm_variable_shape(inputs: list):
    """
    variable shape for norm ops
    """
    def _gen_variable_inputs():
        _variable_outputs = []
        _output_non_duplicate_list = []
        # all inputs are unknown broadcast, vars according to shape_to_variable do not need to generate
        # unknown_broadcast: dim + input_type + index
        # others:  dim + index
        if mode_non_duplicate_list == ["broadcast_unknown"] * len(mode_non_duplicate_list):
            for _idx, _input_type in enumerate(input_type_non_duplicate_list):
                _shape = input_non_duplicate_list[_idx].get("shape")
                _range = input_non_duplicate_list[_idx].get("range")
                _single_output = []
                for _index in range(len(_shape)):
                    if _shape[_index] == -1:
                        _var = operation.var_inner(f"_dim_{_input_type}_{_index}", _range[_index])
                        _single_output.append(_var)
                        continue
                    _single_output.append(_shape[_index])
                _output_non_duplicate_list.append(_single_output)
        else:
            _shape_after_variable = []
            for _index in range(len(shape_to_variable)):
                if shape_to_variable[_index] == -1:
                    _var = operation.var_inner(f"_dim_{_index}", range_to_variable[_index])
                    _shape_after_variable.append(_var)
                    continue
                _shape_after_variable.append(shape_to_variable[_index])
            for _idx, _input_type in enumerate(input_type_non_duplicate_list):
                _shape = input_non_duplicate_list[_idx].get("shape")
                _single_output = []
                if mode_non_duplicate_list[_idx] == "broadcast_unknown":
                    _range = input_non_duplicate_list[_idx].get("range")
                    for _index in range(len(_shape)):
                        if _shape[_index] == -1:
                            _var = operation.var_inner(f"_dim_{_input_type}_{_index}", _range[_index])
                            _single_output.append(_var)
                            continue
                        _single_output.append(_shape[_index])
                    continue
                for _index in range(len(_shape)):
                    if _shape[_index] == -1:
                        _single_output.append(_shape_after_variable[_index])
                        continue
                    _single_output.append(_shape[_index])

                _output_non_duplicate_list.append(_single_output)

        for _, _input_type in enumerate(input_type_list):
            _variable_outputs.append(_output_non_duplicate_list[input_type_non_duplicate_list.index(_input_type)])

        return _variable_outputs

    shape_len = len(inputs[0].get("shape"))
    shape_to_variable = [-1] * shape_len
    range_to_variable = [(1, None) for _ in range(shape_len)]

    broadcast_axis = None
    exist_after_broadcast_input = False

    input_type_list = []
    # input type are not duplicated
    input_type_non_duplicate_list = []
    input_non_duplicate_list = []
    mode_non_duplicate_list = []

    norm_pattern = inputs[0].get("norm_pattern")

    for single_input in inputs:
        input_type = single_input.get("input_type")
        input_type_list.append(input_type)
        input_shape = single_input.get("shape")
        mode_str = single_input.get("mode")

        if mode_str == "broadcast_axis_known":
            broadcast_axis = single_input.get("broadcast_axis")

        if input_type not in input_type_non_duplicate_list:
            input_type_non_duplicate_list.append(input_type)
            mode_non_duplicate_list.append(mode_str)
            input_non_duplicate_list.append(single_input)

        if input_type == 0:
            # normalize shape_to_variable according to after broadcast inputs
            shape_to_variable = input_shape
            range_to_variable = single_input.get("range")
            exist_after_broadcast_input = True

    # normalize shape_to_variable according to before broadcast inputs when there is no after broadcast input
    if not exist_after_broadcast_input:
        for idx in range(shape_len):
            input_dim = [x.get("shape")[idx] for x in inputs]
            max_dim = max(input_dim)
            min_dim = min(input_dim)
            if max_dim > 1 or (max_dim == min_dim == 1):
                shape_to_variable[idx] = max_dim
                range_to_variable[idx] = (max_dim, max_dim)

    variable_inputs = _gen_variable_inputs()

    current_operator = operation.get_context()
    if current_operator:
        current_compute = current_operator.get_current_compute()
        if current_compute:
            # ori inputs of ops
            current_compute.add("_input_shapes", variable_inputs)
            if broadcast_axis is not None:
                current_compute.add("_broadcast_axis", broadcast_axis)
            current_compute.add("_norm_pattern", norm_pattern)

    return variable_inputs


def simplify_axis_shape(shape, axis):
    """
    simplify the shape and aixs
    """
    axis1 = []
    shape1 = []
    merge_num = 0
    length = shape[0]

    for i in range(len(axis)):
        if axis[i] < 0:
            axis_i = axis[i] + length(shape)
            if axis_i < 0:
                dict_args = {"errCode": "E90001",
                             "detailed_cause": "input axis[%d] is %d, which is out of range." % (i, axis[i])}
                raise RuntimeError(dict_args, get_error_message(dict_args))
            axis[i] = axis_i
        if i == 0:
            length = shape[axis[0]]
            axis1.append(axis[0])
        else:
            if axis[i] - axis[i - 1] == 1:
                length = length*shape[axis[i]]
                merge_num = merge_num + 1
            else:
                shape1.append(length)
                for j in range(axis[i - 1], axis[i] - 1):
                    shape1.append(shape[j + 1])
                axis1.append(axis[i] - merge_num)
                length = shape[axis[i]]
    shape1.append(length)
    if axis1 == []:
        axis1 = [0]
    else:
        shape1 = list(shape[:axis[0]]) + shape1 + list(shape[axis[-1] + 1:])

    shape_final = []
    axis_final = []
    axis_fuse_sum = 0
    pre_axis = -1
    for axes in axis1:
        shape_noreduce = shape1[pre_axis + 1 : axes]
        if len(shape_noreduce) > 1:
            shape_final.append(reduce(lambda x, y:x*y, shape_noreduce))
        else:
            shape_final += shape_noreduce

        if len(shape_noreduce) > 0:
            axis_fuse_sum += len(shape_noreduce) - 1
        axis_final.append(axes - axis_fuse_sum)
        shape_final.append(shape1[axes])
        pre_axis = axes

    shape_noreduce = shape1[pre_axis + 1:]
    if len(shape_noreduce) > 1:
        shape_final.append(reduce(lambda x, y: x*y, shape_noreduce))
    else:
        shape_final += shape_noreduce

    return shape_final, axis_final


def shape_refine(shape, reduce_axis=None, keep_dims=True):
    """
    refine shape to drop 1 in shape according to reduce axis,
    if input is just shape, result is shape, and if inputs are shape and axis,
    result is a tuple of (shape, axis)

    Parameters
    ----------
    shape : shape of data

    reduce_axis : list, tuple or int
        axis want to reduce

    keepdims: if keepdims = True, we should not refine the shape

    Returns
    -------
    shape : list
        refined shape

    reduce_axis : list
        if input parameters send reduce axis, this will be the output.
        if all the reduce axis is illegal like the length of reduce axis is 1,
        a empty list([]) will be returned.

    """

    def __refine_shape_no_reduce(shape_local):
        refined_shape = [i for i in shape_local if i > 1]
        if not refined_shape:
            refined_shape = [1]
        return refined_shape

    res_shape = []
    res_reduce_axis = []
    if reduce_axis is not None:
        # if the reduce axis correspond to shape[axis] is 1,
        # we can not refine the shape,or the reduce axis will be wrong
        if not check_reduce_need_refine(shape, reduce_axis, keep_dims):
            if hasattr(reduce_axis, 'index'):
                return shape, reduce_axis
            return shape, [reduce_axis]

        if isinstance(reduce_axis, (tuple, list)):
            res_reduce_axis = reduce_axis[:]
        else:
            res_reduce_axis = [reduce_axis]
        res_reduce_axis = sorted(refine_axis(reduce_axis, shape))
        if not res_reduce_axis:
            return __refine_shape_no_reduce(shape), []
        res_shape = shape[:]
        refined_shape = []
        count = 0
        for i in res_shape:
            if i > 1:
                refined_shape.append(i)
                count += 1
            else:
                for j in range(len(res_reduce_axis)):
                    if res_reduce_axis[j] > count:
                        res_reduce_axis[j] -= 1

        return refined_shape, res_reduce_axis
    return __refine_shape_no_reduce(shape)


def refine_axis(axis, shape):
    """
    refine axis

    Parameters
    ----------
    axis :
        axis want to reduce

    shape : shape of data

    Returns
    -------
    res_reduce_axis : list
        refined axis
    """
    if isinstance(axis, (tuple, list)):
        local_axis = axis
    else:
        local_axis = [axis]
    res_axis = []
    shape_len = len(shape)
    for i in local_axis:
        if i < 0:
            laxis = shape_len + i
        else:
            laxis = i
        if (laxis >= shape_len) or (laxis < 0):
            raise RuntimeError("wrong axis.")
        res_axis.append(laxis)
    res_reduce_axis = []
    for i in res_axis:
        if shape[i] > 1:
            res_reduce_axis.append(i)
    return res_reduce_axis


def _axis_value_type_check(shape_len, value):
    """
    Check the value of the axis
    """
    if type(value) != int:
        raise RuntimeError("type of axis value should be int")
    if value >= shape_len or value < -shape_len:
        raise RuntimeError(
            "input axis is out of range, axis value can be from %d to %d"
            % (-shape_len, shape_len - 1))
    if value < 0:
        value = shape_len + value
    return value


def axis_check(shape_len, axis):
    """
    Check the value of axis and return the sorted axis
    """
    if not hasattr(axis, 'index'):
        axis = _axis_value_type_check(shape_len, axis)
        return axis
    for i in range(len(axis)):
        axis[i] = _axis_value_type_check(shape_len, axis[i])

    axis = list(set(axis))
    axis.sort()
    return axis


def produce_shapes(shape1, shape2):
    """
    two input shapes produce three output shape
    """
    shape1 = list(shape1)
    shape2 = list(shape2)
    flag = 0
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        flag = 1

    output_shape_len = len(shape1)
    dec = output_shape_len - len(shape2)
    for i in range(dec):
        shape2 = [1] + shape2

    out_shape = []
    for i in range(output_shape_len):
        if (shape1[i] != shape2[i]) and (shape1[i] != 1) and (shape2[i] != 1):
            raise RuntimeError("input shapes not match!")
        out_shape.append(shape1[i] if shape1[i] > shape2[i] else shape2[i])

    if flag == 1:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def check_reduce_need_refine(shape, reduce_axis, keep_dims):
    """
    # if the reduce axis correspond to shape[axis] is 1,
    we can not refine the shape,or the reduce axis will be wrong
    shape : shape of data

    reduce_axis : list, tuple or int  axis want to reduce

    :return: True or False
    """

    # if the reduce axis correspond to shape[axis] is 1,
    # we can not refine the shape,or the reduce axis will be wrong
    if hasattr(reduce_axis, 'index'):
        if not keep_dims:
            for i in reduce_axis:
                if shape[i] != 1:
                    return True
            return False

        for i in reduce_axis:
            if shape[i] == 1:
                return False
    else:
        if shape[reduce_axis] == 1:
            return False

    return True


def scalar2tensor_one(shape):
    """
    if the input_shape is [],convert the input_shape to [1]
    ----------
    shape: shape of input tensor

    Returns
    -------
    list:[1]
    """
    if isinstance(shape, (list, tuple)):
        if not shape:
            return [1]
    return shape


def axis_transform_5d(axis, data_format):
    """
    4d format axis to 5d mapping
    """
    if data_format == "NCHW":
        if axis < 0:
            axis = axis - 1
    elif data_format == "NHWC":
        if axis == -4:
            axis = -5
        elif axis == -1:
            axis = -4
        elif axis == 1:
            axis = 2
        elif axis == 2:
            axis = 3
        elif axis == 3:
            axis = 1
    return axis


def compare_tensor_dict_key(dict1, dict2, dict_key):
    """
    compare the key value between dict1 and dict2,
    the value is not equal, will raise error

    Parameters
    ----------
    dict1: dict
        input dict1
    dict2: dict
        input dict2
    dict_key: str
        the key that will be compare

    Returns
    -------
    None
    """
    if not isinstance(dict1, dict):
        raise RuntimeError("the input dict1 is not dict")
    if not isinstance(dict2, dict):
        raise RuntimeError("the input dict2 is not dict")

    if dict_key not in dict1.keys():
        raise RuntimeError("There is no value for this input type,"
                           "please check the input!")
    if dict_key not in dict2.keys():
        raise RuntimeError("There is no value for this input type,"
                           "please check the input")

    value1 = dict1.get(dict_key)
    value2 = dict2.get(dict_key)

    if isinstance(value1, (list, tuple)):
        value1 = list(value1)
    if isinstance(value2, (list, tuple)):
        value2 = list(value2)

    if not isinstance(value1, type(value2)):
        raise RuntimeError("The two input types are inconsistent!."
                           "The input types must be the same")
    if isinstance(value1, (str,)):
        if value1.lower() != value2.lower():
            raise RuntimeError("Input one and input two are not equal!")
    elif isinstance(value1, (list, tuple,)):
        if value1 != value2:
            raise RuntimeError("Input one and input two are not equal!")


def get_shape_size(shape):
    """
    get all dimension.
    ----------
    shape: shape of data

    Returns
    -------
    """
    from functools import reduce
    product = reduce(lambda x, y: x * y, shape[:])

    return product


def cast(x, dtype):
    """Cast input to specified data type.

    Parameters
    ----------
    x : tvm.Tensor or Expr
        Input argument.

    dtype : str
        Data type.

    Returns
    -------
    y : tvm.Tensor
        The result.
    """
    if isinstance(x, _tensor.Tensor):
        return tvm.compute(
            x.shape, lambda *i: x(*i).astype(dtype), tag="elemwise")
    return _make._cast(dtype, x)


def shape_to_list(shape):
    """
    translate tvm.shape to list type in python
    """
    tmp = []
    for i in shape:
        if isinstance(i, _expr.ConstExpr):
            tmp.append(i.value)
        else:
            tmp.append(i)

    return tmp
