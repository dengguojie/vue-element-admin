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
from te.tvm._ffi.node import register_node
from te.platform import operation
from te.platform.fusion_manager import fusion_manager
from te.tvm import expr as _expr
from te.tvm import make as _make
from te.tvm import api as tvm
from te.tvm import tensor as _tensor
from te.utils import check_para
from functools import reduce

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


def broadcast_shapes(shape1, shape2, op_name=check_para.OP_NAME,
                     param_name_input1='', param_name_input2=''):
    """
    two input shapes produce three output shape
    """
    def _generate_dynamic_output(_shape1_i, _shape2_i, out_shape, index):
        if not _equal(_shape1_i, _shape2_i):
            if isinstance(_shape1_i, int):
                if _shape1_i == 1:
                    out_shape.append(_shape2_i)
                else:
                    out_shape.append(_shape1_i)
            elif isinstance(_shape2_i, int):
                if _shape2_i == 1:
                    out_shape.append(_shape1_i)
                else:
                    out_shape.append(_shape2_i)
            else:
                var_name = "dim_" + str(index) + "_2"
                _var = operation.get_te_var(var_name)
                if _var is None:
                    bound_x = operation.get_te_var(_shape1_i.name).get_bound()
                    bound_y = operation.get_te_var(_shape2_i.name).get_bound()
                    bound = (min(bound_x[0], bound_y[0]),
                             max(bound_x[1], bound_y[1]))
                    _var = operation.var(var_name, bound)
                else:
                    _var = _var.tvm_var
                out_shape.append(_var)
        else:
            out_shape.append(_shape1_i)

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
        if not _equal(shape1_i, shape2_i) and \
                (isinstance(shape1_i, int) and shape1_i != 1) \
                and (isinstance(shape2_i, int) and shape2_i != 1):
            error_info = {
                'errCode': check_para.OP_ERROR_CODE_013, 'op_name': op_name,
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
        if operation.in_dynamic():
            _generate_dynamic_output(shape1_i, shape2_i, out_shape, i)
        else:
            out_shape.append(shape1_i if _equal(shape2_i, 1) else shape2_i)

    if swapped:
        shape1, shape2 = shape2, shape1

    return shape1, shape2, out_shape


def refine_shapes_for_broadcast(shape1, shape2):
    """
    Fusing the axes for the input shapes
    """
    def _dynamic_refine_shapes_for_broadcast(shape1, shape2):
        """
        Fusing the axes for the input shapes
        """
        def _equals_one(_x):
            if isinstance(_x, _expr.ConstExpr):
                return _x.value == 1
            if isinstance(_x, int):
                return _x == 1
            return False

        def _get_state(_a, _b):
            if _equal(_a, _b):
                return 1
            if _equals_one(_a):
                return 2
            if _equals_one(_b):
                return 3
            return 4

        fused_shape1 = [1]
        fused_shape2 = [1]
        fusion_index = []
        current_index = []
        state = None
        mode = operation.get_context().get("mode")
        if mode != check_para.ORIGINAL:
            return shape1, shape2
        for index, (i_a, i_b) in enumerate(zip(shape1, shape2)):
            if _equals_one(i_a) and _equals_one(i_b):
                pass
            elif state is None:
                fused_shape1[-1] *= i_a
                fused_shape2[-1] *= i_b
                state = _get_state(i_a, i_b)
                current_index.append(index)
            elif _get_state(i_a, i_b) == 4:
                fused_shape1.append(i_a)
                fused_shape2.append(i_b)
                state = _get_state(i_a, i_b)
                fusion_index.append(current_index)
                current_index = [index]
            elif state == _get_state(i_a, i_b):
                fused_shape1[-1] *= i_a
                fused_shape2[-1] *= i_b
                current_index.append(index)
            else:
                fused_shape1.append(i_a)
                fused_shape2.append(i_b)
                state = _get_state(i_a, i_b)
                fusion_index.append(current_index)
                current_index = [index]

        fusion_index.append(current_index)
        operation.add_compile_info("_fusion_index", fusion_index)

        return fused_shape1, fused_shape2

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

    if fusion_manager.get_build_cfg() == "disable":
        return shape1, shape2

    shape1, shape2 = list(shape1), list(shape2)
    swapped = False
    if len(shape1) < len(shape2):
        shape1, shape2 = shape2, shape1
        swapped = True

    _dv = len(shape1) - len(shape2)
    shape2 = [1] * _dv + shape2

    if operation.in_dynamic():
        operation.add_compile_info("_fusion", 2)
        fused_shape1, fused_shape2 = \
            _dynamic_refine_shapes_for_broadcast(shape1, shape2)
    else:
        fused_shape1, fused_shape2 = \
            _const_refine_shapes_for_broadcast(shape1, shape2)

    if swapped:
        fused_shape1, fused_shape2 = fused_shape2, fused_shape1

    return fused_shape1, fused_shape2


def _equal(expr_a, expr_b):
    """
    :param expr_a:
    :param expr_b:
    :return:
    """
    elements1 = {}
    elements2 = {}

    single_types = (int, float, _expr.Var)
    const_types = (_expr.IntImm,)
    for expr, elements in zip((expr_a, expr_b), (elements1, elements2)):
        if isinstance(expr, single_types):
            elements[expr] = elements.get(expr, 0) + 1
        elif isinstance(expr, const_types):
            elements[expr.value] = elements.get(expr.value, 0) + 1
        elif isinstance(expr, _expr.Expr):
            _parse_expr(expr, elements)
        else:
            error_info = {
                'errCode': check_para.OP_ERROR_CODE_025,
                'op_name': operation.get_context().get_op_type(),
                'param_expr': expr}
            raise RuntimeError(
                error_info,
                "In op[%s], unsupported expr: [%s]"
                % (error_info['op_name'], error_info['param_expr']))

    return elements1 == elements2


def _parse_expr(expr, elements: dict):
    if isinstance(expr, _expr.Mul):
        _parse_mul(expr, elements)
    else:
        error_info = {
            'errCode': check_para.OP_ERROR_CODE_025,
            'op_name': operation.get_context().get_op_type(),
            'param_expr': expr}
        raise RuntimeError(error_info,
                           "In op[%s], unsupported expr: [%s]"
                           % (error_info['op_name'], error_info['param_expr']))


def _parse_mul(expr, elements: dict):
    if not isinstance(expr, _expr.Mul):
        error_info = {
            'errCode': check_para.OP_ERROR_CODE_026,
            'op_name': operation.get_context().get_op_type(),
            'param_expr': expr}
        raise RuntimeError(error_info,
                           "In op[%s], it is not mul expr: [%s]"
                           % (error_info['op_name'], error_info['param_expr']))

    const_types = (_expr.IntImm,)
    var_types = (_expr.Var,)
    for _x in (expr.a, expr.b):
        if isinstance(_x, const_types):
            elements[_x.value] = elements.get(_x.value, 0) + 1
        elif isinstance(_x, var_types):
            elements[_x] = elements.get(_x, 0) + 1
        else:
            _parse_mul(_x, elements)


def variable_shape(inputs: list, support_broadcast=False):
    """
    :param inputs: all inputs
    :param support_broadcast: whether to support broadcast
    :return:
    """
    def _has_intersection(range0, range1):
        _range0 = list(range0)
        _range1 = list(range1)
        if _range0[1] is None:
            _range0[1] = check_para.MAX_UNKNOWN_SHAPE_NUM
        if _range1[1] is None:
            _range1[1] = check_para.MAX_UNKNOWN_SHAPE_NUM
        return max(_range0[0], _range1[0]) <= min(_range0[1], _range1[1])

    def _select(cond, then_case, else_case):
        if cond:
            return then_case
        else:
            return else_case

    def _update_range(shape0, range0, shape1, range1):
        for index in range(len(range0)):
            verify_shape = (shape0[index] != -1 and shape1[index] != -1) or \
                           shape0[index] == 1 or shape1[index] == 1
            if verify_shape:
                continue
            range_x = list(range0[index])
            range_y = list(range1[index])
            for j, (_rx, _ry) in enumerate(zip(range_x, range_y)):
                if _rx is None:
                    range_x[j] = check_para.MAX_UNKNOWN_SHAPE_NUM
                if _ry is None:
                    range_y[j] = check_para.MAX_UNKNOWN_SHAPE_NUM
            x_const = shape0[index] != -1 and shape1[index] == -1
            y_const = shape0[index] == -1 and shape1[index] != -1
            variable_intersection = \
                _has_intersection(range_x, range_y) and \
                (range_x[0] > 1) and (range_y[0] > 1)
            if x_const:
                range_y = (_select(range_y[0] <= 1, range_y[0],
                                   shape0[index]),
                           _select(range_y[1] >= shape0[index],
                                   shape0[index], 1))
            elif y_const:
                range_y = (_select(range_x[0] <= 1, range_x[0],
                                   shape1[index]),
                           _select(range_x[1] >= shape1[index],
                                   shape1[index], 1))
            elif variable_intersection:
                range_x = (max(range_x[0], range_y[0]),
                           min(range_x[1], range_y[1]))
                range_y = range_x
            elif not _has_intersection(range_x, range_y):
                if range_x[0] <= 1:
                    range_x = (1, 1)
                if range_y[0] <= 1:
                    range_y = (1, 1)
            range0[index] = tuple(range_x)
            range1[index] = tuple(range_y)
            if range_x[0] == range_x[1]:
                shape0[index] = range_x[0]
            if range_y[0] == range_y[1]:
                shape1[index] = range_y[0]

    def _fill(_inputs):
        if support_broadcast:
            if len(inputs) != 2:
                error_info = {
                    'errCode': check_para.OP_ERROR_CODE_027,
                    'op_name': operation.get_context().get_op_type(),
                    'param_name': check_para.PARAM_NAME}
                raise RuntimeError(
                    error_info,
                    "In op[%s], only support two inputs for broadcast"
                    % (error_info['op_name']))
            x_0, x_1 = _inputs
            shape0, range0 = list(x_0["shape"]), list(x_0["range"])
            shape1, range1 = list(x_1["shape"]), list(x_1["range"])
            swapped = False
            if len(shape0) < len(shape1):
                shape0, range0, shape1, range1 = shape1, range1, shape0, range0
                swapped = True
            d_v = len(shape0) - len(shape1)
            shape1 = [1] * d_v + shape1
            range1 = [(1, 1)] * d_v + range1
            if swapped:
                shape0, range0, shape1, range1 = shape1, range1, shape0, range0
            _update_range(shape0, range0, shape1, range1)
            return [shape0, shape1], [range0, range1]

        _shapes, _ranges = [], []
        for _input in inputs:
            _shapes.append(_input["shape"])
            _ranges.append(_input["range"])
        return _shapes, _ranges

    def _maybe_broadcast():
        if support_broadcast:
            for _r in ranges:
                if _r[i][0] <= 1:
                    return True
        return False

    def _mode_process():
        if mode == check_para.CONST:
            if support_broadcast:
                input1 = inputs[0]["shape"]
                input2 = inputs[1]["shape"]
                const_shape = [a & b for a, b in zip(input1, input2)]
            else:
                const_shape = inputs[0]["shape"]
            operation.get_context().get_current_compute(). \
                add("const_shape", const_shape)
        elif mode == check_para.SPECIAL:
            pattern = inputs[0].get("pattern")
            operation.get_context().\
                get_current_compute().add("pattern", pattern)
            if support_broadcast:
                for i, _pattern in enumerate(pattern):
                    if _pattern == check_para.COMMON:
                        for j in range(len(shapes)):
                            shapes[j][i] = -77
        elif mode == check_para.SPECIAL_SCALAR:
            pattern = inputs[0].get("pattern")
            operation.get_context(). \
                get_current_compute().add("pattern", pattern)

    if len(inputs) < 1:
        return []
    mode = inputs[0].get("mode")
    if mode is None:
        mode = check_para.ORIGINAL
    operation.get_context().add("mode", mode)
    operation.get_context().add("support_broadcast", support_broadcast)

    shapes, ranges = _fill(inputs)
    _mode_process()

    d_shapes = [[] for _ in shapes]
    for i in range(len(shapes[0])):
        _var = None
        need_two_vars = _maybe_broadcast()
        _suffix = 0
        for d_shape, shape, _range in zip(d_shapes, shapes, ranges):
            if shape[i] == -1:
                if _var is None or need_two_vars:
                    _var = operation.var("dim_" + str(i) + "_" + str(_suffix),
                                         _range[i])
                d_shape.append(_var)
            elif shape[i] == -77:
                if _var is None:
                    _var = operation.var("dim_" + str(i) + "_" + str(_suffix),
                                         _range[i])
                d_shape.append(_var)
            else:
                d_shape.append(shape[i])
            _suffix += 1

    return d_shapes


def simplify_axis_shape(shape, axis):
    """
    simplify the shape and aixs
    """
    axis1 = []
    shape1 = []
    merge_num = 0
    length = shape[0]

    for i in range(len(axis)):
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
        shape_final.append(reduce(lambda x, y:x*y, shape_noreduce))
    else:
        shape_final += shape_noreduce

    return shape_final, axis_final


def shape_refine(shape, reduce_axis=None):
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
        if not check_reduce_need_refine(shape, reduce_axis):

            if hasattr(reduce_axis, 'index'):
                return shape, reduce_axis
            else:
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

    else:
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
    else:
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


def check_reduce_need_refine(shape, reduce_axis):
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
    if product >= check_para.SHAPE_SIZE_LIMIT + 1:
        raise RuntimeError(
            "The shape size for operator has exceeded the maximum")

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
        if isinstance(i, _expr.Var):
            tmp.append(i)
        else:
            tmp.append(i.value)
    return tmp
