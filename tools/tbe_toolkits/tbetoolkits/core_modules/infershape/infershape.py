#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Shape inference methods
"""
Development Tips for shape inference

Inside a method, a shape can be a list.
BUT DO NOT pass a shape as list between public methods!!! Always treat it as a tuple!!!
"""
# Standard Packages
import copy
import math
import logging
import inspect
from typing import Tuple
from typing import Union
from typing import Sequence

# Third-Party Packages
import tbetoolkits
from ...utilities import get
from ...utilities import is_shape
from ...utilities import get_tf_func
from ...utilities import eliminate_scalar_shapes


def __reversed_broadcast_rule_with_size(size: int, shape1: list, shape2: list) -> bool:
    """
    Reversed compare two shapes and modify shape1 with broadcast rule
    """
    result = True
    for idx in range(-1, -size - 1, -1):
        both_dim = (shape1[idx], shape2[idx])
        if shape1[idx] != shape2[idx] and (1 not in both_dim and -1 not in both_dim):
            raise RuntimeError("Cannot perform Elewise operation on %s and %s"
                               % (str(shape1), str(shape2)))
        shape1[idx] = -1 if (-1 in both_dim and 1 in both_dim) else max(shape1[idx], shape2[idx])
    return result


def _produce_by_broadcast_rule(*input_shapes) -> tuple:
    """
    Produce an output shape by using input_shapes, this complies to broadcast rule
    """
    maximum_shape = []
    for shape in input_shapes:
        # Convert shape to list
        shape = list(shape)
        if len(shape) > len(maximum_shape):  # new shape has higher dimension
            # Insert missing dimensions to maximum_shape
            difference = len(shape) - len(maximum_shape)
            for diff_idx in range(difference):
                maximum_shape.insert(diff_idx, shape[diff_idx])
    return tuple(maximum_shape)


def _elewise_check_input_parameters(args, output_num, relations):
    # Perform param check
    if len(args) < 1:
        # Number of input shapes must be higher than 1
        raise ValueError("Cannot perform shape inference mode ELEWISE if there is no input shapes")
    if output_num < 1:
        # Number of output shapes must be higher than 1
        raise ValueError("Cannot perform shape inference mode ELEWISE if there is no output shapes")
    # Only shape<int> can be passed as args
    for arg in args:
        if not is_shape(arg):
            raise TypeError("Elewise infershape function received invalid shapes: %s" % (str(args)))
    # If relations are specified, it must be a tuple and its length must be the same as output_num
    # Also, related relations must use in-range int or NoneType as an index to indicate the
    # related input shape
    if relations is not None:
        if not isinstance(relations, tuple):
            raise TypeError(
                "Elewise infershape function received %s instead of tuple as shape relationships"
                % str(type(relations)))
        for relation in relations:
            if not isinstance(relation, (tuple, type(None))):
                raise TypeError(
                    "Elewise infershape function received %s instead of "
                    "tuple or NoneType as shape relationship"
                    % str(type(relation)))
            if isinstance(relation, tuple):
                if not len(relation) > 0:
                    raise ValueError("Cannot perform shape inference mode ELEWISE if "
                                     "output shape isn't related with any input shape")
                for related_shape_idx in relation:
                    if not isinstance(related_shape_idx, int):
                        raise TypeError(
                            "Elewise infershape function received %s instead of "
                            "int as related input shape index"
                            % str(type(related_shape_idx)))
                    if related_shape_idx >= len(args):
                        raise ValueError("Cannot perform shape inference mode ELEWISE if "
                                         "output shape is related with input shape index "
                                         "that is out of range %s" % str(len(args)))


def elewise(*args, output_num=1, relations=None) -> Tuple[Tuple[int], ...]:
    """
    Produce output shapes by input shapes, this method complies to the broadcast rule
    """
    _elewise_check_input_parameters(args, output_num, relations)
    # Perform broadcast rules
    output_shapes = [(-10000,)] * output_num
    for output_idx in range(output_num):
        if relations is None:
            output_shapes[output_idx] = _produce_by_broadcast_rule(*args)
            continue
        related_shapes = tuple(args[shape_idx] for shape_idx in range(len(args))
                               if relations[output_idx] is None
                               or shape_idx in relations[output_idx])
        output_shapes[output_idx] = _produce_by_broadcast_rule(*related_shapes)
    return tuple(output_shapes)


def reduce(input_tensor: tuple, axes) -> tuple:
    """
    Produce output shapes by input shapes, this method complies to the reduce rule
    """
    if not is_shape(input_tensor):
        raise RuntimeError("%s is not a shapelike object" % str(input_tensor))
    mutable_shape = list(input_tensor)
    # For dynamic reduce axes
    if axes is None:
        return tuple(-1 if ori_axis != 1 else 1 for ori_axis in mutable_shape)
    for axis in axes:
        if axis >= len(mutable_shape) or (axis < 0 and -axis > len(mutable_shape)):
            continue
        mutable_shape[axis] = 1
    return tuple(mutable_shape)


def tensorflow_inference(context: "tbetoolkits.UniversalTestcaseStructure", use_original=False) -> tuple:
    import tensorflow as tf
    from ...user_defined_modules.davinci_to_tf import dav2tf_registry
    from ...user_defined_modules.shape_funcs import shape_func_registry
    if context.op_name in shape_func_registry.special_shape_inference_func_map:
        return shape_func_registry.special_shape_inference_func_map[context.op_name](copy.deepcopy(context))
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.disable_eager_execution()
    if context.op_name in dav2tf_registry.dav_op_to_tf_map:
        context = dav2tf_registry.dav_op_to_tf_map[context.op_name](copy.deepcopy(context))
    params = {**context.other_compilation_params, **context.other_runtime_params}
    op_name = context.op_name
    input_dtypes = context.stc_input_dtypes
    input_formats = context.stc_input_formats
    input_ori_shapes = context.stc_ori_inputs
    original_format = get(context.stc_input_ori_formats, 0)
    target_format = get(input_formats, 0)
    tf_func = get_tf_func(op_name)
    if tf_func is not None:
        # Do format transformation
        with tf.compat.v1.variable_scope("", reuse=tf.compat.v1.AUTO_REUSE):
            placeholders = [tf.compat.v1.placeholder(shape=get(input_ori_shapes, i),
                                                     dtype=get(input_dtypes, i)) if i not in context.input_as_variable
                            else tf.compat.v1.get_variable(name="temp_var_%s_%d" % (context.testcase_name, i),
                                                           shape=get(input_ori_shapes, i),
                                                           dtype=get(input_dtypes, i),
                                                           initializer=tf.compat.v1.random_normal_initializer())
                            for i in range(len(input_ori_shapes))]
        # input_as_list
        temp_placeholders = []
        if context.stc_input_as_list_distribution:
            last_num = 0
            for num in context.stc_input_as_list_distribution:
                if num == 0:
                    temp_placeholders.append(placeholders[last_num])
                    last_num += 1
                else:
                    temp_placeholders.append(placeholders[last_num:last_num + num])
                    last_num += num
            if last_num < len(placeholders):
                for tensor in placeholders[last_num:]:
                    temp_placeholders.append(tensor)
            placeholders = temp_placeholders

        # Remove useless keys
        tf_func_params = tuple(inspect.signature(tf_func).parameters.keys())
        increased_index = 0
        for idx, value in enumerate(placeholders):
            if idx + increased_index >= len(tf_func_params):
                break
            while tf_func_params[idx + increased_index] in params:
                increased_index += 1
            else:
                params[tf_func_params[idx + increased_index]] = value
        for param in tuple(params.keys()):
            if param not in tf_func_params:
                logging.warning("Automatic tensorflow infershape removing redundant parameter: %s" % param)
                del params[param]

        # noinspection PyBroadException
        params["name"] = context.testcase_name
        try:
            result = tf_func(**params)
        except:
            logging.exception("TF Function failed with params: %s" % str(params))
            raise
        if context.input_as_variable:
            result = [placeholders[i] for i in context.input_as_variable]
        if isinstance(result, Sequence):
            result_shapes = []
            for output in result:
                if not isinstance(output, Sequence):
                    result_shapes.append(tuple(tuple(map(int, output.shape))))
                else:
                    for inner_output in output:
                        result_shapes.append(tuple(tuple(map(int, inner_output.shape))))
            result_shapes = eliminate_scalar_shapes(tuple(result_shapes))
        else:
            result_shapes = eliminate_scalar_shapes((tuple(map(int, result.shape)),))
        # Do format transformation
        if not use_original:
            result_shapes = tuple(transform(shape, original_format, target_format)
                                  if target_format in ["NC1HWC0", "NDC1HWC0"] else shape for shape in result_shapes)
        return result_shapes
    else:
        raise NotImplementedError("Automatic shape inference failed: Could not find tensorflow Implementation")


def range_inference(shape: tuple) -> Union[tuple, type(None)]:
    """
    Produce range by input shape, this method complies to simple range inference rule
    """
    if shape is None:
        return None
    if not is_shape(shape):
        raise TypeError("Range inference function received invalid shape: %s" % str(shape))
    _range = []
    for dim in shape:
        if dim in [-1, -2]:
            _range.append((1, None))
        else:
            _range.append((dim, dim))
    return tuple(_range)


def shape_inference(shapes: tuple, args: tuple, mode: str) -> tuple:
    """
    Produce output shape by input shapes, this method distribute arguments to their shape inference function
    :param shapes:
    :param args:
    :param mode:
    :return:
    """
    shapes = eliminate_scalar_shapes(shapes)
    if mode == "ELEWISE":
        return elewise(*shapes, output_num=args[0], relations=args[1])
    if mode == "REDUCE":
        reduce_results = []
        for shape in elewise(*shapes, output_num=args[1], relations=args[2]):
            result_shape = reduce(shape, args[0])
            reduce_results.append(result_shape)
        return tuple(reduce_results)
    if mode == "RANGE":
        return tuple(range_inference(shape) for shape in shapes)
    raise RuntimeError("UNKNOWN MODE %s" % mode)


def transform(shape, cur_format, target_format):
    """
    Produce target format shape with original shape
    :param shape:
    :param cur_format:
    :param target_format:
    :return:
    """
    format_transformation_map = {
        "NHWC": {
            "NC1HWC0": _NHWC25Dinfer,
            "FRACTAL_NZ": _NZinfer
        },
        "NCHW": {
            "NC1HWC0": _NCHW25Dinfer,
            "FRACTAL_NZ": _NZinfer
        },
        "NWC": {
            "NC1HWC0": _NWC25Dinfer,
            "FRACTAL_NZ": _NZinfer
        },
        "NDHWC": {
            "NDC1HWC0": _NDHWC25Dinfer,
            "FRACTAL_NZ": _NZinfer
        }
    }
    if cur_format in format_transformation_map:
        if target_format in format_transformation_map[cur_format]:
            return format_transformation_map[cur_format][target_format](shape)
        else:
            raise NotImplementedError("Cannot transform format %s to %s" % (cur_format, target_format))
    else:
        raise NotImplementedError(f"Cannot transform format {cur_format} to any format including {target_format}")


def _25Dinfer(before_c, c, after_c):
    c1 = math.ceil(c / 16)
    return (*before_c, c1, *after_c, 16)


def _NHWC25Dinfer(shape: tuple):
    if not len(shape) == 4:
        raise RuntimeError(f'4D to 5D transformation only supports 4 dimension input, but received {shape}')
    n, h, w, c = shape
    return _25Dinfer((n,), c, (h, w))


def _NDHWC25Dinfer(shape: tuple):
    if len(shape) != 5:
        raise RuntimeError(f'NDHWC to 6D transformation only supports 5 dimension input, but received {shape}')
    n, d, h, w, c = shape
    return _25Dinfer((n, d), c, (h, w))


def _NCHW25Dinfer(shape: tuple):
    if not len(shape) == 4:
        raise RuntimeError(f'4D to 5D transformation only supports 4 dimension input, but received {shape}')
    n, c, h, w = shape
    return _25Dinfer((n,), c, (h, w))


def _NWC25Dinfer(shape: tuple):
    if not len(shape) == 3:
        raise RuntimeError(f'3D to 5D transformation only supports 3 dimension input, but received {shape}')
    n, w, c = shape
    return _25Dinfer((n,), c, (1, w))


def _NZinfer(shape: tuple):
    if len(shape) < 2:
        shape = (2 - len(shape)) * (1,) + shape
    a, b = shape[-2:]
    return shape[:-2] + (math.ceil(b / 16), math.ceil(a / 16), 16, 16)
