# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""DSL Golden"""

# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_golden
from ...utilities import bfloat16_conversion


def _cast_dsttype_conversion(_dst_type):
    """
    This function is determined by TBE operator implementation
    Update is needed if TBE operator changed
    :param _dst_type: int
    :return: str
    """
    if _dst_type == 0:
        _dst_type = "float32"
    if _dst_type == 1:
        _dst_type = "float16"
    if _dst_type == 2:
        _dst_type = "int8"
    if _dst_type == 3:
        _dst_type = "int32"
    if _dst_type == 4:
        _dst_type = "uint8"
    if _dst_type == 5:
        _dst_type = "int64"
    if _dst_type == 10:
        _dst_type = "uint64"
    if _dst_type == 12:
        _dst_type = "bool"
    return _dst_type


@register_golden(["dsl_trunc"])
def dsl_trunc(context: "tbetoolkits.core_modules.dynamic_shape.ProfilingContextStructure"):
    if 6 == context.other_runtime_params["dst_type"]:
        return numpy.trunc(context.input_arrays[0]).astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return numpy.trunc(context.input_arrays[0]).astype(getattr(numpy,
                                                                   _cast_dsttype_conversion(
                                                                       context.other_runtime_params["dst_type"])))


@register_golden(["dsl_round"])
def dsl_round(input0: numpy.ndarray, dst_type):
    if 6 == dst_type:
        return numpy.rint(input0).astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return numpy.rint(input0).astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["dsl_round_half_up"])
def dsl_round_half_up(input0: numpy.ndarray, dst_type):
    if 6 == dst_type:
        return numpy.round(input0).astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return numpy.round(input0).astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["dsl_ceil"])
def dsl_ceil(input0: numpy.ndarray, dst_type):
    if 6 == dst_type:
        return numpy.ceil(input0).astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return numpy.ceil(input0).astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["dsl_floor"])
def dsl_floor(input0: numpy.ndarray, dst_type):
    if 6 == dst_type:
        return numpy.floor(input0).astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return numpy.floor(input0).astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["dsl_cast"])
def dsl_cast(input0: numpy.ndarray, dst_type):
    if 6 == dst_type:
        return input0.astype(bfloat16_conversion(["bfloat16"])[0])
    else:
        return input0.astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["dsl_vand"])
def dsl_vand(input1, input2):
    return numpy.bitwise_and(input1, input2)


@register_golden(["dsl_vor"])
def dsl_vor(input1, input2):
    return numpy.bitwise_or(input1, input2)


@register_golden(["dsl_vnot"])
def dsl_vnot(input1):
    return numpy.bitwise_not(input1)


@register_golden(["sqrt_reduce_sum_d_square_exp_relu"])
def sqrt_reduce_sum_d_square_exp_relu(input0):
    import logging
    import tensorflow as tf
    output0 = input0.astype("float32")
    output1 = numpy.sqrt(output0)
    output2 = output1.astype("float16")
    output3 = output2.astype("float32")
    output4 = numpy.sum(output3, axis=(1,3), keepdims=False)
    #output0 = tf.reduce_sum(input3,axis=(1,3), keepdims=False)  
    output5 = output4.astype("float16") 
    output6 = numpy.multiply(output5, output5)
    #logging.debug("output2++++++++++++++++++++ dtype is %s" % output2.dtype)
    #print("output2++++++++++++++++++++ dtype is ",output2.dtype,"++++",output2)
    #output3 = numpy.exp(input4)
    res = numpy.maximum(output6, 0)
    return res

@register_golden(["floordiv_add_sqrt_mul_relu"])
def floordiv_add_sqrt_mul_relu(input0, input1, input2, input3):
    import numpy as np
    res_floor_div = np.floor_divide(input0, input1)
    res_add = np.add(res_floor_div, input2)
    res_sqrt = np.sqrt(res_add)
    res_mul = np.multiply(res_sqrt, input3)
    res = np.maximum(res_mul, 0)
    return res