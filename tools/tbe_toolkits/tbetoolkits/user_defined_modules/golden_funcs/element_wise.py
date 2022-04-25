#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Special golden data generation function for element_wise pattern
"""
# Standard Packages
import math

# Third-Party Packages
import numpy
import tbetoolkits
from .registry import register_golden


@register_golden(["floor_mod"])
def _floor_mod(context: "tbetoolkits.UniversalTestcaseStructure"):
    input0 = context.input_arrays[0]
    input1 = context.input_arrays[1]
    return input0 - numpy.multiply(numpy.floor_divide(input0, input1), input1)


@register_golden(["add_n"])
def _add_n(context: "tbetoolkits.UniversalTestcaseStructure"):
    inputs = context.input_arrays
    result = inputs[0]
    for inp in inputs[1:]:
        result = numpy.add(result, inp, dtype=result.dtype)
    return result


@register_golden(["sigmoid_grad"])
def _sigmoid_grad(context: "tbetoolkits.UniversalTestcaseStructure"):
    input0 = context.input_arrays[0]
    input1 = context.input_arrays[1]
    return numpy.multiply(numpy.subtract(input0, numpy.square(input0)), input1)


@register_golden(["fill_d", "fill"])
def _fill_d(possible_const_dims, input0, dims):
    return numpy.tile(input0, dims)


@register_golden(["tile_d", "tile"])
def _tile_d(input0, multiples):
    return numpy.tile(input0, multiples)


@register_golden(["relu"])
def _relu(input0):
    return numpy.maximum(input0, 0)


@register_golden(["sqrt_grad"])
def _sqrt_grad(input0, input1):
    return numpy.divide(input1, numpy.multiply(input0, 2))


@register_golden(["leaky_relu_grad"])
def _leaky_relu_grad(x0, x1, negative_slope=0):
    x0_r = numpy.where(x0 > 0, 1, 0)
    return x1 * (numpy.abs(x0_r - 1) * negative_slope + x0_r)


@register_golden(["power"])
def _power(context: "tbetoolkits.UniversalTestcaseStructure"):
    input_0 = context.input_arrays[0]
    power = context.other_runtime_params["power"]
    scale = context.other_runtime_params["scale"]
    shift = context.other_runtime_params["shift"]
    return (input_0 * scale + shift) ** power


@register_golden(["fused_mul_add_n"])
def _fused_mul_add_n(x0, x1, x2):
    return numpy.add(x1, numpy.multiply(x0, x2))


@register_golden(["apply_rms_prop_d"])
def _power(var, ms, mom, lr, grad, rho, momentum, epsilon):
    grad_square = numpy.multiply(grad, grad)
    grad_square_ms = numpy.subtract(grad_square, ms)

    rho_gs = numpy.multiply(grad_square_ms, 1.0 - rho)
    ms_t = numpy.add(ms, rho_gs)

    m_mom = numpy.multiply(mom, momentum)
    lr_grad = numpy.add(grad, lr)

    e_ms = numpy.add(ms_t, epsilon)

    sqrt_ms = numpy.sqrt(e_ms)

    tmp_grad = numpy.divide(lr_grad, sqrt_ms)

    mom_t = numpy.add(m_mom, tmp_grad)

    var_t = numpy.subtract(var, mom_t)

    return [var_t, ms_t, mom_t]


@register_golden(["cast"])
def _cast(input0: numpy.ndarray, dst_type):
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
        if _dst_type == 9:
            _dst_type = "int64"
        if _dst_type == 10:
            _dst_type = "uint64"
        if _dst_type == 12:
            _dst_type = "bool"
        return _dst_type

    return input0.astype(getattr(numpy, _cast_dsttype_conversion(dst_type)))


@register_golden(["vaxpy"])
def _vaxpy(input0, input1, scalar=2):
    return input0*scalar + input1


@register_golden(["vmla"])
def _vmla(input0, input1, input2):
    return input0*input1 + input2


@register_golden(["vmadd"])
def _vmadd(input0, input1, input2):
    return input0*input2 + input1


@register_golden(["vmaddrelu"])
def _vmaddrelu(input0, input1, input2):
    vmadd = input0*input2 + input1
    return numpy.maximum(vmadd, 0)


@register_golden(["vcmp_bit"])
def _vcmp_bit(input1, input2):
    input1_list = input1.reshape(-1)
    input2_list = input2.reshape(-1)
    input_length = len(input1_list)

    if input_length % 8 == 0:
        output_shape = input_length // 8
        output = numpy.zeros((output_shape,), numpy.uint8)
        conditon = numpy.zeros((input_length,), numpy.uint8)
        index = 0
        tmp_condtion = 0
        for i in range(input_length):
            output_flag = 1 if input1_list[i] < input2_list[i] else 0
            conditon[i] = output_flag
            inner_index = i % 8
            tmp_condtion += output_flag*math.pow(2, inner_index)
            if inner_index == 7:
                output[index] = tmp_condtion
                tmp_condtion = 0
                index += 1

        print(conditon)
        return output
    else:
        output_shape = input_length // 8 + 1
        output = numpy.zeros((output_shape,), numpy.uint8)
        conditon = numpy.zeros((input_length,), numpy.uint8)

        index = 0
        tmp_condtion = 0
        for i in range(input_length):
            output_flag = 1 if input1_list[i] < input2_list[i] else 0
            conditon[i] = output_flag
            inner_index = i % 8
            tmp_condtion += output_flag*math.pow(2, inner_index)
            if inner_index == 7:
                output[index] = tmp_condtion
                tmp_condtion = 0
                index += 1

        output[index] = tmp_condtion
        print(conditon)
        return output


@register_golden(["vsel_bit"])
def _vsel_bit(conditon, input1, input2):
    input1_list = input1.reshape(-1)
    input2_list = input2.reshape(-1)
    conditon = conditon.reshape(-1)
    conditon_length = len(conditon)

    output = numpy.zeros((conditon_length*8,), input1_list.dtype)

    for i in range(conditon_length):
        mask = conditon[i]
        for j in range(8):
            flag = mask%2
            mask //= 2
            real_index = i*8 + j
            output[real_index] = input1_list[real_index] if flag == 1 else input2_list[real_index]

    print(conditon)
    return output


@register_golden(["vcmp_bool"])
def _vcmp_bool(input1, input2):
    input1_list = input1.reshape(-1)
    input2_list = input2.reshape(-1)
    input_length = len(input1_list)

    conditon = numpy.zeros((input_length,), numpy.uint8)
    for i in range(input_length):
        output_flag = 1 if input1_list[i] < input2_list[i] else 0
        conditon[i] = output_flag

    return conditon


@register_golden(["vsel_bool"])
def _vsel_bool(conditon, input1, input2):
    input1_list = input1.reshape(-1)
    input2_list = input2.reshape(-1)
    conditon_length = len(conditon)

    output = numpy.zeros((conditon_length,), input1_list.dtype)

    for i in range(conditon_length):
        output[i] = input1_list[i] if conditon[i] == 1 else input2_list[i]

    return output


@register_golden(["select"])
def _select(context):
    raise NotImplementedError("Select not supported")
