#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
'''
Special golden data generation function for convolution pattern
'''
# Third-Party Packages
import numpy as np

def _infer_axes(input_data_format, data_format, shape):
    """
    To infer sum operate axis by input_data format and data_format
    to keep compute Architecture, so use global parameter send variable
    Parameters:
    ----------
    input_data_format: str
        op's input data format
    data_format: str
        'NCHW' or 'NHWC'
    shape : tuple or list
        the input data shape

    Returns
    -------
    g_shape_list. list
    """
    g_shape_list = []
    if input_data_format == 'FRACTAL_NZ':
        if data_format == "NCHW":
            if len(shape) == 4:
                for i in range(-1 * len(shape), 0):
                    if i not in (-1, -4):
                        g_shape_list += [i + len(shape)]
            elif len(shape) == 5:
                for i in range(-1 * len(shape), 0):
                    if i not in (-2, -3):
                        g_shape_list += [i + len(shape)]
            else:
                g_shape_list.append(0)
                for i in range(2, len(shape)):
                    g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 4:
                error_manager_vector.raise_err_specific_reson("bias_add_grad",
                                                              "cce_bias_add_grad_nz_2_nhwc \
                                                              only support shape larger than 4D")
            for i in range(-1 * len(shape), 0):
                if i not in (-1, -4):
                    g_shape_list += [i + len(shape)]
    elif input_data_format in ("FRACTAL_Z", "FRACTAL_Z_3D", "NC1HWC0", "NDC1HWC0"):
        if input_data_format == "FRACTAL_Z":
            # mean format is FRACTAL_Z, shape is C1HWNiNoC0
            g_shape_list = [1, 2, 3, 4]
        elif input_data_format == "FRACTAL_Z_3D":
            # mean format is FRACTAL_Z_3D, shape is DC1HWNiNoC0
            g_shape_list = [0, 2, 3, 4, 5]
        elif input_data_format == "NC1HWC0":
            # mean format is NC1HWC0, shape is NC1HWC0
            g_shape_list = [0, 2, 3]
        elif input_data_format == "NDC1HWC0":
            # mean format is NDC1HWC0, shape is NDC1HWC0
            g_shape_list = [0, 1, 3, 4]
    else:
        if data_format == "NCHW":
            g_shape_list = [0]
            for i in range(2, len(shape)):
                g_shape_list = g_shape_list + [i]
        else:
            if len(shape) < 2:
                error_manager_vector.raise_err_specific_reson("bias_add_grad", "cce_bias_add_grad \
                                                              only support shape larger than 2D")
            g_shape_list = list(range(len(shape) - 1))

    return tuple(g_shape_list)

def calc_expect_func_nz(x, y, data_format):
    reduce_axis = _infer_axes("FRACTAL_NZ", data_format, x.get("value").shape)
    res = np.sum(x.get("value"), axis=reduce_axis)
    return [res]

def calc_expect_func_fz3d(x, y, data_format):
    reduce_axis = _infer_axes("FRACTAL_Z_3D", data_format, x.get("value").shape)
    res = np.sum(x.get("value"), axis=reduce_axis)
    return [res]

def calc_expect_func_ndc1hwc0(x, y, data_format):
    reduce_axis = _infer_axes("NDC1HWC0", data_format, x.get("value").shape)
    res = np.sum(x.get("value"), axis=reduce_axis)
    return [res]

def calc_expect_func_other(x, y, data_format):
    reduce_axis = _infer_axes("NHWC", data_format, x.get("value").shape)
    res = np.sum(x.get("value"), axis=reduce_axis)
    return [res]
