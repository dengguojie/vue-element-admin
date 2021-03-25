# # -*- coding:utf-8 -*-
import sys
import torch
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("quantize")

#pylint: disable=unused-argument
def calc_expect_func(input_x, scales, zero_points, output_z, dtype, axis):
    if(dtype == "torch.qint32"):
        dtype = torch.qint32
    elif(dtype == "torch.qint8"):
        dtype = torch.qint8
    elif(dtype == "torch.quint8"):
        dtype = torch.quint8
    if(scales['value'].shape[0] == 1 and zero_points['value'].shape[0] == 1):
        res = torch.quantize_per_tensor(torch.from_numpy(input_x['value']).float(), scales['value'][0], zero_points['value'][0], dtype).int_repr().numpy()
    else:
        res = torch.quantize_per_channel(torch.from_numpy(input_x['value']).float(), torch.from_numpy(scales['value']), torch.from_numpy(zero_points['value']), axis, torch.qint32).int_repr().numpy()
    return [res, ]

#pylint: disable=unused-argument
def calc_expect_func_2(input_x, scales, zero_points, output_z, dtype, axis):
    if(dtype == "torch.qint32"):
        dtype = torch.qint32
    elif(dtype == "torch.qint8"):
        dtype = torch.qint8
    elif(dtype == "torch.quint8"):
        dtype = torch.quint8
    res = torch.quantize_per_tensor(torch.from_numpy(input_x['value']).float(), 1, 1, dtype).int_repr().numpy()
    return [res, ]

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "input", "value_range": [1, 1]},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4), "shape": (3, 4),
                "param_type": "output", "value_range": [1, 1]},
               'torch.qint32', 0],
    "calc_expect_func": calc_expect_func_2
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint8', 1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.quint8', 1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6, 7, 8), "shape": (3, 4, 5, 6, 7, 8),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6, 7, 8), "shape": (3, 4, 5, 6, 7, 8),
                "param_type": "output"},
               'torch.qint32', -1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', -5],
    "calc_expect_func": calc_expect_func
})

ut_case.add_precision_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', -1],
    "calc_expect_func": calc_expect_func
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "uint8", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (4,), "shape": (4,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})

ut_case.add_case("Ascend910A", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (8,), "shape": (8,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (8,), "shape": (8,),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (3, 4, 5, 6), "shape": (3, 4, 5, 6),
                "param_type": "output"},
               'torch.qint32', 1],
    "calc_expect_func": calc_expect_func,
    "expect": RuntimeError
})