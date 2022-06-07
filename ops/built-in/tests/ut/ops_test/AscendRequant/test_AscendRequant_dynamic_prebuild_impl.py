#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
import tbe.common.context.op_info as operator_info  # pylint: disable=import-outside-toplevel
from te.tvm import build_module
from op_test_frame.ut import OpUT
from impl.dynamic.ascend_requant import ascend_requant


def test_ascend_requant_prebuild0(params):
    with tbe.common.context.op_context.OpContext("dynamic"):
        current_build_config = build_module.current_build_config()
        current_build_config.set_attr("enable_op_prebuild", 1)

        with current_build_config:
            op_info = operator_info.OpInfo("AscendRequant", "AscendRequant")
            tbe.common.context.op_context.get_context().add_op_info(op_info)

            ascend_requant(params["x"], params["req_scale"],
                           params["y"], params["relu_flag"], params["kernel_name"])


def test_ascend_requant_prebuild1001(_):
    x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
         "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (1, 1, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0",
         "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": True, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)


def test_ascend_requant_prebuild2001(_):
    x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
         "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
    req_scale = {"shape": (1, 2, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (32,), "ori_format": "ND"}
    y = {"shape": (1, 1, 4, 4, 32), "dtype": "int8", "format": "NC1HWC0",
         "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)


def test_ascend_requant_prebuild3001(_):
    x = {"shape": (1, 1, 16, 16), "dtype": "int32",
         "format": "FRACTAL_NZ", "ori_shape": (2, 4), "ori_format": "ND"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)


def test_ascend_requant_prebuild4001(_):
    x = {"shape": (1, 1, 16, 16), "dtype": "int32",
         "format": "FRACTAL_NZ", "ori_shape": (2, 4), "ori_format": "ND"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)


def test_ascend_requant_prebuild5001(_):
    x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32",
         "format": "FRACTAL_NZ", "ori_shape": (2, 4, 4), "ori_format": "ND"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (2, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)

def test_ascend_requant_prebuild6001(_):
    x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32",
         "format": "FRACTAL_NZ", "ori_shape": (2, 4, 4), "ori_format": "ND", "name": "mad1"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (2, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)

def test_ascend_requant_prebuild7001(_):
    x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32",
         "format": "FRACTAL_NZ", "ori_shape": (2, 4, 4), "ori_format": "ND", "name": "convolution_c_col_bias"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (16,), "ori_format": "ND"}
    y = {"shape": (2, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}

    params = {"x": x, "req_scale": req_scale, "y": y,
              "relu_flag": False, "kernel_name": "ascend_requant"}
    test_ascend_requant_prebuild0(params)


ut_case = OpUT("AscendRequant", "impl.dynamic.ascend_requant",
               "ascend_requant")
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild1001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild2001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild3001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild4001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild5001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild6001)
ut_case.add_cust_test_func(
    ["Ascend310", "Ascend310P3", "Ascend910A"], test_func=test_ascend_requant_prebuild7001)


from impl.dynamic.ascend_requant import ascend_requant_compute
from te import tvm

def test_ascend_requant_compute(test_arg):
    x=tvm.placeholder((2, 1, 1, 16, 16), name="mad1", dtype="int32", attrs={"ori_shape":(2, 4, 4), "format":"FRACTAL_NZ", "ori_format":"ND","shape":(2, 1, 1, 16, 16)})
    req_scale=tvm.placeholder((1, 1, 1, 1, 16), name="req", dtype="uint64", attrs={"ori_shape":(1,), "format":"NC1HWC0", "ori_format":"ND","shape":(1, 1, 1, 1, 16)})
    y = {"shape": (2, 1, 1, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ",
         "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
    ascend_requant_compute(x, req_scale, y, False)

ut_case.add_cust_test_func(test_func=test_ascend_requant_compute)

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
