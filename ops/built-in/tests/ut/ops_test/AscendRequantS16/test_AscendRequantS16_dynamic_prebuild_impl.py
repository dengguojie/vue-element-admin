#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
import tbe.common.context.op_info as operator_info  # pylint: disable=import-outside-toplevel
from te.tvm import build_module
from op_test_frame.ut import OpUT
from impl.dynamic.ascend_requant_s16 import ascend_requant_s16


def test_ascend_requant_s16_prebuild0(params):
    with tbe.common.context.op_context.OpContext("dynamic"):
        current_build_config = build_module.current_build_config()
        current_build_config.set_attr("enable_op_prebuild", 1)

        with current_build_config:
            op_info = operator_info.OpInfo(
                "AscendRequantS16", "AscendRequantS16")
            tbe.common.context.op_context.get_context().add_op_info(op_info)

            ascend_requant_s16(params["x0"],
                               params["req_scale"],
                               params["x1"],
                               params["y0"],
                               params["y1"],
                               params["dual_output"],
                               params["relu_flag"],
                               params["kernel_name"])


def test_ascend_requant_s16_prebuild1001(_):
    x0 = {"shape": (1, 1, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
    req_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    x1 = None
    y0 = {"shape": (1, 1, 1, 1, 32), "dtype": "int8", "format": "NC1HWC0",
          "ori_shape": (1, 1, 1, 1, 32), "ori_format": "NC1HWC0"}
    y1 = {"shape": (1, 1, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

    params = {"x0": x0, "req_scale": req_scale, "x1": x1, "y0": y0, "y1": y1,
              "dual_output": True, "relu_flag": False, "kernel_name": "ascend_requant_s16"}
    test_ascend_requant_s16_prebuild0(params)


def test_ascend_requant_s16_prebuild2001(_):
    x0 = {"shape": (1, 2, 4, 4, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
    req_scale = {"shape": (1, 2, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (32,), "ori_format": "ND"}
    x1 = {"shape": (1, 2, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 2, 1, 1, 16), "ori_format": "NC1HWC0"}
    y0 = {"shape": (1, 1, 4, 4, 32), "dtype": "int8", "format": "NC1HWC0",
          "ori_shape": (1, 1, 4, 4, 32), "ori_format": "NC1HWC0"}
    y1 = None

    params = {"x0": x0, "req_scale": req_scale, "x1": x1, "y0": y0, "y1": y1,
              "dual_output": False, "relu_flag": False, "kernel_name": "ascend_requant_s16"}
    test_ascend_requant_s16_prebuild0(params)


def test_ascend_requant_s16_prebuild3001(_):
    x0 = {"shape": (1, 2, 4, 4, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
    req_scale = {"shape": (1, 2, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (32,), "ori_format": "ND"}
    x1 = {"shape": (1, 2, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 2, 1, 1, 16), "ori_format": "NC1HWC0"}
    y0 = {"shape": (1, 1, 4, 4, 32), "dtype": "int8", "format": "NC1HWC0",
          "ori_shape": (1, 1, 4, 4, 32), "ori_format": "NC1HWC0"}
    y1 = None

    params = {"x0": x0, "req_scale": req_scale, "x1": x1, "y0": y0, "y1": y1,
              "dual_output": False, "relu_flag": True, "kernel_name": "ascend_requant_s16"}
    test_ascend_requant_s16_prebuild0(params)


ut_case = OpUT("AscendRequantS16",
               "impl.dynamic.ascend_requant_s16", "ascend_requant_s16")
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_requant_s16_prebuild1001)
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_requant_s16_prebuild2001)
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_requant_s16_prebuild3001)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
