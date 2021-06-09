#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
import tbe.common.context.op_info as operator_info  # pylint: disable=import-outside-toplevel
from te.tvm import build_module
from op_test_frame.ut import OpUT
from impl.dynamic.ascend_dequant_s16 import ascend_dequant_s16


def test_ascend_dequant_s16_prebuild0(params):
    with tbe.common.context.op_context.OpContext("dynamic"):
        current_build_config = build_module.current_build_config()
        current_build_config.set_attr("enable_op_prebuild", 1)

        with current_build_config:
            op_info = operator_info.OpInfo("AscendDequantS16", "AscendDequantS16")
            tbe.common.context.op_context.get_context().add_op_info(op_info)

            ascend_dequant_s16(params["x0"], params["deq_scale"], params["x1"],
                               params["y"], params["relu_mode"], params["kernel_name"])


def test_ascend_dequant_s16_prebuild1001(_):
    x0 = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
          "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
    deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    x1 = {"shape": (1,), "dtype": "int16", "format": "ND",
          "ori_shape": (1,), "ori_format": "ND"}
    y = {"shape": (1, 1, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
         "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

    params = {"x0": x0, "deq_scale": deq_scale, "x1": x1, "y": y,
              "relu_mode": False, "kernel_name": "ascend_dequant_s16"}
    test_ascend_dequant_s16_prebuild0(params)


def test_ascend_dequant_s16_prebuild2001(_):
    x0 = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
          "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
    deq_scale = {"shape": (1, 2, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (32,), "ori_format": "ND"}
    x1 = None
    y = {"shape": (1, 2, 4, 4, 16), "dtype": "int16", "format": "NC1HWC0",
         "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

    params = {"x0": x0, "deq_scale": deq_scale, "x1": x1, "y": y,
              "relu_mode": True, "kernel_name": "ascend_dequant_s16"}
    test_ascend_dequant_s16_prebuild0(params)


def test_ascend_dequant_s16_prebuild3001(_):
    x0 = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
          "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
    deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "uint64",
                 "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
    x1 = {"shape": (1, 2, 1, 1, 16), "dtype": "int16", "format": "NC1HWC0",
          "ori_shape": (1, 2, 1, 1, 16), "ori_format": "NC1HWC0"}
    y = {"shape": (1, 4, 4, 16, 16), "dtype": "int16", "format": "NC1HWC0",
         "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

    params = {"x0": x0, "deq_scale": deq_scale, "x1": x1, "y": y,
              "relu_mode": False, "kernel_name": "ascend_dequant_s16"}
    test_ascend_dequant_s16_prebuild0(params)


ut_case = OpUT("AscendDequantS16",
               "impl.dynamic.ascend_dequant_s16", "ascend_dequant_s16")
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_dequant_s16_prebuild1001)
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_dequant_s16_prebuild2001)
ut_case.add_cust_test_func(["Ascend310", "Ascend710", "Ascend910A"],
                           test_func=test_ascend_dequant_s16_prebuild3001)


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
