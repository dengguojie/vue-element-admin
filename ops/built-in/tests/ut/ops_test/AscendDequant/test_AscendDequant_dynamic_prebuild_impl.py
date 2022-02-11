#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe
import tbe.common.context.op_info as operator_info  # pylint: disable=import-outside-toplevel
from te import tvm
from te import platform as cce_conf
from te.tvm import build_module
from op_test_frame.ut import OpUT
from impl.dynamic.ascend_dequant import ascend_dequant
from impl.dynamic.ascend_dequant import ascend_dequant_compute
from impl.dynamic.ascend_dequant import _scalar_dequant_v200
from impl.dynamic.ascend_dequant import _vector_dequant_v200
from impl.util.platform_adapter import tbe as platform_tbe


ut_case = OpUT("AscendDequant", "impl.dynamic.ascend_dequant", "ascend_dequant")


def test_ascend_dequant_dynamic_prebuild_0(params):
    with tbe.common.context.op_context.OpContext("dynamic"):
        current_build_config = build_module.current_build_config()
        current_build_config.set_attr("enable_op_prebuild", 1)

        with current_build_config:
            op_info = operator_info.OpInfo("AscendDequant", "AscendDequant")
            tbe.common.context.op_context.get_context().add_op_info(op_info)

            ascend_dequant(params["x"], params["deq_scale"], params["y"],
                           params["sqrt_mode"], params["relu_mode"], params["kernel_name"])


def add_test_1():

    def test_ascend_dequant_dynamic_prebuild_1001(_):
        x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1002(_):
        x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1003(_):
        x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1004(_):
        x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1005(_):
        x = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1006(_):
        x = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1007(_):
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    def test_ascend_dequant_dynamic_prebuild_1008(_):
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_ascend_dequant_dynamic_prebuild_0(params)

    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1001)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1002)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1003)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1004)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1005)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1006)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1007)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_1008)


def add_test_ascend710():
    def test_ascend_dequant_dynamic_prebuild_2001(_):
        x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 1), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 1), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}

        cce_conf.cce_conf.te_set_version("Ascend710")
        test_ascend_dequant_dynamic_prebuild_0(params)
        cce_conf.cce_conf.te_set_version("Ascend310")

    def test_ascend_dequant_dynamic_prebuild_2002(_):
        x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 1), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 1), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}

        cce_conf.cce_conf.te_set_version("Ascend710")
        test_ascend_dequant_dynamic_prebuild_0(params)
        cce_conf.cce_conf.te_set_version("Ascend310")

    def test_ascend_dequant_dynamic_prebuild_2003(_):
        x = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 1), "dtype": "uint64", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 1), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}

        cce_conf.cce_conf.te_set_version("Ascend710")
        test_ascend_dequant_dynamic_prebuild_0(params)
        cce_conf.cce_conf.te_set_version("Ascend310")

    def test_ascend_dequant_dynamic_prebuild_2004(_):
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 1), "dtype": "uint64",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}

        cce_conf.cce_conf.te_set_version("Ascend710")
        test_ascend_dequant_dynamic_prebuild_0(params)
        cce_conf.cce_conf.te_set_version("Ascend310")

    ut_case.add_cust_test_func("Ascend710", test_func=test_ascend_dequant_dynamic_prebuild_2001)
    ut_case.add_cust_test_func("Ascend710", test_func=test_ascend_dequant_dynamic_prebuild_2002)
    ut_case.add_cust_test_func("Ascend710", test_func=test_ascend_dequant_dynamic_prebuild_2003)
    ut_case.add_cust_test_func("Ascend710", test_func=test_ascend_dequant_dynamic_prebuild_2004)


def add_test_matmul():
    def test_compute_prebuild_matmul(params):
        with tbe.common.context.op_context.OpContext("dynamic"):
            current_build_config = build_module.current_build_config()
            current_build_config.set_attr("enable_op_prebuild", 1)

            with current_build_config:
                op_info = operator_info.OpInfo("AscendDequant", "AscendDequant")
                tbe.common.context.op_context.get_context().add_op_info(op_info)

                schedules, tensors = [], []
                with platform_tbe.compute():
                    x = params["x"]
                    shape_x = x["shape"]
                    dtype_x = x["dtype"]
                    input_x = tvm.placeholder(shape_x, dtype_x, "x")
                    matmul_attr = {"shape": shape_x}
                    matmul = tvm.compute(
                        shape_x, lambda *indice: input_x(*indice), name="matmul", tag="matmul", attrs=matmul_attr)

                    deq_scale = params["deq_scale"]
                    shape_deq = deq_scale["shape"]
                    dtype_deq = deq_scale.get("dtype")
                    attr = {"ori_shape": deq_scale.get("ori_shape")}
                    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)

                    res = ascend_dequant_compute(
                        matmul, input_deq, None, params["sqrt_mode"], params["relu_mode"], params["kernel_name"])

                    tensors.append([x, matmul, input_deq, res])

                with tvm.target.cce():
                    sch = platform_tbe.auto_schedule(res)
                    schedules.append(sch)

                kernel_name = params["kernel_name"]
                config = {"name": kernel_name, "tensor_list": tensors}
                platform_tbe.build(schedules, config)

    def test_ascend_dequant_dynamic_prebuild_3001(_):
        x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_matmul(params)

    def test_ascend_dequant_dynamic_prebuild_3002(_):
        x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_matmul(params)

    def test_ascend_dequant_dynamic_prebuild_3003(_):
        x = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                    "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": True,
                  "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_matmul(params)

    def test_ascend_dequant_dynamic_prebuild_3004(_):
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_matmul(params)

    def test_ascend_dequant_dynamic_prebuild_3005(_):
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "sqrt_mode": False,
                  "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_matmul(params)

    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_3001)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_3002)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_3003)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_3004)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_3005)


def add_test_conv():
    def test_compute_prebuild_conv(params):
        with tbe.common.context.op_context.OpContext("dynamic"):
            current_build_config = build_module.current_build_config()
            current_build_config.set_attr("enable_op_prebuild", 1)

            with current_build_config:
                op_info = operator_info.OpInfo("AscendDequant", "AscendDequant")
                tbe.common.context.op_context.get_context().add_op_info(op_info)

                schedules, tensors = [], []
                with platform_tbe.compute():
                    mad1 = params["mad1"]
                    shape_mad1 = mad1["shape"]
                    dtype_mad1 = mad1["dtype"]
                    tensor_mad1 = tvm.placeholder(shape_mad1, dtype_mad1, "input_mad1")

                    x = params["x"]
                    shape_x = x["shape"]
                    op_x = tvm.compute(shape_mad1, lambda *indice: tensor_mad1(*indice), name="mad1", tag="mad1")
                    attr_x = {"invalid_data_rm_flag": 1,
                              "conv_shape": params["y"]["shape"], "remove_padded_column_in_next_op": 0}
                    input_x = tvm.compute(shape_x, lambda *indice: op_x(*indice), name="x", tag="x", attrs=attr_x)

                    deq_scale = params["deq_scale"]
                    shape_deq = deq_scale["shape"]
                    dtype_deq = deq_scale.get("dtype")
                    attr_deq = {"ori_shape": deq_scale.get("ori_shape")}
                    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr_deq)

                    res = ascend_dequant_compute(
                        input_x, input_deq, None, params["sqrt_mode"], params["relu_mode"], params["kernel_name"])

                    tensors.append([tensor_mad1, op_x, input_x, input_deq, res])

                with tvm.target.cce():
                    sch = platform_tbe.auto_schedule(res)
                    schedules.append(sch)

                kernel_name = params["kernel_name"]
                config = {"name": kernel_name, "tensor_list": tensors}
                platform_tbe.build(schedules, config)

    def test_ascend_dequant_dynamic_prebuild_4001(_):
        mad1 = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
                "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        x = {"shape": (1, 1, 1, 1, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}

        params = {"mad1": mad1, "x": x, "deq_scale": deq_scale, "y": y,
                  "sqrt_mode": False, "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_conv(params)

    def test_ascend_dequant_dynamic_prebuild_4002(_):
        mad1 = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
                "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        x = {"shape": (1, 2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"mad1": mad1, "x": x, "deq_scale": deq_scale, "y": y,
                  "sqrt_mode": False, "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_conv(params)

    def test_ascend_dequant_dynamic_prebuild_4003(_):
        mad1 = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
                "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        x = {"shape": (1, 4, 4, 16, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 4, 4, 16, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"mad1": mad1, "x": x, "deq_scale": deq_scale, "y": y,
                  "sqrt_mode": True, "relu_mode": True, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_conv(params)

    def test_ascend_dequant_dynamic_prebuild_4004(_):
        mad1 = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
                "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        x = {"shape": (2, 1, 1, 16, 16), "dtype": "int32", "format": "FRACTAL_NZ",
             "ori_shape": (2, 4, 4), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16",
                     "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "ND"}
        y = {"shape": (2, 1, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
             "ori_shape": (1, 4, 4, 16, 16), "ori_format": "NC1HWC0"}

        params = {"mad1": mad1, "x": x, "deq_scale": deq_scale, "y": y,
                  "sqrt_mode": False, "relu_mode": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_conv(params)

    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_4001)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_4002)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_4003)
    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_4004)


def add_test_scalar_dequant_v200():
    def test_compute_prebuild_scalar_dequant_v200(params):
        with tbe.common.context.op_context.OpContext("dynamic"):
            current_build_config = build_module.current_build_config()
            current_build_config.set_attr("enable_op_prebuild", 1)

            with current_build_config:
                op_info = operator_info.OpInfo("AscendDequant", "AscendDequant")
                tbe.common.context.op_context.get_context().add_op_info(op_info)

                schedules, tensors = [], []
                with platform_tbe.compute():
                    x = params["x"]
                    shape_x = x["shape"]
                    dtype_x = x["dtype"]
                    input_x = tvm.placeholder(shape_x, dtype_x, "x")

                    deq_scale = params["deq_scale"]
                    shape_deq = deq_scale["shape"]
                    dtype_deq = deq_scale.get("dtype")
                    attr = {"ori_shape": deq_scale.get("ori_shape")}
                    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)

                    res = _scalar_dequant_v200(input_x, shape_x, shape_x, input_deq, params["conv_flag"])

                    tensors.append([input_x, input_deq, res])

                with tvm.target.cce():
                    sch = platform_tbe.auto_schedule(res)
                    schedules.append(sch)

                kernel_name = params["kernel_name"]
                config = {"name": kernel_name, "tensor_list": tensors}
                platform_tbe.build(schedules, config)

    def test_ascend_dequant_dynamic_prebuild_5001(_):
        x = {"shape": (2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "conv_flag": False, "kernel_name": "ascend_dequant"}
        test_compute_prebuild_scalar_dequant_v200(params)


    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_5001)


def add_test_vector_dequant_v200():
    def test_compute_prebuild_vector_dequant_v200(params):
        with tbe.common.context.op_context.OpContext("dynamic"):
            current_build_config = build_module.current_build_config()
            current_build_config.set_attr("enable_op_prebuild", 1)

            with current_build_config:
                op_info = operator_info.OpInfo("AscendDequant", "AscendDequant")
                tbe.common.context.op_context.get_context().add_op_info(op_info)

                schedules, tensors = [], []
                with platform_tbe.compute():
                    x = params["x"]
                    shape_x = x["shape"]
                    dtype_x = x["dtype"]
                    input_x = tvm.placeholder(shape_x, dtype_x, "x")

                    deq_scale = params["deq_scale"]
                    shape_deq = deq_scale["shape"]
                    dtype_deq = deq_scale.get("dtype")
                    attr = {"ori_shape": deq_scale.get("ori_shape")}
                    input_deq = tvm.placeholder(shape_deq, name="deq_scale", dtype=dtype_deq, attrs=attr)

                    res = _vector_dequant_v200(input_x, shape_x, shape_x, input_deq, params["relu_mode"],
                              params["conv_flag"])

                    tensors.append([input_x, input_deq, res])

                with tvm.target.cce():
                    sch = platform_tbe.auto_schedule(res)
                    schedules.append(sch)

                kernel_name = params["kernel_name"]
                config = {"name": kernel_name, "tensor_list": tensors}
                platform_tbe.build(schedules, config)

    def test_ascend_dequant_dynamic_prebuild_6001(_):
        x = {"shape": (2, 4, 4, 16), "dtype": "int32", "format": "NC1HWC0",
             "ori_shape": (2, 4, 4, 16), "ori_format": "NC1HWC0"}
        deq_scale = {"shape": (1, 1, 1, 1, 16), "dtype": "float16", "format": "NC1HWC0",
                     "ori_shape": (1, 1, 1, 1, 16), "ori_format": "NC1HWC0"}
        y = {"shape": (1, 2, 4, 4, 16), "dtype": "float16", "format": "NC1HWC0",
             "ori_shape": (1, 2, 4, 4, 16), "ori_format": "NC1HWC0"}

        params = {"x": x, "deq_scale": deq_scale, "y": y, "relu_mode": False, "conv_flag": False,
                  "kernel_name": "ascend_dequant"}
        test_compute_prebuild_vector_dequant_v200(params)


    ut_case.add_cust_test_func(["Ascend310", "Ascend910A"], test_func=test_ascend_dequant_dynamic_prebuild_6001)


add_test_1()
add_test_ascend710()
add_test_matmul()
add_test_conv()
add_test_scalar_dequant_v200()
add_test_vector_dequant_v200()


if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)
