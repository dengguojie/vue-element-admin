#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import traceback
from sch_test_frame.ut import OpUT
import te
import tbe
from te import tvm
from impl.util.platform_adapter import operation
from impl.dynamic.conv2d import conv2d_fusion_compute
from impl.dynamic.ascend_dequant import ascend_dequant_compute
from tbe.common.context.op_info import OpInfo

ut_case = OpUT("conv_schedule", "conv_schedule.test_dynamic_conv_schedule_impl")
def test_bias_not_support_preload_910A(test_arg):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            with tbe.dsl.base.operation.compute():
                tbe.dsl.base.operation.get_op_context().add_op_info(OpInfo("conv2d_fusion_dequant", "Conv2D"))
                shape_x = [-1, 1, 32, -1, 32]
                range_x = [[1, 100], [1, 1], [32, 32], [12, 52]]
                var_list = ["batch_n", "fmap_w"]
                for idx, value in enumerate([0, 3]):
                    shape_x[value] = operation.var(var_list[idx], range_x[value])
                    operation.add_exclude_bound_var(shape_x[value])
                x_ = tvm.placeholder(shape_x, name="x", dtype="int8", attrs={"ori_shape": [-1, 1, 32, -1], "format": "NC1HWC0", "ori_format": "NCHW", "range": range_x})
                filter_ = tvm.placeholder([9, 4, 16, 32], name="filter", dtype="int8", attrs={"ori_shape": [64, 1, 3, 3], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([64], name="bias", dtype="int32", attrs={"ori_shape": [64], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "int32", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 2, 2]
                pads = [0, 1, 0, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_fusion_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
                scale_dtype = "float16"
                v200_version = ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
                if te.platform.cce_conf.get_soc_spec("SOC_VERSION") in v200_version:
                    scale_dtype = "uint64"
                deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [64]})
                dequant_out = ascend_dequant_compute(conv_out, deq_scale, None)
                tensor_list = [x_, filter_, bias_, deq_scale, dequant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(dequant_out)
                config = {
                    "name": "bias_not_support_preload_910A",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

def test_bias_not_support_preload_710(test_arg):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            with tbe.dsl.base.operation.compute():
                shape_x = [-1, 1, 32, -1, 32]
                range_x = [[1, 100], [1, 1], [32, 32], [12, 52]]
                var_list = ["batch_n", "fmap_w"]
                for idx, value in enumerate([0, 3]):
                    shape_x[value] = operation.var(var_list[idx], range_x[value])
                    operation.add_exclude_bound_var(shape_x[value])
                x_ = tvm.placeholder(shape_x, name="x", dtype="int8", attrs={"ori_shape": [-1, 1, 32, -1], "format": "NC1HWC0", "ori_format": "NCHW", "range": range_x})
                filter_ = tvm.placeholder([9, 4, 16, 32], name="filter", dtype="int8", attrs={"ori_shape": [64, 1, 3, 3], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([64], name="bias", dtype="int32", attrs={"ori_shape": [64], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "int32", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 2, 2]
                pads = [0, 1, 0, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_fusion_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
                scale_dtype = "float16"
                v200_version = ("Ascend710", "Ascend610", "Ascend615", "Hi3796CV300CS", "SD3403")
                if te.platform.cce_conf.get_soc_spec("SOC_VERSION") in v200_version:
                    scale_dtype = "uint64"
                deq_scale = tvm.placeholder((1, 4, 1, 1, 16), name="deq_scale", dtype=scale_dtype, attrs={"ori_shape": [64]})
                dequant_out = ascend_dequant_compute(conv_out, deq_scale, None)
                tensor_list = [x_, filter_, bias_, deq_scale, dequant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(dequant_out)
                config = {
                    "name": "bias_not_support_preload_710",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError, AttributeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_bias_not_support_preload_910A")
ut_case.add_cust_test_func("Ascend910A",test_func=test_bias_not_support_preload_910A)

print("adding test_bias_not_support_preload_710")
ut_case.add_cust_test_func("Ascend710",test_func=test_bias_not_support_preload_710)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])
