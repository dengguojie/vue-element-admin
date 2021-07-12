#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import traceback
from sch_test_frame.ut import OpUT
import te
import tbe
from te import tvm
from impl.util.platform_adapter import operation
from impl.conv2d import conv2d_compute
from impl.ascend_dequant import ascend_dequant_compute

ut_case = OpUT("conv_schedule", "conv_schedule.test_static_conv_schedule_impl")
def test_bias_preload(test_arg):
    try:
        with tbe.common.context.op_context.OpContext(None):
            with tbe.dsl.base.operation.compute():
                x_ = tvm.placeholder([1, 1, 32, 32, 32], name="x", dtype="int8", attrs={"ori_shape": [1, 1, 32, 32], "format": "NC1HWC0", "ori_format": "NCHW"})
                filter_ = tvm.placeholder([9, 4, 16, 32], name="filter", dtype="int8", attrs={"ori_shape": [64, 1, 3, 3], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([64], name="bias", dtype="int32", attrs={"ori_shape": [64], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "int32", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 2, 2]
                pads = [0, 1, 0, 1]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
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
                    "name": "bias_preload",
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

print("adding test_bias_preload")
ut_case.add_cust_test_func(test_func=test_bias_preload)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])