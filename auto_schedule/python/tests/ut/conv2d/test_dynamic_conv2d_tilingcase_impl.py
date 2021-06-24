#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import traceback
from sch_test_frame.ut import OpUT
import te
import tbe
from te import tvm
from impl.util.platform_adapter import operation
from impl.dynamic.conv2d import conv2d_fusion_compute
from impl.ascend_quant import ascend_quant_compute

ut_case = OpUT("conv2d_tilingcase", "conv2d_tilingcase.test_dynamic_conv2d_tilingcase_impl")

def test_h_out_1_case(test_arg):
    try:
        with tbe.common.context.op_context.OpContext("dynamic"):
            with tbe.dsl.base.operation.compute():
                shape_x = [-1, 96, -1, -1, 16]
                range_x = [[1, 101], [1536, 1536], [3, 3], [1, 110]]
                var_list = ["batch_n", "fmap_h", "fmap_w"]
                for idx, value in enumerate([0, 2, 3]):
                    shape_x[value] = operation.var(var_list[idx], range_x[value])
                    operation.add_exclude_bound_var(shape_x[value])
                x_ = tvm.placeholder(shape_x, name="x", dtype="float16", attrs={"ori_shape": [-1, 1536, -1, -1], "format": "NC1HWC0", "ori_format": "NCHW", "range": range_x})
                filter_ = tvm.placeholder([96, 128, 16, 16], name="filter", dtype="float16", attrs={"ori_shape": [2048, 1536, 1, 1], "format": "FRACTAL_Z", "ori_format": "NCHW"})
                bias_ = tvm.placeholder([2048], name="bias", dtype="float16", attrs={"ori_shape": [2048], "format": "ND", "ori_format": "ND"})
                output_ = {"dtype": "float16", "format": "NC1HWC0", "ori_format": "NCHW"}
                strides = [1, 1, 3, 3]
                pads = [0, 0, 0, 0]
                dilations = [1, 1, 1, 1]
                conv_out = conv2d_fusion_compute(x_, filter_, bias_, None, output_, strides, pads, dilations, 1, "NCHW")
                quant_out = ascend_quant_compute(conv_out, None, 0, -128)
                tensor_list = [x_, filter_, bias_, quant_out]
                with te.tvm.target.cce():
                    sch = te.utils.cce.auto_schedule(quant_out)
                config = {
                    "name": "conv2d_tilingcase_h_out_1",
                    "tensor_list": tensor_list,
                    "build_args": {"constant_realize_extent_in_infer_bound": False}
                }
                tbe.dsl.unify_schedule.build.build(sch, config)

    except (RuntimeError, ValueError, TypeError):
        msg = traceback.format_exc()
        print(msg)
        return False
    else:
        return True

print("adding test_h_out_1_case")
ut_case.add_cust_test_func(test_func=test_h_out_1_case)

if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend710"])
