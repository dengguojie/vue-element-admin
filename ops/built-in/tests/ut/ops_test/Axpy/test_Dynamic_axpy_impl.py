#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#  pylint: disable=invalid-name,missing-module-docstring
#!/usr/bin/env python
import te
from op_test_frame.ut import OpUT

ut_case = OpUT("Axpy", "impl.dynamic.axpy", "axpy")

# pylint: disable=too-many-arguments
def gen_dynamic_axpy_case(shape_x, shape_y, range_x, range_y, dtype_val, alpha,format_arg,
                         kernel_name_val, expect):
    """
    gen_params fun.
    """
    return {"params": [
        {"ori_shape": shape_x, "shape": shape_x, "ori_format": format_arg,
         "format": format_arg, "dtype": dtype_val, "range": range_x},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": format_arg,
         "format": format_arg, "dtype": dtype_val, "range": range_y},
        {"ori_shape": shape_y, "shape": shape_y, "ori_format": format_arg,
         "format": format_arg, "dtype": dtype_val, "range": range_y},
         alpha],
        "case_name": kernel_name_val, "expect": expect, "format_expect": [],
        "support_expect": True}


ut_case.add_case("all",
                 gen_dynamic_axpy_case((-1,), (1,), ((1, 64),), ((1, 1),),
                                       "float16", 2.1, "ND", "dynamic_axpy_fp16_ND",
                                       "success"))
ut_case.add_case("all",
                 gen_dynamic_axpy_case((-1,2,3,4), (1,2,3,4), ((2, 16),(2,2),(3,3),(4,4)),
                                       ((1, 1),(2,2),(3,3),(4,4)),
                                       "float32", 2.1, "NCHW", "dynamic_axpy_fp32_NCHW",
                                       "success"))

if __name__ == '__main__':
    with te.op.dynamic():
        ut_case.run("Ascend910")
