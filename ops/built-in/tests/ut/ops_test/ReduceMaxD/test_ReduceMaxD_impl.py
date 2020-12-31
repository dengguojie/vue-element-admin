# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import ReduceOpUT

# ut_case = ReduceOpUT("ReduceMaxD", None, None)


# # ============ auto gen ["Ascend910"] test cases start ===============
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1,), (0,), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1,), 0, False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 1), (1,), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 1), (1,), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (101, 10241), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (101, 10241), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1023*255, ), (-1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1023*255, ), (-1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (0, 1, 2), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (51, 101, 1023), (0, 1, 2), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (99991, 10), (0, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (99991, 10), (0, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 99991), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 99991), (1, ), False)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 99991, 10), (1, ), True)
# ut_case.add_reduce_case_simple(["Ascend910"], ["float32", "float16", "int8", "uint8", "bool", "int32"], (1, 99991, 10), (1, ), False)

# # ============ auto gen ["Ascend910"] test cases end =================

# if __name__ == '__main__':
#     # ut_case.run("Ascend910")
#     ut_case.run()

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("ReduceMaxD", "impl.reduce_max_d", "reduce_max_d")

def gen_dynamic_reduce_max_d_case(shape_x, range_x, dtype_val, format,
                                  ori_shape_x, axes, keepdims,
                                  kernel_name_val, expect):

    return {"params": [{"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       {"shape": shape_x, "dtype": dtype_val,
                        "range": range_x, "format": format,
                        "ori_shape": ori_shape_x, "ori_format": format},
                       axes, keepdims],
            "case_name": kernel_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

ut_case.add_case("all",
                 gen_dynamic_reduce_max_d_case((1024, 1000),
                                               [(1024,1024),(1000,1000)],
                                               "int32", "ND",
                                               (1024, 1), [1,], True,
                                               "reduce_max_d_int32_last_dim",
                                               "success"))

ut_case.add_case("all",
                 gen_dynamic_reduce_max_d_case((1024, 1000),
                                               [(1024,1024),(1000,1000)],
                                               "float32", "ND",
                                               (1024, 1), [1,], True,
                                               "reduce_max_d_fp32_last_dim",
                                               "success"))

if __name__ == '__main__':
    ut_case.run("Ascend910")

