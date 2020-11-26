# TODO fix me, run failed
# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
#
# ut_case = OpUT("TensorScatterSub", "impl.tensor_scatter_sub",
#                "tensor_scatter_sub")
#
#
# def gen_tensor_scatter_sub_case(x_shape, indices_shape, updates_shape,
#                                 dtype_x, case_name_val, expect):
#     return {"params": [{"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": "ND", "format": "ND"},
#                        {"shape": indices_shape, "dtype": "int32", "ori_shape": indices_shape, "ori_format": "ND", "format": "ND"},
#                        {"shape": updates_shape, "dtype": dtype_x, "ori_shape": updates_shape, "ori_format": "ND", "format": "ND"},
#                        {"shape": x_shape, "dtype": dtype_x, "ori_shape": x_shape, "ori_format": "ND", "format": "ND"}],
#             "case_name": case_name_val,
#             "expect": expect,
#             "format_expect": [],
#             "support_expect": True}
#
#
# ut_case.add_case("all",
#                  gen_tensor_scatter_sub_case((33,5), (33,25,1), (33,25,5),
#                                             "float32", "valid_fp32", "success"))
#
# if __name__ == '__main__':
#     ut_case.run()
#     exit(0)
