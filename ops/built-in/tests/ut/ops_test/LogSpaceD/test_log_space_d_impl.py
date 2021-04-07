# # -*- coding:utf-8 -*-
import numpy as np
from op_test_frame.common import precision_info
from op_test_frame.ut import BroadcastOpUT

ut_case = BroadcastOpUT("log_space_d")

#pylint: disable=unused-argument
def calc_expect_func(input_x, y, start, end, steps=100, base=10.0, dtype=1):
    output_dtype_dict = {0:"float16", 1:"float32", 2:"int8", 3:"int32", 4:"uint8", 6:"int16",
                         7:"uint16", 8:"uint32", 9:"int64", 10:"uint64", 11:"double", 12:"bool"}
    outputArr = np.zeros(shape=input_x["shape"]).astype(output_dtype_dict[dtype])

    if steps == 1:
        res_log = pow(base, start)
        outputArr[0] = res_log
    else:
        diff = (end-start) / (steps-1)
        for i in range(steps):
            res_lin = start + i * diff
            res_log = pow(base, res_lin)
            outputArr[i] = res_log
    return [outputArr,]

ut_case.add_precision_case("all", {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input", "value": np.arange(0, 2, 1, np.float16)},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "output"}, 2.0, 3.0, 2, 0.2, 0],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_precision_log_space_d_case_1"
})

ut_case.add_precision_case(support_soc="Ascend910A", case={
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input", "value": np.arange(0, 2, 1, np.float32)},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "output"}, 1.0, 2.0, 2, -0.5, 1],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_precision_log_space_d_case_2"
})

ut_case.add_precision_case(support_soc="Ascend910A", case={
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (110,), "shape": (110,),
                "param_type": "input", "value": np.arange(0, 110, 1, np.float32)},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (110,), "shape": (110,),
                "param_type": "output"}, 14.0, 15.0, 110, 0.4, 1],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_precision_log_space_d_case_3"
})

ut_case.add_precision_case(support_soc="Ascend310", case={
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (20,), "shape": (20,),
                "param_type": "input", "value": np.arange(0, 20, 1, np.float32)},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (20,), "shape": (20,),
                "param_type": "output"}, 0.0, 0.1, 20, 1.2, 1],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_precision_log_space_d_case_4",
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case(support_soc="Ascend310", case={
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input", "value": np.arange(0, 2, 1, np.float32)},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "output"}, 1.0, 2.0, 2, -0.5, 1],
    "calc_expect_func": calc_expect_func,
    "case_name": "test_is_precision_log_space_d_case_5",
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
                "param_type": "output"}, -1.0, 10.0, 100],
    "case_name": "test_is_log_space_d_case_1"
})

ut_case.add_case("all", {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (2,), "shape": (2,),
                "param_type": "output"}, 1.0, 2.0, 2, -0.5],
    "case_name": "test_is_log_space_d_case_2"
})

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
#                 "param_type": "output"}, 0.0, 0.0, 1, 0.0],
#     "case_name": "test_is_log_space_d_case_3"
# })

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "output"}, -1.0, 10.0, -1],
#     "case_name": "test_is_log_space_d_case_4",
#     "expect": RuntimeError
# })

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "output"}, -1.0, 10.0, 1],
#     "case_name": "test_is_log_space_d_case_5",
#     "expect": RuntimeError
# })

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
#                 "param_type": "output"}, 2.0, 3.0, 1, 0.2, 0],
#     "case_name": "test_is_log_space_d_case_6",
#     "expect": RuntimeError
# })

# ut_case.add_case("all", {
#     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "input"},
#                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (100,), "shape": (100,),
#                 "param_type": "output"}, -1.0, 10.0, 100, 10.0, 2],
#     "case_name": "test_is_log_space_d_case_7",
#     "expect": RuntimeError
# })
