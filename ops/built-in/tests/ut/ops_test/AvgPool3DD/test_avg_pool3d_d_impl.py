"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

AvgPool3DD ut case
"""
from op_test_frame.common import precision_info
import tensorflow as tf
from op_test_frame.ut import OpUT

ut_case = OpUT("AvgPool3DD", "impl.avg_pool3d_d", "avg_pool3d_d")


ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,6,64,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,6,7,7,1024), "ori_format":"NDHWC"},
    None, None,
    {"shape": (1,5,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,5,7,7,1024), "ori_format":"NDHWC"},
    (1,2,7,7,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_d_001"})

ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,1,1,3,3,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,1,3,3,1), "ori_format":"NDHWC"},
    {"shape": (4,1,16,16), "format":"FRACTAL_Z_3D","dtype": "float16","ori_shape":(1,2,2,1,1),"ori_format":"DHWCN"},
    None,
    {"shape": (1,1,1,2,2,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,1,2,2,1), "ori_format":"NDHWC"},
    (1,1,2,2,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_d_002"})

ut_case.add_case(["Ascend910A"], {"params":[
    {"shape": (1,1,1,3,3,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,1,3,3,1), "ori_format":"NDHWC"},
    {"shape": (4,1,16,16), "format":"FRACTAL_Z_3D","dtype": "float16","ori_shape":(1,2,2,1,1),"ori_format":"DHWCN"},
    None,
    {"shape": (1,1,1,2,2,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,1,2,2,1), "ori_format":"NDHWC"},
    (1,1,2,2,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    True,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_d_003"})

ut_case.add_case(["Ascend910A"], {"params":[
    {"shape": (25,8,37,9,40,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (25,8,9,40,580), "ori_format":"NDHWC"},
    None,
    None,
    {"shape": (25,2,37,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (25,2,1,1,580), "ori_format":"NDHWC"},
    (1,1,9,40,1),
    (1,7,3,28,1),
    (0,0,0,0,0,0),
    False,
    False,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_d_004"})

# Test Case In Conv3D
ut_case.add_case(["Ascend910A"], {"params":[
    {"shape": (23,19,71,88,2,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (23,19,88,2,1124), "ori_format":"NDHWC"},
    {"shape": (12496,1,16,16), "format": "FRACTAL_Z_3D", "dtype": "float16", "ori_shape": (1,88,2,1124,1), "ori_format":"DHWCN"},
    {"shape": (23,2,71,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (23,2,71,1,1,16), "ori_format":"NDHWC"},
    {"shape": (23,2,71,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (), "ori_format":"NDHWC"},
    (1,1,88,2,1),
    (1,17,18,2,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_d_005"})

def calc_expect_func(x, y, ksize, strides):
    data = x["value"].transpose((0, 1, 3, 4, 2, 5)).reshape(x["ori_shape"])
    data=tf.Variable(data,dtype="float32")
    padding="VALID"
    ret=tf.nn.avg_pool3d(data,ksize,strides,padding)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result=sess.run(ret)

    n, d, c1, h, w, c0 = y["shape"]
    return result.reshape((n, d, h, w, c1, c0)).transpose((0, 1, 4, 2, 3, 5))

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,3,7,9,9,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,9,9,112), "ori_format":"NDHWC", "param_type": "input"},
               None, None,
               {"shape": (1,2,7,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,112), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,9,9,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,3,3,2,2,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,2,2,48), "ori_format":"NDHWC", "param_type": "input"},
               None, None,
               {"shape": (1,2,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,2,2,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,3,3,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,7,7,48), "ori_format":"NDHWC", "param_type": "input"},
               None, None,
               {"shape": (1,2,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,7,7,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,6,3,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,6,7,7,48), "ori_format":"NDHWC", "param_type": "input"},
               None, None,
               {"shape": (1,5,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,5,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,7,7,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
     "params": [{"shape": (25,8,37,9,40,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (25,8,9,40,580), "ori_format":"NDHWC", "param_type": "input"},
               None, None,
               {"shape": (25,2,37,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (25,2,1,1,580), "ori_format":"NDHWC", "param_type": "output"},
               (1,1,9,40,1),
               (1,7,3,28,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
