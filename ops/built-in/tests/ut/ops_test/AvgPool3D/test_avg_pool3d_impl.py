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

AvgPool3D ut case
"""
from op_test_frame.common import precision_info
import tensorflow as tf
from op_test_frame.ut import OpUT

ut_case = OpUT("AvgPool3D", "impl.avg_pool3d", "avg_pool3d")
def _gen_data_case(case, expect, case_name_val, support_expect=True):
    return {"params": case,
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": support_expect}

ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,6,64,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,6,64,7,7,16), "ori_format":"NDC1HWC0"},
    {"shape": (1,5,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,5,64,7,7,16), "ori_format":"NDC1HWC0"},
    (1,2,7,7,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_001"})

ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,3,64,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,64,7,7,16), "ori_format":"NDC1HWC0"},
    {"shape": (1,2,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,64,1,1,16), "ori_format":"NDC1HWC0"},
    (1,2,7,7,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_002"})

ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,4,64,8,8,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,4,64,8,8,16), "ori_format":"NDC1HWC0"},
    {"shape": (1,3,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,64,1,1,16), "ori_format":"NDC1HWC0"},
    (1,2,8,8,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_003"})

ut_case.add_case(["Ascend910"], {"params":[
    {"shape": (1,3,64,9,9,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,64,9,9,16), "ori_format":"NDC1HWC0"},
    {"shape": (1,2,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,64,1,1,16), "ori_format":"NDC1HWC0"},
    (1,2,9,9,1),
    (1,1,1,1,1),
    (0,0,0,0,0,0),
    False,
    True,
    0,
    "NDHWC"],
    "expect": "success",
    "case_name":"test_avg_pool3d_004"})

def calc_expect_func(x, y, ksize, strides, pads):
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
               {"shape": (1,2,7,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,112), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,9,9,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,3,3,2,2,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,2,2,48), "ori_format":"NDHWC", "param_type": "input"},
               {"shape": (1,2,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,2,2,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,3,3,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,3,7,7,48), "ori_format":"NDHWC", "param_type": "input"},
               {"shape": (1,2,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,2,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,7,7,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})

ut_case.add_precision_case("Ascend910", {
    "params": [{"shape": (1,6,3,7,7,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,6,7,7,48), "ori_format":"NDHWC", "param_type": "input"},
               {"shape": (1,5,3,1,1,16), "format": "NDC1HWC0", "dtype": "float16", "ori_shape": (1,5,1,1,48), "ori_format":"NDHWC", "param_type": "output"},
               (1,2,7,7,1),
               (1,1,1,1,1),
               (0,0,0,0,0,0)],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)
})


ut_case.add_case({'ori_shape': (5,), 'shape': (5,), 'format': "ND",'ori_format': 'ND', 'dtype': 'int32', 'range': ((5,5),)},
         {'ori_shape': (16,2,3,-1,1), 'shape': (16,2,1,3,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16),(2,2),(1,1),(3,3),(3,40),(16,16))},
         {'ori_shape': (1,5,1,16,1), 'shape': (5,1,16,16), 'format': "FRACTAL_Z_3D", 'ori_format': 'DHWCN', 'dtype': 'float16',
            'range':((5,5),(1,1),(16,16),(16,16))},
         {'ori_shape': (16,12,12,-1,1), 'shape': (16,12,1,12,-1,16), 'format': "NDC1HWC0", 'ori_format': 'NDHWC', 'dtype': 'float16',
            'range': ((16,16), (12,12), (1,1), (12,12), (12,294), (16,16))},
         (5,1,1),
         (4,2,10),
         (-1,-1,-1,-1,-1,-1),
         False,
         False,
         0,
         "NDHWC")

if __name__ == '__main__':
    ut_case.run("Ascend910")
