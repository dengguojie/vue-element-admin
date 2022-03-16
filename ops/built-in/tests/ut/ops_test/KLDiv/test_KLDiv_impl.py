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

KLDiv ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("KLDiv", "impl.kl_div", "kl_div")

def calc_expect_func(x, target, y, reduction):
    output_pos = target["value"] * (np.log(target["value"]) - x["value"])
    cond_gt_0 = target["value"] > 0
    tmpResult = np.where(cond_gt_0, output_pos, 0)
    if reduction == "none":
        return tmpResult
    batch_size = x["shape"][0]
    if reduction == "batchmean":
        tmpResult = tmpResult * 1.0 / batch_size

    result = tmpResult.sum()
    out = np.ones((1,), dtype=y['dtype']) * result
    return out

# ut_case.add_test_cfg_cov_case("all")
ut_case.add_case("all", {
    "params": [{'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               "sum"],
    "expect": "success"
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               "batchmean"],
    "expect": "success"
})

ut_case.add_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (16, 1, 8, 8), 'dtype': 'float32', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (16, 1, 8, 8), 'dtype': 'float32', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND'},
               "sum"],
    "expect": "success"
})

ut_case.add_precision_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',"param_type":"output"},
               "batchmean"],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (7, 1, 17, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (7, 1, 17, 8), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (7, 1, 17, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (7, 1, 17, 8), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',"param_type":"output"},
               "batchmean"],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})


ut_case.add_precision_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 77), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 77), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',"param_type":"output"},
               "batchmean"],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_precision_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (1,), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape':(1,), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (1,), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (1,), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (1,), 'dtype': 'float16', 'format': 'ND',
                'ori_shape': (1,), 'ori_format': 'ND',"param_type":"output"},
               "batchmean"],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

ut_case.add_case("all", {
    "params": [{'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               {'shape': (16, 1, 8, 8), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW'},
               "none"],
    "expect": "success"
})

ut_case.add_precision_case(["Ascend910", "Ascend310"], {
    "params": [{'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 77), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 77), 'ori_format': 'NCHW',"param_type":"input"},
               {'shape': (16, 1, 77), 'dtype': 'float16', 'format': 'NCHW',
                'ori_shape': (16, 1, 77), 'ori_format': 'NCHW',"param_type":"output"},
               "none"],
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
})

def test_op_select_format(test_arg):

    from impl.kl_div import op_select_format
    op_select_format({"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},
                     {"shape":(1,1,2,1,1,16), "ori_shape":(1,1,2,1,1,16), "dtype":"float32", "format":"NDC1HWC0", "ori_format":"NDC1HWC0"},"none")
    op_select_format({"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,2,1,1), "ori_shape":(1,2,1,1), "dtype":"float32", "format":"NCHW", "ori_format":"NCHW"},
                     {"shape":(1,), "ori_shape":(1,), "dtype":"float32", "format":"ND", "ori_format":"ND"},"sum")
    op_select_format({"shape":(-1,-1,-1,-1,-1), "ori_shape":(-1,-1,-1,-1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(-1,-1,-1,-1,-1), "ori_shape":(-1,-1,-1,-1), "dtype":"float32", "format":"NC1HWC0", "ori_format":"NCHW"},
                     {"shape":(1,), "ori_shape":(1,), "dtype":"float32", "format":"ND", "ori_format":"ND"},"sum")

ut_case.add_cust_test_func(test_func=test_op_select_format)