#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("KLDiv", "impl.kl_div", "kl_div")

def calc_expect_func(x, target, y, reduction):
    output_pos = target["value"] * (np.log(target["value"]) - x["value"])
    cond_gt_0 = target["value"] > 0
    tmpResult = np.where(cond_gt_0, output_pos, 0)
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

# ut_case.add_precision_case(["Ascend910"], {
#     "params": [{'shape': (16, 1, 8, 8), 'dtype': 'float32', 'format': 'NCHW',
#                 'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW',"param_type":"input"},
#                {'shape': (16, 1, 8, 8), 'dtype': 'float32', 'format': 'NCHW',
#                 'ori_shape': (16, 1, 8, 8), 'ori_format': 'NCHW',"param_type":"input"},
#                {'shape': (1,), 'dtype': 'float32', 'format': 'ND',
#                 'ori_shape': (1,), 'ori_format': 'ND',"param_type":"output"},
#                "sum"],
#     "expect": "success",
#     "calc_expect_func": calc_expect_func,
#     "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)
# })

ut_case.add_precision_case(["Ascend910"], {
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

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")

