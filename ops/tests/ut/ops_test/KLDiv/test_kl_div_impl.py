#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("KLDiv", "impl.kl_div", "kl_div")

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


if __name__ == '__main__':
    ut_case.run()
    exit(0)
