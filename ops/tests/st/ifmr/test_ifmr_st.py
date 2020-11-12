"""
Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

test ifmr
"""

import unittest
import os
import shutil
from impl.ifmr import ifmr
from run_testcase import run_testcase,get_path_val,print_func_name

testcases = {
    "op_name": "ifmr",
    "all": {
        "test_st_ifmr_1": ((1, 1), "float32", "cce_ifmr_1_1_float32"),
        "test_st_ifmr_2": ((16, 32), "float32", "cce_ifmr_16_32_float32"),
    },
    "mini": {},
    "cloud": {},
}

bin_path_val = get_path_val(testcases)


def test_ifmr(shape_x, dtype_val, kernel_name_val):
    ifmr({'shape': shape_x, 'dtype': dtype_val, 'format': 'ND',
          'ori_shape': shape_x, 'ori_format': 'ND'},
         {'shape': (1,), 'dtype': dtype_val, 'format': 'ND', 'ori_shape': (1,),
          'ori_format': 'ND'},
         {'shape': (1,), 'dtype': dtype_val, 'format': 'ND', 'ori_shape': (1,),
          'ori_format': 'ND'},
         {'shape': (512,), 'dtype': 'int32', 'format': 'ND',
          'ori_shape': (512,), 'ori_format': 'ND'},
         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
          'ori_format': 'ND'},
         {'shape': (1,), 'dtype': 'float32', 'format': 'ND', 'ori_shape': (1,),
          'ori_format': 'ND'},
         0.9,
         0.9,
         [0.7, 1.3],
         0.01,
         True,
         kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"
    if os.path.isfile(kernel_meta_path + lib_kernel_name):
        shutil.move(kernel_meta_path + lib_kernel_name,
                    bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o",
                    bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json",
                    bin_path_val + "/" + kernel_name_val + ".json")


class Test_ifmr_cce(unittest.TestCase):
    def tearDown(self):
        pass

    def setUp(self):
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUpClass(self):
        pass

    @print_func_name
    def test_cce_ifmr(self):
        run_testcase(testcases, test_ifmr)


if __name__ == "__main__":
    unittest.main()
