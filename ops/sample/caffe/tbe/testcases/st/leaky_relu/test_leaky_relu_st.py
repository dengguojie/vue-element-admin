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

test add
"""

import unittest
import os
import shutil
from impl.leaky_relu_demo import leaky_relu_demo
from run_testcase import run_testcase,get_path_val,print_func_name

testcases = {
    "op_name": "leaky_relu",
    "all": {"test_leaky_relu_1_16_10_10_float16":
                ((1,16,10,10), "float16", 0, "ND", "leaky_relu_1_16_10_10_float16"),
            "test_leaky_relu_1_11_8_8_float32" :
                ((1,11,8,8), "float32", -0.3, "ND", "leaky_relu_1_11_8_8_float32"),
            "test_leaky_relu_1_128_int8" :
                ((1,128), "int8", -5, "ND", "leaky_relu_1_128_int8")},
    "mini": {},
    "cloud": {},
}

bin_path_val = get_path_val(testcases)

def test_leaky_relu(shape, dtype, neg, format,kernel_name_val):
    leaky_relu_demo(
        {"shape": shape, "dtype": dtype,"format":format},
        {},neg, kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"
    if (os.path.isfile(kernel_meta_path + lib_kernel_name)):
        shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/" + kernel_name_val + ".json")

class Test_leaky_relu_cce(unittest.TestCase):
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
    def test_cce_leaky_relu(self):
        run_testcase(testcases, test_leaky_relu)


if __name__ == "__main__":
    unittest.main()
