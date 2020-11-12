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

test test_op
"""

import unittest
import os
import shutil
from impl.test_op import test_op
from run_testcase import run_testcase, get_path_val, print_func_name

testcases = {
    "op_name": "test_op",
    "all": {},
}

bin_path_val = get_path_val(testcases)


def test_test_op(shape_val, dtype_val, kernel_name_val):
    test_op({"shape": shape_val, "dtype": dtype_val}, {"shape": shape_val, "dtype": dtype_val}, kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"
    if (os.path.isfile(kernel_meta_path + lib_kernel_name)):
        shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/" + kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/" + kernel_name_val + ".json")


class Test_test_op_cce(unittest.TestCase):
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
    def test_cce_test_op(self):
        run_testcase(testcases, test_test_op)


if __name__ == "__main__":
    unittest.main()
