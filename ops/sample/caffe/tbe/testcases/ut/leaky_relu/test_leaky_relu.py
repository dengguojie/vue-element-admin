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

leaky_relu ut case
"""
import unittest
from impl.leaky_relu_demo import leaky_relu_demo
from te import platform as tbe_platform

class Test_leaky_relu_cce(unittest.TestCase):
    def tearDown(self):
        #Do this after each test case is executed.
        pass

    def setUp(self):
        #Do this before each test case is executed.
        pass

    @classmethod
    def tearDownClass(self):
        # Must use the @classmethod decorator, run once after all tests have run
        pass

    @classmethod
    def setUpClass(self):
        # Must use the @classmethod decorator, run once before all tests have run
        pass

    """
    Code the testcases here.
    """
    def test_leaky_ut(self):
        tbe_platform.cce_conf.te_set_version("Ascend310")
        leaky_relu_demo({"shape":(1,16,10,10), "dtype":"float16", "format":"ND"},{},0,"leaky_relu")
        leaky_relu_demo({"shape":(1,16), "dtype":"float16", "format":"ND"},{},-1,"leaky_relu")
        leaky_relu_demo({"shape":(1,16,16), "dtype":"float16", "format":"ND"},{},5,"leaky_relu")
        leaky_relu_demo({"shape":(1,16,16), "dtype":"float32", "format":"ND"},{},0,"leaky_relu")
        leaky_relu_demo({"shape":(1,16,8,32), "dtype":"int8", "format":"ND"},{},0.1,"leaky_relu")
        leaky_relu_demo({"shape":(1,16,8,32), "dtype":"int32", "format":"ND"},{},10,"leaky_relu")
        try:
            leaky_relu_demo({"shape":(1,16,8,32), "dtype":"float64", "format":"ND"},{},10,"leaky_relu")
        except:
            pass
if __name__ == "__main__":
    unittest.main()
