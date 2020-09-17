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

cumulativelogsumexp_d ut case
"""
import unittest
from impl.cumulativelogsumexp_d import cumulative_logsumexp_d


# pylint: disable=missing-function-docstring, no-self-use
# pylint: disable=too-many-arguments, missing-class-docstring
def cce_cum_lse(shape_x, dtype_x, axis, exclusive=False, reverse=False,
                kernel_name="cumulative_logsumexp", formats="ND"):
    cumulative_logsumexp_d(
        {"shape": shape_x, "dtype": dtype_x, "format": formats,
         "ori_shape": shape_x, "ori_format": formats},
        {"shape": shape_x, "dtype": dtype_x, "format": formats,
         "ori_shape": shape_x, "ori_format": formats},
        axis=axis, exclusive=exclusive, reverse=reverse,
        kernel_name=kernel_name
    )


class TestCumulativelogsumexpDCce(unittest.TestCase):
    def tearDown(self):
        # Do this after each test case is executed.
        pass

    def setUp(self):
        # Do this before each test case is executed.
        pass

    @classmethod
    def tearDownClass(cls):
        # Must use the @classmethod deco, run once after all tests have run
        pass

    @classmethod
    def setUpClass(cls):
        # Must use the @classmethod deco, run once before all tests have run
        pass

    # case fp32, 1d, ex=False, re=False
    def test_function_case0(self):
        cce_cum_lse((1, ), "float16", axis=0)

    # case fp32, 1d, ex=False, re=False
    def test_function_case1(self):
        cce_cum_lse((1, 16, 16), "float32", axis=0)

    # case fp 16, 1d, ex=True, re=False
    def test_function_case2(self):
        cce_cum_lse((1, 16, 16), "float16", axis=0, exclusive=True)

    # case fp 16, 1d, ex=False, re=True
    def test_function_case3(self):
        cce_cum_lse((1, 16, 16), "float16", axis=0, reverse=True)

    # case fp 16, 1d, ex=False, re=False
    def test_function_case4(self):
        cce_cum_lse((1, 16, 16), "float16", axis=0,
                    exclusive=True, reverse=True)

    # case fp 16, 1d, ex=False, re=False, no aligned well
    def test_function_case5(self):
        cce_cum_lse((1, 16, 13), "float16", axis=0,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False, no aligned well
    def test_function_case6(self):
        cce_cum_lse((1, 12, 13), "float32", axis=0,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False, neg axis
    def test_function_case13(self):
        cce_cum_lse((1, 12, 13), "float32",
                    axis=-2, exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False, axis in the middle
    def test_function_case7(self):
        cce_cum_lse((1, 12, 13, 15), "float32", axis=2,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False, axis in the last
    def test_function_case8(self):
        cce_cum_lse((1, 12, 13, 15), "float32", axis=3,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False, each*rdsize > 25*1024,MAXSIZE
    def test_function_case9(self):
        cce_cum_lse((10, 12, 13, 15, 223), "float32", axis=1,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=False
    def test_function_case16(self):
        cce_cum_lse((1, 12, 13, 15), "float32", axis=2,
                    exclusive=True, reverse=True)

    # case fp32, 1d, ex=False, re=True, move_tail
    def test_function_case17(self):
        cce_cum_lse((10, 12, 13, 15, 223), "float32", axis=1,
                    exclusive=False, reverse=True)

    # case fp32, 1d, ex=True, re=False, , move_tail
    def test_function_case18(self):
        cce_cum_lse((10, 12, 13, 15, 223), "float32", axis=1,
                    exclusive=True, reverse=False)

    # case fp32, 1d, ex=False, re=False, , move_tail
    def test_function_case19(self):
        cce_cum_lse((10, 12, 13, 15, 223), "float32", axis=1)

    # case fp32, 1d, ex=False, re=False, , move_tail
    def test_function_case20(self):
        cce_cum_lse((10, 1, 13, 15, 223), "float32", axis=1, exclusive=True)

    # case fp32, 1d, ex=False, re=False, outer_tail!= 0
    def test_function_case21(self):
        cce_cum_lse((35, 1, 13, 15, 223), "float32", axis=1, exclusive=True)

    # case fp32, 1d, ex=False, re=False, outer_tail!= 0
    def test_function_case22(self):
        cce_cum_lse((35, 1, 13, 15, 223), "float32", axis=-1, exclusive=True)

    # type error
    def test_function_case11(self):
        try:
            cce_cum_lse((1, 2, 3, 4), "int16", axis=2)
        except RuntimeError:
            pass

    # exceed the range of axes
    def test_function_case12(self):
        try:
            cce_cum_lse((1, 2, 4, 5), "float32", axis=10)
        except RuntimeError:
            pass


if __name__ == "__main__":
    unittest.main()
