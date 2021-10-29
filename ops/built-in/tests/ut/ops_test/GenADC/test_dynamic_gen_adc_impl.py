#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this file
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

Dynamic GenADC ut case
"""
from op_test_frame.ut import OpUT
from impl.dynamic.gen_adc import gen_adc
from tbe.common.platform.platform_info import set_current_compile_soc_info
import tbe


ut_case = OpUT("GenADC", "impl.dynamic.gen_adc", "gen_adc")


def test_gen_adc_case001(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (32,), "dtype": "float16", "format": "ND",
                 "ori_shape": (32,), "ori_format": "ND", "range": ((32, 32),)},
                {"shape": (16, 256, 2), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 256, 2), "ori_format": "ND", "range": ((16, 16), (256, 256), (2, 2))},
                {"shape": (1000000, 32), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 32), "ori_format": "ND", "range": ((1000000, 1000000), (32, 32))},
                {"shape": (1,), "dtype": "int32", "format": "ND",
                 "ori_shape": (1,), "ori_format": "ND", "range": ((1, 1),)},
                {"shape": (1, 16, 256), "dtype": "float16", "format": "ND",
                 "ori_shape": (1, 16, 256), "ori_format": "ND", "range": ((1, 1), (16, 16), (256, 256))})
    set_current_compile_soc_info(test_args)


def test_gen_adc_case002(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (64,), "dtype": "float32", "format": "ND",
                 "ori_shape": (64,), "ori_format": "ND", "range": ((64, 64),)},
                {"shape": (32, 256, 2), "dtype": "float16", "format": "ND",
                 "ori_shape": (32, 256, 2), "ori_format": "ND", "range": ((32, 32), (256, 256), (2, 2))},
                {"shape": (2000000, 64), "dtype": "float16", "format": "ND",
                 "ori_shape": (2000000, 64), "ori_format": "ND", "range": ((2000000, 2000000), (64, 64))},
                {"shape": (17,), "dtype": "int32", "format": "ND",
                 "ori_shape": (17,), "ori_format": "ND", "range": ((17, 17),)},
                {"shape": (17, 32, 256), "dtype": "float16", "format": "ND",
                 "ori_shape": (17, 32, 256), "ori_format": "ND", "range": ((17, 17), (32, 32), (256, 256))})
    set_current_compile_soc_info(test_args)


def test_gen_adc_case003(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (64,), "dtype": "float16", "format": "ND",
                 "ori_shape": (64,), "ori_format": "ND", "range": ((64, 64),)},
                {"shape": (16, 256, 4), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 256, 4), "ori_format": "ND", "range": ((16, 16), (256, 256), (4, 4))},
                {"shape": (3000000, 64), "dtype": "float16", "format": "ND",
                 "ori_shape": (3000000, 64), "ori_format": "ND", "range": ((3000000, 3000000), (64, 64))},
                {"shape": (1024,), "dtype": "int64", "format": "ND",
                 "ori_shape": (1024,), "ori_format": "ND", "range": ((1024, 1024),)},
                {"shape": (1024, 16, 256), "dtype": "float16", "format": "ND",
                 "ori_shape": (1024, 16, 256), "ori_format": "ND", "range": ((1024, 1024), (16, 16), (256, 256))})
    set_current_compile_soc_info(test_args)


def test_gen_adc_case004(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (128,), "dtype": "float32", "format": "ND",
                 "ori_shape": (128,), "ori_format": "ND", "range": ((128, 128),)},
                {"shape": (32, 512, 4), "dtype": "float16", "format": "ND",
                 "ori_shape": (32, 512, 4), "ori_format": "ND", "range": ((32, 32), (512, 512), (4, 4))},
                {"shape": (9000000, 128), "dtype": "float16", "format": "ND",
                 "ori_shape": (9000000, 128), "ori_format": "ND", "range": ((9000000, 9000000), (128, 128))},
                {"shape": (384,), "dtype": "int64", "format": "ND",
                 "ori_shape": (384,), "ori_format": "ND", "range": ((384, 384),)},
                {"shape": (384, 32, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (384, 32, 512), "ori_format": "ND", "range": ((384, 384), (32, 32), (512, 512))})
    set_current_compile_soc_info(test_args)


def test_gen_adc_case005(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (128,), "dtype": "float32", "format": "ND",
                 "ori_shape": (128,), "ori_format": "ND", "range": ((128, 128),)},
                {"shape": (32, 512, 4), "dtype": "float16", "format": "ND",
                 "ori_shape": (32, 512, 4), "ori_format": "ND", "range": ((32, 32), (512, 512), (4, 4))},
                {"shape": (1000000, 128), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 128), "ori_format": "ND", "range": ((1000000, 1000000), (128, 128))},
                {"shape": (384,), "dtype": "int64", "format": "ND",
                 "ori_shape": (384,), "ori_format": "ND", "range": ((384, 384),)},
                {"shape": (384, 32, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (384, 32, 512), "ori_format": "ND", "range": ((384, 384), (32, 32), (512, 512))},
                "l2sqr")
    set_current_compile_soc_info(test_args)


def test_gen_adc_case006(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (128,), "dtype": "float32", "format": "ND",
                 "ori_shape": (128,), "ori_format": "ND", "range": ((128, 128),)},
                {"shape": (32, 512, 4), "dtype": "float16", "format": "ND",
                 "ori_shape": (32, 512, 4), "ori_format": "ND", "range": ((32, 32), (512, 512), (4, 4))},
                {"shape": (1000000, 128), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 128), "ori_format": "ND", "range": ((1000000, 1000000), (128, 128))},
                {"shape": (384,), "dtype": "int64", "format": "ND",
                 "ori_shape": (384,), "ori_format": "ND", "range": ((384, 384),)},
                {"shape": (384, 32, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (384, 32, 512), "ori_format": "ND", "range": ((384, 384), (32, 32), (512, 512))},
                "inner_product")
    set_current_compile_soc_info(test_args)


def test_gen_adc_case007(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (256,), "dtype": "float32", "format": "ND",
                 "ori_shape": (256,), "ori_format": "ND", "range": ((256, 256),)},
                {"shape": (16, 512, 16), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 512, 16), "ori_format": "ND", "range": ((16, 16), (512, 512), (16, 16))},
                {"shape": (1000000, 256), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 256), "ori_format": "ND", "range": ((1000000, 1000000), (256, 256))},
                {"shape": (10,), "dtype": "int64", "format": "ND",
                 "ori_shape": (10,), "ori_format": "ND", "range": ((10, 10),)},
                {"shape": (10, 16, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (10, 16, 512), "ori_format": "ND", "range": ((10, 10), (16, 16), (512, 512))},
                "l2sqr")
    set_current_compile_soc_info(test_args)


def test_gen_adc_case008(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (256,), "dtype": "float32", "format": "ND",
                 "ori_shape": (256,), "ori_format": "ND", "range": ((256, 256),)},
                {"shape": (16, 512, 16), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 512, 16), "ori_format": "ND", "range": ((16, 16), (512, 512), (16, 16))},
                {"shape": (1000000, 256), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 256), "ori_format": "ND", "range": ((1000000, 1000000), (256, 256))},
                {"shape": (10,), "dtype": "int64", "format": "ND",
                 "ori_shape": (10,), "ori_format": "ND", "range": ((10, 10),)},
                {"shape": (10, 16, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (10, 16, 512), "ori_format": "ND", "range": ((10, 10), (16, 16), (512, 512))},
                "inner_product")
    set_current_compile_soc_info(test_args)


def test_gen_adc_case009(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (512,), "dtype": "float32", "format": "ND",
                 "ori_shape": (512,), "ori_format": "ND", "range": ((512, 512),)},
                {"shape": (16, 512, 32), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 512, 32), "ori_format": "ND", "range": ((16, 16), (512, 512), (32, 32))},
                {"shape": (1000000, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 512), "ori_format": "ND", "range": ((1000000, 1000000), (512, 512))},
                {"shape": (10,), "dtype": "int64", "format": "ND",
                 "ori_shape": (10,), "ori_format": "ND", "range": ((10, 10),)},
                {"shape": (10, 16, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (10, 16, 512), "ori_format": "ND", "range": ((10, 10), (16, 16), (512, 512))},
                "l2sqr")
    set_current_compile_soc_info(test_args)


def test_gen_adc_case010(test_args):
    """
    Compute adc distance.
    """
    set_current_compile_soc_info("Ascend710")
    with tbe.common.context.op_context.OpContext("dynamic"):
        gen_adc({"shape": (512,), "dtype": "float32", "format": "ND",
                 "ori_shape": (512,), "ori_format": "ND", "range": ((512, 512),)},
                {"shape": (16, 512, 32), "dtype": "float16", "format": "ND",
                 "ori_shape": (16, 512, 32), "ori_format": "ND", "range": ((16, 16), (512, 512), (32, 32))},
                {"shape": (1000000, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (1000000, 512), "ori_format": "ND", "range": ((1000000, 1000000), (512, 512))},
                {"shape": (10,), "dtype": "int64", "format": "ND",
                 "ori_shape": (10,), "ori_format": "ND", "range": ((10, 10),)},
                {"shape": (10, 16, 512), "dtype": "float16", "format": "ND",
                 "ori_shape": (10, 16, 512), "ori_format": "ND", "range": ((10, 10), (16, 16), (512, 512))},
                "inner_product")
    set_current_compile_soc_info(test_args)


test_gen_adc_case101 = {"params": [{"shape": (32,), "dtype": "float16", "format": "ND", "ori_shape": (32,),
                                    "ori_format": "ND", "range": ((32, 32),)},
                                   {"shape": (16, 256, 2), "dtype": "float16", "format": "ND",
                                    "ori_shape": (16, 256, 2), "ori_format": "ND",
                                    "range": ((16, 16), (256, 256), (2, 2))},
                                   {"shape": (1000000, 32), "dtype": "float16", "format": "ND",
                                    "ori_shape": (1000000, 32), "ori_format": "ND",
                                    "range": ((1000000, 1000000), (32, 32))},
                                   {"shape": (-1,), "dtype": "int32", "format": "ND",
                                    "ori_shape": (10,), "ori_format": "ND", "range": ((10, 10),)},
                                   {"shape": (-1, 16, 256), "dtype": "float16", "format": "ND",
                                    "ori_shape": (10, 16, 256), "ori_format": "ND",
                                    "range": ((10, 10), (16, 16), (256, 256))}],
                        "case_name": "gen_adc_case101",
                        "expect": "success",
                        "format_expect": [],
                        "support_expect": True}


ut_case.add_cust_test_func(test_func=test_gen_adc_case001)
ut_case.add_cust_test_func(test_func=test_gen_adc_case002)
ut_case.add_cust_test_func(test_func=test_gen_adc_case003)
ut_case.add_cust_test_func(test_func=test_gen_adc_case004)
ut_case.add_cust_test_func(test_func=test_gen_adc_case005)
ut_case.add_cust_test_func(test_func=test_gen_adc_case006)
ut_case.add_cust_test_func(test_func=test_gen_adc_case007)
ut_case.add_cust_test_func(test_func=test_gen_adc_case008)
ut_case.add_cust_test_func(test_func=test_gen_adc_case009)
ut_case.add_cust_test_func(test_func=test_gen_adc_case010)

ut_case.add_case(["Ascend710", "Ascend910"], test_gen_adc_case101)


if __name__ == '__main__':
    ut_case.run("Ascend710")
    exit(0)
