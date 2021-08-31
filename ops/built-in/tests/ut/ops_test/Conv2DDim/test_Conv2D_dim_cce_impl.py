#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_dim_conv_cce(test_arg):
    import te
    from te import tvm
    from tbe.common import utils
    import te.lang.cce
    from te.lang.cce import cce_build_code
    from tbe.dsl import auto_schedule


    def _dim_conv_four2five_cce(input_shape, dtype, kernel_name="dim_conv_four2five_cce", need_build=True,
                                need_print=False):
        """
        Dimension conversion from 4D to 5D

        Parameters
        ----------
        input_shape : shape of input data

        dtype : input data type, only support float16

        kernel_name : cce kernel name, default value is "dim_conv_four2five_cce"


        need_buid : if need to build CCEC kernel, default value is True

        need_print : if need to print the ir, default value is False

        Returns
        -------
        None

        """
        utils.check_shape_rule(input_shape)

        check_list = ["float16"]
        if not (dtype.lower() in check_list):
            raise RuntimeError(
                "dim_conv_four2five_cce input only support %s while dtype is %s" % (",".join(check_list), dtype))

        input = tvm.placeholder(input_shape, name='input', dtype=dtype)

        with tvm.target.cce():
            raw_shape = input_shape
            output = te.lang.cce.compute_four2five(input, raw_shape)
            sch = auto_schedule(output)

        config = {"print_ir": need_print,
                "need_build": need_build,
                "name": kernel_name,
                "tensor_list": [input, output]}
        te.lang.cce.cce_build_code(sch, config)


    def _dim_conv_five2four_cce(input_shape, dtype, kernel_name="dim_conv_five2four_cce", need_build=True,
                                need_print=False):
        """
        Dimension conversion from 5D to 4D

        Parameters
        ----------
        input_shape : shape of input data

        dtype : input data type, only support float16

        kernel_name : cce kernel name, default value is "dim_conv_five2four_cce"


        need_buid : if need to build CCEC kernel, default value is True

        need_print : if need to print the ir, default value is False

        Returns
        -------
        None

        """
        utils.check_shape_rule(input_shape)

        check_list = ["float16"]
        if not (dtype.lower() in check_list):
            raise RuntimeError(
                "dim_conv_five2four_cce input only support %s while dtype is %s" % (",".join(check_list), dtype))

        input = tvm.placeholder(input_shape, name='input', dtype=dtype)

        with tvm.target.cce():
            raw_shape = input_shape
            output = te.lang.cce.compute_five2four(input, raw_shape)
            sch = auto_schedule(output)

        config = {"print_ir": need_print,
                "need_build": need_build,
                "name": kernel_name,
                "tensor_list": [input, output]}
        te.lang.cce.cce_build_code(sch, config)

    # 测试用例
    _dim_conv_four2five_cce((2, 32, 4, 4), "float16", kernel_name="dim_conv_four2five_2_32_4_4_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_four2five_cce((2, 32, 10, 4), "float16", kernel_name="dim_conv_four2five_2_32_10_4_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_four2five_cce((2, 32, 16, 128), "float16", kernel_name="dim_conv_four2five_2_32_16_128_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_four2five_cce((2, 36, 17, 129), "float16", kernel_name="dim_conv_four2five_2_36_17_129_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_four2five_cce((2, 36, 224, 224), "float16", kernel_name="dim_conv_four2five_2_36_224_224_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_four2five_cce((2, 36, 225, 225), "float16", kernel_name="dim_conv_four2five_2_36_255_255_float16",
                            need_build=True)


    # 测试用例
    _dim_conv_four2five_cce((2, 36, 4, 4), "float16", kernel_name="dim_conv_four2five_2_36_4_4_float16",
                            need_build=True)


    # 测试用例
    _dim_conv_four2five_cce((2, 36, 5, 5), "float16", kernel_name="dim_conv_four2five_2_36_5_5_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 32, 4, 4), "float16", kernel_name="dim_conv_five2four_2_32_4_4_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 32, 10, 4), "float16", kernel_name="dim_conv_five2four_2_32_10_4_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 32, 16, 128), "float16", kernel_name="dim_conv_five2four_2_32_16_128_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 32, 17, 129), "float16", kernel_name="dim_conv_five2four_2_32_17_129_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 36, 17, 129), "float16", kernel_name="dim_conv_five2four_2_36_17_129_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 36, 224, 224), "float16", kernel_name="dim_conv_five2four_2_36_224_224_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 36, 225, 225), "float16", kernel_name="dim_conv_five2four_2_36_255_255_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 36, 4, 4), "float16", kernel_name="dim_conv_five2four_2_36_4_4_float16",
                            need_build=True)

    # 测试用例
    _dim_conv_five2four_cce((2, 36, 5, 5), "float16", kernel_name="dim_conv_five2four_2_36_5_5_float16",
                            need_build=True)

    # 测试用例
    try:
        _dim_conv_five2four_cce((2, 32, 10, 4), "float32", kernel_name="dim_conv_five2four_2_32_10_4_float32",
                                need_build=True)
    except RuntimeError:
        pass

print("adding Conv2D dim testcases")
ut_case.add_cust_test_func(test_func=test_dim_conv_cce)