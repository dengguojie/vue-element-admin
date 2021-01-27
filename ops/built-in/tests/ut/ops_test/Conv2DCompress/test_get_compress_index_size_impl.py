#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_get_compress_index_size(test_arg):
    import types

    from tbe.dsl.static_schedule.conv_schedule import CceConvOp

    def freeVar(val):
        def nested():
            return val
        return nested.__closure__[0]

    def get_get_compress_index_size(conv_op, **freeVars):
        freeVars["int_ceil_div"] = get_int_ceil_div(conv_op)
        for obj1 in conv_op.schedule.__code__.co_consts:
            if isinstance(obj1, types.CodeType) and obj1.co_name == 'intrin_mapping':
                for obj2 in obj1.co_consts:
                    if isinstance(obj2, types.CodeType) and obj2.co_name == 'get_compress_index_size':
                        return types.FunctionType(obj2, globals(), None, None, tuple(freeVar(freeVars[name]) for name in obj2.co_freevars))

    def get_int_ceil_div(conv_op):
        for obj in conv_op.schedule.__code__.co_consts:
            if isinstance(obj, types.CodeType) and obj.co_name == 'int_ceil_div':
                return types.FunctionType(obj, globals(), None, None, None)



    def test_get_compress_index_size_case1():
        """存在尾块，且尾块不能被block_size整除的场景"""
        print("run test_get_compress_index_size_case1")
        conv_op = CceConvOp()
        conv_op.unzip_parameters["compress_tiling"] = [72, 2]
        weight_shape = [144, 31, 16, 32]
        mode_index_size = 8
        weight_shape_size = 2285568
        block_size = 24576
        func_get_compress_index_size = get_get_compress_index_size(conv_op, self=conv_op)
        assert(func_get_compress_index_size(weight_shape, mode_index_size, weight_shape_size, block_size) == 752)

    def test_get_compress_index_size_case2():
        """存在尾块，且尾块不能被block_size整除的场景"""
        conv_op = CceConvOp()
        conv_op.unzip_parameters["compress_tiling"] = [144, 4]
        weight_shape = [144, 31, 16, 32]
        mode_index_size = 8
        weight_shape_size = 2285568
        block_size = 32678
        func_get_compress_index_size = get_get_compress_index_size(conv_op, self=conv_op)
        assert(func_get_compress_index_size(weight_shape, mode_index_size, weight_shape_size, block_size) == 560)

    def test_get_compress_index_size_case3():
        """不存在尾块的场景"""
        conv_op = CceConvOp()
        conv_op.unzip_parameters["compress_tiling"] = [144, 31]
        weight_shape = [144, 31, 16, 32]
        mode_index_size = 8
        weight_shape_size = 2285568
        block_size = 24576
        func_get_compress_index_size = get_get_compress_index_size(conv_op, self=conv_op)
        assert(func_get_compress_index_size(weight_shape, mode_index_size, weight_shape_size, block_size) == 744)

    def test_get_compress_index_size_case4():
        """存在尾块，且尾块能被block_size整除的场景"""
        conv_op = CceConvOp()
        conv_op.unzip_parameters["compress_tiling"] = [144, 2]
        weight_shape = [144, 31, 16, 32]
        mode_index_size = 8
        weight_shape_size = 2285568
        block_size = 24576
        func_get_compress_index_size = get_get_compress_index_size(conv_op, self=conv_op)
        assert(func_get_compress_index_size(weight_shape, mode_index_size, weight_shape_size, block_size) == 744)

    test_get_compress_index_size_case1()
    test_get_compress_index_size_case2()
    test_get_compress_index_size_case3()
    test_get_compress_index_size_case4()

# ut_case.add_cust_test_func(test_func=test_get_compress_index_size)

if __name__ == "__main__":
    # ut_case.add_cust_test_func(test_func=test_get_compress_index_size)
    exit(0)
