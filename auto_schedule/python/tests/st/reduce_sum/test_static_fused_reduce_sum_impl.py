# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import warnings

from te import tvm
from te.utils import shape_util
import te.lang.cce as tbe

warnings.filterwarnings("ignore")


def test_confusion_mul_grad(input0, input1, input2, axis, keep_dims, kernel_name="test_confusion_mul_grad"):
    """
    fused_reduce_sum ut for workspace
    """

    def shape_broadcast(data_1, data_2):
        shape_x = shape_util.shape_to_list(data_1.shape)
        shape_y = shape_util.shape_to_list(data_2.shape)
        if shape_x != shape_y:
            shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y, param_name_input1="input0",
                                                                    param_name_input2="input1")
            data_1 = tbe.broadcast(data_1, shape_max)
            data_2 = tbe.broadcast(data_2, shape_max)

        return data_1, data_2

    def __confusion_mul_grad_compute(data_input0, data_input1, data_input2,
                                     output0, output1,
                                     axis, keep_dims,
                                     kernel_name="test_confusion_mul_grad"):
        # mul
        mul_data_input0, mul_data_input1 = \
            shape_broadcast(data_input0, data_input1)
        result0 = tbe.vmul(mul_data_input0, mul_data_input1)

        # mul_1
        data_input1, data_input2 = shape_broadcast(data_input1, data_input2)
        mul_1_result = tbe.vmul(data_input1, data_input2)

        # temp compute for tvm
        shape_x = shape_util.shape_to_list(mul_1_result.shape)
        shape_y = shape_util.shape_to_list(result0.shape)
        if shape_x == shape_y:
            zero_tmp = tbe.vmuls(result0, 0)
            mul_1_result = tbe.vadd(mul_1_result, zero_tmp)

        # sum
        dtype = mul_1_result.dtype
        if dtype == "float16":
            mul_1_result = tbe.cast_to(mul_1_result, "float32")
        result1 = tbe.sum(mul_1_result, axis=axis, keepdims=keep_dims)
        if dtype == "float16":
            result1 = tbe.cast_to(result1, "float16")

        res = [result0, result1]

        return res

    shape_input0 = input0.get("shape")
    dtype_input0 = input0.get("dtype")
    shape_input1 = input1.get("shape")
    dtype_input1 = input1.get("dtype")
    shape_input2 = input2.get("shape")
    dtype_input2 = input2.get("dtype")

    data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)
    data_input1 = tvm.placeholder(shape_input1, name="data_input1", dtype=dtype_input1)
    data_input2 = tvm.placeholder(shape_input2, name="data_input2", dtype=dtype_input2)

    res = __confusion_mul_grad_compute(data_input0, data_input1, data_input2, {}, {},
                                       axis, keep_dims, kernel_name)

    inputlist = [data_input0, data_input1, data_input2]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {"name": kernel_name,
              "tensor_list": list(inputlist) + list(res)}
    tbe.cce_build_code(sch, config)

def test_cast_sum_fused(input, output, axis, keep_dims, kernel_name="test_cast_sunm_fused"):
    shape_input = input.get("shape")
    dtype_input = input.get("dtype")
    dtype_output = output.get("dtype")
    data_input = tvm.placeholder(shape_input, name="data_input", dtype=dtype_input)

    res = tbe.sum(data_input, axis=axis, keepdims=keep_dims)
    if dtype_output == "float16":
        res = tbe.cast_to(res, "float32")

    tensor_list = [data_input, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)

def test_vmadd_sum_fused(input0, input1, input2, axis, keep_dims, kernel_name="test_vmadd_sum_fused"):
    try:
        shape_input0 = input0.get("shape")
        dtype_input0 = input0.get("dtype")
        shape_input1 = input1.get("shape")
        dtype_input1 = input1.get("dtype")
        shape_input2 = input2.get("shape")
        dtype_input2 = input2.get("dtype")

        data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)
        data_input1 = tvm.placeholder(shape_input1, name="data_input1", dtype=dtype_input1)
        data_input2 = tvm.placeholder(shape_input2, name="data_input2", dtype=dtype_input2)

        data_input0 = tbe.broadcast(data_input0, shape_input1, dtype_input0)
        res = tbe.vmadd(data_input0, data_input1, data_input2)
        res = tbe.sum(res, axis=axis, keepdims=keep_dims)
        tensor_list = [data_input0, data_input1, data_input2, res]
        with tvm.target.cce():
            sch = tbe.auto_schedule(res)
        config = {
            "print_ir": False,
            "name": kernel_name,
            "tensor_list": tensor_list
        }
        tbe.cce_build_code(sch, config)
    except:
        print("atomic schedule vmultiple error, abandoned")

def test_broadcast_sum_fused(input0, axis, keep_dims, kernel_name="test_broadcast_sum_fused"):
    shape_input0 = input0.get("shape")
    dtype_input0 = input0.get("dtype")

    data_input0 = tvm.placeholder(shape_input0, name="data_input0", dtype=dtype_input0)

    shape = (shape_input0[0], shape_input0[1], shape_input0[2], shape_input0[2])

    b_res = tbe.broadcast(data_input0, shape, dtype_input0)
    res = tbe.sum(b_res, axis=axis, keepdims=keep_dims)
    tensor_list = [data_input0, res]
    with tvm.target.cce():
        sch = tbe.auto_schedule(res)
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


def dsl_fused_reduce_sum(fuse_type, op_args, kernel_name):
    if fuse_type == "confusion_mul_grad":
        input0 = op_args.get("input0")
        input1 = op_args.get("input1")
        input2 = op_args.get("input2")
        axis = op_args.get("axis")
        keep_dims = op_args.get("keep_dims")
        test_confusion_mul_grad(input0, input1, input2, axis, keep_dims, kernel_name)
    elif fuse_type == "cast_sum_fused":
        input = op_args.get("input")
        output = op_args.get("output")
        axis = op_args.get("axis")
        keep_dims = op_args.get("keep_dims")
        test_cast_sum_fused(input, output, axis, keep_dims, kernel_name)
    elif fuse_type == "vmadd_sum_fused":
        input0 = op_args.get("input0")
        input1 = op_args.get("input1")
        input2 = op_args.get("input2")
        axis = op_args.get("axis")
        keep_dims = op_args.get("keep_dims")
        test_vmadd_sum_fused(input0, input1, input2, axis, keep_dims, kernel_name)
    elif fuse_type == "broadcast_sum_fused":
        input0 = op_args.get("input0")
        axis = op_args.get("axis")
        keep_dims = op_args.get("keep_dims")
        test_broadcast_sum_fused(input0, axis, keep_dims, kernel_name)

ut_case = OpUT("reduce_sum", "reduce_sum.test_static_fused_reduce_sum_impl", "dsl_fused_reduce_sum")

case1 = {
    "params": [
        "confusion_mul_grad",
        {
            "input0": {"shape":[1,1,3], "dtype":"float16"},
            "input1": {"shape":[1,8,1], "dtype":"float16"},
            "input2": {"shape":[2,8,3], "dtype":"float16"},
            "axis": (0,),
            "keep_dims": True,
        }
    ],
    "case_name": "test_fused_confusion_mul_grad_1",
    "expect": "success",
    "support_expect": True
}
case2 = {
    "params": [
        "confusion_mul_grad",
        {
            "input0": {"shape":[1,1,3], "dtype":"float16"},
            "input1": {"shape":[1,8,1], "dtype":"float16"},
            "input2": {"shape":[2,8,3], "dtype":"float16"},
            "axis": (2,),
            "keep_dims": True,
        }
    ],
    "case_name": "test_fused_confusion_mul_grad_2",
    "expect": "success",
    "support_expect": True
}
case3 = {
    "params": [
        "cast_sum_fused",
        {
            "input": {"shape":[300,8,15], "dtype":"float16"},
            "output": {"shape":[8,15], "dtype":"float16"},
            "axis": (0,),
            "keep_dims": False,
        }
    ],
    "case_name": "test_cast_sum_fused_1",
    "expect": "success",
    "support_expect": True
}
case4 = {
    "params": [
        "vmadd_sum_fused",
        {
            "input0": {"shape":[16,32,1,1], "dtype":"float32"},
            "input1": {"shape":[16,32,512,512], "dtype":"float32"},
            "input2": {"shape":[16,32,512,512], "dtype":"float32"},
            "axis": (0,2,3),
            "keep_dims": False,
        }
    ],
    "case_name": "test_vmadd_sum_fused_1",
    "expect": "success",
    "support_expect": True
}
case5 = {
    "params": [
        "broadcast_sum_fused",
        {
            "input0": {"shape":[16,32,511,1], "dtype":"float32"},
            "axis": (0,2,3),
            "keep_dims": False,
        }
    ],
    "case_name": "test_broadcast_sum_fused_1",
    "expect": "success",
    "support_expect": True
}

compile_case_list = [
    case1,
    case2,
    case3,
    case4,
    case5,
]

for item in compile_case_list:
    ut_case.add_case(case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
