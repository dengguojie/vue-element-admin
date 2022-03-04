# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("UT_Split_d")
def split_d(input_s, output, split_dim, num_split, kernel_name="dsl_split_d"):
    dtype_x = input_s.get("dtype")

    extra_params = {"avg_split": True, "num_split": num_split}
    ins = classify([input_s, split_dim], "split", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_, size_splits_) in ins:
        with tbe.dsl.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            res = tbe.dsl.split(input_tensors, axis_, size_splits)

            tensors.append([input_tensors, *res])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("UT_Split_d", "split.test_dynamic_split_impl", "split_d")


@register_operator("UT_Split")
def split(split_dim, input_s, y, num_split, kernel_name="dsl_split"):
    dtype_x = input_s.get("dtype")

    input0 = tvm.placeholder((1,), dtype=split_dim.get("dtype"), name="input0")

    extra_params = {"avg_split": True, "num_split": num_split}
    ins = classify([input_s, split_dim], "split", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_, size_splits_) in ins:
        with tbe.dsl.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            res = tbe.dsl.split(input_tensors, axis_, size_splits)

            tensors.append([input0, input_tensors, *res])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case1 = OpUT("UT_Split", "split.test_dynamic_split_impl", "split")


@register_operator("UT_Split_v")
def split_v(input_s, size_splits, split_dim, y, num_split, kernel_name="dsl_split_v"):
    dtype_x = input_s.get("dtype")

    size_type = size_splits.get("dtype")
    split_size = tbe.dsl.var("split_size", dtype=size_type)
    size_shape = (split_size,)
    input1 = tvm.placeholder(size_shape, dtype=size_type, name="input1")

    input2 = tvm.placeholder((1,), dtype=split_dim.get("dtype"), name="input2")

    extra_params = {"avg_split": False, "num_split": num_split}
    ins = classify([input_s, split_dim, size_splits], "split", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_, size_splits_) in ins:
        with tbe.dsl.compute():
            shape_x, size_splits = shape_util.variable_shape([input_x_, size_splits_], "split")
            input_tensors = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            res = tbe.dsl.split(input_tensors, axis_, size_splits)

            tensors.append([input_tensors, input1, input2, *res])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case2 = OpUT("UT_Split_v", "split.test_dynamic_split_impl", "split_v")

# split_d
# b8, b16, b32, b64
# dims: 1, 2, 5, 8
# splits: 1, 2, 5, 63
# split_dims: first, middle, end, -first, -second, -middle
# shape 0, shape -2

# split

# split_v

case0 = {
    "params": [{
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }] * 2, 1, 1],
    "case_name": "test_dynamic_split_d_0",
    "expect": "success",
    "support_expect": True
}

case1 = {
    "params": [{
        "shape": (-1, ) * 1,
        "dtype": "float16",
        "range": [(1, None)] * 1
    }, [{
        "shape": (-1, ) * 1,
        "dtype": "float16",
        "range": [(1, None)] * 1
    }] * 1, 0, 1],
    "case_name": "test_dynamic_split_d_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (-1, ) * 1,
        "dtype": "int8",
        "range": [(1, None)] * 1
    }, [{
        "shape": (-1, ) * 1,
        "dtype": "int8",
        "range": [(1, None)] * 1
    }] * 2, 0, 2],
    "case_name": "test_dynamic_split_d_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{
        "shape": (-1, ) * 5,
        "dtype": "float32",
        "range": [(1, None)] * 5
    }, [{
        "shape": (-1, ) * 5,
        "dtype": "float32",
        "range": [(1, None)] * 5
    }] * 3, 1, 3],
    "case_name": "test_dynamic_split_d_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{
        "shape": (-1, ) * 8,
        "dtype": "int64",
        "range": [(1, None)] * 8
    }, [{
        "shape": (-1, ) * 8,
        "dtype": "int64",
        "range": [(1, None)] * 8
    }] * 63, -1, 63],
    "case_name": "test_dynamic_split_d_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (0, None), (1, None)]
    }, [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(1, None), (0, None), (1, None)]
    }] * 4, 1, 4],
    "case_name": "test_dynamic_split_d_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(0, None), (1, None), (1, None)]
    }, [{
        "shape": (-1, -1, -1),
        "dtype": "float16",
        "range": [(0, None), (1, None), (1, None)]
    }] * 4, 1, 4],
    "case_name": "test_dynamic_split_d_6",
    "expect": "success",
    "support_expect": True
}

case7 = {
    "params": [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }, [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }] * 2, 1, 2],
    "case_name": "test_dynamic_split_d_7",
    "expect": "success",
    "support_expect": True
}

case8 = {
    "params": [{
        "shape": (100, 200, 300),
        "dtype": "float16",
        "range": [(100, 100), (200, 200), (300, 300)]
    }, [{
        "shape": (100, 50, 300),
        "dtype": "float16",
        "range": [(100, 100), (50, 50), (300, 300)]
    }] * 4, 1, 4],
    "case_name": "test_dynamic_split_d_8",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend710"], case0)
ut_case.add_case(["Ascend910A", "Ascend710"], case1)
ut_case.add_case(["Ascend910A", "Ascend710"], case2)
ut_case.add_case(["Ascend910A", "Ascend710"], case3)
ut_case.add_case(["Ascend910A", "Ascend710"], case4)
ut_case.add_case(["Ascend910A", "Ascend710"], case5)
ut_case.add_case(["Ascend910A", "Ascend710"], case6)
ut_case.add_case(["Ascend910A", "Ascend710"], case7)
ut_case.add_case(["Ascend910A", "Ascend710"], case8)

case100 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, {
        "shape": (-1, ) * 2,
        "dtype": "uint8",
        "range": [(1, None)] * 2
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "uint8",
        "range": [(1, None)] * 2
    }] * 2, 1],
    "case_name": "test_dynamic_split_100",
    "expect": "success",
    "support_expect": True
}

case101 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, {
        "shape": (-1, ) * 2,
        "dtype": "float32",
        "range": [(1, None)] * 2
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "float32",
        "range": [(1, None)] * 2
    }] * 2, 2],
    "case_name": "test_dynamic_split_101",
    "expect": "success",
    "support_expect": True
}

case102 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, {
        "shape": (-1, -1),
        "dtype": "uint64",
        "range": [(0, None), (1, None)]
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "uint64",
        "range": [(0, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_102",
    "expect": "success",
    "support_expect": True
}

case103 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, {
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }, [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }] * 2, 2],
    "case_name": "test_dynamic_split_103",
    "expect": "success",
    "support_expect": True
}

case104 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, {
        "shape": (100, 5880),
        "dtype": "uint64",
        "range": [(100, 100), (5880, 5880)]
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "uint64",
        "range": [(1, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_104",
    "expect": "success",
    "support_expect": True
}

case105 = {
    "params": [{
        "shape": (1, ),
        "dtype": "int32",
        "value": [1],
        "range": [(1, 1)]
    }, {
        "shape": (100, 5880),
        "dtype": "uint64",
        "range": [(100, 100), (5880, 5880)]
    }, [{
        "shape": (100, 2940),
        "dtype": "uint64",
        "range": [(1, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_105",
    "expect": "success",
    "support_expect": True
}

ut_case1.add_case(["Ascend910A", "Ascend710"], case100)
ut_case1.add_case(["Ascend910A", "Ascend710"], case101)
ut_case1.add_case(["Ascend910A", "Ascend710"], case102)
ut_case1.add_case(["Ascend910A", "Ascend710"], case103)
ut_case1.add_case(["Ascend910A", "Ascend710"], case104)
ut_case1.add_case(["Ascend910A", "Ascend710"], case105)

case200 = {
    "params": [{
        "shape": (-1, ) * 2,
        "dtype": "int8",
        "range": [(1, None)] * 2
    }, {
        "shape": (-1, ),
        "dtype": "int32",
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "int8",
        "range": [(1, None)] * 2
    }] * 1, 1],
    "case_name": "test_dynamic_split_200",
    "expect": "success",
    "support_expect": True
}

case201 = {
    "params": [{
        "shape": (-1, ) * 2,
        "dtype": "float32",
        "range": [(1, None)] * 2
    }, {
        "shape": (-1, ),
        "dtype": "int32",
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, [{
        "shape": (-1, ) * 2,
        "dtype": "float32",
        "range": [(1, None)] * 2
    }] * 2, 2],
    "case_name": "test_dynamic_split_201",
    "expect": "success",
    "support_expect": True
}

case202 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(0, None), (1, None)]
    }, {
        "shape": (-1, ),
        "dtype": "int32",
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, [{
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(0, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_202",
    "expect": "success",
    "support_expect": True
}

case203 = {
    "params": [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }, {
        "shape": (-1, ),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "float16",
        "range": [(1, 1)]
    }, [{
        "shape": (-2, ),
        "dtype": "float16",
        "range": []
    }] * 2, 2],
    "case_name": "test_dynamic_split_203",
    "expect": "success",
    "support_expect": True
}

case204 = {
    "params": [{
        "shape": (100, 4488),
        "dtype": "int64",
        "range": [(100, 100), (4488, 4488)]
    }, {
        "shape": (-1, ),
        "dtype": "int32",
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "int32",
        "range": [(1, 1)]
    }, [{
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(1, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_204",
    "expect": "success",
    "support_expect": True
}

case205 = {
    "params": [{
        "shape": (100, 4488),
        "dtype": "int64",
        "range": [(100, 100), (4488, 4488)]
    }, {
        "shape": (2, ),
        "dtype": "int32",
        "value": [4400, 88],
        "range": [(1, None)]
    }, {
        "shape": (1, ),
        "dtype": "int32",
        "value": [1],
        "range": [(1, 1)]
    }, [{
        "shape": (-1, -1),
        "dtype": "int64",
        "range": [(1, None), (1, None)]
    }] * 2, 2],
    "case_name": "test_dynamic_split_205",
    "expect": "success",
    "support_expect": True
}

ut_case2.add_case(["Ascend910A", "Ascend710"], case200)
ut_case2.add_case(["Ascend910A", "Ascend710"], case201)
ut_case2.add_case(["Ascend910A", "Ascend710"], case202)
ut_case2.add_case(["Ascend910A", "Ascend710"], case203)
ut_case2.add_case(["Ascend910A", "Ascend710"], case204)
ut_case2.add_case(["Ascend910A", "Ascend710"], case205)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    ut_case1.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
    ut_case2.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
