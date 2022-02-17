# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("UT_Concat")
def concat(input_values, output_data, axis, kernel_name="concat"):
    dtype_x = input_values[0].get("dtype")

    extra_params = {"axis": axis}
    ins = classify([input_values], "concat", extra_params)
    schedules, tensors = [], []
    for (input_x_, axis_) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([input_x_], "concat")
            input_tensors = []
            for index, shape in enumerate(shape_x):
                data = tvm.placeholder(shape, dtype=dtype_x, name=f"data_{index}")
                input_tensors.append(data)
            res = tbe.dsl.concat(input_tensors, axis_)

            tensors.append([*input_tensors, res])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)
    compile_info = tbe.dsl.base.operation.get_compile_info()
    import json
    print(json.dumps(compile_info))


ut_case = OpUT("UT_Concat", "concat.test_dynamic_concat_impl", "concat")

case0 = {
    "params": [[{
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }] * 2, {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, 0],
    "case_name":
    "test_dynamic_concat_0",
    "expect":
    "success",
    "support_expect":
    True
}

case1 = {
    "params": [[{
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }] * 2, {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, 1],
    "case_name":
    "test_dynamic_concat_1",
    "expect":
    "success",
    "support_expect":
    True
}

case2 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "float16",
        "range": [(1, None)] * 3
    }] * 2, {
        "shape": (-1, ) * 3,
        "dtype": "float16",
        "range": [(1, None)] * 3
    }, 0],
    "case_name":
    "test_dynamic_concat_2",
    "expect":
    "success",
    "support_expect":
    True
}

case3 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "int32",
        "range": [(1, None)] * 3
    }] * 2, {
        "shape": (-1, ) * 3,
        "dtype": "int32",
        "range": [(1, None)] * 3
    }, 1],
    "case_name":
    "test_dynamic_concat_3",
    "expect":
    "success",
    "support_expect":
    True
}

case4 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "float32",
        "range": [(1, None)] * 3
    }] * 2, {
        "shape": (-1, ) * 3,
        "dtype": "float32",
        "range": [(1, None)] * 3
    }, 2],
    "case_name":
    "test_dynamic_concat_4",
    "expect":
    "success",
    "support_expect":
    True
}

case5 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "uint64",
        "range": [(1, None)] * 3
    }] * 4, {
        "shape": (-1, ) * 3,
        "dtype": "uint64",
        "range": [(1, None)] * 3
    }, 1],
    "case_name":
    "test_dynamic_concat_5",
    "expect":
    "success",
    "support_expect":
    True
}

case6 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "int64",
        "range": [(1, None)] * 3
    }] * 5, {
        "shape": (-1, ) * 3,
        "dtype": "int64",
        "range": [(1, None)] * 3
    }, 1],
    "case_name":
    "test_dynamic_concat_6",
    "expect":
    "success",
    "support_expect":
    True
}

case7 = {
    "params": [[{
        "shape": (-1, ) * 3,
        "dtype": "uint8",
        "range": [(1, None)] * 3
    }] * 2, {
        "shape": (-1, ) * 3,
        "dtype": "uint8",
        "range": [(1, None)] * 3
    }, 2],
    "case_name":
    "test_dynamic_concat_7",
    "expect":
    "success",
    "support_expect":
    True
}

case8 = {
    "params": [[{
        "shape": (256, 3120),
        "dtype": "float32",
        "range": [(1, None)] * 2
    }] * 2, {
        "shape": (-1, ) * 2,
        "dtype": "float32",
        "range": [(1, None)] * 2
    }, 1],
    "case_name":
    "test_dynamic_concat_8",
    "expect":
    "success",
    "support_expect":
    True
}

case9 = {
    "params": [[{
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }] * 1, {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, 1],
    "case_name":
    "test_dynamic_concat_9",
    "expect":
    "success",
    "support_expect":
    True
}

case10 = {
    "params": [[{
        "shape": (-2, ),
        "dtype": "float16",
        "range": [(1, None)]
    }] * 3, {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, 3],
    "case_name":
    "test_dynamic_concat_10",
    "expect":
    "success",
    "support_expect":
    True
}

case11 = {
    "params": [[{
        "shape": (-2, ),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (2, ),
        "dtype": "float16",
        "range": [(2, 2)]
    }], {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, 0],
    "case_name":
    "test_dynamic_concat_11",
    "expect":
    "success",
    "support_expect":
    True
}

case12 = {
    "params": [[{
        "shape": (-2, ),
        "dtype": "float16",
        "range": [(1, None)]
    }, {
        "shape": (3, 4, 100),
        "dtype": "float16",
        "range": [(3, 3), (4, 4), (100, 100)]
    }], {
        "shape": (-1, ) * 2,
        "dtype": "float16",
        "range": [(1, None)] * 2
    }, -1],
    "case_name":
    "test_dynamic_concat_12",
    "expect":
    "success",
    "support_expect":
    True
}

ut_case.add_case(["all"], case0)
ut_case.add_case(["all"], case1)
ut_case.add_case(["all"], case2)
ut_case.add_case(["all"], case3)
ut_case.add_case(["all"], case4)
ut_case.add_case(["all"], case5)
ut_case.add_case(["all"], case6)
ut_case.add_case(["all"], case7)
ut_case.add_case(["all"], case8)
ut_case.add_case(["all"], case9)
ut_case.add_case(["all"], case10)
ut_case.add_case(["all"], case11)
ut_case.add_case(["all"], case12)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
