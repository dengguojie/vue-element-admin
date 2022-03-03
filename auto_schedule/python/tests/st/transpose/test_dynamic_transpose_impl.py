# # -*- coding:utf-8 -*-
import numpy as np
import tbe
from sch_test_frame.common import precision_info
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.register import register_operator
from tbe.common.utils import shape_util
from tbe.dsl import classify


@register_operator("Transpose")
def transpose(x, y, perm, kernel_name="transpose"):
    dtype_x = x.get("dtype")

    extra_params = {"axes": perm}
    ins = classify([x], "transpose", extra_params)
    schedules, tensors = [], []
    for (input_x_, perm_) in ins:
        with tbe.dsl.compute():
            shape_x = shape_util.variable_shape([input_x_], "transpose")
            data_x = tvm.placeholder(shape_x, dtype=dtype_x, name="data_x")
            res = tbe.dsl.transpose(data_x, perm_)

            tensors.append([data_x, res])
        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)
    compile_info = tbe.dsl.base.operation.get_compile_info()
    import json
    print(json.dumps(compile_info))


ut_case = OpUT("Transpose", "transpose.test_dynamic_transpose_impl", "transpose")

case0 = {
    "params": [{
        "shape": (-1,) * 2, "dtype": "float16", "range": [(1, None)] * 2
    }, {"shape": (-1,) * 2, "dtype": "float16", "range": [(1, None)] * 2
        }, [1, 0]],
    "case_name": "test_dynamic_transpose_0",
    "expect": "success",
    "support_expect": True
}

case1 = {
    "params": [{
        "shape": (-1,) * 2, "dtype": "float16", "range": [(1, None)] * 2
    }, {"shape": (-1,) * 2, "dtype": "float16", "range": [(1, None)] * 2
        }, [0, 1]],
    "case_name": "test_dynamic_transpose_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [{
        "shape": (-1,) * 3, "dtype": "float16", "range": [(1, None)] * 3
    }, {"shape": (-1,) * 3, "dtype": "float16", "range": [(1, None)] * 3
        }, [2, 0, 1]],
    "case_name": "test_dynamic_transpose_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [{
        "shape": (-1,) * 3, "dtype": "float16", "range": [(1, None)] * 3
    }, {"shape": (-1,) * 3, "dtype": "float16", "range": [(1, None)] * 3
        }, [2, 1, 0]],
    "case_name": "test_dynamic_transpose_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
        }, [3, 2, 1, 0]],
    "case_name": "test_dynamic_transpose_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
        }, [0, 2, 1, 3]],
    "case_name": "test_dynamic_transpose_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{
        "shape": (10, 100, 186, 20), "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (10, 186, 100, 20), "dtype": "float16", "range": [(1, None)] * 4
        }, [0, 2, 1, 3]],
    "case_name": "test_dynamic_transpose_6",
    "expect": "success",
    "support_expect": True
}

case7 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float32", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float32", "range": [(1, None)] * 4
        }, [3, 2, 1, 0]],
    "case_name": "test_dynamic_transpose_7",
    "expect": "success",
    "support_expect": True
}

case8 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float32", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float32", "range": [(1, None)] * 4
        }, [0, 2, 1, 3]],
    "case_name": "test_dynamic_transpose_8",
    "expect": "success",
    "support_expect": True
}

case9 = {
    "params": [{
        "shape": (256, 8, 16, 64), "dtype": "float32", "range": [(1, None)] * 4
    }, {"shape": (8, 256, 64, 16), "dtype": "float32", "range": [(1, None)] * 4
        }, [1, 0, 3, 2]],
    "case_name": "test_dynamic_transpose_9",
    "expect": "success",
    "support_expect": True
}

case10 = {
    "params": [{
        "shape": (-1, -1, -1, 1, -1), "dtype": "float16", "range": [(1, None)] * 5
    }, {"shape": (-1, 1, -1, -1, -1), "dtype": "float16", "range": [(1, None)] * 5
        }, [4, 3, 1, 2, 0]],
    "case_name": "test_dynamic_transpose_10",
    "expect": "success",
    "support_expect": True
}

case11 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
        }, [2, 1, 0, 3]],
    "case_name": "test_dynamic_transpose_11",
    "expect": "success",
    "support_expect": True
}

case12 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
        }, [2, 1, 3, 0]],
    "case_name": "test_dynamic_transpose_12",
    "expect": "success",
    "support_expect": True
}

case13 = {
    "params": [{
        "shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
    }, {"shape": (-1,) * 4, "dtype": "float16", "range": [(1, None)] * 4
        }, [1, 0, 3, 2]],
    "case_name": "test_dynamic_transpose_13",
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
ut_case.add_case(["Ascend910A", "Ascend710"], case9)
ut_case.add_case(["Ascend910A", "Ascend710"], case10)
ut_case.add_case(["Ascend910A", "Ascend710"], case11)
ut_case.add_case(["Ascend910A", "Ascend710"], case12)
ut_case.add_case(["Ascend910A", "Ascend710"], case13)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
