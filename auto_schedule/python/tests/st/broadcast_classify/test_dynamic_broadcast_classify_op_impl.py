# # -*- coding:utf-8 -*-
import tbe
from sch_test_frame.ut import OpUT
from tbe import tvm
from tbe.common.utils import shape_util


def dsl_dync_vadd(x, y, z, kernel_name="dsl_dync_vadd"):
    input_dtype = x.get("dtype")

    ins = tbe.dsl.classify([x, y], "broadcast")
    schedules, tensors = [], []

    for (x, y) in ins:
        with tbe.dsl.compute():
            shape_x, shape_y = shape_util.variable_shape([x, y])
            data1 = tvm.placeholder(shape_x, name='data1', dtype=input_dtype)
            data2 = tvm.placeholder(shape_y, name='data2', dtype=input_dtype)

            shape_x, shape_y, shape_max = shape_util.broadcast_shapes(shape_x, shape_y,
                                                                      param_name_input1="input_x",
                                                                      param_name_input2="input_y")
            input1 = tbe.dsl.broadcast(data1, shape_max)
            input2 = tbe.dsl.broadcast(data2, shape_max)
            res = tbe.dsl.vadd(input1, input2)

            tensors.append((data1, data2, res))

        with tvm.target.cce():
            sch = tbe.dsl.auto_schedule(res)
        schedules.append(sch)

    config = {"name": kernel_name, "tensor_list": tensors}
    tbe.dsl.build(schedules, config)


ut_case = OpUT("vadd", "broadcast_classify.test_dynamic_broadcast_classify_op_impl", "dsl_dync_vadd")

case2 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(None, None), (2, 10), (0, 5)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(None, 2147483647), (5, 18), (0, 5)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(None, 2147483647), (2, 10), (0, 5)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_2",
    "expect":
        "success",
    "support_expect":
        True
}

case3 = {
    "params": [{
        "shape": (50, -1),
        "dtype": "float32",
        "range": [(50, 50), (1, 10000)]
    }, {
        "shape": (50, 10000),
        "dtype": "float32",
        "range": [(50, 50), (10000, 10000)]
    }, {
        "shape": (-1, 10),
        "dtype": "float32",
        "range": [(50, 50), (10, 10)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_3",
    "expect":
        "success",
    "support_expect":
        True
}

case4 = {
    "params": [{
        "shape": (1, -1),
        "dtype": "float32",
        "range": [(1, 1), (1, None)]
    }, {
        "shape": (-1, 1),
        "dtype": "float32",
        "range": [(1, None), (1, 1)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_4",
    "expect":
        "success",
    "support_expect":
        True
}

case5 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_5",
    "expect":
        "success",
    "support_expect":
        True
}

case6 = {
    "params": [{
        "shape": (1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, 1), (2, None), (2, None), (1, None)]
    }, {
        "shape": (-1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (2, None), (1, None)]
    }, {
        "shape": (-1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (2, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_6",
    "expect":
        "success",
    "support_expect":
        True
}

case7 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(0, 0), (1, None)]
    }, {
        "shape": (-1, 1),
        "dtype": "float32",
        "range": [(0, 0), (1, 1)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(0, 1), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_7",
    "expect":
        "success",
    "support_expect":
        True
}

case8 = {
    "params": [{
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, 10), (1, 10)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(11, 100), (2, 7)]
    }, {
        "shape": (-1, -1),
        "dtype": "float32",
        "range": [(1, 100), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_8",
    "expect":
        "success",
    "support_expect":
        True
}

case9 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, 10), (2, None), (10, 100)]
    }, {
        "shape": (1, 1, 11),
        "dtype": "float32",
        "range": [(1, 1), (1, 1), (11, 11)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, 10), (1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_9",
    "expect":
        "success",
    "support_expect":
        True
}

case10 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, 10), (1, None), (10, 100)]
    }, {
        "shape": (1, -1, 1),
        "dtype": "float32",
        "range": [(1, 1), (1, None), (1, 1)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, 10), (1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_10",
    "expect":
        "success",
    "support_expect":
        True
}

case11 = {
    "params": [{
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (1, None)]
    }, {
        "shape": (-1, 1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, 1), (1, None)]
    }, {
        "shape": (-1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (1, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_11",
    "expect":
        "success",
    "support_expect":
        True
}

case12 = {
    "params": [{
        "shape": (-1, -1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (2, None), (2, None), (1, None)]
    }, {
        "shape": (-1, -1, 1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (1, 1), (2, None), (1, None)]
    }, {
        "shape": (-1, -1, -1, -1, -1),
        "dtype": "float32",
        "range": [(1, None), (2, None), (1, None), (2, None), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_12",
    "expect":
        "success",
    "support_expect":
        True
}

case13 = {
    "params": [{
        "shape": (-1, -1, 1, -1, 20, -1),
        "dtype": "float32",
        "range": [(2, None), (1, None), (1, 1), (1, None), (20, 20), (1, None)]
    }, {
        "shape": (1, -1, -1, -1, 20, -1),
        "dtype": "float32",
        "range": [(1, 1), (1, None), (2, None), (1, None), (20, 20), (1, None)]
    }, {
        "shape": (-1, -1, -1, -1, -1, -1),
        "dtype": "float32",
        "range": [(2, None), (1, None), (2, None), (1, None), (20, 20), (1, None)]
    }],
    "case_name":
        "test_dynamic_broadcast_classify_op_impl_13",
    "expect":
        "success",
    "support_expect":
        True
}

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
