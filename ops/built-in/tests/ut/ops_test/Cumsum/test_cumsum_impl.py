"""
todo copyright

Cumsum ut case
"""
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("Cumsum", "impl.dynamic.cumsum", "cumsum")

case1 = {
    "params": [
        {
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        },  #x
        {
            "shape": (1,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "value": [1]
        },  #axis 
        {
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }  #y
    ],
    "case_name": "Cumsum_1",
    "expect": "success",
    "support_expect": True
}

case2 = {
    "params": [
        {
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        },  #x
        {
            "shape": (1,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "value": [2]
        },  #axis 
        {
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }  #y
    ],
    "case_name": "Cumsum_2",
    "expect": "success",
    "support_expect": True
}

ut_case.add_case(["Ascend910A", "Ascend310"], case1)
ut_case.add_case(["Ascend910A", "Ascend310"], case2)


def calc_expect_func(x, y, z):
    x_value = x.get("value")
    res = np.cumsum(x_value, axis=y.get("value")[0])
    return (res,)

precision_case1 = {
       "params": [{
        "shape": (-1, 28, 52),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(16, 200),],
        "ori_shape": (16, 28, 52),
        "run_shape": (16, 28, 52),
        "param_type": "input",
        "value": np.ones((16, 28, 52), dtype=np.float32)
    }, {
        "shape": (1,),
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(1, 1)],
        "run_shape": (1,),
        "ori_shape": (1,),
        "param_type": "input",
        "value_need_in_tiling": True,
        "value": np.array([1]),
        "const_value": [1],
    }, {
        "shape": (-1, 28, 52),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(16, 200),],
        "run_shape": (16,28, 52),
        "ori_shape": (16, 28, 52),
        "param_type": "output"
    }],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
}

precision_case2 = {
       "params": [{
        "shape": (-1, 28, 52),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(16, 200),],
        "ori_shape": (16, 28, 52),
        "run_shape": (16, 28, 52),
        "param_type": "input",
        "value": np.ones((16, 28, 52), dtype=np.float32)
    }, {
        "shape": (1,),
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(1, 1)],
        "run_shape": (1,),
        "ori_shape": (1,),
        "param_type": "input",
        "value_need_in_tiling": True,
        "value": np.array([2]),
        "const_value": [2],
    }, {
        "shape": (-1, 28, 52),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(16, 200),],
        "run_shape": (16,28, 52),
        "ori_shape": (16, 28, 52),
        "param_type": "output"
    }],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
}

precision_case3 = {
       "params": [{
        "shape": (-1, 1),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(1, 1000),],
        "ori_shape": (-1, 1),
        "run_shape": (16,1),
        "param_type": "input",
        "value": np.ones((16, 1), dtype=np.float32)
    }, {
        "shape": (1,),
        "dtype": "int32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(1, 10)],
        "run_shape": (1,),
        "ori_shape": (1,),
        "param_type": "input",
        "value_need_in_tiling": True,
        "value": np.array([1]),
        "const_value": [1],
    }, {
    "shape": (-1, 1),
        "dtype": "float32",
        "format": "ND",
        "ori_format": "ND",
        "range": [(1, 1000)],
        "ori_shape": (-1, 1),
        "run_shape": (16, 1),
        "param_type": "output"
    }],
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001),
}

# ut_case.add_precision_case("all", precision_case1)
# ut_case.add_precision_case("all", precision_case2)
# ut_case.add_precision_case("all", precision_case3)


def test_import_const(_):
    import sys
    import importlib
    importlib.reload(sys.modules.get("impl.dynamic.cum_computer"))
    import impl.dynamic.cum_computer as computer
    assert computer.Constant.MAX_COMPUTE_SIZE == 256 * 255

def test_check_support_1(_):
    from impl.dynamic.cumsum import check_supported
    res  = check_supported({
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, {
            "shape": (1,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "value": [1]
        },{
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, True, True)
    assert not res[0]

def test_check_support_2(_):
    from impl.dynamic.cumsum import check_supported
    res  = check_supported({
            "shape": (-1, 28, 52),
            "dtype": "int8",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, {
            "shape": (1,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "value": [1]
        },{
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, False, False)
    assert not res[0]

def test_check_support_3(_):
    from impl.dynamic.cumsum import check_supported
    res  = check_supported({
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, {
            "shape": (1,),
            "dtype": "int32",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "value": [1]
        },{
            "shape": (-1, 28, 52),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (16, 28, 52),
            "ori_format": "ND"
        }, False, False)
    assert res[0]

ut_case.add_cust_test_func(test_func=test_import_const)
ut_case.add_cust_test_func(test_func=test_check_support_1)
ut_case.add_cust_test_func(test_func=test_check_support_2)
ut_case.add_cust_test_func(test_func=test_check_support_3)

if __name__ == "__main__":
    simulator_lib_path = "/usr/local/Ascend/latest/toolkit/tools/simulator/"
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
