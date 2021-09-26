# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings
from te import platform as tbe_platform
from te import tvm
import te.lang.cce as tbe
from te.utils import shape_util

EPSLON = 1e-12
warnings.filterwarnings("ignore")

def layer_norm_beta_gamma_backprop_v2(input_dy, res_for_gamma, res1, res2, shape_gamma,
                                      kernel_name="layer_norm_x_backprop_v2"):
    input_shape1 = input_dy.get("shape")
    input_dtype1 = input_dy.get("dtype")
    input_shape2 = res_for_gamma.get("shape")
    input_dtype2 = res_for_gamma.get("dtype")
    attr = {"op_type": "layer_norm_beta_gamma_backprop_v2"}

    data_dy = tvm.placeholder(input_shape1, name='data_dy', dtype=input_dtype1, attrs=attr)
    data_gamma = tvm.placeholder(input_shape2, name='data_gamma', dtype=input_dtype2)

    data_x = tbe.vmul(data_gamma, data_dy)
    res1, res2 = tbe.tuple_sum([data_x, data_dy], (0, 1), keepdims=True)

    tensor_list = [data_dy, data_gamma, res1, res2]
    with tvm.target.cce():
        sch = tbe.auto_schedule([res1, res2])
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": tensor_list
    }
    tbe.cce_build_code(sch, config)


ut_case = OpUT("layer_norm_beta_gamma_backprop", "layer_norm_beta_gamma_backprop.test_layer_norm_beta_gamma_backprop_impl",
               "layer_norm_beta_gamma_backprop_v2")

case1 = {
    "params": [{"shape": (64, 1, 192), "dtype": "float32", "format": "ND", "ori_shape":(64, 1, 192),"ori_format":"ND"},
               {"shape": (64, 1, 192), "dtype": "float32", "format": "ND", "ori_shape":(64, 1, 192),"ori_format":"ND"},
               {"shape": (192,), "dtype": "float32", "format": "ND", "ori_shape":(192,),"ori_format":"ND"},
               {"shape": (192,), "dtype": "float32", "format": "ND", "ori_shape":(192,),"ori_format":"ND"},
               [192,]],
    "case_name": "test_layer_norm_grad_1",
    "expect": "success",
    "support_expect": True
}

compile_case = [
    case1,
]

for item in compile_case:
    ut_case.add_case(["Ascend910A"], case=item)


if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
