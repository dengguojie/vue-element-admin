from sch_test_frame.ut import OpUT
import warnings

import tbe
from tbe import tvm
from tbe.common.register import register_operator
from impl.mul import mul_compute
from impl.add import add_compute
from impl.rsqrt import rsqrt_compute
from impl.sqrt import sqrt_compute
from impl.real_div import real_div_compute

warnings.filterwarnings("ignore")

@register_operator("dsl_mul_add_rsqrt_fuse")
def dsl_mul_add_rsqrt_fuse(input_x, input_y, input_z, input_a, kernel_name = "mul_add_rsqrt_fuse"):
    input_1 = tvm.placeholder(input_x.get("shape"), name = "data0", dtype = input_x.get("dtype"))
    input_2 = tvm.placeholder(input_y.get("shape"), name = "data1", dtype = input_y.get("dtype"))
    input_3 = tvm.placeholder(input_z.get("shape"), name = "data2", dtype = input_z.get("dtype"))
    input_4 = tvm.placeholder(input_a.get("shape"), name = "data3", dtype = input_a.get("dtype"))
    res = mul_compute(input_1, input_2, {})
    res1 = add_compute(res, input_3, {})
    res2 = rsqrt_compute(res1, {})
    res3 = sqrt_compute(res1, {})
    res4 = real_div_compute(res3, input_4, {})
    res4 = real_div_compute(res4, res3, {})
    with tvm.target.cce():
        sch = tbe.dsl.auto_schedule([res2, res3, res4])
    config = {
        "print_ir": False,
        "name": kernel_name,
        "tensor_list": [input_1, input_2, input_3, input_4, res, res1, res2, res3, res4]
    }
    tbe.dsl.build(sch, config)


ut_case = OpUT("multi_outputs", "multi_outputs.test_multi_outputs_impl", "dsl_mul_add_rsqrt_fuse")

case1 = {"params": [
    {"shape": (256,256,1),"dtype": "float32"},
    {"shape": (256,1,1),"dtype": "float32"},
    {"shape": (256,1,1),"dtype": "float32"},
    {"shape": (256,256,128),"dtype": "float32"}],
    "case_name": "test_dsl_mul_add_rsqrt_fuse",
    "expect": "success",
    "support_expect": True
}

case2 = {"params": [
    {"shape": (256,42,1),"dtype": "float32"},
    {"shape": (256,1,1),"dtype": "float32"},
    {"shape": (256,1,1),"dtype": "float32"},
    {"shape": (256,42,128),"dtype": "float32"}],
    "case_name": "test_dsl_mul_add_rsqrt_fuse_002",
    "expect": "success",
    "support_expect": True
}


case4 = {
    "params": [{"shape": (1, 1, 1), "dtype": "float32", "format": "ND"},
               {"shape": (1, 1, 1), "dtype": "float32", "format": "ND"},
               {"shape": (1, 1, 1), "dtype": "float32", "format": "ND"},
               {"shape": (5, 5, 512), "dtype": "float32", "format": "ND"}
               ],
    "case_name": "test_dsl_mul_add_rsqrt_fuse_004",
    "expect": "success",
    "support_expect": True
}


ut_case.add_case(["all",], case1)
ut_case.add_case(["all",], case2)
ut_case.add_case(["all",], case4)

