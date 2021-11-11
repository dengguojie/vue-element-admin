from op_test_frame.ut import OpUT
from te import platform as cce_conf
import tbe

ut_case = OpUT("IsFinite", "impl.dynamic.is_finite", "is_finite")


def is_finite_fp16_001(test_args):
    from impl.dynamic import is_finite
    shape = (3000,)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float16", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


def is_finite_fp16_002(test_args):
    from impl.dynamic import is_finite
    shape = (3000, 100)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float16", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


def is_finite_fp16_003(test_args):
    from impl.dynamic import is_finite
    shape = (3000, 100, 100)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float16", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


def is_finite_fp32_001(test_args):
    from impl.dynamic import is_finite
    shape = (3000,)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float32", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


def is_finite_fp32_002(test_args):
    from impl.dynamic import is_finite
    shape = (3000, 100)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float32", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


def is_finite_fp32_003(test_args):
    from impl.dynamic import is_finite
    shape = (16, 8, 7, 15, 16)
    with tbe.common.context.op_context.OpContext("dynamic"):
        is_finite({"shape": shape, "ori_shape": shape, "dtype": "float32", "format": "ND", "ori_format": "ND",
                   "param_type": "input"},
                  {"shape": shape, "ori_shape": shape, "dtype": "bool", "format": "ND", "ori_format": "ND",
                   "param_type": "output"})
    cce_conf.cce_conf.te_set_version(test_args)


#ut_case.add_cust_test_func(test_func=is_finite_fp16_001)
#ut_case.add_cust_test_func(test_func=is_finite_fp16_002)
#ut_case.add_cust_test_func(test_func=is_finite_fp16_003)
#ut_case.add_cust_test_func(test_func=is_finite_fp32_001)
#ut_case.add_cust_test_func(test_func=is_finite_fp32_002)
#ut_case.add_cust_test_func(test_func=is_finite_fp32_003)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
