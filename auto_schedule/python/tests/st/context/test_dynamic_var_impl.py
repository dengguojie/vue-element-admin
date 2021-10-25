# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from tbe.dsl.base import var

warnings.filterwarnings("ignore")

ut_case = OpUT("context", "context.test_dynamic_op_context_impl", "dsl_vsub")


def test_var_addition(_):
    var1 = var.Var("name", "bound", "float32", var.Category.NORMAL, "addition")
    addition = var1.get_addition()
    if addition == "addition":
        return True
    return False


test_func_list = [
    test_var_addition
]
for item in test_func_list:
    ut_case.add_cust_test_func(test_func=item)

if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend310", "Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
