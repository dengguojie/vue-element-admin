# # -*- coding:utf-8 -*-
import warnings

from tbe.common.utils.varshape import variable_shape
from tbe.common.utils.varshape.variable_shape import get_variable
from tbe.common.utils.varshape.variable_shape import register_variable

from sch_test_frame.ut import OpUT

warnings.filterwarnings("ignore")
ut_case = OpUT("variable_shape", "varshape.test_variable_shape_impl")


def test_register_variable_with_func_register(_):
    
    @register_variable("mode_1")
    def mode_1_variable_shape():
        pass

    variable_shape_func = variable_shape._variables.get("mode_1")

    return variable_shape_func is not None


def test_register_variable_with_func_exec(_):

    @register_variable("mode_2")
    def mode_2_variable_shape(inputs):
        return inputs

    variable_shape_func = variable_shape._variables.get("mode_2")
    inputs = [{"shape": (-1, -1), "range": [(2, 10), (1, None)]}]
    ret = variable_shape_func(inputs)

    return inputs == ret


def test_get_variable(_):

    @register_variable("mode_3")
    def mode_3_variable_shape(inputs):
        return inputs

    variable_shape_func = get_variable("mode_3")

    return variable_shape_func is not None


test_funcs = [
    test_register_variable_with_func_register,
    test_register_variable_with_func_exec,
    test_get_variable,
]

for func in test_funcs:
    ut_case.add_cust_test_func(test_func=func)
