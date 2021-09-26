# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
from sch_test_frame.common import precision_info
import numpy as np
import tbe

"""
ut_case = OpUT("ConfusionSoftmaxGrad", "impl.dynamic.confusion_softmax_grad", "confusion_softmax_grad")

def gen_bn_training_update_grad_case(shape_grads, shape_x, shape_y, dtype_grads, dtype_x, dtype_y,
                                     case_name_val):
    return {"params": [{"shape":shape_grads, "ori_shape": shape_grads, "dtype": dtype_grads, "format":"ND", "ori_format":"ND", "range": [(1,None), (1,None), (1,None)]},
                       {"shape":shape_x, "ori_shape": shape_x, "dtype": dtype_x, "format":"ND", "ori_format":"ND", "range": [(1,None), (1,None), (1,None)]},
                       {"shape":shape_y, "ori_shape": shape_y, "dtype": dtype_y, "format":"ND", "ori_format":"ND", "range": [(1,None), (1,None), (1,None)]}],
            "case_name": case_name_val,
            "expect": "success",
            "support_expect": True}

case1 = gen_bn_training_update_grad_case((-1,-1,-1), (1,1,32), (-1,-1,32), "float32", "float32", "float32",
                                         "confusion_softmax_grad_1")
case2 = gen_bn_training_update_grad_case((-1,-1,-1), (1,1,48), (-1,-1,48), "float32", "float32", "float32",
                                         "confusion_softmax_grad_1")
case3 = gen_bn_training_update_grad_case((-1,-1,-1), (1,1,16), (-1,-1,16), "float16", "float16", "float16",
                                         "confusion_softmax_grad_1")

compile_case_list = [
    case1,
    case2,
    case3
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
if __name__ == '__main__':
    with tbe.common.context.op_context.OpContext("dynamic"):
        ut_case.run("Ascend910A")
"""
