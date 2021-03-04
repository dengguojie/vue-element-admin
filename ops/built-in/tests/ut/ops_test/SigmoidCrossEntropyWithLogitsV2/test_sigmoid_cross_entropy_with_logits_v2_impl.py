'''
test code
'''
import numpy as np
from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
from impl.sigmoid_cross_entropy_with_logits_v2 import op_select_format


ut_case = OpUT("SigmoidCrossEntropyWithLogitsV2", "impl.sigmoid_cross_entropy_with_logits_v2",
               "sigmoid_cross_entropy_with_logits_v2")

# pylint: disable=redefined-builtin,too-many-arguments,too-many-locals,unused-argument
def calc_expect_func(predict, target, weight, pos_weight, y, reduction):
    '''
    calc_expect_func
    '''
    predict_value = predict["value"]

    target_value = target["value"]

    weight_value = weight["value"]

    pos_weight_value = pos_weight["value"]

    max_val = np.maximum(-predict_value, 0)
    if pos_weight is not None:
        log_weight = (pos_weight_value - 1) * target_value + 1
        loss = (1 - target_value) * predict_value + (
            log_weight * (np.log(np.exp(-max_val) + np.exp(-predict_value - max_val)) + max_val))
    else:
        loss = (1 -
                target_value) * predict_value + max_val + np.log(np.exp(-max_val) + np.exp(-predict_value - max_val))

    if weight is not None:
        loss = loss * weight_value

    if reduction == "mean":
        res = np.mean(loss)
        return res.reshape(y['shape'])

    if reduction == "sum":
        res = np.sum(loss)
        return res.reshape(y['shape'])

    return loss


case1 = {
    "params": [
        {
            "shape": (128, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (128, 128),
            "dtype": "float32",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "none"
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_1",
    "expect": "success",
    "calc_expect_func": calc_expect_func,
    "precision_standard": precision_info.PrecisionStandard(0.0001, 0.0001)
}

case2 = {
    "params": [
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "none"
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_2",
    "expect": "success",
    "support_expect": True
}

case3 = {
    "params": [
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "mean"
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "sum"
    ],
    "case_name": "sigmoid_cross_entropy_with_logits_v2_4",
    "expect": "success",
    "support_expect": True
}


# pylint: disable=redefined-builtin,too-many-arguments,too-many-locals,unused-argument
def test_op_select_format(test_arg):
    '''
    def test_op_select_format
    '''
    op_select_format(
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "none")

    op_select_format(
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "mean")

    op_select_format(
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #predict
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #target
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #weight
        {
            "shape": (128, 128),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (128, 128),
            "ori_format": "ND",
            "param_type": "input"
        },  #pos_weight
        {
            "shape": (1,),
            "dtype": "float16",
            "format": "ND",
            "ori_shape": (1,),
            "ori_format": "ND",
            "param_type": "output"
        },  #loss
        "sum")


ut_case.add_precision_case(["Ascend910A"], case1)
ut_case.add_case(["Ascend910A"], case2)
ut_case.add_case(["Ascend910A"], case3)
ut_case.add_case(["Ascend910A"], case4)
ut_case.add_cust_test_func(test_func=test_op_select_format)
