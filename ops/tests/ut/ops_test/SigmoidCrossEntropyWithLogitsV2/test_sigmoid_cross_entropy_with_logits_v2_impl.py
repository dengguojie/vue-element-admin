from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np

ut_case = OpUT("SigmoidCrossEntropyWithLogitsV2", "impl.sigmoid_cross_entropy_with_logits_v2", "sigmoid_cross_entropy_with_logits_v2")

def calc_expect_func(predict, target, weight, pos_weight, reduction):
    predict_shape = predict["shape"]
    predict_value = predict["value"]
    predict_dtype = predict["dtype"]
    
    target_shape = target["shape"]
    target_value = target["value"]
    target_dtype = target["dtype"]
    
    weight_shape = weight["shape"]
    weight_value = weight["value"]
    weight_dtype = weight["dtype"]
    
    pos_weight_shape = pos_weight["shape"]
    pos_weight_value = pos_weight["value"]
    pos_weight_dtype = pos_weight["dtype"]
    
    max_val = -np.maximum(predict_value, 0)
    if pos_weight is not None:
        log_weight = (pos_weight_value - 1) * target_value + 1
        loss = (1 - target_value) * predict_value + (log_weight * (np.log(np.exp(-max_val) + np.exp(-predict_value-max_val)) + max_val))
    else:
        loss = (1 - target_value) * predict_value + max_val + np.log(np.exp(-max_val) + np.exp(-predict_value-max_val))        
    
    if weight is not None:
        loss = loss * weight_value
    
    if reduction == "mean":
        return np.mean(loss)
    
    if reduction == "sum":
        return np.sum(loss)
        
    return loss
    
case1 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"output"}, #loss
                    "none"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_1",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}

case2 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}, #loss
                    "mean"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_2",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}

case3 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}, #loss
                    "sum"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_3",
         "expect": "success",
         "calc_expect_func": calc_expect_func,
         "precision_standard": precision_info.PrecisionStandard(0.1, 0.1)}

case4 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}, #loss
                    "none"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_4",
         "expect": "success",
         "support_expect": True}

case5 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}, #loss
                    "mean"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_5",
         "expect": "success",
         "support_expect": True}

case6 = {"params": [{"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #predict
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #target
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #weight
                    {"shape": (128, 128), "dtype": "float16", "format": "ND", "ori_shape": (128, 128),"ori_format": "ND", "param_type":"input"}, #pos_weight
                    {"shape": (1,), "dtype": "float16", "format": "ND", "ori_shape": (1,),"ori_format": "ND", "param_type":"output"}, #loss
                    "sum"],
         "case_name": "sigmoid_cross_entropy_with_logits_v2_6",
         "expect": "success",
         "support_expect": True}

ut_case.add_precision_case(["Ascend910","Ascend310"], case1)
ut_case.add_precision_case(["Ascend910","Ascend310"], case2)
ut_case.add_precision_case(["Ascend910","Ascend310"], case3)
ut_case.add_case(["Ascend910","Ascend310"], case4)
ut_case.add_case(["Ascend910","Ascend310"], case5)
ut_case.add_case(["Ascend910","Ascend310"], case6)

if __name__ == '__main__':
    ut_case.run(["Ascend910"], simulator_mode="pv",
                simulator_lib_path="/disk1/ty_mindstudio/.mindstudio/huawei/adk/1.75.T15.0.B150/toolkit/tools/simulator")
