#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT
import numpy as np
from op_test_frame.common import precision_info

ut_case = OpUT("CosineEmbeddingLoss", "impl.cosine_embedding_loss",
               "cosine_embedding_loss")


# pylint: disable=locally-disabled,too-many-arguments
def gen_cosine_embedding_loss_case(input_shape1, input_shape2, target_shape,
                                   output_shape, dtype_val1, dtype_val2,
                                   dtype_val3, margin, reduction,
                                   case_name_val, expect):
    return {"params": [{"shape": input_shape1, "dtype": dtype_val1},
                       {"shape": input_shape2, "dtype": dtype_val2},
                       {"shape": target_shape, "dtype": dtype_val3},
                       {"shape": output_shape, "dtype": "float32"},
                       margin, reduction],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}

def calc_expect_func(input_x1, input_x2, input_target, y,
                     margin=0, reduction='mean', epsilon=1e-5):
    x1 = input_x1["value"]
    x2 = input_x2["value"]
    target = input_target["value"]

    prod_num = np.sum(x1 * x2, axis=1)
    mag_square1 = np.sum(x1 ** 2, axis=1) + epsilon
    mag_square2 = np.sum(x2 ** 2, axis=1) + epsilon
    denom = np.sqrt(mag_square1 * mag_square2)
    cos = prod_num / denom

    zeros = np.zeros_like(target)
    pos = 1 - cos
    neg = np.maximum(cos - margin, 0)

    output_pos = np.where(target == 1, pos, zeros)
    output_neg = np.where(target == -1, neg, zeros)
    output = output_pos + output_neg

    if reduction == 'mean':
        output = np.mean(output, keepdims=True)

    if reduction == 'sum':
        output = np.sum(output, keepdims=True)

    return output

# pylint: disable=locally-disabled,too-many-arguments
def gen_cosine_embedding_loss_precision_case(input_shape1, input_shape2, target_shape,
                                   output_shape, dtype_val1, dtype_val2,
                                   dtype_val3, margin, reduction,
                                   case_name_val, expect):
    return {"params":
          [{"shape": input_shape1, "dtype": dtype_val1,
                  "param_type": "input", "value_range": [-10.0, 10.0]},
                 {"shape": input_shape2, "dtype": dtype_val2,
                  "param_type": "input", "value_range": [-10.0, 10.0]},
                 {"shape": target_shape, "dtype": dtype_val3,
                  "param_type": "input", "value_range": [-1, 1]},
                 {"shape": output_shape, "dtype": "float32",
                  "param_type": "output"},
                  margin, reduction],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True,
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.001, 0.001)}

ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'mean',
                                                "valid_fp32_mean",
                                                "success"))

ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float16", "float16",
                                                "float32", 0.3,
                                                'mean',
                                                "valid_fp16_x",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "int8", "int8", "float32", 0.3,
                                                'mean',
                                                "valid_int8_x",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "uint8", "uint8", "float32",
                                                0.3,
                                                'mean',
                                                "valid_uint8_x",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "int32", "int32", "float32",
                                                0.3,
                                                'mean',
                                                "valid_int32_x",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32", "int8",
                                                0.3,
                                                'mean',
                                                "valid_int8_target",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32", "int32",
                                                0.3,
                                                'mean',
                                                "valid_int32_target",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4),
                                                (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'sum',
                                                "valid_fp32_sum",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4),
                                                (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'none',
                                                "valid_fp32_none",
                                                "success"))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (2, 3, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32",
                                                "float32",
                                                0.3, 'mean',
                                                "invalid_shape_x",
                                                RuntimeError))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (2, 2, 4), (1, 3, 4),
                                                "float32", "float32",
                                                "float32",
                                                0.3, 'mean',
                                                "invalid_shape_target",
                                                RuntimeError))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "uint16", "float32", "float32",
                                                0.3, 'mean',
                                                "invalid_dtype_x1",
                                                RuntimeError))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "uint16", "float32",
                                                0.3, 'mean',
                                                "invalid_dtype_x2",
                                                RuntimeError))
ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32", "uint16",
                                                0.3, 'mean',
                                                "invalid_dtype_target",
                                                RuntimeError))

ut_case.add_case("all",
                 gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32",
                                                "float32",
                                                0.3, 'min',
                                                "invalid_reduction",
                                                RuntimeError))

ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_001",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float16", "float16",
                                                "float32", 0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_002",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "int8", "int8", "float32", 0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_003",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "uint8", "uint8", "float32",
                                                0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_004",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "int32", "int32", "float32",
                                                0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_005",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32", "int8",
                                                0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_006",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4), (1, 3, 4),
                                                "float32", "float32", "int32",
                                                0.3,
                                                'mean',
                                                "cosine_embedding_loss_precision_case_007",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4),
                                                (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'sum',
                                                "cosine_embedding_loss_precision_case_008",
                                                "success"))
ut_case.add_precision_case("all",
                 gen_cosine_embedding_loss_precision_case((1, 2, 3, 4), (1, 2, 3, 4),
                                                (1, 3, 4),
                                                (1, 3, 4),
                                                "float32", "float32",
                                                "float32", 0.3,
                                                'none',
                                                "cosine_embedding_loss_precision_case_009",
                                                "success"))

if __name__ == '__main__':
    ut_case.run("Ascend310")
    ut_case.run(["Ascend310"], simulator_mode="pv", simulator_lib_path="/usr/local/Ascend/toolkit/tools/simulator")
    exit(0)
