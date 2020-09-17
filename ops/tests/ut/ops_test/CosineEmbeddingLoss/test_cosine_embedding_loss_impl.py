# TODO fix me, run failed
# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
#
# ut_case = OpUT("CosineEmbeddingLoss", "impl.cosine_embedding_loss",
#                "cosine_embedding_loss")
#
#
# # pylint: disable=locally-disabled,too-many-arguments
# def gen_cosine_embedding_loss_case(input_shape1, input_shape2, target_shape,
#                                    output_shape, dtype_val1, dtype_val2,
#                                    dtype_val3, margin, reduction,
#                                    case_name_val, expect):
#     return {"params": [{"shape": input_shape1, "dtype": dtype_val1},
#                        {"shape": input_shape2, "dtype": dtype_val2},
#                        {"shape": target_shape, "dtype": dtype_val3},
#                        {"shape": output_shape, "dtype": "float32"},
#                        margin, reduction],
#             "case_name": case_name_val,
#             "expect": expect,
#             "format_expect": [],
#             "support_expect": True}
#
#
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32", 0.3,
#                                                 'mean',
#                                                 "valid_fp32_mean",
#                                                 "success"))
#
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float16", "float16",
#                                                 "float32", 0.3,
#                                                 'mean',
#                                                 "valid_fp16_x",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "int8", "int8", "float32", 0.3,
#                                                 'mean',
#                                                 "valid_int8_x",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "uint8", "uint8", "float32",
#                                                 0.3,
#                                                 'mean',
#                                                 "valid_uint8_x",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "int32", "int32", "float32",
#                                                 0.3,
#                                                 'mean',
#                                                 "valid_int32_x",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32", "int8",
#                                                 0.3,
#                                                 'mean',
#                                                 "valid_int8_target",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32", "int32",
#                                                 0.3,
#                                                 'mean',
#                                                 "valid_int32_target",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4),
#                                                 (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32", 0.3,
#                                                 'sum',
#                                                 "valid_fp32_sum",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4),
#                                                 (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32", 0.3,
#                                                 'none',
#                                                 "valid_fp32_none",
#                                                 "success"))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (2, 3, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32",
#                                                 0.3, 'mean',
#                                                 "invalid_shape_x",
#                                                 RuntimeError))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (2, 2, 4), (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32",
#                                                 0.3, 'mean',
#                                                 "invalid_shape_target",
#                                                 RuntimeError))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "uint16", "float32", "float32",
#                                                 0.3, 'mean',
#                                                 "invalid_dtype_x1",
#                                                 RuntimeError))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "uint16", "float32",
#                                                 0.3, 'mean',
#                                                 "invalid_dtype_x2",
#                                                 RuntimeError))
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32", "uint16",
#                                                 0.3, 'mean',
#                                                 "invalid_dtype_target",
#                                                 RuntimeError))
#
# ut_case.add_case("all",
#                  gen_cosine_embedding_loss_case((1, 2, 3, 4), (1, 2, 3, 4),
#                                                 (1, 3, 4), (1, 3, 4),
#                                                 "float32", "float32",
#                                                 "float32",
#                                                 0.3, 'min',
#                                                 "invalid_reduction",
#                                                 RuntimeError))
#
# if __name__ == '__main__':
#     ut_case.run()
#     exit(0)
