#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.confusion_transpose_d import confusion_transpose_d_compute
from impl.dynamic.batch_matmul import batch_matmul_generalization
from impl.batch_matmul import batch_matmul_compute
from impl.batch_matmul_v2 import batch_matmul_compute as batch_matmul_v2_compute
from impl.ascend_dequant import ascend_dequant_compute
from impl.mul import mul_compute
from impl.sigmoid import sigmoid_compute
from impl.add import add_compute
from tbe.dsl import auto_schedule
from te import tvm
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce
from tbe.common.context import op_context

def test_batch_matmul_generalization_range_check():
    input_x1_dynamic = {'ori_shape': [-1, -1, -1, -1], 'shape': (-1, -1, -1, -1), 'range': ((4, 7), (1, 3), (1, 3), (1, 3)), 'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_range': ((4, 7), (1, 3), (1, 3), (1, 3))}
    input_x2_dynamic = {"ori_shape": [-1, -1], "shape":  [-1, -1], "range": ((1,3), (1,3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND", "ori_range": ((1,3), (1,3))}
    output_dynamic = {"ori_shape": (-1, -1, -1, -1), "shape": (-1, -1, -1, -1), "ori_range": ((4,7), (1,3), (1,3), (1, 3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": ((1, 48),)}
    batch_matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_z=output_dynamic,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul_generalization",
                                generalize_config={"mode": "keep_rank"})

def test_batch_matmul_generalization_not_valid():
    input_x1_dynamic = {'ori_shape': [-1, -1, -1, -1], 'shape': (-1, -1, -1, -1), 'range': ((4, 7), (1, 3), (1, 3), (1, 3)), 'dtype': 'float16', 'format': 'ND', 'ori_format': 'ND', 'ori_range': ((4, 7), (1, 3), (1, 3), (1, 3))}
    input_x2_dynamic = {"ori_shape": [-1, -1], "shape":  [-1, -1], "range": ((1,3), (1,3)), "dtype": 'float32', "format": "ND", "ori_format" : "ND", "ori_range": ((1,3), (1,3))}
    output_dynamic = {"ori_shape": (-1, -1, -1, -1), "shape": (-1, -1, -1, -1), "ori_range": ((4,7), (1,3), (1,3), (1, 3)), "dtype": 'float16', "format": "ND", "ori_format" : "ND"}
    bias_dynamic = {"ori_shape": (5, ), "dtype": 'float16', "shape": (5,), "format": "ND", "ori_format": "ND", "range": ((1, 48),)}
    batch_matmul_generalization(input_x1_dynamic, input_x2_dynamic, bias_dynamic, output_z=output_dynamic,
                                trans_a=False, trans_b=False, kernel_name="batchmatmul_generalization",
                                generalize_config={"mode": "keep_rank"})

def test_batchmatmul_confusion_transpose_910():
    te_set_version("Ascend910")
    with cce():
        x1 = tvm.placeholder((24*16, 32, 32, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512)}, dtype="float16")
        x2 = tvm.placeholder((24*16, 4, 32, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (24, 16, 512, 64)}, dtype="float16")
        output_y = {"shape": (24*16, 4, 32, 16, 16), "dtype": "float16", "ori_shape": (24, 16, 512, 64), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)
        y = {"shape": (64, 768, 16, 16), "ori_shape": (12288, 1024), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (12288, 1024), True)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_confusion_transpose_910",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_confusion_transpose_710():
    te_set_version("Ascend710")
    with cce():
        x1 = tvm.placeholder((8*12, 8, 8, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (8, 12, 128, 128)}, dtype="float16")
        x2 = tvm.placeholder((8*12, 4, 8, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (8, 12, 128, 64)}, dtype="float16")
        output_y = {"shape": (8*12, 4, 8, 16, 16), "dtype": "float16", "ori_shape": (8, 12, 128, 64), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(x1, x2, None, output_y)
        y = {"shape": (48, 64, 16, 16), "ori_shape": (1024, 768), "dtype": "float16", "format": "FRACTAL_NZ", "ori_format": "ND"}
        out = confusion_transpose_d_compute(matmul_out, y, [0, 2, 1, 3], (1024, 768), True)
        tensor_list = [x1, x2, out]
        sch = auto_schedule(out)
        config = {
            "print_ir": False,
            "need_build": True,
            "name": "batch_matmul_confusion_transpose_710",
            "tensor_list": tensor_list,
        }
        cce_build_code(sch, config)

def test_batchmatmul_dequant_mul_add_710():
    case = [
        {"shape": (8, 25, 2, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 784), "ori_format": "ND"},
        {"shape": (200, 6, 16, 32), "dtype": "int8", "format": "FRACTAL_Z", "ori_shape": (8, 784, 96), "ori_format": "HWCN"},
        None,
        False,
        False,
        {"shape": (8, 3, 2, 16, 32), "dtype": "int8", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        "batchmatmul_v2_dequant_mul_add_fusion_test",
        {"shape": (1, 1, 1, 1, 16), "dtype": "uint64", "format": "NC1HWC0", "ori_shape": (1,), "ori_format": "NCHW"},
        {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        "dequant",
        {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        "mul",
        {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        {"shape": (8, 6, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (8, 31, 96), "ori_format": "ND"},
        "add"]

    with op_context.OpContext():
        te_set_version("Ascend710")
        with cce():
            tensor_a = tvm.placeholder(case[0].get("shape"), name='tensor_a',
                                    attrs={'format': case[0].get("format"),
                                            "ori_shape": case[0].get("ori_shape")},
                                    dtype=case[0].get("dtype"))

            tensor_b = tvm.placeholder(case[1].get("shape"), name='tensor_b',
                                    attrs={'format': case[1].get("format"),
                                            "ori_shape": case[1].get("ori_shape"),
                                            "ori_format": case[1].get("ori_format")},
                                    dtype=case[1].get("dtype"))

            res = batch_matmul_v2_compute(tensor_a, tensor_b, bias=None, output_z=case[5], trans_a=case[3],
                                trans_b=case[4], offset_x=0, kernel_name=case[6])

            deq_tensor = tvm.placeholder(case[7].get("shape"), name='deq_tensor',
                                        attrs={'format': case[7].get("format"),
                                        "ori_shape": case[7].get("ori_shape")},
                                        dtype=case[7].get("dtype"))
            res = ascend_dequant_compute(res, deq_tensor, case[8], sqrt_mode=False, relu_flag=False, kernel_name=case[9])

            mul_tensor = tvm.placeholder(case[10].get("shape"), name='mul_tensor',
                                        attrs={'format': case[10].get("format"),
                                        "ori_shape": case[10].get("ori_shape")},
                                        dtype=case[10].get("dtype"),
            )
            res = mul_compute(res, mul_tensor, case[11], kernel_name=case[12])

            add_tensor = tvm.placeholder(case[13].get("shape"), name='add_tensor',
                                        attrs={'format': case[13].get("format"),
                                        "ori_shape": case[13].get("ori_shape")},
                                        dtype=case[13].get("dtype"),
            )
            out = add_compute(res, add_tensor, case[14], kernel_name=case[15])

            tensor_list = [tensor_a, tensor_b, deq_tensor, mul_tensor, add_tensor, out]
            sch = auto_schedule(out)
            config = {
                    "print_ir": False,
                    "need_build": True,
                    "name": "test_batchmatmul_dequant_mul_add_910",
                    "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)
        te_set_version("Ascend310")


def test_batch_matmul_elementwise_ub_fusion_710():
    case = [
        {"shape": (77, 32, 7, 16, 16), "ori_shape": (77, 100, 512),
         "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"},
        {"shape": (32, 128, 16, 16), "ori_shape": (2048, 512),
         "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"},
        None,
        None,
        {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
         "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
        False,
        True,
        0,
        "batch_matmul_elementwise_ub_fusion_test",
        {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
         "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
        {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
         "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
        "mul1",
        {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
         "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
        "sigmoid",
        {"shape": (77, 128, 7, 16, 16), "ori_shape": (77, 100, 2048),
         "format": "FRACTAL_NZ", "ori_format": "NHWC", "dtype": "float16"},
        "mul2"
    ]

    with op_context.OpContext():
        te_set_version("Ascend710")
        with cce():
            tensor_a = tvm.placeholder(case[0].get("shape"), name='tensor_a',
                                       attrs={'format': case[0].get("format"),
                                              "ori_shape": case[0].get("ori_shape"),
                                              "ori_format": case[0].get("ori_format")},
                                       dtype=case[0].get("dtype"))
            tensor_b = tvm.placeholder(case[1].get("shape"), name='tensor_b',
                                       attrs={'format': case[1].get("format"),
                                              "ori_shape": case[1].get("ori_shape"),
                                              "ori_format": case[1].get("ori_format")},
                                       dtype=case[1].get("dtype"))
            res_bmm = batch_matmul_v2_compute(tensor_a, tensor_b, bias=case[2], offset_w=case[3],
                                              output_z=case[4], trans_a=case[5], trans_b=case[6],
                                              offset_x=case[7], kernel_name=case[8])
            mul_tensor = tvm.placeholder(case[9].get("shape"), name='mul_tensor',
                                         attrs={'format': case[9].get("format"),
                                                "ori_shape": case[9].get("ori_shape")},
                                         dtype=case[9].get("dtype"))
            res = mul_compute(res_bmm, mul_tensor, case[10], kernel_name=case[11])
            res = sigmoid_compute(res, case[12], kernel_name=case[13])
            out = mul_compute(res_bmm, res, case[14], kernel_name=case[15])

            tensor_list = [tensor_a, tensor_b, mul_tensor, out]
            sch = auto_schedule(out)
            config = {
                "print_ir": False,
                "need_build": True,
                "name": "test_batch_matmul_elementwise_ub_fusion_func",
                "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)
        te_set_version("Ascend310")


if __name__ == '__main__':
    test_batch_matmul_generalization_range_check()
    test_batch_matmul_generalization_not_valid()
    test_batchmatmul_confusion_transpose_910()
    test_batchmatmul_confusion_transpose_710()
    test_batchmatmul_dequant_mul_add_710()
    test_batch_matmul_elementwise_ub_fusion_710()
