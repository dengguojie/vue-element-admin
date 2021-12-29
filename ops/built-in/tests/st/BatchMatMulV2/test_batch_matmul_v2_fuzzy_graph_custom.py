#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.confusion_transpose_d import confusion_transpose_d_compute
from impl.dynamic.batch_matmul import batch_matmul_generalization
from impl.batch_matmul import batch_matmul_compute
from tbe.dsl import auto_schedule
from te import tvm
from te.lang.cce import cce_build_code
from te.platform.cce_conf import te_set_version
from te.tvm.target import cce

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

if __name__ == '__main__':
    test_batch_matmul_generalization_range_check()
    test_batch_matmul_generalization_not_valid()
    test_batchmatmul_confusion_transpose_910()
    test_batchmatmul_confusion_transpose_710()