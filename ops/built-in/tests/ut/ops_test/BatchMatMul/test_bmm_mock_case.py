#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from te import tvm
from te.tvm.target import cce
from tbe.dsl import auto_schedule
from impl.batch_matmul import batch_matmul_compute
from impl.trans_data import trans_data_compute
from impl.fix_pipe import fixpipe_compute


def test_matmul_ND2ND_fp16():
    with cce():
        x1 = tvm.placeholder((4, 16, 32), name="x1", dtype="float16", attrs={"ori_shape": (4, 16, 32), "format": "ND", "ori_format": "ND"})
        x2 = tvm.placeholder((4, 32, 16), name="x2", dtype="float16", attrs={"ori_shape": (4, 32, 16), "format": "ND", "ori_format": "ND"})
        x1_trans = trans_data_compute(x1, None, "ND", "FRACTAL_NZ")
        x2_trans = trans_data_compute(x2, None, "ND", "FRACTAL_NZ")
        y = {"shape": (4, 1, 1, 16, 16), "ori_shape": (4, 16, 16), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        dx_res = batch_matmul_compute(x1_trans, x2_trans, None, y, False, False)
        trans_out = {"shape": (4, 16, 16), "ori_shape": (4, 16, 16), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        out = trans_data_compute(dx_res, trans_out, "FRACTAL_NZ", "ND")
        sch = auto_schedule(out)

def test_matmul_ND2ND_int8():
    with cce():
        tensor_a_ori = tvm.placeholder((6, 64, 96), name="tensor_a_ori", dtype="int8")
        tensor_b_ori = tvm.placeholder((6, 64, 128), name="tensor_b_ori", dtype="int8")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (6, 8, 6, 16, 16), "dtype": "int32", "ori_shape": (6, 96, 128), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(tensor_a, tensor_b, None, output_y, True, False)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_ND2ND_fp32():
    with cce():
        tensor_a_ori = tvm.placeholder((3, 64, 96), name="tensor_a_ori", dtype="float32")
        tensor_b_ori = tvm.placeholder((3, 64,128), name="tensor_b_ori", dtype="float32")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (3, 16, 6, 16, 8), "dtype": "float32", "ori_shape": (3, 96, 128), "format": "ND", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(tensor_a, tensor_b, None, output_y, True, False)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_ND2ND_fp32_1():
    with cce():
        tensor_a_ori = tvm.placeholder((10, 64, 32), name="tensor_a_ori", dtype="float32")
        tensor_b_ori = tvm.placeholder((96, 32), name="tensor_b_ori", dtype="float32")
        tensor_a = trans_data_compute(tensor_a_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        tensor_b = trans_data_compute(tensor_b_ori, None, src_format="ND", dst_format="FRACTAL_NZ")
        output_y = {"shape": (10, 12, 4, 16, 8), "dtype": "float32", "ori_shape": (10, 64, 96), 
                    "format": "ND", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(tensor_a, tensor_b, None, output_y, False, True)
        out = trans_data_compute(matmul_out, None, src_format="FRACTAL_NZ", dst_format="ND")
        tensor_list = [tensor_a_ori, tensor_b_ori, out]
        sch = auto_schedule(out)

def test_matmul_NZ2ND_fp16():
    with cce():
        x1 = tvm.placeholder((2, 1, 2, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (2, 32, 16), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((2, 1, 2, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (2, 32, 16), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (2, 2, 2, 16, 16), "ori_shape": (2, 32, 32), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        dx_res = batch_matmul_compute(x1, x2, None, y, False, True)
        trans_out = {"shape": (2, 32, 32), "ori_shape": (2, 32, 32), "format": "ND", "ori_format": "ND", "dtype": "float16"}
        out = trans_data_compute(dx_res, trans_out, "FRACTAL_NZ", "ND")
        sch = auto_schedule(out)

def test_matmul_ND2NZ_fp16():
    with cce():
        x1 = tvm.placeholder((10, 64, 1024), name="x1", dtype="float16", attrs={"ori_shape": (10, 64, 1024), "format": "ND", "ori_format": "ND"})
        x2 = tvm.placeholder((10, 1024, 32), name="x2", dtype="float16", attrs={"ori_shape": (10, 1024, 32), "format": "ND", "ori_format": "ND"})
        x1_trans = trans_data_compute(x1, None, "ND", "FRACTAL_NZ")
        x2_trans = trans_data_compute(x2, None, "ND", "FRACTAL_NZ")
        y = {"shape": (10, 2, 4, 16, 16), "ori_shape": (10, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        bias = tvm.placeholder((32,), name="bias", dtype="float32", attrs={"ori_shape": (32,), "format": "ND", "ori_format": "ND"})
        out = batch_matmul_compute(x1_trans, x2_trans, bias, y, False, False)
        sch = auto_schedule(out)


def test_matmul_NZ2NZ_fp16():
    with cce():
        x1 = tvm.placeholder((4, 8, 4, 16, 16), name="x1", dtype="float16", attrs={"ori_shape": (4, 64, 128), "format": "FRACTAL_NZ", "ori_format": "ND"})
        x2 = tvm.placeholder((4, 4, 8, 16, 16), name="x2", dtype="float16", attrs={"ori_shape": (4, 128, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
        y = {"shape": (4, 8, 8, 16, 16), "ori_shape": (4, 128, 128), "format": "FRACTAL_NZ", "ori_format": "ND", "dtype": "float16"}
        out = batch_matmul_compute(x1, x2, None, y, True, True)
        sch = auto_schedule(out)

def test_matmul_NZ2NZ_int8():
    with cce():
        tensor_a_ori = tvm.placeholder((16, 3, 4, 16, 32), name="tensor_a", dtype="int8", attrs={"format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (16, 64, 96)})
        tensor_b_ori = tvm.placeholder((16, 4, 4, 16, 32), name="tensor_b", dtype="int8", attrs={"format": "FRACTAL_NZ", "ori_format": "ND", "ori_shape": (16, 64, 128)})
        bias = tvm.placeholder((1024,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND"})
        output_y = {"shape": (8, 6, 16, 16), "dtype": "int32", "ori_shape": (96, 128), "format": "FRACTAL_NZ", "ori_format": "ND"}
        matmul_out = batch_matmul_compute(tensor_a_ori, tensor_b_ori, bias, output_y, True, False)
        sch = auto_schedule(matmul_out)


def test_matmul_fixpipe_0():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((3, 4, 2, 16, 16), name="tensor_a", dtype="float16", attrs={"ori_shape": (3, 32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((3, 2, 4, 16, 16), name="tensor_b", dtype="float16", attrs={"ori_shape": (3, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="float32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (3, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (3, 32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = batch_matmul_compute(x1, x2, bias, output_y, False, False)
            y = {"shape": (3, 32, 32), "dtype": "float16", "ori_shape": (3, 32, 32), "format": "ND", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, res]
            sch = auto_schedule(res)

def test_matmul_fixpipe_1():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((3, 4, 2, 16, 16), name="tensor_a", dtype="float16", attrs={"ori_shape": (3, 32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((3, 2, 4, 16, 16), name="tensor_b", dtype="float16", attrs={"ori_shape": (3, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="float32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (3, 4, 2, 16, 8), "dtype": "float32", "ori_shape": (3, 32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = batch_matmul_compute(x1, x2, bias, output_y, False, False)
            y = {"shape": (3, 4, 2, 16, 8), "dtype": "float32", "ori_shape": (3, 32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, res]
            sch = auto_schedule(res)

def test_matmul_fixpipe_2():
    import tbe
    with tbe.common.context.op_context.OpContext("pre-static"):
        with cce():
            x1 = tvm.placeholder((5, 2, 2, 16, 32), name="tensor_a", dtype="int8", attrs={"ori_shape": (5, 32, 64), "format": "FRACTAL_NZ", "ori_format": "ND"})
            x2 = tvm.placeholder((5, 1, 4, 16, 32), name="tensor_b", dtype="int8", attrs={"ori_shape": (5, 64, 32), "format": "FRACTAL_NZ", "ori_format": "ND"})
            bias = tvm.placeholder((32,), name="tensor_bias", dtype="int32", attrs={"format": "ND", "ori_format": "ND", "ori_shape": (32,)})
            output_y = {"shape": (5, 2, 2, 16, 16), "dtype": "int32", "ori_shape": (32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = batch_matmul_compute(x1, x2, bias, output_y, False, False)
            deq = tvm.placeholder((5, 1, 2, 1, 1, 16), name='deq', dtype="uint64", attrs={"ori_shape": (32, ), "format": "NC1HWC0", "ori_format": "ND"})
            y = {"shape": (5, 2, 2, 16, 16), "dtype": "float16", "ori_shape": (5, 32, 32), "format": "FRACTAL_NZ", "ori_format": "ND"}
            res = fixpipe_compute(matmul_out, None, None, None, None, None, None, None, None, None, y, [], [], "")
            tensor_list = [x1, x2, bias, deq, res]
            sch = auto_schedule(res)
