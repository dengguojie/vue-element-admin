#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import tbe.common.context.op_context as op_context
from impl.batch_matmul import batch_matmul_compute
from impl.real_div import real_div_compute
from impl.add import add_compute
from te import tvm
from te.lang.cce import cce_build_code
from te.tvm.target import cce
from tbe.dsl import auto_schedule


def test_batchmatmul_realdiv_add():
    with op_context.OpContext("pre_static"):
        with cce():
            x1 = tvm.placeholder((8, 48, 24, 16, 16), name="x1", attrs={'format': "FRACTAL_NZ", "ori_shape": (8, 384, 768)}, dtype="float16")
            x2 = tvm.placeholder((48, 48, 16, 16), name="x2", attrs={'format': "FRACTAL_NZ", "ori_shape": (768,768)}, dtype="float16")
            output_y = {"shape": (8, 48, 24, 16, 16), "dtype": "float16", "ori_shape": (8, 384, 768), "format": "FRACTAL_NZ", "ori_format": "ND"}
            matmul_out = batch_matmul_compute(x1, x2, None, output_y)

            real_div_tensor = tvm.placeholder((1, ), name='tensor_div', dtype="float16", attrs={"format": "ND", "ori_shape": (1,)})
            add_tensor =  tvm.placeholder((8, 48, 24, 16, 16), name='tensor_add', dtype="float16", attrs={"format": "FRACTAL_NZ", "ori_shape": (16, 1, 96, 96)})
            real_div_out = real_div_compute(matmul_out, real_div_tensor, {})
            out = add_compute(real_div_out, add_tensor, {})

            tensor_list = [x1, x2, real_div_tensor, add_tensor, out]
            sch = auto_schedule(out)
            config = {
                "print_ir": False,
                "need_build": True,
                "name": "batch_matmul_realdiv_fused_mul_add",
                "tensor_list": tensor_list,
            }
            cce_build_code(sch, config)


if __name__ == "__main__":
    test_batchmatmul_realdiv_add()
