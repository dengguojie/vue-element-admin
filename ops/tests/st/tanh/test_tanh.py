# -*- coding:utf-8 -*-
import unittest
#from impl import tanh
from impl.tanh_compute import tanh_compute
import os
from te import tvm
import time
import sys
import shutil
import sys
from te.platform.cce_build import build_config
import te.platform.cce_params as cce
#sys.path.append("./llt/ops/st_all/cce_all/testcase_python")
from run_testcase import run_testcase, get_path_val, print_func_name

test_cases = {
    "op_name": "tanh",
    "all": {
        "test_cce_tanh_3_3_5_6_7_float16": ((3, 3, 5, 6, 7), "float16", "cce_tanh_3_3_5_6_7_float16"),
        "test_cce_tanh_3_3_5_6_7_float32": ((3, 3, 5, 6, 7), "float32", "cce_tanh_3_3_5_6_7_float32"),
        "test_cce_tanh_1_1_float16": ((1, 1), "float16", "cce_tanh_1_1_float16"),
        "test_cce_tanh_1_1_float32": ((1, 1), "float32", "cce_tanh_1_1_float32"),
    },
    "mini": {},
    "lite": {},
    "cloud": {
        "test_cce_tanh_3_3_5_6_7_float16": ((3, 3, 5, 6, 7), "float16", "cce_tanh_3_3_5_6_7_float16"),
        "test_cce_tanh_3_3_5_6_7_float32": ((3, 3, 5, 6, 7), "float32", "cce_tanh_3_3_5_6_7_float32"),
        "test_cce_tanh_1_1_float16": ((1, 1), "float16", "cce_tanh_1_1_float16"),
        "test_cce_tanh_1_1_float32": ((1, 1), "float32", "cce_tanh_1_1_float32"),
        #"test_cce_tanh_1_1024_float16" : ((1, 1024), "float16", "cce_tanh_1_1024_float16"),
        #"test_cce_tanh_1_1024_float32" : ((1, 1024), "float32", "cce_tanh_1_1024_float32"),
        #"test_cce_tanh_4_1024_float16" : ((4, 1024), "float16", "cce_tanh_4_1024_float16"),
        #"test_cce_tanh_4_1024_float32" : ((4, 1024), "float32", "cce_tanh_4_1024_float32"),
        #"test_cce_tanh_16_1024_float16" : ((16, 1024), "float16", "cce_tanh_16_1024_float16"),
        #"test_cce_tanh_16_1024_float32" : ((16, 1024), "float32", "cce_tanh_16_1024_float32"),
        #"test_cce_tanh_32_1024_float16" : ((32, 1024), "float16", "cce_tanh_32_1024_float16"),
        #"test_cce_tanh_32_1024_float32" : ((32, 1024), "float32", "cce_tanh_32_1024_float32"),
        #"test_cce_tanh_64_1024_float16" : ((64, 1024), "float16", "cce_tanh_64_1024_float16"),
        #"test_cce_tanh_64_1024_float32" : ((64, 1024), "float32", "cce_tanh_64_1024_float32"),
        #"test_cce_tanh_512_4096_float16" : ((512, 4096), "float16", "cce_tanh_512_4096_float16"),
        #"test_cce_tanh_512_4096_float32" : ((512, 4096), "float32", "cce_tanh_512_4096_float32")
    },
    "tiny": {}
}

bin_path_val = get_path_val(test_cases)


def test_tanh(shape_val, dtype_val, kernel_name_val):
    '''
    tanh({"shape": shape_val, "dtype": dtype_val, "format": "NCHW", "ori_shape": shape_val, "ori_format": "NCHW"},
         {"shape": shape_val, "dtype": dtype_val, "format": "NCHW", "ori_shape": shape_val, "ori_format": "NCHW"},
         kernel_name=kernel_name_val)
    '''
    input_x = tvm.placeholder(
        shape_val, name="x", dtype=dtype_val)

    ub_ht_tmp2 = tvm.compute(shape_val, lambda *i: input_x(*i),
                             name="ub_ht_tmp2")
    tanh_ot_tensor, tanh_ot_operator, tanh_ot_scope = \
        tanh_compute(shape_val, ub_ht_tmp2, 'ht', "high_precision")
    output_gm = tvm.compute(
        shape_val, lambda *i: tanh_ot_tensor["ub_tanh_ht"](*i),
        name="output_gm")

    #schedule_list = [tanh_ot_tensor["ub_tanh_ht"].op]
    schedule_list = [output_gm.op]
    sch = tvm.create_schedule(schedule_list)

    sch[ub_ht_tmp2].emit_insn(ub_ht_tmp2.op.axis[0], "dma_copy")
    sch[ub_ht_tmp2].set_scope(cce.scope_ubuf)
    sch[output_gm].emit_insn(output_gm.op.axis[0], "dma_copy")
    sch[output_gm].set_scope(cce.scope_gm)
    for key in tanh_ot_tensor.keys():
        sch[tanh_ot_tensor[key]].set_scope(tanh_ot_scope[key])
        sch[tanh_ot_tensor[key]].emit_insn(tanh_ot_tensor[key].op.axis[0], tanh_ot_operator[key])
    build_list = [input_x, output_gm]
    print(tvm.lower(sch, build_list, simple_mode=True))
    with build_config:
        tvm.build(sch, build_list, "cce", name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"

    if(os.path.isfile(kernel_meta_path + lib_kernel_name)):
        shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/" + lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/"+ kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/"+ kernel_name_val +".json")


class TestTanh(unittest.TestCase):
    def tearDown(self):
        # 每个测试用例执行之后做操作
        pass

    def setUp(self):
        # 每个测试用例执行之前做操作
        pass

    @classmethod
    def tearDownClass(self):
        pass

    @classmethod
    def setUpClass(self):
        pass

    @print_func_name
    def test_tanh(self):
        run_testcase(test_cases, test_tanh)


def main():
    unittest.main()
    exit(0)


if __name__ == "__main__":
    main()
