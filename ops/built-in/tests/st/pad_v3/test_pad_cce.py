# -*- coding:utf-8 -*-
import unittest
from from impl.pad_v3_d import pad_d
import os
from te import tvm
import time
import sys
import shutil

import sys
sys.path.append("./llt/ops/st_all/cce_all/testcase_python")
from run_testcase import run_testcase,get_path_val,print_func_name


class PadTestcase:
    def __init__(self,input_shape,paddings,mode,constant_values, input_type):
        self.input_shape     = input_shape
        self.paddings    = paddings
        self.input_type      = input_type
        self.mode            = mode.lower()
        self.constant_values = constant_values

    def get_test_case(self,dic):
        output_shape = [0 for _ in range(len(self.input_shape))]
        for i in range(len(self.input_shape)):
            output_shape[i]=self.paddings[i][0]+self.paddings[i][1]+self.input_shape[i]

            ex_str = str(self.input_shape)+'_'+self.input_type + '_' + str(self.paddings) + '_' + self.mode + '_' + str(self.constant_values)
            ex_str = ex_str.replace(',', '_')
            ex_str = ex_str.replace(' ', '')
            ex_str = ex_str.replace('[', '').replace(']', '')
            ex_str = ex_str.replace('(', '').replace(')', '')
        title = "test_cce_pad_"+ex_str
        stubFunc = "cce_pad_"+ex_str

        dic[title]=(self.input_shape, self.paddings, self.constant_values, self.input_type, stubFunc)


def get_all():

    PadingsTestcases = [
        PadTestcase([1], [[1, 1]], "constant", 0, "int32"),
        PadTestcase([17], [[1, 17]], "constant", 0, "float16"),

        PadTestcase([3, 17], [[3, 3], [0, 17]], "constant", 0, "int8"),
        PadTestcase([3, 1024 + 3], [[3, 0], [0, 0]], "constant", 0, "uint8"),


        PadTestcase([2, 2, 16], [[0, 0], [1, 3], [0, 0]], "constant", 0, "float16"),
        PadTestcase([2, 2, 9], [[7, 7], [7, 7], [7, 7]], "constant", 0, "float32"),
        PadTestcase([2, 2, 1027], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32"),
        PadTestcase([2, 2, 1027], [[0, 0], [0, 7], [0, 7]], "constant", 0, "float16"),
        PadTestcase([2, 2, 9], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32"),
        PadTestcase([2, 2, 63], [[0, 0], [0, 7], [0, 7]], "constant", 0, "float32"),
        PadTestcase([2, 2, 63], [[0, 0], [0, 16], [0, 0]], "constant", 0, "float32"),

        PadTestcase([3, 17], [[3, 3], [0, 3]], "constant", 0, "float32"),
    ]

    dic={}
    for case in PadingsTestcases:
        case.get_test_case(dic)
    return dic


testcases = {
    "op_name": "pad",
    "all": {},
    "mini": {},
    "lite": {},
    "cloud": get_all(),
    "tiny": {},
}

bin_path_val = get_path_val(testcases)

def test_cce_pad(shape_val, paddings, constant_values, dtype_val, kernel_name_val,
                 need_build_val = True, need_print_val = False):

    pad_d({"shape": shape_val, "dtype": dtype_val}, {"shape": shape_val, "dtype": dtype_val}, paddings,kernel_name=kernel_name_val)
    kernel_meta_path = "./kernel_meta/"
    lib_kernel_name = "lib" + kernel_name_val + ".so"
    if os.path.isfile(kernel_meta_path + lib_kernel_name):
        shutil.move(kernel_meta_path + lib_kernel_name, bin_path_val + "/"+ lib_kernel_name)
    else:
        shutil.move(kernel_meta_path + kernel_name_val + ".o", bin_path_val + "/"+ kernel_name_val + ".o")
        shutil.move(kernel_meta_path + kernel_name_val + ".json", bin_path_val + "/" + kernel_name_val + ".json")

class Test_pad_cce(unittest.TestCase):
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
    def test_cce_pad(self):
        run_testcase(testcases, test_cce_pad)

def main():
    unittest.main()
    exit(0)


if __name__ == "__main__":
    main()
