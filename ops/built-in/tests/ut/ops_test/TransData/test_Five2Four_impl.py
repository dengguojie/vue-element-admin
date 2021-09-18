#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("TransData", "impl.trans_data", "trans_data")


def gen_trans_data_case(src, dst, dtype, case_name_val, expect,
                        dst_format, src_format="NC1HWC0"):
    return {"params": [{"shape": src, "dtype": dtype, "ori_shape": src,
                        "ori_format": src_format, "format": src_format},
                       {"shape": dst, "dtype": dtype, "ori_shape": dst,
                        "ori_format": dst_format, "format": dst_format},
                       src_format, dst_format],
            "case_name": case_name_val,
            "expect": expect,
            "format_expect": [],
            "support_expect": True} 

ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "float16", "nchw_1", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "int8", "nchw_2", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
                                     "float32", "nchw_3", "success", "NCHW"))
# ut_case.add_case(["Ascend910"],
#                  gen_trans_data_case((3,1,16,16,16), (3, 2, 16,16),
#                                      "uint8", "nchw_4", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((16, 20, 13, 7, 16), (16, 311, 13, 7),
                                     "float32", "nchw_5", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 16, 48, 72, 16), (2, 256, 48, 72),
                                     "float32", "nchw_6", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 2, 41, 101, 16), (2, 31, 41, 101),
                                     "float32", "nchw_7", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2, 2, 9, 11, 16), (2, 29, 9, 11),
                                     "float16", "nchw_8", "success", "NCHW"))
ut_case.add_case(["Ascend910", "Ascend310"],
                 gen_trans_data_case((2, 4, 65, 65, 16), (2, 58, 65, 65),
                                     "float16", "nchw_9", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3200, 25, 1, 304, 16), (3200, 400, 1, 304),
                                     "float16", "nchw_10", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3200, 25, 1, 304, 16), (3200, 400, 1, 304),
                                     "bfloat16", "nchw_10", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3200, 25, 1, 304, 16), (3200, 1, 304, 400),
                                     "bfloat16", "nhwc_11", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 2, 4, 5, 16), (3, 4, 5, 19),
                                     "float16", "nhwc_1", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 3968, 4, 5, 16), (3, 4, 5, 63488),
                                     "float32", "nhwc_2", "success", "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3, 16, 31, 5001, 16), (3, 31, 5001, 16*16),
                                     "float32", "nhwc_2", "success", "NHWC"))
ut_case.add_case(["Ascend310"],
                 gen_trans_data_case((10000, 1, 127, 127, 16), (10000, 127, 127, 1),
                                     "float32", "nhwc_127_127", "success", "NHWC"))
#invalid
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,30,4,5),
                                     "float32", "err_1", RuntimeError, "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,3,5,48),
                                     "float32", "err_2", RuntimeError, "NHWC"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((2,3,4,5,16), (2,48,3,5),
                                     "float32", "err_3", RuntimeError, "NCHW"))


#five 2 four int8
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((1,1,1,16,16), (1, 1, 1,16),
                                     "int8", "int8_1", "success", "NCHW"))
ut_case.add_case(["Ascend910"],
                 gen_trans_data_case((3,1,2,16,16), (3, 2, 16,16),
                                     "int8", "int8_2", "success", "NHWC"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3,1,2,16,32), (3, 2, 2, 16),
                                     "int8", "int8_3", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((3,3,2001,16,32), (3, 95, 2001, 16),
                                     "int8", "int8_4", "success", "NCHW"))
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((31,3,201,16,32), (31, 92, 201, 16),
                                     "int8", "int8_5", "success", "NCHW"))

#five 2 four float 
ut_case.add_case(["Ascend910A", "Ascend310"],
                 gen_trans_data_case((2560, 32, 4, 26, 16), (2560, 512, 4, 26),
                                     "float16", "float16_1", "success", "NCHW"))


case1 = {'params': [{'shape': (16, 1, 16, 16, 16),'dtype': 'float32','ori_shape': (16, 1, 16, 16, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 16, 16, 26),'dtype': 'float32','ori_shape': (16, 16, 16, 26),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NCHW'],
        'case_name': 'Five2Four_case1',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case2 = {'params': [{'shape': (1, 1, 1, 1, 16),'dtype': 'float32','ori_shape': (1, 1, 1, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (1, 1, 1, 16),'dtype': 'float32','ori_shape': (1, 1, 1, 16),'ori_format': 'NHWC','format': 'NHWC'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case2',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case3 = {'params': [{'shape': (1, 1, 16, 1, 16),'dtype': 'float32','ori_shape': (1, 1, 16, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (1, 16, 1, 16),'dtype': 'float32','ori_shape': (1, 16, 1, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NCHW'],
        'case_name': 'Five2Four_case3',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case4 = {'params': [{'shape': (2, 1, 16, 1, 16),'dtype': 'float16','ori_shape': (2, 1, 16, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (2, 16, 1, 9),'dtype': 'float16','ori_shape': (2, 16, 1, 9),'ori_format': 'NHWC','format': 'NHWC'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case4',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case5 = {'params': [{'shape': (2, 1, 1, 1, 16),'dtype': 'float16','ori_shape': (2, 1, 1, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (2, 1, 1, 16),'dtype': 'float16','ori_shape': (2, 1, 1, 16),'ori_format': 'NHWC','format': 'NHWC'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case5',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case6 = {'params': [{'shape': (2, 1, 2, 1, 16),'dtype': 'float32','ori_shape': (2, 1, 2, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (2, 2, 1, 16),'dtype': 'float32','ori_shape': (2, 2, 1, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case6',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case7 = {'params': [{'shape': (16, 4, 16, 64, 16),'dtype': 'float32','ori_shape': (16, 4, 16, 64, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 16, 64, 64),'dtype': 'float32','ori_shape': (16, 16, 64, 64),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case7',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case8 = {'params': [{'shape': (16, 4, 1, 64, 16),'dtype': 'float32','ori_shape': (16, 4, 1, 64, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 1, 64, 64),'dtype': 'float32','ori_shape': (16, 1, 64, 64),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case8',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}     #line 9716    return true todo 
case9 = {'params': [{'shape': (16, 4, 32, 64, 16),'dtype': 'float32','ori_shape': (16, 4, 32, 64, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 32, 64, 64),'dtype': 'float32','ori_shape': (16, 32, 64, 64),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case9',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}       
case10 = {'params': [{'shape': (16, 4, 8, 64, 16),'dtype': 'float32','ori_shape': (16, 4, 8, 64, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 8, 64, 64),'dtype': 'float32','ori_shape': (16, 8, 64, 64),'ori_format': 'NHWC','format': 'NHWC'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case10',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True} 
case11 = {'params': [{'shape': (2, 1, 8, 1, 16),'dtype': 'float16','ori_shape': (2, 1, 8, 1, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (2, 8, 1, 16),'dtype': 'float16','ori_shape': (2, 8, 1, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case11',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}   #src_shape[4] <= 128
case12 = {'params': [{'shape': (16, 1, 16, 16, 16),'dtype': 'float32','ori_shape': (16, 1, 16, 16, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 16, 2, 16),'dtype': 'float32','ori_shape': (16, 16, 2, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NCHW'],
        'case_name': 'Five2Four_case12',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}     #src_shape[4] <= 128
case13 = {'params': [{'shape': (16, 1, 2, 5, 16),'dtype': 'float32','ori_shape': (16, 1, 2, 5, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 2, 2, 5),'dtype': 'float32','ori_shape': (16, 2, 2, 5),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NCHW'],
        'case_name': 'Five2Four_case13',
        'expect': 'success',
        'format_expect': [],
        'support_expect': True}
case14 = {'params': [{'shape': (16, 1, 16, 16, 16),'dtype': 'float32','ori_shape': (16, 1, 16, 16, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 2, 1, 2),'dtype': 'float32','ori_shape': (16, 2, 1, 2),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NCHW'],
        'case_name': 'Five2Four_case14',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case15 = {'params': [{'shape': (16, 1, 16, 16, 16),'dtype': 'float32','ori_shape': (16, 1, 16, 16, 16),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (300, 2, 2, 3),'dtype': 'float32','ori_shape': (300, 2, 2, 3),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case15',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case16 = {'params': [{'shape': (8, 1, 16, 16, 50),'dtype': 'float32','ori_shape': (8, 1, 16, 16, 50),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (10, 2, 5, 10),'dtype': 'float32','ori_shape': (10, 2, 5, 10),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case16',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case17 = {'params': [{'shape': (8, 1, 16, 16, 50),'dtype': 'float32','ori_shape': (8, 1, 16, 16, 50),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 5, 10),'dtype': 'float32','ori_shape': (50, 2, 5, 10),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case17',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case18 = {'params': [{'shape': (20, 4, 500, 500, 500),'dtype': 'float32','ori_shape': (100, 100, 100, 100, 100),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (16, 2, 1, 2),'dtype': 'float32','ori_shape': (16, 2, 1, 2),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case18',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case19 = {'params': [{'shape': (500, 500, 100, 20, 20),'dtype': 'float32','ori_shape': (500, 500, 100, 20, 20),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 5, 10),'dtype': 'float32','ori_shape': (50, 2, 5, 10),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case19',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case20 = {'params': [{'shape': (10, 10, 10, 20, 60),'dtype': 'float32','ori_shape': (10, 10, 10, 20, 60),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 5, 10),'dtype': 'float32','ori_shape': (50, 2, 5, 10),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case20',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case21 = {'params': [{'shape': (10, 10, 10, 40, 60),'dtype': 'float32','ori_shape': (10, 10, 10, 40, 60),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 5, 10),'dtype': 'float32','ori_shape': (50, 2, 5, 10),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case21',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case22 = {'params': [{'shape': (10, 1, 10, 40, 2),'dtype': 'float16','ori_shape': (10, 1, 10, 40, 2),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 20, 16),'dtype': 'float16','ori_shape': (50, 2, 20, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case22',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}
case23 = {'params': [{'shape': (10, 1, 10, 40, 2),'dtype': 'float16','ori_shape': (10, 1, 10, 40, 2),'ori_format': 'NC1HWC0','format': 'NC1HWC0'},
        {'shape': (50, 2, 20, 16),'dtype': 'float16','ori_shape': (50, 2, 20, 16),'ori_format': 'NCHW','format': 'NCHW'},
        'NC1HWC0',
        'NHWC'],
        'case_name': 'Five2Four_case23',
        'expect': RuntimeError,
        'format_expect': [],
        'support_expect': True}

ut_case.add_case(["Ascend910A", "Ascend310"],case1)
ut_case.add_case(["Ascend910A", "Ascend310"],case2)
ut_case.add_case(["Ascend910A", "Ascend310"],case3)
ut_case.add_case(["Ascend910A", "Ascend310"],case4)
ut_case.add_case(["Ascend910A", "Ascend310"],case5)
ut_case.add_case(["Ascend910A", "Ascend310"],case6)
ut_case.add_case(["Ascend910A", "Ascend310"],case7)
ut_case.add_case(["Ascend910A", "Ascend310"],case8)
ut_case.add_case(["Ascend910A", "Ascend310"],case9)
ut_case.add_case(["Ascend910A", "Ascend310"],case10)
ut_case.add_case(["Ascend910A", "Ascend310"],case11)
ut_case.add_case(["Ascend910A", "Ascend310"],case12)
ut_case.add_case(["Ascend910A", "Ascend310"],case13)
ut_case.add_case(["Ascend910A", "Ascend310"],case14)
ut_case.add_case(["Ascend910A", "Ascend310"],case15)
ut_case.add_case(["Ascend910A", "Ascend310"],case16)
ut_case.add_case(["Ascend910A", "Ascend310"],case17)
ut_case.add_case(["Ascend910A", "Ascend310"],case18)
ut_case.add_case(["Ascend910A", "Ascend310"],case19)
ut_case.add_case(["Ascend910A", "Ascend310"],case20)
ut_case.add_case(["Ascend910A", "Ascend310"],case21)
if __name__ == '__main__':
    ut_case.run(["Ascend910A", "Ascend310"])
    exit(0)
