# # -*- coding:utf-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("masked_fill_range")

case1 = {
     "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                 "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                 "param_type": "input"},
                {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                 "param_type": "input"},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                 "param_type": "input"},
                {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32,), "shape": (32,),
                 "param_type": "output"}, 0],
     "case_name": "case_1",
     "expect": "success",
     "support_expect": True
 }

case2 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "output"}, 0],
    "case_name": "case_2",
    "expect": "success",
    "support_expect": True
}

case3= {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "output"}, 0],
    "case_name": "case_3",
    "expect": "success",
    "support_expect": True
}

case4 = {
    "params": [{"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "output"}, 0],
    "case_name": "case_4",
    "expect": "success",
    "support_expect": True
}

case5 = {
    "params": [{"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "int8", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "output"}, 0],
    "case_name": "case_5",
    "expect": "success",
    "support_expect": True
}

case6 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 5), "shape": (32, 5),
                "param_type": "output"}, 1],
    "case_name": "case_6",
    "expect": "success",
    "support_expect": True
}

case7 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "output"}, 0],
    "case_name": "case_7",
    "expect": "success",
    "support_expect": True
}

case8 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "output"}, 1],
    "case_name": "case_8",
    "expect": "success",
    "support_expect": True
}

case9 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 16), "shape": (1, 16),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, ), "shape": (1, ),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (16, 32, 5), "shape": (16, 32, 5),
                "param_type": "output"}, 2],
    "case_name": "case_9",
    "expect": "success",
    "support_expect": True
}

case10 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 32, 5), "shape": (4, 32, 5),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 32, 5), "shape": (4, 32, 5),
                "param_type": "output"}, 2],
    "case_name": "case_10",
    "expect": "success",
    "support_expect": True
}

case11 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 6, 1023),
                "shape": (4, 6, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 6, 1023),
                "shape": (4, 6, 1023), "param_type": "output"}, 2],
    "case_name": "case_11",
    "expect": "success",
    "support_expect": True
}

case12 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "output"}, 1],
    "case_name": "case_12",
    "expect": "success",
    "support_expect": True
}

case13 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float32", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "output"}, 1],
    "case_name": "case_13",
    "expect": RuntimeError,
    "support_expect": True
}

case14 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 4), "shape": (1, 4),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (4, 1023),
                "shape": (4, 1023), "param_type": "output"}, 2],
    "case_name": "case_14",
    "expect": RuntimeError,
    "support_expect": True
}

case15 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1638480000,),
                "shape": (1638480000,), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1638480000,),
                "shape": (1638480000,), "param_type": "output"}, 0],
    "case_name": "case_15",
    "expect": RuntimeError,
    "support_expect": True
}

case16 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 163848000),
                "shape": (32, 163848000), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 32), "shape": (1, 32),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (32, 163848000),
                "shape": (32, 163848000), "param_type": "output"}, 2],
    "case_name": "case_16",
    "expect": RuntimeError,
    "support_expect": True
}

case17 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1023),
                "shape": (1, 1, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1023),
                "shape": (1, 1, 1023), "param_type": "output"}, 2],
    "case_name": "case_17",
    "expect": "success",
    "support_expect": True
}

case18 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1023),
                "shape": (1, 1, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1), "shape": (1, 1),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1, 1, 1023),
                "shape": (1, 1, 1023), "param_type": "output"}, -1],
    "case_name": "case_18",
    "expect": "success",
    "support_expect": True
}

case19 = {
    "params": [{"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1, 1, 1023),
                "shape": (3, 1, 1, 1023), "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 3), "shape": (1, 3),
                "param_type": "input"},
               {"dtype": "int32", "format": "ND", "ori_format": "ND", "ori_shape": (1, 3), "shape": (1, 3),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (1,), "shape": (1,),
                "param_type": "input"},
               {"dtype": "float16", "format": "ND", "ori_format": "ND", "ori_shape": (3, 1, 1, 1023),
                "shape": (3, 1, 1, 1023), "param_type": "output"}, 3],
    "case_name": "case_19",
    "expect": RuntimeError,
    "support_expect": True
}


ut_case.add_case(['Ascend910A'], case1)
ut_case.add_case(['Ascend910A'], case2)
ut_case.add_case(['Ascend910A'], case3)
ut_case.add_case(['Ascend910A'], case4)
ut_case.add_case(['Ascend910A'], case5)
ut_case.add_case(['Ascend910A'], case6)
ut_case.add_case(['Ascend910A'], case7)
ut_case.add_case(['Ascend910A'], case8)
ut_case.add_case(['Ascend910A'], case9)
ut_case.add_case(['Ascend910A'], case10)
ut_case.add_case(['Ascend910A'], case11)
ut_case.add_case(['Ascend910A'], case12)
ut_case.add_case(['Ascend910A'], case13)
ut_case.add_case(['Ascend910A'], case14)
ut_case.add_case(['Ascend910A'], case15)
ut_case.add_case(['Ascend910A'], case16)
ut_case.add_case(['Ascend910A'], case17)
ut_case.add_case(['Ascend910A'], case18)
ut_case.add_case(['Ascend910A'], case19)

if __name__ == "__main__":
    ut_case.run("Ascend910A")
