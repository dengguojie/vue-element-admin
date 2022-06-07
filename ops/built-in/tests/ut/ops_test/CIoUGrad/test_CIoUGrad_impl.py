"""
Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this fil
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

CIoUGrad ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("CIoUGrad","impl.ciou_grad","ciou_grad")

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test1",
     "expect": "success"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test2",
     "expect": "success"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test3",
     "expect": "failed"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                False, False, "iou"],
     "case_name": "test4",
     "expect": "failed"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                True, True, "iou"],
     "case_name": "test5",
     "expect": "failed"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test6",
     "expect": "success"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (30720,),  "ori_shape": (30720,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 30720), "ori_shape": (4, 30720),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 30720), "ori_shape": (4, 30720),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (30720,),  "ori_shape": (30720,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 30720), "ori_shape": (4, 30720),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 30720), "ori_shape": (4, 30720),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test7",
     "expect": "success"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (7680,),  "ori_shape": (7680,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 7680), "ori_shape": (4, 7680),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 7680), "ori_shape": (4, 7680),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (7680,),  "ori_shape": (7680,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 7680), "ori_shape": (4, 7680),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 7680), "ori_shape": (4, 7680),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test8",
     "expect": "success"})

ut_case.add_case(["Ascend310P3", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (3840,),  "ori_shape": (3840,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 3840), "ori_shape": (4, 3840),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 3840), "ori_shape": (4, 3840),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (3840,),  "ori_shape": (3840,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 3840), "ori_shape": (4, 3840),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 3840), "ori_shape": (4, 3840),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test9",
     "expect": "success"})

if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)