"""
Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

This program is free software; you can redistribute it and/or modify
it under the terms of the Apache License Version 2.0.You may not use this fil
except in compliance with the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
Apache License for more details at
http://www.apache.org/licenses/LICENSE-2.0

GIoUGrad ut case
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("GIoUGrad","impl.giou_grad","giou_grad")

ut_case.add_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 1), "ori_shape": (4, 1),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test1",
     "expect": "success"})

ut_case.add_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (15360,),  "ori_shape": (15360,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (4, 15360), "ori_shape": (4, 15360),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test2",
     "expect": "success"})

ut_case.add_case(["Ascend710", "Ascend910A"], {
    "params": [{"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1,),  "ori_shape": (1,),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "input"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "output"},
                {"dtype": "float32", "format": "ND", "ori_format": "ND", "shape": (1, 4), "ori_shape": (1, 4),
                "param_type": "output"},
                True, False, "iou"],
     "case_name": "test3",
     "expect": "failed"})
     
if __name__ == '__main__':
    ut_case.run("Ascend910A")
    exit(0)