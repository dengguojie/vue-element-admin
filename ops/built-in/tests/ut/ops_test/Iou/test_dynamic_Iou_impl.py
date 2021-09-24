import tbe
from op_test_frame.ut import OpUT
ut_case = OpUT("Iou", "impl.dynamic.iou", "iou")

case1 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(4,4)]}, #bboxes
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,4096),(4,4)]}, #gtboxes
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(1,None)]}, 
                    "iou",
                    ],
         "case_name": "iou_dynamic_1",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(4,4)]}, #bboxes
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(4096,None),(4,4)]}, #gtboxes
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(1,None)]}, 
                    "iou",
                    ],
         "case_name": "iou_dynamic_1",
         "expect": "success",
         "support_expect": True}

case3 = {"params": [{"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(4,4)]}, #bboxes
                    {"shape": (-1, 4), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(4096,None),(4,4)]}, #gtboxes
                    {"shape": (-1, -1), "dtype": "float16", "format": "ND", "ori_shape": (-1, 4),"ori_format": "ND",
                     "range":[(1,None),(1,None)]}, 
                    "iou", 0.01,
                    ],
         "case_name": "iou_dynamic_3",
         "expect": "success",
         "support_expect": True}


ut_case.add_case(["Ascend910","Ascend310"], case1)
ut_case.add_case(["Ascend910","Ascend310"], case2)
ut_case.add_case(["Ascend910","Ascend310"], case3)


ut_case.add_case(["Ascend910","Ascend310"], case1)

if __name__ == '__main__':
    ut_case.run(["Ascend910","Ascend310"])
    exit(0)