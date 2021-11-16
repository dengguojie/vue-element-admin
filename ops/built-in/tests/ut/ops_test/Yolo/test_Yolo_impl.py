#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from op_test_frame.ut import OpUT

ut_case = OpUT("Yolo", None, None)

def ceil_x(total_len, align_value):
    """
    ceil align
    Parameters
    ----------
    total_len: len before align
    align_value: align byte
    Returns
    -------
    len after ceil align
    """
    align_len = (total_len + align_value - 1) // align_value * align_value
    return align_len

def gen_yolo_data(batch_val, grid_h_val, grid_w_val, dtype_val,
                  boxes_val, coords_val, classes_val, case_name, version, expect,
                  softmax=False, background=False, softmaxtree=False):
    """
    feature_dic, boxes, coords, classes, yolo_version="V3",
    softmax=False, background=False, kernel_name="yolo"
    feature_dic{'shape':shape, 'format':"NCHW", 'dtype':dtype}
    """
    dtype_size = 2 if dtype_val == "float16" else 4
    shape_val = (batch_val, 1, grid_h_val, grid_w_val)
    input_dic = {'shape': shape_val, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":shape_val}
    coord_shape = (batch_val, boxes_val*coords_val,
                   ceil_x(grid_h_val*grid_w_val*dtype_size, 32)//dtype_size)
    obj_shape = (batch_val, ceil_x(boxes_val * grid_h_val * grid_w_val *
                                   dtype_size + 32, 32) // dtype_size)
    class_shape = (batch_val, classes_val,
                   ceil_x(boxes_val * grid_h_val * grid_w_val *
                          dtype_size + 32, 32) // dtype_size)
    coords_dic = {'shape': coord_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":coord_shape}
    obj_dic = {'shape': obj_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":obj_shape}
    class_dic = {'shape': class_shape, 'format': "NCHW", 'dtype': dtype_val, 'ori_format':"NCHW", "ori_shape":class_shape}

    return {"params": [input_dic, coords_dic, obj_dic, class_dic, boxes_val,
                       coords_val, classes_val, version, softmax, background, softmaxtree],
            "case_name": case_name,
            "expect": expect,
            "format_expect": [],
            "support_expect": True}


case1 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_1", "V2", "success", False, True)
case2 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_2", "V2", "success", True, False)
case3 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_3", "V2", "success", True, True)
case4 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_4", "V3", "success")

case5 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_5", "V2", "success", False, True)
case6 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_5", "V2", "success", False, False)
# case6 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_6", "V2", RuntimeError, True, False)
# case7 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_7", "V2", RuntimeError, True, True)
case8 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_8", "V3", "success")

case9 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_9", "V2", "success")
case10 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_10", "V3", "success")

case11 = gen_yolo_data(1, 13, 13, "float16", 0, 4, 80, "yolo_11", "V3", RuntimeError)
case12 = gen_yolo_data(1, 13, 13, "float16", 5, 3, 80, "yolo_12", "V3", RuntimeError)
case13 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 0, "yolo_13", "V3", RuntimeError)
case14 = gen_yolo_data(1024, 13, 13, "float16", 5, 4, 80, "yolo_14", "V3", RuntimeError)

ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case4)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case5)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case6)
ut_case.add_case(["Hi3796CV300CS"], case9)
ut_case.add_case(["Hi3796CV300CS"], case10)
# ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case7)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case8)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case9)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case10)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case11)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case12)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case13)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case14)


if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
    exit(0)
