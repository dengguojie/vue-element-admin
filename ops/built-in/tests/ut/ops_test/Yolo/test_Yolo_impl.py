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
                  boxes_val, coords_val, classes_val, case_name):
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
                       coords_val, classes_val],
            "case_name": case_name,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


case1 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_1")
case2 = gen_yolo_data(1, 1024, 1024, "float16", 5, 4, 80, "yolo_2")
case3 = gen_yolo_data(1, 13, 13, "float16", 5, 4, 80, "yolo_3")



ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case1)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case2)
ut_case.add_case(["Ascend310", "Ascend710", "Ascend910A"], case3)
ut_case.add_case(["Hi3796CV300CS"], case3)

if __name__ == '__main__':
    ut_case.run(["Ascend310", "Ascend710", "Ascend910A"])
    exit(0)
