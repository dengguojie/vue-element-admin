#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info
import numpy as np
import random
ut_case = OpUT("FastrcnnPredictions", None, None)

def gen_fastrcnn_predictions_case(shape_x, shape_y, dtype_x, dtype_y, nms_threshold, score_threshold,
                                  k, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_y, "dtype": dtype_y, "ori_shape": shape_y, "ori_format": "ND", "format": "ND"},
                       {"shape": [k,4], "dtype": dtype_x, "ori_shape": [k,4], "ori_format": "ND", "format": "ND"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND"},
                       nms_threshold,
                       score_threshold,
                       k],
            "case_name": case_name_val,
            "expect": "success",
            "format_expect": [],
            "support_expect": True}

def gen_fastrcnn_predictions_case_err(shape_x, shape_y, dtype_x, dtype_y, nms_threshold, score_threshold,
                                  k, case_name_val):
    return {"params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": "ND", "format": "ND"},
                       {"shape": shape_y, "dtype": dtype_y, "ori_shape": shape_y, "ori_format": "ND", "format": "ND"},
                       {"shape": [k,4], "dtype": dtype_x, "ori_shape": [k,4], "ori_format": "ND", "format": "ND"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND"},
                       nms_threshold,
                       score_threshold,
                       k],
            "case_name": case_name_val,
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

case1 = gen_fastrcnn_predictions_case((32 * 5, 4), (32, 6), "float16", "float16", 0.5, 0.01, 32, "fastrcnn_predictions_1")
case2 = gen_fastrcnn_predictions_case((16 * 15, 4), (16, 16), "float16", "float16", 0.5, 0.01, 16, "fastrcnn_predictions_2")
case3 = gen_fastrcnn_predictions_case_err((16 * 32, 4), (16, 33), "float16", "float32", 0.5, 0.01, 16, "fastrcnn_predictions_3")
case4 = gen_fastrcnn_predictions_case_err((96, 4), (96, 2), "float32", "float16", 0.5, 0.01, 96, "fastrcnn_predictions_4")
case5 = gen_fastrcnn_predictions_case_err((96, 16, 4), (96, 2), "float16", "float16", 0.5, 0.01, 96, "fastrcnn_predictions_5")
case6 = gen_fastrcnn_predictions_case_err((96, 16, 4), (96, 32, 2), "float16", "float16", 0.5, 0.01, 96, "fastrcnn_predictions_6")
case7 = gen_fastrcnn_predictions_case_err((96, 4), (128, 2), "float16", "float16", 0.5, 0.01, 96, "fastrcnn_predictions_7")
case8 = gen_fastrcnn_predictions_case_err((96, 4), (96, 64), "float16", "float16", 0.5, 0.01, 96, "fastrcnn_predictions_8")

ut_case.add_case(["Ascend610", "Ascend710", "Ascend615"], case1)
ut_case.add_case(["Ascend610", "Ascend710", "Ascend615"], case2)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case3)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case4)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case5)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case6)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case7)
ut_case.add_case(["Ascend910A","Ascend310","Ascend310"], case8)


### precision cases
def nms_proposal_aadd(area_0, area_1, thresh):
    return (area_0+area_1)*thresh

def nms_proposal_iou(proposal_0, proposal_1):
    xx1 = max(proposal_0[0], proposal_1[0])
    yy1 = max(proposal_0[1], proposal_1[1])
    xx2 = min(proposal_0[2], proposal_1[2])
    yy2 = min(proposal_0[3], proposal_1[3])
    w = max(0, xx2-xx1)
    h = max(0, yy2-yy1)
    return w*h

def calc_expect_func(x, y, out1, out2, out3, nms_threshold, score_threshold, k):
    classes = y['shape'][1] - 1
    num = x['shape'][0] // classes
    downscale = 0.05
    rec_downscale = 1 / downscale

    data_x = x['value']
    data_y = y['value']
    src_type = x['dtype']

    nms_threshold_new = nms_threshold / (1.0 + nms_threshold)
    score_threshold_list = []
    for i in range(classes):
        score_threshold_list.append(score_threshold)

    data_x_trs = np.reshape(data_x, [num, classes, 4])
    x_trs = np.reshape(np.transpose(data_x_trs, [1, 0, 2]), [num * classes, 4])
    y_trs = np.reshape(np.transpose(data_y[:, 1:], [1, 0]), [num * classes, ])


    label = np.zeros((classes, num), dtype=src_type)
    for i in range(classes):
        label[i:, ] = i + 1
    label_trs = np.reshape(label, [num * classes, ])

    propsals = np.zeros((classes * num, 8), dtype=src_type)
    propsals[:, 0:4] = x_trs
    propsals[:, 4] = y_trs
    propsals[:, 5] = label_trs

    nms_after_propsal = np.zeros((num * classes, 8), dtype=src_type)
    finaily_propsal = np.zeros((k, 8), dtype=src_type)

    topk_num = []
    nms_num = []
    nms_after_num = 0
    after_topk_num = 0

    sorted_rois = np.zeros((k, 4), dtype=src_type)
    sorted_scores = np.zeros((k, 1), dtype=src_type)
    sorted_classes = np.zeros((k, 1), dtype=src_type)

    for i in range(classes):
        area = np.zeros((num,), dtype=src_type)
        topk_output_proposal = np.zeros((num, 8), dtype=src_type)
        topk_output_proposal1 = np.zeros((num, 8), dtype=src_type)
        supVec = np.zeros(num, dtype=np.uint16)
        supVec[0] = 0
        sort_proposal_box = []
        for j in range(num):
            sort_proposal_box.append(propsals[j + i * num, :])

        sort_proposal_box.sort(reverse=True, key=lambda x: x[4])
        count = 0
        for j in range(0, num):
            topk_output_proposal[j, :] = sort_proposal_box[j]
            if topk_output_proposal[j, 4] > score_threshold_list[i]:
                count = count + 1
        topk_num.append(count)
        after_topk_num = after_topk_num + topk_num[i]
        for j in range(topk_num[i]):
            topk_output_proposal1[j, :] = topk_output_proposal[j, :]
        topk_output_proposal1[0:topk_num[i], 0:4] = topk_output_proposal1[0:topk_num[i],
                                                    0:4] * downscale
        for j in range(topk_num[i]):
            area[j] = (topk_output_proposal1[j][2] - topk_output_proposal1[j][0]) \
                      * (topk_output_proposal1[j][3] - topk_output_proposal1[j][1])
        for j in range(0, topk_num[i] - 1):
            for m in range((j + 1), topk_num[i]):
                if (supVec[j] == 0):  # not suppressed by former
                    intersec = nms_proposal_iou(topk_output_proposal1[j],
                                                topk_output_proposal1[m])
                    join = nms_proposal_aadd(area[j], area[m], nms_threshold_new)
                    if (intersec > join):
                        supVec[m] = 1
        cout_count = 0
        for j in range(0, topk_num[i]):
            if supVec[j] == 0:
                nms_after_propsal[i * num + j, :] = topk_output_proposal1[j, :]
                cout_count = cout_count + 1
        nms_num.append(cout_count)
        nms_after_num = nms_num[i] + nms_after_num

    sort_proposal_box1 = []
    for i in range(num * classes):
        sort_proposal_box1.append(nms_after_propsal[i, :])
    sort_proposal_box1.sort(reverse=True, key=lambda x: x[4])
    for i in range(0, k):
        finaily_propsal[i, :] = sort_proposal_box1[i]
    finaily_propsal[:, 0:4] = finaily_propsal[:, 0:4] * rec_downscale
    for j in range(4):
        sorted_rois[:, j] = finaily_propsal[:, j]
    sorted_scores[:, 0] = finaily_propsal[:, 4]
    sorted_classes[:, 0] = finaily_propsal[:, 5]
    return sorted_rois, sorted_scores, sorted_classes

def gen_precision_case(shape_x, shape_y, dtype_x, dtype_y, nms_threshold, score_threshold,
                       k):
    classes = shape_y[1] - 1
    num = shape_x[0] // classes
    data_a = np.zeros((num * classes * 4,), dtype=np.float16)

    for i in range(num * classes):
        data_x1 = random.uniform(0, 1856)
        data_y1 = random.uniform(0, 2880)
        data_x2 = random.uniform(data_x1, 1856)
        data_y2 = random.uniform(data_y1, 2880)
        data_a[4 * i] = data_x1
        data_a[4 * i + 1] = data_y1
        data_a[4 * i + 2] = data_x2
        data_a[4 * i + 3] = data_y2
    data_x = np.reshape(data_a, shape_x).astype(np.float16)

    return {"params": [{"shape": shape_x, "dtype": dtype_x, "ori_shape": shape_x, "ori_format": "ND", "format": "ND", "param_type":"input","value":data_x},
                       {"shape": shape_y, "dtype": dtype_y, "ori_shape": shape_y, "ori_format": "ND", "format": "ND", "param_type":"input"},
                       {"shape": [k,4], "dtype": dtype_x, "ori_shape": [k,4], "ori_format": "ND", "format": "ND", "param_type":"output"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND", "param_type":"output"},
                       {"shape": [k,1], "dtype": dtype_x, "ori_shape": [k,1], "ori_format": "ND", "format": "ND", "param_type":"output"},
                       nms_threshold,
                       score_threshold,
                       k],
            "expect": "success",
            "calc_expect_func": calc_expect_func,
            "precision_standard": precision_info.PrecisionStandard(0.005, 0.005)}

ut_case.add_precision_case("Ascend910", gen_precision_case((16 * 1, 4), (16, 2), "float16", "float16", 0.5, 0.01, 16))
ut_case.add_precision_case("Ascend910", gen_precision_case((16 * 15, 4), (16, 16), "float16", "float16", 0.5, 0.01, 16))
ut_case.add_precision_case("Ascend910", gen_precision_case((32 * 5, 4), (32, 6), "float16", "float16", 0.5, 0.01, 32))
ut_case.add_precision_case("Ascend615", gen_precision_case((16 * 1, 4), (16, 2), "float16", "float16", 0.5, 0.01, 16))


