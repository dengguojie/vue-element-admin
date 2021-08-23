#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
ut_case = OpUT("SsdDetectionOutput", None, None)

def run_ssd_detection_out_case(
        num_batch, num_classes, num_loc, code_type,
        dtype="float16",
        share_location=True,
        background_label_id=0,
        iou_threshold=0.45,
        top_k=400,
        eta=1.0,
        variance_encoded_in_target=False,
        keep_top_k=-1,
        confidence_threshold=0.0,):
    mbox_loc_dict = {"shape":(num_batch, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * 4)}
    mbox_conf_dict = {"shape":(num_batch, num_loc * num_classes), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * num_classes)}
    mbox_priorbox_dict = {"shape":(num_batch, 2, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, 2, num_loc * 4)}
    out_box_num_dict = {"shape":(8,), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(8,)}

    if keep_top_k == -1:
        out_box_dict = {"shape":(num_batch, top_k, 8), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, top_k, 8)}
    else:
        out_box_dict = {"shape":(num_batch, keep_top_k, 8), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, keep_top_k, 8)}

    return {"params": [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict,
                       out_box_num_dict, out_box_dict, num_classes],
            "expect": "success",
            "format_expect": [],
            "support_expect": True}


def run_ssd_detection_runtimeerror_case1(
        num_batch, num_classes, num_loc, code_type,
        dtype="float16",
        share_location=True,
        background_label_id=0,
        iou_threshold=0.45,
        top_k=400,
        eta=1.0,
        variance_encoded_in_target=False,
        keep_top_k=-1,
        confidence_threshold=0.0,):
    mbox_loc_dict = {"shape":(num_batch, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * 4)}
    mbox_conf_dict = {"shape":(num_batch, num_loc * num_classes *2), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * num_classes)}
    mbox_priorbox_dict = {"shape":(num_batch, 2, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, 2, num_loc * 4)}
    out_box_num_dict = {"shape":(8,), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(8,)}
    out_box_dict = {"shape":(num_batch, top_k, 8), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, top_k, 8)}

    return {"params": [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict,
                       out_box_num_dict, out_box_dict, num_classes],
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

def run_ssd_detection_runtimeerror_case2(
        num_batch, num_classes, num_loc, code_type,
        dtype="float16",
        share_location=True,
        background_label_id=0,
        iou_threshold=0.45,
        top_k=400,
        eta=1.0,
        variance_encoded_in_target=False,
        keep_top_k=-1,
        confidence_threshold=0.0,):
    mbox_loc_dict = {"shape":(num_batch, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * 4)}
    mbox_conf_dict = {"shape":(num_batch, num_loc * num_classes), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * num_classes)}
    mbox_priorbox_dict = {"shape":(num_batch, 3, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, 2, num_loc * 4)}
    out_box_num_dict = {"shape":(8,), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(8,)}
    out_box_dict = {"shape":(num_batch, top_k, 8), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, top_k, 8)}

    return {"params": [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict,
                       out_box_num_dict, out_box_dict, num_classes],
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

def run_ssd_detection_runtimeerror_case3(
        num_batch, num_classes, num_loc, code_type,
        dtype="float16",
        share_location=True,
        background_label_id=0,
        iou_threshold=0.45,
        top_k=400,
        eta=1.0,
        variance_encoded_in_target=False,
        keep_top_k=-1,
        confidence_threshold=0.0,):
    mbox_loc_dict = {"shape":(num_batch, num_loc * 4), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * 4)}
    mbox_conf_dict = {"shape":(num_batch, num_loc * num_classes), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, num_loc * num_classes)}
    mbox_priorbox_dict = {"shape":(num_batch, 2, num_loc * 5), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, 2, num_loc * 4)}
    out_box_num_dict = {"shape":(8,), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(8,)}
    out_box_dict = {"shape":(num_batch, top_k, 8), "dtype":dtype, "format":"NCHW", "ori_format":"NCHW", "ori_shape":(num_batch, top_k, 8)}

    return {"params": [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict,
                       out_box_num_dict, out_box_dict, num_classes],
            "expect": RuntimeError,
            "format_expect": [],
            "support_expect": True}

case1 = run_ssd_detection_out_case(1, 11, 9, 1)
case2 = run_ssd_detection_runtimeerror_case1(1, 21, 9, 1)
case3 = run_ssd_detection_runtimeerror_case2(1, 32, 9, 1)
case4 = run_ssd_detection_runtimeerror_case3(1, 33, 9, 1)
case5 = run_ssd_detection_out_case(1, 65, 9, 1)
case6 = run_ssd_detection_out_case(1, 81, 9, 1)
case7 = run_ssd_detection_out_case(1, 1024, 256, 2)
case8 = run_ssd_detection_out_case(1, 2, 1, 1)
case9 = run_ssd_detection_out_case(1, 2, 256, 2)

# run time too long
ut_case.add_case("Ascend910A", case1)
ut_case.add_case("Ascend910A", case2)
ut_case.add_case("Ascend910A", case3)
ut_case.add_case("Ascend910A", case4)
# ut_case.add_case("Ascend910", case5)
# ut_case.add_case("Ascend910", case6)
# ut_case.add_case("Ascend910", case7)
# ut_case.add_case("Ascend910", case8)
# ut_case.add_case("Ascend910", case9)


if __name__ == '__main__':
    ut_case.run()
