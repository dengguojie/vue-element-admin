#!/usr/bin/env python
# -*- coding: UTF-8 -*-
from impl.ssd_detection_output import ssd_detection_output


def run_ssd_detection_out_case(
    num_batch,
    num_classes,
    num_loc,
    code_type,
    dtype="float16",
    share_location=True,
    background_label_id=0,
    iou_threshold=0.45,
    top_k=400,
    eta=1.0,
    variance_encoded_in_target=False,
    keep_top_k=-1,
    confidence_threshold=0.0,
):
    mbox_loc_dict = {
        "shape": (num_batch, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * 4)
    }
    mbox_conf_dict = {
        "shape": (num_batch, num_loc * num_classes),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * num_classes)
    }
    mbox_priorbox_dict = {
        "shape": (num_batch, 2, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, 2, num_loc * 4)
    }
    out_box_num_dict = {"shape": (8,), "dtype": dtype, "format": "NCHW", "ori_format": "NCHW", "ori_shape": (8,)}

    if keep_top_k == -1:
        out_box_dict = {
            "shape": (num_batch, top_k, 8),
            "dtype": dtype,
            "format": "NCHW",
            "ori_format": "NCHW",
            "ori_shape": (num_batch, top_k, 8)
        }
    else:
        out_box_dict = {
            "shape": (num_batch, keep_top_k, 8),
            "dtype": dtype,
            "format": "NCHW",
            "ori_format": "NCHW",
            "ori_shape": (num_batch, keep_top_k, 8)
        }

    return [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict, out_box_num_dict, out_box_dict, num_classes]


def run_ssd_detection_runtimeerror_case1(
    num_batch,
    num_classes,
    num_loc,
    code_type,
    dtype="float16",
    share_location=True,
    background_label_id=0,
    iou_threshold=0.45,
    top_k=400,
    eta=1.0,
    variance_encoded_in_target=False,
    keep_top_k=-1,
    confidence_threshold=0.0,
):
    mbox_loc_dict = {
        "shape": (num_batch, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * 4)
    }
    mbox_conf_dict = {
        "shape": (num_batch, num_loc * num_classes * 2),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * num_classes)
    }
    mbox_priorbox_dict = {
        "shape": (num_batch, 2, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, 2, num_loc * 4)
    }
    out_box_num_dict = {"shape": (8,), "dtype": dtype, "format": "NCHW", "ori_format": "NCHW", "ori_shape": (8,)}
    out_box_dict = {
        "shape": (num_batch, top_k, 8),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, top_k, 8)
    }

    return [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict, out_box_num_dict, out_box_dict, num_classes]


def run_ssd_detection_runtimeerror_case2(
    num_batch,
    num_classes,
    num_loc,
    code_type,
    dtype="float16",
    share_location=True,
    background_label_id=0,
    iou_threshold=0.45,
    top_k=400,
    eta=1.0,
    variance_encoded_in_target=False,
    keep_top_k=-1,
    confidence_threshold=0.0,
):
    mbox_loc_dict = {
        "shape": (num_batch, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * 4)
    }
    mbox_conf_dict = {
        "shape": (num_batch, num_loc * num_classes),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * num_classes)
    }
    mbox_priorbox_dict = {
        "shape": (num_batch, 3, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, 2, num_loc * 4)
    }
    out_box_num_dict = {"shape": (8,), "dtype": dtype, "format": "NCHW", "ori_format": "NCHW", "ori_shape": (8,)}
    out_box_dict = {
        "shape": (num_batch, top_k, 8),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, top_k, 8)
    }

    return [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict, out_box_num_dict, out_box_dict, num_classes]


def run_ssd_detection_runtimeerror_case3(
    num_batch,
    num_classes,
    num_loc,
    code_type,
    dtype="float16",
    share_location=True,
    background_label_id=0,
    iou_threshold=0.45,
    top_k=400,
    eta=1.0,
    variance_encoded_in_target=False,
    keep_top_k=-1,
    confidence_threshold=0.0,
):
    mbox_loc_dict = {
        "shape": (num_batch, num_loc * 4),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * 4)
    }
    mbox_conf_dict = {
        "shape": (num_batch, num_loc * num_classes),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, num_loc * num_classes)
    }
    mbox_priorbox_dict = {
        "shape": (num_batch, 2, num_loc * 5),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, 2, num_loc * 4)
    }
    out_box_num_dict = {"shape": (8,), "dtype": dtype, "format": "NCHW", "ori_format": "NCHW", "ori_shape": (8,)}
    out_box_dict = {
        "shape": (num_batch, top_k, 8),
        "dtype": dtype,
        "format": "NCHW",
        "ori_format": "NCHW",
        "ori_shape": (num_batch, top_k, 8)
    }

    return [mbox_loc_dict, mbox_conf_dict, mbox_priorbox_dict, out_box_num_dict, out_box_dict, num_classes]


def test_ssd_detection_out_case():
    ssd_detection_output(*(run_ssd_detection_out_case(1, 11, 9, 1)))
    ssd_detection_output(*(run_ssd_detection_out_case(1, 65, 9, 1)))
    ssd_detection_output(*(run_ssd_detection_out_case(1, 81, 9, 1)))
    ssd_detection_output(*(run_ssd_detection_out_case(1, 1024, 256, 2)))
    ssd_detection_output(*(run_ssd_detection_out_case(1, 2, 1, 1)))
    ssd_detection_output(*(run_ssd_detection_out_case(1, 2, 256, 2)))


def test_run_ssd_detection_runtimeerror():
	try:
		ssd_detection_output(*(run_ssd_detection_runtimeerror_case1(1, 21, 9, 1)))
	except RuntimeError as e:
		pass

	try:
		ssd_detection_output(*(run_ssd_detection_runtimeerror_case2(1, 32, 9, 1)))
	except RuntimeError as e:
		pass

	try:
		ssd_detection_output(*(run_ssd_detection_runtimeerror_case3(1, 33, 9, 1)))
	except RuntimeError as e:
		pass


if __name__ == '__main__':
    test_ssd_detection_out_case()
    test_run_ssd_detection_runtimeerror()
