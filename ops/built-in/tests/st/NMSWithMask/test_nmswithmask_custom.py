from unittest.mock import MagicMock
from unittest.mock import patch
import te
from te.platform.cce_conf import te_set_version
from op_test_frame.ut import OpUT
from tbe.common.context import op_context
from tbe.common.context.op_context import OpContext

ut_case = OpUT("NMSWithMask", "impl.dynamic.nms_with_mask", "nms_with_mask")



def test_op_mask_generalization_1(test_arg):
    from impl.dynamic.nms_with_mask import nms_with_mask_generalization

    box_scores = {
        'shape': (-1, -1),
        'dtype': "float16",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 8),
        "range": [(16, 16), (8, 8)]
    }
    selected_boxes = {
        'shape': (-1, 5),
        'dtype': "float16",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 5),
        "range": [(16, 16), (5, 5)]
    }
    selected_idx = {
        'shape': (-1, ),
        'dtype': "int32",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(16, 16)]
    }
    selected_mask = {
        'shape': (-1, ),
        'dtype': "uint8",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(16, 16)]
    }
    iou_thr = 0.7
    generalize_config = {"mode": "keep_rank"}

    if not nms_with_mask_generalization(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr,
                                        generalize_config):
        raise Exception("Failed to call nms_with_mask_generalization in nms_with_mask.")

def side_effects(*args):
    # return vals[args]
    return True

def test_op_mask_1981_test_2():
    from impl.dynamic.nms_with_mask_common import nms_with_mask_single_core
    box_scores = {
        'shape': (-1, 8),
        'dtype': "float32",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 8),
        "range": [(2048, 2048), (8, 8)]
    }
    selected_boxes = {
        'shape': (-1, 5),
        'dtype': "float32",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, 5),
        "range": [(2048, 2048), (5, 5)]
    }
    selected_idx = {
        'shape': (-1, ),
        'dtype': "int32",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(2048, 2048)]
    }
    selected_mask = {
        'shape': (-1, ),
        'dtype': "uint8",
        "format": "ND",
        "ori_format": "ND",
        "ori_shape": (-1, ),
        "range": [(2048, 2048)]
    }
    iou_thr = 0.7
    with op_context.OpContext("dynamic"):
        with patch("te.platform.api_check_support",MagicMock(side_effect=side_effects)):
            nms_with_mask_single_core(box_scores, selected_boxes, selected_idx, selected_mask, iou_thr,
                                    "test_op_mask_1981_test_2")

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_op_mask_generalization_1)
    ut_case.run("Ascend910A")
    soc_version = te.platform.cce_conf.get_soc_spec("SOC_VERSION")
    te_set_version("Ascend920A", "VectorCore")
    test_op_mask_1981_test_2()
    te_set_version(soc_version)
