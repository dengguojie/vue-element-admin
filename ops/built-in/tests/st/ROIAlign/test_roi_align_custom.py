#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
from unittest.mock import MagicMock
from unittest.mock import patch
from te import platform as cce_conf
from impl.util.platform_adapter import tbe_context
from impl.roi_align import roi_align as roi_align_static
from impl.dynamic.roi_align import roi_align as roi_align_dynamic
from tbe.common.context import op_context


def test_roi_align_static():
    def test_roi_align_static_001():
        input_list = [{"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                        "ori_format": "NHWC"},
                        {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),
                        "ori_format": "NHWC"},
                        None,
                        {"shape": (1, 16, 38, 64, 16), "dtype": "float32", "format": "NHWC",
                        "ori_shape": (2, 260, 1, 1, 16), "ori_format": "NHWC"},
                        0.25,
                        7,
                        7,
                        2,
                        1,
                        "roi_align_static"]
        roi_align_static(*input_list)
        input_list[7] = -2
        roi_align_static(*input_list)

    def test_roi_align_static_002():
        input_list = [{"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                        "ori_format": "NHWC"},
                        {"shape": (256, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (256, 5),
                        "ori_format": "NHWC"},
                        None,
                        {"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC",
                        "ori_shape": (2, 260, 1, 1, 16), "ori_format": "NHWC"},
                        0.25,
                        7,
                        7,
                        2,
                        1,
                        "roi_align_static"
                        ]
        roi_align_static(*input_list,impl_mode="high_precision")
        input_list[7] = -2
        roi_align_static(*input_list,impl_mode="high_precision")

    def test_roi_align_static_003():
        input_list = [{"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC", "ori_shape": (2, 260, 1, 16),
                        "ori_format": "NHWC"},
                        {"shape": (256, 5), "dtype": "float16", "format": "NHWC", "ori_shape": (256, 5),
                        "ori_format": "NHWC"},
                        None,
                        {"shape": (1, 16, 38, 64, 16), "dtype": "float16", "format": "NHWC",
                        "ori_shape": (2, 260, 1, 1, 16), "ori_format": "NHWC"},
                        0.25,
                        7,
                        7,
                        2,
                        1,
                        "roi_align_static"]
        roi_align_static(*input_list)
        input_list[7] = -2
        roi_align_static(*input_list)

    with op_context.OpContext():
        TEST_PLATFORM = ["Ascend610", "Ascend910"]
        for soc in TEST_PLATFORM:
            cce_conf.te_set_version(soc)
            test_roi_align_static_001()
            test_roi_align_static_002()
            test_roi_align_static_003()

def test_roi_align_dynamic():
    input_list = [{"shape": (2, 16, 336, 336, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (2, 256, 336, 336),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    {"shape": (1280, 5), "dtype": "float32", "format": "ND", "ori_shape": (1280, 5),
                     "ori_format": "ND", "range":((1,1),(1,1))},
                    {"shape": (1280,), "dtype": "int32", "format": "ND", "ori_shape": (1280,),
                     "ori_format": "ND", "range":((1280,1280),)},
                    {"shape": (1280, 16, 7, 7, 16), "dtype": "float32", "format": "NC1HWC0",
                     "ori_shape": (1280, 256, 7, 7),"ori_format": "NCHW", "range":((1,1), (1,1), (1,1), (1,1), (1,1))},
                    0.25, 7, 7, 2, 1, "roi_align_dynamic"]
    with tbe_context.op_context.OpContext("dynamic"):
        roi_align_dynamic(*input_list)



if __name__ == '__main__':
    test_roi_align_static()
    soc_version = cce_conf.get_soc_spec("SHORT_SOC_VERSION")
    cce_conf.te_set_version("Ascend310P3")
    vals = {("tik.vextract", "float16"): False,
            ("tik.vextract", "float32"): False,
            ("tik.vdiv", "float32"): True
            # ("tik.vgatherb", "float32"): True
            }
    def side_effects(*args):
        return vals[args]
    # with patch("te.platform.api_check_support", MagicMock(side_effect=side_effects)):
    #     test_roi_align_static()
    # with patch("impl.util.platform_adapter.tik.Dprofile.get_l1_buffer_size", MagicMock(return_value=0)):
    #    test_roi_align_dynamic()

    cce_conf.te_set_version(soc_version)
