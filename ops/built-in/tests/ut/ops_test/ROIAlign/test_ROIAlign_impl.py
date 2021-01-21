
#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT
from op_test_frame.common import precision_info

ut_case = OpUT("ROIAlign", "impl.roi_align", "roi_align")

case1 = {"params": [{"shape": (1,16,38,64,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    {"shape": (256, 5), "dtype": "float32", "format": "NHWC", "ori_shape": (256, 5),"ori_format": "NHWC"},
                    None,
                    {"shape": (1,16,38,64,16), "dtype": "float32", "format": "NHWC", "ori_shape": (2,260,1,1,16),"ori_format": "NHWC"},
                    0.25,
                    7,
                    7,
                    2,
                    1
                    ],
         "case_name": "roi_align_01",
         "expect": "success",
         "format_expect": [],
         "support_expect": True}


ut_case.add_case(["Ascend310", "Ascend710", "Ascend910", "Hi3796CV300CS"], case1)

ut_case.run(["Ascend310", "Ascend710", "Ascend910", "Hi3796CV300CS"])
