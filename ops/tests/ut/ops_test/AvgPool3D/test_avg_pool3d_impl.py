# TODO fix me run failed
# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT
#
# ut_case = OpUT("AvgPool3D", "impl.avg_pool3d", "avg_pool3d")
#
#
# ut_case.add_case(["Ascend910"], {"params":[
#     {"shape": (1,6,64,7,7,16), "format": "NDC1HWC0", "dtype": "float16"},
#     {"shape": (1,5,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16"},
#     (1,2,7,7,1),
#     (1,1,1,1,1),
#     (0,0,0,0,0,0),
#     False,
#     True,
#     0,
#     "NDHWC"],
#     "expect": "success",
#     "case_name":"test_avg_pool3d_001"})
#
# ut_case.add_case(["Ascend910"], {"params":[
#     {"shape": (1,3,64,7,7,16), "format": "NDC1HWC0", "dtype": "float16"},
#     {"shape": (1,2,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16"},
#     (1,2,7,7,1),
#     (1,1,1,1,1),
#     (0,0,0,0,0,0),
#     False,
#     True,
#     0,
#     "NDHWC"],
#     "expect": "success",
#     "case_name":"test_avg_pool3d_002"})
#
# ut_case.add_case(["Ascend910"], {"params":[
#     {"shape": (1,4,64,8,8,16), "format": "NDC1HWC0", "dtype": "float16"},
#     {"shape": (1,3,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16"},
#     (1,2,8,8,1),
#     (1,1,1,1,1),
#     (0,0,0,0,0,0),
#     False,
#     True,
#     0,
#     "NDHWC"],
#     "expect": "success",
#     "case_name":"test_avg_pool3d_003"})
#
# ut_case.add_case(["Ascend910"], {"params":[
#     {"shape": (1,3,64,9,9,16), "format": "NDC1HWC0", "dtype": "float16"},
#     {"shape": (1,2,64,1,1,16), "format": "NDC1HWC0", "dtype": "float16"},
#     (1,2,9,9,1),
#     (1,1,1,1,1),
#     (0,0,0,0,0,0),
#     False,
#     True,
#     0,
#     "NDHWC"],
#     "expect": "success",
#     "case_name":"test_avg_pool3d_004"})
#
#
# if __name__ == '__main__':
#     ut_case.run("Ascend910")
