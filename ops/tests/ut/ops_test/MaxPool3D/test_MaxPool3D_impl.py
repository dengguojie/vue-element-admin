# #!/usr/bin/env python
# # -*- coding: UTF-8 -*-
# from op_test_frame.ut import OpUT

# ut_case = OpUT("MaxPool3D", "max_pool3d", "max_pool3d")

# case_6hd = {
#      "params": [{"shape" : (2, 16, 3, 16, 16, 16), "format" : "NDHWC",  "dtype" : "float16"},
#                 {"shape" : (2, 4,  3, 4,  4,  16), "format" : "NDHWC",  "dtype" : "float16"},
#                 (1, 4, 4, 4, 1),   #window
#                 (1, 4, 4, 4, 1),   #stride
#                 "SAME",            #padding_mode
#                 (0,0,0,0,0,0),     #pad
#                 (1,1,1),           #dilation
#                 1,                 #data_mode
#                 0,                 #ceil_mode
#                 "NDHWC",           #data_format
#                 #"max_pool3d"       #kernel_name
#                ],
#      "expect": "success",
#      "format_expect":[],
#      "support_expect": True
# }


# case_5d = {
#      "params": [{"shape" : (2, 16, 16, 16, 16), "format" : "NDHWC",  "dtype" : "float16"},
#                 {"shape" : (2, 8,  8,  8,  16), "format" : "NDHWC",  "dtype" : "float16"},
#                 (1, 2, 2, 2, 1),   #window
#                 (1, 2, 2, 2, 1),   #stride
#                 "SAME",            #padding_mode
#                 (0,0,0,0,0,0),     #pad
#                 (1,1,1),           #dilation
#                 1,                 #data_mode
#                 0,                 #ceil_mode
#                 "NDHWC",           #data_format
#                 #"max_pool3d"       #kernel_name
#                ],
#      "expect": "success",
#      "format_expect":[],
#      "support_expect": True
# }

# # ============ auto gen ["Ascend910"] test cases start ===============
# ut_case.add_case("all",case_6hd)
# #ut_case.add_case(["Ascend310"],case_5d)

# # ============ auto gen ["Ascend910"] test cases end =================

# if __name__ == '__main__':
#     # ut_case.run("Ascend910")
#     ut_case.run()
#     exit(0)
