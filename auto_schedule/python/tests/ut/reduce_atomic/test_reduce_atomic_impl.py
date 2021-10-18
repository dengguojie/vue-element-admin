# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT

ut_case = OpUT("reduce_atomic", "impl.kl_div", "kl_div")

case1 = {"params": [{"shape": (1000, 1, 7), "dtype": "float16", "format": "ND",
                     "ori_shape": (1000, 1, 7), "ori_format": "ND", "addr_type": 0,
                     "total_shape": [1000, 1, 7], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_x'},
                    {"shape": (1000, 1, 7), "dtype": "float16", "format": "ND",
                     "ori_shape": (1000, 1, 7), "ori_format": "ND", "addr_type": 0,
                     "total_shape": [1000, 1, 7], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_y'},
                    {"shape": (1, ), "dtype": "float16", "format": "ND",
                     "ori_shape": (1, ), "ori_format": "ND", "addr_type": 0,
                     "total_shape": [1, ], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_x'},
                     "batchmean"],
         "case_name": "dsl_reduce_multi_1",
         "expect": "success",
         "support_expect": True}

compile_case_list = [
    case1,
]

for item in compile_case_list:
    ut_case.add_case(["Ascend710"], case=item)
    
if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend710"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
