# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import tbe


ut_case = OpUT("LayerNorm", "impl.layer_norm", "layer_norm")

case1 = {"params": [{"shape": (1000, 21, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (1000, 48, 324), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [1000, 21, 3, 16, 16], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_x'},
                    {"shape": (324, ), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (324, ), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [324, ], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_gamma'},
                    {"shape": (324,), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (324,), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [324, ], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'input_beta'},
                    {"shape": (1000, 21, 3, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ",
                     "ori_shape": (1000, 48, 324), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [1000, 21, 3, 16, 16], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'output_y'},
                    {"shape": (1000, 48, 1), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1000, 48, 1), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [1000, 48, 1], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'output_mean'},
                    {"shape": (1000, 48, 1), "dtype": "float16", "format": "NCHW",
                     "ori_shape": (1000, 48, 1), "ori_format": "NCHW", "addr_type": 0,
                     "total_shape": [1000, 48, 1], "slice_offset": (), "L1_addr_offset": 0,
                     "L1_fusion_type": -1, "L1_workspace_size": -1, 'valid_shape': (), "split_index": 0,
                     "param_name": 'output_variance'},
                    -1, -1, 9.999999960041972e-13],
         "case_name": "layer_norm_1",
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
