# # -*- coding:utf-8 -*-
from sch_test_frame.ut import OpUT
import warnings

from layer_norm import layer_norm

warnings.filterwarnings("ignore")


def dsl_reduce_multi_ln(input_x, input_gamma, input_beta, output_y, output_mean, output_variance,
                        begin_norm_axis, begin_params_axis, epsilon=1e-12, kernel_name="layer_norm"):
    layer_norm(input_x, input_gamma, input_beta,
               output_y, output_mean, output_variance,
               begin_norm_axis, begin_params_axis,
               epsilon)


ut_case = OpUT("reduce_multi", "reduce_multi.test_reduce_multi_ln_impl", "dsl_reduce_multi_ln")


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
         "case_name": "test_dsl_reduce_multi_ln_01",
         "expect": "success",
         "support_expect": True}

compile_case_list = [
    case1,
]

for item in compile_case_list:
    ut_case.add_case(["Ascend910A"], case=item)
    
if __name__ == '__main__':
    import os
    from pathlib import Path

    _ASCEND_TOOLCHAIN_PATH_ENV = "TOOLCHAIN_HOME"
    simulator_lib_path = Path(os.environ.get(_ASCEND_TOOLCHAIN_PATH_ENV,
                                             "/usr/local/Ascend/toolkit")).joinpath("tools/simulator")
    ut_case.run(["Ascend910A"], simulator_mode="pv", simulator_lib_path=simulator_lib_path)
