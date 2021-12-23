from unittest.mock import MagicMock
from unittest.mock import patch
from op_test_frame.ut import OpUT
from impl.batch_matmul import batch_matmul
from impl.batch_matmul import op_select_format as op_select_format_v1
from impl.batch_matmul_v2 import batch_matmul_v2
from impl.batch_matmul_v2 import op_select_format as op_select_format_v2


from tbe.common.context import op_context

ut_case = ut_case = OpUT("BatchMatMul", "impl.batch_matmul", "batch_matmul")

vals = {("CORE_NUM", ): 48,
        ("CUBE_VECTOR_SPLIT",): True,
        ("UB_SIZE", ): 196608,
        ("L0A_SIZE", ): 65536,
        ("L0B_SIZE", ): 65536,
        ("L1_SIZE", ): 524288,
        ("L0C_SIZE", ): 131072,
        ("Intrinsic_fix_pipe_l0c2out",): True,
        ("Intrinsic_fix_pipe_unit_list",): True,
        ("Intrinsic_fix_pipe_unit_list", "post_eltwise"): True
        }
def side_effects(*args):
    return vals[args]

case1 = {"params": [{"shape": (2, 4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 32, 64),"ori_format": "ND"},
                    {"shape": (2, 4, 1, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (2, 4, 64, 16),"ori_format": "ND"},
                    None,
                    {"shape": (1, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (32, 16),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v1__impl_001",
         "expect": "success",
         "support_expect": True}

case2 = {"params": [{"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 64),"ori_format": "ND"},
                     {"shape": (4, 4, 4, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 64, 49),"ori_format": "ND"},
                     {"shape": (49,), "dtype": "float32", "format": "ND", "ori_shape": (49,),"ori_format": "ND"},
                    None,
                    {"shape": (4, 4, 2, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (4, 32, 49),"ori_format": "ND"},
                    False,False,
                    ],
         "case_name": "BatchMatmul_v2_impl_002",
         "expect": "success",
         "support_expect": True}

# test mock case
def test_mock_cases():
    with patch("tbe.common.platform.platform_info.get_soc_spec", MagicMock(side_effect=side_effects)):
        with patch("tbe.common.platform.platform_info.intrinsic_check_support", MagicMock(side_effect=side_effects)):
            op_select_format_v1(*case1["params"])
            op_select_format_v2(*case2["params"])

#ut_case.add_cust_test_func(test_func=test_mock_cases)

if __name__ == "__main__":
    with op_context.OpContext():
        test_mock_cases()