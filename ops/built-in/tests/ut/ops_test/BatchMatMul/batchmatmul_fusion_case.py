batchmatmul_ut_fusion_case = [
    {
        "params": [
            {"shape": (195, 146, 16, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (195, 256, 2336), "ori_format": "ND"},
            {"shape": (195, 96, 16, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (195, 256, 1536), "ori_format": "ND"}, 
            None, 
            {"shape": (195, 96, 146, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (195, 2336, 1536), "ori_format": "ND"},
            True, False, 
            {"shape": (1, 96, 146, 16, 16), "dtype": "float32", "format": "FRACTAL_NZ", "ori_shape": (2336, 1536), "ori_format": "ND"}, 
            0
        ],
        "case_name": "reduce_sum_case_1",
        "expect": "success",
        "support_expect": True
    },

    {
        "params": [
            {"shape": (24, 16, 32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512), "ori_format": "ND"},
            {"shape": (24, 16, 32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512), "ori_format": "ND"},
            None,
            {"shape": (24, 16, 32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512), "ori_format": "ND"},
            True, False,
            {"shape": (1, 1, 1, 1, 1, 1), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (), "ori_format": "ND"},
            {"shape": (24, 16, 32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (24, 16, 512, 512), "ori_format": "ND"},
        ],
        "case_name": "fused_mul_add_case_1",
        "expect": "success",
        "support_expect": True
    }
]
