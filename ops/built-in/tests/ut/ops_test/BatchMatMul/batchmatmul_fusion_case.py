batchmatmul_ut_fusion_case =[
    # batch_matmul + reduce_sum
    (
        ["Ascend910A"],
        {"shape":(195,146,16,16,16), "dtype":"float16", "format":"FRACTAL_NZ", "ori_shape": (195,256,2336),"ori_format":"ND"},
        {"shape":(195,96,16,16,16), "dtype":"float16", "format":"FRACTAL_NZ", "ori_shape": (195,256,1536),"ori_format":"ND"},
        None,
        {"shape":(195,96,146,16,16), "dtype":"float32", "format":"FRACTAL_NZ", "ori_shape": (195,2336,1536),"ori_format":"ND"},
        True,
        False,
        {"shape":(1,96,146,16,16), "dtype":"float32", "format":"FRACTAL_NZ", "ori_shape": (2336,1536),"ori_format":"ND"},
        0
    )
]
