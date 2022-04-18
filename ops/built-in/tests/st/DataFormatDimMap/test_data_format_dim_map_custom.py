from impl.dynamic.data_format_dim_map import get_op_support_info

def test_get_op_support_info():
    get_op_support_info({"shape": (128, 128, 128, 128), "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"},
                        {"shape": (128, 128, 128, 128),  "dtype": "float16", "format": "NHWC", "ori_shape": (128, 128, 128, 128),"ori_format": "NHWC"})

if __name__ == '__main__':
    test_get_op_support_info()
