from op_test_frame.ut import OpUT

ut_case = OpUT("BasicRNNCell", "impl.basic_rnn_cell", "basic_rnn_cell")


case1 = {"params": [{"shape": (64, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1, 1024),"ori_format": "ND"}, #x
                    {"shape": (16,), "dtype": "float16", "format": "NCHW", "ori_shape": (16,),"ori_format": "NCHW"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "NCHW"}, #h,
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "ND"}, #h
                    {"shape": (64, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 1024),"ori_format": "HWCN"}, #c
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),"ori_format": "NCHW"},
                    {"shape": (32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 512),"ori_format": "HWCN"}, #c
                    {"shape": (32, 32, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (512, 512),"ori_format": "HWCN"}, #c
                    {"shape": (512,), "dtype": "float16", "format": "NCHW", "ori_shape": (512,),"ori_format": "NCHW"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 512),"ori_format": "ND"},
                    {"shape": (32, 1, 16, 16), "dtype": "float16", "format": "FRACTAL_NZ", "ori_shape": (1, 1, 512),"ori_format": "ND"},
                    True],
         "case_name": "BasicRNNCell1",
         "expect": "success",
         "support_expect": True}



'''
(x=None, cont=None, w_xc_x_static=None,
                       h=None, c=None, w_xh=None, bias=None,
                       w_xh_deqscale=None,
                       h_t=None, c_t=None, num_output=0,
                       expose_hidden=False, xh_scale=0.0,
                       sqrt_mode=False, xh_offset=0, w_xh_offset=0,
                       kernel_name="BasicRNNCell",
                       impl_mode="high_performance"):
'''

ut_case.add_case("Ascend910A", case1)

if __name__ == '__main__':
    ut_case.run("Ascend910A")