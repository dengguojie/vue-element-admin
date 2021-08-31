#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
Copyright 2018 Huawei Technologies Co., Ltd

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")
def test_conv_quant_cce_conf(test_arg):
    import te
    from te import tvm
    from tbe.common import utils
    import te.lang.cce
    from te.lang.cce import cce_build_code
    from impl.conv2d import _conv_layer_cce

    def test_FP16(bias):
        shape_in = (1, 64, 8, 8)
        shape_w = (128, 64, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias,
                                kernel_name="test23_TBE_FP1622", need_build=True,
                                need_print=False)

        shape_in = (8, 64, 8, 8)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias,
                                kernel_name="test23_TBE_FP1622",  need_build=True, need_print=False)

        shape_in = (20, 64, 8, 8)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias,
                                kernel_name="test23_TBE_FP1622",  need_build=True, need_print=False)

    def test_conv_layer_dilation():
        shape_in = (1, 16, 257, 257)
        shape_w = (16, 16, 2, 2)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        dilateh = 1
        dilatew = 1

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                            padh, padw, strideh, stridew, dilateh=dilateh, dilatew=dilatew)

        shape_in = (1, 16, 257, 2)
        shape_w = (16, 16, 2, 1)
        dilateh = 255
        dilatew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                            padh, padw, strideh, stridew, dilateh=dilateh, dilatew=dilatew)

        shape_w = (512,512,3,3)
        shape_in = (2, 512, 48, 72)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                2, 2, strideh, stridew, dilateh=2, dilatew=2)

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                4, 4, strideh, stridew, dilateh=4, dilatew=4)

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                8, 8, strideh, stridew, dilateh=8, dilatew=8)

        shape_in = (2, 512, 24, 36)

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                2, 2, strideh, stridew, dilateh=2, dilatew=2)

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                4, 4, strideh, stridew, dilateh=4, dilatew=4)

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                8, 8, strideh, stridew, dilateh=8, dilatew=8)

    def test_conv_layer_dilation_check():
        shape_in = (1, 16, 257, 257)
        shape_w = (16, 16, 2, 2)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        dilateh = 256
        dilatew = 1
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, dilateh, dilatew,
                                bias=False, kernel_name="dilate_value_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        dilateh = 3
        dilatew = -1
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, dilateh, dilatew,
                                bias=False, kernel_name="dilate_value_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        dilateh = 3
        dilatew = 3
        w_dtype = "int8"
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, dilateh, dilatew,
                                bias=False, kernel_name="dilate_w_dtype_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

    def test_conv_layer_c0_optim():
        shape_in = (1, 3, 8, 8)
        shape_w = (128, 3, 2, 2)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        bias = True
        optim_dict = {"c0_optim_flg": True}

        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="test_c0_optim_1", need_build=True, need_print=False)
        shape_in = (8, 3, 8, 8)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="test_c0_optim_2",  need_build=True, need_print=False)
        shape_in = (20, 3, 8, 8)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="test_c0_optim_3",  need_build=True, need_print=False)
        shape_in = (20, 3, 8, 8)
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="test_c0_optim_3",  need_build=True, need_print=False)

        shape_in = (1,3,224,224)
        shape_w = (64,3,7,7)
        padh = 3
        padw = 3
        strideh = 2
        stridew = 2
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case01",
                                need_build=True, need_print=True)

        shape_in = (1,1,299,299)
        shape_w = (32,1,3,3)
        padh = 0
        padw = 0
        strideh = 2
        stridew = 2
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case02",
                                need_build=True, need_print=True)

        shape_in = (1,4,4,4)
        shape_w = (32,4,3,3)
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case03",
                                need_build=True, need_print=True)

        shape_in = (1,4,4,4)
        shape_w = (32,4,2,2)
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case04",
                                need_build=True, need_print=True)

        shape_in = (1,1,25,25)
        shape_w = (64,1,5,5)
        padh = 2
        padw = 2
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case05",
                                need_build=True, need_print=True)

        shape_in = (1,1,95,95)
        shape_w = (64,1,5,5)
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case09",
                                need_build=True, need_print=True)

        shape_in = (1,1,96,128)
        shape_w = (8,1,3,3)
        padh = 1
        padw = 1
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=False,
                                optim_dict={"c0_optim_flg": True},
                                kernel_name="conv_case10",
                                need_build=True, need_print=True)

    def test_conv_layer_c0_optim_check():
        shape_in = (1, 5, 8, 8)
        shape_w = (128, 5, 2, 2)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        bias = True
        optim_dict = {"c0_optim_flg": True}

        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="c0_optim_shape_check", need_build=True, need_print=False)
        except Exception as e:
            print(e)

        optim_dict = {"c0_optim_flg": 1234}
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew, bias=bias, optim_dict=optim_dict,
                                kernel_name="c0_optim_dtype_check", need_build=True, need_print=False)
        except Exception as e:
            print(e)

        w_dtype = "int8"
        shape_in = (1, 4, 8, 8)
        shape_w = (128, 4, 2, 2)
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=bias, optim_dict=optim_dict, kernel_name="c0_optim_dtype_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

    def shape_check():
        shape_in = (1, 80, 16, 48)
        shape_w = (96, 96, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # shape_in[1] must equal to shape_w[1]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 1024, 1024)
        shape_w = (1024, 96, 1, 1)
        # feature H*feature W*Cout must <= 2^24
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 2049, 8)
        shape_w = (32, 96, 1, 1)
        # feature H must be in [1,2048]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 8, 3000)
        shape_w = (32, 96, 1, 1)
        # feature W must be in [1,2048]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        w_dtype = "int8"
        shape_in = (1, 96, 8, 1555)
        shape_w = (32, 96, 1, 1)
        # feature H/W must be in [1,1280]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 8, 8)
        shape_w = (32, 96, 10, 10)
        strideh = 1
        stridew = 1
        # feature H must >= kernel H
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_w = (32, 96, 2, 10)
        # feature W must >= kernel W
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 16, 48)
        shape_w = (96, 96, 8, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 9
        padw = 0
        strideh = 1
        stridew = 1
        # kernel H must > Pad H
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 18, 18)
        shape_w = (32, 96, 15, 15)
        padh = 12
        padw = 1
        # padh must be in [0,11]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        padh = 1
        padw = 12
        # padw must be in [0,11]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 96, 16, 48)
        shape_w = (96, 96, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 64
        stridew = 1
        # strideh must be in [1,63]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        strideh = 1
        stridew = 64
        # stridew must be in [1,63]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 16, 290, 16)
        shape_w = (1, 16, 280, 8)
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # kernel H must be in [1,255]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 16, 16, 290)
        shape_w = (1, 16, 8, 280)
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # kernel W must be in [1,255]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="shape_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 16, 255, 255)
        shape_w = (1, 16, 255, 255)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # kernel H*W must be in [1,255]
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

    def buffer_check():
        shape_in = (1, 2048, 8, 1024)
        shape_w = (8, 2048, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # min cut is out of half of L1 memory fp16
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 2048, 8, 1024)
        shape_w = (8, 2048, 1, 1)
        in_dtype = "int8"
        w_dtype = "int8"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # min cut is out of half of L1 memory int8
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 2048, 1024, 1024)
        shape_w = (8, 2048, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # im2col row major shape exceed 32bit limitation
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 1024, 1024, 1024)
        shape_w = (8, 1024, 1024, 1024)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # im2col fractal shape exceed 32bit limitation
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (1, 16, 255, 64)
        shape_w = (1, 16, 255, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 63
        stridew = 1
        # L1 buffer overflow!
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (100000, 16, 255, 64)
        shape_w = (1, 16, 255, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # Input feature exceed 32 bit limitations
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

        shape_in = (100000, 16, 255, 64)
        shape_w = (1000, 16, 1, 1)
        in_dtype = "float16"
        w_dtype = "float16"
        res_dtype = "float16"
        padh = 0
        padw = 0
        strideh = 1
        stridew = 1
        # Output feature exceed 32 bit limitations!
        try:
            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="buffer_check",  need_build=True, need_print=False)
        except Exception as e:
            print(e)

    def test_default_tiling():
        shape_in = (1, 3, 416, 988)
        shape_w = (32, 3, 3, 3)
        in_dtype = "float16"
        w_dtype = "int8"
        res_dtype = "int8"
        padh = 1
        padw = 9
        strideh = 1
        stridew = 1
        _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                padh, padw, strideh, stridew,
                                bias=False, kernel_name="test_default_tiling",
                                need_build=True, need_print=False)

    def test_speculate_conv():
            m = 13
            k = 3
            n = 4

            shape_in = (1, 64, 104, 104)
            shape_w = (128, 64, 3, 3)
            in_dtype = "float16"
            w_dtype = "float16"
            res_dtype = "float16"
            padh = 1
            padw = 1
            strideh = 1
            stridew = 1

            _conv_layer_cce(shape_in, shape_w, in_dtype, w_dtype, res_dtype,
                                    padh, padw, strideh, stridew, bias=False,
                                    kernel_name="wlf",  need_build=True, need_print=False)
    def test_conv1d_split_w():
        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 82),
        in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 82),
        in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=2)

        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 1),
        in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 1), shape_w = (16, 301, 1, 1),
        in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 1), shape_w = (16, 301, 1, 1),
        in_dtype = "int8", w_dtype = "int8", res_dtype = "int32",
        padh = [0, 0], padw = [34, 23], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 82),
        in_dtype = "int8", w_dtype = "int8", res_dtype = "int32",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 82),
        in_dtype = "int8", w_dtype = "int8", res_dtype = "int32",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 750232), shape_w = (47151, 301, 1, 1),
        in_dtype = "int8", w_dtype = "int8", res_dtype = "int32",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

        _conv_layer_cce(shape_in = (7, 301, 1, 1), shape_w = (16, 301, 1, 1),
        in_dtype = "int8", w_dtype = "int8", res_dtype = "int32",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)

    def test_conv_H_extend_to_100000():
        _conv_layer_cce(shape_in = (1, 16, 99999, 32), shape_w = (16, 16, 3, 3),
        in_dtype = "float16", w_dtype = "float16", res_dtype = "float16",
        padh = [0, 0], padw = [0, 0], strideh = 1, stridew = 1, dilateh=1, dilatew=1)



    """
    The UT for conv layer cce
    """

    print("-----------------------------------------------------")
    # print("[conv UNITTEST START topi/python/topi/cce/conv_layer.py]")
    # def test_conv_layer_cce_case1():

    #     print("[conv layer ut test case 1: test FP16]")
    #     test_FP16(bias=False)
    #     test_FP16(bias=True)

    def test_conv_layer_cce_case2():

        print("[conv layer ut test case 2: dilation]")
        test_conv_layer_dilation()
        test_conv_layer_dilation_check()
        test_conv1d_split_w()
        test_conv_H_extend_to_100000()

    # def test_conv_layer_cce_case3():

    #     print("[conv layer ut test case 3: C0 = 4 optimization]")
    #     test_conv_layer_c0_optim()
    #     test_conv_layer_c0_optim_check()

    # def test_conv_layer_cce_case4():
    #     print("[conv layer ut test case 4: shape check and buffer check]")
    #     shape_check()
    #     buffer_check()

    # def test_conv_layer_cce_case5():
    #     print("[conv layer ut test case 5: default tiling check]")
    #     test_default_tiling()

    # def test_conv_layer_cce_case6():
    #     print("[conv layer ut test case 5: test_storage_rewrite_cce]")
    #     test_speculate_conv()

print("adding Conv2D cce testcases")
ut_case.add_cust_test_func(test_func=test_conv_quant_cce_conf)
