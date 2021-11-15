# -*- coding:utf-8 -*-
from __future__ import absolute_import
from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv2d_int4(test_arg):
    import unittest
    import os
    import shutil
    import sys
    from impl.conv2d import conv2d_compute
    from impl.eltwise import eltwise_compute
    from te import tvm
    from topi import generic
    import te.lang.cce
    from te import platform as cceconf
    from functools import reduce
    from tbe.common.buildcfg import build_config

    import conv_int4_testcase as tc
    from impl.ascend_dequant import ascend_dequant_compute
    from impl.ascend_quant import ascend_quant_compute
    from impl.strided_write import strided_write_compute
    from impl.strided_read import strided_read_compute
    from impl.write_select import write_select_compute
    from impl.relu6 import relu6_compute
    from impl.leaky_relu import leaky_relu_compute
    from impl.conv2d_data_rm import conv2d_data_rm_compute
    from impl import load_to_l1
    from impl import store_to_gm
    from te.platform.cce_policy import disableL2
    from te.platform.cce_policy import enableL2
    from te.platform.cce_policy import set_L1_info
    from tbe.common.context import op_context
    from tbe.common.context import get_context
    from tbe.tvm.buffer_manager import get_buffer_manager

    def conv_rmpad_case(version, dataflow, l1fusion_dict,
                        shape_in, shape_w, pads, strides, offset_d,
                        bias_flag, relu_flag, vector_flag, stride_swrite,
                        quant_scale=1, quant_offset=0,
                        kernel_name="conv_rmpad", invalid_data_rm_flag=False):
        
        Ni, Ci, Hi, Wi = shape_in
        Co, _, Hk, Wk = shape_w
        
        if dataflow in (0, 1, 9):
            Ci1 = (Ci + 15) // 16
            Ci0 = 16
        elif dataflow in (2, 3, 4, 11, 12):
            Ci1 = (Ci + 31) // 32
            Ci0 = 32
        else:
            Ci1 = (Ci + 63) // 64
            Ci0 = 64

        if dataflow in (2, 3, 4, 11, 12):
            Co = ((Co + 31) // 32)*32
        
        if dataflow in (5, 7, 8, 9, 13, 14):
            Co = ((Co + 63) // 64)*64
        
        Co1 = (Co + 15) // 16
        Co0 = 16

        shape_in_5HD = (Ni, Ci1, Hi, Wi, Ci0)
        shape_w_fracz = (Hk*Wk*Ci1, Co1, Co0, Ci0)

        if vector_flag:
            shape_scale = (1, Co1, 1, 1, 16)
        else:
            shape_scale = (1, 1, 1, 1, 1)

        dilations = [1, 1, 1, 1]
        strides = [1, 1, strides[0], strides[1]]

        fm_addr_type = l1fusion_dict["fm_addr_type"]
        l1_fusion_type = l1fusion_dict["l1_fusion_type"]
        out_addr_type = l1fusion_dict["out_addr_type"]
        out_16_addr_type = l1fusion_dict["out_16_addr_type"]

        fm_valid_shape = l1fusion_dict["fm_valid_shape"]
        fm_offset = l1fusion_dict["fm_offset"]

        ws_s4_valid_shape = l1fusion_dict["ws_s4_valid_shape"]
        ws_s8_valid_shape = l1fusion_dict["ws_s8_valid_shape"]
        ws_fp16_valid_shape = l1fusion_dict["ws_fp16_valid_shape"]

        l1fusion_stride_swrite = l1fusion_dict["l1fusion_stride_swrite"]

        l1_space = l1fusion_dict["l1_space"]

        if l1_fusion_type != -1:
            set_L1_info("op_L1_space", l1_space*1024)
            set_L1_info("L1_fusion_enabled", True)

        # L1 valid size
        if fm_valid_shape:
            L1_valid_size = reduce((lambda x, y: x*y), fm_valid_shape)
        else:
            L1_valid_size = reduce((lambda x, y: x*y), shape_in_5HD)
        if dataflow in (5, 7, 8, 13, 14):
            L1_valid_size = int(L1_valid_size*0.5)
        outputs = {"addr_type": out_addr_type}

        in_dtype_map = {5: "int4"}
        bias_dtype_map = {5: "int32"}
        dequant_scale_dtype_map = {"v100": "float16", "v200": "uint64"}

        strided_read_flag = False

        buffer_manager = get_buffer_manager()
        buffer_manager.set_l1_fusion_type(int(l1_fusion_type))
        with build_config(enable_L1_fusion = int(l1_fusion_type)):
            with tvm.target.cce():
                if fm_addr_type:
                    load_to_l1({"shape": shape_in_5HD, "dtype": in_dtype_map[dataflow]}, None, kernel_name=kernel_name + "_fm_l1")
                
                fm = tvm.placeholder(shape_in_5HD, name='fmap', dtype=in_dtype_map[dataflow],
                                    attrs={'ori_format': 'NCHW', 'addr_type': fm_addr_type,
                                            'valid_shape': fm_valid_shape, 'slice_offset': fm_offset,
                                            'L1_fusion_type': l1_fusion_type, "L1_valid_size": L1_valid_size, "L1_addr_flag": 1})
                if strided_read_flag:
                    fm_ori = fm
                    fm = strided_read_compute(fm_ori, {"shape": shape_in_5HD}, 1, 0, "strided_read")
                
                filter_w = tvm.placeholder(shape_w_fracz, name='filter_w', dtype=in_dtype_map[dataflow],
                                        attrs={'ori_shape': shape_w, 'ori_format': 'NCHW'})
                
                if bias_flag:
                    bias_tensor = tvm.placeholder((Co1*Co0, ), name='bias_tensor', dtype=bias_dtype_map[dataflow])
                else:
                    bias_tensor =None

                conv_res = conv2d_compute(fm, filter_w, bias_tensor, None, outputs, strides, pads, dilations, offset_x=offset_d, options={"invalid_data_rm": invalid_data_rm_flag})
                if dataflow in (2, 3, 4, 5, 7, 8, 11, 12, 13, 14):
                    vdeq = tvm.placeholder(shape_scale, name='vreq_reg', dtype=dequant_scale_dtype_map[version],
                                        attrs={'ori_shape': [Co1*Co0 if vector_flag else 1]})
                
                if dataflow == 5:
                    # conv + dequant + relu6
                    dequant = ascend_dequant_compute(conv_res, vdeq, None, sqrt_mode=False, relu_flag=relu_flag)
                    # out = relu6_compute(dequant, None)
                    out = dequant
                    # out = leaky_relu_compute(dequant, None)
                    if ws_fp16_valid_shape:
                        out = write_select_compute(out, {"valid_shape": ws_fp16_valid_shape})
                    
                    if l1fusion_stride_swrite:
                        y = {"shape": tuple(i.value for i in out.shape)}
                        out = strided_write_compute(out, y, 1, l1fusion_stride_swrite, "strided_write")
                    
                    if stride_swrite: # for stridedwrite in no L1fusion
                        y = {"shape": tuple(i.value for i in out.shape)}
                        out = strided_write_compute(out, y, 1, stride_swrite, "strided_write")
                    
                    if invalid_data_rm_flag:
                        out = conv2d_data_rm_compute(out)
                    out.op.attrs["addr_type"] = out_addr_type
                    if out_addr_type:
                        store_to_gm({"shape": ws_fp16_valid_shape if ws_fp16_valid_shape else out.shape, "dtype": "float16"},
                                    None, kernel_name=kernel_name + "_out_gm")
                    if bias_flag:
                        tensor_list = [fm, filter_w, bias_tensor, vdeq, out]
                    else:
                        tensor_list = [fm, filter_w, vdeq, out]
                    
                sch = generic.auto_schedule(out)
            
        config = {
            "print_ir": False,
            "need_build": True,
            "name": kernel_name,
            "tensor_list": tensor_list}
        te.lang.cce.cce_build_code(sch, config)

    def run_testcase(version, config_dict):
        aaa = 0
        for i in config_dict:
            aaa = aaa + 1
            print(f"*************{aaa}****************")
            print("case {}".format(i))
            if i[0] in (1, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14):
                dataflow, shape_in, shape_w, pads, strides, offset_d, bias_flag, relu_flag, quant_scale, quant_offset, stride_swrite, l1fusion_dict = i
            else:
                dataflow, shape_in, shape_w, pads, strides, offset_d, bias_flag, relu_flag, dequant_flag, stride_swrite, l1fusion_dict = i

            kernel_name = "df{}_x_{}_w_{}_p_{}_s_{}_b_{}_r_{}_ss_{}".format(dataflow, shape_in, shape_w, pads, strides, bias_flag, relu_flag, stride_swrite)

            if i[0] in (1, 3, 4, 6, 7, 8, 11, 12, 13, 14):
                kernel_name += "_sc_" + str(quant_scale) + "_of_" + str(quant_offset)
            
            kernel_name += "_dic"

            for key, item  in l1fusion_dict.items():
                if item == []:
                    kernel_name += "_n"
                else:
                    kernel_name += "_" + str(item)

            kernel_name = kernel_name.replace("(", "")
            kernel_name = kernel_name.replace(")", "")
            kernel_name = kernel_name.replace("[", "")
            kernel_name = kernel_name.replace("]", "")
            kernel_name = kernel_name.replace(", ", "_")
            kernel_name = kernel_name.replace("-", "_")

            print("[generate .o and json]", kernel_name)
            if i[0] in (1, 3, 4, 6, 7, 8, 11, 12, 13, 14):
                conv_rmpad_case(version, dataflow, l1fusion_dict,
                                shape_in, shape_w, pads, strides, offset_d,
                                bias_flag, relu_flag, True, stride_swrite,
                                quant_scale=quant_scale, quant_offset=quant_offset, kernel_name=kernel_name, invalid_data_rm_flag=False)
            else:
                conv_rmpad_case(version, dataflow, l1fusion_dict,
                                shape_in, shape_w, pads, strides, offset_d,
                                bias_flag, relu_flag, dequant_flag, stride_swrite, kernel_name=kernel_name, invalid_data_rm_flag=True)

        dst_path = "./res"
        if os.path.exists(dst_path):
            shutil.rmtree(dst_path)
        shutil.move("./kernel_meta", dst_path)

    class Test_conv_rmpad(unittest.TestCase):
        def tearDown(self):
            pass

        def setUp(self):
            pass

        @classmethod
        def tearDownClass(self):
            pass

        @classmethod
        def setUpClass(self):
            pass

        def test_cce_conv_rmpad(self):
            if tc.rmpad_testcase["v200"]["st"]:
                cceconf.te_set_version('Ascend710')
                run_testcase("v200", tc.rmpad_testcase["v200"]["st"])

    from te import platform as cce_conf
    cce_conf.te_set_version("Ascend710")
    run_testcase("v200", tc.rmpad_testcase["v200"]["st"])
    cce_conf.te_set_version("Ascend310")


print("adding Conv2D v200 int4 ut testcases")
ut_case.add_cust_test_func(["Ascend910A"], test_func=test_conv2d_int4)