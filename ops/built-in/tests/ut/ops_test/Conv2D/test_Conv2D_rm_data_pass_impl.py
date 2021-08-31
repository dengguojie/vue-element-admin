#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from op_test_frame.ut import OpUT

ut_case = OpUT("Conv2D", "impl.conv2d", "conv2d")

def test_conv_rm_data_pass(test_arg):

    import impl
    import json
    import te
    import copy
    from tbe.common.register.fusion_pass.conv2d_data_rm_fusion_pass import conv2d_data_rm_build_pass

    op_list = []
    op_new = {}
    op_new["name"] = "-1_0_res3d_branch2a_quant_layer_fuse0__0"
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_quant_layer_fuse0__0"}]
    op_new["type"] = "Data"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["name"] = "-1_0_new_sub_graph1_PlaceHolder63__0"
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder63__0"}]
    op_new["type"] = "Data"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["name"] = "-1_0_new_sub_graph1_PlaceHolder64__0"
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder64__0"}]
    op_new["type"] = "Data"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["name"] = "-1_0_res3d_branch2a_fuse0OPT"
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_fuse0OPT"}]
    op_new["type"] = "Data"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["name"] = "-1_0_new_sub_graph1_PlaceHolder65__0"
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder65__0"}]
    op_new["type"] = "Data"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["func_name"] = "conv2d"
    op_new["id"] = 142
    op_new["input_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_quant_layer_fuse0__0"},
                            {"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder63__0"},
                            {"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder64__0"},
                            {"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_fuse0OPT"}]
    op_new["name"] = "-1_0_res3d_branch2a_fuse0"
    op_new["ori_name"] = ["res3d_branch2a_fuse0"]
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_fuse0__0", "output_index":0}]
    op_new["prebuild_outs_attrs"] = {"kwds_args":{},"list_args":[]}
    op_new["pattern"] = "Convolution"
    op_new["type"] = "Conv2D"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["func_name"] = "ascend_dequant"
    op_new["id"] = 144
    op_new["input_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_fuse0__0"},
                            {"L1_fusion_type":-1, "name":"-1_0_new_sub_graph1_PlaceHolder65__0"}]
    op_new["name"] = "-1_0_res3d_branch2a_dequant_layer_fuse0"
    op_new["ori_name"] = ["res3d_branch2a_dequant_layer_fuse0"]
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_dequant_layer_fuse0__0", "output_index":0},
                            {"L1_fusion_type":-1, "name":"-1_0_xxxxxxxxx__0", "output_index":1}]
    op_new["prebuild_outs_attrs"] = {"kwds_args":{},"list_args":[]}
    op_new["pattern"] = "dequant"
    op_new["type"] = "AscendDequant"
    op_list.append(copy.deepcopy(op_new))
    op_new = {}
    op_new["func_name"] = "ascend_quant"
    op_new["id"] = 145
    op_new["input_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2a_dequant_layer_fuse0__0"}]
    op_new["name"] = "-1_0_res3d_branch2b_quant_layer_fuse0"
    op_new["ori_name"] = ["res3d_branch2b_quant_layer_fuse0"]
    op_new["output_desc"] = [{"L1_fusion_type":-1, "name":"-1_0_res3d_branch2b_quant_layer_fuse0__0", "output_index":0}]
    op_new["prebuild_outs_attrs"] = {"kwds_args":{},"list_args":[]}
    op_new["pattern"] = "quant"
    op_new["type"] = "AscendQuant"
    op_list.append(copy.deepcopy(op_new))

    conv2d_data_rm_build_pass(op_list)

print("adding Tefusion Conv2D rmpad ut case")
ut_case.add_cust_test_func(test_func=test_conv_rm_data_pass)

if __name__ == '__main__':
    ut_case.add_cust_test_func(test_func=test_conv_rm_data_pass)
    exit(0)
