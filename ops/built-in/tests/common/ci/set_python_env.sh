#!/bin/bash

#current dir
cur_dir=$1
test_type=$2
product=$3


if [ "Xpy_ut" = "X$test_type" ];then
    test_type="ut"
    # fix set ddk version issue
elif [ "Xpy_st" = "X$test_type" ];then
    test_type="st"
fi

# mini
TOOLCHAIN_HOME="bin/toolchain/x86/ubuntu/ccec_libs/ccec_x86_ubuntu_16_04_adk"

mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend920
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend920/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in

cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910A_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910ProA_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910B_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910ProB_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910PremiumA_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910A_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910ProA_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910B_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910ProB_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910PremiumA_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910A_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910A_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910B_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910B_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910ProA_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910ProA_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910ProB_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910ProB_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910PremiumA_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910PremiumA_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910A_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910ProA_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910B_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_L1_FUSION_cost_model_Ascend910PremiumA_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_L1_FUSION_cost_model_Ascend910ProB_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910A_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910ProA_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910B_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_32_NO_L1_FUSION_cost_model_Ascend910PremiumA_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/AIC_30_NO_L1_FUSION_cost_model_Ascend910ProB_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910B_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910A_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProA_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910ProB_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend910/built-in/cost_model_ascend910PremiumA_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_L1_FUSION_cost_model_Ascend710_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend710_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_L1_FUSION_cost_model_Ascend710Pro_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend710Pro_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_L1_FUSION_cost_model_Ascend710_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend710_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_L1_FUSION_cost_model_Ascend710Pro_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend710Pro_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_L1_FUSION_cost_model_Ascend710_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend710_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_L1_FUSION_cost_model_Ascend710Pro_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend710Pro_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710Pro_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710Pro_conv3d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710Pro_conv3d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710Pro_conv3d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_Ascend710_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_ascend710_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend710/built-in/cost_model_ascend710Pro_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_NO_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_NO_L1_FUSION_cost_model_Ascend610_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_L1_FUSION_cost_model_Ascend610_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend610_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend610_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/cost_model_Ascend610_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/cost_model_Ascend610_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_10_NO_L1_FUSION_cost_model_Ascend610_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_8_NO_L1_FUSION_cost_model_Ascend610_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_4_NO_L1_FUSION_cost_model_Ascend610_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend610_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend610/built-in/AIC_1_NO_L1_FUSION_cost_model_Ascend610_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_L1_FUSION_cost_model_Ascend310_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend310_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_L1_FUSION_cost_model_Ascend310_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend310_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_L1_FUSION_cost_model_Ascend310_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/AIC_2_NO_L1_FUSION_cost_model_Ascend310_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend310/built-in/cost_model_ascend310_matmul.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300ES_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300ES_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300ES_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300ES_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300ES_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300ES_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/cost_model_Hi3796CV300ES_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/cost_model_Hi3796CV300ES_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300ES/built-in/cost_model_Hi3796CV300ES_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300CS_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300CS_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300CS_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300CS_conv2d_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_L1_FUSION_cost_model_Hi3796CV300CS_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/AIC_1_NO_L1_FUSION_cost_model_Hi3796CV300CS_depthwise_conv2d_forward.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/cost_model_Hi3796CV300CS_conv2d_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/cost_model_Hi3796CV300CS_depthwise_bp_input.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/Hi3796CV300CS/built-in/cost_model_Hi3796CV300CS_depthwise_bp_filter.bin
cp $cur_dir/llt/tensor_engine/ut/testcase/auto_tiling/cost_model.bin $cur_dir/out/${product}/llt/$test_type/obj/data/tiling/ascend920/built-in/cost_model_ascend920A_matmul.bin
cp $cur_dir/tensor_engine/tiling_config $cur_dir/out/${product}/llt/$test_type/obj/
mkdir -p $cur_dir/out/${product}/llt/$test_type/obj/lib/simulator/common/data
cp $cur_dir/bin/cmodel/data/* $cur_dir/out/${product}/llt/$test_type/obj/lib/simulator/common/data/.
#PYTHONPATH
export PYTHONPATH=$cur_dir/tensor_engine/python:$cur_dir/tensor_engine/topi/python:$cur_dir/asl/ops/cann/ops/built-in/tbe:$cur_dir/toolchain/tensor_utils/op_test_frame/python:$PYTHONPATH

#PATH
export PATH=$cur_dir/${TOOLCHAIN_HOME}/bin:$PATH

#LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$cur_dir/out/${product}/llt/$test_type/obj/lib:$LD_LIBRARY_PATH

#TVM_AICPU_INCLUDE_PATH and TVM_AICPU_LIBRARY_PATH
export TVM_AICPU_INCLUDE_PATH=$cur_dir/inc/tensor_engine:$cur_dir/inc/cce:$cur_dir/cce/inc
export TVM_AICPU_LIBRARY_PATH=$cur_dir/out/${product}/llt/lib:$cur_dir/${TOOLCHAIN_HOME}/aicpu_lib:$TVM_AICPU_LIBRARY_PATH

export TVM_AICPU_OS_SYSROOT=$cur_dir/prebuilts/hcc/linux-x86/aarch64/aarch64-linux-gnu/sysroot:$cur_dir/build/prebuilts/hcc/linux-x86/aarch64/aarch64-linux-gnu/sysroot
#LD_PRELOAD
export LD_PRELOAD=$LD_PRELOAD:$cur_dir/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/lib64/libasan.so:$cur_dir/build/prebuilts/gcc/linux-x86/x86/x86_64-unknown-linux-gnu-4.9.3/lib64/libasan.so
# for tiling cost_model and repository
export ASCEND_OPP_PATH=$cur_dir/out/${product}/llt/$test_type/obj
unset LD_PRELOAD
env
