#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolWithArgmaxV2Tiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "MaxPoolWithArgmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPoolWithArgmaxV2Tiling TearDown" << std::endl;
    }
};

static string to_string(const std::stringstream &tiling_data) {
    auto data = tiling_data.str();
    string result;
    int32_t tmp = 0;
    for (size_t i = 0; i < data.length(); i += sizeof(int32_t)) {
        memcpy(&tmp, data.c_str() + i, sizeof(tmp));
        result += std::to_string(tmp);
        result += " ";
    }
    return result;
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_0) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 1, \"ksize_w\": 1, \"strides_h\": 1, "
        "\"strides_w\": 1, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 35, 35, 16};
    std::vector<int64_t> output0{4, 5, 35, 35, 16};
    std::vector<int64_t> output1{4, 5, 1, 78, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_0";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 766 754 35 35 35 35 35 35 0 0 0 0 1 1 1 0 766 0 754 20 16 768 48 1248 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_1) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 28, 28, 16};
    std::vector<int64_t> output0{4, 5, 13, 13, 16};
    std::vector<int64_t> output1{4, 5, 9, 12, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_1";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 20 1 1 28 28 13 13 27 27 0 0 0 0 1 1 1 1 0 1 0 20 16 0 16 192 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_2) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 2, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 100, 100, 16};
    std::vector<int64_t> output0{4, 5, 50, 50, 16};
    std::vector<int64_t> output1{4, 5, 4, 158, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_2";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "4 20 1 1 100 100 50 50 100 100 0 0 0 0 1 5 1 10 0 10 0 20 16 0 64 2528 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_3) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 2, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{4, 5, 100, 1000, 16};
    std::vector<int64_t> output0{4, 5, 50, 500, 16};
    std::vector<int64_t> output1{4, 5, 4, 1564, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_3";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "3 20 1 1 100 1000 50 500 100 1000 0 0 0 0 1 1 330 1 170 1 170 20 336 176 512 25024 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_4) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 255, \"strides_h\": 61, "
        "\"strides_w\": 1, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 0, \"pad_bottom\": 0, \"pad_left\": 0, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{3, 4, 320, 320, 16};
    std::vector<int64_t> output0{3, 4, 6, 66, 16};
    std::vector<int64_t> output1{3, 4, 510, 26, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_4";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "3 12 1 1 320 320 6 66 307 320 0 0 0 0 1 1 32 2 2 2 2 12 32 16 80 416 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_5) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 3, \"ksize_w\": 3, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 0, \"pad_left\": 1, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{32, 4, 112, 112, 16};
    std::vector<int64_t> output0{32, 4, 56, 56, 16};
    std::vector<int64_t> output1{32, 4, 9, 197, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_5";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "6 32 4 4 112 112 56 56 113 113 1 0 1 0 1 1 1 0 0 0 0 128 16 0 64 3152 ");
}

TEST_F(MaxPoolWithArgmaxV2Tiling, maxpool_with_argmax_v2_tiling_6) {
    using namespace optiling;
    std::string op_name = "MaxPoolWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"ksize_h\": 2, \"ksize_w\": 2, \"strides_h\": 2, "
        "\"strides_w\": 2, \"padding\": 0, \"ceil_mode\": 0, \"pad_top\": 1, \"pad_bottom\": 0, \"pad_left\": 1, \"pad_right\": 0, \"global\": 0}}";

    std::vector<int64_t> input0{8, 3, 640, 640, 16};
    std::vector<int64_t> output0{8, 3, 321, 321, 16};
    std::vector<int64_t> output1{8, 3, 4, 6442, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";
    TeOpTensor tensor_output1;
    tensor_output1.shape = output1;
    tensor_output1.dtype = "uint16";
    tensor_output1.format = "NC1HWC0";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg1;
    tensor_output_arg1.tensor.push_back(tensor_output1);
    tensor_output_arg1.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.outputs.push_back(tensor_output_arg1);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpool_with_argmax_v2_tiling_6";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 24 1 1 640 640 321 321 642 642 1 1 1 1 1 1 1 321 0 321 0 24 16 0 336 103072 ");
}