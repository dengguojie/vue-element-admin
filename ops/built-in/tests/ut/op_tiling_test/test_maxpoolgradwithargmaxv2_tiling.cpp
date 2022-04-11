#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class MaxPoolGradWithArgmaxV2Tiling : public testing::Test {
    protected:
    static void SetUpTestCase() {
        std::cout << "MaxPoolGradWithArgmaxV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "MaxPoolGradWithArgmaxV2Tiling TearDown" << std::endl;
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


// tiling ho nc1h
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_0) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
        "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 2, \"stride_h\": 2, "
        "\"stride_w\": 2, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 4, 112, 112, 16};
    std::vector<int64_t> input1{1, 4, 56, 56, 16};
    std::vector<int64_t> input2{1, 4, 4, 197, 16};
	std::vector<int64_t> output0{1, 4, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
	TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_0";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 4 1 18 384 0 0 0 0 0 5 0 10 4 56 56 112 112 0 4 1 4 7 8 0 32 1 1 7 14 3152 1 ");
}


// tiling ho
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_1) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 3, \"kw\": 3, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 34, 112, 112, 16};
    std::vector<int64_t> input1{1, 34, 56, 56, 16};
    std::vector<int64_t> input2{1, 34, 9, 197, 16};
    std::vector<int64_t> output0{1, 34, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_1";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "3 2 2 0 0 1 0 1 0 0 0 0 0 34 56 56 112 112 0 0 0 0 0 0 0 17 0 0 0 0 0 1 ");
}

// not tiling nc1h
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_2) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 2, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 4, 24, 24, 16};
    std::vector<int64_t> input1{1, 4, 12, 12, 16};
    std::vector<int64_t> input2{1, 4, 4, 10, 16};
    std::vector<int64_t> output0{1, 4, 24, 24, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "NC1HWC0";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    tensor_input1.format = "NC1HWC0";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    tensor_input2.format = "NC1HWC0";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";
    tensor_output0.format = "NC1HWC0";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_2";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 4 1 2 128 0 0 0 0 0 12 0 47 4 12 12 24 24 0 4 1 4 2 6 0 24 1 1 2 4 160 1 ");
}

// not tiling
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_3) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 2, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 32, 24, 24, 16};
    std::vector<int64_t> input1{1, 32, 12, 12, 16};
    std::vector<int64_t> input2{1, 32, 4, 10, 16};
    std::vector<int64_t> output0{1, 32, 24, 24, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_3";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "0 32 1 9 256 0 0 0 0 0 12 0 47 32 12 12 24 24 0 32 0 0 0 0 0 32 0 0 0 0 160 1 ");
}


// tiling ho wo
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_4) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 2, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 1, 624, 624, 16};
    std::vector<int64_t> input1{1, 1, 312, 312, 16};
    std::vector<int64_t> input2{1, 1, 4, 6085, 16};
    std::vector<int64_t> output0{1, 1, 624, 624, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_4";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 1 18 384 0 0 0 0 283 1 0 2 1 312 312 624 624 0 1 1 1 10 32 0 32 1 1 10 20 97360 1 ");
}

// tiling ho wo , kernel expand
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_5) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 31, \"kw\": 31, \"stride_h\": 31, "
     "\"stride_w\": 31, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 1, 124, 124, 16};
    std::vector<int64_t> input1{1, 1, 4, 4, 16};
    std::vector<int64_t> input2{1, 1, 4, 2, 16};
    std::vector<int64_t> output0{1, 1, 124, 124, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_5";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "2 1 1 1 128 0 0 0 0 1 1 0 31 1 4 4 124 124 1 1 1 1 1 4 1 1 0 0 0 0 32 1 ");
}

// pad_h != pad_w case
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_6) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 5, \"kw\": 3, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 2, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 34, 112, 112, 16};
    std::vector<int64_t> input1{1, 34, 56, 56, 16};
    std::vector<int64_t> input2{1, 34, 15, 197, 16};
    std::vector<int64_t> output0{1, 34, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_6";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 2 11 256 1 0 2 1 0 3 0 10 34 56 56 112 112 0 34 0 0 0 0 0 17 0 0 0 0 3152 1 ");
}

// wrong output shape case
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_7) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 3, \"kw\": 3, \"stride_h\": 2, "
     "\"stride_w\": 2, \"pad_h\": 1, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{1, 34, 112, 112, 16};
    std::vector<int64_t> input1{1, 34, 55, 55, 16};
    std::vector<int64_t> input2{1, 34, 9, 197, 16};
    std::vector<int64_t> output0{1, 34, 112, 112, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_7";
    OpRunInfo runInfo;
    iter->second.tiling_func_(opParas, op_compile_info, runInfo);
}

// case of overlap is too large to support
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_8) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 2, \"stride_h\": 1, "
     "\"stride_w\": 61, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 1}}";

    std::vector<int64_t> input0{32, 64, 400, 672, 16};
    std::vector<int64_t> input1{32, 64, 399, 12, 16};
    std::vector<int64_t> input2{32, 64, 4, 301, 16};
    std::vector<int64_t> output0{32, 64, 400, 672, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_8";
    OpRunInfo runInfo;
    iter->second.tiling_func_(opParas, op_compile_info, runInfo);
}

// overlap too large to support
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_9) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 255, \"kw\": 2, \"stride_h\": 61, "
     "\"stride_w\": 1, \"pad_h\": 7, \"pad_w\": 1,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 1}}";

    std::vector<int64_t> input0{3, 4, 320, 320, 16};
    std::vector<int64_t> input1{3, 4, 3, 321, 16};
    std::vector<int64_t> input2{3, 4, 510, 62, 16};
    std::vector<int64_t> output0{3, 4, 320, 320, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";


    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_9";
    OpRunInfo runInfo;
    iter->second.tiling_func_(opParas, op_compile_info, runInfo);
}

// kernel size or stride too large
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_10) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 2, \"kw\": 255, \"stride_h\": 61, "
     "\"stride_w\": 1, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 0}}";

    std::vector<int64_t> input0{3, 4, 320, 320, 16};
    std::vector<int64_t> input1{3, 4, 6, 66, 16};
    std::vector<int64_t> input2{3, 4, 510, 26, 16};
    std::vector<int64_t> output0{3, 4, 320, 320, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_10";
    OpRunInfo runInfo;
    iter->second.tiling_func_(opParas, op_compile_info, runInfo);
}

// case of fix overlap arrange
TEST_F(MaxPoolGradWithArgmaxV2Tiling, maxpoolgrad_with_argmax_v2_tiling_11) {
    using namespace optiling;
    std::string op_name = "MaxPoolGradWithArgmaxV2";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo =
     "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"kh\": 200, \"kw\": 2, \"stride_h\": 1, "
     "\"stride_w\": 61, \"pad_h\": 0, \"pad_w\": 0,\"dilation_h\": 1, \"dilation_w\": 1, \"ceil_mode\": 1}}";

    std::vector<int64_t> input0{250, 64, 400, 672, 16};
    std::vector<int64_t> input1{250, 64, 201, 12, 16};
    std::vector<int64_t> input2{250, 64, 400, 301, 16};
    std::vector<int64_t> output0{250, 64, 400, 672, 16};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_input2;
    tensor_input2.shape = input2;
    tensor_input2.dtype = "uint16";
    TeOpTensor tensor_output0;
    tensor_output0.shape = output0;
    tensor_output0.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg2;
    tensor_input_arg2.tensor.push_back(tensor_input2);
    tensor_input_arg2.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg0;
    tensor_output_arg0.tensor.push_back(tensor_output0);
    tensor_output_arg0.arg_type = TensorArgType::TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.inputs.push_back(tensor_input_arg2);
    opParas.outputs.push_back(tensor_output_arg0);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "maxpoolgrad_with_argmax_v2_tiling_11";
    OpRunInfo runInfo;
    iter->second.tiling_func_(opParas, op_compile_info, runInfo);
}
