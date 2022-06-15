#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class YoloxBoundingBoxDecodeTiling : public testing::Test {
protected:
    static void SetUpTestCase() {
        std::cout << "YoloxBoundingBoxDecodeTiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
        std::cout << "YoloxBoundingBoxDecodeTiling TearDown" << std::endl;
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

TEST_F(YoloxBoundingBoxDecodeTiling, yolox_bounding_box_decode_tiling_0) {
    using namespace optiling;
    std::string op_name = "YoloxBoundingBoxDecode";
    auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

    std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"bboxes_data_each_block\": 16}}";
    std::vector<int64_t> input0{8400,4};
    std::vector<int64_t> input1{1,8400,4};
    std::vector<int64_t> output{1,8400,4};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "float16";
    tensor_input0.format = "ND";
    tensor_input0.ori_format = "ND";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    tensor_input1.format = "ND";
    tensor_input1.ori_format = "ND";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float16";
    tensor_output.format = "ND";
    tensor_output.ori_format = "ND";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TensorArgType::TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;
  
    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "yoloxboundingboxdecode_8400_4";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 8400 ");
}