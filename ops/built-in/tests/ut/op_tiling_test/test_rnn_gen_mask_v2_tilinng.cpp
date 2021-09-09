#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class RnnGenMaskV2Tiling : public testing::Test {
protected:
    static void SetUpTestCase() {
      std::cout << "RnnGenMaskV2Tiling SetUp" << std::endl;
    }

    static void TearDownTestCase() {
      std::cout << "RnnGenMaskV2Tiling TearDown" << std::endl;
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

TEST_F(RnnGenMaskV2Tiling, rnn_gen_mask_v2_tiling_0) {
    using namespace optiling;
    std::string op_name = "RnnGenMaskV2";
    auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
    ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

    // std::string compileInfo = "{\"vars\": {\"cal_mode\": 1,\"available_aicore_num\": 32, \"core_used\": 9, \"block\": 16, \"core_num\": 32, \"batch_size\": 3, \"num_step\": 3, \"hidden_size\": 3, \"hidden_size_block\": 8, \"repeat\": 1, \"rounds\": 9, \"batch_num_per_aicore\": 1, \"batch_tail\": 0}}";
    std::string compileInfo = "{\"vars\": {\"cal_mode\": 1,\"available_aicore_num\": 32, \"core_used\": 9, \"block\": 16, \"core_num\": 32, \"batch_size\": 3, \"num_step\": 3, \"rounds\": 9, \"batch_num_per_aicore\": 1, \"batch_tail\": 0}}";

    std::vector<int64_t> input0{32};
    std::vector<int64_t> input1{2, 32, 64};
    std::vector<int64_t> output{2, 32, 64};

    TeOpTensor tensor_input0;
    tensor_input0.shape = input0;
    tensor_input0.dtype = "int32";
    TeOpTensor tensor_input1;
    tensor_input1.shape = input1;
    tensor_input1.dtype = "float16";
    TeOpTensor tensor_output;
    tensor_output.shape = output;
    tensor_output.dtype = "float16";

    TeOpTensorArg tensor_input_arg0;
    tensor_input_arg0.tensor.push_back(tensor_input0);
    tensor_input_arg0.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_input_arg1;
    tensor_input_arg1.tensor.push_back(tensor_input1);
    tensor_input_arg1.arg_type = TA_SINGLE;
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TA_SINGLE;

    TeOpParas opParas;
    opParas.inputs.push_back(tensor_input_arg0);
    opParas.inputs.push_back(tensor_input_arg1);
    opParas.outputs.push_back(tensor_output_arg);
    opParas.op_type = op_name;
    OpCompileInfo op_compile_info;
    op_compile_info.str = compileInfo;
    op_compile_info.key = "12345671";
    OpRunInfo runInfo;
    ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
    EXPECT_EQ(to_string(runInfo.tiling_data), "1 32 32 2 64 2 0 ");

}