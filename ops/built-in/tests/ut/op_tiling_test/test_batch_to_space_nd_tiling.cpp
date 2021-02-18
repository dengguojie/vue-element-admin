#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"

using namespace std;

class BatchToSpaceNDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "BatchToSpaceNDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "BatchToSpaceNDTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream& tiling_data) {
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

TEST_F(BatchToSpaceNDTiling, batchtospacend_tiling_0) {
  using namespace optiling;
  std::string op_name = "BatchToSpaceND";
  auto iter = optiling::OpTilingRegistryInterf::RegisteredOpInterf().find(op_name);
  ASSERT_TRUE(iter != optiling::OpTilingRegistryInterf::RegisteredOpInterf().end());

  std::string compileInfo = "{\"vars\": {\"ub_ele\": 126976, \"core_num\": 32, \"block_size\": 2}}";

  std::vector<int64_t> input{16, 2, 2, 2, 16};
  std::vector<int64_t> input_crops{2, 2};
  std::vector<int32_t> crops{1, 1, 1, 1};
  std::vector<int64_t> output{4, 2, 2, 2, 16};

  TeOpTensor tensor_input;
  tensor_input.shape = input;
  tensor_input.dtype = "float16";
  tensor_input.format = "NC1HWC0";
  tensor_input.ori_format = "NHWC";
  TeOpTensor tensor_input_crops;
  tensor_input_crops.shape = input_crops;
  tensor_input_crops.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float16";
  tensor_output.format = "NC1HWC0";
  tensor_output.ori_format = "NHWC";

  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_input_crops_arg;
  tensor_input_crops_arg.tensor.push_back(tensor_input_crops);
  tensor_input_crops_arg.arg_type = TA_SINGLE;
  TeOpTensorArg tensor_output_arg;
  tensor_output_arg.tensor.push_back(tensor_output);
  tensor_output_arg.arg_type = TA_SINGLE;

  TeOpParas opParas;
  opParas.const_inputs["crops"] =
      std::tuple<const uint8_t*, size_t, ge::Tensor>((const uint8_t*)crops.data(), crops.size() * 4, ge::Tensor());
  opParas.inputs.push_back(tensor_input_arg);
  opParas.inputs.push_back(tensor_input_crops_arg);
  opParas.outputs.push_back(tensor_output_arg);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 1 1 16 0 2 2 0 0 1 1 1 1 0 2 2 2 16 4 0 2 2 ");
}