#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"

using namespace std;

class NMSWithMaskTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "NMSWithMaskTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "NMSWithMaskTiling TearDown" << std::endl;
  }
};

static string to_string(const std::stringstream &tiling_data) {
  auto data = tiling_data.str();
  string result;
  int32_t tmp = 0;
  memcpy(&tmp, data.c_str(), sizeof(tmp));
  result = std::to_string(tmp);

  return result;
}

TEST_F(NMSWithMaskTiling, nms_with_mask_tiling_test) {
  using namespace optiling;
  std::string op_name = "NMSWithMask";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("NMSWithMask");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  
  std::string compileInfo = "{\"vars\": {\"max_boxes_num\":2960}}";

  std::vector<int64_t> input_shape {16, 8};
  std::string input_dtype = "float16";
  std::vector<std::vector<int64_t>> output_shape {{16, 5}, {16, }, {16, }};
  std::vector<std::string> output_dtype {"float16", "int32", "uint8"};

  TeOpParas opParas;

  TeOpTensor tensor_input;
  tensor_input.shape = input_shape;
  tensor_input.dtype = input_dtype;
  TeOpTensorArg tensor_input_arg;
  tensor_input_arg.tensor.push_back(tensor_input);
  tensor_input_arg.arg_type = TensorArgType::TA_SINGLE;
  opParas.inputs.push_back(tensor_input_arg);

  for (size_t i = 0; i < output_shape.size(); i++) {
    TeOpTensor tensor_output;
    tensor_output.shape = output_shape[i];
    tensor_output.dtype = output_dtype[i];
    TeOpTensorArg tensor_output_arg;
    tensor_output_arg.tensor.push_back(tensor_output);
    tensor_output_arg.arg_type = TensorArgType::TA_SINGLE;
    opParas.outputs.push_back(tensor_output_arg);
  }

  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "nms_with_mask_dynamic_tiling";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(runInfo.block_dim, 1);
  EXPECT_EQ(to_string(runInfo.tiling_data), "16");
}

