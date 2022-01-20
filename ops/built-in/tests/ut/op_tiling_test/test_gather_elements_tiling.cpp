#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "common/utils/ut_op_util.h"
#include "selection_ops.h"
#include "array_ops.h"

using namespace std;
using namespace optiling;
using namespace ge;
using namespace ut_util;

class GatherElementsTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "GatherElementsTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "GatherElementsTiling TearDown" << std::endl;
  }
};

/*
 * be careful of the to_string fuction
 * the type of tiling_data in other ops is int64 while int32 here
 */
static string to_string(const std::stringstream& tiling_data) {
  auto data = tiling_data.str();
  string result;
  int64_t tmp = 0;
  for (size_t i = 0; i < data.length(); i += sizeof(int64_t)) {
    memcpy(&tmp, data.c_str() + i, sizeof(tmp));
    result += std::to_string(tmp);
    result += " ";
  }

  return result;
}

/*
.INPUT(x, TensorType::BasicType())
    .INPUT(indices, TensorType::IndexNumberType())
    .OUTPUT(y, TensorType::BasicType())
*/

TEST_F(GatherElementsTiling, gather_elements_tiling_0) {
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"indices_dsize\":8, "
      "\"params_dsize\":4, \"axis\":0}}";

  std::vector<int64_t> inputA{25600};
  std::vector<int64_t> inputB{10};
  std::vector<int64_t> output{10};

  TeOpTensor tensor_input;
  tensor_input.shape = inputA;
  tensor_input.ori_shape = inputA;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_indices;
  tensor_indices.shape = inputB;
  tensor_indices.ori_shape = inputB;
  tensor_indices.dtype = "int64";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argInput;
  tensor_argInput.tensor.push_back(tensor_input);
  tensor_argInput.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argIndices;
  tensor_argIndices.tensor.push_back(tensor_indices);
  tensor_argIndices.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argOutput;
  tensor_argOutput.tensor.push_back(tensor_output);
  tensor_argOutput.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argInput);
  opParas.inputs.push_back(tensor_argIndices);
  opParas.outputs.push_back(tensor_argOutput);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "1 1 25600 1 10 1 10 0 0 0 10 25600 0 0 2 10 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_1) {
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"indices_dsize\":4, "
      "\"params_dsize\":4, \"axis\":0}}";

  std::vector<int64_t> inputA{3, 3, 3};
  std::vector<int64_t> inputB{3, 3, 3};
  std::vector<int64_t> output{3, 3, 3};

  TeOpTensor tensor_input;
  tensor_input.shape = inputA;
  tensor_input.ori_shape = inputA;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_indices;
  tensor_indices.shape = inputB;
  tensor_indices.ori_shape = inputB;
  tensor_indices.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argInput;
  tensor_argInput.tensor.push_back(tensor_input);
  tensor_argInput.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argIndices;
  tensor_argIndices.tensor.push_back(tensor_indices);
  tensor_argIndices.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argOutput;
  tensor_argOutput.tensor.push_back(tensor_output);
  tensor_argOutput.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argInput);
  opParas.inputs.push_back(tensor_argIndices);
  opParas.outputs.push_back(tensor_argOutput);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "2";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "2 1 3 9 27 1 27 0 0 0 27 27 0 0 1 3 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_2) {
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"indices_dsize\":4, "
      "\"params_dsize\":4, \"axis\":0}}";

  std::vector<int64_t> inputA{1024, 1024};
  std::vector<int64_t> inputB{1024, 1024};
  std::vector<int64_t> output{1024, 1024};

  TeOpTensor tensor_input;
  tensor_input.shape = inputA;
  tensor_input.ori_shape = inputA;
  tensor_input.dtype = "int32";
  TeOpTensor tensor_indices;
  tensor_indices.shape = inputB;
  tensor_indices.ori_shape = inputB;
  tensor_indices.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "int32";

  TeOpTensorArg tensor_argInput;
  tensor_argInput.tensor.push_back(tensor_input);
  tensor_argInput.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argIndices;
  tensor_argIndices.tensor.push_back(tensor_indices);
  tensor_argIndices.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argOutput;
  tensor_argOutput.tensor.push_back(tensor_output);
  tensor_argOutput.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argInput);
  opParas.inputs.push_back(tensor_argIndices);
  opParas.outputs.push_back(tensor_argOutput);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "3";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "3 1 1024 1024 1048576 32 32768 0 1 32512 256 1048576 0 0 1 1024 ");
}

TEST_F(GatherElementsTiling, gather_elements_tiling_3) {
  std::string op_name = "GatherElements";
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("GatherElements");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  std::string compileInfo =
      "{\"vars\": {\"ub_size\": 262144, \"core_num\": 32, \"indices_dsize\":4, "
      "\"params_dsize\":4, \"axis\":0}}";

  std::vector<int64_t> inputA{17, 17, 17};
  std::vector<int64_t> inputB{17, 17, 17};
  std::vector<int64_t> output{17, 17, 17};

  TeOpTensor tensor_input;
  tensor_input.shape = inputA;
  tensor_input.ori_shape = inputA;
  tensor_input.dtype = "float32";
  TeOpTensor tensor_indices;
  tensor_indices.shape = inputB;
  tensor_indices.ori_shape = inputB;
  tensor_indices.dtype = "int32";
  TeOpTensor tensor_output;
  tensor_output.shape = output;
  tensor_output.dtype = "float32";

  TeOpTensorArg tensor_argInput;
  tensor_argInput.tensor.push_back(tensor_input);
  tensor_argInput.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argIndices;
  tensor_argIndices.tensor.push_back(tensor_indices);
  tensor_argIndices.arg_type = TensorArgType::TA_SINGLE;
  TeOpTensorArg tensor_argOutput;
  tensor_argOutput.tensor.push_back(tensor_output);
  tensor_argOutput.arg_type = TensorArgType::TA_SINGLE;

  TeOpParas opParas;

  opParas.inputs.push_back(tensor_argInput);
  opParas.inputs.push_back(tensor_argIndices);
  opParas.outputs.push_back(tensor_argOutput);
  opParas.op_type = op_name;
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "4";
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.tiling_data),
            "4 1 17 289 4913 32 152 49 0 32512 152 4913 1 6 1 17 ");
}