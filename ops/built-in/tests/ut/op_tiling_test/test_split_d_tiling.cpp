#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"

using namespace std;

class SplitDTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitDTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitDTiling TearDown" << std::endl;
  }
};

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

using namespace ge;
#include "test_common.h"
/*
REG_OP(SplitD)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(SplitD)
*/

TEST_F(SplitDTiling, SplitD_tiling_0) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {1, 8};
  vector<vector<int64_t>> output_shapes = {
      {1, 8},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[0];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":1, \"num_split\":1}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234560";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "0 8 8 1 0 0 8 0 0 8 0 8 253952 8 0 0 0 0 ");
}

TEST_F(SplitDTiling, SplitD_tiling_1) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {1023, 23, 23, 9};
  vector<vector<int64_t>> output_shapes = {
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[0];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);
  
  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":3, \"num_split\":3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234561";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "2 9 3 32 16912 16895 152208 152055 0 152208 0 152055 190458 4870503 16912 16912 16895 16895 ");
}

TEST_F(SplitDTiling, SplitD_tiling_2) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {1023, 23, 23, 9};
  vector<vector<int64_t>> output_shapes = {
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[1];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[1];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":3, \"num_split\": 0}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234562";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_FALSE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
}

TEST_F(SplitDTiling, SplitD_tiling_3) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {1023, 23, 23, 9};
  vector<vector<int64_t>> output_shapes = {
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
      {1023, 23, 23, 3},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[1];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[1];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 0, \"ub_size\":253952, \"split_dim\":3, \"num_split\": 3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234563";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
}
TEST_F(SplitDTiling, SplitD_tiling_4) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {21, 21, 6, 9};
  vector<vector<int64_t>> output_shapes = {
      {21, 7, 6, 9},
      {21, 7, 6, 9},
      {21, 7, 6, 9},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[1];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[1];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":1, \"num_split\": 3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234564";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "4 1134 378 31 21 0 12 18 0 12 0 18 253952 23814 0 0 0 0 ");
}
TEST_F(SplitDTiling, SplitD_tiling_5) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {21, 21, 9, 9};
  vector<vector<int64_t>> output_shapes = {
      {21, 7, 9, 9},
      {21, 7, 9, 9},
      {21, 7, 9, 9},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[0];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":1, \"num_split\": 3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234565";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 1701 567 1 21 0 18 9 0 0 0 0 253952 35721 0 0 0 0 ");
}
TEST_F(SplitDTiling, SplitD_tiling_6) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {21, 21, 6, 18};
  vector<vector<int64_t>> output_shapes = {
      {21, 21, 2, 18},
      {21, 21, 2, 18},
      {21, 21, 2, 18},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[1];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[1];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":2, \"num_split\": 3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234566";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "1 108 36 32 14 7 0 0 0 36 0 0 253952 47628 0 0 0 0 ");
}
TEST_F(SplitDTiling, SplitD_tiling_7) {
  using namespace optiling;
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("SplitD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());

  TeOpTensorArg tensorInputs, tensorOutputs;
  TeOpParas opParas;

  std::vector<int64_t> input_shapes = {7, 21, 6, 6};
  vector<vector<int64_t>> output_shapes = {
      {7, 21, 2, 6},
      {7, 21, 2, 6},
      {7, 21, 2, 6},
  };
  vector<std::string> dtypes = {"int8", "int32"};

  tensorInputs.tensor.clear();
  TeOpTensor tensorInput;
  tensorInput.shape = input_shapes;
  tensorInput.dtype = dtypes[0];
  tensorInputs.tensor.push_back(tensorInput);
  tensorInputs.arg_type = TA_SINGLE;
  opParas.inputs.push_back(tensorInputs);

  for (size_t i = 0; i < output_shapes.size(); i++) {
    tensorOutputs.tensor.clear();
    TeOpTensor tensorOutput;
    tensorOutput.shape = output_shapes[i];
    tensorOutput.dtype = dtypes[0];
    tensorOutputs.tensor.push_back(tensorOutput);
    tensorOutputs.arg_type = TA_SINGLE;
    opParas.outputs.push_back(tensorOutputs);
  }
  opParas.op_type = "SplitD";

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_size\":253952, \"split_dim\":2, \"num_split\": 3}}";
  OpCompileInfo op_compile_info;
  op_compile_info.str = compileInfo;
  op_compile_info.key = "1234567";

  // do tilling, get runInfo
  OpRunInfo runInfo;
  ASSERT_TRUE(iter->second.tiling_func_(opParas, op_compile_info, runInfo));
  std::cout << "SplitD tiling_data:" << to_string(runInfo.tiling_data) << std::endl;
  EXPECT_EQ(to_string(runInfo.tiling_data), "3 36 12 1 147 2 0 0 0 0 0 0 253952 5292 0 0 0 0 ");
}