#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"

using namespace std;

class SplitTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitTiling TearDown" << std::endl;
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
REG_OP(Split)
    .INPUT(split_dim, TensorType({DT_INT32}))
    .INPUT(x, TensorType::BasicType())
    .DYNAMIC_OUTPUT(y, TensorType::BasicType())
    .REQUIRED_ATTR(num_split, Int)
    .OP_END_FACTORY_REG(Split)
*/

TEST_F(SplitTiling, Split_tiling1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Split");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Split("Split");
  vector<vector<int64_t>> input_shapes = {
      {1820, 232},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {1820, 232},
  };
  vector<ge::DataType> dtypes = {ge::DT_INT8, ge::DT_INT32};
  vector<int32_t> split_dim{0};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSplitDim.SetDataType(dtypes[1]);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  opParas.SetAttr("num_split", {1820});

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":253952, \"num_split\":1}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()),
            "1 32 422240 1820 13195 13195 0 13195 13195 0 13195 13195 232 1 422240 0 0 0 0 0 0 0 0 1820 ");
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}

TEST_F(SplitTiling, Split_tiling2) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Split");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  auto opParas = op::Split("Split");
  vector<vector<int64_t>> input_shapes = {
      {40000, 84},
      {1},
  };
  vector<vector<int64_t>> output_shapes = {
      {40000, 1},
      {40000, 2},
      {40000, 1},
      {40000, 80},
  };
  vector<ge::DataType> dtypes = {ge::DT_FLOAT16, ge::DT_INT32};
  std::vector<int32_t> split_dim{1};

  TensorDesc tensorInputx;
  tensorInputx.SetShape(ge::Shape(input_shapes[0]));
  tensorInputx.SetDataType(dtypes[0]);
  tensorInputx.SetFormat(ge::FORMAT_FRACTAL_NZ);
  TENSOR_INPUT(opParas, tensorInputx, x);

  TensorDesc tensorInputSplitDim;
  tensorInputSplitDim.SetShape(ge::Shape(input_shapes[1]));
  tensorInputSplitDim.SetDataType(dtypes[1]);
  tensorInputSplitDim.SetFormat(ge::FORMAT_FRACTAL_NZ);
  TENSOR_INPUT_CONST(opParas, tensorInputSplitDim, split_dim, (const uint8_t*)split_dim.data(), split_dim.size() * 4);

  opParas.SetAttr("num_split", {1, 2, 1, 80});

  for (size_t i = 0; i < output_shapes.size(); i++) {
    TensorDesc tensorOutput;
    tensorOutput.SetShape(ge::Shape(output_shapes[i]));
    tensorOutput.SetDataType(dtypes[0]);
    tensorOutput.SetFormat(ge::FORMAT_FRACTAL_NZ);
    TENSOR_OUTPUT(opParas, tensorOutput, y);
  }

  std::string compileInfo = "{\"vars\": {\"core_num\": 32, \"ub_elems\":126976, \"num_split\":4}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo);
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), to_string(runInfo.GetAllTilingData()));
  int64_t tiling_test_num = 0;
  for (int64_t i = 0; i < tiling_test_num; i++) {
    iter->second(opParas, op_compile_info, runInfo);
  }
}