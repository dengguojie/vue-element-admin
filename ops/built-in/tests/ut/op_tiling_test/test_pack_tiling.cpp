#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "../../../op_tiling/op_tiling_util.h"

using namespace std;
using namespace ge;

class PackTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "PackTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "PackTiling TearDown" << std::endl;
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

TEST_F(PackTiling, Pack_tiling1) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Pack");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::Pack("Pack");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":0, \"input_size\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 256 256 768 3 0 0 256 0 256 256 256 512 ");
}

TEST_F(PackTiling, Pack_tiling2) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Pack");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 4, 4, 4},
      {4, 4, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::Pack("Pack");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 256 1 1 3 3 0 0 1 0 1 1 1 2 ");
  EXPECT_EQ(4, optiling::GetByteLenByString("int32"));  
}

TEST_F(PackTiling, Pack_tiling3) {
  auto iter = optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().find("Pack");
  ASSERT_TRUE(iter != optiling::utils::OpTilingRegistryInterf_V2::RegisteredOpInterf().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 5},
      {4, 5},
      {4, 5},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::Pack("Pack");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":2, \"input_size\":3}}";
  optiling::utils::OpCompileInfo op_compile_info(this->test_info_->name(), compileInfo.c_str());
  optiling::utils::OpRunInfo runInfo;
  ASSERT_TRUE(iter->second(opParas, op_compile_info, runInfo));
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 20 1 1 3 3 0 0 1 0 1 1 1 2 ");
}
