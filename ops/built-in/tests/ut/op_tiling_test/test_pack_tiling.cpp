#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "../../../op_tiling/op_tiling_util.h"
#include "common/utils/ut_op_util.h"

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
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Pack");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":0, \"input_size\":3}, \"is_tik\": true}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 256 256 768 3 0 0 256 0 256 256 256 512 ");
}

TEST_F(PackTiling, Pack_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Pack");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":3}, \"is_tik\": true}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 256 1 1 3 3 0 0 1 0 1 1 1 2 ");
  EXPECT_EQ(4, optiling::GetByteLenByString("int32"));  
}

TEST_F(PackTiling, Pack_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Pack");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":2, \"input_size\":3}, \"is_tik\": true}";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 20 1 1 3 3 0 0 1 0 1 1 1 2 ");
}

static string to_string_int32(const std::stringstream& tiling_data) {
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

TEST_F(PackTiling, Pack_dsl_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("Pack");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
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
  opParas.SetAttr("axis", -1);

  std::string compileInfo = R"({"concat_dim": 4, "_ori_axis": 4, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string_int32(runInfo.GetAllTilingData()), "256 ");
}
