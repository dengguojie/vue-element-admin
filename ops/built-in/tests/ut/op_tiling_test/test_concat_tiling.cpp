#include <iostream>
#include <vector>

#include <gtest/gtest.h>
#define private public
#include "register/op_tiling_registry.h"
#include "split_combination_ops.h"
#include "array_ops.h"
#include "common/utils/ut_op_util.h"

using namespace std;
using namespace ge;

class ConcatTiling : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "ConcatTiling SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "ConcatTiling TearDown" << std::endl;
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

TEST_F(ConcatTiling, Concat_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {5, 4, 4, 4},
      {6, 4, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 32 0 0 ");
}

TEST_F(ConcatTiling, Concat_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 5, 4, 4},
      {4, 6, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 1 4 ");
}

TEST_F(ConcatTiling, Concat_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1},
      {4, 5},
      {4, 6},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": -1, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 ");
}

TEST_F(ConcatTiling, Concat_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {2, 1, 32, 16},
      {2, 1, 32, 32},
      {2, 1, 32, 48},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_HWCN);
  tensor_input2.SetOriginFormat(FORMAT_HWCN);
  tensor_input3.SetOriginFormat(FORMAT_HWCN);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 3, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 1 1 ");
}

TEST_F(ConcatTiling, Concat_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {16, 32, 2, 1},
      {32, 32, 2, 1},
      {48, 32, 2, 1},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_NCHW);
  tensor_input2.SetOriginFormat(FORMAT_NCHW);
  tensor_input3.SetOriginFormat(FORMAT_NCHW);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 192 64 0 ");
}

TEST_F(ConcatTiling, Concat_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {16, 2, 2, 1, 16},
      {32, 2, 2, 1, 16},
      {48, 2, 2, 1, 16},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_input2.SetOriginFormat(FORMAT_NC1HWC0);
  tensor_input3.SetOriginFormat(FORMAT_NC1HWC0);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 192 64 0 ");
}

TEST_F(ConcatTiling, Concat_tiling7) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {16, 16, 2, 1, 2},
      {32, 16, 2, 1, 2},
      {48, 16, 2, 1, 2},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_NCDHW);
  tensor_input2.SetOriginFormat(FORMAT_NCDHW);
  tensor_input3.SetOriginFormat(FORMAT_NCDHW);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 192 64 0 ");
}

TEST_F(ConcatTiling, Concat_tiling8) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {16, 2, 1, 2, 16},
      {32, 2, 1, 2, 16},
      {48, 2, 1, 2, 16},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_NDHWC);
  tensor_input2.SetOriginFormat(FORMAT_NDHWC);
  tensor_input3.SetOriginFormat(FORMAT_NDHWC);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 0, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "1 1 1 192 64 0 ");
}

TEST_F(ConcatTiling, Concat_tiling9) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 1, 16, 16},
      {4, 2, 16, 16},
      {4, 3, 16, 16},
  };
  vector<vector<int64_t>> input_ori_shapes = {
      {2, 1, 2, 16, 16},
      {2, 1, 2, 16, 32},
      {2, 1, 2, 16, 48},
  };

  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_FRACTAL_Z_3D, DT_FLOAT16);
  tensor_input1.SetOriginShape(ge::Shape(input_ori_shapes[0]));
  tensor_input2.SetOriginShape(ge::Shape(input_ori_shapes[1]));
  tensor_input3.SetOriginShape(ge::Shape(input_ori_shapes[2]));
  tensor_input1.SetOriginFormat(FORMAT_DHWCN);
  tensor_input2.SetOriginFormat(FORMAT_DHWCN);
  tensor_input3.SetOriginFormat(FORMAT_DHWCN);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = R"({"_ori_axis": 4, "_pattern": "Concat", "_core_num": 32, "_ub_size": 262144, "_only_const_tiling": false, "_is_const": false, "_concat_vars": [[true, false], [false, false], [false, false]], "_align_vars": [0, 1, 2], "_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "3000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_1", "_offset_1", "_offset_2"], "4000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_offset_1", "_offset_2"], "4100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0_0"], "2000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "2100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"],"5000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5000001": ["_dim_0_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "_align_factor_0", "_align_factor_1", "_align_factor_2", "_offset_1", "_offset_2"], "5100000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"], "6000000": ["_dim_0_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}, "_custom_vars": {"3000000": [], "3000001": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "2000000": [], "2000001": [], "2100000": [], "5000000": [], "5000001": [], "5100000": [], "6000000": []}})";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string(runInfo.GetAllTilingData()), "4 1 1 ");
}

static string to_string_int64(const std::stringstream& tiling_data) {
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

TEST_F(ConcatTiling, Concat_tik_tiling1) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {5, 4, 4, 4},
      {6, 4, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":0, \"input_size\":3}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), "1 1 384 256 960 3 0 0 256 0 320 256 384 576 ");
}

TEST_F(ConcatTiling, Concat_tik_tiling2) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4, 4, 4},
      {4, 5, 4, 4},
      {4, 6, 4, 4},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":1, \"input_size\":3}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(to_string_int64(runInfo.GetAllTilingData()), "1 4 96 64 240 3 0 0 64 0 80 64 96 144 ");
}

TEST_F(ConcatTiling, Concat_tik_tiling3) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 5},  {4, 6},  {4, 7},  {4, 8},  {4, 9},  {4, 10}, {4, 11}, {4, 12}, {4, 13}, {4, 14},
      {4, 15}, {4, 16}, {4, 17}, {4, 18}, {4, 19}, {4, 20}, {4, 21}, {4, 22}, {4, 23}, {4, 24},
      {4, 25}, {4, 26}, {4, 27}, {4, 28}, {4, 29}, {4, 30}, {4, 31}, {4, 32}, {4, 33}, {4, 34},
  };
  TensorDesc tensor_input0(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input1(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[3]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input4(ge::Shape(input_shapes[4]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input5(ge::Shape(input_shapes[5]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input6(ge::Shape(input_shapes[6]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input7(ge::Shape(input_shapes[7]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input8(ge::Shape(input_shapes[8]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input9(ge::Shape(input_shapes[9]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input10(ge::Shape(input_shapes[10]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input11(ge::Shape(input_shapes[11]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input12(ge::Shape(input_shapes[12]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input13(ge::Shape(input_shapes[13]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input14(ge::Shape(input_shapes[14]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input15(ge::Shape(input_shapes[15]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input16(ge::Shape(input_shapes[16]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input17(ge::Shape(input_shapes[17]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input18(ge::Shape(input_shapes[18]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input19(ge::Shape(input_shapes[19]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input20(ge::Shape(input_shapes[20]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input21(ge::Shape(input_shapes[21]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input22(ge::Shape(input_shapes[22]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input23(ge::Shape(input_shapes[23]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input24(ge::Shape(input_shapes[24]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input25(ge::Shape(input_shapes[25]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input26(ge::Shape(input_shapes[26]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input27(ge::Shape(input_shapes[27]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input28(ge::Shape(input_shapes[28]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input29(ge::Shape(input_shapes[29]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(30);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input0);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 3, tensor_input3);
  opParas.UpdateDynamicInputDesc("x", 4, tensor_input4);
  opParas.UpdateDynamicInputDesc("x", 5, tensor_input5);
  opParas.UpdateDynamicInputDesc("x", 6, tensor_input6);
  opParas.UpdateDynamicInputDesc("x", 7, tensor_input7);
  opParas.UpdateDynamicInputDesc("x", 8, tensor_input8);
  opParas.UpdateDynamicInputDesc("x", 9, tensor_input9);
  opParas.UpdateDynamicInputDesc("x", 10, tensor_input10);
  opParas.UpdateDynamicInputDesc("x", 11, tensor_input11);
  opParas.UpdateDynamicInputDesc("x", 12, tensor_input12);
  opParas.UpdateDynamicInputDesc("x", 13, tensor_input13);
  opParas.UpdateDynamicInputDesc("x", 14, tensor_input14);
  opParas.UpdateDynamicInputDesc("x", 15, tensor_input15);
  opParas.UpdateDynamicInputDesc("x", 16, tensor_input16);
  opParas.UpdateDynamicInputDesc("x", 17, tensor_input17);
  opParas.UpdateDynamicInputDesc("x", 18, tensor_input18);
  opParas.UpdateDynamicInputDesc("x", 19, tensor_input19);
  opParas.UpdateDynamicInputDesc("x", 20, tensor_input20);
  opParas.UpdateDynamicInputDesc("x", 21, tensor_input21);
  opParas.UpdateDynamicInputDesc("x", 22, tensor_input22);
  opParas.UpdateDynamicInputDesc("x", 23, tensor_input23);
  opParas.UpdateDynamicInputDesc("x", 24, tensor_input24);
  opParas.UpdateDynamicInputDesc("x", 25, tensor_input25);
  opParas.UpdateDynamicInputDesc("x", 26, tensor_input26);
  opParas.UpdateDynamicInputDesc("x", 27, tensor_input27);
  opParas.UpdateDynamicInputDesc("x", 28, tensor_input28);
  opParas.UpdateDynamicInputDesc("x", 29, tensor_input29);
  opParas.SetAttr("N", 30);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":30}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3(opParas, iter->second, compileInfo, runInfo);
  EXPECT_EQ(
      to_string_int64(runInfo.GetAllTilingData()),
      "1 4 34 5 585 30 0 0 5 0 6 5 7 11 8 18 9 26 10 35 11 45 12 56 13 68 14 81 15 95 16 110 17 126 18 143 19 161 20 "
      "180 21 200 22 221 23 243 24 266 25 290 26 315 27 341 28 368 29 396 30 425 31 455 32 486 33 518 34 551 ");
}

TEST_F(ConcatTiling, Concat_tik_tiling4) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {},
      {4, 5},
      {4, 6},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":3}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ConcatTiling, Concat_tik_tiling5) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {4, 4},
      {4, 5},
      {4, 6},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}

TEST_F(ConcatTiling, Concat_tik_tiling6) {
  auto iter = optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().find("ConcatD");
  ASSERT_TRUE(iter != optiling::OpTilingFuncRegistry::RegisteredOpFuncInfo().end());
  vector<vector<int64_t>> input_shapes = {
      {9, 4},
      {4, 5},
      {4, 6},
  };
  TensorDesc tensor_input1(ge::Shape(input_shapes[0]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input2(ge::Shape(input_shapes[1]), FORMAT_ND, DT_FLOAT16);
  TensorDesc tensor_input3(ge::Shape(input_shapes[2]), FORMAT_ND, DT_FLOAT16);

  auto opParas = op::ConcatD("ConcatD");
  opParas.create_dynamic_input_x(3);
  opParas.UpdateDynamicInputDesc("x", 0, tensor_input1);
  opParas.UpdateDynamicInputDesc("x", 1, tensor_input2);
  opParas.UpdateDynamicInputDesc("x", 2, tensor_input3);
  opParas.SetAttr("N", 3);
  opParas.SetAttr("concat_dim", -1);

  std::string compileInfo = "{\"vars\": {\"block_dim\": 32, \"concat_dim\":-1, \"input_size\":3}, \"is_tik\": true}";
  // do tilling, get runInfo
  optiling::utils::OpRunInfo runInfo;
  RUN_TILING_V3_FALSE(opParas, iter->second, compileInfo, runInfo);
}
