#include <iostream>
#include <fstream>
#include <vector>

#include <gtest/gtest.h>
#include "graph/utils/op_desc_utils.h"
#include "graph/graph.h"
#include "register/op_tiling_registry.h"
#include "op_tiling/split_dsl.h"
#include "op_tiling/tiling_handler.h"
#include "common_autotiling_util.h"

using namespace optiling;
class SplitDslTilingRt2 : public testing::Test {
 protected:
  static void SetUpTestCase() {
    std::cout << "SplitDslTilingRt2 SetUp" << std::endl;
  }

  static void TearDownTestCase() {
    std::cout << "SplitDslTilingRt2 TearDown" << std::endl;
  }
};

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case0) {
  std::vector<std::vector<int64_t>> inputs {
    {100, 50}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "100, 50, 25";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetTilingKey(), 0);
}


TEST_F(SplitDslTilingRt2, split_dsl_tiling_case1) {
  std::vector<std::vector<int64_t>> inputs {
    {2591, 2, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 2;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 2, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "5182, 170, 85, 162, 85, 162, 85";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case2) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 2590, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "2, 440300, 220150, 1, 13760, 1, 13760";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case3) {
  std::vector<std::vector<int64_t>> inputs {
    {5182, 22, 46, 799}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "5182, 808588, 404294, 162, 404294, 1, 65536";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case4) {
  std::vector<std::vector<int64_t>> inputs {
    {5182, 69237}
  };
  int64_t split_num = 63;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "5182, 69237, 1099, 162, 1099, 59, 1099";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case5) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 800}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1000, 800, 400, 1, 32, 800";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 4000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case6) {
  std::vector<std::vector<int64_t>> inputs {
    {10, 8000}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "10, 8000, 4000, 1, 10, 8000";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 1);
  EXPECT_EQ(test.GetTilingKey(), 4000001);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case7) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 144}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1000, 144, 72, 1";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 4);
  EXPECT_EQ(test.GetTilingKey(), 5000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case8) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 111}
  };
  int64_t split_num = 2;
  int64_t split_dim = 0;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 0, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [false, true], "_vars": {"3000000": ["_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_1", "_split_0"], "5000000": ["_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_1", "_split_0"], "5000000": ["_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "111000, 55500, 1, 1735, 1, 1735";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case9) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1110}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": true, "_only_const_tiling": false, "_is_const": true, "_const_dims": 32, "_split_vars": [false, false], "_vars": {"1000000": []}, "_normal_vars": {"1000000": []}, "_attr_vars": {"1000000": []}, "_custom_vars": {"1000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 1000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case10) {
  std::vector<std::vector<int64_t>> inputs {
    {1000, 1110}
  };
  int64_t split_num = 1;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 0, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [false, true], "_vars": {"3000000": ["_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"]}, "_attr_vars": {"3000000": []}, "_custom_vars": {"3000000": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "1110000, 1110000, 1, 34688, 1, 34688";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case11) {
  std::vector<std::vector<int64_t>> inputs {
    {100, 50}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_INT8;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "100, 50, 25, 4, 25, 4, 25";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 25);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case12) {
  std::vector<std::vector<int64_t>> inputs {
    {2591, 2, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 2;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 2, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "5182, 170, 85, 162, 85, 162, 85";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case13) {
  std::vector<std::vector<int64_t>> inputs {
    {5182, 22, 46, 799}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_UINT64;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  EXPECT_EQ(test.Test(), true);
  std::string expect_tiling_data = "5182, 808588, 404294, 162, 404294, 1, 16384";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case14) {
  std::vector<std::vector<int64_t>> inputs {
    {100, 50}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_INT8;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "100, 50, 25, 4, 25, 4, 25";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 25);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case15) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 2590, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0"], "2000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_1"], "2000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_1"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": [], "2000000": [], "2000001": []}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "2, 440300, 220150, 1, 13760, 1, 13760";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case16) {
  std::vector<std::vector<int64_t>> inputs {
    {2591, 2, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 2;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "5182, 170, 85, 162, 85, 162, 85";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case17) {
  std::vector<std::vector<int64_t>> inputs {
    {5182, 22, 46, 799}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": true, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor", "_avg_block_factor", "_ub_factor_0", "_ub_factor_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}})";
  ge::DataType dtype = ge::DT_UINT64;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "5182, 808588, 404294, 162, 404294, 1, 16384";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case18) {
  std::vector<std::vector<int64_t>> inputs {
    {100, 50}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": false, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "split_size"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "split_size"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "split_size"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "split_size"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": ["split_size"], "4000000": ["split_size"], "4000001": ["split_size"], "4100000": ["split_size"]}})";
  ge::DataType dtype = ge::DT_INT8;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({10, 40});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "100, 50, 10, 40, 4, 4, 10, 4, 40";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 25);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case19) {
  std::vector<std::vector<int64_t>> inputs {
    {2, 2590, 170}
  };
  int64_t split_num = 2;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": false, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "split_size"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "split_size"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "split_size"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "split_size"], "0": ["_dim_0", "_dim_1", "_split_0", "_split_1", "split_size"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "split_size"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0", "_ub_factor_0"], "0": ["_dim_0", "_dim_1", "_split_0", "_split_1"], "5000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_block_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": [], "0": [], "5000000": []}, "_custom_vars": {"3000000": ["split_size"], "4000000": ["split_size"], "4000001": ["split_size"], "4100000": ["split_size"], "0": ["split_size"], "5000000": ["split_size"]}})";
  ge::DataType dtype = ge::DT_FLOAT16;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({2500, 90});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "2, 440300, 425000, 15300, 1, 1, 65536, 1, 15300";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 2);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case20) {
  std::vector<std::vector<int64_t>> inputs {
    {2591, 2, 170}
  };
  int64_t split_num = 3;
  int64_t split_dim = 2;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": false, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "_ub_factor_0_2", "_ub_factor_1_2", "split_size"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "split_size"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "split_size"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "split_size"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "_ub_factor_0_2", "_ub_factor_1_2"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": ["split_size"], "4000000": ["split_size"], "4000001": ["split_size"], "4100000": ["split_size"]}})";
  ge::DataType dtype = ge::DT_FLOAT;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({70, 50, 50});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "5182, 170, 70, 50, 50, 162, 162, 70, 162, 50, 162, 50";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}

TEST_F(SplitDslTilingRt2, split_dsl_tiling_case21) {
  std::vector<std::vector<int64_t>> inputs {
    {5182, 22, 46, 799}
  };
  int64_t split_num = 3;
  int64_t split_dim = 1;
  std::vector<std::vector<int64_t>> outputs(split_num, inputs[0]);
  int64_t split_shape = inputs[0][split_dim] / split_num;
  for (auto& output: outputs) {
    output[split_dim] = split_shape;
  }

  std::string compile_str = R"({"_ori_axis": 1, "_pattern": "Split", "_core_num": 32, "_ub_size": 262144, "_avg_split": false, "_split_is_const": false, "_only_const_tiling": false, "_is_const": false, "_split_vars": [true, true], "_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "_ub_factor_0_2", "_ub_factor_1_2", "split_size"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1", "split_size"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1", "split_size"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "split_size"]}, "_normal_vars": {"3000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor", "_ub_factor_0_0", "_ub_factor_1_0", "_ub_factor_0_1", "_ub_factor_1_1", "_ub_factor_0_2", "_ub_factor_1_2"], "4000000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0", "_ub_factor_1"], "4000001": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_1", "_ub_factor_0", "_ub_factor_1"], "4100000": ["_dim_0", "_dim_1", "_split_0", "_split_1", "_split_2", "_block_factor_0", "_ub_factor_0"]}, "_attr_vars": {"3000000": [], "4000000": [], "4000001": [], "4100000": []}, "_custom_vars": {"3000000": ["split_size"], "4000000": ["split_size"], "4000001": ["split_size"], "4100000": ["split_size"]}})";
  ge::DataType dtype = ge::DT_UINT64;

  AutoTilingTest test(inputs, outputs, dtype, dtype);
  optiling::split::SplitCompileInfo split_info;
  test.SetCompileInfo(compile_str, &split_info);

  inputs.push_back({19, 2, 1});
  std::vector<int64_t> axes{split_dim};
  optiling::OpInfo op_info(&split_info);
  op_info.SetInputShape(&inputs);
  op_info.SetInputType(&dtype);
  op_info.SetAxes(&axes);
  EXPECT_EQ(test.Test(&op_info), true);
  std::string expect_tiling_data = "5182, 808588, 698326, 73508, 36754, 162, 1, 16384, 1, 16384, 1, 16384";
  EXPECT_EQ(test.GetInt32TilingData(), expect_tiling_data);
  EXPECT_EQ(test.GetBlockDims(), 32);
  EXPECT_EQ(test.GetTilingKey(), 3000000);
}